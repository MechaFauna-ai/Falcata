# coding: utf-8
"""DatasetLoader validates forced-bins files and label_column.

- GetForcedBins ignored the JSON parse error (failing later in a bare CHECK),
  only upper-bounded the feature index (a negative one indexed out of
  bounds), and removed only adjacent duplicate bounds.
- SetHeader erased feature_names_[label_idx_] with an unchecked numeric
  label_column.
- LoadFromFile's distributed "query id needs pre-partition" guard tested
  group_idx_ > 0, so a query id in column 0 slipped through.
- CheckCategoricalFeatureNumBin kept its per-thread flags in a
  std::vector<bool>: concurrent writes to bits of one word could drop the
  flag of the thread that saw an oversized categorical feature.
"""

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import falcata as flc
from falcata.basic import FalcataError


@pytest.fixture
def csv_with_header(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 4, size=(500, 3))
    y = X[:, 0] + rng.normal(scale=0.1, size=500)
    path = tmp_path / "data.csv"
    with open(path, "w") as f:
        f.write("y,a,b,c\n")
        np.savetxt(f, np.column_stack([y, X]), delimiter=",")
    return path, X, y


def _num_bins(X, y, forced, tmp_path, name):
    path = tmp_path / name
    path.write_text(json.dumps(forced))
    ds = flc.Dataset(X, label=y, params={"forcedbins_filename": str(path), "max_bin": 255, "verbose": -1})
    ds.construct()
    return ds.feature_num_bin(0)


def test_forced_bins_non_adjacent_duplicates_removed(csv_with_header, tmp_path):
    _, X, y = csv_with_header
    dup = _num_bins(X, y, [{"feature": 0, "bin_upper_bound": [1.0, 2.0, 1.0]}], tmp_path, "dup.json")
    uniq = _num_bins(X, y, [{"feature": 0, "bin_upper_bound": [1.0, 2.0]}], tmp_path, "uniq.json")
    assert dup == uniq


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ('[{"feature": 0, "bin_upper_bound": [1.0', "Could not parse forced bins"),
        ('{"feature": 0}', "must contain a JSON array"),
        ('[{"feature": -1, "bin_upper_bound": [1.0]}]', "names feature -1"),
        ('[{"feature": 3, "bin_upper_bound": [1.0]}]', "names feature 3"),
    ],
)
def test_forced_bins_bad_file_raises(csv_with_header, tmp_path, content, match):
    _, X, y = csv_with_header
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(FalcataError, match=match):
        flc.Dataset(X, label=y, params={"forcedbins_filename": str(path), "verbose": -1}).construct()


def test_label_column_out_of_range_raises(csv_with_header):
    path, _, _ = csv_with_header
    with pytest.raises(FalcataError, match="label_column 9 is out of range"):
        flc.Dataset(str(path), params={"header": True, "label_column": "9", "verbose": -1}).construct()


_DISTRIBUTED_QUERY_CHILD = textwrap.dedent(
    """
    import ctypes
    import sys

    import falcata as flc
    from falcata.basic import _LIB, FalcataError, _safe_call

    # two machines through external collectives that are never reached when
    # the loader refuses the file up front
    REDUCE_SCATTER = ctypes.CFUNCTYPE(
        None, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_char_p, ctypes.c_int32, ctypes.c_void_p,
    )
    ALLGATHER = ctypes.CFUNCTYPE(
        None, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int32,
    )
    rs = REDUCE_SCATTER(lambda *a: None)
    ag = ALLGATHER(lambda *a: None)
    _safe_call(_LIB.FLC_NetworkInitWithFunctions(
        ctypes.c_int(2), ctypes.c_int(0), ctypes.cast(rs, ctypes.c_void_p), ctypes.cast(ag, ctypes.c_void_p)))
    params = {"group_column": sys.argv[2], "label_column": "1", "header": False, "verbose": -1}
    try:
        flc.Dataset(sys.argv[1], params=params).construct()
    except FalcataError as err:
        print("raised:", err)
        sys.exit(0 if "without pre-partitioning" in str(err) else 4)
    print("loaded")
    sys.exit(3)
    """
)


@pytest.mark.parametrize("group_column", ["0", "2"])
def test_distributed_query_column_needs_pre_partition(tmp_path, group_column):
    rng = np.random.default_rng(0)
    path = tmp_path / "query.csv"
    cols = [np.repeat(np.arange(20), 10), rng.normal(size=200), rng.normal(size=200)]
    if group_column == "2":
        cols = [cols[2], cols[1], cols[0]]
    np.savetxt(path, np.column_stack(cols), delimiter=",")
    proc = subprocess.run(
        [sys.executable, "-c", _DISTRIBUTED_QUERY_CHILD, str(path), group_column],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"


_CATEGORICAL_WARNING_CHILD = textwrap.dedent(
    """
    import sys

    import numpy as np

    import falcata as flc

    runs, num_threads = int(sys.argv[1]), int(sys.argv[2])
    num_features = 4096  # >= 1024 takes the threaded check
    X = np.random.default_rng(0).integers(0, 3, size=(400, num_features)).astype(np.float64)
    cat = num_features // num_threads  # first feature of the second thread's block
    X[:, cat] = np.arange(400)  # 400 categories, more than max_bin
    params = {"max_bin": 63, "min_data_in_bin": 1, "num_threads": num_threads, "verbose": 0}
    for _ in range(runs):
        flc.Dataset(X, label=X[:, 0], categorical_feature=[cat], params=params).construct()
        print("CONSTRUCTED", flush=True)
    """
)


def test_threaded_categorical_bin_check_always_warns():
    runs = 100
    proc = subprocess.run(
        [sys.executable, "-c", _CATEGORICAL_WARNING_CHILD, str(runs), "8"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.count("CONSTRUCTED") == runs
    warned = proc.stdout.count("Categorical features with more bins than the configured maximum bin number found.")
    assert warned == runs, f"the oversized categorical feature was reported in only {warned} of {runs} constructions"
