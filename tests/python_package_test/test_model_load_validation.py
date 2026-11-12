"""A malformed model must be refused at load time, never crash or hang later.

The text and FALB loaders took tree structure and header counts on trust. A
fuzzed model could make prediction loop forever (a child pointing back at an
ancestor), read out of bounds (a child, split feature or category bitset index
past its array), overflow the recursive writers' stack, divide by zero
(num_tree_per_iteration=0) or overflow the caller's prediction buffer
(num_tree_per_iteration > num_class). Each case runs in a child process so a
crash or hang on an unfixed build fails the test instead of the session.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import falcata as flc

_CHILD = textwrap.dedent(
    """
    import sys
    import numpy as np
    import falcata as flc

    text = open(sys.argv[1], encoding="utf-8").read()
    try:
        booster = flc.Booster(model_str=text)
    except flc.basic.FalcataError as err:
        print("REFUSED", err)
        sys.exit(0)
    x = np.array([[0.0, np.nan, 3.0, -1.0]] * 2)[:, : booster.num_feature()]
    for kw in ({}, {"raw_score": True}, {"pred_leaf": True}, {"pred_contrib": True}):
        try:
            booster.predict(x, **kw)
        except flc.basic.FalcataError:
            pass
    booster.dump_model()
    booster.model_to_string()
    print("LOADED")
    """
)


@pytest.fixture(scope="module")
def model_text():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 4))
    x[:, 3] = rng.integers(0, 6, size=400)
    y = 0.1 * x[:, 0] + 5.0 * np.isin(x[:, 3], [2, 4])
    ds = flc.Dataset(x, label=y, categorical_feature=[3])
    params = {"verbose": -1, "num_threads": 1, "num_leaves": 4, "min_data_in_leaf": 5, "max_cat_to_onehot": 1}
    return flc.train(params, ds, num_boost_round=2).model_to_string()


def _first_tree_line(text, key):
    start = text.index("Tree=0\n")
    i = text.index(f"\n{key}=", start) + 1
    j = text.index("\n", i)
    return i, j


def _without_tree_sizes(text):
    # edits change the tree block sizes; without the line the loader reads
    # the blocks sequentially instead of refusing the stale sizes
    i = text.index("tree_sizes=")
    return text[:i] + text[text.index("\n", i) + 1 :]


def _replace_in_first_tree(text, key, value):
    i, j = _first_tree_line(text, key)
    return _without_tree_sizes(text[:i] + f"{key}={value}" + text[j:])


def _run(tmp_path, text):
    path = tmp_path / "model.txt"
    path.write_text(text, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(path)], capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.returncode == 0, f"child died with {proc.returncode}: {proc.stderr[-1500:]}"
    return proc.stdout


def _tree0(text, key):
    i, j = _first_tree_line(text, key)
    return text[i:j].split("=", 1)[1].split(" ")


def test_valid_model_still_loads(tmp_path, model_text):
    assert "LOADED" in _run(tmp_path, model_text)


def test_child_cycle_is_refused(tmp_path, model_text):
    left = _tree0(model_text, "left_child")
    right = _tree0(model_text, "right_child")
    # point the last internal node's child back at the root
    node = len(left) - 1
    if int(left[node]) < 0:
        left[node] = "0"
        bad = _replace_in_first_tree(model_text, "left_child", " ".join(left))
    else:
        right[node] = "0"
        bad = _replace_in_first_tree(model_text, "right_child", " ".join(right))
    assert "REFUSED" in _run(tmp_path, bad)


@pytest.mark.parametrize("child", ["7", "-9"])
def test_child_out_of_range_is_refused(tmp_path, model_text, child):
    left = _tree0(model_text, "left_child")
    left[0] = child
    bad = _replace_in_first_tree(model_text, "left_child", " ".join(left))
    assert "REFUSED" in _run(tmp_path, bad)


@pytest.mark.parametrize("feature", ["-1", "4", "100000"])
def test_split_feature_out_of_range_is_refused(tmp_path, model_text, feature):
    feats = _tree0(model_text, "split_feature")
    feats[0] = feature
    bad = _replace_in_first_tree(model_text, "split_feature", " ".join(feats))
    assert "REFUSED" in _run(tmp_path, bad)


def test_bad_categorical_bitset_index_is_refused(tmp_path, model_text):
    dtypes = _tree0(model_text, "decision_type")
    thresholds = _tree0(model_text, "threshold")
    cat_nodes = [k for k, d in enumerate(dtypes) if int(d) & 1]
    assert cat_nodes, "fixture must contain a categorical split"
    thresholds[cat_nodes[0]] = "5"  # the tree has fewer bitsets than that
    bad = _replace_in_first_tree(model_text, "threshold", " ".join(thresholds))
    assert "REFUSED" in _run(tmp_path, bad)


def test_negative_num_cat_is_refused(tmp_path, model_text):
    bad = _replace_in_first_tree(model_text, "num_cat", "-1")
    assert "REFUSED" in _run(tmp_path, bad)


def test_zero_trees_per_iteration_is_refused(tmp_path, model_text):
    head = model_text[: model_text.index("Tree=0\n")]
    tail = model_text[model_text.index("end of trees") :]
    empty = head.replace("num_tree_per_iteration=1", "num_tree_per_iteration=0")
    empty = empty.replace(
        empty[empty.index("tree_sizes=") : empty.index("\n", empty.index("tree_sizes="))], "tree_sizes="
    )
    assert "REFUSED" in _run(tmp_path, empty + tail)


def test_more_trees_per_iteration_than_classes_is_refused(tmp_path, model_text):
    bad = model_text.replace("num_tree_per_iteration=1", "num_tree_per_iteration=2", 1)
    assert "num_tree_per_iteration=2" in bad
    assert "REFUSED" in _run(tmp_path, bad)


def test_empty_objective_is_refused(tmp_path, model_text):
    start = model_text.index("objective=")
    end = model_text.index("\n", start)
    bad = model_text[:start] + "objective= " + model_text[end:]
    assert "REFUSED" in _run(tmp_path, bad)


def test_malformed_parameter_line_is_ignored(tmp_path, model_text):
    bad = model_text.replace("\nparameters:\n", "\nparameters:\nx\n[\n", 1)
    assert "LOADED" in _run(tmp_path, bad)


@pytest.mark.parametrize("value", ["-5", "2147483647"])
def test_invalid_max_feature_idx_is_refused(tmp_path, model_text, value):
    start = model_text.index("max_feature_idx=")
    end = model_text.index("\n", start)
    bad = model_text[:start] + f"max_feature_idx={value}" + model_text[end:]
    assert "REFUSED" in _run(tmp_path, bad)
