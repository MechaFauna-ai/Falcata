# coding: utf-8
"""colmajor_fill must engage when the device memory left after its copy covers what training can still allocate,
decline cleanly otherwise, and never change the model.

cuda_plan key colmajor_fill makes a one-time column-major copy of the 4-bit packed row matrix for the per-tree
compact fill to gather from. The copy is optional (the fill reads the row-major matrix without it and writes the
same bytes), so it is made before the second tree, once everything sized by the data exists, and only if free device
memory then covers the copy plus a reserve for what can still grow (CUDASingleGPUTreeLearner::ColMajorFillReserveBytes).
FALCATA_DEBUG=vramfree=N caps the free-memory figure that decision sees, which lets these tests put the decision on
either side of the line without filling the GPU; FALCATA_DEBUG=diag logs the decision. The token is read once per
process, so every run here is a subprocess.
"""

import os
import re
import subprocess
import sys

import pytest

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

_CHILD = """
import hashlib, re
import numpy as np
import falcata as flc

def data(rows, feats, seed):
    rng = np.random.default_rng(seed)
    # 14 values: 4-bit packed rows whose bins span the full nibble
    X = rng.integers(0, 14, size=(rows, feats)).astype(np.int8)
    y = (X[:, :8].astype(np.float64) @ rng.standard_normal(8) + rng.standard_normal(rows)).astype(np.float32)
    return X, y

params = {{"objective": "regression", "device_type": "cuda", "num_leaves": 31, "max_depth": 6,
           "min_data_in_leaf": 50, "max_bin": 15, "learning_rate": 0.1, "feature_fraction": 0.2, "seed": 3,
           "verbosity": 1, **{params!r}}}
X, y = data({rows}, {feats}, 0)
train = flc.Dataset(X, label=y, params=params)
bst = flc.Booster(params, train)
for _ in range({rounds}):
    bst.update()
if {reset_rounds}:
    # same features and bin mappers, different rows: ResetTrainingData swaps the row data under the booster
    X2, y2 = data({rows}, {feats}, 1)
    bst.update(train_set=flc.Dataset(X2, label=y2, params=params, reference=train))
    for _ in range({reset_rounds} - 1):
        bst.update()
# the parameter dump records the plan string itself
model = re.sub(r"^\\[cuda_plan: .*\\]$", "", bst.model_to_string(), flags=re.M)
print("MODEL_MD5", hashlib.md5(model.encode()).hexdigest(), flush=True)
"""

_DECISION = re.compile(
    r"colmajor_fill: (engaged|declined)[^(]*\(copy (\d+) MiB, reserve (\d+) MiB, free (\d+) MiB before the copy\)"
)

ROWS, FEATS = 200_000, 400


def _run(params, debug="diag", rounds=12, reset_rounds=0):
    """(decisions, model md5) of one training run in a fresh process; decisions are (kind, copy, reserve, free)."""
    code = _CHILD.format(params=params, rows=ROWS, feats=FEATS, rounds=rounds, reset_rounds=reset_rounds)
    env = {**os.environ, "FALCATA_DEBUG": debug}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout
    decisions = [(k, int(c), int(r), int(f)) for k, c, r, f in _DECISION.findall(out)]
    md5 = re.search(r"^MODEL_MD5 (\w+)$", out, re.M).group(1)
    return decisions, md5


QUANT = pytest.mark.parametrize("quant", ["stochastic", "fixedpoint"])


@_REQUIRES_CUDA
@QUANT
def test_colmajor_fill_engages_with_little_headroom_cuda(quant):
    """The copy engages when free memory covers it plus the computed reserve, a few hundred MiB here, and declines
    just below that; all three runs and colmajor_fill:off train the same model."""
    params = {"quant_mode": quant}
    probe, md5_probe = _run(params)
    assert [d[0] for d in probe] == ["engaged"], probe
    _, copy_mib, reserve_mib, _ = probe[0]
    # The rule this replaces needed more than max(2 GiB, copy / 2) free beyond the copy and the per-tree view;
    # the cap below leaves far less, so that rule would have declined this run.
    engage_cap = copy_mib + reserve_mib + 32
    assert engage_cap - copy_mib < max(2048, copy_mib // 2)

    engaged, md5_engaged = _run(params, debug=f"diag,vramfree={engage_cap}")
    assert [d[:3] for d in engaged] == [("engaged", copy_mib, reserve_mib)], engaged
    assert engaged[0][3] == engage_cap

    declined, md5_declined = _run(params, debug=f"diag,vramfree={copy_mib + reserve_mib - 32}")
    assert [d[:3] for d in declined] == [("declined", copy_mib, reserve_mib)], declined

    off, md5_off = _run({**params, "cuda_plan": "auto,colmajor_fill:off"})
    assert off == []
    assert md5_probe == md5_engaged == md5_declined == md5_off


@_REQUIRES_CUDA
def test_colmajor_fill_declines_when_the_copy_alone_does_not_fit_cuda():
    params = {"quant_mode": "stochastic"}
    declined, md5_declined = _run(params, debug="diag,vramfree=1")
    assert [d[0] for d in declined] == ["declined"], declined
    _, md5_off = _run({**params, "cuda_plan": "auto,colmajor_fill:off"})
    assert md5_declined == md5_off


@_REQUIRES_CUDA
def test_colmajor_fill_decides_once_before_the_second_tree_cuda():
    """No decision during the first tree (it sizes everything else first), exactly one before the second."""
    params = {"quant_mode": "stochastic"}
    assert _run(params, rounds=1)[0] == []
    assert [d[0] for d in _run(params, rounds=2)[0]] == ["engaged"]


@_REQUIRES_CUDA
@QUANT
def test_colmajor_fill_follows_reset_training_data_cuda(quant):
    """Swapping the training data drops the old rows' column-major copy and decides again on the new rows (a stale
    copy would feed the old rows' bins to the fill)."""
    params = {"quant_mode": quant}
    decisions, md5_on = _run(params, rounds=4, reset_rounds=6)
    assert [d[0] for d in decisions] == ["engaged", "engaged"], decisions
    _, md5_off = _run({**params, "cuda_plan": "auto,colmajor_fill:off"}, rounds=4, reset_rounds=6)
    assert md5_on == md5_off
