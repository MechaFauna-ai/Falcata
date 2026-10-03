# coding: utf-8
"""colmajor_direct keeps one device copy of the 4-bit bin matrix, built from the Dataset's columns, and never changes
the model.

cuda_plan key colmajor_direct (with colmajor_fill) builds no row-major matrix at Init. The training regime picks the
one copy the device holds:
  - compact (feature_fraction below view_mask_ff): the column-major store (the Dataset's 4-bit columns end to end),
    from which every tree fills its compact view of the sampled columns;
  - mask (feature_fraction at or above view_mask_ff): the full view (the row-major layout, filled once from the
    columns a chunk at a time), read by every tree under its column masks; no store.
cuda_plan view_mode:compact|mask forces a regime. A compact-regime reader that needs every column (a tree that
samples every column, the split view of a pack-codec tree) gets the full view, filled from the store and kept while
trees use it. colmajor_direct:off builds the row-major matrix at Init, as before. Every model must be byte-identical
across all of these, and after ResetTrainingData (update(train_set=...)) and reset_parameter(feature_fraction=...),
which decide the regime again.

FALCATA_VERIFY=1 compares the store and the full view with host references and FALCATA_DEBUG=diag logs every
decision; both are read once per process, so those runs are subprocesses.
"""

import os
import re
import subprocess
import sys

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

BASE = {
    "objective": "regression",
    "device_type": "cuda",
    "num_leaves": 63,
    "max_depth": 8,
    "min_data_in_leaf": 20,
    "learning_rate": 0.1,
    "max_bin": 15,
    "verbosity": -1,
    "seed": 3,
}

# (rows, columns, distinct values, extra params); every shape is 4-bit (at most 16 bins per column)
SHAPES = {
    # sampled columns over several feature partitions, odd row count (pad nibble)
    "sampled": (30_003, 700, 6, {"feature_fraction": 0.2}),
    # feature_fraction 1
    "ff1": (20_001, 60, 9, {"feature_fraction": 1.0}),
    # columns over 1 MiB: copied to the device one by one instead of through host gather runs
    "tall": (2_100_001, 8, 7, {"feature_fraction": 0.5}),
    # 600 KB columns: the mask regime's 256 MiB staging takes 447 of a partition's 500 columns per chunk
    "chunked": (1_200_001, 500, 6, {"feature_fraction": 0.9}),
}

OFF = "auto,colmajor_direct:off"


def _data(shape, seed=0):
    rows, cols, values, _ = SHAPES[shape]
    rng = np.random.default_rng(seed)
    X = rng.integers(0, values, size=(rows, cols)).astype(np.int8 if rows * cols > 100_000_000 else np.float32)
    y = X[:, :5].astype(np.float64) @ rng.standard_normal(5) + rng.standard_normal(rows)
    return X, y.astype(np.float32)


def _strip(model):
    # the parameter dump records the plan string itself
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.M)


def _params(shape, quant, plan, **extra):
    p = {**BASE, **SHAPES[shape][3], "quant_mode": quant, **extra}
    if quant == "none":
        # the deterministic non-quantized kernels make the float histograms run-to-run identical; the batched level
        # flow keeps the atomic kernel for compact views, so the classic loop (as lattice's sampled/nonquant-det)
        p["deterministic"] = True
        plan = plan + ",hybrid:off"
    p["cuda_plan"] = plan
    return p


def _train(shape, quant, plan, rounds=8, **extra):
    X, y = _data(shape)
    p = _params(shape, quant, plan, **extra)
    return _strip(flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=rounds).model_to_string())


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("shape", "quant"),
    [
        ("sampled", "stochastic"),
        ("sampled", "fixedpoint"),
        ("sampled", "none"),
        ("ff1", "stochastic"),
        ("ff1", "none"),
        ("tall", "stochastic"),
        ("chunked", "stochastic"),
    ],
)
def test_colmajor_direct_is_bit_identical_cuda(shape, quant):
    assert _train(shape, quant, "auto") == _train(shape, quant, OFF)


@_REQUIRES_CUDA
@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.99, 1.0])
@pytest.mark.parametrize("quant", ["stochastic", "none"])
def test_view_regimes_are_bit_identical_cuda(fraction, quant):
    """compact and mask train the same model at every feature_fraction, and the same as the row-major matrix."""
    models = {
        plan: _train("sampled", quant, plan, feature_fraction=fraction)
        for plan in ("auto,view_mode:compact", "auto,view_mode:mask", OFF)
    }
    assert models["auto,view_mode:compact"] == models["auto,view_mode:mask"] == models[OFF]


@_REQUIRES_CUDA
def test_colmajor_direct_pack_codec_is_bit_identical_cuda():
    """The pack-codec fill reads the store (base c * pad, stride 1); the split readers cannot read codec words and
    take the full view instead."""
    on = _train("sampled", "stochastic", "auto,pack_bit3:on")
    assert on == _train("sampled", "stochastic", "auto,pack_bit3:on,colmajor_direct:off")


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "none"])
def test_colmajor_direct_train_twice_on_one_dataset_cuda(quant):
    """Each train() decides its own regime on the same Dataset: compact, then mask at feature_fraction 0.95."""
    X, y = _data("sampled")
    models = {}
    for plan in ("auto", OFF):
        p = _params("sampled", quant, plan)
        ds = flc.Dataset(X, label=y, params=p)
        first = _strip(flc.train(p, ds, num_boost_round=6).model_to_string())
        second = _strip(flc.train({**p, "feature_fraction": 0.95}, ds, num_boost_round=6).model_to_string())
        models[plan] = (first, second)
    assert models["auto"] == models[OFF]
    assert models["auto"][0] != models["auto"][1]


def _train_with_reset(plan, quant="stochastic", rounds=5):
    """ResetTrainingData: the booster continues on a second Dataset with the same bin mappers but other rows."""
    X, y = _data("sampled", seed=0)
    X2, y2 = _data("sampled", seed=1)
    p = _params("sampled", quant, plan)
    train = flc.Dataset(X, label=y, params=p)
    bst = flc.Booster(p, train)
    for _ in range(rounds):
        bst.update()
    bst.update(train_set=flc.Dataset(X2, label=y2, params=p, reference=train))
    for _ in range(rounds - 1):
        bst.update()
    return _strip(bst.model_to_string())


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "none"])
def test_colmajor_reset_training_data_cuda(quant):
    """After ResetTrainingData the view is rebuilt from the new Dataset; colmajor_fill:off (no store at all) is the
    reference every arm must reproduce."""
    reference = _train_with_reset("auto,colmajor_fill:off", quant)
    for plan in ("auto", "auto,view_mode:mask", OFF):
        assert _train_with_reset(plan, quant) == reference, plan


_CHILD = """
import sys
sys.path.insert(0, {here!r})
import falcata as flc
from test_colmajor_direct import _data, _params, _strip
X, y = _data({shape!r})
p = {{**_params({shape!r}, "stochastic", {plan!r}), "verbosity": 1}}
train = flc.Dataset(X, label=y, params=p)
bst = flc.Booster(p, train)
for step in {steps!r}:
    if step == "reset":
        X2, y2 = _data({shape!r}, seed=1)
        print("STEP reset", flush=True)
        bst.update(train_set=flc.Dataset(X2, label=y2, params=p, reference=train))
        continue
    if step is not None:
        print("STEP", step, flush=True)
        bst.reset_parameter({{"feature_fraction": step}})
    for _ in range(3):
        bst.update()
print("MODEL", __import__("hashlib").md5(_strip(bst.model_to_string()).encode()).hexdigest(), flush=True)
"""

STORE = "colmajor_direct: compact regime"
MASK = "colmajor_direct: mask regime"
FULL_VIEW = "colmajor_direct: full view filled from the "
RELEASED = "colmajor_direct: full view released"


def _child(shape, plan="auto", steps=(None,), verify=False):
    code = _CHILD.format(here=os.path.dirname(os.path.abspath(__file__)), shape=shape, plan=plan, steps=list(steps))
    env = {**os.environ, "FALCATA_DEBUG": "diag"}
    if verify:
        env["FALCATA_VERIFY"] = "1"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout
    md5 = re.search(r"^MODEL (\w+)$", out, re.M).group(1)
    return out, md5


def _md5_off(shape, steps):
    return _child(shape, plan=OFF, steps=steps)[1]


@_REQUIRES_CUDA
@pytest.mark.parametrize("shape", ["sampled", "tall"])
def test_compact_regime_store_matches_the_columns_cuda(shape):
    """Low feature_fraction: the store is uploaded from the columns, equals the host reference, and no full view or
    row-major matrix is ever built."""
    out, md5 = _child(shape, verify=True)
    assert out.count(STORE) == 1, out[-3000:]
    assert "column-major store built from the Dataset's columns" in out
    assert "FALCATA_VERIFY: column-major 4-bit store matches the Dataset's columns" in out, out[-3000:]
    assert FULL_VIEW not in out, out[-3000:]
    assert MASK not in out, out[-3000:]
    assert md5 == _md5_off(shape, (None,))


@_REQUIRES_CUDA
@pytest.mark.parametrize("shape", ["ff1", "chunked"])
def test_mask_regime_holds_no_store_cuda(shape):
    """High feature_fraction: the full view is filled once from the columns (several staging chunks on the wide
    shape), byte-exact, and the store is never on the device."""
    out, md5 = _child(shape, verify=True)
    assert out.count(MASK) == 1, out[-3000:]
    assert "no column-major store on the device" in out
    assert "the full view filled from the Dataset's columns matches the host-packed 4-bit row data" in out
    assert STORE not in out, out[-3000:]
    assert FULL_VIEW not in out, out[-3000:]
    assert md5 == _md5_off(shape, (None,))


@_REQUIRES_CUDA
def test_regime_follows_feature_fraction_changes_cuda():
    """reset_parameter(feature_fraction) moves the matrix to the other regime: compact -> mask releases the store,
    mask -> compact releases the full view; the model matches colmajor_direct:off throughout."""
    steps = (None, 0.95, 0.2)
    out, md5 = _child("sampled", steps=steps)
    lines = [ln for ln in out.splitlines() if "regime (" in ln or ln.startswith("STEP")]
    assert len(lines) == 5, lines
    assert STORE in lines[0], lines
    assert lines[1] == "STEP 0.95", lines
    assert MASK in lines[2], lines
    assert "column-major store released" in lines[2], lines
    assert lines[3] == "STEP 0.2", lines
    assert STORE in lines[4], lines
    assert "full view released" in lines[4], lines
    assert md5 == _md5_off("sampled", steps)


@_REQUIRES_CUDA
@pytest.mark.parametrize("plan", ["auto", "auto,view_mode:mask"])
def test_regime_is_decided_again_on_update_train_set_cuda(plan):
    """update(train_set=...) replaces the row data: the regime is decided again and its view built from the new
    Dataset's columns."""
    steps = (None, "reset", None)
    out, md5 = _child("sampled", plan=plan, steps=steps)
    kind = MASK if "mask" in plan else STORE
    decisions = [ln for ln in out.splitlines() if "regime (" in ln or ln.startswith("STEP")]
    assert [kind in decisions[0], decisions[1], kind in decisions[2]] == [True, "STEP reset", True], decisions
    assert md5 == _md5_off("sampled", steps)


@_REQUIRES_CUDA
def test_compact_regime_full_view_cache_rule_cuda():
    """Forced compact: the full view lives while consecutive trees sample every column, is released on the first
    sampled tree, and filled again when the trees go back to every column."""
    # 0.999 rounds to every column; a reset to exactly 1.0 would keep the previous tree's sample (ColSampler::SetConfig
    # stops resampling at 1.0 without restoring the mask)
    steps = (None, 0.3, 0.999)
    out, md5 = _child("ff1", plan="auto,view_mode:compact", steps=steps)
    assert out.count(FULL_VIEW + "column-major store for a tree without a compact view") == 2, out[-3000:]
    assert out.count(RELEASED) == 1, out[-3000:]
    assert md5 == _md5_off("ff1", steps)


@_REQUIRES_CUDA
def test_compact_regime_pack_codec_split_view_cuda():
    """Codec trees: the fill reads the store; the split view builder takes the full view once and keeps it."""
    plan = "auto,pack_bit3:on"
    out, md5 = _child("sampled", plan=plan, steps=(None, None), verify=True)
    assert out.count(FULL_VIEW + "column-major store for a split view the compact matrix cannot serve") == 1
    assert RELEASED not in out, out[-3000:]
    assert md5 == _child("sampled", plan=plan + ",colmajor_direct:off", steps=(None, None))[1]


def test_view_mode_plan_keys_parse():
    """view_mode takes auto|compact|mask and view_mask_ff a number; anything else is refused (the plan is resolved
    at Dataset construction too, so this needs no GPU)."""
    X, y = _data("ff1")
    for bad in ("view_mode:sometimes", "view_mask_ff:high", "view_mask_ff:-1"):
        p = {**BASE, "device_type": "cpu", "cuda_plan": f"auto,{bad}"}
        with pytest.raises(flc.basic.FalcataError, match="cuda_plan: bad value"):
            flc.Dataset(X[:2000], label=y[:2000], params=p).construct()
