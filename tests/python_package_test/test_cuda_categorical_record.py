# coding: utf-8
"""CUDA categorical split record: the learner-owned growth buffers (CUDATreeCatArena) and what a tree keeps.

Each tree takes the learner's categorical record buffers (both bitsets and boundary arrays) over, appends one
bitset per categorical split, reads the record back in ToHost and hands the buffers back to the learner for the next
tree, keeping exact-size device copies of the inner bitset and boundaries. Those copies are what
AddPredictionToScore traverses for the tree's whole lifetime: the validation scores, the out-of-bag pass under
bagging and DART's re-scoring of dropped trees.

Per shape, the test checks that
- two trainings give the same model (md5 of the model text without its parameters),
- the model survives a text round trip (same text, same predictions),
- the scores the booster maintained on the device (validation set; the training set under bagging) equal the host
  prediction of the final model, i.e. every retained tree's device record matches its host record.
The shapes cover one-hot and many-vs-many splits, a high-cardinality feature whose per-tree record exceeds the
1 MiB presize, multiclass, DART, bagging and a random forest grown on concurrent forest slots (one learner, so one
arena, per slot and host thread) and solo (one-tree groups).

Every shape is trained quantized (stochastic) and non-quantized. Quantized, every depth-limited shape grows on the
complete device tree: its select lists each level's categorical splits, one
kernel per level builds their bitsets, the apply reads them there and the tree records the level with one kernel at
offsets summed on the device, into record buffers presized for the tree's worst case; the record is read back once
per tree. The host level loop (cuda_plan key device_tree:off, also the forest slots' flow) records each level with
CUDATree::SplitLevel, growing its buffers when a level's worst case passes their capacity; every shape is also
trained there and must give the same model text and predictions. Non-quantized, the trees grow on the one-sync host
level loop (reason `categorical`), which records by level with SplitLevel too. max_cat_32 and max_cat_64 have a categorical
feature whose largest value is a multiple of 32 (its bitset needs max_cat_value / 32 + 1 words; with values 0..64 a
bitset sized one word short failed training with an invalid-argument CUDA copy). FALCATA_DEBUG=diag reports per tree
what the level record did; the test checks that the device tree grew the tree and the fast path ran: one tree-end
copy, no growth inside a tree, and on the high-cardinality shape the four-copy readback of a record bound past the
one-copy limit (and, on the host level loop, the exact-count readback on a capacity overflow). The presized record
is fixed scratch of the device tree's memory plan: when it does not fit beside one pair's search in half of free
device memory, the tree takes the host level loop with reason `scratch_too_large` and the same model; a re-plan (a
parameter reset) counts an allocated record once.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

#: CUDATreeCatArena presize cap in words (cuda_single_gpu_tree_learner.cpp AllocateBitset)
_PRESIZE_CAP_WORDS = 1 << 18


def _data(rows, seed, cards=(12, 40, 200), high=None, num_class=2, exact_max=None):
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, c, rows).astype(np.float64) for c in cards]
    if exact_max is not None:
        # every value 0..exact_max occurs
        cols.append(rng.integers(0, exact_max + 1, rows).astype(np.float64))
    if high is not None:
        # 200 distinct category values spread up to `high`: the real-value bitset of a split holding the largest
        # ones is ~high/32 words long
        values = np.random.default_rng(99).choice(high, 200, replace=False)
        cols.append(rng.choice(values, rows).astype(np.float64))
    num = rng.standard_normal((rows, 4))
    effect = num[:, 0] - 0.5 * num[:, 1]
    for j, c in enumerate(cols):
        _, codes = np.unique(c, return_inverse=True)
        effect += np.random.default_rng(10 + j).standard_normal(codes.max() + 1)[codes]
    if num_class > 2:
        y = np.digitize(effect, np.quantile(effect, np.linspace(0, 1, num_class + 1)[1:-1])).astype(np.float64)
    else:
        y = (effect + 0.5 * rng.standard_normal(rows) > 0).astype(np.float64)
    return np.column_stack([*cols, num]), y, list(range(len(cols)))


def _strip_parameters(model_str):
    out, skip = [], False
    for line in model_str.splitlines():
        if line.startswith("parameters:"):
            skip = True
        if not skip:
            out.append(line)
        if line.startswith("end of parameters"):
            skip = False
    return "\n".join(out)


def _md5(text):
    return hashlib.md5(_strip_parameters(text).encode()).hexdigest()


BASE = {
    "device_type": "cuda",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "max_depth": 5,
    "max_bin": 63,
    "min_data_in_leaf": 20,
    "learning_rate": 0.1,
    "seed": 7,
    "verbose": -1,
}
SHAPES = {
    "one_hot": ({"objective": "regression", "metric": "l2", "max_cat_to_onehot": 256}, {}, 20),
    "many_vs_many": ({}, {}, 20),
    "high_cardinality": (
        {"num_leaves": 63, "max_depth": 6, "max_cat_threshold": 128, "min_data_per_group": 5, "cat_smooth": 1},
        {"high": 2_000_000},
        3,
    ),
    "multiclass": ({"objective": "multiclass", "num_class": 3, "metric": "multi_logloss"}, {"num_class": 3}, 15),
    "dart": ({"boosting": "dart", "drop_rate": 0.3}, {}, 20),
    "bagging": ({"bagging_fraction": 0.7, "bagging_freq": 1, "metric": "auc"}, {}, 20),
    "rf_forest_slots": (
        {"boosting": "rf", "bagging_fraction": 0.632, "bagging_freq": 1, "cuda_plan": "auto,forest_group:4"},
        {},
        12,
    ),
    "rf_solo": (
        {"boosting": "rf", "bagging_fraction": 0.632, "bagging_freq": 1, "cuda_plan": "auto,forest_group:1"},
        {},
        12,
    ),
    "max_cat_32": ({"max_cat_to_onehot": 1}, {"cards": (12, 40), "exact_max": 32}, 20),
    "max_cat_64": ({"max_cat_to_onehot": 1}, {"cards": (65,), "exact_max": 64}, 20),
}

_DIAG = re.compile(
    r"cat_level_record: levels=(\d+) cat_splits=(\d+) record_launches=\d+ saved_launches=\d+ saved_d2d=\d+ "
    r"saved_length_readbacks=\d+ tohost_copies=(\d+) overflow_readbacks=(\d+)"
)
_ROUTE = re.compile(r"\[hybrid-diag\] device_tree: (?:(complete) |fallback \((\w+)\) route=(\w+))")
_DIAG_TRAIN = """
import hashlib, json, sys
import numpy as np
import falcata as flc
d = np.load(sys.argv[1])
params = json.loads(sys.argv[2])
ds = flc.Dataset(d["x"], d["y"], categorical_feature=[int(c) for c in d["cats"]], params=params)
bst = flc.train(params, ds, num_boost_round=int(sys.argv[3]))
text = bst.model_to_string()
out, skip = [], False
for line in text.splitlines():
    if line.startswith("parameters:"):
        skip = True
    if not skip:
        out.append(line)
    if line.startswith("end of parameters"):
        skip = False
print("MD5 " + hashlib.md5("\\n".join(out).encode()).hexdigest())
"""


def _diag_run(params, data_kw, rounds, debug="diag"):
    """A training under FALCATA_DEBUG=<debug> in a fresh process (the debug options are read once per process):
    ((levels, cat_splits, tohost_copies, overflow_readbacks) per tree, the routes reported in first-seen order
    (("complete",) or (reason, route)), the model md5)."""
    x, y, cats = _data(20000, 3, **data_kw)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.npz")
        np.savez(path, x=x, y=y, cats=np.array(cats))
        env = dict(os.environ, FALCATA_DEBUG=debug)
        out = subprocess.run(
            [sys.executable, "-c", _DIAG_TRAIN, path, json.dumps(params), str(rounds)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    lines = [tuple(int(v) for v in m.groups()) for m in _DIAG.finditer(out.stderr)]
    routes = []
    for complete, reason, route in _ROUTE.findall(out.stderr):
        entry = ("complete",) if complete else (reason, route)
        if entry not in routes:
            routes.append(entry)
    md5 = next(line[4:] for line in out.stdout.splitlines() if line.startswith("MD5 "))
    return lines, routes, md5


def _train(params, data_kw, rounds):
    x, y, cats = _data(20000, 3, **data_kw)
    xv, yv, _ = _data(5000, 4, **data_kw)
    ds = flc.Dataset(x, y, categorical_feature=cats, params=params, free_raw_data=False)
    dv = flc.Dataset(xv, yv, reference=ds, categorical_feature=cats)
    bst = flc.train(params, ds, num_boost_round=rounds, valid_sets=[dv], valid_names=["v"], keep_training_booster=True)
    return bst, x, xv


def _device_scores(bst, data_idx, rows, num_class):
    """The scores the booster maintained on the device, in output space, shaped like Booster.predict."""
    score = bst._Booster__inner_predict(data_idx=data_idx)
    return score.reshape(rows, num_class) if num_class > 1 else score


def _per_tree_words(text, key):
    return [len(line.split("=", 1)[1].split()) for line in text.splitlines() if line.startswith(key + "=")]


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "none"])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_cuda_categorical_record(shape, quant):
    extra, data_kw, rounds = SHAPES[shape]
    params = {**BASE, "quant_mode": quant, **extra}
    bst, x, xv = _train(params, data_kw, rounds)
    text = bst.model_to_string()
    assert sum(int(line[8:]) for line in text.splitlines() if line.startswith("num_cat=")) > 0, "no categorical split"

    # repeated training: the reused buffers carry no state from one training (or tree) into the model
    again, _, _ = _train(params, data_kw, rounds)
    assert _md5(again.model_to_string()) == _md5(text)

    # text round trip
    loaded = flc.Booster(model_str=text)
    assert _md5(loaded.model_to_string()) == _md5(text)
    np.testing.assert_array_equal(loaded.predict(xv), bst.predict(xv))

    # device-maintained scores (retained inner bitsets) == host prediction of the final model
    num_class = params.get("num_class", 1)
    # (both in output space: the booster's inner prediction applies the objective's output transform)
    np.testing.assert_allclose(_device_scores(bst, 1, xv.shape[0], num_class), bst.predict(xv), rtol=1e-9, atol=1e-12)
    if params.get("bagging_freq", 0) > 0 and params.get("boosting") != "rf":
        # the out-of-bag rows are scored through AddPredictionToScore on the finished tree
        np.testing.assert_allclose(_device_scores(bst, 0, x.shape[0], num_class), bst.predict(x), rtol=1e-9, atol=1e-12)

    if shape == "high_cardinality":
        # one tree's real-value record is longer than the presize, so the record buffers hold more than it
        assert max(_per_tree_words(text, "cat_threshold")) > _PRESIZE_CAP_WORDS
    # the host level loop (cuda_plan key device_tree:off; SplitLevel per level) gives the same model
    plan = params.get("cuda_plan", "auto")
    host_loop, _, _ = _train({**params, "cuda_plan": plan + ",device_tree:off"}, data_kw, rounds)
    assert _md5(host_loop.model_to_string()) == _md5(text)
    np.testing.assert_array_equal(host_loop.predict(xv), bst.predict(xv))

    # the level record ran (FALCATA_DEBUG=diag): a silent fallback or the four-copy readback fails
    lines, routes, _ = _diag_run(params, data_kw, min(rounds, 4))
    assert lines, "no tree took the level record"
    assert all(levels > 0 and splits > 0 for levels, splits, _, _ in lines)
    if shape == "rf_forest_slots":
        assert routes == [("forest_slots", "forest_slots")], routes
    elif quant == "none":
        # the non-quantized one-sync host loop (its SplitLevel record grows past its capacity on high cardinality)
        assert routes == [("categorical", "one_sync")], routes
        if shape == "high_cardinality":
            assert any(overflow > 0 for _, _, _, overflow in lines)
            assert any(copies == 4 for _, _, copies, _ in lines)
        else:
            assert all(copies == 1 and overflow == 0 for _, _, copies, overflow in lines), lines
        return
    else:
        # quantized, every depth-limited shape grows on the complete device tree
        assert routes == [("complete",)], routes
    # the device tree's record is presized for the whole tree: nothing grows inside it
    assert all(overflow == 0 for _, _, _, overflow in lines), lines
    if shape == "high_cardinality":
        # the stride-bound words pass the one-copy limit
        assert any(copies == 4 for _, _, copies, _ in lines)
        # the host level loop grows its buffers past their capacity while a tree records (exact-count readback)
        host_lines, host_routes, _ = _diag_run({**params, "cuda_plan": "auto,device_tree:off"}, data_kw, 3)
        assert host_routes == [("plan_off", "host_two_sync")], host_routes
        assert any(overflow > 0 for _, _, _, overflow in host_lines)
        assert any(copies == 4 for _, _, copies, _ in host_lines)
    else:
        assert all(copies == 1 for _, _, copies, _ in lines), lines

    if shape == "rf_forest_slots":
        # concurrent slots grow the same forest as sequential training
        seq, _, _ = _train({**params, "cuda_plan": "auto,forest_batched:off"}, data_kw, rounds)
        assert _md5(seq.model_to_string()) == _md5(text)


@_REQUIRES_CUDA
def test_cuda_categorical_record_memory_fallback():
    """The device tree presizes the tree's categorical record for its worst case (every split categorical at the
    bitset stride: about 16 MB on the high-cardinality shape), as part of its fixed scratch. Under a device memory cap
    (FALCATA_DEBUG=vramfree=N MiB) in which that does not fit beside one pair's search scratch in half of the free
    memory, the tree takes the host level loop (reason scratch_too_large), and the model is the device tree's."""
    extra, data_kw, _ = SHAPES["high_cardinality"]
    params = {**BASE, "quant_mode": "stochastic", **extra}
    _, routes, md5 = _diag_run(params, data_kw, 3)
    assert routes == [("complete",)], routes
    _, capped_routes, capped_md5 = _diag_run(params, data_kw, 3, debug="diag,vramfree=16")
    assert capped_routes == [("scratch_too_large", "host_two_sync")], capped_routes
    assert capped_md5 == md5


_CHUNK_TRAIN = """
import hashlib, sys
import numpy as np
import falcata as flc
rng = np.random.default_rng(17)
rows = 20000
cats = [rng.integers(0, c, rows) for c in (12, 40, 200)]
num = rng.standard_normal((rows, 200))
effect = num[:, 0] - 0.5 * num[:, 1] + 0.3 * num[:, 2:20].sum(axis=1)
for j, c in enumerate(cats):
    effect += np.random.default_rng(10 + j).standard_normal(c.max() + 1)[c]
x = np.column_stack([*[c.astype(np.float64) for c in cats], num])
y = (effect + 0.5 * rng.standard_normal(rows) > 0).astype(np.float64)
params = {"device_type": "cuda", "objective": "binary", "num_leaves": 1023, "max_depth": 10, "max_bin": 63,
          "min_data_in_leaf": 5, "quant_mode": "stochastic", "seed": 7, "verbose": -1}
ds = flc.Dataset(x, y, categorical_feature=[0, 1, 2], params=params)
bst = flc.train(params, ds, num_boost_round=3)
text = bst.model_to_string().split("parameters:")[0]
assert sum(int(line[8:]) for line in text.splitlines() if line.startswith("num_cat=")) > 0
print("MD5 " + hashlib.md5(text.encode()).hexdigest())
"""


def _chunk_run(debug):
    out = subprocess.run(
        [sys.executable, "-c", _CHUNK_TRAIN],
        env=dict(os.environ, FALCATA_DEBUG=debug),
        capture_output=True,
        text=True,
        check=True,
    )
    chunks = [int(m) for m in re.findall(r"searched pairs, (\d+) chunks of", out.stderr)]
    completes = len(re.findall(r"device_tree: complete ", out.stderr))
    md5 = next(line[4:] for line in out.stdout.splitlines() if line.startswith("MD5 "))
    return chunks, completes, md5


@_REQUIRES_CUDA
def test_cuda_categorical_record_pair_chunks():
    """Under a device memory cap the device tree searches each wide level in pair chunks (PlanDeviceTreeChunks); the
    categorical list, bitset slab and level record are per level and presized, so a chunked categorical tree records
    the same model as the unchunked one."""
    chunks, completes, md5 = _chunk_run("diag")
    assert chunks == [1], chunks
    assert completes == 3
    capped_chunks, capped_completes, capped_md5 = _chunk_run("diag,vramfree=8")
    assert len(capped_chunks) == 1, capped_chunks
    assert capped_chunks[0] > 1, capped_chunks
    assert capped_completes == 3
    assert capped_md5 == md5


@_REQUIRES_CUDA
def test_cuda_categorical_record_stump():
    """Quantized categorical stumps (num_leaves 2) grow as depth-1 complete device trees (the growth depth of a
    quantized stump is 1) and record their categorical root split by level, with the host level loop's model."""
    params = {**BASE, "quant_mode": "stochastic", "num_leaves": 2, "max_depth": -1}
    lines, routes, md5 = _diag_run(params, {}, 6)
    assert routes == [("complete",)], routes
    assert any(splits > 0 for _, splits, _, _ in lines), lines
    _, host_routes, host_md5 = _diag_run({**params, "cuda_plan": "auto,device_tree:off"}, {}, 6)
    assert host_routes[0][0] == "plan_off", host_routes
    assert host_md5 == md5


_REPLAN_TRAIN = """
import json, sys
import numpy as np
import falcata as flc
d = np.load(sys.argv[1])
params = json.loads(sys.argv[2])
ds = flc.Dataset(d["x"], d["y"], categorical_feature=[int(c) for c in d["cats"]], params=params)
# a parameter reset every round: ResetConfig releases the device levels and the next tree plans its scratch again
flc.train(params, ds, num_boost_round=4, callbacks=[flc.reset_parameter(learning_rate=[0.1, 0.09, 0.08, 0.07])])
"""


@_REQUIRES_CUDA
def test_cuda_categorical_record_replan_counts_record_once():
    """PlanDeviceTreeChunks counts the categorical record's growth where the record buffers are at plan time (the
    tree being grown holds the learner's arena then): the first plan counts the whole high-cardinality record, a
    re-plan after a parameter reset counts nothing for it (counted twice, an allocated record could refuse a plan the
    first one accepted)."""
    extra, data_kw, _ = SHAPES["high_cardinality"]
    params = {**BASE, "quant_mode": "stochastic", **extra}
    x, y, cats = _data(20000, 3, **data_kw)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.npz")
        np.savez(path, x=x, y=y, cats=np.array(cats))
        out = subprocess.run(
            [sys.executable, "-c", _REPLAN_TRAIN, path, json.dumps(params)],
            env=dict(os.environ, FALCATA_DEBUG="diag"),
            capture_output=True,
            text=True,
            check=True,
        )
    to_allocate = [
        float(m) for m in re.findall(r"categorical record estimate [\d.]+ MiB \(to allocate ([\d.]+),", out.stderr)
    ]
    assert len(to_allocate) >= 2, out.stderr[-2000:]
    assert to_allocate[0] > 10.0, to_allocate
    assert all(mib < 0.5 for mib in to_allocate[1:]), to_allocate
    assert "scratch too large" not in out.stderr
