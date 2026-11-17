# coding: utf-8
"""CUDA training must give the same trees whatever multiprocessor count the card reports.

The launch-shape decisions of the CUDA tree learner that read the device's multiprocessor (SM) count read it through
include/Falcata/cuda/cuda_device_sm.hpp (FalcataSMSite): the batched construct's saturation floor, the fused root
fill's tiles per block, the device-level construct's target blocks, the count histogram's tiling, the split finder's
warps per wide task, the pair-joint construct's device slots and the live-grid budget. FALCATA_DEBUG=smcount=N makes
them see N multiprocessors, so one card runs the launch schedule that a card of the same architecture with N SMs
would run (an RTX 5090 has 170, an RTX PRO 6000 188). It emulates the SM count only, not the per-SM limits of
another architecture. The first override at each site is logged at Info level, and test_each_case_reaches_its_sites
checks from those lines which sites a case's training reaches, so a case that stops reaching a site is noticed. One
site, the pair-joint code view's shape, is reached by no case here (NOT_REACHED).
The token is read once per process, so every run here is a subprocess.

Measured 2026-11-17 for the PR that added the token: the same models on an RTX 5090 (170 SMs) and an RTX PRO 6000
Max-Q (188 SMs), and on each card under smcount=170 and 188.
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

rng = np.random.default_rng(5)
n, p = {rows}, {feats}
X = rng.standard_normal((n, p)).astype(np.float32)
X[:, : p // 4] = np.round(X[:, : p // 4] * 3) / 3
if {levels}:
    X = rng.integers(0, {levels}, size=(n, p)).astype(np.float32)
if {missing}:
    X[rng.random((n, p)) < 0.3] = np.nan
y = (X[:, :8] @ rng.standard_normal(8) * 0.05 + rng.standard_normal(n)).astype(np.float32)
w = (rng.gamma(2.0, 0.5, n) + 0.05).astype(np.float32) if {weighted} else None
params = {{"objective": "regression", "device_type": "cuda", "seed": 3, "verbosity": {verbosity},
           "min_data_in_leaf": 25, "learning_rate": 0.1, **{params!r}}}
ds = flc.Dataset(X, label=y, weight=w, params={{"max_bin": params.get("max_bin", 63), "device_type": "cuda"}})
bst = flc.train(params, ds, {rounds})
model = bst.model_to_string()
print("MODEL_MD5", hashlib.md5(model.encode()).hexdigest(), flush=True)
"""

# Each case lists the sites its training reaches on an RTX 5090 (measured from the per-site log lines with
# FALCATA_DEBUG=smcount=188; test_each_case_reaches_its_sites keeps the list true). Together they reach every site
# but NOT_REACHED.
CASES = {
    # the shape of a production fit that diverged across cards (later traced to its inputs): weighted rows, 30%
    # missing, bagging and column sampling, many leaves, fixed-point quantized gradients
    "fixedpoint_weighted": {
        "rows": 60_000,
        "feats": 120,
        "missing": True,
        "weighted": True,
        "rounds": 40,
        "params": {
            "quant_mode": "fixedpoint",
            "cuda_precision": "fp32",
            "num_leaves": 255,
            "max_bin": 51,
            "feature_fraction": 0.2,
            "bagging_fraction": 0.5,
            "bagging_freq": 1,
            "lambda_l2": 10.0,
        },
        "sites": {"level_target_blocks", "live_grid_budget", "saturation_floor"},
    },
    "stochastic_large": {
        "rows": 400_000,
        "feats": 60,
        "missing": True,
        "weighted": False,
        "rounds": 20,
        "params": {
            "quant_mode": "stochastic",
            "num_leaves": 127,
            "max_bin": 63,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
        },
        "sites": {"find_split_cap", "level_target_blocks", "live_grid_budget", "saturation_floor"},
    },
    "unquantized_fp32": {
        "rows": 400_000,
        "feats": 60,
        "missing": False,
        "weighted": True,
        "rounds": 20,
        "params": {"quant_mode": "none", "cuda_precision": "fp32", "num_leaves": 127, "max_bin": 63},
        "sites": {"count_tiles"},
    },
    "unquantized_fp64": {
        "rows": 200_000,
        "feats": 60,
        "missing": True,
        "weighted": True,
        "rounds": 15,
        "params": {"quant_mode": "none", "cuda_precision": "fp64", "num_leaves": 63, "max_bin": 63},
        "sites": {"count_tiles"},
    },
    # a random forest without quantization: the count histograms (cuda_plan key forest_count_hist)
    "random_forest_fp64": {
        "rows": 200_000,
        "feats": 60,
        "missing": False,
        "weighted": False,
        "rounds": 15,
        "params": {
            "boosting": "rf",
            "quant_mode": "none",
            "cuda_precision": "fp64",
            "num_leaves": 63,
            "max_bin": 63,
            "bagging_fraction": 0.6,
            "bagging_freq": 1,
            "feature_fraction": 0.5,
        },
        "sites": {"count_tiles"},
    },
    # 4-bit packed bins (max_bin 15) with quantized gradients and column sampling: the per-tree compact view, filled
    # by the fused root histogram fill, and the pair-joint construct on its 5-bit code view
    "fourbit_compact": {
        "rows": 300_000,
        "feats": 100,
        "missing": False,
        "weighted": False,
        "rounds": 15,
        "params": {"quant_mode": "fixedpoint", "num_leaves": 63, "max_bin": 15, "feature_fraction": 0.5},
        "sites": {"find_split_cap", "fused_root_fill", "level_target_blocks", "live_grid_budget", "saturation_floor"},
    },
    # the same with five levels per feature: the quantized pair_hist construct (it does not reach the pair-joint
    # code-view shape, NOT_REACHED)
    "fourbit_five_levels": {
        "rows": 300_000,
        "feats": 100,
        "levels": 5,
        "missing": False,
        "weighted": False,
        "rounds": 15,
        "params": {"quant_mode": "fixedpoint", "num_leaves": 63, "max_bin": 15, "feature_fraction": 0.5},
        "sites": {"fused_root_fill", "level_target_blocks", "saturation_floor"},
    },
    "wide_255_bins": {
        "rows": 60_000,
        "feats": 800,
        "missing": False,
        "weighted": False,
        "rounds": 15,
        "params": {"quant_mode": "fixedpoint", "num_leaves": 63, "max_bin": 255, "feature_fraction": 0.8},
        "sites": {"find_split_cap", "level_target_blocks", "live_grid_budget", "saturation_floor"},
    },
}

ALL_SITES = {
    "saturation_floor",
    "fused_root_fill",
    "level_target_blocks",
    "count_tiles",
    "find_split_cap",
    "pair_joint_shape",
    "live_grid_budget",
}

# Sites that no case reaches (searched 2026-11-17: 4-bit data with 3 to 5 levels per feature, 40 to 100 features, full
# and half column sampling, 63 and 255 leaves; the fill logs "pair_code5 off ... no code"). Not covered by this test.
NOT_REACHED = {"pair_joint_shape"}

# the real count, two cards of the architecture that ships (5090, PRO 6000), and the extremes of the schedule:
# one multiprocessor (every per-device bound at its minimum) and far more than any card has
SM_COUNTS = [None, 170, 188, 1, 2048]

_SITE_LINE = re.compile(r"FALCATA_DEBUG smcount: site (\w+) sees (\d+) multiprocessors \(device \d+ has \d+\)")


def _run(case, debug, verbosity=-1):
    """(model md5, child's stdout + stderr) of one training run in a fresh process."""
    c = CASES[case]
    code = _CHILD.format(**{"levels": 0, **{k: v for k, v in c.items() if k != "sites"}}, verbosity=verbosity)
    env = dict(os.environ)
    env.pop("FALCATA_DEBUG", None)
    if debug is not None:
        env["FALCATA_DEBUG"] = debug
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)
    log = r.stdout + r.stderr
    m = re.search(r"^MODEL_MD5 (\w+)$", r.stdout, re.M)
    if r.returncode != 0 or m is None:
        pytest.fail(f"{case} with FALCATA_DEBUG={debug!r}: exit {r.returncode}, no model\n{log[-4000:]}")
    return m.group(1), log


@_REQUIRES_CUDA
@pytest.mark.parametrize("case", sorted(CASES))
def test_trees_do_not_depend_on_the_sm_count_cuda(case):
    md5 = {n: _run(case, None if n is None else f"smcount={n}")[0] for n in SM_COUNTS}
    assert len(set(md5.values())) == 1, md5


@_REQUIRES_CUDA
@pytest.mark.parametrize("case", sorted(CASES))
def test_each_case_reaches_its_sites_cuda(case):
    _, log = _run(case, "smcount=188", verbosity=1)
    seen = dict(_SITE_LINE.findall(log))
    assert set(seen.values()) <= {"188"}, seen
    assert set(seen) == set(CASES[case]["sites"]), (sorted(seen), log[-4000:])


def test_the_cases_reach_every_site_but_the_declared_ones():
    reached = set().union(*(c["sites"] for c in CASES.values()))
    assert reached == ALL_SITES - NOT_REACHED, sorted(reached)


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("token", "warning"),
    [
        ("smcount=abc", 'smcount="abc" is not an integer'),
        ("smcount=12x", 'smcount="12x" is not an integer'),
        ("smcount=0", "smcount=0 clamped to 1"),
        ("smcount=999999", "smcount=999999 clamped to 65536"),
        ("smcount=188,smsites=zz", 'smsites="zz" is not a bit mask'),
        ("smcount=188,smsites=0", "smsites=0 applies smcount to no site"),
    ],
)
def test_bad_tokens_are_refused_loudly_cuda(token, warning):
    _, log = _run("fixedpoint_weighted", token, verbosity=1)
    assert warning in log, log[-4000:]
