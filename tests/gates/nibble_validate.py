"""GPU gate for nibble-packed device columns (kNibbleColumnBitType).

Runs standalone and self-asserts:

    python tests/gates/nibble_validate.py            # gate
    python tests/gates/nibble_validate.py baseline   # same, tagged for A/B

The checks are tolerance-based, never bit-exact. A CUDA model's md5 is not a
valid contract on this fork: gain math runs in fp32 and histogram bins are
accumulated by atomics, so summation order varies between launches and two
runs of one unchanged build already produce different trees. The md5 is still
printed as a fingerprint, but nothing compares it -- an md5 assertion here
fails on noise and says nothing about whether the nibble packing is read
correctly.

What a broken nibble read actually looks like is a signal collapse, not a
digit change: two bins fuse per byte and rows past (N+1)/2 read out of bounds,
which took cat+bagging correlation from 0.86 to 0.003. So the contracts are
prediction-level -- CPU-vs-CUDA agreement at max_bin=5, and a correlation floor
on the categorical+bagging path that exercises CopySubrow and the categorical
apply. Peak device memory is reported for A/B comparison (the feature exists
for that win) but not asserted, since its baseline is the other build's run.
"""

import hashlib
import os
import subprocess
import sys
import threading
import time

import numpy as np

import falcata as lgb

# CPU-vs-CUDA agreement at max_bin=5. Observed max delta is ~1e-07 (the fp32
# run-to-run noise floor, which the unpacked build shares); these bounds sit
# orders of magnitude above it and orders of magnitude below a fused-nibble
# misread.
CPU_CUDA_MIN_CORR = 0.9999
CPU_CUDA_MAX_DELTA_IN_SIGMA = 0.02
# cat+bagging correlation: 0.8598 healthy, 0.0031 with a byte-per-row kernel
# pointed at a nibble-packed buffer.
CAT_BAGGING_MIN_CORR = 0.80

failures = []


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        failures.append(name)


def used_mb():
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in out.stdout.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 2 and p[0].isdigit() and int(p[0]) == os.getpid():
            return int(p[1])
    return 0


tag = sys.argv[1] if len(sys.argv) > 1 else "nibble"
rng = np.random.default_rng(5)

# --- correctness: 4-bit (max_bin=5) must agree with the CPU learner
X = rng.integers(0, 5, size=(300_000, 80)).astype(np.float32)
y = X[:, :4].sum(axis=1) * 0.3 + rng.normal(size=300_000) * 0.1
base = {
    "objective": "regression",
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "learning_rate": 0.1,
    "verbosity": -1,
    "num_threads": 8,
    "seed": 3,
    "deterministic": True,
    "quant_mode": "none",
}
preds = {}
for dev in ("cpu", "cuda"):
    p = {**base, "max_bin": 5, "device_type": dev}
    b = lgb.train(p, lgb.Dataset(X, label=y, params=p), num_boost_round=25)
    preds[dev] = np.asarray(b.predict(X, raw_score=True))
    if dev == "cuda":
        model_body = b.model_to_string().split(chr(10) + "parameters:")[0]
        # fingerprint only -- see the module docstring on why this is not asserted
        print(f"{tag} model-md5 {hashlib.md5(model_body.encode()).hexdigest()[:12]}")
delta = np.abs(preds["cuda"] - preds["cpu"])
sigma = float(np.std(preds["cpu"]))
corr = float(np.corrcoef(preds["cuda"], preds["cpu"])[0, 1])
check(
    "4bit cpu-vs-cuda corr",
    corr >= CPU_CUDA_MIN_CORR,
    f"{corr:.6f} >= {CPU_CUDA_MIN_CORR}",
)
check(
    "4bit cpu-vs-cuda max delta",
    delta.max() <= CPU_CUDA_MAX_DELTA_IN_SIGMA * sigma,
    f"max={delta.max():.3e} <= {CPU_CUDA_MAX_DELTA_IN_SIGMA} * sigma({sigma:.4f}) "
    f"= {CPU_CUDA_MAX_DELTA_IN_SIGMA * sigma:.3e}",
)

# --- categorical + bagging exercises CopySubrow and the categorical apply,
# which read columns through GetColumnData: the path a nibble/byte bit-type
# mismatch destroys.
Xc = X.copy()
Xc[:, 0] = rng.integers(0, 5, size=300_000)
pc = {**base, "max_bin": 5, "device_type": "cuda", "bagging_fraction": 0.6, "bagging_freq": 1}
bc = lgb.train(pc, lgb.Dataset(Xc, label=y, params=pc, categorical_feature=[0]), num_boost_round=25)
pred_c = np.asarray(bc.predict(Xc))
cat_corr = float(np.corrcoef(pred_c, y)[0, 1])
check("cat+bagging finite", bool(np.all(np.isfinite(pred_c))), "all predictions finite")
check(
    "cat+bagging corr",
    cat_corr >= CAT_BAGGING_MIN_CORR,
    f"{cat_corr:.4f} >= {CAT_BAGGING_MIN_CORR}",
)

# --- memory: per-column arrays live below the 8GB skip threshold
peak = [0]
stop = [False]


def sample_peak():
    while not stop[0]:
        peak[0] = max(peak[0], used_mb())
        time.sleep(0.05)


threading.Thread(target=sample_peak, daemon=True).start()
Xm = rng.integers(0, 5, size=(3_000_000, 500)).astype(np.float32)
ym = Xm[:, 0] * 0.4 + rng.normal(size=3_000_000) * 0.1
pm = {**base, "max_bin": 5, "device_type": "cuda", "num_leaves": 63}
dm = lgb.Dataset(Xm, label=ym, params=pm)
dm.construct()
lgb.train(pm, dm, num_boost_round=5)
stop[0] = True
time.sleep(0.3)
print(f"{tag} 3M x 500 4-bit peak={peak[0]}MiB (per-column arrays = 1.4 GiB at 8-bit, 0.7 at 4-bit)")

if failures:
    print(f"{tag} FAILED: {', '.join(failures)}")
    sys.exit(1)
print(f"{tag} all nibble invariants pass")
