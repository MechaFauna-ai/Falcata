"""GPU A/B harness for nibble-packed device columns (kNibbleColumnBitType).

Run once per build with a tag argument, e.g.

    python tests/gates/nibble_validate.py baseline   # unpacked build
    python tests/gates/nibble_validate.py nibble     # packed build

and compare the printed lines: the model md5 and cpu-vs-cuda max deltas
must be identical across builds; only the peak-memory line should shrink.
"""

import hashlib
import os
import subprocess
import sys
import threading
import time

import numpy as np

import falcata as lgb


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


tag = sys.argv[1]
rng = np.random.default_rng(5)

# --- correctness: 4-bit (max_bin=5) must match CPU, and match an 8-bit-binned twin
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
        print(f"{tag} model-md5 {hashlib.md5(model_body.encode()).hexdigest()[:12]}")
d = np.abs(preds["cuda"] - preds["cpu"])
print(f"{tag} 4bit cpu-vs-cuda: max={d.max():.3e}")

# --- categorical + bagging exercises CopySubrow and the categorical apply
Xc = X.copy()
Xc[:, 0] = rng.integers(0, 5, size=300_000)
pc = {**base, "max_bin": 5, "device_type": "cuda", "bagging_fraction": 0.6, "bagging_freq": 1}
bc = lgb.train(pc, lgb.Dataset(Xc, label=y, params=pc, categorical_feature=[0]), num_boost_round=25)
pred_c = np.asarray(bc.predict(Xc))
print(f"{tag} cat+bagging: finite={np.all(np.isfinite(pred_c))} corr={np.corrcoef(pred_c, y)[0, 1]:.4f}")

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
