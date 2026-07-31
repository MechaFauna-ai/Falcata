"""Top-4 lever measurement battery. Run when Felix frees the GPU:
  .gates-venv/bin/python top4_measure.py
Bit-identity REQUIRED for every key (all are equality-class)."""
import hashlib, json, time
import numpy as np, falcata as flc

CACHE = "/home/felixjk/Documents/exaboost-bench/data/cache"

def run(dataset, params, rounds, plan, reps=3):
    if dataset == "numerai":
        meta = json.load(open(f"{CACHE}/numerai/meta.json"))
        n, m, tr = meta["n_rows"], meta["n_features"], meta["train_end"]
        X = np.memmap(f"{CACHE}/numerai/X.i8.mem", dtype=np.int8, mode="r", shape=(n, m))[:tr]
        y = np.load(f"{CACHE}/numerai/y.npy")[:tr]
    else:
        X = np.load(f"{CACHE}/{dataset}/X_train.npy")
        y = np.load(f"{CACHE}/{dataset}/y_train.npy")
    p = {**params, "cuda_plan": plan, "device_type": "cuda", "seed": 42,
         "verbose": -1, "metric": "None", "num_threads": 32}
    ts, md5 = [], None
    for _ in range(reps):
        ds = flc.Dataset(X, label=y, params=p)
        t0 = time.perf_counter()
        bst = flc.train(p, ds, num_boost_round=rounds)
        ts.append(rounds / (time.perf_counter() - t0))
        if md5 is None:
            md5 = hashlib.md5(bst.model_to_string().split("\nparameters:")[0].encode()).hexdigest()[:12]
    return md5, sorted(ts)[len(ts)//2]

DEEP = {"objective": "regression", "learning_rate": 0.001, "num_leaves": 1024,
        "max_depth": 10, "max_bin": 255, "feature_fraction": 0.1,
        "min_data_in_leaf": 10000, "quant_mode": "stochastic"}
COV = {"objective": "multiclass", "num_class": 7, "learning_rate": 0.1,
       "num_leaves": 1023, "max_depth": 10, "max_bin": 255, "lambda_l2": 1.0,
       "quant_mode": "stochastic"}
YEAR = {"objective": "regression", "learning_rate": 0.1, "num_leaves": 63,
        "max_depth": 6, "max_bin": 255, "quant_mode": "stochastic"}

CELLS = [("numerai", DEEP, 300), ("covtype", COV, 100), ("year", YEAR, 100)]
KEYS = ["l2_policy", "colmajor_fill", "tuner", "wide_partitions"]

for dataset, params, rounds in CELLS:
    base_md5, base_t = run(dataset, params, rounds, "auto")
    print(f"\n== {dataset} baseline {base_t:.1f} t/s md5={base_md5}", flush=True)
    for key in KEYS:
        md5, t = run(dataset, params, rounds, f"auto,{key}:on")
        tag = "OK-identical" if md5 == base_md5 else ("TUNER-OK" if key == "tuner" and md5 == base_md5 else "*** MD5 DIFFERS ***")
        if key == "tuner" and md5 != base_md5:
            tag = "*** MD5 DIFFERS (tuner must be invariant under quant!) ***"
        print(f"  {key:16s} {t:6.1f} t/s ({(t/base_t-1)*100:+5.1f}%) {tag}", flush=True)
    md5, t = run(dataset, params, rounds, "auto," + ",".join(f"{k}:on" for k in KEYS))
    print(f"  {'ALL FOUR':16s} {t:6.1f} t/s ({(t/base_t-1)*100:+5.1f}%) "
          f"{'OK-identical' if md5 == base_md5 else '*** MD5 DIFFERS ***'}", flush=True)
print("\nTOP4 MEASURE DONE — now run: .gates-venv/bin/python tests/gates/lattice.py --check", flush=True)
