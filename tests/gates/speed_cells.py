"""The fixed cell set of the pre-merge speed gate (speed_gate.py).

gates/*   the bench tier's real datasets and parameters (bench_tier.py), the Numerai cells on the first million
          training rows so that one run stays at a few seconds (the gate runs every cell 10-24 times)
h60/*     the production h60 sparse shape from the frozen binary dataset (saved_dataset.frozen_h60; needs the pinned
          manifest, as bench_tier.py)
v6like/*  tree fits shaped like a downstream user's walk-forward run (numerCrypto run v6, 2026-11): multiclass with
          3-16 classes, fixed-point quantized gradients with fp32 sums, strong L2, 63-6,000 float32 columns with 7%
          missing values, feature fractions down to 0.03 (the wide-sample compact skip of #613), bagging. The columns
          are factor-structured synthetic data generated once per machine (numpy, seed 0) into $FALCATA_SPEED_DATA
          (default ~/.cache/falcata-gates/speed-data) and memory-mapped by every run, so both arms read the same bits.

A cell whose data is missing is reported by `missing()` and listed as not run in the gate's summary; it is never
replaced by another cell.
"""

import json
import os
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("FALCATA_BENCH_CACHE", "bench-cache"))
DATA = Path(os.environ.get("FALCATA_SPEED_DATA", Path.home() / ".cache" / "falcata-gates" / "speed-data"))
GEN_VERSION = 1
BASE = {"device_type": "cuda", "seed": 42, "verbose": -1, "metric": "None", "num_threads": min(32, os.cpu_count() or 1)}


class Cell:
    def __init__(self, params, rounds, load, construct=None, test=None, needs=()):
        self.params, self.rounds, self.load = {**BASE, **params}, rounds, load
        self.construct = construct or _construct
        self.test = test or (lambda d: d.get("Xt"))
        self.needs = needs

    def missing(self):
        for n in self.needs:
            if callable(n):
                why = n()
                if why:
                    return why
            elif not Path(n).exists():
                return f"missing {n}"
        return None


def _construct(flc, data, params):
    """As bench_tier.py: the Dataset takes the training parameters."""
    ds = flc.Dataset(data["X"], label=data["y"], params=params)
    ds.construct()
    return ds, None


def _construct_v6(flc, data, params):
    """As the downstream walk-forward trainer (presage.models.wf): binning parameters only, weights, raw data kept."""
    dp = {"max_bin": params["max_bin"], "feature_pre_filter": False, "device_type": "cuda"}
    ds = flc.Dataset(data["X"], label=data["y"], weight=data["w"], params=dp, free_raw_data=False)
    ds.construct()
    return ds, None


def _cache(name):
    def load():
        d = CACHE / name
        return {"X": np.load(d / "X_train.npy"), "y": np.load(d / "y_train.npy"), "Xt": np.load(d / "X_test.npy")}

    return load


def _numerai(rows):
    def load():
        d = CACHE / "numerai"
        meta = json.loads((d / "meta.json").read_text())
        n, m, tr = meta["n_rows"], meta["n_features"], meta["train_end"]
        X = np.memmap(d / "X.i8.mem", dtype=np.int8, mode="r", shape=(n, m))
        y = np.load(d / "y.npy")
        return {"X": np.ascontiguousarray(X[:rows]), "y": y[:rows], "Xt": np.asarray(X[tr : tr + 50_000], np.float32)}

    return load


def _h60_missing():
    if not os.environ.get("FALCATA_H60_MANIFEST") or not os.environ.get("FALCATA_H60_MANIFEST_SHA256"):
        return "FALCATA_H60_MANIFEST / FALCATA_H60_MANIFEST_SHA256 not set"
    return None


def _h60_load():
    from saved_dataset import frozen_h60

    return frozen_h60()


def _h60_construct(flc, frozen, params):
    from saved_dataset import check_loaded_h60

    ds = flc.Dataset(str(frozen["path"]), params={**params, **frozen["params"]})
    ds.construct()
    check_loaded_h60(ds, frozen)
    ds.set_label(frozen["y"])
    return ds, None


def synthetic(tag, rows, cols, classes, nan=0.07):
    """Factor-structured float32 columns (8 factors + noise), `nan` missing, a rank-bucketed label with `classes`
    classes (a centred rank for 1), weights in [0.5, 1.5). Written once to DATA/<tag>_v<GEN_VERSION>_*.npy."""
    stem = DATA / f"{tag}_v{GEN_VERSION}"
    fx, fy, fw = (Path(f"{stem}_{s}.npy") for s in ("X", "y", "w"))
    if not fw.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        fac = rng.standard_normal((rows, 8), dtype=np.float32)
        tmp = Path(f"{stem}_X.tmp.npy")
        X = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(rows, cols))
        for c in range(0, cols, 256):
            k = min(256, cols - c)
            blk = fac @ (rng.standard_normal((8, k), dtype=np.float32) * 0.5)
            blk += rng.standard_normal((rows, k), dtype=np.float32)
            blk[rng.random((rows, k), dtype=np.float32) < nan] = np.nan
            X[:, c : c + k] = blk
        sig = np.nan_to_num(X[:, : min(20, cols)]).sum(1) * 0.05 + fac[:, 0] * 0.3
        sig += rng.standard_normal(rows, dtype=np.float32)
        X.flush()
        del X
        os.replace(tmp, fx)
        r = np.argsort(np.argsort(sig, kind="stable"), kind="stable") / rows
        y = np.minimum((r * classes).astype(np.int64), classes - 1).astype(np.float32) if classes > 1 else r - 0.5
        np.save(fy, y.astype(np.float32))
        np.save(fw, (0.5 + rng.random(rows, dtype=np.float32)).astype(np.float32))
    X = np.load(fx, mmap_mode="r")
    return {"X": np.ascontiguousarray(X), "y": np.load(fy), "w": np.load(fw), "Xt": np.asarray(X[:50_000])}


V6 = {
    "lambda_l2": 163.0,
    "cuda_precision": "fp32",
    "quant_mode": "fixedpoint",
    "seed": 341791130,
    "bagging_seed": 341791130,
    "feature_fraction_seed": 341791130,
}


def cells():
    out = {}
    q = {"objective": "multiclass", "num_class": 7, "learning_rate": 0.1, "max_bin": 255, "lambda_l2": 1.0}
    out["gates/covtype-deep-quant"] = Cell(
        {**q, "num_leaves": 1023, "max_depth": 10, "use_quantized_grad": True},
        100,
        _cache("covtype"),
        needs=[CACHE / "covtype"],
    )
    out["gates/covtype-shallow-nonquant"] = Cell(
        {**q, "num_leaves": 63, "max_depth": 6}, 100, _cache("covtype"), needs=[CACHE / "covtype"]
    )
    out["gates/year-shallow-fp32"] = Cell(
        {
            "objective": "regression",
            "learning_rate": 0.1,
            "num_leaves": 63,
            "max_depth": 6,
            "max_bin": 255,
            "cuda_precision": "fp32",
        },
        100,
        _cache("year"),
        needs=[CACHE / "year"],
    )
    nm = {
        "objective": "regression",
        "learning_rate": 0.01,
        "num_leaves": 32,
        "max_depth": 5,
        "max_bin": 255,
        "feature_fraction": 0.1,
    }
    out["gates/numerai1m-example-quant"] = Cell(
        {**nm, "use_quantized_grad": True}, 100, _numerai(1_000_000), needs=[CACHE / "numerai" / "X.i8.mem"]
    )
    out["gates/numerai1m-nonquant-fp32"] = Cell(
        {**nm, "cuda_precision": "fp32"}, 100, _numerai(1_000_000), needs=[CACHE / "numerai" / "X.i8.mem"]
    )
    out["h60/numerai-h60-sparse"] = Cell(
        {
            "objective": "regression",
            "learning_rate": 0.003375,
            "num_leaves": 250,
            "max_depth": 12,
            "min_data_in_leaf": 40_000,
            "feature_fraction": 0.15,
            "max_bin": 5,
        },
        60,
        _h60_load,
        construct=_h60_construct,
        test=lambda d: None,
        needs=[_h60_missing],
    )
    # (tag, rows, columns, feature fraction, classes, leaves, max_bin, min_data_in_leaf, bagging, iterations)
    for tag, rows, cols, ff, k, leaves, mb, ml, bag, it in [
        ("f91-5class-2568col", 727_000, 2568, 0.594, 5, 6, 126, 231, 1.0, 40),
        ("f17-9class-2000col-bag", 735_000, 2000, 0.4375, 9, 21, 69, 182, 0.83, 30),
        ("f81-16class-222col", 693_000, 222, 0.78, 16, 40, 64, 3756, 1.0, 30),
        ("f131-3class-63col", 704_000, 63, 0.96, 3, 19, 81, 450, 1.0, 300),
        ("f27-5class-122col-bag", 178_000, 122, 0.38, 5, 10, 32, 500, 0.93, 300),
        ("f86-3class-6000col-ff033", 250_000, 6000, 0.033, 3, 4, 204, 1000, 1.0, 300),
    ]:
        p = {
            **V6,
            "objective": "multiclass",
            "num_class": k,
            "learning_rate": 0.004,
            "num_leaves": leaves,
            "min_data_in_leaf": ml,
            "feature_fraction": ff,
            "max_bin": mb,
        }
        if bag < 1:
            p.update(bagging_fraction=bag, bagging_freq=1)
        out[f"v6like/{tag}"] = Cell(
            p, it, (lambda t=tag, r=rows, c=cols, kk=k: synthetic(t, r, c, kk)), construct=_construct_v6
        )
    return out
