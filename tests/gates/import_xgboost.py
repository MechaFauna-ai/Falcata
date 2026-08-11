"""XGBoost import gate: converted models must predict what XGBoost predicts.

Covers every supported objective (including missing values and multiclass),
the accepted input forms, interop with FALB, and -- as important -- that the
unsupported cases REFUSE explicitly instead of silently producing a model that
predicts something else.

Inputs are float32 because that is what XGBoost consumes: DMatrix truncates to
float32, so comparing on float64 data would measure that truncation rather
than the conversion (see falcata/importers.py).
"""

import sys

import numpy as np
import xgboost as xgb

from falcata.importers import from_xgboost


def case(name, objective, num_class=0, nan_frac=0.0, depth=5, rounds=25, n=4000, m=8):
    rng = np.random.default_rng(abs(hash(name)) % 10_000)
    # XGBoost consumes float32 (DMatrix truncates); feed both engines the
    # SAME values so the comparison is apples to apples
    X = rng.standard_normal((n, m)).astype(np.float32)
    if nan_frac:
        X[rng.random((n, m)) < nan_frac] = np.nan
    lin = np.nan_to_num(X) @ rng.standard_normal(m)
    if objective.startswith("multi"):
        y = (np.nan_to_num(X) @ rng.standard_normal((m, num_class))).argmax(1).astype(float)
    elif objective.startswith("binary") or objective == "reg:logistic":
        y = (lin > 0).astype(float)
    elif objective == "count:poisson":
        y = rng.poisson(np.exp(np.clip(lin * 0.2, -2, 2))).astype(float)
    else:
        y = lin + 0.3 * rng.standard_normal(n)
    params = {"objective": objective, "max_depth": depth, "eta": 0.3, "seed": 1}
    if num_class:
        params["num_class"] = num_class
    bst = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=rounds)
    ref = bst.predict(xgb.DMatrix(X))
    got = from_xgboost(bst).predict(X)
    got = np.asarray(got)
    if ref.ndim == 2 and got.ndim == 2:
        pass
    elif ref.ndim == 2:
        got = got.reshape(ref.shape)
    scale = max(1e-9, float(np.abs(ref).max()))
    rel = float(np.abs(got - ref).max() / scale)
    ok = rel < 1e-6
    print(f"  {'PASS' if ok else 'FAIL'}  {name:38s} max_rel_diff={rel:.3e}")
    return ok


results = [
    case("regression", "reg:squarederror"),
    case("regression + missing", "reg:squarederror", nan_frac=0.15),
    case("regression deep", "reg:squarederror", depth=9, rounds=60),
    case("binary:logistic", "binary:logistic"),
    case("binary:logistic + missing", "binary:logistic", nan_frac=0.2),
    case("reg:logistic", "reg:logistic"),
    case("binary:logitraw", "binary:logitraw"),
    case("multi:softprob k=4", "multi:softprob", num_class=4),
    case("multi:softprob k=3 + missing", "multi:softprob", num_class=3, nan_frac=0.15),
    case("count:poisson", "count:poisson"),
    case("stumps (depth 1)", "reg:squarederror", depth=1),
]

# --- input forms, refusals, and FALB interop --------------------------------
import json
import os
import tempfile

import falcata as flc

extra = []


def chk(name, cond, note=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name:38s} {note}")
    extra.append(cond)


rng = np.random.default_rng(7)
Xs = rng.standard_normal((1500, 6)).astype(np.float32)
ys = (Xs @ rng.standard_normal(6).astype(np.float32)).astype(float)
bst = xgb.train(
    {"objective": "reg:squarederror", "max_depth": 4, "seed": 1}, xgb.DMatrix(Xs, label=ys), num_boost_round=15
)
ref = bst.predict(xgb.DMatrix(Xs))
base = np.asarray(from_xgboost(bst).predict(Xs))

d = tempfile.mkdtemp()
jp = os.path.join(d, "m.json")
bst.save_model(jp)
chk("from .json path (no xgboost needed)", np.allclose(np.asarray(from_xgboost(jp).predict(Xs)), ref, atol=1e-6))
chk("from parsed dict", np.allclose(np.asarray(from_xgboost(json.load(open(jp))).predict(Xs)), ref, atol=1e-6))
sk = xgb.XGBRegressor(n_estimators=10, max_depth=3).fit(Xs, ys)
chk("from sklearn wrapper", np.asarray(from_xgboost(sk).predict(Xs)).shape == (len(Xs),))

# imported model round-trips through FALB and keeps its predictions
fb = os.path.join(d, "imported.falb")
from_xgboost(bst).save_model(fb)
chk(
    "imported -> FALB -> load",
    np.allclose(np.asarray(flc.Booster(model_file=fb).predict(Xs)), base, atol=0),
    f"({os.path.getsize(jp):,}B json vs {os.path.getsize(fb):,}B falb)",
)

# refusals must be explicit, never a silently wrong model
lin = xgb.train({"booster": "gblinear", "objective": "reg:squarederror"}, xgb.DMatrix(Xs, label=ys), num_boost_round=3)
try:
    from_xgboost(lin)
    chk("refuses gblinear", False)
except ValueError as e:
    chk("refuses gblinear", "gbtree" in str(e), str(e)[:44])
# booster=dart is deprecated in xgboost >= 3.4: dropout applies during
# training only and the model SAVES as gbtree, so it must convert with parity
drt = xgb.train(
    {"booster": "dart", "objective": "reg:squarederror", "max_depth": 3, "rate_drop": 0.1, "seed": 1},
    xgb.DMatrix(Xs, label=ys),
    num_boost_round=5,
)
dref = drt.predict(xgb.DMatrix(Xs))
chk("dart-trained (saves as gbtree)", np.allclose(np.asarray(from_xgboost(drt).predict(Xs)), dref, atol=1e-6))
# ...but a legacy dart MODEL (older xgboost) scales leaves at predict time, so
# a plain tree walk would predict something else -- must refuse, not approximate
legacy = json.loads(drt.save_raw("json").decode())
legacy["learner"]["gradient_booster"]["name"] = "dart"
try:
    from_xgboost(legacy)
    chk("refuses legacy dart model", False)
except ValueError as e:
    chk("refuses legacy dart model", "gbtree" in str(e) and "dart" in str(e), str(e)[:44])
Xc = Xs.copy()
dm = xgb.DMatrix(
    np.c_[Xc, rng.integers(0, 5, len(Xc))].astype(np.float32),
    label=ys,
    feature_types=["q"] * 6 + ["c"],
    enable_categorical=True,
)
cat = xgb.train({"objective": "reg:squarederror", "max_depth": 4}, dm, num_boost_round=8)
try:
    from_xgboost(cat)
    chk("refuses native categorical", False, "(model had no cat split)")
except ValueError as e:
    chk("refuses native categorical", "categorical" in str(e), str(e)[:44])
try:
    from_xgboost(
        xgb.train({"objective": "rank:pairwise"}, xgb.DMatrix(Xs, label=(ys > 0).astype(int)), num_boost_round=3)
    )
    chk("refuses unsupported objective", False)
except ValueError as e:
    chk("refuses unsupported objective", "not supported" in str(e), str(e)[:44])

ok = all(results) and all(extra)
print("XGBOOST IMPORT GATE " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
