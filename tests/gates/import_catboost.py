"""CatBoost import gate: converted models must predict what CatBoost predicts.

CatBoost's symmetric (oblivious) trees are unrolled into ordinary binary
trees. Parity lands at machine precision rather than the ~1e-7 of the XGBoost
importer, because CatBoost's JSON stores float64 borders and leaf values --
there is no float32 round-trip to recover.

Also checks the accepted input forms, FALB interop, and that the cases which
cannot be represented (multiclass, categorical/CTR features, non-oblivious
grow policies) REFUSE rather than silently converting to something else.
"""
import sys, numpy as np, catboost as cb
from falcata.importers import from_catboost

def case(name, loss, depth=4, iters=25, n=3000, m=6, nan_frac=0.0, cls=False):
    rng = np.random.default_rng(abs(hash(name)) % 10_000)
    X = rng.standard_normal((n, m))
    if nan_frac:
        X[rng.random((n, m)) < nan_frac] = np.nan
    lin = np.nan_to_num(X) @ rng.standard_normal(m)
    y = (lin > 0).astype(int) if cls else lin + 0.2 * rng.standard_normal(n)
    Model = cb.CatBoostClassifier if cls else cb.CatBoostRegressor
    mdl = Model(iterations=iters, depth=depth, learning_rate=0.3, loss_function=loss,
                verbose=0, allow_writing_files=False, random_seed=1)
    mdl.fit(X, y)
    ref = (mdl.predict(X, prediction_type="RawFormulaVal") if cls
           else mdl.predict(X))
    got = np.asarray(from_catboost(mdl).predict(X, raw_score=cls))
    scale = max(1e-9, float(np.abs(ref).max()))
    rel = float(np.abs(got - ref).max() / scale)
    ok = rel < 1e-6
    print(f"  {'PASS' if ok else 'FAIL'}  {name:36s} max_rel_diff={rel:.3e}")
    return ok

res = [
    case("RMSE depth4", "RMSE"),
    case("RMSE depth6", "RMSE", depth=6, iters=40),
    case("RMSE depth1", "RMSE", depth=1),
    case("RMSE + missing", "RMSE", nan_frac=0.15),
    case("MAE", "MAE"),
    case("Quantile", "Quantile:alpha=0.7"),
    case("Logloss (binary)", "Logloss", cls=True),
    case("Logloss + missing", "Logloss", nan_frac=0.2, cls=True),
    case("Poisson", "Poisson", iters=15),
]

# --- input forms, refusals, FALB interop ------------------------------------
import json, os, tempfile
import falcata as flc

extra = []
def chk(name, cond, note=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name:36s} {note}")
    extra.append(cond)

rng = np.random.default_rng(11)
Xs = rng.standard_normal((2000, 5))
ys = Xs @ rng.standard_normal(5)
mdl = cb.CatBoostRegressor(iterations=20, depth=4, learning_rate=0.3, verbose=0,
                           allow_writing_files=False, random_seed=1).fit(Xs, ys)
ref = mdl.predict(Xs)
base = np.asarray(from_catboost(mdl).predict(Xs))
d = tempfile.mkdtemp(); jp = os.path.join(d, "m.json"); mdl.save_model(jp, format="json")
chk("from .json path (no catboost needed)",
    np.allclose(np.asarray(from_catboost(jp).predict(Xs)), ref, atol=1e-9))
chk("from parsed dict",
    np.allclose(np.asarray(from_catboost(json.load(open(jp))).predict(Xs)), ref, atol=1e-9))
fb = os.path.join(d, "cb.falb"); from_catboost(mdl).save_model(fb)
chk("imported -> FALB -> load",
    np.allclose(np.asarray(flc.Booster(model_file=fb).predict(Xs)), base, atol=0),
    f"({os.path.getsize(jp):,}B json vs {os.path.getsize(fb):,}B falb)")

# refusals: explicit errors, never a silently different model
try:
    mc = cb.CatBoostClassifier(iterations=8, depth=3, loss_function="MultiClass",
                               verbose=0, allow_writing_files=False).fit(
        Xs, rng.integers(0, 3, len(Xs)))
    from_catboost(mc); chk("refuses multiclass", False)
except ValueError as e:
    chk("refuses multiclass", "multiclass" in str(e).lower(), str(e)[:42])
try:
    Xc = np.c_[Xs, rng.integers(0, 4, len(Xs)).astype(str)]
    cm = cb.CatBoostRegressor(iterations=8, depth=3, verbose=0, allow_writing_files=False,
                              cat_features=[5]).fit(Xc, ys)
    from_catboost(cm); chk("refuses categorical features", False)
except ValueError as e:
    chk("refuses categorical features", True, str(e)[:42])
try:
    lg = cb.CatBoostRegressor(iterations=8, depth=4, grow_policy="Lossguide",
                              verbose=0, allow_writing_files=False).fit(Xs, ys)
    from_catboost(lg); chk("refuses non-oblivious trees", False)
except ValueError as e:
    chk("refuses non-oblivious trees", "oblivious" in str(e), str(e)[:42])

ok = all(res) and all(extra)
print("CATBOOST IMPORT GATE " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
