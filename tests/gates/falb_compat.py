"""Frozen-fixture compatibility gate for Falcata model files.

The fixtures under ``falb_fixtures/`` are model files written by an earlier
Falcata (the set was first frozen at v1.0.0) together with an input matrix and
the predictions that build produced; ``meta.json`` records the version the
writing build reported. Every future Falcata must keep loading BOTH formats of
every fixture and reproduce those predictions -- this is the promise that a
saved model outlives the library version that wrote it.

  python tests/gates/falb_compat.py             # check (the gate)
  python tests/gates/falb_compat.py --freeze    # (re)write fixtures: only for
                                                # a NEW fixture generation with
                                                # an understood format change
  python tests/gates/falb_compat.py --freeze NAME [NAME...]
                                                # write only these, leaving the
                                                # other fixtures' bytes frozen
                                                # -- how a fixture is ADDED
"""

import json
import sys
from pathlib import Path

import numpy as np

import falcata as flc

FIXTURE_DIR = Path(__file__).resolve().parent / "falb_fixtures"
RTOL = 1e-10

#: name -> (params, data shape knobs). CPU, single thread, fixed seeds: the
#: freeze is deterministic, but determinism is not load-bearing -- fixtures
#: are frozen FILES, not recipes.
SHAPES = {
    "regression": ({"objective": "regression", "num_leaves": 31}, {}),
    "regression-missing": ({"objective": "regression", "num_leaves": 31}, {"nan_frac": 0.3}),
    "binary": ({"objective": "binary", "num_leaves": 63}, {}),
    "multiclass": ({"objective": "multiclass", "num_class": 4, "num_leaves": 31}, {}),
    "categorical": ({"objective": "regression", "num_leaves": 31}, {"cat_cols": 2}),
    "deep": ({"objective": "regression", "num_leaves": 255, "min_data_in_leaf": 2}, {}),
    # a class that never occurs has nothing learnable left in it, so its
    # per-class trees come back CONSTANT (1 leaf) every round while the other
    # classes keep growing: the mix that the binary writer used to mis-size
    "multiclass-constant": ({"objective": "multiclass", "num_class": 4, "num_leaves": 31}, {}),
}


def make_data(name, nan_frac=0.0, cat_cols=0, n=3000, m=8):
    rng = np.random.default_rng(abs(hash(name)) % 100_000)
    x = rng.standard_normal((n, m))
    for j in range(cat_cols):
        x[:, j] = rng.integers(0, 40, size=n)
    if nan_frac:
        x[rng.random((n, m)) < nan_frac] = np.nan
    y = np.nan_to_num(x) @ rng.standard_normal(m) + rng.standard_normal(n) * 0.1
    if "binary" in name:
        y = (y > np.median(y)).astype(float)
    if "multiclass" in name:
        y = (np.nan_to_num(x) @ rng.standard_normal((m, 4))).argmax(1).astype(float)
    if "constant" in name:
        # fold the last class away: it never occurs, so its trees stay constant
        y = np.minimum(y, 2.0)
    return x, y, list(range(cat_cols))


def leaf_counts(model_text):
    return [int(t.split("num_leaves=")[1].split("\n")[0]) for t in model_text.split("\nTree=")[1:]]


def freeze(only=None):
    for name, (params, knobs) in SHAPES.items():
        if only and name not in only:
            continue
        d = FIXTURE_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        x, y, cats = make_data(name, **knobs)
        p = dict(params, verbose=-1, seed=42, num_threads=1, deterministic=True)
        ds = flc.Dataset(x, label=y, params=p, categorical_feature=cats or "auto")
        bst = flc.train(p, ds, num_boost_round=20)
        np.save(d / "X.npy", x.astype(np.float64))
        np.save(d / "preds.npy", bst.predict(x))
        bst.save_model(str(d / "model.falb"), format="falb")
        bst.save_model(str(d / "model.txt"), format="text")
        (d / "meta.json").write_text(
            json.dumps({"falcata_version": flc.__version__, "params": params}, indent=1) + "\n"
        )
        leaves = leaf_counts((d / "model.txt").read_text())
        if "constant" in name and not (min(leaves) == 1 < max(leaves)):
            raise SystemExit(
                f"{name}: this fixture exists to carry constant trees next to grown ones, "
                f"but the recipe produced leaf counts {sorted(set(leaves))}"
            )
        print(f"froze {name} (writer {flc.__version__}, {len(leaves)} trees, {min(leaves)}-{max(leaves)} leaves)")


def check():
    fixtures = sorted(p.parent for p in FIXTURE_DIR.glob("*/meta.json"))
    if not fixtures:
        print("FAIL: no fixtures found -- run --freeze once from a release build")
        return 1
    bad = 0
    for d in fixtures:
        meta = json.loads((d / "meta.json").read_text())
        x = np.load(d / "X.npy")
        want = np.load(d / "preds.npy")
        for fmt in ("model.falb", "model.txt"):
            try:
                got = flc.Booster(model_file=str(d / fmt)).predict(x)
                ok = got.shape == want.shape and np.allclose(got, want, rtol=RTOL, atol=0)
            except Exception as e:  # noqa: BLE001 -- a load failure IS the finding
                ok, got = False, f"raised {e!r}"
            tag = f"{d.name}/{fmt} (writer {meta['falcata_version']})"
            print(f"  {'PASS' if ok else 'FAIL'}  {tag}")
            bad += 0 if ok else 1
        if "constant" in d.name:
            # a fixture regenerated into an all-grown model would keep passing
            # while covering nothing: say so instead
            leaves = leaf_counts((d / "model.txt").read_text())
            ok = min(leaves) == 1 < max(leaves)
            print(f"  {'PASS' if ok else 'FAIL'}  {d.name} still carries constant trees next to grown ones")
            bad += 0 if ok else 1
    print(f"{'MODEL COMPAT GATE PASS' if bad == 0 else f'MODEL COMPAT GATE: {bad} FAILURE(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        freeze([a for a in sys.argv[sys.argv.index("--freeze") + 1 :] if not a.startswith("-")])
        sys.exit(check())
    sys.exit(check())
