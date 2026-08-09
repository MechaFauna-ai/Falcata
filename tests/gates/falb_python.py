"""FALB M2 gate: Python plumbing.

Covers the surface a user actually touches: save_model format selection,
loading by magic (not extension), the bytes API, the pickle flip and its
escape hatch, and the CLI converter -- including that converting is LOSSLESS
by default, so an artifact does not quietly lose the per-node data that
pred_contrib and gain importance need.
"""

import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

import falcata as flc

rng = np.random.default_rng(0)
X = rng.standard_normal((5000, 20))
y = X @ rng.standard_normal(20) + 0.3 * rng.standard_normal(5000)
p = {"objective": "regression", "num_leaves": 31, "verbose": -1, "num_threads": 8}
bst = flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=40)
ref = bst.predict(X)
fails = []


def chk(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        fails.append(name)


d = tempfile.mkdtemp()
# 1. save_model format=auto by extension
txt, falb = os.path.join(d, "m.txt"), os.path.join(d, "m.falb")
bst.save_model(txt)
bst.save_model(falb)
st, sf = os.path.getsize(txt), os.path.getsize(falb)
chk(
    "save_model auto-detects .falb",
    open(falb, "rb").read(4) == b"FALB",
    f"({st:,}B txt vs {sf:,}B falb = {st / sf:.1f}x)",
)
chk("save_model .txt still text", open(txt, "rb").read(4) != b"FALB")

# 2. load by magic sniffing, incl. a misleading extension
chk("Booster(model_file=.falb)", np.array_equal(flc.Booster(model_file=falb).predict(X), ref))
lying = os.path.join(d, "lying.txt")
open(lying, "wb").write(open(falb, "rb").read())
chk("magic beats extension", np.array_equal(flc.Booster(model_file=lying).predict(X), ref))
chk("Booster(model_file=.txt)", np.array_equal(flc.Booster(model_file=txt).predict(X), ref))

# 3. bytes round-trip
blob = bst.model_to_binary()
chk("model_to_binary/Booster(model_bin=)", np.array_equal(flc.Booster(model_bin=blob).predict(X), ref))

# 4. pickle defaults to FALB and shrinks
pk = pickle.dumps(bst)
chk("pickle payload is FALB", b"FALB" in pk)
chk("unpickle predicts identically", np.array_equal(pickle.loads(pk).predict(X), ref))
flc.set_pickle_format("text")
pk_txt = pickle.dumps(bst)
flc.set_pickle_format("falb")
chk(
    "pickle FALB smaller than text",
    len(pk) < len(pk_txt),
    f"({len(pk):,} vs {len(pk_txt):,} = {len(pk_txt) / len(pk):.1f}x)",
)
chk("text pickles still load", np.array_equal(pickle.loads(pk_txt).predict(X), ref))
chk("get_pickle_format", flc.get_pickle_format() == "falb")

# 5. CLI convert both ways
out_falb = os.path.join(d, "cli.falb")
out_txt = os.path.join(d, "cli.txt")
r1 = subprocess.run(
    [sys.executable, "-m", "falcata", "convert", txt, out_falb], check=False, capture_output=True, text=True
)
r2 = subprocess.run(
    [sys.executable, "-m", "falcata", "convert", out_falb, out_txt], check=False, capture_output=True, text=True
)
chk("CLI txt->falb", r1.returncode == 0 and open(out_falb, "rb").read(4) == b"FALB", r1.stdout.strip()[:60])
chk("CLI falb->txt", r2.returncode == 0 and np.array_equal(flc.Booster(model_file=out_txt).predict(X), ref))
chk("CLI round-trip text is byte-identical", open(out_txt, "rb").read() == open(txt, "rb").read())

# 6. no LightGBM tokens in the binary payload
chk("no LightGBM token in FALB bytes", b"LightGBM" not in blob and b"lightgbm" not in blob)

# 7. pandas_categorical still travels with the file
chk("pandas_categorical preserved", flc.Booster(model_file=falb).pandas_categorical == bst.pandas_categorical)

print("M2 PASS" if not fails else f"M2 FAIL: {fails}")
sys.exit(1 if fails else 0)
