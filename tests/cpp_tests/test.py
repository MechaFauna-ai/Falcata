# coding: utf-8
from pathlib import Path

import numpy as np

pred_files = sorted(Path(__file__).absolute().parent.glob("*.pred"))
assert len(pred_files) == 2, f"expected exactly two .pred files to compare, found {[p.name for p in pred_files]}"
preds = [np.loadtxt(str(name)) for name in pred_files]
np.testing.assert_allclose(preds[0], preds[1])
