# coding: utf-8
"""A zero-column Arrow table as init_score is rejected instead of read.

InitScoreView concatenated one view per struct field, and View(views)
read views[0] without checking for an empty list, so a table with no
columns dereferenced past an empty vector.
"""

import numpy as np
import pytest

import falcata as flc
from falcata.basic import FalcataError

pa = pytest.importorskip("pyarrow")


def test_empty_arrow_init_score_raises():
    rng = np.random.default_rng(0)
    ds = flc.Dataset(rng.normal(size=(50, 2)), label=rng.normal(size=50)).construct()
    with pytest.raises(FalcataError, match="empty list of Arrow views"):
        ds.set_init_score(pa.table({}))
