"""A text model with Windows (CRLF) line endings must load and predict the same.

tree_sizes records each tree block's size in bytes as written with '\\n'.
After a CRLF conversion (a Windows editor, git autocrlf) every line has an
extra '\\r', every tree offset was off, and loading failed with "Model format
error, expect a tree here". (Stock LightGBM 4.6 aborts the process on such a
file.)
"""

import numpy as np

import falcata as flc


def test_crlf_model_text_loads(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 4))
    x[:, 3] = rng.integers(0, 5, size=300)
    ds = flc.Dataset(x, label=x[:, 0] + (x[:, 3] == 2), categorical_feature=[3])
    booster = flc.train({"verbose": -1, "num_threads": 1, "num_leaves": 7}, ds, num_boost_round=5)
    crlf = booster.model_to_string().replace("\n", "\r\n")
    np.testing.assert_array_equal(flc.Booster(model_str=crlf).predict(x), booster.predict(x))
    path = tmp_path / "model_crlf.txt"
    path.write_bytes(crlf.encode("utf-8"))
    loaded = flc.Booster(model_file=path)
    np.testing.assert_array_equal(loaded.predict(x), booster.predict(x))
    assert loaded.num_trees() == booster.num_trees()
