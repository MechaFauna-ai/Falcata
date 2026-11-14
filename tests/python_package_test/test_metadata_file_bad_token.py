"""A bad token in a companion .weight or .init file raises FalcataError.

The weight and init-score files are parsed inside an OpenMP loop. The
"Unknown token" (and the init-score column count) errors were thrown inside it,
which called std::terminate and killed the process.
"""

import pytest

import falcata as flc


def _write_train(path, num_rows=30):
    path.write_text("".join(f"{i % 2},{i},{(i * 7) % 5}\n" for i in range(num_rows)))


@pytest.mark.parametrize("bad", ["none", "?", "abc"])
def test_bad_weight_token_raises(tmp_path, bad):
    data = tmp_path / "train.csv"
    _write_train(data)
    weights = ["1.0"] * 30
    weights[7] = bad
    (tmp_path / "train.csv.weight").write_text("\n".join(weights) + "\n")
    with pytest.raises(flc.basic.FalcataError, match="Unknown token"):
        flc.Dataset(str(data), params={"verbose": -1}).construct()


def test_bad_init_score_token_raises(tmp_path):
    data = tmp_path / "train.csv"
    _write_train(data)
    scores = ["0.5"] * 30
    scores[3] = "nan?"
    (tmp_path / "train.csv.init").write_text("\n".join(scores) + "\n")
    with pytest.raises(flc.basic.FalcataError, match="Unknown token"):
        flc.Dataset(str(data), params={"verbose": -1}).construct()


def test_ragged_multiclass_init_score_raises(tmp_path):
    data = tmp_path / "train.csv"
    _write_train(data)
    scores = ["0.1\t0.2\t0.3"] * 30
    scores[11] = "0.1\t0.2"
    (tmp_path / "train.csv.init").write_text("\n".join(scores) + "\n")
    with pytest.raises(flc.basic.FalcataError, match="Redundant or insufficient columns"):
        flc.Dataset(str(data), params={"verbose": -1}).construct()
