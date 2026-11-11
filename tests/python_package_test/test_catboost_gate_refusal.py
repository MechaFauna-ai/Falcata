"""The categorical refusal gate must reject unrelated failures."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("stage", ["fit", "convert"])
def test_catboost_gate_rejects_unrelated_valueerror(stage):
    path = Path(__file__).resolve().parents[1] / "gates" / "import_catboost.py"
    tree = ast.parse(path.read_text())
    # Execute only the real refusal block, isolating it from unrelated parity checks.
    blocks = [
        node
        for node in tree.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.Constant) and child.value == "refuses categorical features"
            for child in ast.walk(node)
        )
    ]
    assert len(blocks) == 1

    def fit(*args):
        if stage == "fit":
            raise ValueError("unrelated training failure")
        return object()

    def convert(*args):
        raise ValueError("unrelated conversion failure")

    checks = []
    namespace = {
        "np": np,
        "Xs": np.zeros((2, 5)),
        "ys": np.zeros(2),
        "rng": np.random.default_rng(0),
        "cb": SimpleNamespace(CatBoostRegressor=lambda **kwargs: SimpleNamespace(fit=fit)),
        "from_catboost": convert,
        "chk": lambda *args: checks.append(args),
    }
    exec(compile(ast.Module(body=blocks, type_ignores=[]), str(path), "exec"), namespace)
    assert len(checks) == 1
    assert checks[0][1] is False
