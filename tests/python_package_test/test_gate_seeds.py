"""Gate data seeds must be stable across interpreter processes."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("gate", ["falb_compat", "falb_roundtrip", "import_catboost", "import_xgboost"])
def test_gate_seed_ignores_pythonhashseed(gate):
    source = (Path(__file__).resolve().parents[1] / "gates" / f"{gate}.py").read_text()
    tree = ast.parse(source)
    # Extract the real seed expression without importing gates that train at import time.
    seeds = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mod)
        and "name" in ast.unparse(node)
        and ("hash(" in ast.unparse(node) or "crc32(" in ast.unparse(node))
    ]
    assert len(seeds) == 1
    script = f'from zlib import crc32; name = "binary"; print({ast.unparse(seeds[0])})'
    outputs = [
        subprocess.check_output([sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": seed})
        for seed in ["1", "2", "3"]
    ]
    assert len(set(outputs)) == 1
