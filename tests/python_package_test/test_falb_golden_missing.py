"""Golden writer entries cannot silently disappear with fixture directories."""

import importlib.util
import json
from pathlib import Path


def test_golden_gate_detects_deleted_fixture(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "gates" / "falb_golden.py"
    spec = importlib.util.spec_from_file_location("falb_golden", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"deleted": {"raw": "hash"}}))
    monkeypatch.setattr(module, "GOLDEN_PATH", golden)
    monkeypatch.setattr(module, "FIXTURE_DIR", tmp_path / "fixtures")
    assert module.check() == 1
