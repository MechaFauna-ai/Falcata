"""Unexpected FIL failures retain their cause when selecting CPU fallback."""

import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest

from falcata import basic


@pytest.mark.parametrize("stage", ["prediction", "host prediction", "conversion"])
def test_fil_fallback_reports_cause(stage, monkeypatch):
    warning = Mock()
    monkeypatch.setattr(basic, "_log_warning", warning)
    if "prediction" in stage:
        monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(asarray=np.asarray))
        model = types.SimpleNamespace(is_classifier=False, predict=Mock(side_effect=RuntimeError("simulated OOM")))
        runner = basic._fil_run_host_slabs if stage == "host prediction" else basic._fil_run_device
        result = runner(model, np.zeros((2, 3), dtype=np.float32))
    else:
        target = Mock()
        target._leaf_value_dim.return_value = 1
        target.model_to_string.return_value = "model text"
        frontend = Mock()
        frontend.load_lightgbm_model.side_effect = RuntimeError("simulated OOM")
        result = basic.Booster._build_fil_model(
            target,
            treelite_frontend=frontend,
            nvforest=Mock(),
            start_iteration=0,
            num_iteration=-1,
            raw_score=False,
            precision="single",
        )
    assert result is None
    warning.assert_called_once()
    message = warning.call_args.args[0]
    assert stage in message
    assert "RuntimeError" in message
    assert "simulated OOM" in message
