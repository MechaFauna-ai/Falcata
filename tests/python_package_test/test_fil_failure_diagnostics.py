"""A FIL failure (use_fil=True) is raised with its cause, never replaced by another predictor."""

import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest

from falcata import basic


def _failing_runner(stage, monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(asarray=np.asarray))
    model = types.SimpleNamespace(is_classifier=False, predict=Mock(side_effect=RuntimeError("simulated OOM")))
    runner = basic._fil_run_host_slabs if stage == "host prediction" else basic._fil_run_device
    return lambda: runner(model, np.zeros((2, 3), dtype=np.float32))


def _failing_conversion():
    target = Mock()
    target._leaf_value_dim.return_value = 1
    target.model_to_string.return_value = "model text"
    frontend = Mock()
    frontend.load_lightgbm_model.side_effect = RuntimeError("simulated OOM")
    return lambda: basic.Booster._build_fil_model(
        target,
        treelite_frontend=frontend,
        nvforest=Mock(),
        start_iteration=0,
        num_iteration=-1,
        raw_score=False,
        precision="single",
    )


@pytest.mark.parametrize("stage", ["prediction", "host prediction", "conversion"])
def test_fil_failure_raises_with_cause(stage, monkeypatch):
    warning = Mock()
    monkeypatch.setattr(basic, "_log_warning", warning)
    call = _failing_conversion() if stage == "conversion" else _failing_runner(stage, monkeypatch)
    with pytest.raises(basic.FalcataError, match=f"FIL {stage} failed \\(RuntimeError: simulated OOM\\)"):
        call()
    warning.assert_not_called()
