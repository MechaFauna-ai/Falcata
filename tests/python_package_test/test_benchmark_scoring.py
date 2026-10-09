"""CPU regressions for benchmark score representations and accepted rounds."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import falcata as lgb

BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"


def _load_bench():
    # The harness uses script-local imports. Load it without changing the test
    # process's sys.path or replacing an unrelated module named common.
    old_common = sys.modules.get("common")
    common_spec = importlib.util.spec_from_file_location("common", BENCHMARKS_DIR / "common.py")
    common = importlib.util.module_from_spec(common_spec)
    common_spec.loader.exec_module(common)
    spec = importlib.util.spec_from_file_location("falcata_benchmark_scoring_bench", BENCHMARKS_DIR / "bench.py")
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["common"] = common
        spec.loader.exec_module(module)
    finally:
        if old_common is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = old_common
    return module


bench = _load_bench()


def test_native_binary_auc_preserves_order_lost_by_sigmoid_saturation():
    x = np.arange(4, dtype=np.float32).reshape(-1, 1)
    y = np.array([0, 1, 0, 1])
    raw = np.array([40.0, 41.0, 42.0, 43.0])
    params = {"device_type": "cpu", "objective": "binary", "metric": "auc", "num_threads": 1, "verbose": -1}
    train = lgb.Dataset(x, label=y, init_score=raw, params=params)
    valid = lgb.Dataset(x, label=y, init_score=raw, reference=train)
    model = lgb.Booster(params=params, train_set=train)
    model.add_valid(valid, "test")
    native_auc = model.eval_valid()[0][2]
    # GetPredictAt applies the objective conversion to the cached init scores.
    probability = model._Booster__inner_predict(data_idx=1)
    np.testing.assert_array_equal(probability, np.ones(4))
    raw_auc = roc_auc_score(y, raw)
    probability_metrics = bench.quality_metrics("binary", probability, y, {})
    assert native_auc == pytest.approx(raw_auc)
    assert native_auc != probability_metrics["auc"]
    assert not probability_metrics["sane"]
    assert bench.curve_metric_ok("binary", [[0, 0, native_auc]], probability_metrics, raw_auc)
    assert not bench.curve_metric_ok("binary", [[0, 0, 0.95]], probability_metrics, raw_auc)
    # Other engines still use the original probability-based integrity check.
    assert not bench.curve_metric_ok("binary", [[0, 0, native_auc]], probability_metrics)


@pytest.fixture
def fake_lightgbm(monkeypatch):
    """A stopped model whose accepted tree count differs from update calls."""
    state = SimpleNamespace(
        dataset_params=[], booster_params=[], model=None, limit=3, raw=np.array([-1.0, 1.0, -1.0, 1.0]), native_auc=None
    )

    class Dataset:
        def __init__(self, data, label=None, params=None, **kwargs):
            if params is not None:
                state.dataset_params.append(dict(params))

        def construct(self):
            return self

    class Booster:
        def __init__(self, params, train_set):
            state.booster_params.append(dict(params))
            state.model = self
            self.calls = 0
            self.accepted = 0

        def add_valid(self, valid, name):
            pass

        def update(self):
            self.calls += 1
            if self.accepted == state.limit:
                return True
            self.accepted += 1
            return False

        def current_iteration(self):
            return self.accepted

        def num_trees(self):
            return self.accepted

        def eval_valid(self):
            auc = float(roc_auc_score([0, 1, 0, 1], state.raw)) if state.native_auc is None else state.native_auc
            return [("test", "auc", auc, True)]

        def predict(self, data, **kwargs):
            assert kwargs["raw_score"] is True
            # Falcata must bypass FIL; upstream has no use_fil argument.
            if "use_fil" in kwargs:
                assert kwargs["use_fil"] is False
            return state.raw[np.asarray(data)[:, 0].astype(int)]

    def train(params, train_set, num_boost_round):
        model = Booster(params, train_set)
        model.accepted = min(num_boost_round, state.limit)
        return model

    module = SimpleNamespace(Dataset=Dataset, Booster=Booster, train=train, __version__="fake")
    monkeypatch.setitem(sys.modules, "falcata", module)
    monkeypatch.setitem(sys.modules, "lightgbm", module)
    monkeypatch.setattr(bench, "predict_in_chunks", lambda bst, x: (1 / (1 + np.exp(-state.raw)), "cpu"))

    class Monitor:
        gpu_peak_mb = 0
        rss_peak_mb = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(bench, "ResourceMonitor", Monitor)
    monkeypatch.setitem(bench.REGIMES, "test", {"rounds": 8, "eval_every": 2, "lr": 0.1, "leaves": 3, "depth": 2})
    return state


def run_fake_cell(library="falcata-noquant", kind="curve", overrides=()):
    args = SimpleNamespace(library=library, dataset="fraud", align_l2=False, overrides=list(overrides))
    x = np.arange(4, dtype=np.float32).reshape(-1, 1)
    y = np.array([0, 1, 0, 1])
    return bench.run_cell(args, "binary", "test", kind, (x, y, x, y, {}), {})


@pytest.mark.parametrize(
    ("limit", "expected_calls", "reason"), [(3, 4, "empty_tree"), (8, 8, "requested_rounds"), (2, 3, "empty_tree")]
)
def test_curve_stops_on_empty_update_and_uses_accepted_iteration_axis(fake_lightgbm, limit, expected_calls, reason):
    fake_lightgbm.limit = limit
    record = run_fake_cell()
    assert record["status"] == "ok"
    assert fake_lightgbm.model.calls == expected_calls
    assert record["requested_rounds"] == 8
    assert record["update_calls"] == expected_calls
    assert record["current_iteration"] == limit
    assert record["num_trees"] == limit
    assert record["stop_reason"] == reason
    assert record["curve"][-1][0] == limit
    assert len({point[0] for point in record["curve"]}) == len(record["curve"])
    assert record["trees_per_s"] == pytest.approx(limit / record["train_s"])
    assert record["curve_auc_raw"] == 1
    assert record["curve_metric_representation"] == "raw_margin"
    assert record["score_s"] >= 0


def test_saturated_probability_model_still_fails_sanity_with_a_correct_raw_curve(fake_lightgbm):
    fake_lightgbm.raw = np.array([40.0, 41.0, 42.0, 43.0])
    record = run_fake_cell()
    assert record["status"] == "insane"
    assert record["metrics"] == {"auc": 0.5, "sane": False}
    assert record["curve_auc_raw"] == 0.75
    assert bench.curve_metric_ok("binary", record["curve"], record["metrics"], record["curve_auc_raw"])


def test_wrong_raw_curve_is_rejected_even_with_sane_probabilities(fake_lightgbm):
    fake_lightgbm.native_auc = 0.5
    record = run_fake_cell()
    assert record["metrics"]["sane"]
    assert record["status"] == "bad_curve"
    assert record["curve_auc_raw"] == 1
    assert "raw_margin" in record["error"]


@pytest.mark.parametrize("library", ["falcata-noquant", "lightgbm"])
def test_explicit_leaf_cap_reaches_dataset_and_booster_without_changing_defaults(fake_lightgbm, library):
    record = run_fake_cell(library, overrides=["max_delta_step=1"])
    assert record["status"] == "ok"
    assert record["overrides"] == {"max_delta_step": 1}
    assert fake_lightgbm.dataset_params[0]["max_delta_step"] == 1
    assert fake_lightgbm.booster_params[0]["max_delta_step"] == 1
    run_fake_cell(library)
    assert "max_delta_step" not in fake_lightgbm.booster_params[-1]


def test_leaf_cap_override_rejects_other_engines_before_running_them(fake_lightgbm):
    record = run_fake_cell("xgboost", overrides=["max_delta_step=1"])
    assert record["status"] == "failed"
    assert "LightGBM-family" in record["error"]
    assert fake_lightgbm.model is None


def test_plain_run_reports_actual_trees_without_inventing_update_calls(fake_lightgbm):
    record = run_fake_cell(kind="timed1")
    assert record["status"] == "ok"
    assert record["num_trees"] == 3
    assert record["update_calls"] is None
    assert record["stop_reason"] == "fewer_iterations_than_requested"
    assert record["trees_per_s"] == pytest.approx(3 / record["train_s"])
    assert "curve_auc_raw" not in record
    assert "score_s" not in record
