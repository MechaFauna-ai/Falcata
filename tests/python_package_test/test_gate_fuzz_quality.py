"""Bounded CPU fixtures for fuzz semantics; all child processes and training are mocked."""

import ast
import copy
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score


@pytest.fixture
def fuzz(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "gates" / "fuzz.py"
    spec = importlib.util.spec_from_file_location("fuzz_quality_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden(*args, **kwargs):
        pytest.fail("unexpected child process: every worker must be mocked")

    monkeypatch.setattr(module.subprocess, "run", forbidden)
    return module


@pytest.fixture
def spec():
    return {
        "n": 200,
        "m": 5,
        "dtype": "float64",
        "int_lo": -6,
        "int_hi": 7,
        "num_cat_features": 0,
        "cat_cards": [],
        "max_cat_threshold": 32,
        "nan_frac": 0.0,
        "sparsity": 0.0,
        "objective": "binary",
        "num_class": None,
        "quant_mode": "none",
        "quant_bins": 0,
        "num_leaves": 7,
        "max_depth": 10,
        "min_data_in_leaf": 20,
        "feature_fraction": 1.0,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "max_bin": 511,
        "learning_rate": 0.1,
        "cuda_plan": "auto",
        "rounds": 3,
        "data_seed": 910229978,
    }


def pair_auc(labels, predictions):
    """Independent definition, deliberately quadratic on tiny fixtures only."""
    positive = [p for y, p in zip(labels, predictions, strict=True) if y == 1]
    negative = [p for y, p in zip(labels, predictions, strict=True) if y == 0]
    return sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


@pytest.fixture
def auc(fuzz):
    # Extract the actual embedded function; no falcata import or worker execution.
    function = next(
        node
        for node in ast.parse(fuzz._WORKER_SOURCE).body
        if isinstance(node, ast.FunctionDef) and node.name == "binary_roc_auc"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<worker auc>", "exec"), namespace)
    return namespace["binary_roc_auc"]


@pytest.mark.parametrize(
    ("labels", "predictions"),
    [
        ([0, 1, 0, 1], [0, 0, 1, 1]),  # old unique-rank formula is order-dependent
        ([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]),
        ([0, 0, 1, 1], [0, 1, 2, 3]),
        ([0, 0, 1, 1], [3, 2, 1, 0]),
        ([0, 1, 0, 0, 1], [0.2, 0.2, 0.5, 0.8, 0.8]),
    ],
)
def test_binary_auc_matches_pair_count_and_sklearn_under_permutations(auc, labels, predictions):
    expected = pair_auc(labels, predictions)
    for permutation in itertools.permutations(range(len(labels))):
        y = np.asarray(labels)[list(permutation)]
        p = np.asarray(predictions)[list(permutation)]
        assert auc(y, p) == pytest.approx(expected, rel=0, abs=1e-15)
        assert auc(y, p) == pytest.approx(roc_auc_score(y, p), rel=0, abs=1e-15)


@pytest.mark.parametrize(
    ("labels", "predictions"),
    [
        ([], []),
        ([0, 0], [0, 1]),
        ([1, 1], [0, 1]),
        ([0, 2], [0, 1]),
        ([-1, 1], [0, 1]),
        ([0, np.nan], [0, 1]),
        ([0, 1], [0, np.inf]),
        ([0, 1], [0, np.nan]),
        ([0, 1], [0]),
        ([[0, 1]], [[0, 1]]),
    ],
)
def test_binary_auc_rejects_undefined_or_invalid_inputs(auc, labels, predictions):
    with pytest.raises(ValueError, match="binary ROC AUC"):
        auc(labels, predictions)


SEEDS = {
    "seed": 42,
    "bagging_seed": 400,
    "data_random_seed": 175,
    "feature_fraction_seed": 30056,
    "drop_seed": 17869,
    "objective_seed": 16083,
    "extra_seed": 12879,
}


class WorkerScript:
    """Exercise real run()/JSON transport and check every spec, device and timeout."""

    def __init__(
        self,
        fuzz,
        monkeypatch,
        spec,
        pairs=(True, True),
        higher=True,
        effective_bag_seed=400,
        initial=None,
        reference=1.0,
        faults=None,
    ):
        self.fuzz = fuzz
        self.original = copy.deepcopy(spec)
        self.calls = []
        self.expected = [(copy.deepcopy(spec), "cuda", 300)] * 2 + [(copy.deepcopy(spec), "cpu", 900)]
        self.seeds = dict(SEEDS, bagging_seed=effective_bag_seed)
        self.higher = higher
        self.reference = reference
        self.loss = reference * (0.6 if higher else 1.4)
        self.initial = self.loss if initial is None else initial
        self.pairs = pairs
        self.faults = faults or {}
        if pairs is not None:
            assert len(pairs) == 2
            for mask in (0x5EED, 0xBA66):
                changed = dict(spec, bagging_seed=effective_bag_seed ^ mask)
                self.expected.extend([(changed, "cuda", 300), (changed, "cpu", 900)])
        monkeypatch.setattr(fuzz.subprocess, "run", self.transport)

    def transport(self, argv, **kwargs):
        index = len(self.calls)
        actual = (json.loads(argv[3]), argv[4], kwargs["timeout"])
        assert index < len(self.expected), f"unexpected extra worker {actual}"
        assert actual == self.expected[index]
        assert argv[:3] == [sys.executable, "-c", self.fuzz._WORKER_SOURCE]
        assert kwargs["env"]["FALCATA_FIL"] == "0"
        assert kwargs["capture_output"]
        assert kwargs["text"]
        assert not kwargs["check"]
        self.calls.append(copy.deepcopy(actual))
        changed, device, timeout = actual
        metric = self.reference if device == "cpu" else self.initial
        if index >= 3 and device == "cuda":
            outcome = self.pairs[(index - 3) // 2]
            metric = (self.reference if outcome else self.loss) if isinstance(outcome, bool) else outcome
        result = {
            "md5": "same-cuda-model",
            "metric": metric,
            "higher_better": self.higher,
            "trees": changed["rounds"],
            "stopped_early": False,
            "data_seed": changed["data_seed"],
            "train_rows": int(changed["n"] * 0.8),
            "effective_seeds": dict(self.seeds, bagging_seed=changed.get("bagging_seed", self.seeds["bagging_seed"])),
        }
        fault = self.faults.get(index)
        if fault == "timeout":
            raise subprocess.TimeoutExpired(argv, timeout)
        if fault == "error":
            return subprocess.CompletedProcess(argv, 1, "", "mock child error")
        if isinstance(fault, dict):
            result.update(fault)
        return subprocess.CompletedProcess(argv, 0, "OUT " + json.dumps(result) + "\n", "")

    def check(self, spec):
        result = self.fuzz.check_spec(spec)
        assert self.calls == self.expected  # no skipped confirmations or extra attempts
        assert spec == self.original  # original caller's spec must remain unchanged
        for changed, _, _ in self.calls[3:]:
            assert {k: v for k, v in changed.items() if k != "bagging_seed"} == {
                k: v for k, v in spec.items() if k != "bagging_seed"
            }
            assert changed["bagging_seed"] != self.seeds["bagging_seed"]
        return result


@pytest.mark.parametrize("higher", [True, False])
@pytest.mark.parametrize("pairs", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("effective_bag_seed", [400, 9231])
def test_strict_confirmation_truth_table_and_only_bag_seed_changes(
    fuzz, monkeypatch, spec, higher, pairs, effective_bag_seed
):
    spec["objective"] = "binary" if higher else "regression"
    script = WorkerScript(fuzz, monkeypatch, spec, pairs, higher, effective_bag_seed)
    fails, known = script.check(spec)
    assert bool(fails) == (not all(pairs))
    assert any("2/2 fixed-data pairs within tolerance" in note for note in known) == all(pairs)
    assert not any("single-run variance" in note for note in known)
    if all(pairs):
        assert "cause unresolved" in known[0]
    else:
        assert any("much worse than cpu" in failure for failure in fails)
    assert len(script.calls) == 7
    assert [s["bagging_seed"] for s, _, _ in script.calls[3::2]] == [
        effective_bag_seed ^ 0x5EED,
        effective_bag_seed ^ 0xBA66,
    ]


def test_explicit_original_bag_seed_is_the_confirmation_anchor(fuzz, monkeypatch, spec):
    spec["bagging_seed"] = 733
    script = WorkerScript(fuzz, monkeypatch, spec, effective_bag_seed=733)
    assert script.check(spec)[0] == []
    assert script.calls[3][0]["bagging_seed"] == 733 ^ 0x5EED


@pytest.mark.parametrize("higher", [True, False])
@pytest.mark.parametrize("beyond", [False, True])
@pytest.mark.parametrize("boundary_in_confirmation", [False, True])
@pytest.mark.parametrize("reference", [0.2, 0.8])
def test_same_relative_tolerance_and_absolute_epsilon_at_boundary(
    fuzz, monkeypatch, spec, higher, beyond, boundary_in_confirmation, reference
):
    assert fuzz.CPU_METRIC_TOLERANCE == 0.10
    spec["objective"] = "binary" if higher else "regression"
    direction = -1 if higher else 1
    boundary = reference + direction * (abs(reference) * 0.10 + 1e-9)
    metric = boundary + direction * (1e-10 if beyond else 0)
    if boundary_in_confirmation:
        script = WorkerScript(fuzz, monkeypatch, spec, pairs=(metric, True), higher=higher, reference=reference)
    else:
        spec["bagging_freq"] = 0
        script = WorkerScript(fuzz, monkeypatch, spec, pairs=None, higher=higher, initial=metric, reference=reference)
    fails, _ = script.check(spec)
    assert bool(fails) == beyond


@pytest.mark.parametrize("inactive", ["fraction_one", "frequency_zero", "frequency_missing"])
def test_inactive_bagging_never_confirms_or_resamples(fuzz, monkeypatch, spec, inactive):
    if inactive == "fraction_one":
        spec["bagging_fraction"] = 1.0
    elif inactive == "frequency_zero":
        spec["bagging_freq"] = 0
    else:
        del spec["bagging_freq"]
    script = WorkerScript(fuzz, monkeypatch, spec, pairs=None)
    fails, known = script.check(spec)
    assert len(fails) == 1
    assert "much worse than cpu" in fails[0]
    assert known == []
    assert len(script.calls) == 3


def test_initial_quality_within_tolerance_does_not_confirm(fuzz, monkeypatch, spec):
    script = WorkerScript(fuzz, monkeypatch, spec, pairs=None, initial=1.0)
    assert script.check(spec) == ([], [])


@pytest.mark.parametrize(
    ("index", "fault"), [(3, "error"), (4, "error"), (5, "error"), (6, "error"), (4, "timeout"), (6, "timeout")]
)
@pytest.mark.parametrize("lowbin", [False, True])
def test_confirmation_child_failure_is_never_exonerated(fuzz, monkeypatch, spec, index, fault, lowbin):
    if lowbin:
        spec.update(quant_mode="stochastic", quant_bins=4, max_bin=63)
    script = WorkerScript(fuzz, monkeypatch, spec, faults={index: fault})
    fails, known = script.check(spec)
    expected = "timed out after" if fault == "timeout" else "mock child error"
    assert any("bagging confirmation bagging_seed=" in failure and expected in failure for failure in fails)
    assert not any("2/2 fixed-data pairs" in note for note in known)
    assert len(script.calls) == 7


@pytest.mark.parametrize("fault", [{"md5": "different-cuda-model"}, "error"])
def test_original_cuda_rerun_failure_survives_two_clean_confirmations(fuzz, monkeypatch, spec, fault):
    script = WorkerScript(fuzz, monkeypatch, spec, faults={1: fault})
    fails, known = script.check(spec)
    assert len(fails) == 1
    assert "cuda nondeterminism" in fails[0] or "cuda rerun failed" in fails[0]
    assert any("2/2 fixed-data pairs" in note for note in known)


@pytest.mark.parametrize("index", [0, 1, 3, 5])
@pytest.mark.parametrize("lowbin", [False, True])
def test_cuda_timeout_stops_immediately_with_original_repro_and_pending_confirmations(
    fuzz, monkeypatch, spec, index, lowbin
):
    if lowbin:
        spec.update(quant_mode="stochastic", quant_bins=4, max_bin=63)
    # A previous nondeterminism failure must survive a later confirmation timeout.
    faults = {index: "timeout"}
    if index >= 3:
        faults[1] = {"md5": "different-cuda-model"}
    script = WorkerScript(fuzz, monkeypatch, spec, faults=faults)
    full_schedule = list(script.expected)
    script.expected = full_schedule[: index + 1]  # any later subprocess fails this fixture
    with pytest.raises(fuzz.UnresolvedCUDATimeout) as caught:
        fuzz.check_spec(spec)
    stopped = caught.value
    assert script.calls == full_schedule[: index + 1]
    assert spec == script.original == stopped.spec
    assert stopped.worker_spec == full_schedule[index][0]
    assert any("UNRESOLVED_CUDA_TIMEOUT" in failure for failure in stopped.failures)
    assert any("not-run confirmations:" in note for note in stopped.known)
    assert any("all later specs not run" in note for note in stopped.known)
    assert not any("2/2 fixed-data pairs" in note for note in stopped.known)
    if index >= 3:
        assert any("cuda nondeterminism" in failure for failure in stopped.failures)
        assert any("much worse than cpu" in failure for failure in stopped.failures)
        pending = "pair 1 CPU; pair 2 CUDA/CPU" if index == 3 else "pair 2 CPU"
        assert pending in stopped.known[-1]


@pytest.mark.parametrize("index", [0, 1, 3, 5])
def test_main_single_spec_cuda_timeout_is_named_failure_with_original_repro(fuzz, monkeypatch, capsys, spec, index):
    script = WorkerScript(fuzz, monkeypatch, spec, faults={index: "timeout"})
    script.expected = script.expected[: index + 1]
    monkeypatch.setattr(sys, "argv", ["fuzz.py", "--spec", json.dumps(spec)])
    assert fuzz.main() == 125
    assert script.calls == script.expected
    output = capsys.readouterr().out
    assert "FAIL UNRESOLVED_CUDA_TIMEOUT" in output
    assert "not-run confirmations:" in output
    assert "STOP: unresolved CUDA timeout" in output
    assert f"repro: python tests/gates/fuzz.py --spec '{json.dumps(spec)}'" in output
    if index >= 3:
        assert "much worse than cpu" in output


@pytest.mark.parametrize("index", [0, 1, 3, 5])
def test_main_corpus_cuda_timeout_preserves_prior_failure_and_stops_all_later_specs(
    fuzz, monkeypatch, capsys, tmp_path, spec, index
):
    earlier = dict(spec, data_seed=12345, bagging_freq=0)
    later = dict(spec, data_seed=54321)
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([earlier, spec, later]))
    monkeypatch.setattr(fuzz, "CORPUS_FILE", corpus)
    first = WorkerScript(fuzz, monkeypatch, earlier, pairs=None)
    stopping = WorkerScript(fuzz, monkeypatch, spec, faults={index: "timeout"})
    stopping.expected = stopping.expected[: index + 1]
    all_calls = []

    def transport(argv, **kwargs):
        current = json.loads(argv[3])
        all_calls.append(current)
        if current == earlier:
            return first.transport(argv, **kwargs)
        # Includes explicit bagging_seed for confirmation workers.
        assert current["data_seed"] == spec["data_seed"], "later corpus spec must not start"
        return stopping.transport(argv, **kwargs)

    def forbidden_sample(rng):
        pytest.fail("sampling must not start after the corpus CUDA timeout")

    monkeypatch.setattr(fuzz.subprocess, "run", transport)
    monkeypatch.setattr(fuzz, "sample_spec", forbidden_sample)
    monkeypatch.setattr(sys, "argv", ["fuzz.py", "--seed", "123", "--minutes", "1"])
    assert fuzz.main() == 125
    assert first.calls == first.expected
    assert stopping.calls == stopping.expected
    assert len(all_calls) == 3 + index + 1
    output = capsys.readouterr().out
    assert "2 specs tried" in output
    assert "FAIL [corpus[0]] cuda metric" in output
    assert "FAIL [corpus[1]] UNRESOLVED_CUDA_TIMEOUT" in output
    assert "FAIL [corpus[2]]" not in output
    assert f"--spec '{json.dumps(earlier)}'" in output
    assert f"--spec '{json.dumps(spec)}'" in output
    assert "not-run confirmations:" in output
    assert "STOP: unresolved CUDA timeout" in output


def test_main_sampled_cuda_timeout_stops_sampling_and_workers(fuzz, monkeypatch, capsys, tmp_path, spec):
    monkeypatch.setattr(fuzz, "CORPUS_FILE", tmp_path / "absent.json")
    script = WorkerScript(fuzz, monkeypatch, spec, faults={3: "timeout"})
    script.expected = script.expected[:4]
    samples = []

    def sample(rng):
        assert not samples, "no spec may be sampled after the CUDA timeout"
        samples.append(copy.deepcopy(spec))
        return dict(spec)

    monkeypatch.setattr(fuzz, "sample_spec", sample)
    monkeypatch.setattr(sys, "argv", ["fuzz.py", "--seed", "123", "--minutes", "1"])
    assert fuzz.main() == 125
    assert len(samples) == 1
    assert script.calls == script.expected
    output = capsys.readouterr().out
    assert "1 specs tried" in output
    assert "FAIL [seed123#1] UNRESOLVED_CUDA_TIMEOUT" in output


@pytest.mark.parametrize("single_spec", [True, False])
@pytest.mark.parametrize(("metric", "status"), [(0.6, 1), (1.0, 0)])
def test_main_ordinary_quality_and_success_statuses_unchanged(
    fuzz, monkeypatch, capsys, tmp_path, spec, single_spec, metric, status
):
    spec["bagging_freq"] = 0
    script = WorkerScript(fuzz, monkeypatch, spec, pairs=None, initial=metric)
    if single_spec:
        argv = ["fuzz.py", "--spec", json.dumps(spec)]
    else:
        corpus = tmp_path / "corpus.json"
        corpus.write_text(json.dumps([spec]))
        monkeypatch.setattr(fuzz, "CORPUS_FILE", corpus)
        argv = ["fuzz.py", "--seed", "123", "--minutes", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    assert fuzz.main() == status
    assert script.calls == script.expected
    output = capsys.readouterr().out
    assert "UNRESOLVED_CUDA_TIMEOUT" not in output
    assert "STOP:" not in output


@pytest.mark.parametrize("index", [3, 4, 5, 6])
@pytest.mark.parametrize("field", ["bagging_seed", "seed", "feature_fraction_seed", "data_random_seed"])
def test_confirmation_effective_seed_drift_fails(fuzz, monkeypatch, spec, index, field):
    seeds = dict(SEEDS, bagging_seed=400 ^ (0x5EED if index < 5 else 0xBA66))
    seeds[field] += 1
    script = WorkerScript(fuzz, monkeypatch, spec, faults={index: {"effective_seeds": seeds}})
    fails, known = script.check(spec)
    assert any("effective seed/data split evidence differs" in failure for failure in fails)
    assert known == []


@pytest.mark.parametrize(("field", "value"), [("data_seed", 999), ("train_rows", 159), ("effective_seeds", {})])
def test_confirmation_missing_or_changed_data_evidence_fails(fuzz, monkeypatch, spec, field, value):
    script = WorkerScript(fuzz, monkeypatch, spec, faults={3: {field: value}})
    fails, known = script.check(spec)
    assert any("effective seed/data split evidence differs" in failure for failure in fails)
    assert known == []


@pytest.mark.parametrize("index", [0, 2])
def test_original_seed_evidence_must_agree_before_confirming(fuzz, monkeypatch, spec, index):
    script = WorkerScript(fuzz, monkeypatch, spec, pairs=None, faults={index: {"effective_seeds": {}}})
    fails, known = script.check(spec)
    assert any("original effective seed/data split evidence missing or inconsistent" in failure for failure in fails)
    assert known == []
    assert len(script.calls) == 3


@pytest.mark.parametrize("fault", ["error", "timeout"])
def test_original_cpu_failure_is_a_failure_without_confirmations(fuzz, monkeypatch, spec, fault):
    script = WorkerScript(fuzz, monkeypatch, spec, pairs=None, faults={2: fault})
    fails, known = script.check(spec)
    assert len(fails) == 1
    assert "cpu run failed" in fails[0]
    assert known == []


@pytest.mark.parametrize(
    ("mode", "bins", "min_leaf", "metric", "classification"),
    [
        ("none", 0, 20, 0.8, "hard"),
        ("stochastic", 4, 20, 0.5, "known"),
        ("fixedpoint", 16, 1, 0.8, "known"),
        ("fixedpoint", 16, 1, 0.6, "envelope"),
        ("fixedpoint", 16, 20, 0.8, "hard"),
    ],
)
def test_existing_lowbin_classifications_outside_active_bagging_unchanged(
    fuzz, monkeypatch, spec, mode, bins, min_leaf, metric, classification
):
    spec.update(quant_mode=mode, quant_bins=bins, min_data_in_leaf=min_leaf, bagging_freq=0)
    script = WorkerScript(fuzz, monkeypatch, spec, pairs=None, initial=metric)
    fails, known = script.check(spec)
    assert bool(fails) == (classification != "known")
    assert any("low-bin quant variance" in note for note in known) == (classification == "known")
    if classification == "envelope":
        assert "beyond the low-bin quant variance envelope" in fails[0]


def embedded_worker(fuzz, monkeypatch, capsys, spec, device, *, seed_drift=False):
    """Execute the actual self-contained worker with a tiny in-memory falcata stub."""
    captured = {}
    fake = types.ModuleType("falcata")

    class Booster:
        def __init__(self, model_str=None):
            pass

        def model_to_string(self):
            effective = dict(SEEDS, bagging_seed=captured["params"].get("bagging_seed", 400))
            if seed_drift:
                effective["bagging_seed"] += 1
            return "tree\nparameters:\n" + "\n".join(f"[{key}: {value}]" for key, value in effective.items())

        def num_trees(self):
            return spec["rounds"]

        def predict(self, data):
            labels = namespace["y"][namespace["k"] :]
            positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
            assert len(positives), "fixture needs both held-out classes"
            assert len(negatives), "fixture needs both held-out classes"
            predictions = np.arange(len(data), dtype=float) / len(data)
            # Exactly one cross-class tie: its half-pair credit cannot equal
            # any result from the old unique-integer-rank calculation.
            predictions[positives[0]] = predictions[negatives[0]]
            return predictions

    def dataset(data, *, label, params, categorical_feature):
        captured.update(
            train_X=data.copy(),
            train_y=label.copy(),
            dataset_params=dict(params),
            categorical_feature=categorical_feature,
        )
        return object()

    def train(params, ds, *, num_boost_round):
        captured.update(params=dict(params), rounds=num_boost_round)
        return Booster()

    fake.Dataset, fake.train, fake.Booster = dataset, train, Booster
    monkeypatch.setitem(sys.modules, "falcata", fake)
    monkeypatch.setattr(sys, "argv", ["-c", json.dumps(spec), device])
    namespace = {"__name__": "fuzz_worker_under_test"}
    exec(compile(fuzz._WORKER_SOURCE, "<embedded fuzz worker>", "exec"), namespace)
    output = [line[4:] for line in capsys.readouterr().out.splitlines() if line.startswith("OUT ")]
    assert len(output) == 1
    return json.loads(output[0]), captured, namespace


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["none", "fixedpoint"])
@pytest.mark.parametrize("optional_defaults", [False, True])
def test_worker_preserves_defaults_data_and_split_and_applies_only_explicit_bag_seed(
    fuzz, monkeypatch, capsys, spec, device, mode, optional_defaults
):
    spec.update(quant_mode=mode, quant_bins=16 if mode == "fixedpoint" else 0, max_bin=63)
    if optional_defaults:
        for key in ("learning_rate", "max_cat_threshold", "bagging_freq"):
            del spec[key]
    before = copy.deepcopy(spec)
    output, captured, namespace = embedded_worker(fuzz, monkeypatch, capsys, spec, device)
    expected_params = {
        "objective": "binary",
        "num_leaves": 7,
        "max_depth": 10,
        "min_data_in_leaf": 20,
        "feature_fraction": 1.0,
        "bagging_fraction": 0.7,
        "max_bin": 63,
        "learning_rate": 0.1,
        "quant_mode": mode,
        "quant_bins": spec["quant_bins"],
        "cuda_plan": "auto",
        "device_type": device,
        "seed": 42,
        "verbose": -1,
        "metric": "None",
        "num_threads": 8,
        "max_cat_threshold": 32,
    }
    if device == "cpu":
        expected_params.update(deterministic=True, quant_mode="none", quant_bins=0)
    if not optional_defaults:
        expected_params["bagging_freq"] = 1
    assert captured["params"] == captured["dataset_params"] == expected_params
    assert output["effective_seeds"] == SEEDS
    assert "bagging_seed" not in captured["params"]
    assert spec == before
    k = int(spec["n"] * 0.8)
    assert output["data_seed"] == spec["data_seed"]
    assert output["train_rows"] == k
    # This checks linkage to the worker scoring path, including tied scores.
    labels, predictions = namespace["y"][k:], namespace["pred"]
    assert output["metric"] == pytest.approx(pair_auc(labels, predictions), rel=0, abs=1e-15)
    assert output["metric"] == pytest.approx(roc_auc_score(labels, predictions), rel=0, abs=1e-15)
    for bag_seed in (0, SEEDS["bagging_seed"] ^ 0x5EED, SEEDS["bagging_seed"] ^ 0xBA66):
        changed = dict(spec, bagging_seed=bag_seed)
        result, retrained, scope = embedded_worker(fuzz, monkeypatch, capsys, changed, device)
        assert retrained["params"] == retrained["dataset_params"] == dict(expected_params, bagging_seed=bag_seed)
        assert result["effective_seeds"] == dict(SEEDS, bagging_seed=bag_seed)
        assert result["data_seed"] == output["data_seed"]
        assert result["train_rows"] == k
        np.testing.assert_array_equal(retrained["train_X"], captured["train_X"])
        np.testing.assert_array_equal(retrained["train_y"], captured["train_y"])
        np.testing.assert_array_equal(scope["X"], namespace["X"])
        np.testing.assert_array_equal(scope["y"], namespace["y"])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_worker_rejects_ignored_explicit_bag_seed(fuzz, monkeypatch, capsys, spec, device):
    spec["bagging_seed"] = 733
    with pytest.raises(AssertionError, match="effective bagging_seed differs"):
        embedded_worker(fuzz, monkeypatch, capsys, spec, device, seed_drift=True)


@pytest.fixture
def nightly_shell(tmp_path):
    """Real nightly shell, isolated temp paths and CPU stubs for every native command."""
    if os.name != "posix":
        pytest.skip("nightly shell fixture requires POSIX")
    repo, fake_home, bin_dir = tmp_path / "repo", tmp_path / "home", tmp_path / "bin"
    gates = repo / "tests" / "gates"
    gates.mkdir(parents=True)
    (repo / ".gates-venv" / "bin").mkdir(parents=True)
    fake_home.mkdir()
    bin_dir.mkdir()
    calls_file = tmp_path / "calls.jsonl"
    calls_file.write_text("")
    contention = tmp_path / "contention.log"
    contention.write_text("")
    stub = f"""#!{sys.executable}
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["NIGHTLY_TEST_CALLS"], "a") as output:
    output.write(json.dumps([name, *args]) + "\\n")
if name == "git" and "rev-parse" in args:
    print("01234567")
if name == "python" and args and args[0] == "tests/gates/fuzz.py":
    sys.exit(int(os.environ["NIGHTLY_TEST_FUZZ_STATUS"]))
sys.exit(0)
"""
    for path in (bin_dir / "git", bin_dir / "gpuq", repo / ".gates-venv" / "bin" / "python"):
        path.write_text(stub)
        path.chmod(0o755)
    build_stub = r"""#!/bin/bash
printf '%s\n' '["ci_build"]' >> "$NIGHTLY_TEST_CALLS"
"""
    (gates / "ci_build.sh").write_text(build_stub)
    (gates / "cpp_suite.sh").write_text(build_stub.replace("ci_build", "cpp_suite"))
    nightly = Path(__file__).resolve().parents[1] / "gates" / "nightly_local.sh"
    env = {
        **os.environ,
        "HOME": str(fake_home),
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "FALCATA_REPO": str(repo),
        "GPUQ_JOB_ID": "1234",
        "GPUQ_CONTENTION_LOG": str(contention),
        "FALCATA_PERF_SEED_RUNS": "1",
        "FALCATA_PERF_BASELINE": str(tmp_path / "baseline.json"),
        "NIGHTLY_TEST_CALLS": str(calls_file),
    }

    def run(fuzz_status):
        proc = subprocess.run(
            ["/bin/bash", str(nightly)],
            cwd=repo,
            env={**env, "NIGHTLY_TEST_FUZZ_STATUS": str(fuzz_status)},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        calls = [json.loads(line) for line in calls_file.read_text().splitlines()]
        log_dir = fake_home / ".cache" / "falcata-gates"
        logs = list(log_dir.glob("nightly-*.log"))
        assert len(logs) == 1
        return proc, calls, logs[0].read_text(), (log_dir / "last-status").read_text()

    return run


@pytest.mark.parametrize("fuzz_status", [125, 1])
def test_nightly_shell_stops_reserved_timeout_but_continues_quality_failure(nightly_shell, fuzz_status):
    proc, calls, log, status = nightly_shell(fuzz_status)
    workloads = [
        call[1] if call[0] == "python" else call[0] for call in calls if call[0] in ("python", "ci_build", "cpp_suite")
    ]
    before_stop = [
        "ci_build",
        "tests/gates/lattice.py",
        "tests/gates/selective_equivalence.py",
        "tests/gates/fuzz.py",
    ]
    assert proc.returncode == fuzz_status, log + proc.stderr
    assert any(call[0] == "git" for call in calls)
    assert any(call[0] == "gpuq" for call in calls)
    assert "=== ALL GATES PASSED" not in log
    if fuzz_status == 125:
        assert workloads == before_stop
        assert calls[-1] == ["python", "tests/gates/fuzz.py", "--minutes", "45"]
        assert status.startswith("UNRESOLVED_CUDA_TIMEOUT (exit 125)")
        assert "fuzz (45 min, corpus first)" in status
        assert "ABORT: unresolved CUDA timeout; remaining nightly gates not run" in log
        assert "-> validation scoring" not in log
    else:
        assert workloads == before_stop + [
            "tests/gates/valid_metric.py",
            "tests/gates/falb_python.py",
            "tests/gates/falb_roundtrip.py",
            "tests/gates/falb_compat.py",
            "tests/gates/falb_golden.py",
            "tests/gates/falb_fuzz.py",
            "-m",  # pytest is handled by the same fake venv Python
            "cpp_suite",
            "tests/gates/import_xgboost.py",
            "tests/gates/import_catboost.py",
            "tests/gates/canonical.py",
            "tests/gates/canonical.py",
            "tests/gates/sparse_column_view.py",
            "tests/gates/nibble_validate.py",
            "tests/gates/leaf_integrity.py",
            "tests/gates/bench_tier.py",
            "tests/gates/perf_gate.py",
        ]
        assert status.startswith("FAILED ")
        assert "fuzz (45 min, corpus first)" in status
        assert "UNRESOLVED_CUDA_TIMEOUT" not in log
        assert "=== GATES FAILED:" in log
