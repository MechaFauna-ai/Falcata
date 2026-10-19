"""Controller intent, actual engagement and complete model equality across resets."""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")
_HOST = "auto,device_levels:off,device_tree:off"


def _model(booster):
    # Only controller intent differs; compare every model field and all other parameters.
    return "\n".join(line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line)


def _data():
    rng = np.random.default_rng(192)
    x = rng.integers(0, 6, size=(4096, 24)).astype(np.float32)
    y = (x[:, 0] + 2 * x[:, 1] - x[:, 2] + (x[:, 3] > 2)).astype(np.float32)
    params = {
        "objective": "regression",
        "device_type": "cuda",
        "quant_mode": "fixedpoint",
        "quant_bins": 64,
        "max_bin": 15,
        "max_depth": 4,
        "num_leaves": 16,
        "min_data_in_leaf": 8,
        "feature_pre_filter": False,
        "num_threads": 2,
        "verbosity": -1,
        "seed": 192,
    }
    return x, y, params


def _explicit_probe(plan, unsupported):
    x, y, params = _data()
    if unsupported:
        params["max_depth"] = -1
    models = []
    for spec in (_HOST, plan):
        p = {**params, "cuda_plan": spec}
        booster = flc.train(p, flc.Dataset(x, label=y, params=p), num_boost_round=3)
        assert all(tree["num_leaves"] > 1 for tree in booster.dump_model()["tree_info"])
        models.append(_model(booster))
    assert models[0] == models[1], "complete host/requested model differs"
    print(json.dumps({"equal": True}))


def _reset_probe():
    x, y, params = _data()
    for initial in ("auto,device_levels:on,device_tree:off", _HOST):
        p = {**params, "cuda_plan": initial}
        dataset = flc.Dataset(x, label=y, params=p, free_raw_data=False).construct()
        booster = flc.Booster(p, train_set=dataset)
        reference = flc.Booster({**p, "cuda_plan": _HOST}, train_set=dataset)
        # The reference construction changes the global plan. The first
        # learner must retain its explicit prefix request regardless.
        booster.update()
        reference.update()
        assert _model(booster) == _model(reference)
        for spec in (
            initial,
            "auto,device_tree:on,device_levels:off,device_level_graph:off",
            _HOST,
            "auto,device_tree:off,device_levels:on",
        ):
            booster.reset_parameter({"cuda_plan": spec})
            booster.update()
            reference.update()
            assert _model(booster) == _model(reference)
        replacement = flc.Dataset(x[:2048], label=y[:2048], reference=dataset, params=p).construct()
        booster.update(train_set=replacement)
        reference.update(train_set=replacement)
        assert _model(booster) == _model(reference)
    print(json.dumps({"equal": True}))


def _auto_probe(mode):
    # Dense fixture: 260 supplied columns, 258 valid features, half-width U=129.
    # Missing low-bin fixtures separately exercise admission without the capacity proxy.
    rng = np.random.default_rng(193)
    dense = mode in {"", "nan_dense", "zero_dense", "retired_dense_prefix"}
    if dense:
        valid_features = 64 if mode == "retired_dense_prefix" else 258
    elif mode in {"nan", "zero"}:
        valid_features = 130
    elif mode == "prefix":
        valid_features = 96
    else:
        valid_features = 28
    values = 9 if dense else 4
    x = np.empty((102400, valid_features + 2), dtype=np.float32)
    x[:, :valid_features] = rng.integers(0, values, size=(len(x), valid_features), dtype=np.int8)
    x[:, valid_features:] = 1
    y = (x[:, 0] + 2 * x[:, 1] - x[:, 2]).astype(np.float32)
    if mode in {"nan", "nan_dense"}:
        x[::16, valid_features - 1] = np.nan
    _, _, params = _data()
    params.update(max_depth=8, num_leaves=256, feature_fraction=0.5 if mode else 1.0, feature_pre_filter=True)
    if mode in {"zero", "zero_dense"}:
        params["zero_as_missing"] = True
    if mode in {"nan", "zero"}:
        params["min_data_in_leaf"] = 10000  # capacity proxy=10 < 256; low-bin whole must not require it
    if mode in {"prefix", "retired_dense_prefix"}:
        params.update(max_depth=11, num_leaves=2048)
    if mode == "retired_dense_prefix":
        params["feature_fraction"] = 1.0
    modifiers = {
        "l1": {"lambda_l1": 0.1},
        "l2": {"lambda_l2": 1.0},
        "path_smooth": {"path_smooth": 1.0},
        "min_gain": {"min_gain_to_split": 1e12},
        "hessian_floor": {"min_sum_hessian_in_leaf": 1e6},
        "objective": {"objective": "poisson"},
    }
    params.update(modifiers.get(mode, {}))
    if mode == "objective":
        y += 4  # valid nonnegative Poisson labels
    auto_plan = "auto,device_level_graph:off" if mode == "graph_off" else "auto"
    auto_params = {**params, "cuda_plan": auto_plan}
    fields = {}
    if mode == "weight":
        fields["weight"] = np.linspace(0.5, 1.5, len(x), dtype=np.float64)
    elif mode == "init_score":
        fields["init_score"] = np.full(len(x), 0.125, dtype=np.float64)
    dataset = flc.Dataset(x, label=y, params=auto_params, free_raw_data=False, **fields).construct()
    booster = flc.Booster(auto_params, train_set=dataset)
    reference = flc.Booster({**params, "cuda_plan": _HOST}, train_set=dataset)

    def step(train_set=None):
        booster.update(train_set=train_set)
        reference.update(train_set=train_set)
        assert _model(booster) == _model(reference), "complete AUTO/host model differs after reset"

    step()
    if not mode:
        for change in (
            {"feature_fraction": 0.5},
            {"num_leaves": 255},
            {"feature_fraction": 1.0},
            {"feature_fraction": 0.5, "num_leaves": 256},
        ):
            booster.reset_parameter(change)
            reference.reset_parameter(change)
            step()
        smaller = flc.Dataset(x[:8192], label=y[:8192], reference=dataset, params=auto_params).construct()
        step(smaller)
        step(dataset)
    print(json.dumps({"equal": True}))


def _run(probe, *args):
    code = "import json,runpy,sys; runpy.run_path(sys.argv[1])[sys.argv[2]](*json.loads(sys.argv[3]))"
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, probe, json.dumps(args)],
        env=dict(os.environ, FALCATA_DEBUG="diag"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\nSTDERR:\n" + result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["equal"]
    return result.stderr


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("plan", "route"),
    [
        ("auto,device_tree:TRUE,device_levels:0", "whole"),
        ("auto,device_levels:false,device_tree:1,row_batch:off", "whole"),
        ("auto,device_tree:off", "prefix"),
        ("auto,device_levels:on,device_level_graph:off", "prefix"),
        ("auto,device_levels:1,device_tree:on,device_tree:false", "prefix"),
        ("auto,device_levels:off,row_batch:on", "host"),
    ],
)
def test_explicit_controller_route_and_model_cuda(plan, route):
    diag = _run("_explicit_probe", plan, False)
    assert f"controller: requested=explicit resolved={route} reason=explicit_controller_overrides" in diag
    if route == "whole":
        assert diag.count("device_tree: complete") == 3
    elif route == "prefix":
        assert "device_levels: 3 device-driven levels per tree" in diag
    else:
        assert "device-driven levels per tree" not in diag
    assert "host fallback" not in diag


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("plan", "route", "fallback"),
    [
        ("auto,device_tree:on,device_levels:off", "whole", "device_tree"),
        ("auto,device_tree:off,device_levels:on", "prefix", "device_levels"),
    ],
)
def test_resolved_request_is_separate_from_actual_fallback_cuda(plan, route, fallback):
    diag = _run("_explicit_probe", plan, True)
    assert f"controller: requested=explicit resolved={route} reason=explicit_controller_overrides" in diag
    assert f"{fallback}: host fallback" in diag
    assert "device_tree: complete" not in diag


@_REQUIRES_CUDA
def test_controller_config_and_data_reset_cuda():
    diag = _run("_reset_probe")
    assert diag.count("resolved=whole reason=explicit_controller_overrides phase=reset_config") == 2
    assert diag.count("resolved=prefix reason=explicit_controller_overrides phase=reset_data") == 2
    assert diag.count("device_tree: complete") == 2
    whole_lines = [line for line in diag.splitlines() if "device_tree: complete" in line]
    assert all("graph=0" in line for line in whole_lines)
    assert diag.count("device_levels: 3 device-driven levels per tree") >= 4
    assert "phase=reset_data levels_explicit=1 tree_explicit=1 rows=2048" in diag


@_REQUIRES_CUDA
def test_auto_controller_valid_sample_count_and_resets_cuda():
    diag = _run("_auto_probe", "")
    decisions = re.findall(r"controller: requested=auto resolved=(\w+) reason=(\w+) phase=(\w+)([^\n]*)", diag)
    assert decisions, diag
    assert "features=258 sampled=258" in decisions[0][3]
    assert all("missing_features=0" in decision[3] for decision in decisions)
    if decisions[0][1] == "unqualified_hardware":
        assert all(decision[0] == "host" for decision in decisions)
        assert "device-driven levels per tree" not in diag
        assert "device_tree: complete" not in diag
    else:
        assert [decision[0] for decision in decisions] == ["host", "whole", "whole", "host", "whole", "host", "whole"]
        assert "features=258 sampled=129" in decisions[1][3]
        assert decisions[2][1] == "v2_whole_dense"  # D=8,L=255 admits the native final single-leaf budget bind
        assert decisions[5][1:3] == ("v2_min_rows", "reset_data")
        assert "rows=8192" in decisions[5][3]
        assert decisions[6][1:3] == ("v2_whole_dense", "reset_data")
        assert diag.count("device_tree: complete") == 4
        assert "device_levels: 7 device-driven levels per tree" not in diag


@_REQUIRES_CUDA
@pytest.mark.parametrize("missing_mode", ["nan", "zero", "nan_dense", "zero_dense"])
def test_auto_controller_uses_all_valid_missing_metadata_cuda(missing_mode):
    diag = _run("_auto_probe", missing_mode)
    decision = re.search(r"controller: requested=auto resolved=(\w+) reason=(\w+)[^\n]*missing_features=(\d+)", diag)
    assert decision is not None, diag
    assert int(decision[3]) > 0
    if decision[2] == "unqualified_hardware":
        assert decision[1] == "host"
        assert "device_tree: complete" not in diag
    elif missing_mode in {"nan", "zero"}:
        assert (decision[1], decision[2]) == ("whole", "v2_whole_lowbin")
        assert diag.count("device_tree: complete") == 1
    else:
        assert (decision[1], decision[2]) == ("host", "v2_missing_mode")
        assert "device-driven levels per tree" not in diag
        assert "device_tree: complete" not in diag


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "modifier",
    ["graph_off", "weight", "init_score", "l1", "l2", "path_smooth", "min_gain", "hessian_floor", "objective"],
)
def test_auto_controller_declines_unqualified_modifiers_cuda(modifier):
    diag = _run("_auto_probe", modifier)
    decision = re.search(r"controller: requested=auto resolved=(\w+) reason=(\w+)", diag)
    assert decision is not None, diag
    assert decision[1] == "host"
    expected = "v2_graph_disabled" if modifier == "graph_off" else "v2_unqualified_modifiers"
    assert decision[2] in {expected, "unqualified_hardware"}
    assert "device-driven levels per tree" not in diag
    assert "device_tree: complete" not in diag


@_REQUIRES_CUDA
@pytest.mark.parametrize("mode", ["prefix", "retired_dense_prefix"])
def test_auto_controller_retains_only_smallbin_prefix_cuda(mode):
    diag = _run("_auto_probe", mode)
    decision = re.search(r"controller: requested=auto resolved=(\w+) reason=(\w+)", diag)
    assert decision is not None, diag
    if decision[2] == "unqualified_hardware":
        assert decision[1] == "host"
        assert "device-driven levels per tree" not in diag
    elif mode == "prefix":
        assert (decision[1], decision[2]) == ("prefix", "v2_prefix_smallbin")
        assert "device_levels: 10 device-driven levels per tree" in diag
    else:
        assert (decision[1], decision[2]) == ("host", "v2_no_region")
        assert "device-driven levels per tree" not in diag
    assert "device_tree: complete" not in diag
