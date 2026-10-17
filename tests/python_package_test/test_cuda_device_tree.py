# coding: utf-8
"""Complete bounded device trees must engage, including final depth and changing sampled layouts.

Every comparison uses one binary and identical data/parameters. The host arm disables both device controllers;
the complete arm disables the prefix to ensure the experimental key supplies the actual growth. Diagnostics
prove engagement on every requested round, including after graph misses, rather than merely comparing a fallback.
"""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")


def _case(name):
    rng = np.random.default_rng(42)
    params = {
        "objective": "regression",
        "device_type": "cuda",
        "num_leaves": 1024,
        "max_depth": 10,
        "learning_rate": 0.001,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.1,
        "quant_mode": "stochastic",
        "seed": 42,
        "num_threads": 32,
        "metric": "None",
        "verbose": -1,
    }
    rounds = 12
    if name == "full":
        # All 1024 final leaves are necessary to represent this ten-bit linear target exactly. Repeated rows keep
        # the integer quantized histograms exact; the final-depth fixture must actually reach all 1024 leaves.
        X = ((np.arange(65536, dtype=np.uint32)[:, None] >> np.arange(10, dtype=np.uint32)) & 1).astype(np.float32)
        y = X @ np.arange(1, 11, dtype=np.float32)
        params.update(feature_fraction=1.0, min_data_in_leaf=2, quant_mode="fixedpoint", quant_bins=64)
        rounds = 3
    else:
        bins = rng.integers(3, 7, size=400) if name == "layouts" else np.full(3555 if name == "reuse" else 400, 5)
        X = np.floor(rng.random((30011, len(bins))) * bins).astype(np.float32)
        y = X[:, :8].sum(axis=1) + rng.standard_normal(len(X))
        if name in {"layouts", "reuse"}:
            rounds = 70 if name == "reuse" else 40
        elif name in {"root_offset", "root_regularized"}:
            # Disabling the initial average leaves a large, deterministic nonzero root output. A split records
            # that output as internal_value[0], even though its child leaf outputs still train correctly if lost.
            y = (32 + X[:, :8].sum(axis=1)).astype(np.float32)
            params.update(boost_from_average=False, quant_mode="fixedpoint")
            if name == "root_regularized":
                # Opposite-sign children still improve gain when the large positive root is clipped. A purely
                # positive offset with every output clipped to the same value would produce a stump instead.
                y = (32 + 64 * (X[:, 0] - 2) + X[:, 1:8].sum(axis=1)).astype(np.float32)
                params.update(feature_fraction=1.0, lambda_l1=0.0, lambda_l2=3.0, max_delta_step=2.5)
            rounds = 3
        elif name == "stump":
            params["min_gain_to_split"] = 1e12
            rounds = 1
        elif name == "early":
            params["min_data_in_leaf"] = 5000
        elif name == "binding":
            params["num_leaves"] = 100
        elif name == "unbounded":
            params["max_depth"] = -1
        elif name == "too_deep":
            params.update(num_leaves=4096, max_depth=12)
        elif name == "bynode":
            params["feature_fraction_bynode"] = 0.5
        elif name == "l1":
            params["lambda_l1"] = 0.7
        elif name == "nonquant":
            params["quant_mode"] = "none"
        elif name == "shallow":
            params.update(num_leaves=4, max_depth=1)
        elif name == "two_leaf":
            params.update(num_leaves=2, max_depth=1)
        elif name not in {"fivebin", "reset", "reset_auto_budget"}:
            raise ValueError(name)
    return X, y, params, rounds


def _probe(name, base_plan, graph):
    X, y, params, rounds = _case(name)
    models = []
    counts = []
    depths = []
    roots = []
    reset_budgets = []
    for suffix in [",device_tree:off", ",device_tree:on,device_level_graph:" + graph]:
        p = {**params, "cuda_plan": base_plan + ",device_levels:off" + suffix}
        ds = flc.Dataset(X, label=y, params=p)
        budgets = []
        if name in {"reset", "reset_auto_budget"}:
            booster = flc.Booster(p, train_set=ds)
            for bound in [1, 10, 3, 10]:
                reset_params = {"max_depth": bound}
                if name == "reset":
                    # Config::Set auto-reduces num_leaves when omitted at a shallow depth. Preserve the explicit
                    # budget so this case actually exercises complete-controller capacity/cache depth resets.
                    reset_params["num_leaves"] = p["num_leaves"]
                booster.reset_parameter(reset_params)
                for _ in range(3):
                    booster.update()
                native_budget = re.search(r"^\[num_leaves: (\d+)\]$", booster.model_to_string(), re.MULTILINE)
                assert native_budget is not None
                budgets.append(int(native_budget.group(1)))
        else:
            booster = flc.train(p, ds, num_boost_round=rounds)
        reset_budgets.append(budgets)
        models.append("\n".join(line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line))
        info = booster.dump_model()["tree_info"]
        counts.append([t["num_leaves"] for t in info])
        roots.append([t["tree_structure"].get("internal_value", 0.0) for t in info])

        def depth(node):
            return (
                0
                if "leaf_index" in node or "split_index" not in node
                else 1 + max(depth(node["left_child"]), depth(node["right_child"]))
            )

        depths.append([depth(t["tree_structure"]) for t in info])
    if name in {"root_offset", "root_regularized"}:
        assert all(count > 1 for count in counts[0])
        assert all(abs(value) > 1e-6 for value in roots[0]), "host root must be nonzero to discriminate this regression"
        print(json.dumps({"host_roots": roots[0], "device_roots": roots[1]}), flush=True)
    assert models[0] == models[1], "host/device model bytes differ"
    assert counts[0] == counts[1]
    assert reset_budgets[0] == reset_budgets[1]
    print(
        json.dumps(
            {"counts": counts[1], "depths": depths[1], "roots": roots[1], "rounds": rounds, "budgets": reset_budgets[1]}
        )
    )


def _run(name, graph="on", base_plan="auto"):
    code = "import runpy,sys; runpy.run_path(sys.argv[1])['_probe'](*sys.argv[2:])"
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, name, base_plan, graph],
        env=dict(os.environ, FALCATA_DEBUG="diag"),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.splitlines()[-1]), result.stderr


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["off", "on"])
@pytest.mark.parametrize("case", ["fivebin", "full", "early", "stump", "shallow", "layouts", "reuse", "reset"])
def test_device_tree_complete_cuda(case, graph):
    data, diag = _run(case, graph)
    events = re.findall(
        r"device_tree: complete depth=(\d+) applied_levels=(\d+) splits=(\d+) "
        r"growth_readbacks=1 host_tail=0 graph=(\d+) hits=(\d+) misses=(\d+)",
        diag,
    )
    assert len(events) == data["rounds"], diag
    assert "device_tree: host fallback" not in diag
    assert all(int(e[2]) + 1 == count for e, count in zip(events, data["counts"], strict=True))
    if case == "full":
        assert data["counts"] == [1024] * data["rounds"]
        assert data["depths"] == [10] * data["rounds"]
        assert all(int(e[1]) == 10 for e in events)
    elif case == "reset":
        assert data["budgets"] == [1024] * 4
        assert [int(e[0]) for e in events] == [1] * 3 + [10] * 3 + [3] * 3 + [10] * 3
        assert all(depth <= int(e[0]) for e, depth in zip(events, data["depths"], strict=True))
    elif case == "early":
        assert all(0 < int(e[1]) < 10 for e in events)
    elif case == "stump":
        assert data["counts"] == [1]
        assert int(events[0][1]) == 0
    elif case == "layouts" and graph == "on":
        # Exceed the ordinary prefix's nine-miss cutoff and still require every complete tree above.
        assert int(events[-1][5]) >= 9
        assert "graphs off" not in diag
    elif case == "reuse" and graph == "on":
        # Five-bin sampled columns spread over several original partitions. The final whole-row code-word launch
        # stays invariant while the preliminary per-partition maxima change with each sample.
        assert int(events[-1][4]) > 0
        assert "graphs off" not in diag
        layouts = re.findall(r"device_tree: layout partcols=(\d+) partjoint=(\d+) construct=(\d+)", diag)
        assert len(layouts) == data["rounds"]
        assert len({layout[0] for layout in layouts}) > 1
        assert any(layout[2] == "4" for layout in layouts)
        assert any(
            layouts[i][0] != layouts[i - 1][0] and int(events[i][4]) > int(events[i - 1][4])
            for i in range(1, len(events))
        ), "no graph reuse across changing preliminary partition maxima"
    if graph == "off":
        assert all(int(e[3]) == 0 for e in events)


@_REQUIRES_CUDA
@pytest.mark.parametrize("case", ["binding", "unbounded", "too_deep", "bynode", "nonquant", "two_leaf", "l1"])
def test_device_tree_unsupported_cuda(case):
    _, diag = _run(case)
    assert "device_tree: host fallback" in diag
    assert "device_tree: complete" not in diag


@_REQUIRES_CUDA
@pytest.mark.parametrize("plan", ["auto,fused_root_hist:off", "auto,pdl_level_chain:off", "auto,skip_empty_tail:off"])
def test_device_tree_dependencies_cuda(plan):
    data, diag = _run("fivebin", base_plan=plan)
    assert diag.count("device_tree: complete") == data["rounds"]
    assert "device_tree: host fallback" not in diag


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["off", "on"])
@pytest.mark.parametrize("case", ["root_offset", "root_regularized"])
def test_device_tree_root_output_cuda(case, graph):
    data, diag = _run(case, graph)
    assert all(abs(value) > 1e-6 for value in data["roots"])
    assert diag.count("device_tree: complete") == data["rounds"]
    assert "device_tree: host fallback" not in diag


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["off", "on"])
def test_device_tree_reset_auto_budget_cuda(graph):
    # A depth-only reset to one automatically reduces the native leaf budget to two. Later depth-only resets do
    # not raise it again: this is the documented Config::Set adjustment, and two-leaf trees use host growth.
    data, diag = _run("reset_auto_budget", graph)
    assert data["budgets"] == [2] * 4
    assert len(data["counts"]) == data["rounds"] == 12
    assert all(count <= 2 for count in data["counts"])
    assert all(depth <= 1 for depth in data["depths"])
    assert "device_tree: host fallback" in diag
    assert "device_tree: complete" not in diag


# Draft V9 coverage: publish/consume markers prove byte reuse; exact models and live training scores guard it.
def _end_readback_probe(graph, gate):
    X, y, params, _ = _case("root_offset")
    # The same Dataset survives min-leaf resets in both directions; retain every feature during construction.
    params.update(feature_fraction=1.0, feature_pre_filter=False)
    # First stump, growth, shrink, another stump, growth after stale staging, capacity growth, graph reuse,
    # early positive exhaustion and a final stump. Repeating an identical stage deliberately avoids a reset.
    stages = [
        (8, 3, 1e12, 20),
        (1024, 10, 0.0, 20),
        (4, 1, 0.0, 20),
        (4, 1, 1e12, 20),
        (16, 3, 0.0, 20),
        (2048, 10, 0.0, 20),
        (2048, 10, 0.0, 20),
        (2048, 10, 0.0, 5000),
        (2048, 10, 1e12, 20),
    ]
    snapshots = []
    for controller in ["off", "on"]:
        first_leaves, first_depth, first_gain, first_minleaf = stages[0]
        p = {
            **params,
            "num_leaves": first_leaves,
            "max_depth": first_depth,
            "min_gain_to_split": first_gain,
            "min_data_in_leaf": first_minleaf,
            "cuda_plan": "auto,device_levels:off,device_tree:" + controller + ",device_level_graph:" + graph + gate,
        }
        booster = flc.Booster(p, train_set=flc.Dataset(X, label=y, params=p))
        previous = stages[0]
        arm = []
        for stage in stages:
            leaves, depth, gain, minleaf = stage
            if stage != previous:
                booster.reset_parameter(
                    {"num_leaves": leaves, "max_depth": depth, "min_gain_to_split": gain, "min_data_in_leaf": minleaf}
                )
            previous = stage
            empty = booster.update()
            live_scores = []

            def capture_scores(preds, _, scores=live_scores):
                scores.append(preds.copy())
                return "score_capture", 0.0, False

            booster.eval_train(feval=capture_scores)
            assert len(live_scores) == 1
            # Explicit regular CPU prediction; FIL's optional fp32 routing is irrelevant to score-cache parity.
            prediction = booster.predict(X, raw_score=True, use_fil=False, num_threads=32)
            np.testing.assert_allclose(live_scores[0], prediction, rtol=0.0, atol=1e-12)
            model = "\n".join(line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line)
            arm.append((empty, model, live_scores[0]))
        snapshots.append(arm)
    for off, on in zip(*snapshots, strict=True):
        assert off[:2] == on[:2], "model bytes or empty-tree result differ"
        np.testing.assert_array_equal(off[2], on[2])
    print(json.dumps({"stages": len(stages)}))


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["off", "on"])
@pytest.mark.parametrize("gate", ["", ",final_readback_first:off", ",readback_kernel:off", ",async_tree_start:off"])
def test_device_tree_end_readback_cuda(graph, gate):
    code = "import runpy,sys; runpy.run_path(sys.argv[1])['_end_readback_probe'](*sys.argv[2:])"
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, graph, gate],
        env=dict(os.environ, FALCATA_DEBUG="diag"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\nSTDERR:\n" + result.stderr
    data = json.loads(result.stdout.splitlines()[-1])
    diag = result.stderr
    events = re.findall(
        r"device_tree: complete depth=(\d+) applied_levels=(\d+) splits=(\d+) "
        r"growth_readbacks=1 host_tail=0 graph=(\d+) hits=(\d+) misses=(\d+)",
        diag,
    )
    staged = re.findall(r"device_tree_end_readback: queued=(\d+) ready=(\d+) leaves=(\d+) counts=(\d+)", diag)
    consumed = re.findall(r"device_tree_end_readback: consume slab=(\d+) counts=(\d+) leaves=(\d+)", diag)
    assert len(events) == len(staged) == data["stages"] == 9, diag
    assert "device_tree: host fallback" not in diag
    assert [int(event[2]) == 0 for event in events] == [True, False, False, True, False, False, False, False, True]
    assert int(events[7][1]) < 10, "early-exhaustion fixture did not stop below the depth bound"
    enabled = not gate
    nonstumps = []
    for event, row in zip(events, staged, strict=True):
        queued, ready, leaves, counts = map(int, row)
        splits = int(event[2])
        assert leaves == splits + 1
        assert queued == int(enabled)
        assert ready == counts == int(enabled and splits > 0)
        if splits > 0:
            nonstumps.append(leaves)
    if gate == ",async_tree_start:off":
        assert not consumed
    else:
        assert len(consumed) == len(nonstumps)
        for row, leaves in zip(consumed, nonstumps, strict=True):
            assert tuple(map(int, row)) == (int(enabled), int(enabled), leaves)
    if graph == "off":
        assert all(int(event[3]) == 0 for event in events)
    else:
        assert int(events[6][4]) > 0, "identical consecutive geometry did not reuse its captured graph"
