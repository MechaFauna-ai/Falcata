# coding: utf-8
"""Complete bounded device trees must engage, including final depth and changing sampled layouts.

Every comparison uses one binary and identical data/parameters. The host arm disables both device controllers
(device_levels:off); the complete arm is the default plan (device_tree on), whose shapes outside the depth bound fall
back to the device_levels prefix. Diagnostics prove engagement on every eligible round, including after graph misses,
rather than merely comparing a fallback.
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
# the host flow vs the complete device tree (the default; outside its bound, the device_levels prefix)
_CONTROLLER = {"off": "device_levels:off", "on": "device_tree:on"}


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
    elif name == "fullbig":
        # "full" on 2^18 rows: a depth-8 leaf (1024 rows x 64 quant bins) needs 32-bit histogram bins and its depth-9
        # children (512 rows) 16-bit ones, so the bit-change copy applies at the search of 256 pairs
        X = ((np.arange(1 << 18, dtype=np.uint32)[:, None] >> np.arange(10, dtype=np.uint32)) & 1).astype(np.float32)
        y = X @ np.arange(1, 11, dtype=np.float32)
        params.update(feature_fraction=1.0, min_data_in_leaf=2, quant_mode="fixedpoint", quant_bins=64)
        rounds = 2
    elif name == "fivebin_wide":
        # five-bin features whose most frequent value (3, in 70% of the rows) is not the zero bin: the level
        # fix/subtract needs its most-frequent-bin fix (the masked kernel); every feature in every tree, so the fused
        # used-feature fix/subtract does not take it
        rng_wide = np.random.default_rng(7)
        X = np.where(rng_wide.random((30011, 400)) < 0.7, 3.0, np.floor(rng_wide.random((30011, 400)) * 5))
        X = X.astype(np.float32)
        y = X[:, :8].sum(axis=1) + rng_wide.standard_normal(len(X))
        params["feature_fraction"] = 1.0
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
    for suffix in [",device_levels:off", ",device_tree:on,device_level_graph:" + graph]:
        p = {**params, "cuda_plan": base_plan + suffix}
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


_LIVE_GRID_LINE = re.compile(
    r"device_tree_live_grid: pairs=(\d+) budget=(\d+) fix_subtract=([\w-]+) grid_x=(\d+) copy=(\w+) copy_grid_x=(\d+)"
)


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["off", "on"])
@pytest.mark.parametrize(
    ("case", "extra", "live_grid", "expect"),
    [
        # no need-fix feature: the plain subtract, live (with the copy) at the 256-pair level only
        ("full", "", 1, {"live-subtract", "copy-live"}),
        # the bit-change copy applies at the 256-pair level, which runs live
        ("fullbig", "", 1, {"live-subtract", "copy-live-at-256"}),
        # need-fix features and every feature sampled: the masked fix/subtract, live on every level
        ("fivebin_wide", "", 1, {"live-masked", "copy-live"}),
        ("fivebin_wide", ",fix_subtract_masked:off", 1, {"live-subtract", "copy-live"}),
        # under the budget everywhere, or the key off: the batched kernels only
        ("fivebin_wide", "", 4096, {"no-live"}),
        ("fivebin_wide", "", 0, {"no-live"}),
        # a sampled layout (feature_fraction 0.1, <= 8 bins) takes the fused used-feature fix/subtract and copy
        ("layouts", "", 1, {"fused"}),
    ],
)
def test_device_tree_live_grid_cuda(case, extra, live_grid, expect, graph):
    # cuda_plan key device_tree_live_grid: a level whose worst-case launch (grid_x * pairs blocks) exceeds
    # device_tree_live_grid blocks per SM runs the fix/subtract and the bit-change copy over its live pairs. The diag
    # line of every enqueued level (direct or captured) names the route; the probe compares the complete tree's model
    # bytes with the host flow's.
    data, diag = _run(case, graph, base_plan=f"auto,device_tree_live_grid:{live_grid}{extra}")
    assert diag.count("device_tree: complete") == data["rounds"]
    assert "device_tree: host fallback" not in diag
    lines = _LIVE_GRID_LINE.findall(diag)
    assert lines, diag
    routes = set()
    for pairs, budget, fix, grid_x, copy, copy_grid_x in lines:
        pairs, budget, grid_x, copy_grid_x = int(pairs), int(budget), int(grid_x), int(copy_grid_x)
        assert (budget == 0) == (live_grid == 0)
        if fix != "fused":
            assert fix.startswith("live") == (budget > 0 and grid_x * pairs > budget), (pairs, budget, fix, grid_x)
        if copy != "fused":
            assert (copy == "live") == (budget > 0 and copy_grid_x * pairs > budget), (pairs, budget, copy)
        routes.add(fix)
        routes.add("copy-" + copy)
        if copy == "live" and pairs == 256:
            routes.add("copy-live-at-256")
    if "no-live" in expect:
        assert not any("live" in route for route in routes), routes
    elif "fused" in expect:
        assert routes == {"fused", "copy-fused"}, routes
    else:
        assert expect <= routes, routes


@_REQUIRES_CUDA
@pytest.mark.parametrize("case", ["binding", "unbounded", "too_deep", "bynode", "nonquant", "two_leaf", "l1"])
def test_device_tree_unsupported_cuda(case):
    _, diag = _run(case)
    # outside the depth bound the prefix grows the tree; unsupported configurations use the host flow
    bound = case in {"binding", "unbounded", "too_deep"}
    assert ("device_tree: fallback to the prefix/host flow" if bound else "device_tree: host fallback") in diag
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
            "cuda_plan": "auto," + _CONTROLLER[controller] + ",device_level_graph:" + graph + gate,
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


def _host_cache_probe(graph, scenario):
    X, y, params, _ = _case("root_offset")
    params.update(feature_fraction=1.0, feature_pre_filter=False, quant_bins=64)
    if scenario == "transition":
        params["quant_mode"] = "stochastic"
    weight = None
    # weighted: the full host cache (device_tree_lean_end:off); weighted_lean: weights under the lean tree end
    lean_gate = ",device_tree_lean_end:off" if scenario == "weighted" else ""
    if scenario in {"weighted", "weighted_lean"}:
        weight = np.linspace(0.5, 1.5, len(y))
    elif scenario == "renewal":
        params["quant_train_renew_leaf"] = True
    # A real host-grown tree between complete device trees must rebuild all current child caches from its new root.
    # cuda_plan is resolved at learner Init, not reset_parameter. Keep each arm's controller fixed, and use depth 11
    # at the same leaf budget (outside the bound: the device_levels prefix, then host levels) between two eligible
    # depth-10 trees.
    stages = [(1024, 10, 20, 0.0), (1024, 11, 20, 0.0), (1024, 10, 20, 0.0)]
    if scenario == "transition":
        stages += [
            (100, 10, 20, 0.0),  # A separate budget-binding fallback, then return to whole growth.
            (1024, 10, 20, 0.0),
            (1024, 10, 5000, 0.0),  # Early positive exhaustion.
            (1024, 10, 20, 1e12),  # Stump after stale child caches.
            (1024, 10, 20, 0.0),
        ]
    snapshots = []
    for candidate in [False, True]:
        p = {
            **params,
            "cuda_plan": "auto,"
            + _CONTROLLER["on" if candidate else "off"]
            + ",device_level_graph:"
            + graph
            + lean_gate,
        }
        booster = flc.Booster(p, train_set=flc.Dataset(X, label=y, weight=weight, params=p))
        arm = []

        def snapshot(current):
            live = []

            def capture(preds, _):
                live.append(preds.copy())
                return "capture", 0.0, False

            current.eval_train(feval=capture)
            assert len(live) == 1
            prediction = current.predict(X, raw_score=True, use_fil=False, num_threads=32)
            np.testing.assert_allclose(live[0], prediction, rtol=0.0, atol=1e-12)
            model = "\n".join(line for line in current.model_to_string().splitlines() if "cuda_plan" not in line)
            return model, live[0]

        for leaves, depth, minleaf, gain in stages:
            booster.reset_parameter(
                {
                    "num_leaves": leaves,
                    "max_depth": depth,
                    "min_data_in_leaf": minleaf,
                    "min_gain_to_split": gain,
                }
            )
            booster.update()
            arm.append(snapshot(booster))
        if scenario == "refit":
            fitted = booster.refit(
                X,
                y + 0.25 * X[:, 0],
                decay_rate=0.4,
                reference=booster.train_set,
                dataset_params={"feature_pre_filter": False},
                use_fil=False,
                num_threads=32,
            )
            arm.append(snapshot(fitted))
            assert arm[-1][0] != arm[-2][0], "refit must change the trained model"
            # Continue the original learner after its eligible whole tree and the separate refit operation.
            booster.update()
            arm.append(snapshot(booster))
        snapshots.append(arm)
    for off, on in zip(*snapshots, strict=True):
        assert off[0] == on[0], "host/device model bytes differ across cache history or refit"
        np.testing.assert_array_equal(off[1], on[1])
    print(json.dumps({"scenario": scenario, "stages": len(stages), "snapshots": len(snapshots[0])}))


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["off", "on"])
@pytest.mark.parametrize("scenario", ["transition", "weighted", "weighted_lean", "renewal", "refit"])
def test_device_tree_host_cache_cuda(graph, scenario):
    code = "import runpy,sys; runpy.run_path(sys.argv[1])['_host_cache_probe'](*sys.argv[2:])"
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, graph, scenario],
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
    caches = re.findall(r"device_tree_host_cache: skipped=(\d+) split_info_bytes=(\d+) stale=(\d+)", diag)
    expected = 6 if scenario == "transition" else 3 if scenario == "refit" else 2
    assert len(events) == len(caches) == expected, diag
    assert "device_tree: fallback to the prefix/host flow" in diag
    skip = scenario not in {"weighted", "renewal"}
    for event, cache in zip(events, caches, strict=True):
        splits = int(event[2])
        assert tuple(map(int, cache)) == (int(skip), 0 if skip else 18 * 4 * 1023, int(skip and splits > 0))
    if scenario == "transition":
        assert int(events[3][1]) < 10, "early fixture did not exhaust below final depth"
        assert int(events[3][2]) > 0, "early fixture must split"
        assert int(events[4][2]) == 0, "stump history did not engage"
        assert int(events[5][2]) > 0, "growth after stump did not engage"
    if scenario == "refit":
        assert data["snapshots"] == data["stages"] + 2
    if graph == "off":
        assert all(int(event[3]) == 0 for event in events)


# Final-depth all-row map coverage; preserve the existing108 cases and four root regression controls.
def _all_map_leaf_records(node, depth=0):
    if "split_index" not in node:
        return [(node.get("leaf_index", 0), depth, node["leaf_value"], node["leaf_count"])]
    return _all_map_leaf_records(node["left_child"], depth + 1) + _all_map_leaf_records(node["right_child"], depth + 1)


def _final_all_map_probe(graph, scenario, gate):
    X, y, params, _ = _case("full")
    params.update(feature_pre_filter=False, boost_from_average=False)
    if scenario != "full":
        # Both root branches split X1, but only X0=1 splits X2 at the final depth. Terminal leaf2 has a
        # different value from leaf0: zeroing the map cannot conceal an omitted terminal-gap map store.
        # Keep min_gain=10 above observed fixedpoint noise (host tree3 split feature8 at gain0.11094).
        # Intended X2 splits have gain>8000 even after the row-count shrink; retain the gap assertion.
        y = (1 + 7 * X[:, 0] + 3 * X[:, 1] + 2 * X[:, 0] * X[:, 2]).astype(np.float32)
    weight = np.linspace(0.5, 1.5, len(y)) if scenario in {"weighted", "weighted_lean"} else None
    # weighted: the full host cache (device_tree_lean_end:off); weighted_lean: weights under the lean tree end
    if scenario == "weighted":
        gate += ",device_tree_lean_end:off"
    if scenario == "renewal":
        params["quant_train_renew_leaf"] = True
    stages = [(1024, 10, 0.0, len(y))] if scenario == "full" else [(16, 3, 10.0, len(y))]
    if scenario == "history":
        stages = [
            (16, 3, 10.0, len(y)),
            (16, 3, 10.0, len(y)),
            (16, 3, 10.0, len(y)),
            (16, 3, 1e12, len(y)),  # Stump after complete inline mapping must not inherit its ready flags.
            (1024, 10, 10.0, len(y)),  # Positive early exhaustion requires the original FULL mapper.
            (16, 3, 10.0, len(y)),
            (4, 1, 10.0, len(y)),
            (16, 3, 10.0, len(y) // 2),
            (16, 3, 10.0, len(y) // 2),
            (16, 3, 10.0, len(y)),
        ]
    snapshots = []
    gap_histories = []
    for controller in ["off", "on"]:
        first_leaves, first_depth, first_gain, first_rows = stages[0]
        p = {
            **params,
            "num_leaves": first_leaves,
            "max_depth": first_depth,
            "min_gain_to_split": first_gain,
            "cuda_plan": "auto," + _CONTROLLER[controller] + ",device_level_graph:" + graph + gate,
        }
        ds = flc.Dataset(X[:first_rows], label=y[:first_rows], weight=weight, params=p)
        booster = flc.Booster(p, train_set=ds)
        previous = stages[0]
        arm = []
        gap_history = []
        for leaves, depth, gain, rows in stages:
            if (leaves, depth, gain) != previous[:3]:
                booster.reset_parameter({"num_leaves": leaves, "max_depth": depth, "min_gain_to_split": gain})
            changed_data = rows != previous[3]
            if changed_data:
                ds = flc.Dataset(X[:rows], label=y[:rows], reference=ds, params=p)
            previous = (leaves, depth, gain, rows)
            empty = booster.update(train_set=ds if changed_data else None)
            live = []

            def capture(preds, _, scores=live):
                scores.append(preds.copy())
                return "capture", 0.0, False

            booster.eval_train(feval=capture)
            assert len(live) == 1
            prediction = booster.predict(X[:rows], raw_score=True, use_fil=False, num_threads=32)
            np.testing.assert_allclose(live[0], prediction, rtol=0.0, atol=1e-12)
            model = "\n".join(line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line)
            arm.append((empty, model, live[0]))
            leaf_details = _all_map_leaf_records(booster.dump_model()["tree_info"][-1]["tree_structure"])
            leaf_zero_value = next(value for leaf, _, value, _ in leaf_details if leaf == 0)
            gaps = [
                (leaf, value)
                for leaf, leaf_depth, value, count in leaf_details
                if leaf > 0 and leaf_depth < depth and count > 0 and value != leaf_zero_value
            ]
            if scenario == "history" and depth == 3 and not empty:
                assert gaps, "mixed fixture must have a nonzero-ID terminal gap with output distinct from leaf0"
            gap_history.append(gaps)
        snapshots.append(arm)
        gap_histories.append(gap_history)
    for off, on in zip(*snapshots, strict=True):
        assert off[:2] == on[:2], "all-row inline mapping changed model or empty-tree result"
        np.testing.assert_array_equal(off[2], on[2])
    assert gap_histories[0] == gap_histories[1]
    sys.stdout.write(json.dumps({"stages": len(stages), "nonzero_distinct_terminal_gaps": gap_histories[1]}) + "\n")


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["off", "on"])
@pytest.mark.parametrize(
    ("scenario", "gate"),
    [
        ("full", ""),
        ("history", ""),
        ("history", ",leaf_map_no_zero:off"),
        ("history", ",pdl_level_chain:off,apply_row_batch:off,leaf_map_no_zero:off"),
        ("history", ",async_tree_start:off"),
        ("history", ",final_readback_first:off,early_leaf_map:off"),
        ("weighted", ""),
        ("weighted_lean", ""),
        ("renewal", ""),
    ],
)
def test_device_tree_final_all_map_cuda(graph, scenario, gate):
    """Verify complete final mapping, including distinct nonzero-ID terminal gaps and fallback histories."""
    code = "import runpy,sys; runpy.run_path(sys.argv[1])['_final_all_map_probe'](*sys.argv[2:])"
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, graph, scenario, gate],
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
    inline = re.findall(
        r"device_tree_final_all_map: enabled=(\d+) complete=(\d+) final_splits=(\d+) leaves=(\d+)",
        diag,
    )
    routes = re.findall(r"device_tree_final_all_map_branch: route=(\S+) leaves=(\d+)", diag)
    assert len(events) == len(inline) == len(routes) == data["stages"], diag
    assert "device_tree: host fallback" not in diag
    for event, row, branch in zip(events, inline, routes, strict=True):
        depth, applied, splits = map(int, event[:3])
        enabled, complete, final_splits, leaves = map(int, row)
        route, mapped_leaves = branch
        assert leaves == int(mapped_leaves) == splits + 1
        assert enabled == int(scenario not in {"weighted", "renewal"})
        if complete:
            assert enabled == 1
            assert applied == depth
            assert final_splits > 0
            assert route == "inline-all", "completed final apply must skip BOTH full and subset materialization"
        else:
            assert final_splits == 0
            if splits:
                assert route == "full"
            else:
                assert route in {"stump", "none"}
    if scenario == "full":
        assert tuple(map(int, inline[0])) == (1, 1, 512, 1024)
        assert int(events[0][2]) == 1023
    elif scenario == "history":
        assert all(int(row[1]) == 1 for row in inline[:3])
        assert all(data["nonzero_distinct_terminal_gaps"][:3]), "fixture did not discriminate omitted gap stores"
        assert int(events[3][2]) == 0
        assert int(inline[3][1]) == 0
        assert int(events[4][2]) > 0
        assert int(events[4][1]) < int(events[4][0])
        assert routes[4][0] == "full"
        assert tuple(map(int, inline[6])) == (1, 1, 1, 2), "depth-one completion missed"
        if graph == "on":
            assert int(events[2][4]) > 0, "consecutive identical geometry did not reuse the graph"
    elif scenario == "weighted_lean":
        assert int(inline[0][0]) == 1, "device_tree_lean_end must give weighted trees the inline final map"
    else:
        assert int(inline[0][0]) == int(inline[0][1]) == 0
        assert routes[0][0] == "full"
    if graph == "off":
        assert all(int(event[3]) == 0 for event in events)
