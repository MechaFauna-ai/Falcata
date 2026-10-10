"""Export retained six-state evidence and render the article's throughput chart.

Live export (CPU only; requires a terminal refreshed summary):
    python plot_six_state.py --summary ../summary.json
Portable reproduction from the exported JSON:
    python plot_six_state.py
An explicit --draft creates clearly marked draft files while work is pending.
Only the directory containing this script receives output; raw inputs are read-only.
"""

# Standalone publication program: stdout reports pending state or output files.
# ruff: noqa: D103, T201
import argparse
import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BUILD = "d0bb8c47789e062acbf6b770918a0589a74ae9b2"
LIBRARY = "e67a7ad314a75cec7414ecdcfaa05895040191f50c86c960c432bce919d0a90c"
GROUPS = (
    ("standard-stoch", "Standard stochastic", "FP64 · 4 gradient levels · depth 10"),
    ("standard-fixed", "Standard fixed point", "FP64 · 64 gradient levels · depth 10"),
    ("recipe-fixed-fp32", "Current recipe probe", "FP32 · fixed point 64 · depth 15"),
)
TERMINAL = {"done", "failed", "canceled", "cancelled", "expired"}
RETAIN = (
    "cell",
    "status",
    "error",
    "stop_reason",
    "runtime_build",
    "library_sha256",
    "measurement_manifest_sha256",
    "data_manifest_sha256",
    "gpuq_job_id",
    "input_dtype",
    "n_train",
    "n_features",
    "requested_rounds",
    "num_trees",
    "current_iteration",
    "model_complete",
    "update_calls",
    "construct_s",
    "construct_shared",
    "train_s",
    "trees_per_s",
    "predict_s",
    "eval_s",
    "gpu_mem_peak_mb",
    "rss_peak_mb",
    "resolved_cuda_precision",
    "parameters",
    "resolved_parameters",
    "metrics",
    "tree_sha256",
    "prediction_sha256",
    "feature_bin_distribution",
    "engagement_scope",
    "rows",
    "features",
    "identity",
    "missing_cells_in_gate",
    "accepted_counts_by_representation",
    "prediction_sanity_by_representation",
    "tree_sha256_by_representation",
    "score_era_codes",
    "curve",
)


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def load(path):
    payload = path.read_bytes()
    return json.loads(payload), {"file": path.name, "sha256": digest(payload)}


def warmup_evidence(root, cell_id):
    path = root / "logs" / f"{cell_id}.log"
    payload = path.read_bytes()
    lines = payload.decode().splitlines()
    markers = (
        "CUDARowData:",
        "compact regime needs",
        "compact regime does not fit",
        "colmajor_direct: compact regime",
        "colmajor_direct: mask regime",
        "[hybrid-diag] hist:",
        "quantized construct:",
        "split finder:",
        "compact fill: pair_code5 on",
        "compact fill: pair_code5 off",
    )
    excerpt = []
    for marker in markers:
        for number, line in enumerate(lines, 1):
            if marker in line:
                if not any(item["line"] == number for item in excerpt):
                    excerpt.append({"line": number, "text": line})
                break
    pattern = r"store (\d+) \+ compact view (\d+) \+ split view (\d+) \+ reserve (\d+) MiB, (\d+) MiB available"
    match = re.search(pattern, payload.decode())
    reservation = None
    if match:
        reservation = dict(
            zip(("store", "compact_view", "split_view", "reserve", "available"), map(int, match.groups()), strict=True)
        )
        reservation["required"] = sum(reservation[key] for key in ("store", "compact_view", "split_view", "reserve"))
        reservation["compact_fits_reservation"] = reservation["required"] <= reservation["available"]
        reservation["units"] = "MiB"
    return {
        "file": path.name,
        "bytes": len(payload),
        "sha256": digest(payload),
        "scope": "diagnostic warmup only; timed/full endpoints do not log hot-path dispatch",
        "excerpt": sorted(excerpt, key=lambda item: item["line"]),
        "reservation": reservation,
    }


def wisdom_evidence(record, root):
    result = {}
    for key in ("wisdom_before", "wisdom_after"):
        if key not in record:
            continue
        state = record[key]
        result[key] = {name: state[name] for name in ("exists", "bytes", "sha256") if name in state}
        if state.get("copy"):
            filename = Path(state["copy"]).name
            copy = root / "wisdom" / filename
            ensure(digest(copy.read_bytes()) == state["sha256"], f"Retained wisdom copy differs: {filename}")
            result[key]["retained_copy"] = filename
    return result


def eligible(record, terminal):
    admission = record.get("admission") or {}
    metrics = record.get("metrics") or {}
    speed = record.get("trees_per_s")
    return (
        terminal.get("job_state") == "done"
        and terminal.get("contended") is False
        and admission.get("adm_class") == "strict"
        and admission.get("contended") == 0
        and record.get("status") == "ok"
        and record.get("model_complete") is True
        and metrics.get("sane") is True
        and metrics.get("predictions_finite") is True
        and isinstance(speed, (int, float))
        and math.isfinite(speed)
        and speed > 0
    )


def export(summary_path, draft=False):
    summary, summary_identity = load(summary_path)
    root = summary_path.parent
    manifest, manifest_identity = load(root / "manifest.json")
    plan, plan_identity = load(root / "measurement-manifest.json")
    blueprint, blueprint_identity = load(root / "blueprint.json")
    completeness, completeness_identity = load(root / "complete-test-eras.json")
    ensure(summary["data_manifest_sha256"] == manifest_identity["sha256"], "Data manifest identity differs")
    ensure(summary["measurement_manifest_sha256"] == plan_identity["sha256"], "Measurement identity differs")
    ensure(plan["runtime"] == summary["runtime"], "Summary runtime differs")
    ensure(plan["runtime"]["build"] == BUILD and plan["runtime"]["library_sha256"] == LIBRARY, "Runtime differs")
    ensure(
        (root / "measurement-manifest.sha256").read_text().split()[0] == plan_identity["sha256"],
        "Frozen plan digest differs",
    )
    ensure(
        (root / "blueprint.sha256").read_text().split()[0] == blueprint_identity["sha256"],
        "Frozen blueprint digest differs",
    )
    ensure(completeness["data_manifest_sha256"] == manifest_identity["sha256"], "Completeness audit identity differs")
    cells = [{"id": "sentinel-equivalence", "kind": "gate"}, *plan["cells"]]
    ensure(len(cells) == 27 and summary["planned_records"] == 27, "Unexpected experiment scope")
    ensure(set(summary["endpoints"]) == {cell["id"] for cell in cells}, "Endpoint scope differs")
    waiting = [cell["id"] for cell in cells if summary["endpoints"][cell["id"]]["status"] == "pending"]
    job_states = {row.get("job_state") for row in summary["endpoints"].values() if row["status"] != "pending"}
    settled = not waiting and bool(job_states) and job_states <= TERMINAL
    if not settled and not draft:
        return None, {
            "status": "pending",
            "counts": summary["counts"],
            "waiting": waiting,
            "job_states": sorted(job_states, key=str),
            "outputs_written": False,
        }
    records = {}
    dates = set()
    for cell in cells:
        name = cell["id"]
        terminal = summary["endpoints"][name]
        path = root / "raw" / f"{name}.json"
        if not path.exists():
            ensure(terminal["status"] == "pending", f"Missing recorded endpoint: {name}")
            records[name] = {"cell": cell, "status": "pending", "headline_eligible": False}
            continue
        record, identity = load(path)
        ensure(record["cell"] == cell, f"Cell identity differs: {name}")
        ensure(record["measurement_manifest_sha256"] == plan_identity["sha256"], f"Plan identity differs: {name}")
        ensure(record["data_manifest_sha256"] == manifest_identity["sha256"], f"Data identity differs: {name}")
        ensure(record["runtime_build"] == BUILD and record["library_sha256"] == LIBRARY, f"Runtime differs: {name}")
        ensure(record["status"] == terminal["status"], f"Summary is stale: {name}")
        retained = {key: record[key] for key in RETAIN if key in record}
        retained.update({key: value for key, value in record.items() if key.endswith("_s")})
        retained.update(wisdom_evidence(record, root))
        retained["raw_identity"] = identity
        retained["terminal_queue"] = {key: terminal.get(key) for key in ("gpuq_job_id", "job_state", "contended")}
        retained["admission_class"] = (record.get("admission") or {}).get("adm_class")
        retained["headline_eligible"] = eligible(record, terminal)
        if cell["kind"] == "warmup":
            retained["diagnostic_log"] = warmup_evidence(root, name)
        records[name] = retained
        start = (record.get("admission") or {}).get("started_at")
        if start is not None:
            dates.add(
                datetime.fromtimestamp(start, timezone.utc).astimezone(ZoneInfo("Europe/Zurich")).date().isoformat()
            )
    ensure(dict(Counter(row["status"] for row in records.values())) == summary["counts"], "Status counts differ")
    chart = []
    for name, label, detail in GROUPS:
        arms = {}
        for arm in ("five", "six"):
            chosen = [records[f"{name}-{arm}-timed{draw}"] for draw in (1, 2, 3)]
            for record in chosen:
                if record["status"] != "pending":
                    ensure(
                        record["resolved_cuda_precision"] == plan["configs"][name]["resolved_cuda_precision"],
                        "Precision differs",
                    )
            valid = all(record["headline_eligible"] for record in chosen)
            values = [record["trees_per_s"] for record in chosen] if valid else []
            if valid:
                ensure(all(record["num_trees"] == 2000 for record in chosen), "Probe did not complete 2000 trees")
                ensure(
                    math.isclose(
                        statistics.median(values),
                        summary["timed_probe_groups"][name]["arms"][arm]["actual_trees_per_s_median"],
                        rel_tol=1e-12,
                    ),
                    "Summary throughput differs",
                )
            arms[arm] = {
                "eligible": valid,
                "draws": values,
                "median": statistics.median(values) if valid else None,
                "range": [min(values), max(values)] if valid else None,
                "endpoint_ids": [record["cell"]["id"] for record in chosen],
            }
        chart.append(
            {
                "group": name,
                "label": label,
                "detail": detail,
                "rounds": 2000,
                "statistic": "median of three timed draws; range is observed min–max",
                "arms": arms,
            }
        )
    arms = {}
    for arm in ("five", "six"):
        record = records[f"standard-stoch-{arm}-full30000"]
        valid = record["headline_eligible"]
        if valid:
            ensure(
                record["num_trees"] == 30000 and record["resolved_cuda_precision"] == "fp64", "Full-pair scope differs"
            )
        arms[arm] = {
            "eligible": valid,
            "draws": [record["trees_per_s"]] if valid else [],
            "median": record.get("trees_per_s") if valid else None,
            "range": None,
            "endpoint_ids": [record["cell"]["id"]],
        }
    chart.append(
        {
            "group": "full30000",
            "label": "Full stochastic run",
            "detail": "FP64 · 4 gradient levels · depth 10",
            "rounds": 30000,
            "statistic": "one complete draw per arm; no uncertainty estimate",
            "arms": arms,
        }
    )
    gate_ok = records["sentinel-equivalence"]["status"] == "ok"
    ensure(gate_ok or draft, "Native sentinel equivalence prerequisite failed")
    timed_ids = [cell["id"] for cell in cells if cell["kind"].startswith("timed") or cell["kind"] == "full30000"]
    excluded = [name for name in timed_ids if not records[name]["headline_eligible"]]
    failed = [name for name, record in records.items() if record["status"] not in ("ok", "pending")]
    return {
        "schema": 1,
        "status": "measured_complete" if settled else "draft_pending",
        "source_summary": summary_identity,
        "source_data_manifest": manifest_identity,
        "source_measurement_manifest": plan_identity,
        "measurement_dates_europe_zurich": sorted(dates),
        "source_blueprint": blueprint_identity,
        "source_completeness_audit": completeness_identity,
        "frozen_helper_and_harness_sha256": plan["files_sha256"],
        "blueprint_helper_and_harness_sha256": blueprint["files_sha256"],
        "runtime": {"build": BUILD, "library_sha256": LIBRARY},
        "device": "RTX 5090",
        "target": manifest["target"],
        "n_train": manifest["train_end"],
        "n_features": manifest["n_features"],
        "n_retained": manifest["n_rows"],
        "split": manifest["split"],
        "sources": manifest["sources"],
        "shared": manifest["shared"],
        "representations": {
            arm: {key: manifest["arms"][arm][key] for key in ("X", "meta", "state_counts")} for arm in ("five", "six")
        },
        "conversion_validation": manifest["validation"],
        "source_row_count": manifest["validation"]["all_source_rows_verified"],
        "conversion_validation_scope": "CPU manifest precedes the GPU gate; its original pending marker is preserved. See retained sentinel-equivalence endpoint for the completed exactness gate.",
        "configs": plan["configs"],
        "expected_records": 27,
        "recorded_records": 27 - len(waiting),
        "status_counts": summary["counts"],
        "retained_counts": {
            "planned": 27,
            "recorded": 27 - len(waiting),
            "ok": summary["counts"].get("ok", 0),
            "pending": len(waiting),
            "failed": len(failed),
            "timed_or_full": len(timed_ids),
            "excluded_timed_or_full": len(excluded),
            "warmups_excluded_from_timing_statistics": 6,
            "prerequisite_gates": 1,
        },
        "failed_record_ids": failed,
        "pending_record_ids": waiting,
        "headline_ineligible_record_ids": excluded,
        "terminal_contention": {name: row.get("terminal_queue") for name, row in records.items()},
        "sentinel_equivalence_passed": gate_ok,
        "endpoints": records,
        "chart": chart,
        "warmup_observed": {
            name: {arm: summary["timed_probe_groups"][name]["arms"][arm]["warmup_observed"] for arm in ("five", "six")}
            for name, _, _ in GROUPS
        },
        "score_era_codes": summary["score_era_codes"],
        "limitations": summary["limitations"],
        "complete_test_era_audit": completeness,
        "chart_eligible": settled
        and gate_ok
        and all(row["arms"][arm]["eligible"] for row in chart for arm in ("five", "six")),
    }, None


def render(data, folder, draft=False):
    ensure(data["runtime"]["build"] == BUILD and data["runtime"]["library_sha256"] == LIBRARY, "Chart runtime differs")
    ensure(draft or data["status"] == "measured_complete", "Final chart requires settled evidence")
    os.environ.setdefault("MPLCONFIGDIR", str(folder / ".mplconfig"))
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415
    from matplotlib.ticker import FixedLocator, ScalarFormatter  # noqa: PLC0415

    colors = {"five": "#647985", "six": "#216A8B"}
    ink, muted = "#192A36", "#586774"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "falcata-six-state-throughput",
        }
    )
    fig = plt.figure(figsize=(11.4, 8.6), facecolor="white")
    fig.subplots_adjust(left=0.29, right=0.86, top=0.75, bottom=0.28, hspace=0.50)
    grid = fig.add_gridspec(2, 1, height_ratios=(3, 1))
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    for ax, rows in zip(axes, (data["chart"][:3], data["chart"][3:]), strict=True):
        ax.set_xscale("log")
        ax.set_xlim(7, 380)
        ax.xaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 300]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(axis="x", which="both", length=0, pad=7, colors=muted, labelsize=10)
        ax.grid(axis="x", which="major", color="#E4EAF0", linewidth=0.8, zorder=0)
        ax.set_yticks([])
        ax.set_ylim(len(rows) - 0.55, -0.65)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for index, row in enumerate(rows):
            ax.text(
                -0.035,
                index - 0.04,
                row["label"],
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=11.5,
                weight="bold",
                color=ink,
            )
            ax.text(
                -0.035,
                index + 0.25,
                row["detail"],
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
                color=muted,
            )
            if all(row["arms"][arm]["range"] is not None for arm in ("five", "six")):
                five_low, five_high = row["arms"]["five"]["range"]
                six_low, six_high = row["arms"]["six"]["range"]
                decimals = 2 if six_high < 20 else 1
                ranges = f"5: {five_low:.1f}–{five_high:.1f} · 6: {six_low:.{decimals}f}–{six_high:.{decimals}f}"
                ax.text(
                    -0.035,
                    index + 0.46,
                    ranges,
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=8,
                    color=muted,
                )
            for arm, offset in (("five", -0.16), ("six", 0.16)):
                entry = row["arms"][arm]
                y = index + offset
                if not entry["eligible"]:
                    ax.text(11, y, f"{arm}: pending / ineligible", ha="left", va="center", fontsize=9, color=muted)
                    continue
                value = entry["median"]
                if entry["range"] is not None:
                    low, high = entry["range"]
                    ax.errorbar(
                        value,
                        y,
                        xerr=[[value - low], [high - value]],
                        fmt="none",
                        ecolor=colors[arm],
                        elinewidth=1.6,
                        capsize=4,
                        capthick=1.3,
                        zorder=3,
                    )
                ax.scatter(value, y, s=85, color=colors[arm], edgecolor="white", linewidth=1, zorder=4)
                ax.annotate(
                    f"{value:.1f}",
                    (value, y),
                    xytext=(8, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    fontsize=11,
                    weight="bold",
                    color=colors[arm],
                )
            five, six = (row["arms"][arm]["median"] for arm in ("five", "six"))
            if five is not None and six is not None:
                ax.text(
                    1.035,
                    index,
                    f"{six / five:.2f}×",
                    transform=ax.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=13,
                    weight="bold",
                    color=colors["six"],
                )
    axes[0].set_title("2,000 trees · median of 3 timed draws per arm", loc="left", pad=16, fontsize=10.5, color=muted)
    axes[1].set_title("30,000 trees · one complete run per arm", loc="left", pad=13, fontsize=10.5, color=muted)
    axes[1].set_xlabel("Actual trees per second   ·   log scale   ·   higher is faster", labelpad=11, color=ink)
    title = "Five states and six states, measured" if not draft else "DRAFT — six-state measurement still pending"
    fig.text(0.045, 0.955, title, fontsize=21, weight="bold", color=ink, va="top")
    fig.text(
        0.045,
        0.895,
        "Same rows, labels and seed; only the era-constant-2 missing treatment changes",
        fontsize=11.5,
        color=muted,
    )
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=colors[arm], markersize=8, label=label)
        for arm, label in (("five", "Five states: raw 0–4"), ("six", "Six states: 0–4 + missing"))
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.04, 0.856),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handletextpad=0.5,
        columnspacing=2.1,
    )
    fig.text(0.88, 0.815, "Six / five\nthroughput", fontsize=9, ha="left", color=muted)
    dates = ", ".join(data["measurement_dates_europe_zurich"])
    queue_label = (
        "strict GPUQ; terminal, uncontended" if data["chart_eligible"] else "endpoint eligibility retained in JSON"
    )
    captions = (
        f"{dates} · RTX 5090 · Falcata d0bb8c47 · {queue_label}",
        f"Build 1230 · {data['n_train']:,} training rows × {data['n_features']:,} features · max_bin=255 · Ender20 target",
        "Ranges list five / six observed min–max across 3 timed draws; warmups excluded. The full pair has no variation estimate.",
        "Training time includes initial cache/tuner work. Models differ; five-era sanity does not establish equivalent quality.",
        "Current recipe: explicit FP32, unscaled leaf floor 101079; this 2,000-tree probe is not a production run.",
    )
    for y, caption in zip((0.198, 0.160, 0.122, 0.084, 0.046), captions, strict=True):
        fig.text(0.045, y, caption, fontsize=9, color=muted)
    prefix = folder / ("six-state-throughput-draft" if draft else "six-state-throughput")
    outputs = {}
    for suffix in ("png", "svg"):
        path = prefix.with_suffix(f".{suffix}")
        metadata = {
            "Description": "Paired five/six-state Numerai trees/s; repeated probe medians and a separate single full pair; no equal-quality claim."
        }
        if suffix == "svg":
            metadata["Date"] = None
        fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15, metadata=metadata)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        outputs[suffix] = str(path)
    plt.close(fig)
    return outputs


def main():
    folder = Path(__file__).absolute().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, help="export from refreshed final summary and adjacent frozen inputs")
    parser.add_argument(
        "--data", type=Path, default=folder / "six-state-data.json", help="portable exported evidence to render"
    )
    parser.add_argument(
        "--draft", action="store_true", help="write clearly marked draft outputs while work remains pending"
    )
    args = parser.parse_args()
    if args.summary:
        data, pending = export(args.summary.absolute(), args.draft)
        if pending:
            print(json.dumps(pending, indent=2))
            return
        destination = folder / ("six-state-data-draft.json" if args.draft else "six-state-data.json")
        destination.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    else:
        data = json.loads(args.data.read_text())
    outputs = render(data, folder, args.draft)
    print(
        json.dumps(
            {
                "status": data["status"],
                "chart_eligible": data["chart_eligible"],
                "counts": data["status_counts"],
                "outputs": outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
