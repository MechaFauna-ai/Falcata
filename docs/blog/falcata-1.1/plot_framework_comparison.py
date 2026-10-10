"""Plot archived launch frameworks alongside measured modern Falcata endpoints.

    python docs/blog/falcata-1.1/plot_framework_comparison.py

Refresh compact chart data from the record-level audit:
    python plot_framework_comparison.py --matrix PATH/benchmark-matrix.json

This CPU-only program copies recorded quality values; it runs no benchmarks and
does not compute CORR, AUC, Sharpe, or any other model-quality metric.
"""

# Standalone plotting program: stdout reports its output files.
# ruff: noqa: D103, T201
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

INK = "#1A2C37"
BLUE = "#216A8B"
LIGHT_BLUE = "#9BBFCF"
MUTED = "#586774"
GRAY = "#BDC7CE"
LINE = "#D8E3E8"
PALE = "#F5F8FA"
PANEL_SPECS = (
    ("numerai", "numerai-deep", "Numerai deep", "30,000 requested rounds · one timed draw", "CORR"),
    ("numerai", "numerai", "Numerai example", "2,000 rounds · median of three timed draws", "CORR"),
    ("epsilon", "deep", "Epsilon deep", "500 rounds · median of three timed draws", "AUC"),
)
FRAMEWORK_LABELS = {
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM CUDA",
    "lightgbm-ocl": "LightGBM OpenCL",
}


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_data(matrix_path):
    matrix = json.loads(matrix_path.read_text())
    catalog = matrix["record_catalog"]
    panels = []
    for dataset, regime, label, detail, metric_label in PANEL_SPECS:
        workload = next(row for row in matrix["workloads"] if (row["dataset"], row["regime"]) == (dataset, regime))
        launch = workload["launch"]["falcata-stoch"]
        modern = workload["latest_measured"]["falcata-stoch"]
        metric = "corr_mean" if dataset == "numerai" else "auc"

        def arm_row(arm, role, library, name, metric=metric):
            ensure(arm["historical_published_timing_usable_with_caveats"], f"Failed or incomplete timing: {name}")
            records = [catalog[key] for key in arm["selected_records"]]
            ensure(all(row["status"] == "ok" and row["metrics"]["sane"] is True for row in records), name)
            seconds = arm["training_seconds"]["median"]
            ensure(math.isfinite(seconds) and seconds > 0, name)
            return {
                "role": role,
                "library": library,
                "label": name,
                "train_s": seconds,
                "training_seconds": arm["training_seconds"],
                "timed_draws": arm["recorded_timed"],
                "metric_name": metric,
                "metric": arm["metrics"][metric],
                "package_versions": sorted({row.get("version", "unrecorded") for row in records}),
                "source": arm["source"],
                "records": arm["selected_records"],
                "recorded_endpoints": records,
                "strict_quiet_verified": arm["strict_quiet_verified"],
                "caveats": arm["caveats"],
            }

        ensure(modern["absolute_timing_eligible"], f"Modern timing lacks strict quiet provenance: {label}")
        rows = [
            arm_row(modern, "modern_falcata", "falcata-stoch", "Falcata · modern measured build"),
            arm_row(launch, "launch_falcata", "falcata-stoch", "Falcata · launch snapshot"),
        ]
        omissions = []
        for library, framework_label in FRAMEWORK_LABELS.items():
            arm = workload["archived_competitors"].get(library)
            if arm and arm["historical_published_timing_usable_with_caveats"]:
                rows.append(arm_row(arm, "archived_competitor", library, framework_label))
            elif arm:
                omissions.append(
                    {
                        "library": library,
                        "reason": "No complete sanity-passing archived timed set; no bar or artificial time assigned",
                        "statuses": arm["timed_statuses"],
                        "failure_records": arm["failure_records_all_kinds"],
                        "records": [catalog[key] for key in arm["all_group_records"]],
                    }
                )
        metadata = matrix["sources"][modern["source"]]
        runtime = metadata.get("runtime_build", metadata.get("runtime_build_verified_by_contemporaneous_build_record"))
        panels.append(
            {
                "dataset": dataset,
                "regime": regime,
                "label": label,
                "detail": detail,
                "metric_label": metric_label,
                "modern_measurement_date": "2026-10-08" if modern["source"] == "oct8" else "2026-10-09",
                "modern_runtime": runtime,
                "modern_library_identity": {key: value for key, value in metadata.items() if "library_" in key},
                "archived_measurement_date": "August 2026",
                "launch_snapshot_div_modern_ratio": launch["training_seconds"]["median"]
                / modern["training_seconds"]["median"],
                "contemporary_framework_gap_established": False,
                "quality_equivalence_established": False,
                "rows": rows,
                "omitted_frameworks": omissions,
            }
        )
    return {
        "schema": 1,
        "matrix_sha256": digest(matrix_path),
        "timer": "Whole training call, excluding Dataset construction and CPU prediction/scoring",
        "hardware": "One RTX 5090; measurements made on an otherwise idle GPU, with no recorded contention",
        "framework_comparison": "Archived August 2026 competitor endpoints, not current framework reruns",
        "quality_contract": "Shown endpoint values are not quality-equivalence or identical-model claims",
        "baseline_contract": "Launch snapshot printed version1.0.0; not a controlled tagged-v1.0.0 runtime comparison",
        "numerai_data_contract": "Historical cache:5463797 train rows ×3555 features; target label unidentified. Oct8 deep uses float32; Oct9 example int8. Both launch panels use float32 per contemporaneous loader.",
        "l2_contract": "Framework defaults retained: Falcata/LightGBM0, XGBoost1, CatBoost3",
        "upstream_cuda_contract": "Archived LightGBM CUDA does not enforce the requested depth cap; its Epsilon model capacity differs",
        "sources": matrix["sources"],
        "caveat_definitions": matrix["caveat_definitions"],
        "panels": panels,
    }


def row_detail(row, panel):
    if row["role"] == "modern_falcata":
        date = panel["modern_measurement_date"][5:]
        return f"Oct {int(date[-2:])} · {panel['modern_runtime'][:8]} · stochastic"
    if row["role"] == "launch_falcata":
        return "Aug 2026 · version label 1.0.0 · stochastic"
    return f"Aug 2026 · version {' / '.join(row['package_versions'])}"


def seconds_label(value):
    if value >= 1000:
        return f"{value:,.0f} s"
    if value >= 10:
        return f"{value:.1f} s"
    return f"{value:.3f} s"


def plot(data, output):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt  # noqa: PLC0415
    from matplotlib.patches import FancyBboxPatch, Patch  # noqa: PLC0415

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(13.4, 13.4), facecolor="white")
    fig.text(0.038, 0.968, "Quantized training since the launch snapshot", fontsize=24, weight="bold")
    fig.text(
        0.038, 0.935, "Falcata stochastic on an RTX 5090 · training time, lower is better", fontsize=12, color=MUTED
    )
    legend = [
        Patch(facecolor=BLUE, label="Modern Falcata measurement"),
        Patch(facecolor=LIGHT_BLUE, label="Launch Falcata snapshot"),
        Patch(facecolor=GRAY, label="Archived competitor measurement"),
    ]
    fig.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0.032, 0.919),
        frameon=False,
        ncol=3,
        fontsize=10.5,
        handlelength=1.4,
        columnspacing=2.0,
    )
    for number, panel in enumerate(data["panels"]):
        card_top = 0.864 - number * 0.267
        card_bottom = card_top - 0.243
        card = FancyBboxPatch(
            (0.033, card_bottom),
            0.934,
            0.243,
            boxstyle="round,pad=0.009,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor="white",
            edgecolor=LINE,
            linewidth=1.0,
            zorder=-1,
        )
        fig.add_artist(card)
        fig.text(0.052, card_top - 0.025, panel["label"], fontsize=17, weight="bold")
        fig.text(0.052, card_top - 0.047, panel["detail"], fontsize=10.5, color=MUTED)
        fig.text(
            0.936,
            card_top - 0.025,
            f"{panel['launch_snapshot_div_modern_ratio']:.2f}×",
            fontsize=21,
            color=BLUE,
            weight="bold",
            ha="right",
        )
        fig.text(0.936, card_top - 0.043, "launch / modern time", fontsize=9.5, color=MUTED, ha="right")
        fig.text(0.888, card_top - 0.059, f"Holdout {panel['metric_label']} ↑", fontsize=10, color=MUTED, ha="center")
        rows = panel["rows"]
        n_rows = len(rows)
        ax = fig.add_axes((0.299, card_bottom + 0.051, 0.448, 0.130))
        ax.set_xscale("log")
        ax.set_xlim(1, 15000)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.set_yticks([])
        ax.set_xticks([1, 10, 100, 1000, 10000], labels=["1", "10", "100", "1,000", "10,000"])
        ax.tick_params(axis="x", length=0, labelsize=9.5, pad=6)
        ax.grid(axis="x", color=LINE, lw=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for index, row in enumerate(rows):
            color = BLUE if row["role"] == "modern_falcata" else LIGHT_BLUE if row["role"] == "launch_falcata" else GRAY
            ax.barh(index, row["train_s"] - 1, left=1, height=0.57, color=color, edgecolor="none")
            ax.annotate(
                seconds_label(row["train_s"]),
                xy=(row["train_s"], index),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                color=INK,
                fontsize=10.5,
                weight="bold" if index == 0 else "normal",
            )
            row_y = ax.get_position().y1 - ax.get_position().height * (index + 0.5) / n_rows
            fig.text(
                0.052,
                row_y + 0.0038,
                row["label"],
                fontsize=10.5,
                weight="bold" if row["role"] == "modern_falcata" else "normal",
                va="center",
            )
            fig.text(0.052, row_y - 0.0073, row_detail(row, panel), fontsize=8.8, color=MUTED, va="center")
            fig.text(
                0.888,
                row_y,
                f"{row['metric']['median']:.7f}",
                fontsize=11,
                ha="center",
                va="center",
                color=BLUE if row["role"] == "modern_falcata" else INK,
            )
        ax.axhline(1.5, color=LINE, linewidth=1, linestyle=(0, (2, 3)))
        if panel["dataset"] == "numerai":
            note = "LightGBM CUDA: archived run failed / OOM, no bar."
        else:
            note = "LightGBM CUDA: archived model exceeds the requested depth cap."
        fig.text(0.052, card_bottom + 0.009, note, fontsize=8.8, color=MUTED)
        fig.text(
            0.523, card_bottom + 0.012, "Training seconds · logarithmic scale", fontsize=9, color=MUTED, ha="center"
        )
    fig.text(
        0.038,
        0.062,
        "Historical comparison, not a rerun of today’s competing frameworks or a tagged-runtime experiment.",
        fontsize=10.5,
        color=INK,
    )
    fig.text(
        0.038,
        0.043,
        "Quality is shown, not declared equivalent. Framework L2 defaults differ; Numerai uses a historical cache with unidentified label.",
        fontsize=9.3,
        color=MUTED,
    )
    fig.text(
        0.038,
        0.026,
        "Dataset construction and prediction are outside the timer. Numerai deep is one full requested-round draw; example’s input changed float32 → int8.",
        fontsize=9.3,
        color=MUTED,
    )
    png = output.with_suffix(".png")
    svg = output.with_suffix(".svg")
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.1, metadata={"Date": None})
    plt.close(fig)
    return [str(png), str(svg)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, help="Optional source audit; refreshes compact data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib-cache"))
    compact_path = args.output_dir / "framework-comparison-data.json"
    if args.matrix:
        data = compact_data(args.matrix)
        compact_path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    else:
        data = json.loads(compact_path.read_text())
    ensure(len(data["panels"]) == 3, "Expected three panels")
    for panel in data["panels"]:
        ensure(panel["rows"][0]["strict_quiet_verified"], "Modern bar lacks strict quiet provenance")
        ensure(
            not panel["contemporary_framework_gap_established"],
            "Historical bars cannot establish a current framework gap",
        )
    outputs = plot(data, args.output_dir / "framework-comparison")
    print(json.dumps({"written": [str(compact_path), *outputs], "gpu_access": False, "metrics_recomputed": False}))


if __name__ == "__main__":
    main()
