"""Render the article's FP64 training-time chart from compact recorded values.

Reproduce with the checked-in training-time-data.json:
    python docs/blog/falcata-1.1/plot_training_time.py

Refresh the compact values from the original strict JSON evidence:
    python plot_training_time.py --evidence PATH/strict-timing-evidence.json
No training, GPU access, or benchmark execution occurs here.
"""

# Standalone plotting program: stdout reports its output files.
# ruff: noqa: D103, T201
import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

BASELINE = "09b6be9724f330f947c86de6e819654b077f70f2"
SELECTED = "d0bb8c47789e062acbf6b770918a0589a74ae9b2"
WORKLOADS = (
    ("year-noquant", "Year", "500 rounds · deep", 3),
    ("epsilon-noquant", "Epsilon", "500 rounds · deep", 3),
    ("higgs-noquant", "Higgs", "500 rounds · deep", 3),
    ("numerai-noquant", "Numerai example", "2,000 rounds", 3),
    ("numerai-deep", "Numerai deep", "30,000 rounds · single pair", 1),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def compact_values(evidence_path):
    evidence = json.loads(evidence_path.read_text())
    jobs = {job["id"]: job for job in evidence["jobs"]}
    rows = []
    for name, label, detail, count in WORKLOADS:
        row = {"workload": name, "label": label, "detail": detail, "timed_runs_per_build": count}
        for arm, runtime in (("baseline", BASELINE), ("candidate", SELECTED)):
            filename = f"strict-{name}-{arm}.jsonl"
            ensure(not evidence["provenance_issues"][filename], f"Provenance issue: {filename}")
            records = [
                record
                for record in evidence["raw_files"][filename]["records"]
                if record.get("kind", "").startswith("timed")
            ]
            ensure(len(records) == count, f"Incorrect timed count: {filename}")
            ensure(
                {record["kind"] for record in records} == {f"timed{n}" for n in range(1, count + 1)},
                "Duplicate or missing timed draw",
            )
            for record in records:
                job = jobs[record["gpuq_job_id"]]
                ensure(
                    job["state"] == "done"
                    and job["exit_code"] == 0
                    and job["adm_class"] == "strict"
                    and job["contended"] == 0,
                    "Timing was not successful strict uncontended work",
                )
                ensure(record["status"] == "ok" and record["metrics"]["sane"] is True, "Nonpassing model endpoint")
                ensure(
                    record["runtime_build"] == runtime and record["resolved_cuda_precision"] == "fp64",
                    "Unexpected runtime/precision",
                )
                ensure(math.isfinite(record["train_s"]) and record["train_s"] > 0, "Invalid training time")
            row[f"{arm}_train_s"] = statistics.median(record["train_s"] for record in records)
            row[f"{arm}_library_sha256"] = records[0]["library_sha256"]
            row[f"{arm}_gpuq_job_ids"] = sorted({record["gpuq_job_id"] for record in records})
            row[f"{arm}_metrics"] = {
                key: {
                    "min": min(record["metrics"][key] for record in records),
                    "median": statistics.median(record["metrics"][key] for record in records),
                    "max": max(record["metrics"][key] for record in records),
                }
                for key in records[0]["metrics"]
                if key != "sane"
            }
        rows.append(row)
    return {
        "schema": 1,
        "baseline_build": BASELINE,
        "selected_build": SELECTED,
        "source_evidence": {
            "file": evidence_path.name,
            "sha256": digest(evidence_path),
            "repository_path": "docs/2026-10-09_noquant-timings/strict-timing-evidence.json",
        },
        "device": "RTX 5090",
        "precision": "fp64",
        "admission": "strict quiet GPUQ",
        "statistic": "100 * (1 - candidate_train_s / baseline_train_s); first four are medians of three timed draws",
        "quality_limits": [
            "Timing reductions do not establish equivalent model quality.",
            "Higgs includes one lower candidate AUC draw.",
            "Numerai deep is one complete run per build; no uncertainty estimate or extrapolation.",
            "Numerai uses a fixed historical build1226 cache with an unidentified label; CORR comes from NumeraiEvaluator.",
        ],
        "rows": rows,
    }


def render(data, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".mplconfig"))
    # Cache placement and headless rendering are set before importing pyplot.
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import PercentFormatter  # noqa: PLC0415

    ensure(data["baseline_build"] == BASELINE and data["selected_build"] == SELECTED, "Build identity differs")
    ensure(data["precision"] == "fp64", "Chart requires FP64")
    ensure([row["workload"] for row in data["rows"]] == [item[0] for item in WORKLOADS], "Chart scope differs")
    rows = data["rows"]
    reductions = [100 * (1 - row["candidate_train_s"] / row["baseline_train_s"]) for row in rows]
    ensure(all(math.isfinite(value) and value > 0 for value in reductions), "Invalid elapsed-time reduction")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "falcata-1.1-fp64-training-time",
        }
    )
    fig, ax = plt.subplots(figsize=(10.0, 5.7), facecolor="white")
    fig.subplots_adjust(left=0.285, right=0.95, top=0.78, bottom=0.30)
    color, muted = "#216A8B", "#586774"
    bars = ax.barh(range(5), reductions, height=0.52, color=color, zorder=3)
    bars[-1].set_facecolor("#EDF3F6")
    bars[-1].set_edgecolor(color)
    bars[-1].set_linewidth(1.3)
    bars[-1].set_hatch("///")
    for index, (row, value) in enumerate(zip(rows, reductions, strict=True)):
        ax.annotate(
            f"{value:.1f}%",
            (value, index),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=13,
            weight="bold",
            color=color,
        )
        ax.text(
            -0.025,
            index - 0.10,
            row["label"],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=12,
            color="#202D38",
            weight="bold",
        )
        ax.text(
            -0.025,
            index + 0.17,
            row["detail"],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9.5,
            color=muted,
        )
    ax.set_ylim(4.55, -0.55)
    ax.set_xlim(0, 45)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_yticks([])
    ax.set_xlabel("Less elapsed training time", labelpad=11, fontsize=11, color="#344552")
    ax.tick_params(axis="x", length=0, pad=8, labelsize=10, colors=muted)
    ax.grid(axis="x", color="#E4EAF0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.055, 0.945, "Less time for FP64 training", fontsize=21, weight="bold", color="#192A36", va="top")
    fig.text(0.055, 0.875, "Selected runtime versus the October 8 baseline", fontsize=12, color=muted, va="top")
    deep = rows[-1]
    baseline_corr = deep["baseline_metrics"]["corr_mean"]["median"]
    selected_corr = deep["candidate_metrics"]["corr_mean"]["median"]
    captions = (
        "09b6be97 → d0bb8c47  ·  RTX 5090  ·  FP64  ·  strict quiet GPUQ",
        "First four: medians of 3 timed runs per build. Numerai deep: one complete run per build; warmups excluded.",
        "Timing reductions do not establish equivalent quality. Higgs included one lower candidate AUC draw.",
        f"Numerai deep CORR: {baseline_corr:.6f} → {selected_corr:.6f} (historical cache; unidentified label).",
    )
    for y, text in zip((0.194, 0.142, 0.095, 0.048), captions, strict=True):
        fig.text(0.055, y, text, fontsize=9, color=muted, va="center")
    paths = {suffix: output.with_suffix(suffix) for suffix in (".png", ".svg")}
    fig.savefig(
        paths[".png"],
        dpi=200,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.15,
        metadata={
            "Description": "FP64 training reductions versus October baseline; single-pair Numerai deep is explicitly labelled."
        },
    )
    fig.savefig(
        paths[".svg"],
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.15,
        metadata={
            "Date": None,
            "Description": "FP64 training reductions versus October baseline; single-pair Numerai deep is explicitly labelled.",
        },
    )
    plt.close(fig)
    return {
        "png": str(paths[".png"]),
        "svg": str(paths[".svg"]),
        "percent_less_time": {row["workload"]: value for row, value in zip(rows, reductions, strict=True)},
    }


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=folder / "training-time-data.json")
    parser.add_argument(
        "--evidence", type=Path, help="optionally rebuild compact values from finalized strict JSON evidence"
    )
    parser.add_argument(
        "--output", type=Path, default=folder / "fp64-training-time", help="output filename prefix without extension"
    )
    args = parser.parse_args()
    if args.evidence:
        data = compact_values(args.evidence)
        args.data.parent.mkdir(parents=True, exist_ok=True)
        args.data.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    else:
        data = json.loads(args.data.read_text())
    print(json.dumps(render(data, args.output), indent=2))


if __name__ == "__main__":
    main()
