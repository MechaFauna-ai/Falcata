"""Render strict timing contrasts directly from archived immutable JSON evidence.

Reproduction after archival:
    python plots/plot_timing_overview.py
Defaults read the strict/auxiliary JSON files beside the plots directory.
Explicit --strict, --auxiliary and --output overrides are also supported.
"""

# Standalone artifact generator: stdout reports output paths.
# ruff: noqa: D103, T201
import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

BASELINE = "09b6be9724f330f947c86de6e819654b077f70f2"
SELECTED = "d0bb8c47789e062acbf6b770918a0589a74ae9b2"
LIBRARIES = {
    BASELINE: "cb6de00388a1d85e32c8a8d1144310777d73767e1e56a8b32ab810e47d636e08",
    SELECTED: "e67a7ad314a75cec7414ecdcfaa05895040191f50c86c960c432bce919d0a90c",
}
NOQUANT = (
    ("year-noquant", "Year · deep (500 rounds)", 3),
    ("epsilon-noquant", "Epsilon · deep (500)", 3),
    ("higgs-noquant", "HIGGS · deep (500)", 3),
    ("numerai-noquant", "Numerai · example (2,000)", 3),
    ("numerai-deep", "Numerai · deep (30,000)\nsingle run per build", 1),
)
GRAPH = (
    ("covtype-deep-quant", "Covtype · deep (100 rounds)"),
    ("covtype-shallow-quant", "Covtype · shallow (100)"),
    ("year-shallow-quant", "Year · shallow (100)"),
    ("fraud-deep-quant", "Fraud · deep (100)"),
    ("higgs-shallow-quant", "HIGGS · shallow (100)"),
    ("epsilon-shallow-quant", "Epsilon · shallow (100)"),
    ("numerai-example-quant", "Numerai · example (200)"),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def eligible_job(job):
    return job["state"] == "done" and job["exit_code"] == 0 and job["adm_class"] == "strict" and job["contended"] == 0


def timed_records(records, count, runtime, jobs):
    timed = [record for record in records if record.get("kind", "").startswith("timed")]
    require(len(timed) == count, f"Expected {count} timed records, found {len(timed)}")
    keyed = {record["kind"]: record for record in timed}
    require(set(keyed) == {f"timed{index}" for index in range(1, count + 1)}, "Timed draw identities differ")
    for record in timed:
        require(record["status"] == "ok" and record["metrics"]["sane"] is True, "Nonpassing timed model")
        require(
            eligible_job(jobs[record["gpuq_job_id"]]), "Timed record is not from a successful strict uncontended job"
        )
        require(
            record["runtime_build"] == runtime and record["library_sha256"] == LIBRARIES[runtime],
            "Runtime/library identity differs",
        )
        require(record["resolved_cuda_precision"] == "fp64", "Unexpected precision")
        require(
            isinstance(record["train_s"], (int, float)) and math.isfinite(record["train_s"]) and record["train_s"] > 0,
            "Invalid recorded train time",
        )
    return keyed


def ratio_row(name, label, reference, optimized, count):
    ref_values = [reference[kind]["train_s"] for kind in sorted(reference)]
    opt_values = [optimized[kind]["train_s"] for kind in sorted(optimized)]
    pairs = [
        {
            "kind": kind,
            "reference_train_s": reference[kind]["train_s"],
            "optimized_train_s": optimized[kind]["train_s"],
            "ratio": reference[kind]["train_s"] / optimized[kind]["train_s"],
        }
        for kind in sorted(reference)
    ]
    return {
        "name": name,
        "label": label,
        "draws_per_arm": count,
        "ratio_of_arm_medians": statistics.median(ref_values) / statistics.median(opt_values),
        "matched_timed_draws": pairs,
        "observed_matched_ratio_min": min(pair["ratio"] for pair in pairs) if count > 1 else None,
        "observed_matched_ratio_max": max(pair["ratio"] for pair in pairs) if count > 1 else None,
        "reference_train_s": ref_values,
        "optimized_train_s": opt_values,
        "gpuq_job_ids": sorted(
            {record["gpuq_job_id"] for record in list(reference.values()) + list(optimized.values())}
        ),
        "reference_runtime": next(iter(reference.values()))["runtime_build"],
        "optimized_runtime": next(iter(optimized.values()))["runtime_build"],
    }


def extract(strict, auxiliary):
    strict_jobs = {job["id"]: job for job in strict["jobs"]}
    noquant = []
    for name, label, count in NOQUANT:
        arms = {}
        for arm, runtime in (("baseline", BASELINE), ("candidate", SELECTED)):
            filename = f"strict-{name}-{arm}.jsonl"
            require(not strict["provenance_issues"][filename], f"Provenance mismatch in {filename}")
            arms[arm] = timed_records(strict["raw_files"][filename]["records"], count, runtime, strict_jobs)
        noquant.append(ratio_row(name, label, arms["baseline"], arms["candidate"], count))
    graph_jobs = {job["id"]: job for job in auxiliary["jobs"]}
    graph = []
    for name, label in GRAPH:
        group = auxiliary["graph_quant"]["groups"][name]
        require(group["headline_eligible"] is True, f"Graph cell {name} is not timing eligible")
        require(group["graph_capture_observed_in_on_warmup"] is True, "Graph capture was not observed")
        off = timed_records(group["arms"]["off"]["records"], 3, SELECTED, graph_jobs)
        on = timed_records(group["arms"]["on"]["records"], 3, SELECTED, graph_jobs)
        row = ratio_row(name, label, off, on, 3)
        require(
            math.isclose(row["ratio_of_arm_medians"], group["off_over_on"], rel_tol=1e-12),
            "Graph ratio differs from finalized report",
        )
        row["observed_all_timed_fingerprints_identical"] = group["observed_all_timed_fingerprints_identical"]
        row["on_warmup_graph_capture_observed"] = True
        graph.append(row)
    return noquant, graph


def draw_panel(ax, rows, title, color, subtitle):
    for y, row in enumerate(rows):
        center = row["ratio_of_arm_medians"]
        if row["draws_per_arm"] == 1:
            ax.plot(
                center,
                y,
                marker="D",
                markersize=7,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.8,
                linestyle="none",
                zorder=4,
            )
        else:
            low, high = row["observed_matched_ratio_min"], row["observed_matched_ratio_max"]
            ax.errorbar(
                center,
                y,
                xerr=[[center - low], [high - center]],
                fmt="o",
                color=color,
                markersize=6.5,
                capsize=4,
                elinewidth=1.6,
                capthick=1.2,
                zorder=4,
            )
        text_anchor = center if row["draws_per_arm"] == 1 else row["observed_matched_ratio_max"]
        ax.annotate(
            f"{center:.3f}×",
            (text_anchor, y),
            xytext=(9, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10,
            color=color,
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7},
        )
    ax.axvline(1.0, color="#657180", linewidth=1.1, linestyle=(0, (4, 4)), zorder=1)
    ax.set_xlim(0.56, 1.84)
    ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
    ax.set_ylim(len(rows) - 0.45, -0.65)
    ax.tick_params(axis="y", length=0, pad=9, labelsize=10)
    ax.tick_params(axis="x", labelsize=9, length=3, color="#AAB3BC")
    ax.grid(axis="x", color="#E8ECF0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#AAB3BC")
    ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=27)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#5C6670", ha="left", va="bottom")


def render(args, noquant, graph):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output.parent / ".mplconfig"))
    # Configure the writable cache and headless backend before importing pyplot.
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "falcata-strict-overview-20261009",
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.0))
    fig.subplots_adjust(left=0.17, right=0.975, top=0.785, bottom=0.285, wspace=0.72)
    draw_panel(
        axes[0],
        noquant,
        "Unquantized: baseline / selected",
        "#156292",
        "FP64 · repeated runs, except the 30,000-round pair",
    )
    draw_panel(
        axes[1], graph, "Stochastic quantized: OFF / ON", "#A75C14", "Selected build · graph_quant full-route contrast"
    )
    fig.text(
        0.035, 0.96, "Falcata strict training-time overview", fontsize=18, weight="bold", va="top", color="#17212B"
    )
    fig.text(
        0.035,
        0.9,
        "RTX 5090 · strict GPUQ admission · no recorded contention · baseline 09b6be97 / selected d0bb8c47",
        fontsize=10,
        color="#5C6670",
        va="top",
    )
    fig.text(0.565, 0.212, "Training speed ratio (higher is faster; 1.0 = equal time)", ha="center", fontsize=11)
    notes = [
        "Filled circles: ratio of arm medians. Whiskers: observed range of three matched timed-draw ratios, not confidence intervals.",
        "Hollow diamond: one complete 30,000-round run per build; no uncertainty estimate or extrapolation. Warmups are excluded.",
        "Noquant fraud/covtype are omitted for failed or unstable quality. Timing ratios alone do not establish equivalent model quality.",
        "Numerai uses the historical build1226 cache with an unidentified label. Quant graph OFF/ON does not isolate the new graph ports.",
    ]
    for y, text in zip((0.145, 0.108, 0.071, 0.034), notes, strict=True):
        fig.text(0.035, y, text, fontsize=9, color="#5C6670", va="center")
    paths = {suffix: args.output.with_suffix(suffix) for suffix in (".png", ".svg")}
    fig.savefig(
        paths[".png"],
        dpi=200,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.15,
        metadata={
            "Description": "Strict training-time contrasts; observed paired ranges are not confidence intervals."
        },
    )
    fig.savefig(
        paths[".svg"],
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.15,
        metadata={
            "Date": None,
            "Description": "Strict training-time contrasts; observed paired ranges are not confidence intervals.",
        },
    )
    plt.close(fig)
    return paths, matplotlib.__version__


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", type=Path, default=directory.parent / "strict-timing-evidence.json")
    parser.add_argument("--auxiliary", type=Path, default=directory.parent / "auxiliary-timing-evidence.json")
    parser.add_argument(
        "--output", type=Path, default=directory / "overview", help="output filename prefix, without extension"
    )
    args = parser.parse_args()
    strict = json.loads(args.strict.read_text())
    auxiliary = json.loads(args.auxiliary.read_text())
    noquant, graph = extract(strict, auxiliary)
    paths, matplotlib_version = render(args, noquant, graph)
    values = {
        "sources": {
            "strict": {"file": args.strict.name, "sha256": sha256(args.strict)},
            "auxiliary": {"file": args.auxiliary.name, "sha256": sha256(args.auxiliary)},
        },
        "plot_generator_sha256": sha256(Path(__file__)),
        "matplotlib_version": matplotlib_version,
        "python_version": sys.version.split()[0],
        "runtime_library_sha256": LIBRARIES,
        "central_statistic": "ratio of arm medians",
        "repeated_range": "min/max of three matched timed-draw ratios; not a confidence interval",
        "noquant": noquant,
        "graph_quant": graph,
    }
    values_path = args.output.parent / f"{args.output.name}-values.json"
    values_path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "png": str(paths[".png"]),
                "svg": str(paths[".svg"]),
                "values": str(values_path),
                "noquant_ratios": {row["name"]: row["ratio_of_arm_medians"] for row in noquant},
                "graph_quant_ratios": {row["name"]: row["ratio_of_arm_medians"] for row in graph},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
