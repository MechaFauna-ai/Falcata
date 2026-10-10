"""Reproduce the six-state throughput chart from published endpoint evidence.

    python plot_six_state.py

The checked-in JSON contains the recorded measurements. This script renders
them without training, GPU access or any private orchestration dependency.
"""

# Standalone publication program: stdout reports output files.
# ruff: noqa: D103, T201
import argparse
import json
import os
from pathlib import Path

BUILD = "d0bb8c47789e062acbf6b770918a0589a74ae9b2"
LIBRARY = "e67a7ad314a75cec7414ecdcfaa05895040191f50c86c960c432bce919d0a90c"


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


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
        "idle GPU; no recorded contention" if data["chart_eligible"] else "endpoint eligibility retained in JSON"
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
    parser.add_argument("--data", type=Path, default=folder / "six-state-data.json")
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    outputs = render(data, folder)
    print(json.dumps({"status": data["status"], "counts": data["status_counts"], "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
