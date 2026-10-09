"""Reproduce the release article's mechanism schematics, with no GPU access.

    python docs/blog/falcata-1.1/plot_mechanisms.py

Facts come from docs/performance.md, sections 10c, 10d and 10e:
- pair_code5 / pair_code5_words: joint cells, six 5-bit codes per word,
  one row-index and packed-gradient load shared across six pairs;
- gradients_no_sync / tree_meta_batch: stream ordering and batched setup;
- apply_struct_fused / gap_copy_fused: disjoint work shares a launch;
- readback_fused: producers write mapped host staging; host reads still wait.

These are explanatory diagrams, not measured timelines or speed estimates.
The example bins, codes and boxes are illustrative; packing is calculated.
"""

# Standalone plotting program: stdout reports its output files.
# ruff: noqa: D103, T201
import argparse
import json
import os
from pathlib import Path

INK = "#1A2C37"
BLUE = "#216A8B"
MUTED = "#586774"
PALE = "#EDF3F6"
LINE = "#D8E3E8"
GRAY = "#F5F7F8"


def text(ax, x, y, value, size=11, color=INK, weight="normal", ha="left", va="center"):
    ax.text(x, y, value, fontsize=size, color=color, weight=weight, ha=ha, va=va, linespacing=1.25)


def box(ax, x, y, width, height, label="", fill=PALE, edge=LINE, size=10, weight="normal"):
    from matplotlib.patches import FancyBboxPatch  # noqa: PLC0415

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.45",
        linewidth=1.05,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    if label:
        text(ax, x + width / 2, y + height / 2, label, size=size, weight=weight, ha="center")
    return patch


def arrow(ax, start, end, color=BLUE, width=1.5):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": width, "mutation_scale": 12},
    )


def line(ax, points, color=LINE, width=1.1):
    ax.plot([point[0] for point in points], [point[1] for point in points], color=color, lw=width)


def canvas(plt, height):
    fig, ax = plt.subplots(figsize=(10.4, height), facecolor="white")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.99, bottom=0.01)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def card(ax, y, height):
    box(ax, 3, y, 94, height, fill="white", edge=LINE)


def save(plt, fig, output):
    png = output.with_suffix(".png")
    svg = output.with_suffix(".svg")
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.12, metadata={"Date": None})
    plt.close(fig)
    return [str(png), str(svg)]


def joint_histogram(plt, folder):
    from matplotlib.patches import Rectangle  # noqa: PLC0415

    fig, ax = canvas(plt, 8.7)
    text(ax, 3, 96, "Read once, update six feature pairs", size=23, weight="bold")
    text(ax, 3, 91.5, "Quantized histogram construction on eligible compact rows", size=12, color=MUTED)

    card(ax, 57, 29)
    text(ax, 7, 83, "1  Pair two feature bins", size=13, weight="bold")
    box(ax, 7, 70, 15, 7, "A = 2", fill=PALE, edge=BLUE, size=14, weight="bold")
    box(ax, 7, 61, 15, 7, "B = 3", fill=PALE, edge=BLUE, size=14, weight="bold")
    arrow(ax, (24, 69), (30, 69))
    grid_x, grid_y, cell = 34, 60, 3.8
    for row in range(5):
        for column in range(5):
            index = row * 5 + column
            x = grid_x + column * cell
            y = grid_y + (4 - row) * cell
            selected = index == 13
            ax.add_patch(
                Rectangle((x, y), cell, cell, facecolor=BLUE if selected else PALE, edgecolor="white", linewidth=1.4)
            )
            text(ax, x + cell / 2, y + cell / 2, str(index), size=9, color="white" if selected else MUTED, ha="center")
        text(ax, grid_x - 1.1, grid_y + (4 - row + 0.5) * cell, f"A{row}", size=9, color=MUTED, ha="right")
    for column in range(5):
        text(ax, grid_x + (column + 0.5) * cell, 80.8, f"B{column}", size=9, color=MUTED, ha="center")
    text(ax, grid_x + cell * 2.5, 58.5, "5 × 5 = 25 joint cells", size=10, color=MUTED, ha="center")
    arrow(ax, (56, 69), (65, 69))
    box(ax, 68, 64, 22, 12, "13  =  01101₂", fill=PALE, edge=BLUE, size=15, weight="bold")
    text(ax, 79, 79.5, "One 5-bit cell index", size=11, color=BLUE, weight="bold", ha="center")
    text(ax, 79, 61, "index = A × 5 + B", size=11, color=MUTED, ha="center")

    card(ax, 35, 18)
    text(ax, 7, 50.4, "2  Pack six pair-codes into one 32-bit word", size=13, weight="bold")
    codes = (13, 7, 21, 0, 18, 4)
    packed = sum(code << (5 * slot) for slot, code in enumerate(codes))
    assert packed < 2**30
    assert tuple((packed >> (5 * slot)) & 31 for slot in range(6)) == codes
    bit_width = 86 / 32
    for slot, code in enumerate(codes):
        x = 7 + slot * 5 * bit_width
        ax.add_patch(Rectangle((x, 40.2), 5 * bit_width, 5.6, facecolor=PALE, edgecolor="white", linewidth=2))
        text(ax, x + 2.5 * bit_width, 43, f"{code:05b}", size=12, color=BLUE, weight="bold", ha="center")
        text(ax, x + 2.5 * bit_width, 47.1, "5 bits", size=9, color=MUTED, ha="center")
        text(ax, x + 2.5 * bit_width, 38.1, f"pair {slot + 1}", size=9, color=MUTED, ha="center")
    x = 7 + 30 * bit_width
    ax.add_patch(Rectangle((x, 40.2), 2 * bit_width, 5.6, facecolor=LINE, edgecolor="white", linewidth=2))
    text(ax, x + bit_width, 43, "00", size=11, color=MUTED, ha="center")
    text(ax, x + bit_width, 47.1, "2 bits", size=9, color=MUTED, ha="center")
    text(ax, x + bit_width, 38.1, "spare", size=9, color=MUTED, ha="center")

    card(ax, 7, 24)
    text(ax, 7, 28.2, "3  Share the loads, keep the six updates", size=13, weight="bold")
    text(ax, 7, 24.4, "Before: one processing unit per byte", size=11, weight="bold")
    text(ax, 54, 24.4, "After: one unit per code word", size=11, weight="bold", color=BLUE)
    for slot in range(6):
        x = 7 + slot * 6.65
        box(ax, x, 15.3, 6.0, 6.4, "index\nG/H\nbyte", size=9)
        text(ax, x + 3, 13.6, f"pair {slot + 1}", size=8.5, color=MUTED, ha="center")
    text(ax, 7, 10.4, "Six index loads + six packed-gradient loads", size=10, color=MUTED)
    for x, label in ((54, "row\nindex"), (68, "packed\nG/H"), (82, "32-bit\nword")):
        box(ax, x, 17.1, 11, 5.3, label, edge=BLUE, size=10)
        line(ax, ((x + 5.5, 17.1), (x + 5.5, 15.1)), color=BLUE)
    line(ax, ((56.7, 15.1), (91.5, 15.1)), color=BLUE)
    for slot in range(6):
        x = 54 + slot * 6.75
        box(ax, x, 10.6, 5.6, 2.8, f"pair {slot + 1}", edge=BLUE, size=8.5)
        arrow(ax, (x + 2.8, 15.1), (x + 2.8, 13.5), width=1.1)
    text(ax, 54, 8.7, "One index + one packed G/H serve all six pairs", size=9.5, color=MUTED)

    text(
        ax,
        3,
        4.1,
        "G/H = quantized gradient + curvature. The compact view requires at most 32 joint cells per pair.",
        size=9.5,
        color=MUTED,
    )
    text(
        ax,
        3,
        1.6,
        "Same integer contributions and marginal histograms. Diagram schematic, not a timing profile.",
        size=9.5,
        color=MUTED,
    )
    return save(plt, fig, folder / "joint-histogram-codewords")


def coordination(plt, folder):
    fig, ax = canvas(plt, 9.5)
    text(ax, 3, 96, "Less coordination between CPU and GPU", size=23, weight="bold")
    text(
        ax, 3, 92, "Queue dependent work, share setup, and let kernels publish their own results.", size=12, color=MUTED
    )
    text(ax, 7, 87, "BEFORE", size=11, color=MUTED, weight="bold")
    text(ax, 55, 87, "AFTER", size=11, color=BLUE, weight="bold")

    card(ax, 66, 17)
    text(ax, 7, 80.6, "1  Stream ordering", size=13, weight="bold")
    text(ax, 7, 76.8, "Extra waits between dependent GPU work", size=10.5, color=MUTED)
    text(ax, 55, 76.8, "Dependencies keep GPU work in order", size=10.5, color=MUTED)
    for x, label in ((7, "gradient\ncalc."), (23, "compact\nfill"), (39, "histogram\nbuild")):
        box(ax, x, 69, 9, 5.5, label, size=9.5)
    for x in (17, 33):
        box(ax, x, 69.8, 5, 4, "CPU\nwait", fill=GRAY, edge=MUTED, size=8)
        arrow(ax, (x - 0.8, 71.8), (x, 71.8), width=1)
        arrow(ax, (x + 5, 71.8), (x + 5.9, 71.8), width=1)
    for x, label in ((55, "gradient\ncalc."), (69, "compact\nfill"), (83, "histogram\nbuild")):
        box(ax, x, 69, 11, 5.5, label, edge=BLUE, size=10)
    arrow(ax, (66, 71.8), (69, 71.8))
    arrow(ax, (80, 71.8), (83, 71.8))
    text(ax, 55, 67.4, "Required host reads still synchronize.", size=9.5, color=MUTED)

    card(ax, 46, 17)
    text(ax, 7, 60.6, "2  Batched metadata", size=13, weight="bold")
    text(ax, 7, 56.8, "Small setup uploads at tree start", size=10.5, color=MUTED)
    text(ax, 55, 56.8, "One packed upload + GPU scatter", size=10.5, color=MUTED)
    box(ax, 7, 48.4, 16, 5.7, "metadata\n1 … N", fill="white", size=10)
    for y in (49.4, 51.2, 53.0):
        arrow(ax, (24, y), (35.7, y), width=1)
    box(ax, 36, 48.4, 12, 5.7, "GPU setup", size=10)
    box(ax, 55, 48.4, 16, 5.7, "metadata\nbundle", fill="white", edge=BLUE, size=10)
    arrow(ax, (71, 51.2), (77, 51.2))
    box(ax, 77, 48.4, 17, 5.7, "scatter to\ndestinations", edge=BLUE, size=10)
    text(ax, 7, 47.1, "Separate CPU submissions", size=9.5, color=MUTED)
    text(ax, 55, 47.1, "The same metadata, fewer submissions.", size=9.5, color=MUTED)

    card(ax, 26, 17)
    text(ax, 7, 40.6, "3  Launch fusion", size=13, weight="bold")
    text(ax, 7, 36.8, "Separate GPU launches", size=10.5, color=MUTED)
    text(ax, 55, 36.8, "One launch; each role writes its own data", size=10.5, color=MUTED)
    for x, label in ((7, "partition\nrows"), (21.5, "update\ntree"), (36, "copy\ngaps")):
        box(ax, x, 28.8, 12, 5.5, label, size=10)
    arrow(ax, (19, 31.5), (21.5, 31.5))
    arrow(ax, (33.5, 31.5), (36, 31.5))
    box(ax, 55, 28.8, 39, 5.5, "partition rows + tree update + gap copy", edge=BLUE, size=10.5)
    text(ax, 55, 27.2, "Different block IDs carry the additional roles.", size=9.5, color=MUTED)

    card(ax, 6, 17)
    text(ax, 7, 20.6, "4  Producer readback", size=13, weight="bold")
    text(ax, 7, 16.8, "Produce the result, then launch a copy", size=10.5, color=MUTED)
    text(ax, 55, 16.8, "The producer also writes host staging", size=10.5, color=MUTED)
    box(ax, 7, 8.7, 12, 5.6, "GPU\nproducer", size=10)
    box(ax, 23, 8.7, 9, 5.6, "copy\nkernel", size=9.5)
    box(ax, 36, 8.7, 12, 5.6, "host\nwait/read", fill="white", edge=MUTED, size=10)
    arrow(ax, (19, 11.5), (23, 11.5))
    arrow(ax, (32, 11.5), (36, 11.5))
    box(ax, 55, 8.7, 22, 5.6, "producer writes\nmapped staging", edge=BLUE, size=10.5)
    box(ax, 82, 8.7, 12, 5.6, "host\nwait/read", fill="white", edge=MUTED, size=10)
    arrow(ax, (77, 11.5), (82, 11.5))
    text(ax, 55, 7.2, "Same result bytes; reads wait for completion.", size=9.5, color=MUTED)

    text(
        ax,
        3,
        3.2,
        "Blue fill: GPU work. White/gray fill: CPU operations. Arrows: dependencies, not durations.",
        size=9.5,
        color=MUTED,
    )
    text(
        ax,
        3,
        0.8,
        "These optimizations run on guarded eligible paths. Diagram schematic, not a timing profile.",
        size=9.5,
        color=MUTED,
    )
    return save(plt, fig, folder / "gpu-coordination")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).absolute().parent)
    args = parser.parse_args()
    folder = args.output_dir.absolute()
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(folder / ".mplconfig"))
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "falcata-1.1-mechanisms",
        }
    )
    outputs = joint_histogram(plt, folder) + coordination(plt, folder)
    print(json.dumps({"outputs": outputs, "type": "mechanism schematic; not a timing profile"}, indent=2))


if __name__ == "__main__":
    main()
