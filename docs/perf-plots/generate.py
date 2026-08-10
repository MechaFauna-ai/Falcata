#!/usr/bin/env python3
"""Render the per-feature effect plots embedded in docs/performance.md.

Sources:
- the cross-library sweep produced by benchmarks/, one JSON record per run.
  Point FALCATA_SWEEP_RUNS at it; default ./results/runs.jsonl
- the leave-one-out ablation battery: benchmarks/ablation_2026-08-09.txt

Run with any python that has matplotlib:
  python docs/perf-plots/generate.py
"""

import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("FALCATA_SWEEP_RUNS", "results/runs.jsonl")
ABLATION = os.path.join(HERE, "..", "..", "benchmarks", "ablation_2026-08-09.txt")

# ---- style: one fixed color per library entity, everywhere -----------------
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
LIB_COLOR = {
    "falcata-stoch": "#2a78d6",  # blue
    "falcata-fixed": "#4a3aa7",  # violet
    "falcata-noquant": "#008300",  # green
    "falcata-stoch64": "#87b7ea",  # light blue (falcata family)
    "xgboost": "#eb6834",  # orange
    "xgboost-lossguide": "#eda100",  # yellow
    "lightgbm": "#1baf7a",  # aqua
    "catboost": "#e87ba4",  # magenta
    "lightgbm-quant": "#8f8e89",  # neutral gray (upstream auxiliary)
    "lightgbm-ocl": "#b5b4ae",  # lighter gray (upstream auxiliary)
}
LIB_LABEL = {
    "falcata-stoch": "falcata (stochastic)",
    "falcata-fixed": "falcata (fixedpoint)",
    "falcata-noquant": "falcata (no quant)",
    "falcata-stoch64": "falcata (stoch, 64 bins)",
    "xgboost": "xgboost",
    "xgboost-lossguide": "xgboost lossguide",
    "lightgbm": "lightgbm CUDA",
    "catboost": "catboost",
    "lightgbm-quant": "lightgbm CUDA quant",
    "lightgbm-ocl": "lightgbm OpenCL",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": TEXT,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT2,
        "xtick.color": TEXT2,
        "ytick.color": TEXT2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "figure.dpi": 160,
    }
)


def fail_label(status, error):
    err = (error or "").lower()
    if "out of memory" in err:
        return "OOM"
    if "quant_bins" in err and "too large" in err:
        return "guard"
    if status == "insane" or "sanity" in err:
        return "diverged"
    if status == "timeout":
        return "timeout"
    return "crash"


def load_sweep():
    rows = [json.loads(line) for line in open(RUNS)]
    med = {}  # (lib, ds, reg) -> dict of medians over timed ok runs
    fail = {}  # (lib, ds, reg) -> honest short failure label
    bykey = {}
    for r in rows:
        k = (r["library"], r["dataset"], r["regime"])
        if r["status"] in ("failed", "crashed", "insane", "timeout"):
            # first recorded failure (usually the warmup) names the cause
            if k not in fail:
                fail[k] = fail_label(r["status"], r.get("error"))
        if r["kind"].startswith("timed") and r["status"] == "ok":
            bykey.setdefault(k, []).append(r)
    for k, rs in bykey.items():
        med[k] = {
            "train_s": statistics.median(x["train_s"] for x in rs),
            "construct_s": statistics.median(x["construct_s"] for x in rs),
            "gpu_mb": statistics.median(x["gpu_mem_peak_mb"] for x in rs),
            "metrics": rs[0]["metrics"],
        }
    curves = {}
    for r in rows:
        if r["kind"] == "curve" and r["status"] == "ok" and r.get("curve"):
            curves[(r["library"], r["dataset"], r["regime"])] = r["curve"]
    return med, fail, curves


def load_ablation():
    """Parse the ablation --out JSONL: Δ% = (baseline_tps / off_tps - 1) * 100,
    i.e. 'what turning the key OFF costs' (matches the printed table)."""
    base = {}
    base_time = {}
    runs = []
    for line in open(ABLATION):
        r = json.loads(line)
        if r["key"] == "BASELINE":
            base[r["cell"]] = r["trees_per_s"]
            base_time[r["cell"]] = r.get("train_s", 0)
        else:
            runs.append(r)
    cells = {}
    for r in runs:
        b = base.get(r["cell"])
        if b and r["trees_per_s"]:
            cells.setdefault(r["cell"], {})[r["key"]] = (b / r["trees_per_s"] - 1) * 100
    return cells, base_time


def cell_noise_band(cell):
    """Sub-2s cells sit inside clock/thermal noise far beyond the +-5% of the
    slow cells (the doc's section-7 methodology note) -- widen their band."""
    return 5.0 if BASE_TIME.get(cell, 0) >= 2.0 else 12.0


def short_cell(c):
    return (
        c.replace("-quant", "")
        .replace("covtype", "covt")
        .replace("numerai-example", "num-ex")
        .replace("numerai", "num")
    )


def bar_ends(ax):
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)


MED, FAIL, CURVES = load_sweep()
LOO, BASE_TIME = load_ablation()

DS_REGS = [
    ("fraud", "deep"),
    ("covtype", "deep"),
    ("year", "deep"),
    ("higgs", "deep"),
    ("epsilon", "deep"),
    ("airline", "deep"),
    ("numerai", "numerai-deep"),
]
DS_LABEL = {("numerai", "numerai-deep"): "numerai-deep (30k)", ("numerai", "numerai"): "numerai-ex (2k)"}


# ---------------------------------------------------------------- 1. §1 scoreboard
def plot_cross_library():
    comps = ["xgboost", "lightgbm", "catboost"]
    ratios = {c: {} for c in comps}  # comp -> {ds_label: (ratio, note)}
    for dr in DS_REGS:
        ds, reg = dr
        base = MED.get(("falcata-stoch", ds, reg))
        for c in comps:
            m = MED.get((c, ds, reg))
            note = ""
            if m is None and c == "lightgbm":
                m = MED.get(("lightgbm-ocl", ds, reg))
                note = "*"
            if m and base:
                ratios[c][dr] = (m["train_s"] / base["train_s"], note)
    import math

    {c: math.exp(sum(math.log(v[0]) for v in ratios[c].values()) / len(ratios[c])) for c in comps}
    fig, ax = plt.subplots(figsize=(11.5, 3.9))
    x = range(len(DS_REGS))
    w = 0.21
    for xi in x:
        ax.bar(
            xi - 1.5 * w,
            1.0,
            width=w - 0.04,
            color=LIB_COLOR["falcata-stoch"],
            label="falcata (stochastic)" if xi == 0 else None,
        )
        ax.text(xi - 1.5 * w, 1.0, "1", ha="center", va="bottom", fontsize=7.5, color=TEXT)
    for i, c in enumerate(comps):
        labeled = False
        for xi, dr in enumerate(DS_REGS):
            xx = xi + (i - 0.5) * w
            if dr in ratios[c]:
                v, note = ratios[c][dr]
                ax.bar(xx, v, width=w - 0.04, color=LIB_COLOR[c], label=None if labeled else LIB_LABEL[c])
                labeled = True
                ax.text(xx, v, f"{v:.1f}{note}", ha="center", va="bottom", fontsize=7.5, color=TEXT)
            else:
                ax.text(xx, 1.08, "✗", ha="center", va="bottom", fontsize=9, color=TEXT2)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 30], ["1×", "2×", "5×", "10×", "30×"])
    ax.minorticks_off()
    ax.set_xticks(list(x), [DS_LABEL.get(dr, dr[0]) for dr in DS_REGS], fontsize=9)
    ax.set_ylabel("training time vs falcata (×, log)")
    ax.set_title("GPU training time relative to falcata — all libraries on CUDA, deep regimes", fontsize=10.5, pad=26)
    fig.text(
        0.5,
        -0.04,
        "cross-library sweep, one GPU, 500 trees (numerai: 30k) at matched-or-better "
        "quality  ·  * = upstream's OpenCL backend (its CUDA build OOMs)  ·  "
        "✗ = no working upstream path (diverged; OpenCL lost to a driver regression)",
        ha="center",
        fontsize=7.5,
        color=TEXT2,
    )
    ax.legend(fontsize=8.5, frameon=False, ncols=4, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    bar_ends(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "cross_library_deep.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 2. §5 quant modes
def plot_quant_modes():
    drs = [
        ("covtype", "deep"),
        ("year", "deep"),
        ("higgs", "deep"),
        ("epsilon", "deep"),
        ("airline", "deep"),
        ("numerai", "numerai-deep"),
        ("numerai", "numerai-leaf"),
    ]

    def dr_label(ds, reg):
        return reg if reg.startswith("numerai") else f"{ds}-{reg}"

    labels = [dr_label(*dr) for dr in drs]
    modes = ["falcata-fixed", "falcata-stoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    x = range(len(drs))
    w = 0.32
    for i, lib in enumerate(modes):
        sp = []
        for ds, reg in drs:
            m = MED.get((lib, ds, reg))
            b = MED.get(("falcata-noquant", ds, reg))
            sp.append(b["train_s"] / m["train_s"] if (m and b) else 0)
        xs_i = [xx + (i - 0.5) * w for xx in x]
        ax1.bar(xs_i, sp, width=w - 0.04, color=LIB_COLOR[lib], label=LIB_LABEL[lib])
        for xx, v, dr in zip(xs_i, sp, drs):
            if v == 0:
                ax1.text(
                    xx,
                    0.08,
                    "✗ " + FAIL.get((lib,) + dr, "n/a"),
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=TEXT,
                    weight="bold",
                )
    ax1.axhline(1.0, color=TEXT, linewidth=1.3, linestyle="--", label="no-quant baseline (1×)")
    ax1.set_xticks(list(x), labels, fontsize=8, rotation=20)
    ax1.set_ylabel("speedup vs no-quant (×)")
    ax1.set_title("Quantized training: speed", fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    bar_ends(ax1)
    # quality: relative metric delta vs noquant, in % (sign = better/worse)
    for i, lib in enumerate(modes):
        dq = []
        for ds, reg in drs:
            m = MED.get((lib, ds, reg))
            b = MED.get(("falcata-noquant", ds, reg))
            if not (m and b):
                dq.append(None)
                continue
            met = (
                "rmse"
                if ds == "year"
                else ("corr_mean" if ds == "numerai" else ("accuracy" if ds == "covtype" else "auc"))
            )
            mv, bv = m["metrics"][met], b["metrics"][met]
            dq.append((bv - mv) / bv * 100 if met == "rmse" else (mv - bv) / bv * 100)
        xs_i = [xx + (i - 0.5) * w for xx in x]
        ax2.bar(xs_i, [v if v is not None else 0 for v in dq], width=w - 0.04, color=LIB_COLOR[lib])
    ax2.axhline(0, color=TEXT2, linewidth=0.8)
    ax2.set_xticks(list(x), labels, fontsize=8, rotation=20)
    ax2.set_ylabel("quality vs no-quant (%)")
    ax2.set_title("Quantized training: quality delta (+ is better)", fontsize=10)
    bar_ends(ax2)
    fig.suptitle("quant_mode effect, measured on the cross-library sweep (deep regimes)", fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "quant_modes.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 3. §4 construct
def plot_construct():
    # one dataset, one point: ingestion on the largest matrix in the suite.
    # upstream lightgbm's Dataset.construct() is CPU binning regardless of
    # training backend, so its one measured number (from the OCL numerai-deep
    # cell) is simply "lightgbm" here.
    libs = ["falcata-stoch", "catboost", "lightgbm-ocl", "xgboost"]
    labels = {"lightgbm-ocl": "lightgbm", "falcata-stoch": "falcata"}
    fig, ax = plt.subplots(figsize=(8.5, 2.7))
    ys = range(len(libs))
    for y, lib in zip(ys, libs):
        m = MED.get((lib, "numerai", "numerai")) or MED.get((lib, "numerai", "numerai-deep"))
        v = m["construct_s"]
        ax.barh(y, v, height=0.6, color=LIB_COLOR["lightgbm" if lib == "lightgbm-ocl" else lib])
        note = {
            "catboost": "   (host copy only — quantization deferred into fit)",
            "lightgbm-ocl": "   (CPU binning)",
        }.get(lib, "")
        ax.text(v + 0.6, y, f"{v:.0f}s" + note, va="center", fontsize=9, color=TEXT)
    ax.set_yticks(list(ys), [labels.get(lib, LIB_LABEL[lib]) for lib in libs], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 64)
    ax.set_xlabel("time until training can start (s)")
    ax.set_title("Ingestion of the numerai matrix (6.8M rows × 3555 features, ~96 GB float32)", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "construct_time.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 4. §1 hybrid ablation
def plot_hybrid_ablation():
    feats = [
        ("hybrid", "hybrid level growth"),
        ("batch_kernels", "batched split kernels"),
        ("batch_apply", "batched apply"),
    ]
    cells = list(LOO)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2), sharey=False)
    for ax, (key, title) in zip(axes, feats):
        vals = [(short_cell(c), LOO[c].get(key, 0)) for c in cells if key in LOO[c]]
        vals.sort(key=lambda t: -t[1])
        ys = range(len(vals))
        ax.barh(list(ys), [v for _, v in vals], height=0.62, color=LIB_COLOR["falcata-stoch"])
        for y, (_c, v) in zip(ys, vals):
            ax.text(v, y, f" +{v:.0f}%" if v > 0 else f" {v:.0f}%", va="center", fontsize=7.5, color=TEXT)
        ax.set_yticks(list(ys), [c for c, _ in vals], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9.5)
        ax.margins(x=0.22)
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "Throughput cost of turning the feature OFF (leave-one-out ablation) — larger = feature worth more",
        fontsize=10,
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "hybrid_ablation.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 5. §2/§3/§4/§7b: one plot per feature
def plot_single_feature(key, title, fname):
    all_vals = [(c, short_cell(c), LOO[c].get(key, 0)) for c in LOO if key in LOO[c]]
    vals = [(sc, v) for c, sc, v in all_vals if abs(v) > cell_noise_band(c)]
    n_neutral = len(all_vals) - len(vals)
    vals.sort(key=lambda t: -t[1])
    fig, ax = plt.subplots(figsize=(7, 0.55 * max(len(vals), 2) + 1.1))
    ys = range(len(vals))
    ax.barh(list(ys), [v for _, v in vals], height=0.62, color=LIB_COLOR["falcata-stoch"])
    for y, (_c, v) in zip(ys, vals):
        ax.text(max(v, 0), y, f" {v:+.0f}%", va="center", fontsize=8.5, color=TEXT)
    ax.axvline(0, color=TEXT2, linewidth=0.8)
    ax.set_yticks(list(ys), [c for c, _ in vals], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"{title}: throughput cost of turning it OFF", fontsize=9.5)
    xlbl = "Δ throughput when disabled (%)"
    if n_neutral:
        xlbl += f"   —   free on the other {n_neutral} suite cells (within run noise, verified)"
    ax.set_xlabel(xlbl, fontsize=8.5)
    ax.margins(x=0.18)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, fname), bbox_inches="tight")
    plt.close(fig)


def plot_small_features():
    plot_single_feature("graph_loop", "CUDA-graph level loop", "ablation_graph_loop.png")
    plot_single_feature("compact_quant", "compact column view", "ablation_compact_quant.png")
    plot_single_feature("fast_rowdata", "fast rowdata build", "ablation_fast_rowdata.png")


# ---------------------------------------------------------------- 5b. §1 how hybrid growth works
def plot_hybrid_diagram():
    from matplotlib.patches import Rectangle

    BLUE = LIB_COLOR["falcata-stoch"]
    GRAY = "#b5b4ae"
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 5.6), gridspec_kw={"height_ratios": [3.2, 1]})
    (axt_l, axt_r), (axs_l, axs_r) = axes
    for ax in (axt_l, axt_r, axs_l, axs_r):
        ax.set_xlim(0, 10)
        ax.axis("off")
    axt_l.set_ylim(-1.55, 3.6)
    axt_r.set_ylim(-1.55, 3.6)

    # node positions for a depth-3 complete tree: (x, y) with root on top
    pos = {0: (5, 3)}
    for d in range(1, 3):
        nodes = range(2**d - 1, 2 ** (d + 1) - 1)
        span = 10 / (2**d + 1)
        for i, node in enumerate(nodes):
            pos[node] = (span * (i + 1), 3 - d * 1.15)
    leaves = range(7, 15)
    for i, node in enumerate(leaves):
        pos[node] = (10 / 9 * (i + 1), 3 - 3 * 1.15)

    def draw_tree(ax, order_labels):
        for child in range(1, 15):
            parent = (child - 1) // 2
            ax.plot(*zip(pos[parent], pos[child]), color=GRID, linewidth=1.2, zorder=1)
        for node in range(7):
            x, y = pos[node]
            ax.scatter([x], [y], s=430, color=BLUE, zorder=2)
            ax.text(
                x, y, order_labels[node], ha="center", va="center", fontsize=9, color="white", weight="bold", zorder=3
            )
        for node in leaves:
            x, y = pos[node]
            ax.scatter([x], [y], s=140, color=GRAY, zorder=2)

    # LEFT: classic leaf-wise -- one split at a time, order follows gain and
    # ping-pongs across the tree; every number is a full GPU<->CPU round trip
    draw_tree(axt_l, {0: "1", 1: "2", 2: "3", 3: "5", 4: "4", 5: "7", 6: "6"})
    axt_l.set_title("classic leaf-wise: one split at a time", fontsize=11)
    axt_l.text(
        5,
        -1.0,
        "7 splits × (histograms → find best → sync to CPU → apply)\n= 7 CPU⇄GPU round trips, ~28 kernel launches",
        ha="center",
        fontsize=9,
        color=TEXT2,
    )

    # RIGHT: hybrid -- level bands, one batch per level
    for d, ytop in ((0, 3), (1, 3 - 1.15), (2, 3 - 2.3)):
        axt_r.add_patch(Rectangle((0.25, ytop - 0.42), 9.5, 0.84, facecolor=BLUE, alpha=0.10, edgecolor="none"))
        axt_r.text(9.85, ytop, f"batch {d + 1}", fontsize=8.5, color=TEXT2, ha="right", va="center")
    draw_tree(axt_r, {0: "1", 1: "2", 2: "2", 3: "3", 4: "3", 5: "3", 6: "3"})
    axt_r.set_title("hybrid level-batched: whole levels at once", fontsize=11)
    axt_r.text(
        5,
        -1.0,
        "while the leaf budget cannot bind, split order doesn't change the tree\n"
        "→ 3 levels × 3 batched launches, 1 sync per level — same final tree (bit-verified)",
        ha="center",
        fontsize=9,
        color=TEXT2,
    )

    # timelines: illustrative proportions -- covtype-deep measured 11x (+1004%)
    def strip(ax, blocks, label):
        ax.set_ylim(0, 1)
        x = 0.2
        for w, busy in blocks:
            ax.add_patch(Rectangle((x, 0.3), w, 0.4, facecolor=BLUE if busy else GRAY, edgecolor="none"))
            x += w + 0.04
        ax.text(0.2, 0.92, label, fontsize=9, color=TEXT2, va="top")

    strip(
        axs_l, [(0.22, True), (1.05, False)] * 7, "GPU timeline: short kernels, long CPU-latency gaps — GPU mostly idle"
    )
    strip(
        axs_r,
        [(1.6, True), (0.28, False)] * 3,
        "GPU timeline: batched kernels back-to-back (the graph loop, §3, then removes the last gaps)",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "hybrid_growth_diagram.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 6. §9 time-to-quality
# Overrides for the line plot ONLY. Bars sit side by side with a label under
# each, so near-neighbours in hue read fine there; five overlapping lines with
# no labels do not. These two pairs were the problem: falcata-fixed's violet
# against falcata-stoch's blue, and lightgbm's aqua against falcata-noquant's
# green. The bar charts keep the shared palette.
CURVE_COLOR = {
    "falcata-fixed": "#9467bd",  # purple, clear of the blue
    "lightgbm": "#8c564b",  # brown, clear of the green
}


def plot_curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, ds, reg in ((ax1, "higgs", "deep"), (ax2, "epsilon", "deep")):
        ymin = 1.0
        for (lib, d, r), curve in CURVES.items():
            if d != ds or r != reg or lib in ("lightgbm-quant", "lightgbm-ocl", "xgboost-lossguide"):
                continue
            if len(curve) < 2:
                continue
            xs = [p[1] for p in curve]
            ys = [p[2] for p in curve]
            # clip the early ramp, but never a curve's FINAL value (an honest
            # quality dip must stay in frame)
            ymin = min(ymin, ys[len(ys) // 4], ys[-1] - 0.0008)
            ax.plot(xs, ys, color=CURVE_COLOR.get(lib, LIB_COLOR[lib]), linewidth=2, label=LIB_LABEL[lib])
        ax.set_xscale("log")
        ax.set_ylim(bottom=ymin)
        ax.set_xlabel("wall time (s, log)")
        ax.set_ylabel("test AUC")
        ax.set_title(f"{ds} deep (500 trees)", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda i: labels[i])
        ax.legend(
            [handles[i] for i in order], [labels[i] for i in order], fontsize=7.5, frameon=False, loc="upper left"
        )
    fig.suptitle("Time-to-quality (eval every 25 iters; cross-library sweep)", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "time_to_quality.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 6b. §9b FALB model size
def plot_model_size():
    # measured on the 45k-tree / 3555-feature numerai production artifact
    # (23cb1ac6, 2026-08-02); bit-exact unless noted
    rows = [
        ("model text (upstream format)", 471.6, ""),
        ("gzip -6 of the text", 148.1, ""),
        ("FALB raw (mmap-able)", 65.5, ""),
        ("FALB zlib-6  — the default", 45.6, ""),
        ("FALB zlib-6 + f32 leaves (opt-in)", 29.6, "  ~3e-08 rel. error"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 2.9))
    ys = range(len(rows))
    for y, (label, mb, note) in enumerate(rows):
        c = LIB_COLOR["falcata-stoch"] if label.startswith("FALB") else "#8f8e89"
        ax.barh(y, mb, height=0.62, color=c)
        mult = 471.6 / mb
        lbl = f" {mb:.0f} MB" + (f"  {mult:.1f}×" if mult >= 1.05 else "") + note
        ax.text(mb, y, lbl, va="center", fontsize=8.5, color=TEXT)
    ax.set_yticks(list(ys), [r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 660)
    ax.set_xlabel(
        "model file size (MB) — 45k-tree / 3555-feature numerai "
        "production model, predictions bit-identical unless noted"
    )
    ax.set_title("FALB binary model format vs the text format", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "model_size.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 7. memory
def plot_memory():
    libs = ["falcata-stoch", "lightgbm", "xgboost", "catboost"]
    drs = [("higgs", "deep"), ("epsilon", "deep"), ("airline", "deep"), ("numerai", "numerai-deep")]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    x = range(len(drs))
    w = 0.2
    for i, lib in enumerate(libs):
        for xi, (ds, reg) in enumerate(drs):
            m = MED.get((lib, ds, reg))
            note = ""
            color = LIB_COLOR[lib]
            if m is None and lib == "lightgbm":
                m = MED.get(("lightgbm-ocl", ds, reg))
                note, color = "*", LIB_COLOR["lightgbm-ocl"]
            xx = xi + (i - 1.5) * w
            if m is None:
                ax.text(xx, 0.4, "✗ OOM", rotation=90, ha="center", va="bottom", fontsize=8, color=TEXT2)
                continue
            v = m["gpu_mb"] / 1024
            ax.bar(xx, v, width=w - 0.03, color=color, label=LIB_LABEL[lib] if xi == 0 else None)
            if note:
                ax.text(xx, v, note, ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_xticks(list(x), [DS_LABEL.get(dr, dr[0]) for dr in drs], fontsize=9)
    ax.set_ylabel("GPU peak (GB)")
    ax.set_title("Peak GPU memory, deep regimes  ·  * = upstream's OpenCL backend (its CUDA build OOMs)", fontsize=10)
    # Below the axes, not floating inside them: the tallest bars (catboost on
    # airline/numerai) ran straight through a legend placed automatically.
    ax.legend(fontsize=8, frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    bar_ends(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "gpu_memory.png"), bbox_inches="tight")
    plt.close(fig)


plot_cross_library()
plot_hybrid_diagram()
plot_quant_modes()
plot_construct()
plot_hybrid_ablation()
plot_small_features()
plot_curves()
plot_model_size()
plot_memory()
print("wrote 7 plots to", HERE)
