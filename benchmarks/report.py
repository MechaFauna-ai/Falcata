"""Aggregate ``<workspace>/results/runs.jsonl`` into REPORT.md + charts."""

import json
import os
import subprocess

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import LIBRARIES, REPORT_DIR, RESULTS_DIR, RUNS_JSONL  # noqa: E402

COLORS = {
    "exaboost": "#d62728",
    "exaboost-quant": "#ff9896",
    "lightgbm": "#1f77b4",
    "lightgbm-quant": "#aec7e8",
    "xgboost": "#2ca02c",
    "catboost": "#9467bd",
}
METRIC_KEY = {
    "higgs": "auc",
    "epsilon": "auc",
    "airline": "auc",
    "fraud": "auc",
    "covtype": "accuracy",
    "year": "rmse",
    "numerai": "corr_mean",
}
NOTES = """
## Measurement notes

- `construct` covers each library's training-data structure: LightGBM `Dataset`
  (binning), XGBoost `QuantileDMatrix` (quantile sketch), CatBoost `Pool` (thin
  wrapper — CatBoost quantizes inside `fit`, i.e. inside `train`).
- CatBoost pre-reserves most of the GPU memory by default, so its "GPU peak"
  reflects the reservation, not the working set.
- Time-to-quality curves for CatBoost use a linearly approximated time axis
  (dashed) because it exposes no per-iteration wall clock.
- `xx-quant` = `use_quantized_grad=true`. Runs flagged `sane=false` produced
  degenerate models and their timings should be ignored.
"""


def fmt(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = pd.DataFrame([json.loads(line) for line in open(RUNS_JSONL)])
    timed = df[df["kind"].str.startswith("timed") & (df["status"] == "ok")].copy()
    for c in ("construct_s", "train_s", "total_s", "gpu_mem_peak_mb", "rss_peak_mb"):
        timed[c] = pd.to_numeric(timed[c], errors="coerce")

    lines = ["# GPU GBDT benchmark: ExaBoost vs LightGBM vs XGBoost vs CatBoost\n"]

    sha_file = os.path.join(RESULTS_DIR, "exaboost_sha.txt")
    sha = open(sha_file).read().strip()[:12] if os.path.exists(sha_file) else "unknown"
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    vers = (
        df[df["status"] == "ok"]
        .groupby("library")["version"]
        .agg(lambda s: s.dropna().iloc[0] if s.notna().any() else "?")
    )
    lines += [
        "## Environment\n",
        f"- GPU: {gpu}",
        f"- ExaBoost: `{sha}`",
        *[f"- {lib}: {v}" for lib, v in vers.items()],
        NOTES,
    ]

    # ---- per dataset x regime tables ------------------------------------
    for (ds, reg), g in timed.groupby(["dataset", "regime"], sort=False):
        mkey = METRIC_KEY.get(ds, "auc")
        lines.append(f"## {ds} — regime `{reg}`\n")
        lines.append(
            f"| library | construct (s) | train (s) | total (s) | {mkey} | GPU peak (MB) | RSS peak (MB) |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        base = g[g["library"] == "exaboost"]["train_s"].median()
        for lib in LIBRARIES:
            gl = g[g["library"] == lib]
            if gl.empty:
                fails = df[
                    (df.dataset == ds)
                    & (df.regime == reg)
                    & (df.library == lib)
                    & (df.status != "ok")
                ]
                note = fails["status"].iloc[0] if not fails.empty else "missing"
                lines.append(f"| {lib} | {note} | | | | | |")
                continue
            met = gl["metrics"].iloc[0] or {}
            spread = gl["train_s"].max() - gl["train_s"].min()
            rel = (
                f" ({base / gl['train_s'].median():.2f}×)"
                if lib != "exaboost" and base and gl["train_s"].median()
                else ""
            )
            flag = "" if met.get("sane", True) else " ⚠️insane"
            lines.append(
                f"| {lib} | {fmt(gl['construct_s'].median())} "
                f"| {fmt(gl['train_s'].median())} ±{fmt(spread / 2)}{rel} "
                f"| {fmt(gl['total_s'].median())} "
                f"| {fmt(met.get(mkey), 4)}{flag} "
                f"| {fmt(gl['gpu_mem_peak_mb'].median(), 0)} "
                f"| {fmt(gl['rss_peak_mb'].median(), 0)} |"
            )
        lines.append("")

    # ---- train-time bar charts ------------------------------------------
    REG_TITLES = {
        "shallow": "regime shallow (500 trees)",
        "deep": "regime deep (500 trees)",
        "numerai": "numerai example config (2000 trees)",
    }
    for reg in ("shallow", "deep", "numerai"):
        sub = timed[timed["regime"] == reg]
        if sub.empty:
            continue
        datasets = [
            d
            for d in [
                "fraud",
                "covtype",
                "year",
                "higgs",
                "epsilon",
                "airline",
                "numerai",
            ]
            if d in set(sub["dataset"])
        ]
        x = np.arange(len(datasets))
        w = 0.13
        fig, ax = plt.subplots(figsize=(11, 5))
        crash_marks = []
        for i, lib in enumerate(LIBRARIES):
            offs = x + (i - 2.5) * w
            vals, hatches = [], []
            for j, d in enumerate(datasets):
                gl = sub[(sub.dataset == d) & (sub.library == lib)]
                if gl.empty:
                    vals.append(np.nan)
                    hatches.append(None)
                    # crashed/failed configs get an explicit marker instead of
                    # silently-absent bars (e.g. upstream quantized on higgs)
                    crashed = df[
                        (df.dataset == d)
                        & (df.regime == reg)
                        & (df.library == lib)
                        & (df.status != "ok")
                    ]
                    if not crashed.empty:
                        crash_marks.append((offs[j], COLORS[lib]))
                    continue
                vals.append(gl["train_s"].median())
                met = gl["metrics"].iloc[0] or {}
                # degenerate models: sane=false, or a regression fit no better
                # than 99% of the target's std (the year quant case squeaks past
                # the sane threshold while still being useless)
                degenerate = not all((m or {}).get("sane", True) for m in gl["metrics"])
                if not degenerate and "rmse" in met and d != "numerai":
                    ok_rmses = [
                        (m or {}).get("rmse")
                        for m in sub[(sub.dataset == d) & (sub.library == "xgboost")][
                            "metrics"
                        ]
                    ]
                    if ok_rmses and ok_rmses[0] and met["rmse"] > 1.15 * ok_rmses[0]:
                        degenerate = True
                hatches.append("///" if degenerate else None)
            bars = ax.bar(offs, vals, w, label=lib, color=COLORS[lib])
            for b, h in zip(bars, hatches):
                if h:
                    b.set_hatch(h)
                    b.set_alpha(0.35)
            # single-dataset charts (numerai) have room for quality labels
            if reg == "numerai" and len(datasets) == 1 and not np.isnan(vals[0]):
                gl = sub[(sub.dataset == datasets[0]) & (sub.library == lib)]
                met = gl["metrics"].iloc[0] or {}
                if met.get("corr_mean") is not None:
                    ax.text(
                        offs[0],
                        vals[0] * 1.03,
                        f"corr {met['corr_mean']:.4f}\nsharpe {met['corr_sharpe']:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        # crash markers: x in data coords, y just above the axis baseline
        for xi, color in crash_marks:
            ax.text(
                xi,
                0.02,
                "✗",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=12,
                color=color,
                fontweight="bold",
            )
        ax.set_yscale("log")
        if reg == "numerai":
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax * 1.4)  # headroom for the quality labels
        ax.set_xticks(x, datasets)
        ax.set_ylabel("train time (s, log)")
        ax.set_title(
            f"GPU training time — {REG_TITLES[reg]}\n"
            "✗ = crashed; hatched = degenerate model (timing not meaningful)"
        )
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, f"train_time_{reg}.png"), dpi=120)
        lines.append(f"![train time {reg}](train_time_{reg}.png)\n")

    # ---- speed vs quality scatter ----------------------------------------
    panels = []
    for ds in ["fraud", "covtype", "year", "higgs", "epsilon", "airline", "numerai"]:
        for reg in ["numerai"] if ds == "numerai" else ["shallow", "deep"]:
            if not timed[(timed.dataset == ds) & (timed.regime == reg)].empty:
                panels.append((ds, reg))
    if panels:
        ncols = 3
        nrows = (len(panels) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax_i, (ds, reg) in zip(axes, panels):
            mkey = METRIC_KEY.get(ds, "auc")
            lower_better = mkey == "rmse"
            for lib in LIBRARIES:
                gl = timed[
                    (timed.dataset == ds)
                    & (timed.regime == reg)
                    & (timed.library == lib)
                ]
                if gl.empty:
                    continue
                met = gl["metrics"].iloc[0] or {}
                q = met.get(mkey)
                if q is None:
                    continue
                degenerate = not all((m or {}).get("sane", True) for m in gl["metrics"])
                ax_i.scatter(
                    [gl["train_s"].median()],
                    [q],
                    s=110 if degenerate else 80,
                    color=COLORS[lib],
                    marker="X" if degenerate else "o",
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=3,
                )
            present = set(
                timed[(timed.dataset == ds) & (timed.regime == reg)]["library"]
            )
            crashed = sorted(
                set(
                    df[(df.dataset == ds) & (df.regime == reg) & (df.status != "ok")][
                        "library"
                    ]
                )
                - present
            )
            title = f"{ds} / {reg}"
            if crashed:
                title += "\n(crashed: " + ", ".join(crashed) + ")"
            ax_i.set_xscale("log")
            ax_i.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if lower_better:
                ax_i.invert_yaxis()
            ax_i.set_title(title, fontsize=10)
            ax_i.set_xlabel("train time (s, log)", fontsize=8)
            ax_i.set_ylabel(
                mkey + (" (lower = better)" if lower_better else ""), fontsize=8
            )
            ax_i.tick_params(labelsize=7)
            ax_i.grid(True, alpha=0.25)
        for ax_i in axes[len(panels) :]:
            ax_i.axis("off")
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=COLORS[lib],
                label=lib,
                markeredgecolor="black",
                markeredgewidth=0.6,
            )
            for lib in LIBRARIES
        ]
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="X",
                linestyle="",
                color="gray",
                label="degenerate model",
                markeredgecolor="black",
            )
        )
        fig.legend(handles=handles, ncol=7, loc="lower center", fontsize=9)
        fig.suptitle("Speed vs quality — up and to the left is better", fontsize=12)
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
        fig.savefig(os.path.join(REPORT_DIR, "speed_vs_quality.png"), dpi=120)
        lines.append("![speed vs quality](speed_vs_quality.png)\n")

    # ---- time-to-quality curves -----------------------------------------
    curves = df[(df["kind"] == "curve") & (df["status"] == "ok")]
    for (ds, reg), g in curves.groupby(["dataset", "regime"], sort=False):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plotted = False
        for _, r in g.iterrows():
            pts = [p for p in (r.get("curve") or []) if p[2] is not None]
            if not pts:
                continue
            style = "--" if r.get("curve_time_approx") else "-"
            ax.plot(
                [p[1] for p in pts],
                [p[2] for p in pts],
                style,
                label=r["library"],
                color=COLORS[r["library"]],
            )
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("wall time (s)")
        ax.set_ylabel("held-out metric")
        ax.set_title(f"time-to-quality — {ds} / {reg}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        name = f"curve_{ds}_{reg}.png"
        fig.savefig(os.path.join(REPORT_DIR, name), dpi=120)
        lines.append(f"![curve {ds} {reg}]({name})\n")

    fails = df[~df["status"].isin(["ok"])]
    if not fails.empty:
        lines.append("## Failed / skipped runs\n")
        for _, r in fails.iterrows():
            err_raw = r.get("error")
            err = err_raw.strip().splitlines() if isinstance(err_raw, str) else []
            lines.append(
                f"- {r['library']}/{r['dataset']}/{r['regime']}/{r['kind']}: {r['status']}"
                + (f" — `{err[-1][:160]}`" if err else "")
            )
        lines.append("")

    with open(os.path.join(REPORT_DIR, "REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {REPORT_DIR}/REPORT.md")


if __name__ == "__main__":
    main()
