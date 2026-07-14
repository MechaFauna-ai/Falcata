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
    for reg in ("shallow", "deep"):
        sub = timed[timed["regime"] == reg]
        if sub.empty:
            continue
        datasets = [
            d
            for d in ["fraud", "covtype", "year", "higgs", "epsilon", "airline"]
            if d in set(sub["dataset"])
        ]
        x = np.arange(len(datasets))
        w = 0.13
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, lib in enumerate(LIBRARIES):
            vals = [
                sub[(sub.dataset == d) & (sub.library == lib)]["train_s"].median()
                for d in datasets
            ]
            ax.bar(x + (i - 2.5) * w, vals, w, label=lib, color=COLORS[lib])
        ax.set_yscale("log")
        ax.set_xticks(x, datasets)
        ax.set_ylabel("train time (s, log)")
        ax.set_title(f"GPU training time — regime {reg} (500 trees)")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, f"train_time_{reg}.png"), dpi=120)
        lines.append(f"![train time {reg}](train_time_{reg}.png)\n")

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
