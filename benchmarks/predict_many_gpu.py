r"""Benchmark for many small GPU predictions: a directory of models, each predicting its own few hundred rows.

This is the shape of a daily scoring run: thousands of model predictions of a few hundred rows each. Models are
loaded in chunks (``--chunk-bytes`` of model text at a time, so host memory stays bounded), and for each chunk the
script times, after one warm-up pass (which packs the models for the device, or converts them for FIL):

- ``native_loop``: one ``Booster.predict`` call per model (device input, default mode);
- ``native_many``: one ``falcata.predict_many`` call for the whole chunk (device input), and ``native_many_host``
  with numpy input;
- with ``--checkpoints FILE`` (JSON: model file stem -> list of iteration counts), the same with one job per
  (model, checkpoint): ``native_ckpt_loop`` and ``native_ckpt_many``;
- with ``--fil``, NVIDIA FIL in its default configuration (no ``optimize()``), one ``predict`` call per model, in
  float32 (``fil_single_loop``) and float64 (``fil_double_loop``); FIL has no iteration ranges, so checkpoints are
  native only.

Rows are drawn per model inside its threshold ranges (``predict_gpu.rows_in_ranges``). Each timing is the median of
``--reps`` passes. The batched results are checked bit for bit against the one-by-one calls, and FIL's largest
difference from the native default mode is reported. Totals over the chunks are written last.

    python benchmarks/predict_many_gpu.py --model-dir DIR --rows 500 --checkpoints DIR/checkpoints.json --fil \\
        --out results.jsonl
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_gpu import fil_model, rows_in_ranges, threshold_ranges  # noqa: E402


def median_time(fn, sync, reps):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def chunks_of(paths, chunk_bytes):
    chunk, size = [], 0
    for p in paths:
        b = p.stat().st_size
        if chunk and size + b > chunk_bytes:
            yield chunk
            chunk, size = [], 0
        chunk.append(p)
        size += b
    if chunk:
        yield chunk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--glob", default="*.txt")
    ap.add_argument("--first", type=int, default=0, help="skip the first N model files (sorted by name)")
    ap.add_argument("--count", type=int, default=0, help="use N model files (0: all)")
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--nan-frac", type=float, default=0.1)
    ap.add_argument("--checkpoints", default="")
    ap.add_argument("--chunk-bytes", type=float, default=4e9)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--fil", action="store_true")
    ap.add_argument("--fil-precisions", default="single,double")
    ap.add_argument("--inputs-dir", default="", help="rows per model as <stem>.npy (made here when missing)")
    ap.add_argument("--make-inputs", action="store_true", help="only write the inputs (CPU) and stop")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.make_inputs:
        paths = sorted(Path(args.model_dir).glob(args.glob))
        Path(args.inputs_dir).mkdir(parents=True, exist_ok=True)
        for p in paths:
            target = Path(args.inputs_dir) / f"{p.stem}.npy"
            if target.exists():
                continue
            text = p.read_text()
            nf = int(next(line for line in text.splitlines() if line.startswith("max_feature_idx=")).split("=")[1]) + 1
            np.save(target, rows_in_ranges(args.rows, *threshold_ranges(text, nf), args.nan_frac))
        return
    import cupy  # noqa: PLC0415

    import falcata as flc  # noqa: PLC0415

    sync = cupy.cuda.get_current_stream().synchronize
    paths = sorted(Path(args.model_dir).glob(args.glob))[args.first :]
    if args.count:
        paths = paths[: args.count]
    ckpts = json.loads(Path(args.checkpoints).read_text()) if args.checkpoints else {}
    out = open(args.out, "a")
    out.write(
        json.dumps(
            {
                "meta": {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "models": len(paths),
                    "rows": args.rows,
                    "variant": os.environ.get("FALCATA_DEBUG", ""),
                }
            }
        )
        + "\n"
    )
    totals = {}
    for ci, chunk in enumerate(chunks_of(paths, args.chunk_bytes)):
        rec = {
            "chunk": ci,
            "models": len(chunk),
            "first": chunk[0].stem,
            "last": chunk[-1].stem,
            "text_bytes": sum(p.stat().st_size for p in chunk),
        }
        t0 = time.perf_counter()
        texts = [p.read_text() for p in chunk]
        boosters = [flc.Booster(model_str=t) for t in texts]
        rec["load_s"] = time.perf_counter() - t0
        rec["trees"] = sum(b.num_trees() for b in boosters)
        X = []
        for p, t, b in zip(chunk, texts, boosters, strict=True):
            cached = Path(args.inputs_dir) / f"{p.stem}.npy" if args.inputs_dir else None
            if cached is not None and cached.exists():
                X.append(np.load(cached)[: args.rows])
            else:
                X.append(rows_in_ranges(args.rows, *threshold_ranges(t, b.num_feature()), args.nan_frac))
        Xd = [cupy.asarray(x) for x in X]
        # warm-up: packs every model (the time a daily run pays once per model and day)
        t0 = time.perf_counter()
        ref = [b.predict(x, raw_score=True) for b, x in zip(boosters, Xd, strict=True)]
        sync()
        rec["native_first_pass_s"] = time.perf_counter() - t0
        rec["native_loop_s"] = median_time(
            lambda bs=boosters, xs=Xd: [b.predict(x, raw_score=True) for b, x in zip(bs, xs, strict=True)],
            sync,
            args.reps,
        )
        many = flc.predict_many(boosters, Xd, raw_score=True)
        rec["native_many_matches_loop"] = all(bool(cupy.array_equal(a, b)) for a, b in zip(many, ref, strict=True))
        rec["native_many_s"] = median_time(
            lambda bs=boosters, xs=Xd: flc.predict_many(bs, xs, raw_score=True), sync, args.reps
        )
        rec["native_many_host_s"] = median_time(
            lambda bs=boosters, xs=X: flc.predict_many(bs, xs, raw_score=True), sync, args.reps
        )
        if ckpts:
            jobs_b, jobs_x, jobs_h, jobs_n = [], [], [], []
            for p, b, x, xh in zip(chunk, boosters, Xd, X, strict=True):
                for k in ckpts.get(p.stem, []):
                    jobs_b.append(b)
                    jobs_x.append(x)
                    jobs_h.append(xh)
                    jobs_n.append(int(k))
            rec["checkpoint_jobs"] = len(jobs_b)
            loop = [
                b.predict(x, num_iteration=k, raw_score=True) for b, x, k in zip(jobs_b, jobs_x, jobs_n, strict=True)
            ]
            got = flc.predict_many(jobs_b, jobs_x, num_iteration=jobs_n, raw_score=True)
            rec["native_ckpt_many_matches_loop"] = all(
                bool(cupy.array_equal(a, b)) for a, b in zip(got, loop, strict=True)
            )
            del loop, got

            def ckpt_loop(bs=jobs_b, xs=jobs_x, ks=jobs_n):
                return [b.predict(x, num_iteration=k, raw_score=True) for b, x, k in zip(bs, xs, ks, strict=True)]

            rec["native_ckpt_loop_s"] = median_time(ckpt_loop, sync, args.reps)
            rec["native_ckpt_many_s"] = median_time(
                lambda bs=jobs_b, xs=jobs_x, ks=jobs_n: flc.predict_many(bs, xs, num_iteration=ks, raw_score=True),
                sync,
                args.reps,
            )
            rec["native_ckpt_many_host_s"] = median_time(
                lambda bs=jobs_b, xs=jobs_h, ks=jobs_n: flc.predict_many(bs, xs, num_iteration=ks, raw_score=True),
                sync,
                args.reps,
            )
        ref_host = [cupy.asnumpy(r).reshape(args.rows, -1) for r in ref]
        del ref, many
        if args.fil:
            for precision in [p for p in args.fil_precisions.split(",") if p]:
                tag = f"fil_{precision}"
                try:
                    t0 = time.perf_counter()
                    models = [fil_model(t, True, precision) for t in texts]
                    rec[f"{tag}_build_s"] = time.perf_counter() - t0
                    preds = [m.predict(x) for m, x in zip(models, Xd, strict=True)]
                    sync()
                    # FIL's predict() gives class labels for multi-output models: compare single-output models only
                    diffs = [
                        float(np.nanmax(np.abs(cupy.asnumpy(p).reshape(args.rows, -1) - r)))
                        for p, r in zip(preds, ref_host, strict=True)
                        if r.shape[1] == 1
                    ]
                    rec[f"{tag}_max_abs_diff"] = max(diffs) if diffs else None
                    del preds
                    rec[f"{tag}_loop_s"] = median_time(
                        lambda ms=models, xs=Xd: [m.predict(x) for m, x in zip(ms, xs, strict=True)], sync, args.reps
                    )
                    del models
                except Exception as exc:  # recorded: one failing FIL configuration is a result
                    rec[f"{tag}_error"] = f"{type(exc).__name__}: {exc}"
                cupy.get_default_memory_pool().free_all_blocks()
        for k, v in rec.items():
            if k.endswith("_s") or k in ("models", "trees", "checkpoint_jobs", "text_bytes"):
                totals[k] = totals.get(k, 0) + v
            elif k.endswith("matches_loop"):
                totals[k] = totals.get(k, True) and v
            elif k.endswith("max_abs_diff") and v is not None:
                totals[k] = max(totals.get(k, 0.0), v)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(json.dumps(rec), flush=True)
        del boosters, texts, X, Xd, ref_host
        gc.collect()
        cupy.get_default_memory_pool().free_all_blocks()
    totals["total"] = True
    out.write(json.dumps(totals) + "\n")
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
