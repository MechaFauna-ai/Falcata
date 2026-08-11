"""Download and preprocess benchmark datasets into ``<workspace>/data/cache/``.

Each dataset is cached as float32 ``X_train/y_train/X_test/y_test`` .npy files
(Numerai as one era-ordered float32 memmap plus metadata) so every library
trains from identical bits. Raw files are downloaded on demand with resume
support.

Run inside the competitors venv (needs sklearn, pandas, pyarrow)::

    python benchmarks/datasets.py all          # everything except numerai
    python benchmarks/datasets.py higgs airline
    NUMERAI_PARQUET=/path/to/v5_all_data.parquet python benchmarks/datasets.py numerai
    python benchmarks/datasets.py numerai-int8  # optional int8 twin (ingest_bench.py)

The Numerai parquet must contain ``feature*`` columns (int8), a ``target``
column, and a string ``era`` column, sorted by era — the "all data" training
file distributed by Numerai (v5) has exactly this layout.
"""

import json
import os
import subprocess
import sys
import zipfile

import numpy as np
from common import CACHE_DIR, DATA_DIR, SEED, dataset_ready

URLS = {
    "higgs.zip": "https://archive.ics.uci.edu/static/public/280/higgs.zip",
    "epsilon_train.bz2": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2",
    "epsilon_test.bz2": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2",
    "airline.data.bz2": "http://kt.ijs.si/elena_ikonomovska/datasets/airline_14col.data.bz2",
    "year.zip": "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip",
    "covtype.data.gz": "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
}

NUMERAI_TEST_ERAS = 200
NUMERAI_EMBARGO_ERAS = 10


def fetch(filename: str) -> str:
    """Download ``filename`` into the data dir if missing; returns its path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return path
    part = path + ".part"
    cmd = [
        "curl",
        "-SL",
        "--retry",
        "10",
        "--retry-all-errors",
        "-C",
        "-",
        "--fail",
        "-o",
        part,
        URLS[filename],
    ]
    print(f"downloading {URLS[filename]}", flush=True)
    subprocess.run(cmd, check=True)
    os.rename(part, path)
    return path


def save(name, x_tr, y_tr, x_te, y_te):
    d = os.path.join(CACHE_DIR, name)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "X_train.npy"), np.ascontiguousarray(x_tr, dtype=np.float32))
    np.save(os.path.join(d, "y_train.npy"), np.asarray(y_tr, dtype=np.float32))
    np.save(os.path.join(d, "X_test.npy"), np.ascontiguousarray(x_te, dtype=np.float32))
    np.save(os.path.join(d, "y_test.npy"), np.asarray(y_te, dtype=np.float32))
    print(f"{name}: train {x_tr.shape} test {x_te.shape} -> {d}", flush=True)


def random_split(x, y, frac=0.2, stratify=False):
    from sklearn.model_selection import train_test_split

    return train_test_split(x, y, test_size=frac, random_state=SEED, stratify=y if stratify else None)


def prep_higgs():
    import pandas as pd

    with zipfile.ZipFile(fetch("higgs.zip")) as z:
        inner = z.namelist()[0]  # HIGGS.csv.gz
        with z.open(inner) as f:
            df = pd.read_csv(
                f,
                header=None,
                dtype=np.float32,
                compression="gzip" if inner.endswith(".gz") else None,
            )
    y = df[0].to_numpy()
    x = df.drop(columns=[0]).to_numpy()
    # canonical split: the last 500K rows are the test set
    save("higgs", x[:-500_000], y[:-500_000], x[-500_000:], y[-500_000:])


def prep_epsilon():
    from sklearn.datasets import load_svmlight_file

    x_tr, y_tr = load_svmlight_file(fetch("epsilon_train.bz2"), n_features=2000)
    x_te, y_te = load_svmlight_file(fetch("epsilon_test.bz2"), n_features=2000)
    save(
        "epsilon",
        x_tr.toarray(),
        (y_tr > 0).astype(np.float32),
        x_te.toarray(),
        (y_te > 0).astype(np.float32),
    )


def _airline_split():
    """Ordinal-encoded airline, split 80/20 (shared by both airline variants)."""
    import pandas as pd

    cols = [
        "Year",
        "Month",
        "DayofMonth",
        "DayOfWeek",
        "CRSDepTime",
        "CRSArrTime",
        "UniqueCarrier",
        "FlightNum",
        "ActualElapsedTime",
        "Origin",
        "Dest",
        "Distance",
        "Diverted",
        "ArrDelay",
    ]
    cat_cols = ["UniqueCarrier", "Origin", "Dest"]
    cat_maps = {c: {} for c in cat_cols}
    chunks = []
    reader = pd.read_csv(fetch("airline.data.bz2"), header=None, names=cols, chunksize=5_000_000)
    for i, ch in enumerate(reader):
        for c in cat_cols:  # ordinal-encode string categoricals
            m = cat_maps[c]
            vals = ch[c].astype(str)
            for v in vals.unique():
                if v not in m:
                    m[v] = len(m)
            ch[c] = vals.map(m)
        y = (ch["ArrDelay"] > 0).astype(np.float32).to_numpy()
        x = ch.drop(columns=["ArrDelay"]).to_numpy(dtype=np.float32)
        chunks.append((x, y))
        print(f"airline chunk {i} ({len(ch)} rows)", flush=True)
    x = np.concatenate([c[0] for c in chunks])
    y = np.concatenate([c[1] for c in chunks])
    del chunks
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(x))
    n_test = int(0.2 * len(x))
    return (
        x[perm[n_test:]],
        y[perm[n_test:]],
        x[perm[:n_test]],
        y[perm[:n_test]],
    )


def prep_airline():
    save("airline", *_airline_split())


#: categorical columns of the airline matrix once ArrDelay is dropped:
#: UniqueCarrier (6), Origin (9), Dest (10)
AIRLINE_CAT_COLS = [6, 9, 10]
#: distinct categories kept per high-cardinality column; everything rarer
#: collapses into one bucket, so the column fits the suite-wide 255-bin budget
#: (catboost's border_count is 254) and no engine is silently given more
#: resolution than another
AIRLINE_CAT_KEEP = 254


def prep_airline_cat():
    """Airline with the three string columns left CATEGORICAL, not ordinal.

    Same rows, split and encoding as ``airline``; the difference is that
    bench.py declares columns 6/9/10 as categorical, so the engines take their
    categorical split paths instead of treating the codes as ordered numbers.

    Origin/Dest carry ~300 airports each, more than the 255-bin budget. Their
    codes are therefore recoded by descending TRAIN-set frequency and the tail
    beyond the top 254 is collapsed into a single rare bucket. Ranking on the
    train split only keeps the test split from informing the encoding.
    """
    x_tr, y_tr, x_te, y_te = _airline_split()
    for col in AIRLINE_CAT_COLS:
        codes, counts = np.unique(x_tr[:, col], return_counts=True)
        # descending frequency; ties broken by code for a deterministic map
        order = np.lexsort((codes, -counts))
        rank = np.empty(len(codes), dtype=np.int64)
        rank[order] = np.arange(len(codes))
        rare = min(len(codes), AIRLINE_CAT_KEEP)
        new_code = np.minimum(rank, rare).astype(np.float32)
        for mat in (x_tr, x_te):
            vals = mat[:, col]
            # searchsorted needs codes sorted by VALUE (np.unique guarantees it);
            # values absent from the train split are test-only categories and
            # land in the rare bucket alongside the frequency tail
            idx = np.searchsorted(codes, vals)
            np.clip(idx, 0, len(codes) - 1, out=idx)
            mat[:, col] = np.where(codes[idx] == vals, new_code[idx], np.float32(rare))
        print(
            f"airline-cat col {col}: {len(codes)} categories -> {rare + 1}",
            flush=True,
        )
    save("airline-cat", x_tr, y_tr, x_te, y_te)


def prep_covtype():
    import pandas as pd

    df = pd.read_csv(fetch("covtype.data.gz"), header=None)
    y = df[54].to_numpy(dtype=np.float32) - 1  # labels 1..7 -> 0..6
    x = df.drop(columns=[54]).to_numpy(dtype=np.float32)
    x_tr, x_te, y_tr, y_te = random_split(x, y, stratify=True)
    save("covtype", x_tr, y_tr, x_te, y_te)


def prep_year():
    import pandas as pd

    with zipfile.ZipFile(fetch("year.zip")) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, dtype=np.float32)
    y = df[0].to_numpy()
    x = df.drop(columns=[0]).to_numpy()
    # canonical split: first 463,715 train / last 51,630 test
    save("year", x[:463_715], y[:463_715], x[463_715:], y[463_715:])


def prep_fraud():
    from sklearn.datasets import fetch_openml

    ds = fetch_openml("creditcard", version=1, as_frame=False, parser="auto")
    x = ds.data.astype(np.float32)
    y = ds.target.astype(np.float32)
    x_tr, x_te, y_tr, y_te = random_split(x, y, stratify=True)
    save("fraud", x_tr, y_tr, x_te, y_te)


def _numerai_roles(f):
    """Row filter shared by the f32 and int8 numerai caches.

    Drops rows without a target, embargoes the eras before the test block,
    and splits era-ordered. Returns ``(feat_cols, keep_all, n_rows,
    train_end, era_int, targets)`` where ``keep_all`` is the absolute row
    mask over the parquet.
    """
    feat_cols = [c for c in f.schema_arrow.names if c.startswith("feature")]

    # pass 1: era + target only, to build the row filter and split boundaries
    et = f.read(columns=["era", "target"]).to_pandas()
    era_int = et["era"].astype(int).to_numpy()
    keep = et["target"].notna().to_numpy()
    if not (np.diff(era_int) >= 0).all():
        sys.exit("numerai: parquet must be sorted by era")

    kept_eras = era_int[keep]
    uniq = np.unique(kept_eras)
    test_eras = uniq[-NUMERAI_TEST_ERAS:]
    embargo_eras = uniq[-(NUMERAI_TEST_ERAS + NUMERAI_EMBARGO_ERAS) : -NUMERAI_TEST_ERAS]
    role = np.full(len(kept_eras), 1, dtype=np.int8)  # 1 train, 0 embargo (drop), 2 test
    role[np.isin(kept_eras, embargo_eras)] = 0
    role[np.isin(kept_eras, test_eras)] = 2

    keep_within = role != 0
    keep_all = keep.copy()
    keep_all[keep] = keep_within  # absolute row filter
    n_rows = int(keep_within.sum())
    train_end = int((role == 1).sum())
    targets = et["target"].to_numpy(dtype=np.float32)
    return feat_cols, keep_all, n_rows, train_end, era_int, targets


def prep_numerai():
    """Era-ordered float32 memmap; last N eras held out with an embargo gap.

    Rows without a target are dropped. Train rows are ``X[:train_end]`` and
    test rows ``X[test_start:]`` so both are zero-copy views of the memmap.
    """
    import pyarrow.parquet as pq

    src = os.environ.get("NUMERAI_PARQUET")
    if not src:
        sys.exit("numerai: set NUMERAI_PARQUET to the v5 'all data' training parquet")
    d = os.path.join(CACHE_DIR, "numerai")
    os.makedirs(d, exist_ok=True)
    f = pq.ParquetFile(src)
    feat_cols, keep_all, n_rows, train_end, era_int, tgt_all = _numerai_roles(f)
    p = len(feat_cols)
    print(f"numerai: {n_rows} rows x {p} features, train_end={train_end}", flush=True)

    x = np.memmap(os.path.join(d, "X.f32.mem"), dtype=np.float32, mode="w+", shape=(n_rows, p))
    y = np.empty(n_rows, dtype=np.float32)
    era_out = np.empty(n_rows, dtype=np.int32)

    # pass 2: stream feature batches into the memmap
    row_abs = row_out = 0
    for batch in f.iter_batches(batch_size=200_000, columns=feat_cols):
        nb = batch.num_rows
        mask = keep_all[row_abs : row_abs + nb]
        if mask.any():
            arr = batch.to_pandas().to_numpy(dtype=np.float32, na_value=np.nan)
            sel = arr[mask]
            x[row_out : row_out + len(sel)] = sel
            y[row_out : row_out + len(sel)] = tgt_all[row_abs : row_abs + nb][mask]
            era_out[row_out : row_out + len(sel)] = era_int[row_abs : row_abs + nb][mask]
            row_out += len(sel)
        row_abs += nb
    assert row_out == n_rows, (row_out, n_rows)
    x.flush()
    np.save(os.path.join(d, "y.npy"), y)
    np.save(os.path.join(d, "era_codes.npy"), era_out)
    meta = {
        "n_rows": n_rows,
        "n_features": p,
        "train_end": train_end,
        "test_start": train_end,
        "n_test_eras": NUMERAI_TEST_ERAS,
        "embargo_eras": NUMERAI_EMBARGO_ERAS,
        "source": src,
    }
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    print(f"numerai done: {n_rows} x {p}", flush=True)


def prep_numerai_int8():
    """Write the optional int8 twin of the numerai cache (``X.i8.mem``), same rows/order.

    Feeds Falcata's native int8 ingestion path (see ingest_bench.py). The
    main cross-library matrix stays float32-fed for fairness. Requires the
    f32 cache to exist; sampled rows are verified against it so the two
    caches cannot drift.
    """
    import pyarrow.parquet as pq

    d = os.path.join(CACHE_DIR, "numerai")
    meta_path = os.path.join(d, "meta.json")
    if not os.path.exists(meta_path):
        sys.exit("numerai-int8: build the numerai cache first")
    meta = json.load(open(meta_path))
    src = os.environ.get("NUMERAI_PARQUET", meta["source"])
    f = pq.ParquetFile(src)
    feat_cols, keep_all, n_rows, train_end, _, _ = _numerai_roles(f)
    if n_rows != meta["n_rows"] or train_end != meta["train_end"]:
        sys.exit("numerai-int8: row filter disagrees with the existing f32 cache")
    p = len(feat_cols)

    x = np.memmap(os.path.join(d, "X.i8.mem"), dtype=np.int8, mode="w+", shape=(n_rows, p))
    row_abs = row_out = 0
    for batch in f.iter_batches(batch_size=200_000, columns=feat_cols):
        nb = batch.num_rows
        mask = keep_all[row_abs : row_abs + nb]
        if mask.any():
            sel = batch.to_pandas().to_numpy(dtype=np.int8)[mask]
            x[row_out : row_out + len(sel)] = sel
            row_out += len(sel)
        row_abs += nb
    assert row_out == n_rows, (row_out, n_rows)
    x.flush()

    xf = np.memmap(os.path.join(d, "X.f32.mem"), dtype=np.float32, mode="r", shape=(n_rows, p))
    rng = np.random.default_rng(SEED)
    for r in rng.integers(0, n_rows, 50):
        if not (x[r].astype(np.float32) == xf[r]).all():
            sys.exit(f"numerai-int8: row {r} mismatches the f32 cache")
    print(f"numerai-int8 done: {n_rows} x {p}, sampled rows verified", flush=True)


PREPS = {
    "higgs": prep_higgs,
    "epsilon": prep_epsilon,
    "airline": prep_airline,
    "airline-cat": prep_airline_cat,
    "covtype": prep_covtype,
    "year": prep_year,
    "fraud": prep_fraud,
    "numerai": prep_numerai,
    "numerai-int8": prep_numerai_int8,
}

if __name__ == "__main__":
    names = sys.argv[1:]
    targets = [n for n in PREPS if not n.startswith("numerai")] if names == ["all"] else names
    for t in targets:
        if dataset_ready(t):
            print(f"{t}: cached, skipping", flush=True)
        else:
            PREPS[t]()
