# coding: utf-8
"""Sparse oblique projection pool: random feature interactions for Falcata.

Axis-aligned trees cannot express y ~ f(w1*x1 + w2*x2) without staircasing.
Sparse random projections ("Sparse Projection Oblique Randomer Forests",
Tomita et al., JMLR 2020; shipped in Yggdrasil DF as SPARSE_OBLIQUE splits)
give every node the chance to split on a random linear combination of a few
features instead.

This module implements the projection pool at the DATASET level rather than
per node: K sparse projections are drawn once, computed over the raw feature
matrix (GPU via cupy when the matrix already lives there), and appended as
ordinary columns before binning. Per-tree diversity then comes from
``feature_fraction`` sampling the pool. Unlike per-node oblique splits, this
needs no new split type, no model-format change and no predictor change --
a split on a projection column is a normal threshold split in z-space -- at
the price of a fixed pool instead of fresh randomness per node. The
projection definitions must be applied at predict time; ``ObliquePool``
serializes itself next to the model for that.

Usage::

    pool = ObliquePool(num_projections=256, density=3, seed=17)
    X_aug = pool.fit_transform(X)               # X plus K z-columns
    booster = falcata.train(params, falcata.Dataset(X_aug, label=y))
    ...
    preds = booster.predict(pool.transform(X_new))
    pool.save(path); pool = ObliquePool.load(path)

Numerical design choices (documented because they differ from the papers):

- Weights are continuous uniform in [-1, 1] (YDF: "consistently gives better
  quality models than binary weights").
- Each projection draws ``density`` distinct features on average (geometric
  around it, min 2): Tomita's lambda, recommended range [1, 5].
- Features are robust-scaled before projecting (median/IQR from a row
  sample), so weights are comparable across feature scales -- YDF ships
  MIN_MAX normalization in its benchmark template for the same reason.
- NaNs are imputed with the feature median inside projections only (the
  original columns keep their NaNs); propagating NaN would make a z-column
  missing whenever ANY of its terms is, which on NaN-heavy data silences
  the pool almost entirely.
"""

import json
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .compat import PANDAS_INSTALLED, pd_DataFrame

try:  # optional: only needed when the caller passes device matrices
    import cupy as _cupy
except ImportError:
    _cupy = None

__all__ = ["ObliquePool"]


def _xp(mat: Any) -> Any:
    """Return numpy or cupy, matching where the matrix lives."""
    if _cupy is not None and type(mat).__module__.startswith("cupy"):
        return _cupy
    return np


def _is_pandas_dataframe(mat: Any) -> bool:
    """Return whether mat is a pandas DataFrame without requiring pandas."""
    return PANDAS_INSTALLED and isinstance(mat, pd_DataFrame)


def _dataframe_to_numpy(frame: Any) -> np.ndarray:
    """Convert a numerical DataFrame to NumPy, including nullable dtypes."""
    bad_columns = []
    for name, pandas_dtype in zip(frame.columns, frame.dtypes, strict=True):
        # Match Falcata's supported pandas types: bool, integer, unsigned
        # integer, and floating-point (including pandas extension dtypes).
        if getattr(pandas_dtype, "kind", None) not in {"b", "i", "u", "f"}:
            bad_columns.append(str(name))
    if bad_columns:
        raise ValueError(f"pandas DataFrame columns must be numeric; invalid columns: {', '.join(bad_columns)}")

    values = frame.to_numpy(copy=False)
    if values.dtype.kind in "biuf":
        return values
    # Nullable pandas dtypes produce an object array; convert pd.NA to NaN.
    return frame.to_numpy(dtype=np.float32, copy=False, na_value=np.nan)


class ObliquePool:
    """A fixed pool of sparse random linear projections over numerical columns.

    Parameters
    ----------
    num_projections : int
        K, the number of z-columns appended.
    density : float
        Average number of features per projection (>= 2 enforced per draw).
    seed : int
        Generator seed; the pool is fully determined by (shape, params, seed).
    sample_rows : int
        Row sample used to estimate per-feature median/IQR for scaling.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        num_projections: int = 256,
        density: float = 3.0,
        seed: int = 0,
        sample_rows: int = 100_000,
    ) -> None:
        if num_projections <= 0:
            raise ValueError("num_projections must be positive")
        if density < 1:
            raise ValueError("density must be >= 1")
        self.num_projections = int(num_projections)
        self.density = float(density)
        self.seed = int(seed)
        self.sample_rows = int(sample_rows)
        # set by fit(): parallel lists, one entry per projection
        self.feature_idx_: Optional[List[np.ndarray]] = None
        self.weights_: Optional[List[np.ndarray]] = None
        self.center_: Optional[np.ndarray] = None  # per-feature median
        self.scale_: Optional[np.ndarray] = None  # per-feature robust scale
        self.num_features_: Optional[int] = None
        self.feature_names_: Optional[List[str]] = None

    # ---- fitting ----

    def fit(self, X: Any) -> "ObliquePool":
        """Draw the pool for a raw numerical feature matrix (rows x features).

        Pandas column names are retained as the fitted feature schema.
        """
        is_dataframe = _is_pandas_dataframe(X)
        matrix = _dataframe_to_numpy(X) if is_dataframe else X
        xp = _xp(matrix)
        n, p = matrix.shape
        if p < 2:
            raise ValueError("need at least 2 features to form projections")
        self.num_features_ = int(p)

        self.feature_names_ = None
        if is_dataframe:
            feature_names = [str(column) for column in X.columns]
            if len(set(feature_names)) != len(feature_names):
                raise ValueError("pandas DataFrame columns must have unique names after conversion to strings")
            generated_names = {f"oblique_{j}" for j in range(self.num_projections)}
            collisions = generated_names.intersection(feature_names)
            if collisions:
                raise ValueError(
                    f"pandas DataFrame columns conflict with generated features: {', '.join(sorted(collisions))}"
                )
            self.feature_names_ = feature_names

        rng = np.random.default_rng(self.seed)
        sample = matrix
        if n > self.sample_rows:
            take = np.sort(rng.choice(n, size=self.sample_rows, replace=False))
            sample = matrix[xp.asarray(take)] if xp is not np else matrix[take]
        sample = xp.asarray(sample, dtype=xp.float32)
        med = xp.nanmedian(sample, axis=0)
        med = xp.nan_to_num(med, nan=0.0)  # all-NaN column: center 0
        q75 = xp.nanpercentile(sample, 75, axis=0)
        q25 = xp.nanpercentile(sample, 25, axis=0)
        iqr = xp.nan_to_num(q75 - q25, nan=0.0)
        scale = xp.where(iqr > 0, iqr, 1.0)
        # host copies: the pool definition is host data regardless of device
        self.center_ = np.asarray(med.get() if xp is not np else med, dtype=np.float64)
        self.scale_ = np.asarray(scale.get() if xp is not np else scale, dtype=np.float64)

        self.feature_idx_ = []
        self.weights_ = []
        for _ in range(self.num_projections):
            # geometric around `density`, floored at 2 (1-feature projections
            # are just rescaled originals and waste a slot)
            k = max(2, int(rng.geometric(1.0 / self.density)))
            k = min(k, p)
            idx = rng.choice(p, size=k, replace=False)
            w = rng.uniform(-1.0, 1.0, size=k)
            self.feature_idx_.append(np.sort(idx).astype(np.int32))
            # keep weights aligned with the sorted index order
            order = np.argsort(idx)
            self.weights_.append(w[order].astype(np.float64))
        return self

    # ---- applying ----

    def transform(self, X: Any, dtype: Any = None) -> Any:
        """Return X with the K projection columns appended (same device).

        A pandas input produces a DataFrame with its index and column names
        preserved. If the pool was fitted on a DataFrame, transform-time
        columns must match the fitted schema in the same order.
        """
        if self.feature_idx_ is None or self.weights_ is None:
            raise RuntimeError("pool is not fitted; call fit() or load()")
        assert self.center_ is not None
        assert self.scale_ is not None
        if X.shape[1] != self.num_features_:
            raise ValueError(f"matrix has {X.shape[1]} features, pool was fitted on {self.num_features_}")
        is_dataframe = _is_pandas_dataframe(X)
        if is_dataframe and self.feature_names_ is not None:
            feature_names = [str(column) for column in X.columns]
            if feature_names != self.feature_names_:
                raise ValueError("pandas DataFrame columns must match the fitted columns in the same order")
        matrix = _dataframe_to_numpy(X) if is_dataframe else X
        xp = _xp(matrix)
        n = matrix.shape[0]
        out_dtype = dtype or (matrix.dtype if matrix.dtype in (xp.float32, xp.float64) else xp.float32)
        Z = xp.empty((n, self.num_projections), dtype=xp.float32)
        center = xp.asarray(self.center_, dtype=xp.float32)
        scale = xp.asarray(self.scale_, dtype=xp.float32)
        for j, (idx_h, w_h) in enumerate(zip(self.feature_idx_, self.weights_, strict=True)):
            idx = xp.asarray(idx_h)
            w = xp.asarray(w_h, dtype=xp.float32)
            cols = xp.asarray(matrix[:, idx], dtype=xp.float32)
            cols = xp.where(xp.isnan(cols), center[idx], cols)
            cols = (cols - center[idx]) / scale[idx]
            Z[:, j] = cols @ w
        augmented = xp.concatenate([xp.asarray(matrix, dtype=out_dtype), Z.astype(out_dtype)], axis=1)
        if is_dataframe:
            columns = list(X.columns) + [f"oblique_{j}" for j in range(self.num_projections)]
            return pd_DataFrame(augmented, index=X.index, columns=columns)
        return augmented

    def fit_transform(self, X: Any, dtype: Any = None) -> Any:
        """Fit the pool on X and return X with the K z-columns appended."""
        return self.fit(X).transform(X, dtype=dtype)

    def feature_names(self, base_names: Optional[List[str]] = None) -> List[str]:
        """Names for the augmented matrix: originals then oblique_<j>."""
        if self.num_features_ is None:
            raise RuntimeError("pool is not fitted; call fit() or load()")
        if base_names is None:
            base_names = self.feature_names_ or [f"Column_{i}" for i in range(self.num_features_)]
        return list(base_names) + [f"oblique_{j}" for j in range(self.num_projections)]

    # ---- persistence (sidecar next to the model file) ----

    def save(self, path: Any) -> None:
        """Write the pool definition as JSON (sidecar next to the model)."""
        if self.feature_idx_ is None or self.weights_ is None:
            raise RuntimeError("pool is not fitted; call fit() or load()")
        assert self.center_ is not None
        assert self.scale_ is not None
        blob = {
            "format_version": self.FORMAT_VERSION,
            "num_projections": self.num_projections,
            "density": self.density,
            "seed": self.seed,
            "num_features": self.num_features_,
            "center": self.center_.tolist(),
            "scale": self.scale_.tolist(),
            "projections": [
                {"features": idx.tolist(), "weights": w.tolist()}
                for idx, w in zip(self.feature_idx_, self.weights_, strict=True)
            ],
        }
        if self.feature_names_ is not None:
            blob["feature_names"] = self.feature_names_
        Path(path).write_text(json.dumps(blob))

    @classmethod
    def load(cls, path: Any) -> "ObliquePool":
        """Read a pool previously written by save()."""
        blob = json.loads(Path(path).read_text())
        if blob["format_version"] != cls.FORMAT_VERSION:
            raise ValueError(f"unsupported oblique pool format {blob['format_version']}")
        pool = cls(
            num_projections=blob["num_projections"],
            density=blob["density"],
            seed=blob["seed"],
        )
        pool.num_features_ = blob["num_features"]
        feature_names = blob.get("feature_names")
        if feature_names is not None:
            if (
                not isinstance(feature_names, list)
                or len(feature_names) != pool.num_features_
                or not all(isinstance(name, str) for name in feature_names)
            ):
                raise ValueError("invalid feature_names in oblique pool")
            pool.feature_names_ = feature_names
        pool.center_ = np.asarray(blob["center"], dtype=np.float64)
        pool.scale_ = np.asarray(blob["scale"], dtype=np.float64)
        pool.feature_idx_ = [np.asarray(p["features"], dtype=np.int32) for p in blob["projections"]]
        pool.weights_ = [np.asarray(p["weights"], dtype=np.float64) for p in blob["projections"]]
        return pool
