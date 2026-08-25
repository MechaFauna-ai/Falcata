# coding: utf-8
"""ObliquePool: the projection pool must (a) be exactly reproducible and
persistable, and (b) actually help on a rotated decision boundary -- the
workload class it exists for."""

import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import falcata as lgb
from falcata import ObliquePool
from falcata.compat import PANDAS_INSTALLED, pd_DataFrame


def _rotated_problem(n=20_000, p=20, seed=7):
    """y depends only on x0 - x1: the worst case for axis-aligned splits,
    padded with distractor features and some NaNs."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = ((X[:, 0] - X[:, 1]) > 0).astype(np.float64)
    X[rng.random(X.shape) < 0.02] = np.nan
    return X, y


def _auc(X_tr, y_tr, X_te, y_te):
    params = {
        "objective": "binary",
        "device_type": "cpu",
        "num_leaves": 4,
        "max_depth": 3,
        "learning_rate": 0.3,
        "feature_fraction": 0.5,
        "verbosity": -1,
        "seed": 0,
        "deterministic": True,
        "num_threads": 4,
    }
    booster = lgb.train(params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=12)
    return roc_auc_score(y_te, booster.predict(X_te))


def test_pool_beats_axis_aligned_on_rotated_boundary():
    # small train set: the axis-aligned staircase around the diagonal is
    # data-starved, while one pool projection landing near x0 - x1 nails it
    X, y = _rotated_problem(n=6_000)
    X_tr, X_te = X[:4_000], X[4_000:]
    y_tr, y_te = y[:4_000], y[4_000:]

    pool = ObliquePool(num_projections=64, density=2.0, seed=3)
    auc_axis = _auc(X_tr, y_tr, X_te, y_te)
    auc_oblique = _auc(pool.fit_transform(X_tr), y_tr, pool.transform(X_te), y_te)
    assert auc_oblique > auc_axis + 0.01, (auc_oblique, auc_axis)


def test_transform_reproducible_and_persistable(tmp_path):
    X, _ = _rotated_problem(n=2_000)
    pool = ObliquePool(num_projections=16, density=3.0, seed=11)
    Z1 = pool.fit_transform(X)

    same = ObliquePool(num_projections=16, density=3.0, seed=11).fit(X)
    np.testing.assert_array_equal(Z1, same.transform(X))

    path = tmp_path / "pool.json"
    pool.save(path)
    loaded = ObliquePool.load(path)
    np.testing.assert_allclose(Z1, loaded.transform(X), rtol=1e-6)

    assert Z1.shape == (X.shape[0], X.shape[1] + 16)
    # augmented matrix must be NaN-free in the projection block
    assert np.isfinite(Z1[:, X.shape[1] :]).all()


def test_shape_mismatch_rejected():
    X, _ = _rotated_problem(n=1_000, p=10)
    pool = ObliquePool(num_projections=4, seed=0).fit(X)
    with pytest.raises(ValueError, match="fitted on 10"):
        pool.transform(X[:, :7])


@pytest.mark.skipif(not PANDAS_INSTALLED, reason="pandas is not installed")
def test_dataframe_transform_preserves_metadata_and_matches_numpy():
    X, _ = _rotated_problem(n=20, p=4)
    columns = ["temperature", "pressure", "humidity", "wind"]
    index = [f"sample-{i}" for i in range(X.shape[0])]
    frame = pd_DataFrame(X, columns=columns, index=index)

    expected = ObliquePool(num_projections=5, density=2.0, seed=7, sample_rows=8).fit_transform(X)
    pool = ObliquePool(num_projections=5, density=2.0, seed=7, sample_rows=8)
    actual = pool.fit_transform(frame)

    assert isinstance(actual, pd_DataFrame)
    assert actual.index.equals(frame.index)
    assert list(actual.columns) == columns + [f"oblique_{i}" for i in range(5)]
    assert pool.feature_names() == list(actual.columns)
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-6)


@pytest.mark.skipif(not PANDAS_INSTALLED, reason="pandas is not installed")
def test_dataframe_transform_rejects_changed_schema():
    frame = pd_DataFrame(np.arange(24, dtype=np.float32).reshape(8, 3), columns=["a", "b", "c"])
    pool = ObliquePool(num_projections=3, seed=4).fit(frame)

    with pytest.raises(ValueError, match="same order"):
        pool.transform(frame[["b", "a", "c"]])
    with pytest.raises(ValueError, match="fitted on 3"):
        pool.transform(frame[["a", "b"]])
    with pytest.raises(ValueError, match="fitted on 3"):
        pool.transform(frame.assign(d=1.0))
    renamed = frame.copy()
    renamed.columns = ["a", "renamed", "c"]
    with pytest.raises(ValueError, match="same order"):
        pool.transform(renamed)


@pytest.mark.skipif(not PANDAS_INSTALLED, reason="pandas is not installed")
@pytest.mark.parametrize(
    ("columns", "error"),
    [
        (["a", "b", "c", "d", "e", "e", "g"], "unique"),
        (["a", "b", "c", "d", "e", "f", "oblique_0"], "conflict"),
    ],
)
def test_rejected_dataframe_refit_preserves_fitted_state(columns, error):
    original = pd_DataFrame(np.arange(30, dtype=np.float32).reshape(6, 5), columns=list("abcde"))
    pool = ObliquePool(num_projections=3, seed=4).fit(original)
    expected = pool.transform(original)

    invalid = pd_DataFrame(np.arange(42, dtype=np.float32).reshape(6, 7), columns=columns)
    with pytest.raises(ValueError, match=error):
        pool.fit(invalid)

    assert pool.num_features_ == 5
    assert pool.feature_names() == list("abcde") + ["oblique_0", "oblique_1", "oblique_2"]
    np.testing.assert_array_equal(pool.transform(original).to_numpy(), expected.to_numpy())
    with pytest.raises(ValueError, match="fitted on 5"):
        pool.transform(invalid.to_numpy(), check_schema=False)


@pytest.mark.skipif(not PANDAS_INSTALLED, reason="pandas is not installed")
def test_dataframe_and_numpy_crossings_require_explicit_schema_opt_out():
    frame = pd_DataFrame(np.arange(24, dtype=np.float32).reshape(8, 3), columns=["a", "b", "c"])
    dataframe_pool = ObliquePool(num_projections=3, seed=4).fit(frame)

    with pytest.raises(ValueError, match="check_schema=False"):
        dataframe_pool.transform(frame.to_numpy())
    expected = dataframe_pool.transform(frame).to_numpy()
    actual = dataframe_pool.transform(frame.to_numpy(), check_schema=False)
    np.testing.assert_array_equal(actual, expected)

    numpy_pool = ObliquePool(num_projections=3, seed=4).fit(frame.to_numpy())
    numpy_expected = numpy_pool.transform(frame.to_numpy())
    dataframe_actual = numpy_pool.transform(frame)
    assert isinstance(dataframe_actual, pd_DataFrame)
    np.testing.assert_array_equal(dataframe_actual.to_numpy(), numpy_expected)


@pytest.mark.skipif(not PANDAS_INSTALLED, reason="pandas is not installed")
def test_dataframe_column_types_are_part_of_schema_and_output_names_are_strings(tmp_path):
    values = np.arange(24, dtype=np.float32).reshape(8, 3)
    integer_columns = pd_DataFrame(values, columns=[0, 1, 2])
    pool = ObliquePool(num_projections=2, seed=4).fit(integer_columns)
    transformed = pool.transform(integer_columns)

    assert list(transformed.columns) == ["0", "1", "2", "oblique_0", "oblique_1"]
    assert pool.feature_names() == list(transformed.columns)

    string_columns = pd_DataFrame(values, columns=["0", "1", "2"])
    with pytest.raises(ValueError, match="same order"):
        pool.transform(string_columns)

    path = tmp_path / "integer-columns.json"
    pool.save(path)
    loaded = ObliquePool.load(path)
    np.testing.assert_array_equal(loaded.transform(integer_columns).to_numpy(), transformed.to_numpy())
    with pytest.raises(ValueError, match="same order"):
        loaded.transform(string_columns)

    tuple_columns = pd_DataFrame(values, columns=[("group", 0), ("group", 1), ("other", 0)])
    tuple_pool = ObliquePool(num_projections=2, seed=5).fit(tuple_columns)
    tuple_result = tuple_pool.transform(tuple_columns)
    assert list(tuple_result.columns) == [
        "('group', 0)",
        "('group', 1)",
        "('other', 0)",
        "oblique_0",
        "oblique_1",
    ]
    assert tuple_pool.feature_names() == list(tuple_result.columns)


@pytest.mark.skipif(not PANDAS_INSTALLED, reason="pandas is not installed")
def test_nullable_float_dataframe_preserves_float64_originals():
    frame = pd_DataFrame(
        {
            "a": [1.0000000001, None, 3.25, 4.5],
            "b": [5.75, 6.1250000001, None, 8.0],
        }
    ).astype("Float64")
    expected_originals = frame.to_numpy(dtype=np.float64, na_value=np.nan)

    transformed = ObliquePool(num_projections=2, seed=8).fit_transform(frame)

    assert transformed.to_numpy().dtype == np.float64
    np.testing.assert_allclose(
        transformed.iloc[:, :2].to_numpy(),
        expected_originals,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    assert np.isfinite(transformed.iloc[:, 2:].to_numpy()).all()


@pytest.mark.skipif(not PANDAS_INSTALLED, reason="pandas is not installed")
def test_dataframe_generated_name_collisions_are_always_rejected():
    values = np.arange(24, dtype=np.float32).reshape(8, 3)
    collision = pd_DataFrame(values, columns=["a", "oblique_0", "c"])

    with pytest.raises(ValueError, match="conflict"):
        ObliquePool(num_projections=2).fit(collision)

    numpy_pool = ObliquePool(num_projections=2).fit(values)
    with pytest.raises(ValueError, match="conflict"):
        numpy_pool.transform(collision)


@pytest.mark.skipif(not PANDAS_INSTALLED, reason="pandas is not installed")
def test_dataframe_rejects_duplicate_stringified_names():
    frame = pd_DataFrame(np.arange(24, dtype=np.float32).reshape(8, 3), columns=[0, "0", "other"])
    with pytest.raises(ValueError, match="unique"):
        ObliquePool(num_projections=2).fit(frame)


@pytest.mark.skipif(not PANDAS_INSTALLED, reason="pandas is not installed")
def test_dataframe_schema_is_persisted_backward_compatibly(tmp_path):
    frame = pd_DataFrame(np.arange(24, dtype=np.float32).reshape(8, 3), columns=["a", "b", "c"])
    pool = ObliquePool(num_projections=3, seed=4).fit(frame)
    path = tmp_path / "pool.json"
    pool.save(path)

    blob = json.loads(path.read_text())
    assert blob["format_version"] == 2
    assert len(blob["feature_schema"]) == 3

    loaded = ObliquePool.load(path)
    assert loaded.feature_names() == [
        "a",
        "b",
        "c",
        "oblique_0",
        "oblique_1",
        "oblique_2",
    ]
    with pytest.raises(ValueError, match="same order"):
        loaded.transform(frame[["b", "a", "c"]])

    # Sidecars written by master before DataFrame schemas were added remain
    # loadable as version 1 and do not claim a fitted schema.
    blob["format_version"] = 1
    del blob["feature_names"]
    del blob["feature_schema"]
    path.write_text(json.dumps(blob))
    legacy = ObliquePool.load(path)
    assert legacy.feature_names() == [
        "Column_0",
        "Column_1",
        "Column_2",
        "oblique_0",
        "oblique_1",
        "oblique_2",
    ]
    assert isinstance(legacy.transform(frame), pd_DataFrame)

    # NumPy-fitted pools remain version 1 so old installs can load them.
    numpy_path = tmp_path / "numpy-pool.json"
    ObliquePool(num_projections=3, seed=4).fit(frame.to_numpy()).save(numpy_path)
    assert json.loads(numpy_path.read_text())["format_version"] == 1
    ObliquePool.load(numpy_path)

    blob["format_version"] = 3
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError, match="unsupported oblique pool format 3"):
        ObliquePool.load(path)


@pytest.mark.skipif(not PANDAS_INSTALLED, reason="pandas is not installed")
def test_version_two_sidecar_requires_complete_schema(tmp_path):
    frame = pd_DataFrame(np.arange(24, dtype=np.float32).reshape(8, 3), columns=["a", "b", "c"])
    path = tmp_path / "pool.json"
    ObliquePool(num_projections=3, seed=4).fit(frame).save(path)
    blob = json.loads(path.read_text())
    del blob["feature_schema"]
    path.write_text(json.dumps(blob))

    with pytest.raises(ValueError, match="invalid feature_schema"):
        ObliquePool.load(path)


@pytest.mark.skipif(not PANDAS_INSTALLED, reason="pandas is not installed")
def test_dataframe_rejects_non_numeric_columns():
    frame = pd_DataFrame({"numeric": [1.0, 2.0], "text": ["one", "two"]})
    with pytest.raises(ValueError, match="numeric"):
        ObliquePool(num_projections=2).fit(frame)
