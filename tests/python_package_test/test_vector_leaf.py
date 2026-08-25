# coding: utf-8
"""Contract tests for the vector-leaf V1 model and prediction plumbing."""

import ctypes
import hashlib
import re
import struct

import numpy as np
import pytest

import falcata as lgb
from falcata.basic import _LIB, _c_int_array, _safe_call


def _training_data():
    rng = np.random.default_rng(20260823)
    X = rng.normal(size=(256, 6))
    y = 1.5 * X[:, 0] - 0.75 * X[:, 2] + 0.2 * X[:, 5]
    return X, y


def _train_scalar(*, tree_mode="scalar"):
    X, y = _training_data()
    params = {
        "objective": "regression",
        "device_type": "cpu",
        "tree_mode": tree_mode,
        "force_col_wise": True,
        "deterministic": True,
        "num_threads": 1,
        "num_leaves": 7,
        "min_data_in_leaf": 5,
        "learning_rate": 0.2,
        "seed": 17,
        "verbosity": -1,
    }
    booster = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=4)
    return booster, X


def _repair_tree_sizes(model: str) -> str:
    """Rebuild the parallel-loader index after editing serialized trees."""
    tree_start = model.index("Tree=0\n")
    tree_end = model.index("end of trees\n", tree_start)
    tree_payload = model[tree_start:tree_end]
    starts = [match.start() for match in re.finditer(r"(?m)^Tree=\d+\n", tree_payload)]
    boundaries = starts[1:] + [len(tree_payload)]
    sizes = [boundary - start for start, boundary in zip(starts, boundaries, strict=True)]
    return re.sub(
        r"(?m)^tree_sizes=.*$",
        "tree_sizes=" + " ".join(map(str, sizes)),
        model,
        count=1,
    )


def _as_two_target_vector_model(model: str) -> str:
    """Widen each scalar leaf x to [x, 2*x] and repair model metadata."""
    output = []
    for line in model.splitlines():
        if line.startswith("num_class="):
            output.append("num_class=2")
        elif line.startswith("objective="):
            output.append("objective=multi_regression num_class:2")
        elif line.startswith("leaf_value="):
            values = [float(value) for value in line.removeprefix("leaf_value=").split()]
            widened = [item for value in values for item in (value, 2.0 * value)]
            output.append("leaf_value_dim=2")
            output.append("leaf_value=" + " ".join(map(repr, widened)))
        else:
            output.append(line)
    return _repair_tree_sizes("\n".join(output) + "\n")


def _vector_booster():
    scalar, X = _train_scalar()
    vector = lgb.Booster(model_str=_as_two_target_vector_model(scalar.model_to_string()))
    return scalar, vector, X


def _falb_tree_index_offset(blob: bytes) -> int:
    assert blob[:4] == b"FALB"
    num_sections = struct.unpack_from("<I", blob, 24)[0]
    section_record = struct.Struct("<IIIIQQQ")
    for section in range(num_sections):
        values = section_record.unpack_from(blob, 32 + section * section_record.size)
        if values[0] == 2:  # kSecTreeIndex
            assert values[2] == 0  # kCodecRaw
            return values[4]
    raise AssertionError("FALB tree-index section is missing")


def test_vector_leaf_t1_is_bit_identical_to_scalar_training():
    scalar, X = _train_scalar(tree_mode="scalar")
    vector, _ = _train_scalar(tree_mode="vector_leaf")

    assert vector.model_to_string() == scalar.model_to_string()
    np.testing.assert_array_equal(vector.predict(X), scalar.predict(X))


def test_scalar_model_text_has_stable_golden_hash():
    scalar, _ = _train_scalar()
    digest = hashlib.sha256(scalar.model_to_string().encode()).hexdigest()
    assert digest == "ab981c613b70aca25c4a897c172410530114111a1f40799b2fa9d6b2eb03a721"


def test_vector_leaf_training_fences_unimplemented_multi_target_path():
    X, y = _training_data()
    labels = np.column_stack((y, -y))
    with pytest.raises(lgb.basic.FalcataError, match="T=1 plumbing milestone"):
        lgb.train(
            {
                "objective": "multi_regression",
                "metric": "multi_rmse",
                "num_class": 2,
                "tree_mode": "vector_leaf",
                "device_type": "cpu",
            },
            lgb.Dataset(X, label=labels),
            num_boost_round=1,
        )


def test_vector_leaf_cuda_deterministic_mode_is_rejected_before_training():
    X, y = _training_data()
    with pytest.raises(lgb.basic.FalcataError, match="does not support deterministic=true"):
        lgb.train(
            {
                "objective": "regression",
                "tree_mode": "vector_leaf",
                "device_type": "cuda",
                "deterministic": True,
            },
            lgb.Dataset(X, label=y),
            num_boost_round=1,
        )


def test_vector_leaf_text_prediction_roundtrip_and_leaf_shape():
    scalar, vector, X = _vector_booster()
    scalar_prediction = scalar.predict(X)
    vector_prediction = vector.predict(X)

    assert vector_prediction.shape == (X.shape[0], 2)
    np.testing.assert_array_equal(vector_prediction[:, 0], scalar_prediction)
    np.testing.assert_allclose(vector_prediction[:, 1], 2.0 * scalar_prediction, rtol=1e-15, atol=0.0)

    vector_text = vector.model_to_string()
    assert vector_text.count("leaf_value_dim=2") == vector.num_trees()
    reloaded = lgb.Booster(model_str=vector_text)
    np.testing.assert_array_equal(reloaded.predict(X), vector_prediction)

    inconsistent = vector_text.replace("num_class=2", "num_class=3", 1)
    with pytest.raises(lgb.basic.FalcataError, match="metadata is inconsistent"):
        lgb.Booster(model_str=inconsistent)

    scalar_leaves = scalar.predict(X, pred_leaf=True)
    vector_leaves = vector.predict(X, pred_leaf=True)
    assert vector_leaves.shape == (X.shape[0], vector.num_trees())
    np.testing.assert_array_equal(vector_leaves, scalar_leaves)

    dumped = vector.dump_model()
    assert all(tree["leaf_value_dim"] == 2 for tree in dumped["tree_info"])

    with pytest.raises(lgb.basic.FalcataError, match="pred_contrib is not supported"):
        vector.predict(X, pred_contrib=True)


def test_vector_leaf_parallel_loader_propagates_malformed_tree_error():
    _, vector, _ = _vector_booster()
    malformed = vector.model_to_string().replace("leaf_value_dim=2\n", "", 1)
    malformed = _repair_tree_sizes(malformed)

    with pytest.raises(lgb.basic.FalcataError, match=r"strs\.size"):
        lgb.Booster(model_str=malformed)


def test_vector_leaf_merge_is_rejected_without_mutating_destination():
    scalar, vector, _ = _vector_booster()
    scalar_before = scalar.model_to_string()

    with pytest.raises(lgb.basic.FalcataError, match="mixes leaf dimensions"):
        _safe_call(_LIB.FLC_BoosterMerge(scalar._handle, vector._handle))

    assert scalar.model_to_string() == scalar_before


def test_vector_leaf_refit_is_rejected_without_mutating_model():
    _, vector, X = _vector_booster()
    vector_before = vector.model_to_string()
    leaf_predictions = vector.predict(X, pred_leaf=True)
    rows, columns = leaf_predictions.shape
    pointer, _, _ = _c_int_array(leaf_predictions.reshape(-1))

    with pytest.raises(lgb.basic.FalcataError, match="Refitting vector-leaf models is not supported"):
        _safe_call(
            _LIB.FLC_BoosterRefit(
                vector._handle,
                pointer,
                ctypes.c_int32(rows),
                ctypes.c_int32(columns),
            )
        )

    assert vector.model_to_string() == vector_before


def test_vector_leaf_init_model_merge_is_rejected():
    _, vector, X = _vector_booster()
    y = _training_data()[1]
    labels = np.column_stack((y, 2.0 * y))

    with pytest.raises(lgb.basic.FalcataError, match="metadata is inconsistent"):
        lgb.train(
            {
                "objective": "multi_regression",
                "metric": "multi_rmse",
                "num_class": 2,
                "tree_mode": "scalar",
                "device_type": "cpu",
            },
            lgb.Dataset(X, label=labels),
            num_boost_round=1,
            init_model=vector,
        )


def test_multiclass_model_cannot_reset_to_vector_leaf_mode():
    X, _ = _training_data()
    labels = np.arange(X.shape[0]) % 3
    booster = lgb.train(
        {
            "objective": "multiclass",
            "num_class": 3,
            "device_type": "cpu",
            "verbosity": -1,
        },
        lgb.Dataset(X, label=labels),
        num_boost_round=2,
    )
    prediction_before = booster.predict(X)

    with pytest.raises(lgb.basic.FalcataError, match="Cannot enable tree_mode=vector_leaf"):
        booster.reset_parameter({"tree_mode": "vector_leaf"})

    np.testing.assert_array_equal(booster.predict(X), prediction_before)


def test_multiclass_pred_leaf_shape_is_unchanged():
    X, _ = _training_data()
    labels = np.arange(X.shape[0]) % 3
    booster = lgb.train(
        {
            "objective": "multiclass",
            "num_class": 3,
            "device_type": "cpu",
            "verbosity": -1,
        },
        lgb.Dataset(X, label=labels),
        num_boost_round=2,
    )

    leaves = booster.predict(X, pred_leaf=True)
    assert leaves.shape == (X.shape[0], booster.num_trees())


def test_vector_leaf_falb_roundtrip_and_dimension_validation():
    _, vector, X = _vector_booster()
    expected = vector.predict(X)
    blob = vector.model_to_binary(
        with_stats=True,
        with_diagnostics=True,
        compress_level=0,
    )
    reloaded = lgb.Booster(model_str=vector.model_to_string()).model_from_binary(blob)
    np.testing.assert_array_equal(reloaded.predict(X), expected)
    assert "leaf_value_dim=2" in reloaded.model_to_string()

    f32_blob = vector.model_to_binary(f32_leaves=True)
    f32_reloaded = lgb.Booster(model_str=vector.model_to_string()).model_from_binary(f32_blob)
    np.testing.assert_allclose(f32_reloaded.predict(X), expected, rtol=1e-7, atol=1e-7)

    tree_index_offset = _falb_tree_index_offset(blob)
    zero_dim = bytearray(blob)
    struct.pack_into("<I", zero_dim, tree_index_offset + 12, 0)
    with pytest.raises(lgb.basic.FalcataError, match="leaf_dim=0"):
        lgb.Booster(model_str=vector.model_to_string()).model_from_binary(bytes(zero_dim))

    mixed_dim = bytearray(blob)
    tree_index_record_size = 56
    struct.pack_into("<I", mixed_dim, tree_index_offset + tree_index_record_size + 12, 3)
    with pytest.raises(lgb.basic.FalcataError, match="mixed leaf dimensions"):
        lgb.Booster(model_str=vector.model_to_string()).model_from_binary(bytes(mixed_dim))


def test_vector_leaf_skips_treelite_fil_conversion():
    _, vector, _ = _vector_booster()

    class UnexpectedTreeliteCall:
        @staticmethod
        def load_lightgbm_model(_):
            raise AssertionError("Treelite must not receive a vector-leaf model")

    assert (
        vector._build_fil_model(  # pylint: disable=protected-access
            treelite_frontend=UnexpectedTreeliteCall,
            nvforest=object(),
            start_iteration=0,
            num_iteration=-1,
            raw_score=False,
            precision="single",
        )
        is None
    )


@pytest.mark.parametrize("tree_mode", ["vector", "multi_output", "vector-leaves"])
def test_tree_mode_rejects_unknown_values(tree_mode):
    X, y = _training_data()
    with pytest.raises(lgb.basic.FalcataError, match="must be scalar or vector_leaf"):
        lgb.train(
            {"objective": "regression", "device_type": "cpu", "tree_mode": tree_mode},
            lgb.Dataset(X, label=y),
            num_boost_round=1,
        )
