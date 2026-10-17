# coding: utf-8
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_svmlight_file

import falcata as lgb

EXAMPLES_DIR = Path(__file__).absolute().parents[2] / "examples"


class FileLoader:
    def __init__(self, directory, prefix, config_file="train.conf"):
        self.directory = directory
        self.prefix = prefix
        self.params = {"gpu_use_dp": True}
        with open(self.directory / config_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = [token.strip() for token in line.split("=")]
                    if "early_stopping" not in key:  # disable early_stopping
                        self.params[key] = value if key not in {"num_trees", "num_threads"} else int(value)

    def load_dataset(self, suffix, is_sparse=False):
        filename = str(self.path(suffix))
        if is_sparse:
            X, Y = load_svmlight_file(filename, dtype=np.float64, zero_based=True)
            return X, Y, filename
        else:
            mat = np.loadtxt(filename, dtype=np.float64)
            return mat[:, 1:], mat[:, 0], filename

    def load_field(self, suffix):
        return np.loadtxt(str(self.directory / f"{self.prefix}{suffix}"))

    def load_cpp_result(self, result_file="LightGBM_predict_result.txt"):
        return np.loadtxt(str(self.directory / result_file))

    def train_predict_check(self, lgb_train, X_test, X_test_fn, sk_pred, sk_booster=None):
        params = dict(self.params)
        params["force_row_wise"] = True
        gbm = lgb.train(params, lgb_train)
        y_pred = gbm.predict(X_test)
        cpp_pred = gbm.predict(X_test_fn)
        np.testing.assert_allclose(y_pred, cpp_pred)
        if sk_booster is None:
            np.testing.assert_allclose(y_pred, sk_pred)
        else:
            # CUDA linear fits can differ between runs: their Gram sums use
            # floating-point atomics. Compare prediction APIs on the same fit.
            np.testing.assert_allclose(sk_booster.predict(X_test), sk_pred)
            np.testing.assert_allclose(sk_booster.predict(X_test_fn), sk_pred)
            # The first tree has constant leaves, before linear fitting starts,
            # so retain independent-fit training consistency for that tree.
            np.testing.assert_allclose(
                gbm.predict(X_test, num_iteration=1), sk_booster.predict(X_test, num_iteration=1)
            )

    def file_load_check(self, lgb_train, name):
        lgb_train_f = lgb.Dataset(self.path(name), params=self.params).construct()
        for f in ("num_data", "num_feature", "get_label", "get_weight", "get_init_score", "get_group"):
            a = getattr(lgb_train, f)()
            b = getattr(lgb_train_f, f)()
            if a is None and b is None:
                pass
            elif a is None:
                assert np.all(b == 1), f
            elif isinstance(b, (list, np.ndarray)):
                np.testing.assert_allclose(a, b)
            else:
                assert a == b, f

    def path(self, suffix):
        return self.directory / f"{self.prefix}{suffix}"


def test_binary():
    fd = FileLoader(EXAMPLES_DIR / "binary_classification", "binary")
    X_train, y_train, _ = fd.load_dataset(".train")
    X_test, _, X_test_fn = fd.load_dataset(".test")
    weight_train = fd.load_field(".train.weight")
    lgb_train = lgb.Dataset(X_train, y_train, params=fd.params, weight=weight_train)
    gbm = lgb.FalcataClassifier(**fd.params)
    gbm.fit(X_train, y_train, sample_weight=weight_train)
    sk_pred = gbm.predict_proba(X_test)[:, 1]
    fd.train_predict_check(lgb_train, X_test, X_test_fn, sk_pred)
    fd.file_load_check(lgb_train, ".train")


@pytest.mark.parametrize("device_type", ["auto", "cpu"])
def test_binary_linear(device_type):
    fd = FileLoader(EXAMPLES_DIR / "binary_classification", "binary", "train_linear.conf")
    if device_type == "cpu":
        fd.params["device_type"] = "cpu"
    X_train, y_train, _ = fd.load_dataset(".train")
    X_test, _, X_test_fn = fd.load_dataset(".test")
    weight_train = fd.load_field(".train.weight")
    lgb_train = lgb.Dataset(X_train, y_train, params=fd.params, weight=weight_train)
    gbm = lgb.FalcataClassifier(**fd.params)
    gbm.fit(X_train, y_train, sample_weight=weight_train)
    sk_pred = gbm.predict_proba(X_test)[:, 1]
    # Retain independent-fit equivalence on CPU. Auto also exercises CUDA on
    # GPU hosts, where linear fitting is explicitly exempt from determinism.
    fd.train_predict_check(
        lgb_train, X_test, X_test_fn, sk_pred, sk_booster=gbm.booster_ if device_type == "auto" else None
    )
    fd.file_load_check(lgb_train, ".train")


def test_multiclass():
    fd = FileLoader(EXAMPLES_DIR / "multiclass_classification", "multiclass")
    X_train, y_train, _ = fd.load_dataset(".train")
    X_test, _, X_test_fn = fd.load_dataset(".test")
    lgb_train = lgb.Dataset(X_train, y_train)
    gbm = lgb.FalcataClassifier(**fd.params)
    gbm.fit(X_train, y_train)
    sk_pred = gbm.predict_proba(X_test)
    fd.train_predict_check(lgb_train, X_test, X_test_fn, sk_pred)
    fd.file_load_check(lgb_train, ".train")


def test_regression():
    fd = FileLoader(EXAMPLES_DIR / "regression", "regression")
    X_train, y_train, _ = fd.load_dataset(".train")
    X_test, _, X_test_fn = fd.load_dataset(".test")
    init_score_train = fd.load_field(".train.init")
    lgb_train = lgb.Dataset(X_train, y_train, init_score=init_score_train)
    gbm = lgb.FalcataRegressor(**fd.params)
    gbm.fit(X_train, y_train, init_score=init_score_train)
    sk_pred = gbm.predict(X_test)
    fd.train_predict_check(lgb_train, X_test, X_test_fn, sk_pred)
    fd.file_load_check(lgb_train, ".train")


def test_lambdarank():
    fd = FileLoader(EXAMPLES_DIR / "lambdarank", "rank")
    X_train, y_train, _ = fd.load_dataset(".train", is_sparse=True)
    X_test, _, X_test_fn = fd.load_dataset(".test", is_sparse=True)
    group_train = fd.load_field(".train.query")
    lgb_train = lgb.Dataset(X_train, y_train, group=group_train)
    params = dict(fd.params)
    params["force_col_wise"] = True
    gbm = lgb.FalcataRanker(**params)
    gbm.fit(X_train, y_train, group=group_train)
    sk_pred = gbm.predict(X_test)
    fd.train_predict_check(lgb_train, X_test, X_test_fn, sk_pred)
    fd.file_load_check(lgb_train, ".train")


def test_xendcg():
    fd = FileLoader(EXAMPLES_DIR / "xendcg", "rank")
    X_train, y_train, _ = fd.load_dataset(".train", is_sparse=True)
    X_test, _, X_test_fn = fd.load_dataset(".test", is_sparse=True)
    group_train = fd.load_field(".train.query")
    lgb_train = lgb.Dataset(X_train, y_train, group=group_train)
    gbm = lgb.FalcataRanker(**fd.params)
    gbm.fit(X_train, y_train, group=group_train)
    sk_pred = gbm.predict(X_test)
    fd.train_predict_check(lgb_train, X_test, X_test_fn, sk_pred)
    fd.file_load_check(lgb_train, ".train")
