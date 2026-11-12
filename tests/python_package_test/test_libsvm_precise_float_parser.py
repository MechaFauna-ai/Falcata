"""precise_float_parser=true also applies to LibSVM feature values.

LibSVMParser parsed the label with the configured parser but every feature value with the
fast Common::Atof, so precise_float_parser was silently ignored for LibSVM features: a value
written exactly at a split threshold could be read one ulp off and take the other branch.
"""

import numpy as np

import falcata as flc


def _thresholds(booster):
    out = []
    for tree in booster.dump_model()["tree_info"]:
        stack = [tree["tree_structure"]]
        while stack:
            node = stack.pop()
            if "threshold" in node:
                out.append((node["split_feature"], node["threshold"]))
                stack += [node["left_child"], node["right_child"]]
    return out


def test_libsvm_features_use_precise_parser(tmp_path):
    rng = np.random.default_rng(0)
    mismatches = 0
    for trial in range(6):
        X = rng.normal(size=(300, 3)) * 10.0 ** rng.integers(-20, 20)
        y = X[:, 0] + X[:, 1] * 0.5
        booster = flc.train({"objective": "regression", "num_leaves": 16, "verbose": -1}, flc.Dataset(X, y), 5)
        rows = []
        for feature, threshold in _thresholds(booster):
            for value in (threshold, np.nextafter(threshold, np.inf), np.nextafter(threshold, -np.inf)):
                row = np.abs(X[0]) + 1.0  # nonzero, so every value is written to the LibSVM line
                row[feature] = value
                rows.append(row)
        T = np.array(rows)
        path = tmp_path / f"t{trial}.svm"
        path.write_text("".join("0 " + " ".join(f"{j}:{float(v)!r}" for j, v in enumerate(r)) + "\n" for r in T))
        got = booster.predict(str(path), precise_float_parser=True)
        mismatches += int(np.sum(got != booster.predict(T)))
    assert mismatches == 0
