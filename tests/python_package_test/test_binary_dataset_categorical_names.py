import numpy as np

import falcata as flc

# Training from the binary file that save_binary wrote, with the same
# parameters, failed when categorical_feature named its columns
# ("name:f3"): DatasetLoader::SetHeader resolved the names against a header
# the binary file does not have. The binary file carries its own categorical
# bin mappers, as it already did for an index list.


def test_binary_dataset_with_categorical_feature_names(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 5))
    X[:, 3] = rng.integers(0, 6, 300)
    y = X[:, 0] + (X[:, 3] == 2) + 0.1 * rng.standard_normal(300)
    csv = tmp_path / "train.csv"
    with open(csv, "w") as f:
        f.write(",".join(["label"] + [f"f{i}" for i in range(5)]) + "\n")
        np.savetxt(f, np.column_stack([y, X]), delimiter=",", fmt="%.10g")
    params = {
        "objective": "regression",
        "header": True,
        "label_column": "name:label",
        "categorical_feature": "name:f3",
        "deterministic": True,
        "num_threads": 1,
        "verbose": -1,
    }
    text_ds = flc.Dataset(str(csv), params=params)
    text_ds.save_binary(str(tmp_path / "train.bin"))
    from_text = flc.train(params, flc.Dataset(str(csv), params=params), 5)
    from_bin = flc.train(params, flc.Dataset(str(tmp_path / "train.bin"), params=params), 5)
    np.testing.assert_array_equal(from_text.predict(X), from_bin.predict(X))
    # f3 is still a categorical feature (a category list, not a [min:max] range) in the binary-trained model
    infos = next(line for line in from_bin.model_to_string().splitlines() if line.startswith("feature_infos="))
    assert not infos.split("=", 1)[1].split(" ")[3].startswith("[")
