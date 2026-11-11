"""Binary category metadata reads skip stored binary section payloads."""

import json
import struct

import pytest

from falcata import basic


@pytest.mark.parametrize("categories", [None, [["a", "b"]], [list(range(10000))]])
def test_binary_category_trailer_read_is_bounded(tmp_path, monkeypatch, categories):
    payload_size = 4 * 1024 * 1024
    path = tmp_path / "container.falb"
    header = struct.pack("<4sIQQII", b"FALB", 1, 0, 1, 1, 0)
    entry = struct.pack("<IIIIQQQ", 1, 0, 0, 0, 72, payload_size, payload_size)
    trailer = b"" if categories is None else b"\npandas_categorical:" + json.dumps(categories).encode() + b"\n"
    path.write_bytes(header + entry + bytes(payload_size) + trailer)
    consumed = []
    original = open

    class CountingFile:
        def __init__(self, *args, **kwargs):
            self.file = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.file, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def read(self, *args):
            data = self.file.read(*args)
            consumed.append(len(data))
            return data

        def readlines(self):
            lines = self.file.readlines()
            consumed.append(sum(map(len, lines)))
            return lines

    monkeypatch.setattr(basic, "open", CountingFile, raising=False)
    assert basic._load_pandas_categorical(file_name=path) == categories
    assert sum(consumed) <= 72 + len(trailer)


def test_native_binary_keeps_pandas_categories(tmp_path):
    pd = pytest.importorskip("pandas")
    import falcata as flc  # noqa: PLC0415

    x = pd.DataFrame({"category": pd.Series(["a", "b", "c"] * 20, dtype="category"), "value": list(range(60))})
    bst = flc.train(
        {"objective": "regression", "device_type": "cpu", "num_threads": 2, "min_data_in_leaf": 1, "verbosity": -1},
        flc.Dataset(x, label=list(range(60))),
        num_boost_round=2,
    )
    path = tmp_path / "model.falb"
    bst.save_model(path, format="falb")
    loaded = flc.Booster(model_file=path)
    assert loaded.pandas_categorical == bst.pandas_categorical
    assert (loaded.predict(x) == bst.predict(x)).all()
