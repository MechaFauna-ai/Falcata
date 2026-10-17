"""Saved binary metadata and explicitly frozen full-scale h60 inputs."""

import hashlib
import json
import os
import struct
from pathlib import Path


def binary_dataset_params(path):
    """Read immutable bin settings from the native 64-bit binary header.

    Dataset::SerializeHeader writes 8-byte-aligned scalar fields after an
    aligned token and a native size_t header length. Dataset.get_params()
    only reflects caller-supplied arguments, so it cannot discover these
    settings before a binary has been successfully loaded. Read this bounded
    prefix rather than guessing from a filename or optional JSON sidecar.
    """
    token = b"______LightGBM_Binary_File_Token______\n"
    aligned_token_size = (len(token) + 7) // 8 * 8
    header_start = aligned_token_size + struct.calcsize("N")
    scalar_size = 10 * 8
    if struct.calcsize("N") != 8:
        raise ValueError("binary dataset gate requires the native 64-bit header layout")
    with open(path, "rb") as file:
        prefix = file.read(header_start + scalar_size)
    if prefix[: len(token)] != token:
        raise ValueError(f"{path}: not a binary dataset (missing binary token)")
    if len(prefix) != header_start + scalar_size:
        raise ValueError(f"{path}: truncated binary dataset header")
    header_size = struct.unpack_from("N", prefix, aligned_token_size)[0]
    if not scalar_size <= header_size <= Path(path).stat().st_size - header_start:
        raise ValueError(f"{path}: invalid binary dataset header size")
    params = {
        name: struct.unpack_from("=i", prefix, header_start + index * 8)[0]
        for index, name in ((4, "max_bin"), (5, "bin_construct_sample_cnt"), (6, "min_data_in_bin"))
    }
    for index, name in ((7, "use_missing"), (8, "zero_as_missing")):
        value = prefix[header_start + index * 8]
        if value not in (0, 1):
            raise ValueError(f"{path}: invalid {name} in binary dataset header")
        params[name] = bool(value)
    return params


MIN_H60_ROWS = 6_000_000
H60_FEATURES = 3555


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def checked_file(spec):
    path = Path(spec["path"])
    if not path.is_absolute() or path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
        raise ValueError(f"input content identity differs: {path}")
    return path


def binary_dataset_shape(path):
    binary_dataset_params(path)  # validate native header layout and bounds
    with Path(path).open("rb") as file:
        prefix = file.read(72)
    return struct.unpack_from("=i", prefix, 48)[0], struct.unpack_from("=i", prefix, 64)[0]


def reject_implicit_companions(path):
    # CheckCanLoadFromBin prefers <path>.bin; LoadInitialScore also consumes
    # <loaded-path>.init. Neither is part of the frozen content manifest.
    for suffix in (".bin", ".init", ".bin.init"):
        companion = Path(str(path) + suffix)
        if companion.exists() or companion.is_symlink():
            raise ValueError(f"unmanifested native companion: {companion}")


def frozen_h60():
    """Fail closed without a reviewed manifest; never discover a latest round."""
    import numpy as np
    import pyarrow.parquet as pq

    name = os.environ.get("FALCATA_H60_MANIFEST")
    expected = os.environ.get("FALCATA_H60_MANIFEST_SHA256")
    if not name or not expected or digest(name) != expected:
        raise ValueError("missing or changed pinned h60 manifest")
    manifest = json.loads(Path(name).read_text())
    if manifest["schema_version"] != 1 or manifest["target"] != "target_ender_60":
        raise ValueError("h60 manifest schema or target differs")
    if manifest["rows"] < MIN_H60_ROWS or manifest["features"] != H60_FEATURES:
        raise ValueError("h60 manifest is not the full-scale regression shape")
    reject_implicit_companions(manifest["dataset"]["path"])
    path = checked_file(manifest["dataset"])
    labels = checked_file(manifest["labels"])
    if binary_dataset_shape(path) != (manifest["rows"], manifest["features"]):
        raise ValueError("h60 binary shape differs from manifest")
    params = binary_dataset_params(path)
    if params != manifest["bin_params"]:
        raise ValueError("h60 saved bin metadata differs from manifest")
    table = pq.read_table(labels, columns=[manifest["target"], "era"])
    y = table.column(0).to_numpy().astype(np.float32)
    era = table.column(1).to_numpy()
    if y.shape != (manifest["rows"],) or not np.isfinite(y).all():
        raise ValueError("h60 label shape or finite target differs")
    if [int(era.min()), int(era.max())] != manifest["era_range"]:
        raise ValueError("h60 label era range differs")
    selection = manifest["selection"]
    if (
        selection["rule"] != "finite-label-source-order"
        or hashlib.sha256(y.tobytes()).hexdigest() != selection["label_sha256"]
    ):
        raise ValueError("h60 selected label order/content differs")
    return {"manifest": manifest, "manifest_sha256": expected, "path": path, "params": params, "y": y}


def check_loaded_h60(ds, frozen):
    import numpy as np

    manifest = frozen["manifest"]
    reject_implicit_companions(frozen["path"])
    if ds.get_init_score() is not None:
        raise ValueError("loaded h60 dataset has unmanifested initial scores")
    if (ds.num_data(), ds.num_feature()) != (manifest["rows"], manifest["features"]):
        raise ValueError("loaded h60 dataset shape differs")
    if not np.array_equal(ds.get_label(), frozen["y"]):
        raise ValueError("loaded h60 binary labels differ from frozen label order")
