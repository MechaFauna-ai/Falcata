"""Dumping or exporting a valid but deep tree must not overflow the stack.

The JSON dump (Tree::NodeToJSON), the C++ if-else export (NodeToIfElse, NodeToIfElseByMap) and the leaf-depth walk
(RecomputeLeafDepths) recursed once per tree level, the dump with a std::stringstream in every frame: a structurally
valid chain of 10,000 nodes -- it passes every load check -- crashed dump_model with SIGSEGV on the default 8 MiB
stack. They walk with an explicit stack now and write the same bytes.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import falcata as flc

_DEPTH = 20000

_CHILD = textwrap.dedent(
    """
    import ctypes, sys
    import numpy as np
    import falcata as flc
    from falcata.basic import _LIB, _safe_call

    booster = flc.Booster(model_file=sys.argv[1])
    n = ctypes.c_int64(0)
    size = 1 << 26
    buf = ctypes.create_string_buffer(size)
    _safe_call(_LIB.FLC_BoosterDumpModel(booster._handle, ctypes.c_int(0), ctypes.c_int(-1), ctypes.c_int(0),
                                         ctypes.c_int64(size), ctypes.byref(n), ctypes.c_char_p(ctypes.addressof(buf))))
    text = buf.value.decode()
    assert n.value <= size
    print("DUMP", text.count('"leaf_index"'), text.count("{"), text.count("}"))
    x = np.array([[123.5], [5.0], [1e9]])
    print("CONTRIB", booster.predict(x, pred_contrib=True).shape)
    """
)


def _chain_model(path, depth):
    x = np.arange(200, dtype=float).reshape(-1, 1)
    model = flc.train({"verbose": -1, "num_leaves": 2, "min_data_in_leaf": 1}, flc.Dataset(x, x[:, 0]), 1)
    head, rest = model.model_to_string().split("Tree=0\n", 1)
    head = "\n".join(line for line in head.split("\n") if not line.startswith("tree_sizes="))
    n = depth
    tree = [
        "Tree=0",
        f"num_leaves={n + 1}",
        "num_cat=0",
        "split_feature=" + " ".join(["0"] * n),
        "split_gain=" + " ".join(["1"] * n),
        "threshold=" + " ".join(str(float(i)) for i in range(n)),
        "decision_type=" + " ".join(["2"] * n),
        "left_child=" + " ".join(str(-(i + 1)) for i in range(n)),
        "right_child=" + " ".join([str(i + 1) for i in range(n - 1)] + [str(-(n + 1))]),
        "leaf_value=" + " ".join(str(float(i % 7)) for i in range(n + 1)),
        "leaf_weight=" + " ".join(["1"] * (n + 1)),
        "leaf_count=" + " ".join(["1"] * (n + 1)),
        "internal_value=" + " ".join(["0"] * n),
        "internal_weight=" + " ".join(["1"] * n),
        "internal_count=" + " ".join(str(n + 1 - i) for i in range(n)),
        "is_linear=0",
        "shrinkage=1",
        "",
        "",
    ]
    path.write_text(head + "\n".join(tree) + "\n" + rest[rest.index("end of trees") :])


@pytest.mark.skipif(sys.platform == "win32", reason="the child relies on the default 8 MiB main-thread stack")
def test_deep_chain_dumps_without_stack_overflow(tmp_path):
    model = tmp_path / "chain.txt"
    _chain_model(model, _DEPTH)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(model)], capture_output=True, text=True, timeout=600, check=False
    )
    assert proc.returncode == 0, f"child died with {proc.returncode}: {proc.stderr[-1500:]}"
    dump = next(line for line in proc.stdout.splitlines() if line.startswith("DUMP")).split()
    assert int(dump[1]) == _DEPTH + 1  # every leaf written
    assert dump[2] == dump[3]  # braces balance
    assert "CONTRIB (3, 2)" in proc.stdout
