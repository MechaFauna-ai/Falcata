"""UNRUN future CPU validator; reject refit-only 'PASS' without six actual ordered Train receipts.

Supply complete stdout+stderr from the exact compiled refit probe with FALCATA_DEBUG=diag.
This file imports no Falcata/CUDA. It is not a launcher and submits no queue work.
"""

import re
import sys
from pathlib import Path


def check(text):
    assert "device_tree: fallback (" not in text, "refit probe fell back"
    assert "PASS: same-learner direct/leaf_pred and reset-smoothed refit" in text
    arm = None
    rows = {
        "eager": {"mode": [], "train": [], "map": [], "route": []},
        "deferred": {"mode": [], "train": [], "map": [], "route": []},
    }
    seen = []
    positions = {"eager": 0, "deferred": 0}
    order = ["map", "mode", "train", "route"]
    for line in text.splitlines():
        marker = re.fullmatch(r"\[map-only-refit-probe\] arm=(eager|deferred) (start|end)", line)
        if marker:
            name, boundary = marker.groups()
            if boundary == "start":
                assert arm is None
                assert name == ("eager" if not seen else "deferred")
                assert name not in seen
                arm = name
                seen.append(name)
            else:
                assert arm == name
                assert positions[name] == 12
                arm = None
            continue
        patterns = {
            "mode": r"device_tree_final_map_only: enabled=(\d+) deferred=(\d+) splits=(\d+) gaps=(\d+) flat=(\d+)",
            "train": r"device_tree: complete depth=(\d+) applied_levels=(\d+) splits=(\d+) growth_readbacks=1 host_tail=0 graph=(\d+) hits=(\d+) misses=(\d+)",
            "map": r"device_tree_final_all_map: enabled=(\d+) complete=(\d+) final_splits=(\d+) leaves=(\d+)",
            "route": r"device_tree_final_all_map_branch: route=(\S+) leaves=(\d+)",
        }
        for kind, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                assert arm is not None, "unbound Train receipt outside a probe arm"
                assert kind == order[positions[arm] % 4], "misordered or unbound current-tree receipt"
                positions[arm] += 1
                rows[arm][kind].append(match.groups())
    assert arm is None
    assert seen == ["eager", "deferred"]
    for name, expected in [("eager", (0, 0)), ("deferred", (1, 1))]:
        assert all(len(rows[name][kind]) == 3 for kind in rows[name]), "need exactly three actual Trains per arm"
        for mode, train, mapped, route in zip(
            *[rows[name][kind] for kind in ["mode", "train", "map", "route"]], strict=True
        ):
            assert tuple(map(int, mode[:2])) == expected
            assert tuple(map(int, train[:3])) == (3, 3, 5)
            assert tuple(map(int, mapped)) == (1, 1, 2, 6)
            assert route == ("inline-all", "6")
            if name == "deferred":
                assert int(mode[2]) == 2
                assert int(mode[3]) == 2
                assert int(mode[4]) >= 2
            else:
                assert tuple(map(int, mode[2:])) == (0, 0, 0)
    return {"ordered_train_receipts": 6, "eager": 3, "deferred": 3}


if __name__ == "__main__":
    print(check(Path(sys.argv[1]).read_text()))
