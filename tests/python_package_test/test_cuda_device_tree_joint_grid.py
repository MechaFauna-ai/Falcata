# coding: utf-8
"""Real whole-tree models and live scores versus the frozen77 reference; gpuq only."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")
_REQUIRES_REFERENCE_QA = pytest.mark.skipif(
    os.environ.get("FALCATA_GRID_REFERENCE_QA") != "1"
    or not os.environ.get("FALCATA_GRID_REFERENCE_RUNTIME")
    or not os.environ.get("FALCATA_GRID_CANDIDATE_RUNTIME"),
    reason="requires explicit frozen-reference QA opt-in and both private runtimes",
)

_CHILD = r"""
import inspect,json,runpy,sys
from pathlib import Path
runtime=Path(sys.argv[4]).resolve()
sys.path.insert(0,str(runtime))
ns=runpy.run_path(sys.argv[1])
actual=Path(ns['flc'].__file__).resolve().parent
assert actual == runtime/'falcata', (str(actual), str(runtime/'falcata'))
source=inspect.getsource(ns['_final_all_map_probe'])
tail='    sys.stdout.write(json.dumps({"stages": len(stages), "nonzero_distinct_terminal_gaps": gap_histories[1]}) + "\\n")'
assert source.count(tail)==1
source=source.replace(tail,'    return snapshots, gap_histories')
exec(source,ns)
snapshots,gaps=ns['_final_all_map_probe'](sys.argv[2],sys.argv[3],'')
data={'snapshots':[[{'empty':s[0],'model':s[1],'score_bits':s[2].tobytes().hex()} for s in arm] for arm in snapshots],
      'gaps':gaps,'stages':len(snapshots[1])}
print(json.dumps(data))
"""


@_REQUIRES_CUDA
@_REQUIRES_REFERENCE_QA
@pytest.mark.parametrize("graph", ["off", "on"])
@pytest.mark.parametrize("scenario", ["full", "history"])
def test_whole_joint_grid_reference_cuda(graph, scenario):
    reference = Path(os.environ["FALCATA_GRID_REFERENCE_RUNTIME"])
    candidate = Path(os.environ["FALCATA_GRID_CANDIDATE_RUNTIME"])
    helper = str(Path(__file__).with_name("test_cuda_device_tree.py"))
    outputs = []
    diagnostics = []
    for runtime in [reference, candidate]:
        result = subprocess.run(
            [sys.executable, "-c", _CHILD, helper, graph, scenario, str(runtime)],
            env=dict(os.environ, PYTHONPATH=str(runtime), FALCATA_DEBUG="diag"),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + "\nSTDERR:\n" + result.stderr
        outputs.append(json.loads(result.stdout.splitlines()[-1]))
        diagnostics.append(result.stderr)
    assert outputs[0] == outputs[1], "full model bytes/live-score bits/gap history differ from frozen77"
    data = outputs[1]
    diag = diagnostics[1]
    events = re.findall(
        r"device_tree: complete depth=(\d+) applied_levels=(\d+) splits=(\d+) "
        r"growth_readbacks=1 host_tail=0 graph=(\d+) hits=(\d+) misses=(\d+)",
        diag,
    )
    assert len(events) == data["stages"], diag
    assert "device_tree: host fallback" not in diag
    if scenario == "full":
        assert tuple(map(int, events[0][:3])) == (10, 10, 1023)
    else:
        assert len(events) == 10
        assert all(data["gaps"][1][:3]), "nonzero-ID heterogeneous terminal gaps disappeared"
        assert int(events[3][2]) == 0, "stump stage changed"
        assert 0 < int(events[4][1]) < 10, "early exhaustion stage changed"
        if graph == "on":
            assert int(events[0][5]) > 0, "first geometry must really miss"
            assert int(events[2][4]) > 0, "repeated geometry must really hit"
    if graph == "off":
        assert all(int(event[3]) == 0 for event in events)
