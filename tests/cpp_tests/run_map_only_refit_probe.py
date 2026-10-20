"""UNRUN future queued runner: a probe PASS without six bound ordered Train receipts is a failure."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from check_map_only_refit_receipts import check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    assert os.environ.get("GPUQ_JOB_ID"), "future CUDA execution requires an authorized gpuq reservation"
    assert os.getsid(0) == int(os.environ["NATIVE_RESERVATION_SID"])
    assert 0 < args.timeout <= 1800
    assert args.executable.is_absolute()
    assert args.executable.is_file()
    assert not args.output.exists(), "preserve every failed/successful raw log; choose a new output path"
    with args.output.open("xb") as log:
        proc = subprocess.Popen(
            [str(args.executable)], env=dict(os.environ, FALCATA_DEBUG="diag"), stdout=log, stderr=subprocess.STDOUT
        )
        assert os.getsid(proc.pid) == os.getsid(0), "probe left the reservation SID"
        try:
            rc = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
    assert rc == 0, args.output.read_text()
    result = check(args.output.read_text())
    print(
        json.dumps(
            {
                "exit_code": rc,
                "gpuq_job_id": os.environ["GPUQ_JOB_ID"],
                "receipts": result,
                "raw_log": str(args.output),
                "performance_claim": False,
            }
        )
    )


if __name__ == "__main__":
    main()
