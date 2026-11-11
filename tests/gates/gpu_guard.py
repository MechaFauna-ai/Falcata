"""GPU resource guard for the gates suite.

The gates GPU is shared (training jobs / other agents run on it). Every
GPU-touching gate step starts by calling this guard: it waits until the GPU has
been sufficiently idle for a sustained window, then exits 0. If the GPU never
frees up within the timeout it exits 3 ("GPU busy" -- rerun later), which CI
surfaces as a distinct, rerunnable failure rather than a regression.

Sufficient resources means BOTH:
  - free VRAM >= --require-free-mb (default 8000 MiB)
  - GPU utilization <= --max-util (default 20 %)
Resource checks must pass at all polls spanning --sustain-sec seconds.
Activity between polls is unobserved; run GPU work through gpuq for exclusive
access. On multi-GPU hosts select a visible GPU UUID or pass --gpu-id explicitly.
"""

import argparse
import os
import subprocess
import sys
import time


def gpu_state(device=None):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    first = visible.split(",")[0].strip() if visible is not None else None
    if (first or "").startswith("MIG-") or str(device).startswith("MIG-"):
        raise ValueError("MIG per-instance resource telemetry is unsupported by this guard")
    if device is None and first is not None:
        if first.startswith("GPU-"):
            device = first
        elif first != "0":
            raise ValueError("Select a GPU UUID or pass --gpu-id with its nvidia-smi ID")
    command = ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    if device is not None:
        command.append(f"--id={device}")
    out = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    rows = out.splitlines()
    if len(rows) != 1:
        raise ValueError("Select exactly one GPU with a visible UUID or --gpu-id")
    free_mb, util = (int(x.strip()) for x in rows[0].split(","))
    return free_mb, util


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-id", help="nvidia-smi GPU index or UUID (required for ambiguous ordinal maps)")
    ap.add_argument("--require-free-mb", type=int, default=8000)
    ap.add_argument("--max-util", type=int, default=20)
    ap.add_argument("--sustain-sec", type=int, default=15)
    ap.add_argument("--timeout-min", type=float, default=45.0)
    ap.add_argument("--poll-sec", type=int, default=20)
    args = ap.parse_args()

    deadline = time.monotonic() + args.timeout_min * 60
    ok_since = None
    while True:
        try:
            free_mb, util = gpu_state(args.gpu_id)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            print(f"gpu_guard: cannot query nvidia-smi ({e})", flush=True)
            sys.exit(2)
        ok = free_mb >= args.require_free_mb and util <= args.max_util
        now = time.monotonic()
        if ok:
            if ok_since is None:
                ok_since = now
            if now - ok_since >= args.sustain_sec:
                print(f"gpu_guard: OK (free={free_mb}MiB util={util}%)", flush=True)
                sys.exit(0)
        else:
            ok_since = None
        if now >= deadline:
            print(
                f"gpu_guard: GPU busy after {args.timeout_min:.0f} min "
                f"(free={free_mb}MiB util={util}%; need free>={args.require_free_mb} "
                f"util<={args.max_util}). Rerun when the GPU frees up.",
                flush=True,
            )
            sys.exit(3)
        time.sleep(min(args.poll_sec, max(1, int(deadline - now))))


if __name__ == "__main__":
    main()
