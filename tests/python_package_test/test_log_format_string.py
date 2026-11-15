# coding: utf-8
"""Text from the input must never be used as a printf format.

OMP_LOOP_EX_END logged a captured exception with Log::Warning(ex.what()), so the
message became the format string. A data file token such as "%s%s%s" is quoted in
the parser's "Unknown token ..." error, and when that row is first parsed inside a
parallel loop (it was not in the bin-construction sample) vsnprintf read
nonexistent arguments: the process segfaulted instead of raising (at any verbosity
that shows warnings, which includes the default).
"""

import subprocess
import sys
import textwrap

_CHILD = textwrap.dedent(
    """
    import sys
    import falcata as flc

    try:
        flc.Dataset(
            sys.argv[1],
            params={"verbose": 0, "num_threads": 4, "bin_construct_sample_cnt": 100, "data_random_seed": 1},
        ).construct()
        print("NO ERROR")
    except flc.basic.FalcataError as err:
        print("RAISED", err)
    """
)


def test_data_token_with_percent_signs_raises_cleanly(tmp_path):
    path = tmp_path / "data.csv"
    rows = [f"{i % 7},{i * 0.5},{(i * 37) % 11}" for i in range(4000)]
    # not in the 100-row bin-construction sample, so the parallel extraction loop parses it first
    rows.insert(2000, "1,%s%s%s%s%s%s%s%s%s%s,2")
    path.write_text("\n".join(rows) + "\n")
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(path)], capture_output=True, text=True, timeout=300, check=False
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "RAISED Unknown token %s%s%s%s%s%s%s%s%s%s in data file" in proc.stdout, proc.stdout
