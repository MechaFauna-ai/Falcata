# coding: utf-8
"""A parse error during two_round loading raises instead of aborting.

PipelineReader::Read ran process_fun while a reader thread was still
running. When process_fun threw (Log::Fatal on a malformed line in the
second pass of two_round loading), unwinding destroyed the joinable
std::thread and the process died in std::terminate.
"""

import pytest

import falcata as flc
from falcata.basic import FalcataError


def test_two_round_malformed_line_raises(tmp_path):
    path = tmp_path / "bad.csv"
    good = "0,0.5,0.25\n1,0.75,0.125\n"
    with open(path, "w") as f:
        f.write(good * 2)
        f.write("1,abc,2\n")
        # > 16 MiB, so the bad line is in a block processed while the
        # pipeline's reader thread is still reading the next one
        f.write(good * 800_000)
    params = {
        "two_round": True,
        "verbose": -1,
        # keep the bad line out of the bin-construction sample (first pass)
        "bin_construct_sample_cnt": 100,
        "data_random_seed": 3,
    }
    with pytest.raises(FalcataError, match="Unknown token abc"):
        flc.Dataset(str(path), params=params).construct()
