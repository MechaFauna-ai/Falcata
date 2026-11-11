"""Unavailable CUDA may skip FIL tests; CUDA training regressions must fail."""

import pytest

import falcata as flc

from .test_basic import fil_cuda_booster


@pytest.mark.parametrize(
    ("message", "skip"),
    [("CUDA Tree Learner was not enabled in this build.", True), ("CUDA kernel regression", False)],
)
def test_fil_fixture_classifies_training_errors(monkeypatch, message, skip):
    monkeypatch.setattr(pytest, "importorskip", lambda *args, **kwargs: None)

    def broken_train(*args, **kwargs):
        raise flc.basic.FalcataError(message)

    monkeypatch.setattr(flc, "train", broken_train)
    with pytest.raises(BaseException) as exc:  # noqa: PT011 - compare exact exception category below
        fil_cuda_booster.__wrapped__()
    assert isinstance(exc.value, pytest.skip.Exception if skip else flc.basic.FalcataError)
