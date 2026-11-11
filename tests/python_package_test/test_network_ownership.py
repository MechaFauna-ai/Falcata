"""Ownership of the process-global native network across boosters."""

import gc
import pickle

import numpy as np
import pytest

import falcata as lgb
from falcata import basic


@pytest.mark.parametrize("machines_in_params", [False, True])
@pytest.mark.parametrize("delete_original", [False, True])
def test_refit_keeps_network_until_last_owner(monkeypatch, machines_in_params, delete_original):
    calls = []
    monkeypatch.setattr(basic._LIB, "FLC_NetworkInit", lambda *args: calls.append("init") or 0)
    monkeypatch.setattr(basic._LIB, "FLC_NetworkFree", lambda: calls.append("free") or 0)
    data = np.random.default_rng(42).normal(size=(50, 3))
    params = {"verbosity": -1, "num_threads": 1, "min_data_in_leaf": 2}
    if machines_in_params:
        params["machines"] = "localhost:12400"
    original = lgb.train(params, lgb.Dataset(data, label=data[:, 0]), num_boost_round=2)
    if not machines_in_params:
        original.set_network("localhost:12400")
    refitted = original.refit(data, data[:, 0])
    try:
        assert original._network
        assert refitted._network
        if delete_original:
            del original
            survivor = refitted
        else:
            del refitted
            survivor = original
        gc.collect()
        assert calls == ["init"]
        assert survivor._network
        survivor.free_network()
        survivor.free_network()
        assert calls == ["init", "free"]
    finally:
        # Also release ownership if an assertion fails, before mocks are restored.
        for name in ("original", "refitted", "survivor"):
            booster = locals().get(name)
            if booster is not None and booster._network:
                booster.free_network()


def test_network_reinitialization_and_failed_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(basic._LIB, "FLC_NetworkInit", lambda *args: calls.append("init") or 0)
    monkeypatch.setattr(basic._LIB, "FLC_NetworkFree", lambda: calls.append("free") or 0)
    data = np.arange(60).reshape(20, 3)
    original = lgb.train({"verbosity": -1, "num_threads": 1}, lgb.Dataset(data, label=data[:, 0]), 1)
    other = lgb.Booster(model_str=original.model_to_string())
    try:
        original.set_network("localhost:12400")
        original.set_network("localhost:12400")
        original.set_network("localhost:12401")
        other.set_network("localhost:12401")
        assert calls == ["init", "init"]
        with pytest.raises(basic.FalcataError, match="other boosters"):
            original.set_network("localhost:12402")
        original.free_network()
        assert calls == ["init", "init"]
        monkeypatch.setattr(basic._LIB, "FLC_GetLastError", lambda: b"network failure")
        monkeypatch.setattr(basic._LIB, "FLC_NetworkFree", lambda: -1)
        with pytest.raises(basic.FalcataError, match="network failure"):
            other.free_network()
        assert other._network
        monkeypatch.setattr(basic._LIB, "FLC_NetworkFree", lambda: calls.append("free") or 0)
        other = None
        gc.collect()
        assert calls == ["init", "init", "free"]
        monkeypatch.setattr(basic._LIB, "FLC_NetworkInit", lambda *args: -1)
        with pytest.raises(basic.FalcataError, match="network failure"):
            original.set_network("localhost:12400")
        assert not original._network
    finally:
        original.free_network()
        if other is not None:
            other.free_network()


@pytest.mark.parametrize("legacy_state", [False, True])
def test_pickle_does_not_copy_network_ownership(monkeypatch, legacy_state):
    calls = []
    monkeypatch.setattr(basic._LIB, "FLC_NetworkInit", lambda *args: calls.append("init") or 0)
    monkeypatch.setattr(basic._LIB, "FLC_NetworkFree", lambda: calls.append("free") or 0)
    data = np.arange(60).reshape(20, 3)
    original = lgb.train({"verbosity": -1, "num_threads": 1}, lgb.Dataset(data, label=data[:, 0]), 1)
    original.set_network("localhost:12400")
    restored = None
    try:
        if legacy_state:
            state = original.__getstate__()
            state["_network"] = True
            restored = lgb.Booster.__new__(lgb.Booster)
            restored.__setstate__(state)
        else:
            restored = pickle.loads(pickle.dumps(original))
        assert not restored._network
        restored = None
        gc.collect()
        assert original._network
        assert calls == ["init"]
        original.free_network()
        assert calls == ["init", "free"]
    finally:
        # A failing ownership assertion must not release an unregistered owner.
        if restored is not None:
            restored._network = False
        original.free_network()
