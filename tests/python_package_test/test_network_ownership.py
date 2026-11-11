"""Ownership of the process-global native network across boosters."""

import gc
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident

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
    else:
        # Refit must also work if the current configuration differs from params.
        original.set_network("localhost:12401")
    refitted = original.refit(data, data[:, 0])
    expected_inits = ["init"] * (3 if machines_in_params else 1)
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
        assert calls == expected_inits
        assert survivor._network
        survivor.free_network()
        survivor.free_network()
        assert calls == expected_inits + ["free"]
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
        other.free_network()
        assert calls == []
        original.set_network("localhost:12400")
        original.set_network("localhost:12400")
        original.set_network("localhost:12401")
        other.set_network("localhost:12401")
        original.set_network("localhost:12402")
        assert calls == ["init"] * 5
        original.free_network()
        assert calls == ["init"] * 5
        monkeypatch.setattr(basic._LIB, "FLC_GetLastError", lambda: b"network failure")
        monkeypatch.setattr(basic._LIB, "FLC_NetworkFree", lambda: -1)
        with pytest.raises(basic.FalcataError, match="network failure"):
            other.free_network()
        assert other._network
        monkeypatch.setattr(basic._LIB, "FLC_NetworkFree", lambda: calls.append("free") or 0)
        other = None
        gc.collect()
        assert calls == ["init"] * 5 + ["free"]
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


@pytest.fixture(autouse=True)
def check_network_ownership_balance():
    gc.collect()
    assert basic._get_network_state().refcount == 0
    yield
    gc.collect()
    assert basic._get_network_state().refcount == 0


@pytest.mark.parametrize("release_on_other_thread", [False, True])
def test_network_ownership_follows_native_thread_state(monkeypatch, release_on_other_thread):
    calls = []
    monkeypatch.setattr(basic._LIB, "FLC_NetworkInit", lambda *args: calls.append(("init", get_ident())) or 0)
    monkeypatch.setattr(basic._LIB, "FLC_NetworkFree", lambda: calls.append(("free", get_ident())) or 0)
    data = np.arange(60).reshape(20, 3)
    original = lgb.train({"verbosity": -1, "num_threads": 1}, lgb.Dataset(data, label=data[:, 0]), 1)
    model = original.model_to_string()
    original.set_network("localhost:12400")

    def worker():
        booster = lgb.Booster(model_str=model)
        booster.set_network("localhost:12401")
        if not release_on_other_thread:
            booster.free_network()
        return booster, get_ident()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            other, worker_ident = executor.submit(worker).result()
            if release_on_other_thread:
                other.free_network()
                assert calls == [("init", get_ident()), ("init", worker_ident)]
                # Deferred cleanup must execute on the native owning thread.
                executor.submit(basic._get_network_state).result()
            assert calls == [("init", get_ident()), ("init", worker_ident), ("free", worker_ident)]
            original.free_network()
            assert calls[-1] == ("free", get_ident())
    finally:
        original.free_network()


def test_network_acquired_before_reentrant_initialization(monkeypatch):
    calls = []
    monkeypatch.setattr(basic._LIB, "FLC_NetworkInit", lambda *args: 0)
    monkeypatch.setattr(basic._LIB, "FLC_NetworkFree", lambda: calls.append("free") or 0)
    data = np.arange(60).reshape(20, 3)
    original = lgb.train({"verbosity": -1, "num_threads": 1}, lgb.Dataset(data, label=data[:, 0]), 1)
    other = lgb.Booster(model_str=original.model_to_string())
    original.set_network("localhost:12400")

    def initialize(*args):
        original.free_network()
        assert calls == []
        return 0

    monkeypatch.setattr(basic._LIB, "FLC_NetworkInit", initialize)
    try:
        other.set_network("localhost:12400")
        assert other._network
        other.free_network()
        assert calls == ["free"]
    finally:
        original.free_network()
        other.free_network()
