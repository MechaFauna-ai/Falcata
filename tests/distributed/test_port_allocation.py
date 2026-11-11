"""The final allowed port-discovery attempt can still succeed."""

import _test_distributed as distributed


def test_ports_found_on_last_attempt(tmp_path, monkeypatch):
    attempts = iter([[1], *([[1]] * 99), [2]])
    monkeypatch.setattr(distributed, "_generate_n_ports", lambda count: next(attempts))
    monkeypatch.setattr(distributed, "TESTS_DIR", tmp_path)
    mockup = distributed.DistributedMockup("unused")
    mockup.n_workers = 2
    mockup._set_ports()
    assert set(mockup.listen_ports) == {1, 2}
