import copy
import io
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_regression
from sklearn.metrics import accuracy_score

TESTS_DIR = Path(__file__).absolute().parent


@pytest.fixture(scope="module")
def executable(pytestconfig) -> str:
    """Returns the path to the falcata executable."""
    return pytestconfig.getoption("execfile")


def _find_random_open_port() -> int:
    """Find a random open port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port  # noqa: RET504


def _generate_n_ports(n: int) -> Generator[int, None, None]:
    return (_find_random_open_port() for _ in range(n))


def _write_dict(d: Dict, file: io.TextIOWrapper) -> None:
    for k, v in d.items():
        file.write(f"{k} = {v}\n")


def create_data(task: str, n_samples: int = 1_000) -> np.ndarray:
    """Create the appropriate data for the task.

    The data is returned as a numpy array with the label as the first column.
    """
    if task == "binary-classification":
        centers = [[-4, -4], [4, 4]]
        X, y = make_blobs(n_samples, centers=centers, random_state=42)
    elif task == "regression":
        X, y = make_regression(n_samples, n_features=4, n_informative=2, random_state=42)
    return np.hstack([y.reshape(-1, 1), X])


class DistributedMockup:
    """Simulate distributed training."""

    default_train_config = {
        "task": "train",
        # Distributed training is CPU-only ("Currently cuda version only supports
        # training on a single machine"), and a CUDA-capable build with
        # device_type unset resolves to cuda. Pin it: what these tests exercise
        # is socket parallelism, not device selection.
        "device_type": "cpu",
        "pre_partition": True,
        "machine_list_file": TESTS_DIR / "mlist.txt",
        "tree_learner": "data",
        "force_row_wise": True,
        "verbose": 0,
        "num_boost_round": 20,
        "num_leaves": 15,
        "num_threads": 2,
    }

    default_predict_config = {
        "task": "predict",
        "device_type": "cpu",
        "data": TESTS_DIR / "train.txt",
        "input_model": TESTS_DIR / "model0.txt",
        "output_result": TESTS_DIR / "predictions.txt",
    }

    def __init__(self, executable: str):
        self.executable = executable

    # A port found free by _find_random_open_port is released before the worker binds it, so another process can
    # take it in between. A worker that then fails to bind ("Binding port ... failed") is retried with fresh ports.
    max_port_attempts = 5

    def worker_train(self, i: int) -> subprocess.Popen:
        """Start the training process on the `i`-th worker; its output goes to train{i}.log."""
        config_path = TESTS_DIR / f"train{i}.conf"
        cmd = [self.executable, f"config={config_path}"]
        with open(TESTS_DIR / f"train{i}.log", "wt") as log:
            return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    def _run_workers(self) -> Optional[int]:
        """Run every worker to completion.

        Returns None when all succeed, or the index of the worker that lost its port to another process (the
        other workers, which would wait for it, are stopped). Any other failure raises.
        """
        procs = [self.worker_train(i) for i in range(self.n_workers)]
        failed = None
        while failed is None and any(p.poll() is None for p in procs):
            failed = next((i for i, p in enumerate(procs) if p.poll() not in (None, 0)), None)
            time.sleep(0.05)
        if failed is None:
            failed = next((i for i, p in enumerate(procs) if p.returncode != 0), None)
        if failed is None:
            return None
        for p in procs:
            if p.poll() is None:
                p.kill()
            p.wait()
        log = (TESTS_DIR / f"train{failed}.log").read_text()
        if f"Binding port {self.listen_ports[failed]} failed" in log:
            return failed
        raise subprocess.CalledProcessError(procs[failed].returncode, procs[failed].args, output=log)

    def _set_ports(self) -> None:
        """Randomly assign a port for training to each worker and save all ports to mlist.txt."""
        ports = set(_generate_n_ports(self.n_workers))
        i = 0
        max_tries = 100
        while i < max_tries and len(ports) < self.n_workers:
            n_ports_left = self.n_workers - len(ports)
            candidates = _generate_n_ports(n_ports_left)
            ports.update(candidates)
            i += 1
        if len(ports) < self.n_workers:
            raise RuntimeError("Unable to find non-colliding ports.")
        self.listen_ports = list(ports)
        with open(TESTS_DIR / "mlist.txt", "wt") as f:
            for port in self.listen_ports:
                f.write(f"127.0.0.1 {port}\n")

    def _write_data(self, partitions: List[np.ndarray]) -> None:
        """Write all training data as train.txt and each training partition as train{i}.txt."""
        all_data = np.vstack(partitions)
        np.savetxt(str(TESTS_DIR / "train.txt"), all_data, delimiter=",")
        for i, partition in enumerate(partitions):
            np.savetxt(str(TESTS_DIR / f"train{i}.txt"), partition, delimiter=",")

    def fit(self, partitions: List[np.ndarray], train_config: Dict) -> None:
        """Run the distributed training process on a single machine.

        For each worker i:
            1. The i-th partition is saved as train{i}.txt.
            2. A random port is assigned for training.
            3. A configuration file train{i}.conf is created.
            4. The falcata binary is called with config=train{i}.conf in another thread.
            5. The trained model is saved as model{i}.txt. Each model file only differs in data and local_listen_port.
        The whole training set is saved as train.txt.
        """
        self.train_config = copy.deepcopy(self.default_train_config)
        self.train_config.update(train_config)
        self.n_workers = self.train_config["num_machines"]
        self._write_data(partitions)
        self.label_ = np.hstack([partition[:, 0] for partition in partitions])
        self.port_attempts = 0
        for _ in range(self.max_port_attempts):
            self.port_attempts += 1
            self._set_ports()
            for i in range(self.n_workers):
                self.write_train_config(i)
            if self._run_workers() is None:
                return
        raise RuntimeError(f"Workers could not bind their ports in {self.max_port_attempts} attempts.")

    def predict(self, predict_config: Dict[str, Any]) -> np.ndarray:
        """Compute the predictions using the model created in the fit step.

        predict_config is used to predict the training set train.txt
        The predictions are saved as predictions.txt and are then loaded to return them as a numpy array.
        """
        self.predict_config = copy.deepcopy(self.default_predict_config)
        self.predict_config.update(predict_config)
        config_path = TESTS_DIR / "predict.conf"
        with open(config_path, "wt") as file:
            _write_dict(self.predict_config, file)
        cmd = [self.executable, f"config={config_path}"]
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            raise RuntimeError("Error in prediction")
        return np.loadtxt(str(TESTS_DIR / "predictions.txt"))

    def write_train_config(self, i: int) -> None:
        """Create a file train{i}.conf with the required configuration to train.

        Each worker gets a different port and piece of the data, the rest are the
        model parameters contained in `self.config`.
        """
        with open(TESTS_DIR / f"train{i}.conf", "wt") as file:
            output_model = TESTS_DIR / f"model{i}.txt"
            data = TESTS_DIR / f"train{i}.txt"
            file.write(f"output_model = {output_model}\n")
            file.write(f"local_listen_port = {self.listen_ports[i]}\n")
            file.write(f"data = {data}\n")
            _write_dict(self.train_config, file)


def test_classifier(executable):
    """Test the classification task."""
    num_machines = 2
    data = create_data(task="binary-classification")
    partitions = np.array_split(data, num_machines)
    train_params = {
        "objective": "binary",
        "num_machines": num_machines,
    }
    clf = DistributedMockup(executable)
    clf.fit(partitions, train_params)
    y_probas = clf.predict(predict_config={})
    y_pred = y_probas > 0.5
    assert accuracy_score(clf.label_, y_pred) == 1.0


def test_regressor(executable):
    """Test the regression task."""
    num_machines = 2
    data = create_data(task="regression")
    partitions = np.array_split(data, num_machines)
    train_params = {
        "objective": "regression",
        "num_machines": num_machines,
    }
    reg = DistributedMockup(executable)
    reg.fit(partitions, train_params)
    y_pred = reg.predict(predict_config={})
    np.testing.assert_allclose(y_pred, reg.label_, rtol=0.2, atol=50.0)


def test_rf_with_an_empty_local_bag(executable):
    """RF skips growing a tree on an empty bag; with data-parallel learners the skip must be collective.

    Worker 1 holds 10 rows, so at bagging_fraction=0.15 its local bag is often empty while worker 0's is not.
    Skipping Train there alone left worker 0 waiting in the histogram Allreduce (a hang).
    """
    num_machines = 2
    data = create_data(task="regression")
    partitions = [data[:-10], data[-10:]]
    train_params = {
        "objective": "regression",
        "boosting": "rf",
        "bagging_fraction": 0.15,
        "bagging_freq": 1,
        "num_machines": num_machines,
    }
    reg = DistributedMockup(executable)
    reg.fit(partitions, train_params)
    y_pred = reg.predict(predict_config={})
    assert np.all(np.isfinite(y_pred))


def test_port_taken_before_the_worker_binds_is_retried(executable, monkeypatch):
    """Another process takes a worker's port between discovery and bind: training retries with fresh ports."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as thief:
        thief.bind(("", 0))
        thief.listen(1)
        stolen = thief.getsockname()[1]
        handed_out = []

        def find_port_first_stolen() -> int:
            port = stolen if not handed_out else _find_random_open_port_unpatched()
            handed_out.append(port)
            return port

        _find_random_open_port_unpatched = _find_random_open_port
        monkeypatch.setattr(f"{__name__}._find_random_open_port", find_port_first_stolen)
        num_machines = 2
        data = create_data(task="binary-classification")
        clf = DistributedMockup(executable)
        clf.fit(np.array_split(data, num_machines), {"objective": "binary", "num_machines": num_machines})
        assert handed_out[0] == stolen
        assert clf.port_attempts == 2
        assert stolen not in clf.listen_ports
        assert accuracy_score(clf.label_, clf.predict(predict_config={}) > 0.5) == 1.0


def test_ports_found_on_last_attempt(tmp_path, monkeypatch):
    attempts = iter([[1], *([[1]] * 99), [2]])
    monkeypatch.setattr(f"{__name__}._generate_n_ports", lambda count: next(attempts))
    monkeypatch.setattr(f"{__name__}.TESTS_DIR", tmp_path)
    mockup = DistributedMockup("unused")
    mockup.n_workers = 2
    mockup._set_ports()
    assert set(mockup.listen_ports) == {1, 2}
