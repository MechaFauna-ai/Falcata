"""Early stopping can be reused after a run where it was disabled."""

import pytest

from falcata.callback import CallbackEnv, EarlyStopException, early_stopping
from falcata.engine import CVBooster


def test_early_stopping_reenabled_after_dart():
    stop = early_stopping(2, verbose=False)

    def env(iteration, boosting):
        return CallbackEnv(
            model=CVBooster(),
            params={"boosting": boosting},
            iteration=iteration,
            begin_iteration=0,
            end_iteration=10,
            evaluation_result_list=[("valid", "l2", 1.0, False)],
        )

    with pytest.warns(UserWarning, match="dart"):
        stop(env(0, "dart"))
    stop(env(0, "gbdt"))
    stop(env(1, "gbdt"))
    with pytest.raises(EarlyStopException) as exc:  # noqa: PT011 - carries iteration state, no message
        stop(env(2, "gbdt"))
    assert exc.value.best_iteration == 0
