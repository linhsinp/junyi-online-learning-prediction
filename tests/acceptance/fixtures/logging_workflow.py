"""Small real Flyte pipeline with isolated SQLite and synthetic input boundaries."""

import logging
import os
import time
from pathlib import Path

import pandas as pd

from junyi_predictor.progress import operation
from junyi_predictor.workflows import training


def fixture_inputs(start_date, end_date, engine):
    """Wait for the test to observe a heartbeat before returning five events."""
    with operation("input.fixture", log=logging.getLogger("junyi_predictor.fixture")):
        gate = os.getenv("JUNYI_TEST_GATE")
        deadline = time.monotonic() + 20
        while gate and not Path(gate).exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Fixture gate was never released")
            time.sleep(0.02)
        if os.getenv("JUNYI_TEST_FAILURE") == "1":
            raise RuntimeError("Synthetic input failure")
        log = pd.DataFrame(
            {
                "timestamp_TW": pd.date_range("2019-06-01", periods=5),
                "uuid": ["u1", "u1", "u2", "u2", "u3"],
                "ucid": ["c1", "c2", "c1", "c2", "c1"],
                "upid": ["p1", "p2", "p1", "p2", "p1"],
                "problem_number": [1, 2, 1, 2, 3],
                "exercise_problem_repeat_session": [0, 1, 0, 1, 0],
                "is_correct": [True, False, True, False, True],
                "total_sec_taken": [11, 12, 13, 14, 15],
                "is_hint_used": [False, True, False, True, False],
                "is_downgrade": [0, 0, 1, 0, 0],
                "is_upgrade": [0, 1, 0, 0, 0],
                "level": [1, 2, 3, 4, 5],
            }
        )
        user = pd.DataFrame(
            {
                "uuid": ["u1", "u2", "u3"],
                "user_grade": [5, 6, 7],
                "gender": ["female", "male", None],
            }
        )
        content = pd.DataFrame({"ucid": ["c1", "c2"], "level4_id": ["l1", "l2"]})
        return log, user, content


training.load_data_for_training = fixture_inputs
training_pipeline = training.training_pipeline
