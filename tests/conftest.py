import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def feature_accuracy_log_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uuid": ["u1", "u1", "u2"],
            "ucid": ["c1", "c1", "c2"],
            "upid": ["p1", "p1", "p2"],
            "level": [1, 2, 3],
            "is_correct": [True, False, True],
            "user_grade": [5, 5, 6],
            "female": [1, 1, 0],
            "male": [0, 0, 0],
            "unspecified": [0, 0, 1],
            "problem_number": [1, 2, 1],
            "exercise_problem_repeat_session": [0, 1, 0],
        }
    )


@pytest.fixture
def feature_log_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uuid": ["u1", "u1", "u2"],
            "ucid": ["c1", "c2", "c1"],
            "upid": ["p1", "p2", "p1"],
            "level": [1, 2, 3],
            "is_correct": [True, False, True],
            "user_grade": [5, 5, 6],
            "female": [1, 1, 0],
            "male": [0, 0, 1],
            "unspecified": [0, 0, 0],
            "problem_number": [1, 2, 1],
            "exercise_problem_repeat_session": [0, 1, 0],
        }
    )


@pytest.fixture
def feature_user_df() -> pd.DataFrame:
    return pd.DataFrame({"uuid": ["u1", "u2"]})


@pytest.fixture
def feature_content_df() -> pd.DataFrame:
    return pd.DataFrame({"ucid": ["c1", "c2"], "level4_id": ["l1", "l2"]})


@pytest.fixture
def preprocess_raw_log_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_TW": ["2024-01-02", "2024-01-01"],
            "uuid": ["u2", "u1"],
            "ucid": ["c2", "c1"],
            "upid": ["p2", "p1"],
            "problem_number": [2, 1],
            "exercise_problem_repeat_session": [0, 1],
            "is_correct": [False, True],
            "total_sec_taken": [20, 10],
            "is_hint_used": [False, True],
            "is_downgrade": [1, 0],
            "is_upgrade": [0, 1],
            "level": [3, 4],
        }
    )


@pytest.fixture
def preprocess_raw_user_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uuid": ["u1", "u2"],
            "user_grade": [5, 6],
            "gender": ["female", None],
        }
    )


@pytest.fixture
def preprocess_raw_content_df() -> pd.DataFrame:
    return pd.DataFrame({"ucid": ["c1", "c2"], "level4_id": ["l1", "l2"]})


@pytest.fixture
def pipeline_raw_log_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_TW": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
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


@pytest.fixture
def pipeline_raw_user_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uuid": ["u1", "u2", "u3"],
            "user_grade": [5, 6, 7],
            "gender": ["female", "male", None],
        }
    )


@pytest.fixture
def pipeline_raw_content_df() -> pd.DataFrame:
    return pd.DataFrame({"ucid": ["c1", "c2"], "level4_id": ["l1", "l2"]})


@pytest.fixture
def training_log_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_grade": [5, 6, 5, 6, 7],
            "female": [1, 0, 1, 0, 0],
            "male": [0, 1, 0, 1, 0],
            "unspecified": [0, 0, 0, 0, 1],
            "v_upid_acc": [0.2, 0.4, 0.6, 0.8, 1.0],
            "level": [1, 2, 3, 4, 5],
            "problem_number": [1, 2, 3, 4, 5],
            "exercise_problem_repeat_session": [0, 1, 0, 1, 0],
            "is_correct": [True, False, True, False, True],
        }
    )


@pytest.fixture
def training_concept_matrix() -> np.ndarray:
    return np.arange(10, dtype=float).reshape(5, 2)


@pytest.fixture
def training_level4_matrix() -> np.ndarray:
    return np.arange(5, dtype=float).reshape(5, 1)


@pytest.fixture
def simple_binary_training_arrays() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    X_test = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_train = np.array([0, 1, 0, 1], dtype=bool)
    y_test = np.array([0, 1], dtype=bool)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def benchmark_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = 10
    df_log = pd.DataFrame(
        {
            "uuid": [f"u{i}" for i in range(rows)],
            "user_grade": pd.Series([5, 6, 5, 6, 5, 6, 5, 6, 5, 6], dtype="int8"),
            "ucid": [f"c{i}" for i in range(rows)],
            "is_correct": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "level": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        }
    )
    df_user = pd.DataFrame(
        {
            "uuid": [f"u{i}" for i in range(rows)],
            "user_grade": pd.Series([5, 6, 5, 6, 5, 6, 5, 6, 5, 6], dtype="int8"),
            "gender": pd.Series(["female", "male"] * 5, dtype="category"),
            "has_teacher_cnt": [1] * rows,
            "is_self_coach": [False, True] * 5,
            "has_student_cnt": [0] * rows,
            "belongs_to_class_cnt": [1] * rows,
            "has_class_cnt": [1] * rows,
        }
    )
    df_content = pd.DataFrame(
        {
            "ucid": [f"c{i}" for i in range(rows)],
            "difficulty": pd.Series(["easy", "hard"] * 5, dtype="category"),
            "learning_stage": pd.Series(["elementary", "junior"] * 5, dtype="category"),
        }
    )
    return df_log, df_user, df_content


@pytest.fixture
def mocked_stage_split():
    return SimpleNamespace(
        X_train=np.array([[0.0], [1.0]]),
        y_train=np.array([0, 1]),
        X_test=np.array([[0.5]]),
        y_test=np.array([1]),
    )


@pytest.fixture
def mock_stage_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_log = pd.DataFrame({"value": [1, 2, 3]})
    df_user = pd.DataFrame({"value": [1, 2]})
    df_content = pd.DataFrame({"value": [1]})
    return df_log, df_user, df_content
