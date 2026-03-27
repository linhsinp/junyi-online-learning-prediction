"""Training stage contracts and reusable helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class TrainingSplit:
    """Explicit output contract for train/test data consumed by model training."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_parquet_dataframe(path_data: str) -> pd.DataFrame:
    """Load a Parquet dataframe used by later pipeline stages."""
    return pd.read_parquet(path_data)


def load_feature_matrices(
    path_m_concept_proficiency: str, path_m_proficiency_level4: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load saved feature matrices used by the training stage."""
    m_concept_proficiency = np.load(path_m_concept_proficiency)["arr_0"]
    m_proficiency_level4 = np.load(path_m_proficiency_level4)["arr_0"]
    return m_concept_proficiency, m_proficiency_level4


def split_training_data(
    df_log: pd.DataFrame,
    m_concept_proficiency: np.ndarray,
    m_proficiency_level4: np.ndarray,
    num_samples: int | None = None,
    random_seed: int = 760,
) -> TrainingSplit:
    """Create a deterministic 80/20 split and append engineered matrices."""
    if num_samples is None:
        num_samples = df_log.shape[0]

    np.random.seed(random_seed)
    num_train_samples = int(num_samples * 0.8)
    samples_train = np.random.choice(
        range(num_samples), num_train_samples, replace=False
    )
    mask_train = np.zeros(num_samples, dtype=bool)
    mask_train[samples_train] = True

    selected_rows = df_log.head(num_samples)
    X_train = np.concatenate(
        (
            selected_rows.loc[mask_train, "user_grade"].to_numpy()[:, np.newaxis],
            selected_rows.loc[mask_train, ["female", "male", "unspecified"]].to_numpy(),
            selected_rows.loc[mask_train, ["v_upid_acc"]].to_numpy(),
            selected_rows.loc[mask_train, "level"].to_numpy()[:, np.newaxis],
            selected_rows.loc[mask_train, "problem_number"].to_numpy()[:, np.newaxis],
            selected_rows.loc[mask_train, "exercise_problem_repeat_session"].to_numpy()[
                :, np.newaxis
            ],
            m_concept_proficiency[:num_samples, :][mask_train, :],
            m_proficiency_level4[:num_samples, :][mask_train, :],
        ),
        axis=1,
    )
    y_train = selected_rows.loc[mask_train, "is_correct"].to_numpy(dtype=bool)

    X_test = np.concatenate(
        (
            selected_rows.loc[~mask_train, "user_grade"].to_numpy()[:, np.newaxis],
            selected_rows.loc[
                ~mask_train, ["female", "male", "unspecified"]
            ].to_numpy(),
            selected_rows.loc[~mask_train, ["v_upid_acc"]].to_numpy(),
            selected_rows.loc[~mask_train, "level"].to_numpy()[:, np.newaxis],
            selected_rows.loc[~mask_train, "problem_number"].to_numpy()[:, np.newaxis],
            selected_rows.loc[
                ~mask_train, "exercise_problem_repeat_session"
            ].to_numpy()[:, np.newaxis],
            m_concept_proficiency[:num_samples, :][~mask_train, :],
            m_proficiency_level4[:num_samples, :][~mask_train, :],
        ),
        axis=1,
    )
    y_test = selected_rows.loc[~mask_train, "is_correct"].to_numpy(dtype=bool)

    return TrainingSplit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def apply_min_max_transformation(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Scale training and test matrices independently using Min-Max scaling."""
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)
    return X_train, X_test


def train_and_evaluate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
) -> dict:
    """Train a supported classifier and return train/test accuracy metrics."""
    if model_type == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(
            X_train, y_train
        )
    elif model_type == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
    elif model_type == "LogisticRegression_L2":
        model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    elif model_type == "LogisticRegression_L1":
        model = LogisticRegression(
            penalty="l1", solver="saga", random_state=0, max_iter=1000
        ).fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return {
        "train_score": model.score(X_train, y_train),
        "test_score": model.score(X_test, y_test),
    }


def train_benchmark_model(
    df_log: pd.DataFrame, df_user: pd.DataFrame, df_content: pd.DataFrame
) -> tuple[LogisticRegression, float]:
    """Train the legacy benchmark logistic-regression model."""
    df1 = pd.merge(
        df_log,
        df_user,
        how="inner",
        left_on=["uuid", "user_grade"],
        right_on=["uuid", "user_grade"],
    )
    df = pd.merge(df1, df_content, on="ucid").dropna()

    required_columns = [
        "is_correct",
        "level",
        "difficulty",
        "learning_stage",
        "gender",
        "user_grade",
        "has_teacher_cnt",
        "is_self_coach",
        "has_student_cnt",
        "belongs_to_class_cnt",
        "has_class_cnt",
    ]
    df_logistic = df[required_columns]
    cat_columns = df_logistic.select_dtypes(["category"]).columns
    df_logistic[cat_columns] = df_logistic[cat_columns].apply(lambda x: x.cat.codes)

    input_data = df_logistic.to_numpy()
    n = input_data.shape[0]
    num_samples = int(n * 0.8)
    samples = np.random.choice(range(n), num_samples, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[samples] = False

    X_train = input_data[samples, 1:]
    y_train = input_data[samples, 0].astype("int")
    X_eval = input_data[mask, 1:]
    y_eval = input_data[mask, 0].astype("int")

    X_train_scaled = MinMaxScaler().fit_transform(X_train)
    model = LogisticRegression(random_state=0).fit(X_train_scaled, y_train)
    X_eval_scaled = MinMaxScaler().fit_transform(X_eval)
    score = model.score(X_eval_scaled, y_eval)

    return model, score
