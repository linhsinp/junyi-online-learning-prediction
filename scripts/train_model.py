import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# TODO: Set to a small number for quick testing to prevent from overflowing the RAM limit
NUM_SAMPLES = (
    10000  # round((df_log.shape[0] / 100))  # 1% of the data for quick testing
)

# Path
PATH_FEATURE_STORE = "data/feature_store"
PATH_PREPROCESSED_INPUT = "data/experiment"
PATH_MODEL = "model"

# Files
FILE_USER_PROCESSED = os.path.join(
    PATH_PREPROCESSED_INPUT, "Processed_Info_UserData_train.parquet.gzip"
)
FILE_CONTENT_PROCESSED = os.path.join(
    PATH_PREPROCESSED_INPUT, "Processed_Info_Content_train.parquet.gzip"
)
FILE_LOG = os.path.join(PATH_FEATURE_STORE, "df_log_with_upid_acc.parquet.gzip")

# Feature files
FILE_M_CONCEPT_PROFICIENCY = os.path.join(
    PATH_FEATURE_STORE, "m_concept_proficiency.npz"
)
FILE_M_PROFICIENCY_LEVEL4 = os.path.join(PATH_FEATURE_STORE, "m_proficiency_level4.npz")


def load_data_into_df(path_data: str) -> pd.DataFrame:
    """Load raw data into pandas dataframe.

    Args:
        path_data (str): Path to feature engineered file

    Returns:
        The DataFrames containing log data / user information / content information.
    """
    return pd.read_parquet(path_data)


def load_matrices(path_m_concept_proficiency, path_m_proficiency_level4):
    m_concept_proficiency = np.load(path_m_concept_proficiency)["arr_0"]
    m_proficiency_level4 = np.load(path_m_proficiency_level4)["arr_0"]
    return m_concept_proficiency, m_proficiency_level4


def train_benchmark_model(
    df_log: pd.DataFrame, df_user: pd.DataFrame, df_content: pd.DataFrame
) -> tuple[LogisticRegression, float]:
    # Join tables based on uuid and ucid
    df1 = pd.merge(
        df_log,
        df_user,
        how="inner",
        left_on=["uuid", "user_grade"],
        right_on=["uuid", "user_grade"],
    )  # NOTE: user_grade is duplicated in both tables
    df = pd.merge(df1, df_content, on="ucid")
    df = df.dropna()

    # Select only required columns
    required_columns = [
        "is_correct",  # y
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

    # Convert DataFrame to numpy array
    input_data = df_logistic.to_numpy()
    n = input_data.shape[0]

    # Split the data into 80 - 20% split for training and testing
    num_samples = int(n * 0.8)
    samples = np.random.choice(range(n), num_samples, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[samples] = False

    X_train = input_data[samples, 1:]
    y_train = input_data[samples, 0]
    y_train = y_train.astype("int")
    X_eval = input_data[mask, 1:]
    y_eval = input_data[mask, 0]
    y_eval = y_eval.astype("int")

    print("X_train shape is = ", np.shape(X_train))
    print("y_train shape is = ", np.shape(y_train))
    print("X_eval shape is = ", np.shape(X_eval))
    print("y_eval shape is = ", np.shape(y_eval))

    X_train_scaled = MinMaxScaler().fit_transform(X_train)
    model = LogisticRegression(random_state=0).fit(X_train_scaled, y_train)

    X_eval_scaled = MinMaxScaler().fit_transform(X_eval)
    score = model.score(X_eval_scaled, y_eval)

    return model, score


def split_data_for_train_and_test(
    df_log: pd.DataFrame,
    m_concept_proficiency: np.array,
    m_proficiency_level4: np.array,
    num_samples: int = NUM_SAMPLES,
) -> tuple[np.array, np.array, np.array, np.array]:
    """Split data for train and test.

    This function append variables from feature stores and perform 80%-20% split on data.

    Args:
        df_log (pd.DataFrame): Preprocessed log data with UPID accuracy.
        num_samples (int): Self-defined number of samples.

    Returns:
        tuple[np.array, np.array, np.array, np.array]: train and test sets
    """

    if not num_samples:
        num_samples = df_log.shape[0]

    np.random.seed(760)
    num_train_samples = int(num_samples * 0.8)
    samples_train = np.random.choice(
        range(num_samples), num_train_samples, replace=False
    )
    mask_train = np.zeros(num_samples, dtype=bool)
    mask_train[samples_train] = True

    X_train = np.concatenate(
        (
            # grade
            df_log.head(num_samples)
            .loc[mask_train, "user_grade"]
            .to_numpy()[:, np.newaxis],
            # gender
            df_log.head(num_samples)
            .loc[mask_train, ["female", "male", "unspecified"]]
            .to_numpy(),
            # Difficulty features
            df_log.head(num_samples).loc[mask_train, ["v_upid_acc"]].to_numpy(),
            # History features
            df_log.head(num_samples).loc[mask_train, "level"].to_numpy()[:, np.newaxis],
            df_log.head(num_samples)
            .loc[mask_train, "problem_number"]
            .to_numpy()[:, np.newaxis],
            df_log.head(num_samples)
            .loc[mask_train, "exercise_problem_repeat_session"]
            .to_numpy()[:, np.newaxis],
            # concept proficiency matrix
            m_concept_proficiency[:num_samples, :][mask_train, :],
            # level-4 proficiency matrix
            m_proficiency_level4[:num_samples, :][mask_train, :],
        ),
        axis=1,
    )
    y_train = (
        df_log.head(num_samples).loc[mask_train, "is_correct"].to_numpy(dtype=bool)
    )

    X_test = np.concatenate(
        (
            # grade
            df_log.head(num_samples)
            .loc[~mask_train, "user_grade"]
            .to_numpy()[:, np.newaxis],
            # gender
            df_log.head(num_samples)
            .loc[~mask_train, ["female", "male", "unspecified"]]
            .to_numpy(),
            # Difficulty features
            df_log.head(num_samples).loc[~mask_train, ["v_upid_acc"]].to_numpy(),
            # History features
            df_log.head(num_samples)
            .loc[~mask_train, "level"]
            .to_numpy()[:, np.newaxis],
            df_log.head(num_samples)
            .loc[~mask_train, "problem_number"]
            .to_numpy()[:, np.newaxis],
            df_log.head(num_samples)
            .loc[~mask_train, "exercise_problem_repeat_session"]
            .to_numpy()[:, np.newaxis],
            # concept proficiency matrix
            m_concept_proficiency[:num_samples, :][~mask_train, :],
            # level-4 proficiency matrix
            m_proficiency_level4[:num_samples, :][~mask_train, :],
        ),
        axis=1,
    )
    y_test = (
        df_log.head(num_samples).loc[~mask_train, "is_correct"].to_numpy(dtype=bool)
    )

    print("X_train shape is = ", np.shape(X_train))
    print("y_train shape is = ", np.shape(y_train))
    print("X_test shape is = ", np.shape(X_test))
    print("y_test shape is = ", np.shape(y_test))

    return X_train, y_train, X_test, y_test


def apply_min_max_transformation(
    X_train: np.array, X_test: np.array
) -> tuple[np.array, np.array]:
    """Overwrite the raw data matrix to reduce RAM usage."""
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)
    return X_train, X_test


def train_and_evaluate_model(
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    model_type: str,
    model_path: str = PATH_MODEL,
) -> dict:
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

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"# Model type = {model_type}", ": ", end="")
    print("train = " + str(train_score) + " ; ", end="")
    print("test = " + str(test_score))

    metrics = {
        "train_score": train_score,
        "test_score": test_score,
    }

    # os.makedirs(model_path, exist_ok=True)
    # with open(f"{model_path}/{model_type}.pkl", "wb") as f:
    #     dump(model, f, protocol=5)
    # with open(f"{model_path}/{model_type}_metrics.json", 'w', encoding='utf-8') as f:
    #     json.dump(metrics, f, ensure_ascii=False, indent=4)

    return metrics


if __name__ == "__main__":
    df_log = load_data_into_df(FILE_LOG)
    df_user = load_data_into_df(FILE_USER_PROCESSED)
    df_content = load_data_into_df(FILE_CONTENT_PROCESSED)
    m_concept_proficiency, m_proficiency_level4 = load_matrices(
        FILE_M_CONCEPT_PROFICIENCY, FILE_M_PROFICIENCY_LEVEL4
    )

    benchmark_model, benchmark_score = train_benchmark_model(
        df_log, df_user, df_content
    )  # Accuracy (n = 3M) = 0.69 %

    X_train, y_train, X_test, y_test = split_data_for_train_and_test(
        df_log, m_concept_proficiency, m_proficiency_level4
    )
    X_train, X_test = apply_min_max_transformation(X_train, X_test)
    metrics_1 = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, model_type="DecisionTreeClassifier"
    )
    metrics_2 = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, model_type="GradientBoostingClassifier"
    )
    metrics_3 = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, model_type="LogisticRegression_L2"
    )
    metrics_4 = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, model_type="LogisticRegression_L1"
    )

    print("Model training and evaluation completed and saved successfully.")
