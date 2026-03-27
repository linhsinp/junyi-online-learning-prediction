"""Pipeline stage interfaces for preprocessing, feature engineering, and training."""

from junyi_predictor.pipeline.feature_engineering import (
    FeatureStageOutput,
    build_feature_stage,
    create_concept_proficiency_matrix,
    create_level4_proficiency_matrix,
    create_upid_accuracy_features,
)
from junyi_predictor.pipeline.preprocessing import (
    PreprocessStageOutput,
    load_data_from_database,
    load_raw_dataframes,
    preprocess_log_frame,
    preprocess_stage,
)
from junyi_predictor.pipeline.training import (
    TrainingSplit,
    apply_min_max_transformation,
    load_feature_matrices,
    load_parquet_dataframe,
    split_training_data,
    train_and_evaluate_model,
    train_benchmark_model,
)

__all__ = [
    "FeatureStageOutput",
    "PreprocessStageOutput",
    "TrainingSplit",
    "apply_min_max_transformation",
    "build_feature_stage",
    "create_concept_proficiency_matrix",
    "create_level4_proficiency_matrix",
    "create_upid_accuracy_features",
    "load_data_from_database",
    "load_feature_matrices",
    "load_parquet_dataframe",
    "load_raw_dataframes",
    "preprocess_log_frame",
    "preprocess_stage",
    "split_training_data",
    "train_benchmark_model",
    "train_and_evaluate_model",
]
