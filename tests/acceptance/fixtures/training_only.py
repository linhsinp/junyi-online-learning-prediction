"""Real Flyte entrypoint with upstream and database access forbidden."""

from junyi_predictor.workflows import training


def forbidden(*args, **kwargs):
    raise AssertionError(
        "Standalone training must not execute upstream or database code"
    )


training.materialize_preprocessed = forbidden
training.materialize_feature_snapshot = forbidden
training.load_data_for_training = forbidden
training.build_feature_stage = forbidden
training.create_engine = forbidden
train_from_features = training.train_from_features
