from junyi_predictor.workflows.training import (
    feature_env,
    pipeline_env,
    preprocess_env,
    train_env,
    training_pipeline,
)


def test_training_workflow_exposes_separate_task_environments():
    assert preprocess_env is not feature_env
    assert feature_env is not train_env
    assert pipeline_env is not train_env
    assert training_pipeline is not None
    assert training_pipeline.triggers[0].name == "weekly-training"
