from junyi_predictor.workflows.preflight import (
    PREFLIGHT_ENVIRONMENT,
    runtime_compatibility,
)
from junyi_predictor.workflows.training import (
    RUNTIME_IMAGE,
    TASK_CONFIG_MAP,
    TASK_SECRET,
    TASK_SERVICE_ACCOUNT,
    feature_env,
    materialize_feature_snapshot,
    materialize_preprocessed,
    pipeline_env,
    preprocess_env,
    train_env,
    train_from_features,
    train_register,
    training_only_env,
    training_pipeline,
)


def test_training_workflow_exposes_separate_task_environments():
    assert preprocess_env is not feature_env
    assert feature_env is not train_env
    assert pipeline_env is not train_env
    assert training_pipeline is not None
    assert not training_pipeline.triggers
    assert materialize_preprocessed.name == "junyi-preprocess.materialize_preprocessed"
    assert (
        materialize_feature_snapshot.name
        == "junyi-features.materialize_feature_snapshot"
    )
    assert train_register.name == "junyi-training.train_register"
    assert train_from_features.name == "junyi-training-only.train_from_features"
    assert training_only_env.depends_on == [train_env]


def test_remote_task_environment_uses_the_fixed_image_and_identity():
    pod_spec = preprocess_env.pod_template.pod_spec
    primary = pod_spec.containers[0]

    assert RUNTIME_IMAGE._ref_name == "runtime"
    assert preprocess_env.image is RUNTIME_IMAGE
    assert preprocess_env.resources.cpu == "500m"
    assert preprocess_env.resources.memory == "3Gi"
    assert preprocess_env.resources.disk == "2Gi"
    assert feature_env.resources.cpu == "1"
    assert feature_env.resources.disk == "3Gi"
    assert pod_spec.service_account_name == TASK_SERVICE_ACCOUNT
    assert primary.env_from[0].config_map_ref.name == TASK_CONFIG_MAP
    assert primary.env_from[1].secret_ref.name == TASK_SECRET


def test_preflight_uses_the_production_task_runtime_contract():
    pod_spec = PREFLIGHT_ENVIRONMENT.pod_template.pod_spec

    assert PREFLIGHT_ENVIRONMENT.image is RUNTIME_IMAGE
    assert PREFLIGHT_ENVIRONMENT.name == "junyi-runtime-preflight"
    assert PREFLIGHT_ENVIRONMENT.resources.cpu == "250m"
    assert pod_spec.service_account_name == TASK_SERVICE_ACCOUNT
    assert runtime_compatibility.name == "junyi-runtime-preflight.runtime_compatibility"
