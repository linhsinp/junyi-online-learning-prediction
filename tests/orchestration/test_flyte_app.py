from orchestration import flyte_app


def test_flyte_app_exposes_flyte_2_entrypoints():
    assert flyte_app.env is not None
    assert flyte_app.full_pipeline is not None
    assert flyte_app.preprocess_from_database is not None
    assert flyte_app.train_from_gcs is not None
