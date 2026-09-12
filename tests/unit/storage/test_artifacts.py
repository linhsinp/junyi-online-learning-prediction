from pathlib import Path

from junyi_predictor.storage.artifacts import LocalArtifactStore, create_artifact_store


def test_local_artifact_store_persists_files_and_json(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "runs")
    source = tmp_path / "source.txt"
    source.write_text("feature-data")

    uri = store.put_file(source, "runs/run-1/features.txt")
    manifest_uri = store.put_json(
        {"training_run_id": "run-1"}, "runs/run-1/manifest.json"
    )
    downloaded = store.get_file("runs/run-1/features.txt", tmp_path / "download.txt")

    assert Path(uri).read_text() == "feature-data"
    assert Path(manifest_uri).exists()
    assert downloaded.read_text() == "feature-data"


def test_create_artifact_store_rejects_incomplete_gcs_configuration(tmp_path: Path):
    try:
        create_artifact_store("gcs", str(tmp_path), None)
    except ValueError as error:
        assert "GCS artifacts" in str(error)
    else:
        raise AssertionError("Expected incomplete GCS configuration to fail")
