"""Feature publications load identically from local and cloud stores."""

import json
import shutil
from unittest.mock import MagicMock, Mock

import pytest

from junyi_predictor.pipeline import experiments
from junyi_predictor.storage.artifacts import GcsArtifactStore
from junyi_predictor.storage.feature_snapshots import file_sha256, load_feature_snapshot


@pytest.mark.parametrize("backend", ["local", "gcs"])
def test_snapshot_private_copies_fingerprints_and_cleanup(
    published_feature_snapshot, monkeypatch, backend
):
    local, key = published_feature_snapshot
    store = local
    if backend == "gcs":
        client = MagicMock()
        client.bucket.return_value.name = "fixture"

        def blob(path):
            result = Mock()
            result.download_to_filename.side_effect = lambda target: shutil.copyfile(
                local.root / path.removeprefix("retained/"), target
            )
            return result

        client.bucket.return_value.blob.side_effect = blob
        monkeypatch.setattr(
            "junyi_predictor.storage.artifacts.storage.Client", lambda: client
        )
        store = GcsArtifactStore("fixture", "retained")
    with load_feature_snapshot(store, key) as loaded:
        paths = [loaded.log_path, loaded.concept_path, loaded.level4_path]
        assert all(path.exists() for path in paths)
        assert loaded.fingerprints["log"] == file_sha256(
            local.root / "runs/source-run/features/log.parquet"
        )
        assert loaded.fingerprints["manifest"] == file_sha256(local.root / key)
        assert loaded.snapshot.training_run_id == "source-run"
    assert all(not path.exists() for path in paths)


def test_temporary_downloads_cleaned_on_failure(
    published_feature_snapshot, monkeypatch
):
    store, key = published_feature_snapshot
    original = store.get_file
    destinations = []

    def failing_download(key, destination):
        destinations.append(destination)
        if key.endswith("concept_proficiency.npy"):
            raise OSError("Synthetic transfer failure")
        return original(key, destination)

    monkeypatch.setattr(store, "get_file", failing_download)
    with pytest.raises(OSError):
        with load_feature_snapshot(store, key):
            pytest.fail("Unreachable")
    assert destinations and all(not path.parent.exists() for path in destinations)


@pytest.mark.parametrize(
    "key",
    [
        "",
        "../manifest.json",
        "/manifest.json",
        "runs/../manifest.json",
        "gs://bucket/manifest.json",
        "runs//manifest.json",
    ],
)
def test_invalid_manifest_key_rejected(published_feature_snapshot, key):
    store, _ = published_feature_snapshot
    with pytest.raises(ValueError, match="store-relative"):
        with load_feature_snapshot(store, key):
            pytest.fail("Unreachable")


@pytest.mark.parametrize(
    "problem",
    [
        "missing_manifest",
        "invalid_json",
        "unsupported_schema",
        "invalid_manifest",
        "missing_data",
        "corrupt_parquet",
        "corrupt_array",
        "row_mismatch",
    ],
)
def test_bad_snapshot_fails_before_fit_and_registration(
    published_feature_snapshot, monkeypatch, problem
):
    store, key = published_feature_snapshot
    manifest = store.root / key
    if problem == "missing_manifest":
        manifest.unlink()
    elif problem == "invalid_json":
        manifest.write_text("invalid json")
    elif problem in ("unsupported_schema", "invalid_manifest", "row_mismatch"):
        data = json.loads(manifest.read_text())
        if problem == "unsupported_schema":
            data["schema_version"] = "v99"
        elif problem == "invalid_manifest":
            del data["log_uri"]
        else:
            data["row_count"] = 21
        manifest.write_text(json.dumps(data))
    else:
        name = (
            "log.parquet" if problem == "corrupt_parquet" else "concept_proficiency.npy"
        )
        path = store.root / "runs/source-run/features" / name
        if problem == "missing_data":
            path.unlink()
        else:
            path.write_bytes(b"corrupt")
    fit = Mock(side_effect=AssertionError("Must not fit"))
    monkeypatch.setattr(experiments, "evaluate_candidates", fit)
    with pytest.raises((ValueError, OSError)):
        experiments.train_snapshot(store, key, "bad-input")
    fit.assert_not_called()
    assert not (store.root / "models/bad-input/manifest.json").exists()
    assert not (store.root / "models/approved.json").exists()
