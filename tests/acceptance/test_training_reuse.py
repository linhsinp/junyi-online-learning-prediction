"""Exercise independent training using the real Flyte local CLI."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from junyi_predictor.storage.feature_snapshots import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def test_two_training_runs_reuse_one_snapshot_without_database(
    published_feature_snapshot, tmp_path
):
    store, key = published_feature_snapshot
    fixture = tmp_path / "training_only.py"
    shutil.copyfile(Path(__file__).parent / "fixtures/training_only.py", fixture)
    source_files = list((store.root / "runs/source-run").rglob("*"))
    fingerprints = {path: file_sha256(path) for path in source_files if path.is_file()}
    store.put_json({"model_version": "existing"}, "models/approved.json")
    approved_before = (store.root / "models/approved.json").read_bytes()
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "ARTIFACT_BACKEND": "local",
        "ARTIFACT_ROOT": str(store.root),
        "JUNYI_LOG_FORMAT": "json",
        "JUNYI_LOG_LEVEL": "INFO",
        "JUNYI_LOG_CONSOLE": "1",
        "JUNYI_LOG_DIR": str(tmp_path / "logs"),
    }
    env.pop("DATABASE_URL", None)
    metadata = []
    for name in ("experiment-one", "experiment-two"):
        result = subprocess.run(
            [
                str(Path(sys.executable).parent / "flyte"),
                "run",
                "--local",
                str(fixture),
                "train_from_features",
                "--feature_snapshot_key",
                key,
                "--training_run_id",
                name,
            ],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        bundle = store.root / "models" / name
        manifest = json.loads((bundle / "manifest.json").read_text())
        assert manifest["training_run_id"] == name
        assert Path(manifest["model_uri"]).exists()
        metadata.append(json.loads(Path(manifest["training_metadata_uri"]).read_text()))
    assert metadata[0]["input_sha256"] == metadata[1]["input_sha256"]
    assert all(item["source_training_run_id"] == "source-run" for item in metadata)
    assert all(item["selection"]["selected_row_count"] == 20 for item in metadata)
    assert (store.root / "models/approved.json").read_bytes() == approved_before
    assert {path: file_sha256(path) for path in fingerprints} == fingerprints
    events = [
        json.loads(line)
        for path in (tmp_path / "logs").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    starts = [event for event in events if event["event"] == "execution.started"]
    assert {event["stage"] for event in starts} == {"pipeline", "training"}
    assert {event["training_run_id"] for event in starts} == {
        "experiment-one",
        "experiment-two",
    }
    inputs = [event for event in events if event["event"] == "training.inputs"]
    assert len(inputs) == 2
    assert all(event["source_training_run_id"] == "source-run" for event in inputs)
