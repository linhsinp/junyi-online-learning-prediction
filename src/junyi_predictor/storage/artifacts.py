"""Artifact stores for local development and Google Cloud Storage."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Protocol

from google.cloud import storage

from junyi_predictor.progress import timed


class ArtifactStore(Protocol):
    """Persist files and JSON under a stable run-relative key."""

    def put_file(self, source: Path, key: str) -> str: ...

    def get_file(self, key: str, destination: Path) -> Path: ...

    def put_json(self, payload: dict, key: str) -> str: ...


class LocalArtifactStore:
    """Filesystem-backed artifacts used by tests and Flyte local mode."""

    def __init__(self, root: Path):
        self.root = root

    @timed("artifact.write_file", fields=("key",))
    def put_file(self, source: Path, key: str) -> str:
        destination = self.root / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return str(destination)

    @timed("artifact.read_file", fields=("key",))
    def get_file(self, key: str, destination: Path) -> Path:
        source = self.root / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    @timed("artifact.write_json", fields=("key",))
    def put_json(self, payload: dict, key: str) -> str:
        destination = self.root / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return str(destination)


class GcsArtifactStore:
    """GCS-backed artifacts used by remote Flyte tasks."""

    def __init__(self, bucket_name: str, prefix: str = ""):
        self.bucket = storage.Client().bucket(bucket_name)
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    @timed("artifact.write_file", fields=("key",))
    def put_file(self, source: Path, key: str) -> str:
        blob_key = self._key(key)
        self.bucket.blob(blob_key).upload_from_filename(source)
        return f"gs://{self.bucket.name}/{blob_key}"

    @timed("artifact.read_file", fields=("key",))
    def get_file(self, key: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.bucket.blob(self._key(key)).download_to_filename(destination)
        return destination

    @timed("artifact.write_json", fields=("key",))
    def put_json(self, payload: dict, key: str) -> str:
        blob_key = self._key(key)
        self.bucket.blob(blob_key).upload_from_string(
            json.dumps(payload, indent=2, sort_keys=True),
            content_type="application/json",
        )
        return f"gs://{self.bucket.name}/{blob_key}"


def create_artifact_store(
    backend: str, root: str, bucket_name: str | None
) -> ArtifactStore:
    """Create the configured store and reject incomplete cloud configuration."""
    if backend == "local":
        return LocalArtifactStore(Path(root))
    if backend == "gcs" and bucket_name:
        return GcsArtifactStore(bucket_name=bucket_name, prefix=root)
    raise ValueError("GCS artifacts require ARTIFACT_BACKEND=gcs and GCS_BUCKET")
