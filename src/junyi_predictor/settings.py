"""Runtime settings shared by local and cloud pipeline execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from junyi_predictor.paths import CURATED_LOG_DIR


@dataclass(frozen=True)
class ArtifactSettings:
    """Artifact-only configuration for training without a database connection."""

    artifact_backend: str
    artifact_root: str
    gcs_bucket: str | None

    @classmethod
    def from_environment(cls) -> "ArtifactSettings":
        return cls(
            artifact_backend=os.getenv("ARTIFACT_BACKEND", "local"),
            artifact_root=os.getenv("ARTIFACT_ROOT", "artifacts/runs"),
            gcs_bucket=os.getenv("GCS_BUCKET"),
        )


@dataclass(frozen=True)
class DataLakeSettings:
    """Location of curated event partitions consumed by preprocessing."""

    backend: str
    curated_log_root: str
    gcs_bucket: str | None
    gcs_prefix: str

    @classmethod
    def from_environment(cls) -> "DataLakeSettings":
        return cls(
            backend=os.getenv("DATA_LAKE_BACKEND", "local"),
            curated_log_root=os.getenv("CURATED_LOG_ROOT", str(CURATED_LOG_DIR)),
            gcs_bucket=os.getenv("DATA_LAKE_BUCKET"),
            gcs_prefix=os.getenv("DATA_LAKE_PREFIX", "data/curated/log_problem"),
        )


@dataclass(frozen=True)
class Settings:
    """Read environment-backed configuration without leaking it into stages."""

    database_url: str
    artifact_backend: str
    artifact_root: str
    gcs_bucket: str | None

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            database_url=os.environ["DATABASE_URL"],
            artifact_backend=os.getenv("ARTIFACT_BACKEND", "local"),
            artifact_root=os.getenv("ARTIFACT_ROOT", "artifacts/runs"),
            gcs_bucket=os.getenv("GCS_BUCKET"),
        )

    @property
    def local_artifact_root(self) -> Path:
        return Path(self.artifact_root)
