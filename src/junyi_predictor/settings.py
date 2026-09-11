"""Runtime settings shared by local and cloud pipeline execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
