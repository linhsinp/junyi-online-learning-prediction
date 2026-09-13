"""Read existing v1 feature publications into isolated temporary files."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from junyi_predictor.contracts import FeatureSnapshot
from junyi_predictor.progress import operation
from junyi_predictor.registry import validate_training_run_id
from junyi_predictor.storage.artifacts import ArtifactStore


@dataclass(frozen=True)
class LoadedFeatureSnapshot:
    """Private copies of the exact manifest and data consumed by one run."""

    snapshot: FeatureSnapshot
    manifest: dict
    log_path: Path
    concept_path: Path
    level4_path: Path
    fingerprints: dict[str, str]


def file_sha256(path: Path) -> str:
    """Fingerprint a file without allocating another full in-memory copy."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def load_feature_snapshot(
    store: ArtifactStore, manifest_key: str
) -> Iterator[LoadedFeatureSnapshot]:
    """Read a published v1 snapshot using its producer's canonical artifact keys.

    URI fields remain provenance; store-relative keys allow retained artifacts to
    be relocated between local/GCS roots without rewriting the original manifest.
    All files are copied before loading so later reads use the fingerprinted bytes.
    """
    key = PurePosixPath(manifest_key)
    if (
        not manifest_key
        or key.is_absolute()
        or ".." in key.parts
        or "\\" in manifest_key
        or ":" in manifest_key
        or key.as_posix() != manifest_key
    ):
        raise ValueError("feature_snapshot_key must be a normalized store-relative key")
    with tempfile.TemporaryDirectory(prefix="junyi-training-input-") as temporary:
        root = Path(temporary)
        with operation("features.download", feature_snapshot_key=manifest_key):
            manifest_path = store.get_file(manifest_key, root / "manifest.json")
            manifest = json.loads(manifest_path.read_text())
            snapshot = FeatureSnapshot.model_validate(manifest)
            if snapshot.schema_version != "v1":
                raise ValueError("Unsupported feature snapshot schema_version")
            validate_training_run_id(snapshot.training_run_id)
            prefix = f"runs/{snapshot.training_run_id}/features"
            files = {
                "log": store.get_file(f"{prefix}/log.parquet", root / "log.parquet"),
                "concept": store.get_file(
                    f"{prefix}/concept_proficiency.npy", root / "concept.npy"
                ),
                "level4": store.get_file(
                    f"{prefix}/level4_proficiency.npy", root / "level4.npy"
                ),
            }
        with operation("training.fingerprint"):
            fingerprints = {
                name: file_sha256(path)
                for name, path in {"manifest": manifest_path, **files}.items()
            }
        yield LoadedFeatureSnapshot(
            snapshot=snapshot,
            manifest=manifest,
            log_path=files["log"],
            concept_path=files["concept"],
            level4_path=files["level4"],
            fingerprints=fingerprints,
        )
