"""Generate private Helm values from non-secret Terraform outputs and a password."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import quote


def build_values(outputs: dict, password: str) -> tuple[dict, dict]:
    """Map cloud outputs to software configuration without logging credentials."""

    def value(key: str) -> str:
        return outputs[key]["value"]

    database = {
        "host": value("cloud_sql_private_ip"),
        "port": 5432,
        "dbname": "flyte",
        "username": "flyte",
        "password": password,
        "options": "sslmode=disable",
    }
    junyi = {
        "serviceAccount": {"googleEmail": value("task_service_account_email")},
        "config": {
            "artifactBucket": value("artifact_bucket"),
            "dataLakeBucket": value("data_lake_bucket"),
        },
        "database": {
            "url": f"postgresql://junyi:{quote(password, safe='')}@{value('cloud_sql_private_ip')}:5432/junyi",
        },
    }
    flyte = {
        "serviceAccount": {
            "annotations": {
                "iam.gke.io/gcp-service-account": value(
                    "control_service_account_email"
                ),
            },
        },
        "configuration": {
            "database": database,
            "storage": {
                "metadataContainer": value("artifact_bucket"),
                "userDataContainer": value("artifact_bucket"),
                "providerConfig": {"gcs": {"project": value("project_id")}},
            },
        },
        "flyte-core-components": {
            "runs": {
                "database": {"postgres": database},
                "storagePrefix": f"gs://{value('artifact_bucket')}/flyte",
            },
        },
    }
    return junyi, flyte


def main() -> None:
    """Write JSON (valid Helm values) with owner-only file permissions."""
    junyi, flyte = build_values(
        json.load(sys.stdin), os.environ["TF_VAR_database_password"]
    )
    directory = Path("artifacts/cloud")
    directory.mkdir(parents=True, exist_ok=True)
    for name, values in (("junyi", junyi), ("flyte", flyte)):
        path = directory / f"{name}.values.json"
        # Refuse accidental overwrites of an existing deployment's configuration.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(values, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
