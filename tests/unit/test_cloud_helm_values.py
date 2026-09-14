"""Verify the Terraform-to-Helm handoff and credential handling."""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/cloud_helm_values.py"


def test_generated_values_preserve_secrets_and_refuse_overwrite(tmp_path):
    outputs = {
        key: {"value": value}
        for key, value in {
            "cloud_sql_private_ip": "10.0.0.2",
            "task_service_account_email": "task@example",
            "control_service_account_email": "control@example",
            "artifact_bucket": "artifacts",
            "data_lake_bucket": "lake",
            "project_id": "demo",
        }.items()
    }
    password = "space and@slash/plus+"
    args = [sys.executable, str(SCRIPT)]
    env = {**os.environ, "TF_VAR_database_password": password}
    result = subprocess.run(
        args,
        input=json.dumps(outputs),
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert password not in result.stdout + result.stderr
    root = tmp_path / "artifacts/cloud"
    junyi = json.loads((root / "junyi.values.json").read_text())
    flyte = json.loads((root / "flyte.values.json").read_text())
    assert junyi["database"]["url"] == (
        "postgresql://junyi:space%20and%40slash%2Fplus%2B@10.0.0.2:5432/junyi"
    )
    assert flyte["configuration"]["database"]["password"] == password
    assert (
        flyte["flyte-core-components"]["runs"]["database"]["postgres"]["password"]
        == password
    )
    assert (
        flyte["serviceAccount"]["annotations"]["iam.gke.io/gcp-service-account"]
        == "control@example"
    )
    assert (root / "junyi.values.json").stat().st_mode & 0o777 == 0o600
    second = subprocess.run(
        args,
        input=json.dumps(outputs),
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
    )
    assert second.returncode != 0
    assert json.loads((root / "junyi.values.json").read_text()) == junyi
