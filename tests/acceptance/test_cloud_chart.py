"""Check rendered Kubernetes contracts without a cluster or cloud credentials."""

import shutil
import subprocess

import pytest
import yaml

from junyi_predictor.workflows.training import (
    TASK_CONFIG_MAP,
    TASK_SECRET,
    TASK_SERVICE_ACCOUNT,
)


def render(*extra: str) -> subprocess.CompletedProcess:
    if not shutil.which("helm"):
        pytest.skip("Helm is required for chart rendering")
    return subprocess.run(
        [
            "helm",
            "template",
            "junyi",
            "infra/helm/junyi-cloud",
            "--namespace",
            "flyte",
            "--set",
            "serviceAccount.googleEmail=task@example.iam.gserviceaccount.com",
            "--set",
            "config.artifactBucket=artifacts",
            "--set",
            "config.dataLakeBucket=lake",
            "--set",
            "database.url=postgresql://fixture",
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_chart_preserves_workflow_contract_and_quota():
    result = render()
    assert result.returncode == 0, result.stderr
    objects = {item["kind"]: item for item in yaml.safe_load_all(result.stdout)}
    assert "Job" not in objects
    assert objects["ServiceAccount"]["metadata"]["name"] == TASK_SERVICE_ACCOUNT
    assert objects["ConfigMap"]["metadata"]["name"] == TASK_CONFIG_MAP
    assert objects["Secret"]["metadata"]["name"] == TASK_SECRET
    assert objects["ResourceQuota"]["spec"]["hard"] == {
        "requests.cpu": "4",
        "requests.memory": "12Gi",
        "requests.ephemeral-storage": "12Gi",
        "pods": "8",
    }


def test_seeder_requires_digest_and_uses_runtime_references():
    invalid = render(
        "--set", "seeder.enabled=true", "--set", "seeder.image=runtime:latest"
    )
    assert invalid.returncode != 0
    result = render(
        "--set",
        "seeder.enabled=true",
        "--set",
        "seeder.image=runtime@sha256:" + "a" * 64,
    )
    assert result.returncode == 0, result.stderr
    job = next(
        item for item in yaml.safe_load_all(result.stdout) if item["kind"] == "Job"
    )
    pod = job["spec"]["template"]["spec"]
    assert job["spec"]["backoffLimit"] == 0
    assert pod["serviceAccountName"] == TASK_SERVICE_ACCOUNT
    assert pod["containers"][0]["envFrom"] == [
        {"configMapRef": {"name": TASK_CONFIG_MAP}},
        {"secretRef": {"name": TASK_SECRET}},
    ]


def test_existing_secret_is_not_rendered():
    result = render("--set", "database.existingSecret=true", "--set", "database.url=")
    assert result.returncode == 0, result.stderr
    assert all(item["kind"] != "Secret" for item in yaml.safe_load_all(result.stdout))
