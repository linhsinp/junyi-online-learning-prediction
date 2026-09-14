"""Check rendered Kubernetes contracts without a cluster or cloud credentials."""

import shutil
import subprocess
from pathlib import Path

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


def test_local_preflight_overlay_targets_only_local_dependencies():
    values_path = Path("infra/helm/flyte/values-local-preflight.yaml")
    values = yaml.safe_load(values_path.read_text())

    postgres = values["flyte-core-components"]["runs"]["database"]["postgres"]
    storage = values["configuration"]["storage"]
    assert (
        postgres["host"] == "flyte-postgres-postgres.flyte-preflight.svc.cluster.local"
    )
    assert postgres["dbname"] == "flyte"
    assert storage["provider"] == "s3"
    assert storage["metadataContainer"] == "flyte-data"
    assert storage["providerConfig"]["s3"]["endpoint"] == "host.docker.internal:9000"
    assert values["rbac"]["extraRules"] == [
        {"apiGroups": [""], "resources": ["namespaces"], "verbs": ["get"]},
        {
            "apiGroups": ["events.k8s.io"],
            "resources": ["events"],
            "verbs": ["get", "list", "watch"],
        },
        {"apiGroups": [""], "resources": ["secrets"], "verbs": ["delete"]},
    ]


def test_local_preflight_auxiliary_pods_have_quota_requests():
    manifests = (
        Path("infra/local/flyte-preflight/minio-deployment.yaml"),
        Path("infra/local/flyte-preflight/minio-bucket-job.yaml"),
    )

    for manifest in manifests:
        pod_spec = yaml.safe_load(manifest.read_text())["spec"]
        while "template" in pod_spec:
            pod_spec = pod_spec["template"]["spec"]
        requests = pod_spec["containers"][0]["resources"]["requests"]
        assert requests["cpu"]
        assert requests["memory"]
