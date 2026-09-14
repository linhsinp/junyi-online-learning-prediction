"""A minimal remote task used to verify the pinned Flyte backend locally."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import flyte

from junyi_predictor.workflows.training import RUNTIME_IMAGE, _task_pod_template

PREFLIGHT_ENVIRONMENT = flyte.TaskEnvironment(
    name="junyi-runtime-preflight",
    image=RUNTIME_IMAGE,
    resources=flyte.Resources(cpu="250m", memory="512Mi", disk="512Mi"),
    pod_template=_task_pod_template(),
)


def _service_account_name() -> str:
    """Read the Kubernetes service-account name without returning its token."""
    token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    if not token_path.exists():
        return "not-mounted"

    try:
        payload = token_path.read_text().split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return (
            claims.get("kubernetes.io", {})
            .get("serviceaccount", {})
            .get("name", "unknown")
        )
    except (IndexError, UnicodeDecodeError, ValueError):
        return "unreadable"


@PREFLIGHT_ENVIRONMENT.task
async def runtime_compatibility() -> dict[str, str]:
    """Report non-sensitive evidence that the scheduled Junyi pod is configured."""
    return {
        "flyte_sdk_version": flyte.__version__,
        "task_service_account": _service_account_name(),
        "task_config_injected": str("GCS_BUCKET" in os.environ).lower(),
        "task_secret_injected": str("DATABASE_URL" in os.environ).lower(),
    }
