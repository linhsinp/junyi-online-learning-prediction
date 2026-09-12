UV := uv

DATABASE_URL ?= postgresql://junyi:junyi-local-only@localhost:30001/junyi
export DATABASE_URL

START_DATE ?= 2019-06-01T00:00:00
END_DATE ?= 2019-06-10T00:00:00
NUM_SAMPLES ?= 1000

.PHONY: help test lint helm-lint helm-template flyte-training-local flyte-training-local-tui postgres-local kind-create download-data materialize-parquet reset-local seed-local

help:
	@echo "Available targets:"
	@echo "  $(UV) sync --all-groups"
	@echo "  make test"
	@echo "  make lint"
	@echo "  make helm-lint"
	@echo "  make helm-template"
	@echo "  make kind-create"
	@echo "  make postgres-local"
	@echo "  make download-data"
	@echo "  make materialize-parquet"
	@echo "  make reset-local DATABASE_URL=..."
	@echo "  make seed-local DATABASE_URL=..."
	@echo "  make flyte-training-local START_DATE=... END_DATE=..."
	@echo "  make flyte-training-local-tui START_DATE=... END_DATE=..."

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check src tests

helm-lint:
	helm lint ./infra/helm/local-postgres

helm-template:
	helm template junyi ./infra/helm/local-postgres >/dev/null

kind-create:
	kind create cluster --name junyi --config infra/local/kind.yaml

postgres-local:
	helm upgrade --install junyi-postgres infra/helm/local-postgres \
		--namespace junyi-local --create-namespace
	kubectl rollout status statefulset/junyi-postgres-postgres \
		--namespace junyi-local --timeout=120s

download-data:
	PYTHONPATH=src $(UV) run python -m junyi_predictor.cli download-data

materialize-parquet:
	PYTHONPATH=src $(UV) run python -m junyi_predictor.cli materialize-parquet

reset-local:
	DATABASE_URL="$(DATABASE_URL)" PYTHONPATH=src $(UV) run python -m junyi_predictor.cli reset-db

seed-local:
	DATABASE_URL="$(DATABASE_URL)" PYTHONPATH=src $(UV) run python -m junyi_predictor.cli seed-db

flyte-training-local:
	@set -a; [ -f .env ] && . ./.env; set +a; \
	JUNYI_LOG_FORMAT="$${JUNYI_LOG_FORMAT:-text}" JUNYI_LOG_DIR="$${JUNYI_LOG_DIR:-artifacts/logs}" \
	ARTIFACT_BACKEND=local ARTIFACT_ROOT=artifacts/runs \
	PYTHONPATH=src $(UV) run flyte run --local src/junyi_predictor/workflows/training.py training_pipeline \
		--start_date "$(START_DATE)" --end_date "$(END_DATE)"

flyte-training-local-tui:
	@set -a; [ -f .env ] && . ./.env; set +a; \
	printf 'Application logs: %s (follow *.jsonl in another terminal)\n' "$${JUNYI_LOG_DIR:-artifacts/logs}"; \
	JUNYI_LOG_CONSOLE=0 JUNYI_LOG_DIR="$${JUNYI_LOG_DIR:-artifacts/logs}" \
	ARTIFACT_BACKEND=local ARTIFACT_ROOT=artifacts/runs \
	PYTHONPATH=src $(UV) run flyte run --local --tui src/junyi_predictor/workflows/training.py training_pipeline \
		--start_date "$(START_DATE)" --end_date "$(END_DATE)"
