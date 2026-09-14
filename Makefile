UV := uv

DATABASE_URL ?= postgresql://junyi:junyi-local-only@localhost:30001/junyi
export DATABASE_URL

START_DATE ?= 2019-06-01T00:00:00
END_DATE ?= 2019-06-10T00:00:00
NUM_SAMPLES ?= 1000
TRAIN_FRACTION ?= 0.70
VALIDATION_FRACTION ?= 0.15

.PHONY: help test lint helm-lint helm-template flyte-training-local flyte-training-local-tui flyte-train-from-features-local postgres-local kind-create download-data materialize-parquet upload-curated-data upload-dimension-data reset-local seed-local

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
	@echo "  make upload-curated-data DATA_LAKE_BUCKET=..."
	@echo "  make upload-dimension-data DATA_LAKE_BUCKET=..."
	@echo "  make reset-local DATABASE_URL=..."
	@echo "  make seed-local DATABASE_URL=..."
	@echo "  make flyte-training-local START_DATE=... END_DATE=..."
	@echo "  make flyte-training-local-tui START_DATE=... END_DATE=..."
	@echo "  make flyte-train-from-features-local FEATURE_SNAPSHOT_KEY=runs/<source-run-id>/feature_snapshot.json"

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check src tests scripts

helm-lint:
	helm lint ./infra/helm/local-postgres
	helm lint infra/helm/junyi-cloud -f infra/helm/junyi-cloud/values-test.yaml

helm-template:
	helm template junyi ./infra/helm/local-postgres >/dev/null
	helm template junyi-cloud infra/helm/junyi-cloud --namespace flyte -f infra/helm/junyi-cloud/values-test.yaml >/dev/null

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

upload-curated-data:
	@test -n "$(DATA_LAKE_BUCKET)" || { echo "DATA_LAKE_BUCKET is required"; exit 2; }
	DATA_LAKE_BACKEND=gcs DATA_LAKE_BUCKET="$(DATA_LAKE_BUCKET)" DATA_LAKE_PREFIX="$${DATA_LAKE_PREFIX:-data/curated/log_problem}" PYTHONPATH=src $(UV) run python -m junyi_predictor.cli upload-curated-data

upload-dimension-data:
	@test -n "$(DATA_LAKE_BUCKET)" || { echo "DATA_LAKE_BUCKET is required"; exit 2; }
	DATA_LAKE_BACKEND=gcs DATA_LAKE_BUCKET="$(DATA_LAKE_BUCKET)" DIMENSION_DATA_PREFIX="$${DIMENSION_DATA_PREFIX:-data/dimensions}" PYTHONPATH=src $(UV) run python -m junyi_predictor.cli upload-dimension-data

reset-local:
	DATABASE_URL="$(DATABASE_URL)" PYTHONPATH=src $(UV) run python -m junyi_predictor.cli reset-db

seed-local:
	DATABASE_URL="$(DATABASE_URL)" PYTHONPATH=src $(UV) run python -m junyi_predictor.cli seed-db

flyte-training-local:
	@set -a; [ -f .env ] && . ./.env; set +a; \
	JUNYI_LOG_FORMAT="$${JUNYI_LOG_FORMAT:-text}" JUNYI_LOG_DIR="$${JUNYI_LOG_DIR:-artifacts/logs}" \
	ARTIFACT_BACKEND=local ARTIFACT_ROOT=artifacts/runs \
	PYTHONPATH=src $(UV) run flyte run --local src/junyi_predictor/workflows/training.py training_pipeline \
		--start_date "$(START_DATE)" --end_date "$(END_DATE)" \
		--train_fraction "$(TRAIN_FRACTION)" --validation_fraction "$(VALIDATION_FRACTION)"

flyte-training-local-tui:
	@set -a; [ -f .env ] && . ./.env; set +a; \
	printf 'Application logs: %s (follow *.jsonl in another terminal)\n' "$${JUNYI_LOG_DIR:-artifacts/logs}"; \
	JUNYI_LOG_CONSOLE=0 JUNYI_LOG_DIR="$${JUNYI_LOG_DIR:-artifacts/logs}" \
	ARTIFACT_BACKEND=local ARTIFACT_ROOT=artifacts/runs \
	PYTHONPATH=src $(UV) run flyte run --local --tui src/junyi_predictor/workflows/training.py training_pipeline \
		--start_date "$(START_DATE)" --end_date "$(END_DATE)" \
		--train_fraction "$(TRAIN_FRACTION)" --validation_fraction "$(VALIDATION_FRACTION)"

# Separate date variables deliberately avoid the combined workflow's sample window.
flyte-train-from-features-local:
	@test -n "$(FEATURE_SNAPSHOT_KEY)" || { echo "FEATURE_SNAPSHOT_KEY is required"; exit 2; }
	@set -a; [ -f .env ] && . ./.env; set +a; \
	set -- --feature_snapshot_key "$(FEATURE_SNAPSHOT_KEY)" \
		--train_fraction "$(TRAIN_FRACTION)" --validation_fraction "$(VALIDATION_FRACTION)"; \
	if [ -n "$(TRAINING_RUN_ID)" ]; then set -- "$$@" --training_run_id "$(TRAINING_RUN_ID)"; fi; \
	if [ -n "$(FEATURE_START_DATE)" ]; then set -- "$$@" --start_date "$(FEATURE_START_DATE)"; fi; \
	if [ -n "$(FEATURE_END_DATE)" ]; then set -- "$$@" --end_date "$(FEATURE_END_DATE)"; fi; \
	JUNYI_LOG_FORMAT="$${JUNYI_LOG_FORMAT:-text}" JUNYI_LOG_DIR="$${JUNYI_LOG_DIR:-artifacts/logs}" \
	ARTIFACT_BACKEND=local ARTIFACT_ROOT="$${ARTIFACT_ROOT:-artifacts/runs}" \
	PYTHONPATH=src $(UV) run flyte run --local src/junyi_predictor/workflows/training.py train_from_features "$$@"
