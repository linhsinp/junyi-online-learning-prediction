UV := uv

DATABASE_URL ?= postgresql://junyi:junyi-local-only@localhost:30001/junyi
export DATABASE_URL

START_DATE ?= 2019-06-01T00:00:00
END_DATE ?= 2019-06-10T00:00:00
NUM_SAMPLES ?= 1000
TRAIN_FRACTION ?= 0.70
VALIDATION_FRACTION ?= 0.15

FLYTE_PREFLIGHT_NAMESPACE := flyte-preflight
FLYTE_PREFLIGHT_IMAGE ?= junyi-runtime:preflight
FLYTE_PREFLIGHT_VERSION ?= local-preflight
FLYTE_PREFLIGHT_PROJECT := flyte-preflight
FLYTE_PREFLIGHT_DOMAIN := development
FLYTE_CHART_VERSION := $(shell tr -d '\n' < infra/helm/flyte/chart-version)

.PHONY: help test lint helm-lint helm-template flyte-training-local flyte-training-local-tui flyte-train-from-features-local postgres-local kind-create download-data materialize-parquet upload-curated-data upload-dimension-data reset-local seed-local flyte-backend-preflight-image flyte-backend-preflight-up flyte-backend-preflight-run flyte-backend-preflight-status flyte-backend-preflight-down

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
	@echo "  make flyte-backend-preflight-image"
	@echo "  make flyte-backend-preflight-up"
	@echo "  make flyte-backend-preflight-run"
	@echo "  make flyte-backend-preflight-status"
	@echo "  make flyte-backend-preflight-down"

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

# This stack is deliberately isolated from junyi-local.  It validates the
# pinned Flyte chart/SDK/task-image boundary without altering local app data.
flyte-backend-preflight-image:
	docker build --platform linux/amd64 --tag "$(FLYTE_PREFLIGHT_IMAGE)" --file infra/docker/Dockerfile .
	kind load docker-image "$(FLYTE_PREFLIGHT_IMAGE)" --name junyi

flyte-backend-preflight-up:
	@kubectl config current-context | grep -qx 'kind-junyi' || { echo "Current context must be kind-junyi"; exit 2; }
	kubectl create namespace "$(FLYTE_PREFLIGHT_NAMESPACE)" --dry-run=client -o yaml | kubectl apply -f -
	helm upgrade --install flyte-postgres infra/helm/local-postgres \
		--namespace "$(FLYTE_PREFLIGHT_NAMESPACE)" \
		--set auth.database=flyte --set auth.username=flyte \
		--set-string auth.password=flyte-preflight-local-only \
		--set service.nodePort=30002
	kubectl delete job minio-create-flyte-data --namespace "$(FLYTE_PREFLIGHT_NAMESPACE)" --ignore-not-found
	kubectl wait --for=delete job/minio-create-flyte-data --namespace "$(FLYTE_PREFLIGHT_NAMESPACE)" --timeout=60s 2>/dev/null || true
	kubectl apply -f infra/local/flyte-preflight/
	kubectl rollout status deployment/minio --namespace "$(FLYTE_PREFLIGHT_NAMESPACE)" --timeout=120s
	kubectl wait --for=condition=complete job/minio-create-flyte-data --namespace "$(FLYTE_PREFLIGHT_NAMESPACE)" --timeout=120s
	helm upgrade --install junyi-preflight infra/helm/junyi-cloud \
		--namespace "$(FLYTE_PREFLIGHT_NAMESPACE)" \
		--set serviceAccount.googleEmail=local-preflight@example.invalid \
		--set config.artifactBucket=flyte-data --set config.dataLakeBucket=not-used-by-preflight \
		--set-string database.url=postgresql://flyte:flyte-preflight-local-only@flyte-postgres-postgres.$(FLYTE_PREFLIGHT_NAMESPACE).svc.cluster.local:5432/flyte \
		--set quota.cpu=6 --set quota.memory=8Gi --set quota.ephemeralStorage=8Gi --set quota.pods=12
	helm upgrade --install flyte-preflight \
		"https://flyteorg.github.io/flyte/flyte-binary-$(FLYTE_CHART_VERSION).tgz" \
		--namespace "$(FLYTE_PREFLIGHT_NAMESPACE)" \
		-f infra/helm/flyte/values-local-preflight.yaml --wait --timeout 10m

flyte-backend-preflight-run:
	@set -eu; \
		kubectl -n "$(FLYTE_PREFLIGHT_NAMESPACE)" port-forward service/flyte-preflight-flyte-binary-http 8090:8090 >/tmp/junyi-flyte-preflight-port-forward.log 2>&1 & pid=$$!; \
		kubectl -n "$(FLYTE_PREFLIGHT_NAMESPACE)" port-forward --address 0.0.0.0 service/minio 9000:9000 >/tmp/junyi-minio-preflight-port-forward.log 2>&1 & minio_pid=$$!; \
		trap 'kill $$pid $$minio_pid 2>/dev/null || true' EXIT; \
		until grep -q 'Forwarding from' /tmp/junyi-flyte-preflight-port-forward.log; do sleep 1; done; \
		until grep -q 'Forwarding from' /tmp/junyi-minio-preflight-port-forward.log; do sleep 1; done; \
		PYTHONPATH=src $(UV) run flyte --endpoint localhost:8090 --insecure get project "$(FLYTE_PREFLIGHT_PROJECT)" >/dev/null 2>&1 || \
			PYTHONPATH=src $(UV) run flyte --endpoint localhost:8090 --insecure create project \
				--id "$(FLYTE_PREFLIGHT_PROJECT)" --name "Junyi local preflight"; \
		PYTHONPATH=src $(UV) run flyte --endpoint localhost:8090 --insecure deploy \
			--project "$(FLYTE_PREFLIGHT_PROJECT)" --domain "$(FLYTE_PREFLIGHT_DOMAIN)" \
			--image "runtime=$(FLYTE_PREFLIGHT_IMAGE)" --version "$(FLYTE_PREFLIGHT_VERSION)" \
			src/junyi_predictor/workflows/preflight.py PREFLIGHT_ENVIRONMENT; \
		PYTHONPATH=src $(UV) run flyte --endpoint localhost:8090 --insecure run \
			--project "$(FLYTE_PREFLIGHT_PROJECT)" --domain "$(FLYTE_PREFLIGHT_DOMAIN)" \
			--image "runtime=$(FLYTE_PREFLIGHT_IMAGE)" \
			src/junyi_predictor/workflows/preflight.py runtime_compatibility

flyte-backend-preflight-status:
	kubectl get all --namespace "$(FLYTE_PREFLIGHT_NAMESPACE)"
	kubectl get pods --namespace "$(FLYTE_PREFLIGHT_NAMESPACE)" -o wide

flyte-backend-preflight-down:
	@helm uninstall flyte-preflight --namespace "$(FLYTE_PREFLIGHT_NAMESPACE)" 2>/dev/null || true
	@helm uninstall junyi-preflight --namespace "$(FLYTE_PREFLIGHT_NAMESPACE)" 2>/dev/null || true
	@helm uninstall flyte-postgres --namespace "$(FLYTE_PREFLIGHT_NAMESPACE)" 2>/dev/null || true
	@kubectl delete namespace "$(FLYTE_PREFLIGHT_NAMESPACE)" --ignore-not-found
