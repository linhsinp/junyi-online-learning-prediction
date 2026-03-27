UV := uv

START_DATE ?= 2019-06-01T00:00:00
END_DATE ?= 2019-06-10T00:00:00
NUM_SAMPLES ?= 1000

.PHONY: help test lint helm-lint helm-template flyte-local flyte-preprocess-local flyte-train-local

help:
	@echo "Available targets:"
	@echo "  $(UV) sync --all-groups"
	@echo "  make test"
	@echo "  make lint"
	@echo "  make helm-lint"
	@echo "  make helm-template"
	@echo "  make flyte-local START_DATE=... END_DATE=... NUM_SAMPLES=..."
	@echo "  make flyte-preprocess-local START_DATE=... END_DATE=..."
	@echo "  make flyte-train-local"

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check junyi_predictor orchestration tests

helm-lint:
	helm lint ./infra/helm/junyi-predictor

helm-template:
	helm template junyi ./infra/helm/junyi-predictor \
		-f ./infra/helm/junyi-predictor/values-full-pipeline.yaml >/dev/null
	helm template junyi ./infra/helm/junyi-predictor \
		-f ./infra/helm/junyi-predictor/values-preprocess.yaml >/dev/null
	helm template junyi ./infra/helm/junyi-predictor \
		-f ./infra/helm/junyi-predictor/values-train-from-gcs.yaml >/dev/null

flyte-local:
	@set -a; [ -f .env ] && . ./.env; set +a; \
	PYTHONPATH=. $(UV) run flyte run --local orchestration/flyte_app.py full_pipeline \
		--start_date "$(START_DATE)" \
		--end_date "$(END_DATE)" \
		--num_samples "$(NUM_SAMPLES)"

flyte-preprocess-local:
	@set -a; [ -f .env ] && . ./.env; set +a; \
	PYTHONPATH=. $(UV) run flyte run --local orchestration/flyte_app.py preprocess_from_database \
		--start_date "$(START_DATE)" \
		--end_date "$(END_DATE)"

flyte-train-local:
	@set -a; [ -f .env ] && . ./.env; set +a; \
	PYTHONPATH=. $(UV) run flyte run --local orchestration/flyte_app.py train_from_gcs
