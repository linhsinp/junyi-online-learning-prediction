FLYTE := ./.venv/bin/flyte
PYTEST := ./.venv/bin/pytest
RUFF := ./.venv/bin/ruff

START_DATE ?= 2019-06-01T00:00:00
END_DATE ?= 2019-06-10T00:00:00
NUM_SAMPLES ?= 1000

.PHONY: help test lint flyte-local flyte-preprocess-local flyte-train-local

help:
	@echo "Available targets:"
	@echo "  make test"
	@echo "  make lint"
	@echo "  make flyte-local START_DATE=... END_DATE=... NUM_SAMPLES=..."
	@echo "  make flyte-preprocess-local START_DATE=... END_DATE=..."
	@echo "  make flyte-train-local"

test:
	$(PYTEST)

lint:
	$(RUFF) check junyi_predictor orchestration data tests

flyte-local:
	@set -a; [ -f .env ] && . ./.env; set +a; \
	PYTHONPATH=. $(FLYTE) run --local orchestration/flyte_app.py full_pipeline \
		--start_date "$(START_DATE)" \
		--end_date "$(END_DATE)" \
		--num_samples "$(NUM_SAMPLES)"

flyte-preprocess-local:
	@set -a; [ -f .env ] && . ./.env; set +a; \
	PYTHONPATH=. $(FLYTE) run --local orchestration/flyte_app.py preprocess_from_database \
		--start_date "$(START_DATE)" \
		--end_date "$(END_DATE)"

flyte-train-local:
	@set -a; [ -f .env ] && . ./.env; set +a; \
	PYTHONPATH=. $(FLYTE) run --local orchestration/flyte_app.py train_from_gcs
