# Repository Guidelines

## Project Structure & Module Organization
Core execution code lives in `junyi_predictor/`. Stage logic is split across `junyi_predictor/pipeline/preprocessing.py`, `junyi_predictor/pipeline/feature_engineering.py`, and `junyi_predictor/pipeline/training.py`, with storage adapters in `junyi_predictor/storage/`. Bootstrap utilities live in `junyi_predictor/bootstrap/`, and Flyte 2 entrypoints live in `orchestration/flyte_app.py`. Keep infrastructure concerns separated under `infra/`: `infra/terraform/` is for cloud resource provisioning, `infra/helm/junyi-predictor/` is for Kubernetes workload deployment, and `infra/docker/` contains container build definitions. Tests are in `tests/`, exploratory work in `notebooks/`, and generated artifacts live under the gitignored `artifacts/{data,model}/` tree.

## Build, Test, and Development Commands
Use `uv` as the only supported environment manager for this repository:

- `uv sync --all-groups`: install project and dev dependencies from `pyproject.toml` and `uv.lock`.
- `uv run pytest`: run the test suite.
- `uv run ruff check .`: lint Python files.
- `uv run ruff format .`: apply formatting.
- `uv run isort .`: normalize import order.
- `helm lint infra/helm/junyi-predictor`: validate the Kubernetes chart.
- `make flyte-local`: run the Flyte 2 full pipeline entrypoint locally with `flyte run --local`.
- `pre-commit run --all-files`: run the same checks used before commits.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, type hints on new or changed functions, and short docstrings for non-trivial behavior. Keep modules and functions in `snake_case`; classes use `PascalCase`; constants use `UPPER_SNAKE_CASE`. Prefer small, stage-oriented functions in `junyi_predictor/` and keep Flyte task/workflow names descriptive and thin.

## Testing Guidelines
Tests use `pytest`. Put focused unit tests under `tests/unit/` and end-to-end or boundary validation under `tests/acceptance/`, with filenames named `test_*.py` and test functions named `test_*`. Prefer focused unit tests with fixtures and mocks, following [tests/unit/test_gcs_utils.py](/Users/hsin-pei/Desktop/github_repo/junyi-online-learning-prediction/tests/unit/test_gcs_utils.py). Target at least 90% test coverage overall. Run all relevant tests after code changes, and run `uv run pytest` before opening a PR; add regression coverage for any bug fix or data-path change.

## Commit & Pull Request Guidelines
Use short, imperative commit messages with a leading gitmoji, for example `:memo: refresh local workflow docs` or `:recycle: modularize pipeline stages`. Keep one logical change per commit. PRs should include a concise summary, affected areas, test evidence, and linked issues. Include screenshots only for UI or notebook-output changes.

## Security & Configuration Tips
Do not commit secrets or local credentials. This repository contains local-only files such as `.env`, `kaggle.json`, and `gcs-service-account.json`; treat them as developer machine state and use environment variables for production configuration.
