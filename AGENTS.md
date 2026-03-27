# Repository Guidelines

## Project Structure & Module Organization
Core execution code lives in `junyi_predictor/`. Stage logic is split across `junyi_predictor/pipeline/preprocessing.py`, `junyi_predictor/pipeline/feature_engineering.py`, and `junyi_predictor/pipeline/training.py`, with storage adapters in `junyi_predictor/storage/`. Flyte orchestration is split between `flyte/tasks/`, `flyte/workflows/`, and top-level workflow entrypoints such as `flyte/full_pipeline_wf.py`. Data utilities live in `data/`, infrastructure code is in `terraform/`, and container builds are under `docker/`. Tests are in `tests/`, exploratory work in `notebooks/`, and generated artifacts are typically written to `model/` and `data/{raw,experiment,feature_store,output}`.

## Build, Test, and Development Commands
Use Python 3.10 to 3.12. Preferred setup is `uv`:

- `uv sync --all-groups`: install project and dev dependencies from `pyproject.toml` and `uv.lock`.
- `uv run pytest`: run the test suite.
- `uv run ruff check .`: lint Python files.
- `uv run ruff format .`: apply formatting.
- `uv run isort .`: normalize import order.
- `pre-commit run --all-files`: run the same checks used before commits.

If `uv` is not installed, review `setup.sh`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, type hints on new or changed functions, and short docstrings for non-trivial behavior. Keep modules and functions in `snake_case`; classes use `PascalCase`; constants use `UPPER_SNAKE_CASE`. Prefer small, stage-oriented functions in `junyi_predictor/` and keep Flyte task/workflow names descriptive and thin.

## Testing Guidelines
Tests use `pytest`. Add new tests under `tests/` with filenames named `test_*.py` and test functions named `test_*`. Prefer focused unit tests with fixtures and mocks, following [tests/test_gcs_utils.py](/Users/hsin-pei/Desktop/github_repo/junyi-online-learning-prediction/tests/test_gcs_utils.py). Run `uv run pytest` before opening a PR; add regression coverage for any bug fix or data-path change.

## Commit & Pull Request Guidelines
Recent commits use short, imperative messages such as `clean up tasks and workflows` and `create scheduled launch plan`. Keep that style, with one logical change per commit. PRs should include a concise summary, affected areas, test evidence, and linked issues. Include screenshots only for UI or notebook-output changes.

## Security & Configuration Tips
Do not commit secrets or local credentials. This repository contains local-only files such as `.env`, `kaggle.json`, and `gcs-service-account.json`; treat them as developer machine state and use environment variables for production configuration.
