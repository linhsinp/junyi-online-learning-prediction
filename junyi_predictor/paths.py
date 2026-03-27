"""Repository-local paths for generated and source artifacts."""

from __future__ import annotations

from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
DATA_DIR = ARTIFACTS_DIR / "data"
MODEL_DIR = ARTIFACTS_DIR / "model"

RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DATA_DIR = DATA_DIR / "output"
EXPERIMENT_DATA_DIR = DATA_DIR / "experiment"
TEST_DATA_DIR = DATA_DIR / "test"
DEMO_DATA_DIR = DATA_DIR / "demo"
FEATURE_STORE_DIR = DATA_DIR / "feature_store"

LOG_FILE = RAW_DATA_DIR / "Log_Problem.csv"
USER_FILE = RAW_DATA_DIR / "Info_UserData.csv"
CONTENT_FILE = RAW_DATA_DIR / "Info_Content.csv"
