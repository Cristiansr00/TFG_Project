from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models/trained"
MODELS_YOLO_DIR = PROJECT_ROOT / "models/YOLO"
TESTS_DIR = PROJECT_ROOT / "tests"
# CONFIGS_DIR = PROJECT_ROOT / "configs"