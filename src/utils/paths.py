from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models/trained"
MODELS_YOLO_DIR = PROJECT_ROOT / "models/YOLO"
TESTS_DIR = PROJECT_ROOT / "tests"
# CONFIGS_DIR = PROJECT_ROOT / "configs"

REQUIRED_DIRS = [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    MODELS_YOLO_DIR,
    TESTS_DIR,
]


def initialize_project_dirs():
    """Crea la estructura mínima para una instalación nueva."""
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
