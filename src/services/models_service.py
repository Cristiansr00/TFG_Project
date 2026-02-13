from pathlib import Path

from utils import paths

def get_trained_models():

    data_path = Path(f"{paths.MODELS_DIR}")

    if not data_path.exists():
        raise FileNotFoundError(f"No existe la carpeta {paths.MODELS_DIR}")

    datasets = [d.name for d in data_path.iterdir() if d.is_dir()]

    return datasets

def get_YOLO_models():
        
    data_path = Path(f"{paths.MODELS_YOLO_DIR}")

    if not data_path.exists():
        raise FileNotFoundError(f"No existe la carpeta {paths.MODELS_YOLO_DIR}")

    datasets = [d.name for d in data_path.iterdir() if d.is_file() and d.suffix == ".pt"]

    return datasets