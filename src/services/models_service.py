from pathlib import Path

from utils import paths

def get_trained_models():

    data_path = Path(f"{paths.MODELS_DIR}")
    data_path.mkdir(parents=True, exist_ok=True)

    datasets = [d.name for d in data_path.iterdir() if d.is_dir()]

    return datasets

def get_YOLO_models():
        
    data_path = Path(f"{paths.MODELS_YOLO_DIR}")
    data_path.mkdir(parents=True, exist_ok=True)

    datasets = [d.stem for d in data_path.iterdir() if d.is_file() and d.suffix == ".pt"]

    return datasets


def resolve_yolo_model(model_name: str) -> str:
    """
    Devuelve la ruta local del modelo si existe en models/YOLO.
    Si no existe, devuelve el nombre .pt para que Ultralytics pueda resolverlo.
    """
    model_name = model_name.strip()
    if not model_name:
        raise ValueError("Debes indicar un modelo YOLO")

    filename = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
    local_path = paths.MODELS_YOLO_DIR / filename

    if local_path.exists():
        return str(local_path)

    return filename
