import os
import cv2
import numpy as np
from ultralytics import YOLO

KERNELS = {                                   # Kernel size
    1: np.ones((2, 2), np.uint8),    # 2x2
    2: np.ones((3, 3), np.uint8),    # 3x3
    3: np.ones((4, 4), np.uint8),    # 4x4
    4: np.ones((5, 5), np.uint8)     # 5x5
}
MORPHS = {                                    # Kernel morphological filter
    1: cv2.MORPH_CLOSE,                    # Close
    2: cv2.MORPH_OPEN                      # Open
}



def limpiar_consola():
    os.system("cls" if os.name == "nt" else "clear")

# def yolo_model_exists(model_path: str, model_name: str):
#     if not os.path.exists(model_path):
#         print(f"Modelo {model_path} no encontrado. Descargando...")
#         try:
#             model = YOLO(f"{model_name}.pt")
#             model.save(model_path)
#             print(f"Modelo descargado exitosamente.")
#         except Exception as e:
#             print(f"No se pudo descargar el modelo: {e}")