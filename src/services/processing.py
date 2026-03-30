
import os
import shutil
import cv2
from transforms.Binarization.BinarizationTransforms import Otsu_binarization, Spline_binarization
from transforms.Texture.TextureTransforms import Mean_texture, Std_texture, Contrast_texture, Dissimilarity_texture, Homogeneity_texture, Asm_texture, Max_texture, Entropy_texture
from transforms.Transform import Original_transform, Transform
from skimage.filters.rank import entropy
from skimage.morphology import disk
from utils.TransformType import TransformType
from utils import paths, tools
import numpy as np
import yaml
from tqdm import tqdm

DESTINO = 'multichannelRGB'

TRANSFORMS_CONFIG = {
    # Original
    TransformType.ORIGINAL: {"class": Original_transform, "required_params": []},

    # Texturas (no necesitan parámetros adicionales)
    TransformType.MEAN: {"class": Mean_texture, "required_params": []},
    TransformType.STD: {"class": Std_texture, "required_params": []},
    TransformType.CONTRAST: {"class": Contrast_texture, "required_params": []},
    TransformType.DISSIMILARITY: {"class": Dissimilarity_texture, "required_params": []},
    TransformType.HOMOGENEITY: {"class": Homogeneity_texture, "required_params": []},
    TransformType.ASM: {"class": Asm_texture, "required_params": []},
    TransformType.MAX: {"class": Max_texture, "required_params": []},
    TransformType.ENTROPY: {"class": Entropy_texture, "required_params": []},
    
    # Binarizaciones (necesitan kernel y morph)
    TransformType.OTSU: {"class": Otsu_binarization, "required_params": ["kernel", "morph"]},
    TransformType.SPLINE: {"class": Spline_binarization, "required_params": ["kernel", "morph"]},
}

# Función que devuelve la clase de la transformación elegida
def get_transform(transform_type, **kwargs):
    """Obtiene transformación y valida parámetros"""

    if transform_type not in TRANSFORMS_CONFIG:
        raise ValueError(f"transform_id {transform_type} no existe")

    config = TRANSFORMS_CONFIG[transform_type]
    required = config["required_params"]

    # Validar parámetros requeridos
    missing = [p for p in required if p not in kwargs]
    if missing:
        raise ValueError(
            f"Transform {transform_type} requiere: {missing}"
        )

    # Crear instancia REAL
    return config["class"](**kwargs)

# Función que devuelve todos los canales de la imagen en escala de grises
def show_all_channels(img_rgb):
    """Muestra cada canal por separado"""
    # OpenCV usa BGR, así que separamos en ese orden
    b_channel, g_channel, r_channel = cv2.split(img_rgb)
    
    # Mostrar cada canal en escala de grises
    cv2.imshow("Canal R (Rojo)", r_channel)
    cv2.imshow("Canal G (Verde)", g_channel)
    cv2.imshow("Canal B (Azul)", b_channel)
    
    # Mostrar imagen RGB completa (convertir BGR a RGB para colores reales)
    img_rgb_correcto = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    cv2.imshow("Imagen RGB Completa", img_rgb_correcto)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Función que devuelve todos los canales de la imagen dependiendo del color del canal
def show_all_channels_colored(img, assume_bgr=True):
    if not assume_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    b_channel, g_channel, r_channel = cv2.split(img)
    
    zeros = np.zeros_like(b_channel)

    r_display = cv2.merge([zeros, zeros, r_channel])
    g_display = cv2.merge([zeros, g_channel, zeros])
    b_display = cv2.merge([b_channel, zeros, zeros])

    cv2.imshow("Canal Rojo", r_display)
    cv2.imshow("Canal Verde", g_display)
    cv2.imshow("Canal Azul", b_display)
    cv2.imshow("Imagen Original", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ELIMINAR
def execute():
    result = generate_synthetic_RGB_image(r"C:\Users\crist\Desktop\TFG\Proyecto\Proyecto\src\services\2Gy-004.JPG",
                                            r={"type": TransformType.ORIGINAL},
                                            g={"type": TransformType.ASM},
                                            b={"type": TransformType.OTSU, "kernel": 3, "morph": 1}
                                        )
    
    print(f"El nombre de la transformación es {result[1]}")
    # result = generate_synthetic_RGB_image(r"C:\Users\Usuario\Desktop\PROYECTOS\TFG_Project\data\processed\SynBRG_nc3_ASM_5kFold\split_1\train\images\2Gy-004.JPG", 0, 6, 20, b_params={'kernel': 3, 'morph': 1})
    # show_all_channels(result[0])

    show_all_channels_colored(result[0])

# Función que genera una imagen con canales RGB sinteticos por transformaciones
def generate_synthetic_RGB_image(img_path, r, g, b):
    """
    Genera una imagen RGB sintética aplicando transformaciones a cada canal.
    
    Parámetros:
        img_path (str): Ruta a la imagen de entrada.
        r (dict): Configuración para el canal R. Ej: {"type": TransformType.ASM}
        g (dict): Configuración para el canal G. Ej: {"type": TransformType.ORIGINAL}
        b (dict): Configuración para el canal B. Ej: {"type": TransformType.OTSU, "kernel": 3, "morph": 1}
    
    Retorna:
        numpy.ndarray: Imagen RGB sintética en formato BGR.
    
    Ejemplo:
        generate_synthetic_RGB_image('img.jpg', 
                                     r={"type": TransformType.ASM},
                                     g={"type": TransformType.ORIGINAL},
                                     b={"type": TransformType.OTSU, "kernel": 3, "morph": 1})
    """
    img = cv2.imread(img_path)

    
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    
    # Aplicar con validación
    r_transform = get_transform(
        r["type"],
        **{k: v for k, v in r.items() if k != "type"}
    )

    g_transform = get_transform(
        g["type"],
        **{k: v for k, v in g.items() if k != "type"}
    )

    b_transform = get_transform(
        b["type"],
        **{k: v for k, v in b.items() if k != "type"}
    )

    # Aplicar transformaciones
    r_channel = r_transform(gray)
    g_channel = g_transform(gray)
    b_channel = b_transform(gray)

    # Merge final
    return cv2.merge([b_channel, g_channel, r_channel]), _build_dataset_name(r_transform,g_transform,b_transform, 0)

# def generate_synthetic_RGB_image(img_path, r, b):
#  #r -> imagen con textura
#  #g -> imagen original
#  #b-> imagen binarizada otsu
#     img = cv2.imread(img_path)

#     # Transformamos la imagen que puede ser en color a escala de grises
#     if len(img.shape) == 3 and img.shape[2] == 3:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = img

#     r_img = None
#     r = int(r)
#     if r == 1:
#         r_img = fast_glcm_ASM(gray)
#     elif r == 2:
#         r_img = fast_glcm_entropy(gray)
#     else:
#         raise ValueError("r must be 1 (ASM) or 2 (Entropy)")

#     g_img = gray
#     b_img = otsu_binarization(gray, b[0], b[1])

#     ## Cambio
#     ## BGR 
#     processed_rgb = cv2.merge([g_img, r_img, b_img])

#     return processed_rgb    


def _build_dataset_name(r: Transform, g: Transform, b: Transform, folds_num: int):
    return f"SynRGB_{r.get_name()}_{g.get_name()}_{b.get_name()}_{folds_num}kFold"


# def generate_multichannel_structure(dataset_name:str, r, b, folds_num:int = 5):
#     name_destino = _build_dataset_name(int(b[0]), int(b[1]), int(r),  folds_num)
#     path_destino =f"{paths.DATA_DIR}/processed/{name_destino}"
#     dataset_path = f"{paths.DATA_DIR}/processed/{dataset_name}"
#     shutil.copytree(dataset_path, path_destino)
#         # ---- actualizar YAMLs ----
#     for root, _, files in os.walk(path_destino):
#         for file in files:
#             if file.endswith(".yaml"):
#                 yaml_path = os.path.join(root, file)
#                 with open(yaml_path, 'r') as f:
#                     data = yaml.safe_load(f)

#                 filename = os.path.basename(data['path'])
#                 data['path'] = f"{path_destino}/{filename}"

#                 with open(yaml_path, 'w') as f:
#                     yaml.dump(data, f)

#       # ---- CONTAMOS cuántas imágenes habrá en total (para la barra) ----
#     total_images = 0
#     for i in range(1, folds_num + 1):
#         for subset in ['train', 'val', 'test']:
#             image_dir = f"{path_destino}/split_{i}/{subset}/images"
#             total_images += len(os.listdir(image_dir))

#     # ================= BARRA DE PROGRESO =================
#     with tqdm(total=total_images,
#               desc="Generando RGB sintético",
#               unit="img") as pbar:

#         # ---- esta función AVANZA la barra ----
#         def process_split_images(split, subset, image_list):
#             for img_name in image_list:
#                 img_path = f"{path_destino}/split_{split}/{subset}/images/{img_name}"
#                 img_res = generate_synthetic_RGB_image(img_path, r, b)
#                 cv2.imwrite(img_path, img_res)

#                 pbar.update(1)   # ← AQUÍ se mueve la barra

#         # ---- aquí se recorren folds y subsets ----
#         for i in range(1, folds_num + 1):
#             for subset in ['train', 'val', 'test']:
#                 image_dir = f"{path_destino}/split_{i}/{subset}/images"
#                 image_list = os.listdir(image_dir)
#                 process_split_images(i, subset, image_list)
#     # ====================================================

def generate_multichannel_structure(dataset_name: str, r: dict, g: dict, b: dict, folds_num: int = 5):
    
        # preparar transformadores UNA vez
    r_transform = get_transform(r["type"], **{k: v for k, v in r.items() if k != "type"})
    g_transform = get_transform(g["type"], **{k: v for k, v in g.items() if k != "type"})
    b_transform = get_transform(b["type"], **{k: v for k, v in b.items() if k != "type"})

    name_destino = _build_dataset_name(r_transform, g_transform, b_transform, folds_num)
    
    path_origen = f"{paths.DATA_DIR}/processed/{dataset_name}"
    path_destino = f"{paths.DATA_DIR}/processed/{name_destino}"

    if os.path.exists(path_destino):
        shutil.rmtree(path_destino)
    shutil.copytree(path_origen, path_destino)

    # actualizar YAMLs con la nueva ruta
    for root, _, files in os.walk(path_destino):
        for file in files:
            if not file.endswith(".yaml"):
                continue
            yaml_path = os.path.join(root, file)
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "path" in data:
                filename = os.path.basename(data["path"])
                data["path"] = os.path.join(path_destino, filename)
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.dump(data, f)

    # contar imágenes
    total_images = 0
    for i in range(1, folds_num + 1):
        for subset in ["train", "val", "test"]:
            img_dir = os.path.join(path_destino, f"split_{i}", subset, "images")
            if os.path.isdir(img_dir):
                total_images += len([n for n in os.listdir(img_dir)
                                     if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))])

    with tqdm(total=total_images, desc="Generando RGB sintético", unit="img") as pbar:
        for i in range(1, folds_num + 1):
            for subset in ["train", "val", "test"]:
                img_dir = os.path.join(path_destino, f"split_{i}", subset, "images")
                if not os.path.isdir(img_dir):
                    continue
                for img_name in os.listdir(img_dir):
                    if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                        continue
                    img_path = os.path.join(img_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        pbar.update(1)
                        continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if (img.ndim == 3 and img.shape[2] == 3) else img

                    out = cv2.merge([
                        b_transform(gray),
                        g_transform(gray),
                        r_transform(gray),
                    ])
                    cv2.imwrite(img_path, out)
                    pbar.update(1)

    return path_destino

if __name__ == "__main__":
    generate_multichannel_structure('./dataset/2026-01-18_5-Fold_Cross-val')