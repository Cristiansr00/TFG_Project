
import os
import shutil
import cv2
from transforms.Binarization.BinarizationTransforms import Otsu_binarization, Spline_binarization
from transforms.Texture.TextureTransforms import Mean_texture, Std_texture, Contrast_texture, Dissimilarity_texture, Homogeneity_texture, Asm_texture, Max_texture, Entropy_texture, Energy_texture
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
    TransformType.ENERGY: {"class": Energy_texture, "required_params": []},
    
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
def execute(dataset):
    # result = generate_synthetic_RGB_image(r"C:\Users\Usuario\Desktop\PROYECTOS\TFG_Project\src\services\2Gy-004.JPG",
    #                                         r={"type": TransformType.ORIGINAL},
    #                                         g={"type": TransformType.ASM},
    #                                         b={"type": TransformType.OTSU, "kernel": 3, "morph": 1}
    #                                     )
    
    result = generate_monochannel_structure(dataset,
                                            t_conf = {"type": TransformType.ASM} )

    print(f"El nombre de la transformación es {result[1]}")

    show_all_channels_colored(result[0])

def execute2():

    generate_multichannel_structure("2026-01-18_5-Fold_Cross-val",
                                        r={"type": TransformType.ORIGINAL},
                                        g={"type": TransformType.ASM},
                                        b={"type": TransformType.OTSU, "kernel": 3, "morph": 1}
                                    )

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


def _build_dataset_name(r: Transform, g: Transform = None, b: Transform = None, folds_num: int = 5):
    if g is None and b is None:
        return f"SynMono_{r.get_name()}_{folds_num}kFold"
    else:
        return f"SynRGB_{r.get_name()}_{g.get_name()}_{b.get_name()}_{folds_num}kFold"

def _contar_imagenes(folds_num: int, path_destino: str):
    total_images = 0
    for i in range(1, folds_num + 1):
        for subset in ["train", "val", "test"]:
            img_dir = os.path.join(path_destino, f"split_{i}", subset, "images")
            if os.path.isdir(img_dir):
                total_images += len([n for n in os.listdir(img_dir)
                                     if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))])
    return total_images

def _update_yaml(path_destino):
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


def generate_monochannel_structure(dataset_name: str, t_conf: dict, folds_num: int = 5):
    t_conf_transform = get_transform(t_conf["type"], **{k: v for k, v in t_conf.items() if k != "type"})
    name_destino = _build_dataset_name(t_conf_transform, folds_num = folds_num)

    path_origen = f"{paths.DATA_DIR}/processed/{dataset_name}"
    path_destino = f"{paths.DATA_DIR}/processed/{name_destino}"

    if os.path.exists(path_destino):
        shutil.rmtree(path_destino)
    shutil.copytree(path_origen, path_destino)

    _update_yaml(path_destino)

    total_images = _contar_imagenes(folds_num, path_destino)

    with tqdm(total=total_images, desc="Generando Monocanal sintético", unit="img") as pbar:
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

                    out = t_conf_transform(gray)
                    cv2.imwrite(img_path, out)
                    pbar.update(1)

    return path_destino

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
    _update_yaml(path_destino)

    # contar imágenes
    total_images = _contar_imagenes(folds_num, path_destino)

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