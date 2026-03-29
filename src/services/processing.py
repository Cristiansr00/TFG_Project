
import os
import shutil
import cv2
from transforms.Binarization.BinarizationTransforms import Otsu_binarization, Spline_binarization
from transforms.Texture.TextureTransforms import Mean_texture, Std_texture, Contrast_texture, Dissimilarity_texture, Homogeneity_texture, Asm_texture, Max_texture, Entropy_texture
from skimage.filters.rank import entropy
from skimage.morphology import disk
from utils import paths, tools
import numpy as np
import yaml
from tqdm import tqdm

DESTINO = 'multichannelRGB'

TRANSFORMS_CONFIG = {
    # Texturas (no necesitan parámetros adicionales)
    1: {"class": Mean_texture(), "required_params": []},
    2: {"class": Std_texture(), "required_params": []},
    3: {"class": Contrast_texture(), "required_params": []},
    4: {"class": Dissimilarity_texture(), "required_params": []},
    5: {"class": Homogeneity_texture(), "required_params": []},
    6: {"class": Asm_texture(), "required_params": []},
    7: {"class": Max_texture(), "required_params": []},
    8: {"class": Entropy_texture(), "required_params": []},
    
    # Binarizaciones (necesitan kernel y morph)
    20: {"class": Otsu_binarization(), "required_params": ["kernel", "morph"]},
    21: {"class": Spline_binarization(), "required_params": ["kernel", "morph"]},
}

def get_transform(transform_id, **kwargs):
    """Obtiene transformación y valida parámetros"""
    if transform_id not in TRANSFORMS_CONFIG:
        raise ValueError(f"transform_id {transform_id} no existe")
    
    config = TRANSFORMS_CONFIG[transform_id]
    required = config["required_params"]
    
    # Validar que estén todos los parámetros requeridos
    for param in required:
        if param not in kwargs:
            raise ValueError(f"Transform {transform_id} requiere el parámetro '{param}'")
    
    return config["class"]


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

def execute():
    result = generate_synthetic_RGB_image(r"C:\Users\Usuario\Desktop\PROYECTOS\TFG_Project\src\services\2Gy-004.JPG", 0, 6, 20, b_params={'kernel': 3, 'morph': 1})
    # result = generate_synthetic_RGB_image(r"C:\Users\Usuario\Desktop\PROYECTOS\TFG_Project\data\processed\SynBRG_nc3_ASM_5kFold\split_1\train\images\2Gy-004.JPG", 0, 6, 20, b_params={'kernel': 3, 'morph': 1})
    cv2.imshow("Resultado", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_synthetic_RGB_image(img_path, r_id, g_id, b_id, r_params=None, g_params=None, b_params=None):
    """
    Ejemplo:
        generate_synthetic_RGB_image('img.jpg', 
                                     r_id=6,      # Asm_texture (sin params)
                                     g_id=0,      # Original
                                     b_id=20,     # Otsu (requiere kernel, morph)
                                     transform_params={'kernel': 2, 'morph': 1})
    """    
    img = cv2.imread(img_path)

    
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    
    # Aplicar con validación
    r_channel = get_transform(r_id, **(r_params or {}))(gray) if r_id != 0 else gray
    g_channel = get_transform(g_id, **(g_params or {}))(gray) if g_id != 0 else gray
    b_channel = get_transform(b_id, **(b_params or {}))(gray) if b_id != 0 else gray
    
    return cv2.merge([b_channel, g_channel, r_channel])
    

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


def otsu_binarization(gray_img, morph, kernel):
    _, binary_otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_p = cv2.morphologyEx(binary_otsu, tools.MORPHS[int(morph)], tools.KERNELS[int(kernel)])
    return img_p

def fast_glcm(img, vmin=0, vmax=255, levels=8, kernel_size=5, distance=1.0, angle=0.0):
    '''
    Parameters
    ----------
    img: array_like, shape=(h,w), dtype=np.uint8
        input image
    vmin: int
        minimum value of input image
    vmax: int
        maximum value of input image
    levels: int
        number of grey-levels of GLCM
    kernel_size: int
        Patch size to calculate GLCM around the target pixel
    distance: float
        pixel pair distance offsets [pixel] (1.0, 2.0, and etc.)
    angle: float
        pixel pair angles [degree] (0.0, 30.0, 45.0, 90.0, and etc.)

    Returns
    -------
    Grey-level co-occurrence matrix for each pixels
    shape = (levels, levels, h, w)
    '''

    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, levels+1)
    gl1 = np.digitize(img, bins) - 1

    # make shifted image
    dx = distance*np.cos(np.deg2rad(angle))
    dy = distance*np.sin(np.deg2rad(-angle))
    mat = np.array([[1.0,0.0,-dx], [0.0,1.0,-dy]], dtype=np.float32)
    gl2 = cv2.warpAffine(gl1, mat, (w,h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REPLICATE)

    # make glcm
    glcm = np.zeros((levels, levels, h, w), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm


def fast_glcm_ASM(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm asm, energy
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            asm  += glcm[i,j]**2

    # ene = np.sqrt(asm)
    asm_img = cv2.normalize(asm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return asm_img

def fast_glcm_entropy(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm entropy
    '''
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
    ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
    ent = cv2.normalize(ent, None, 0, 255, cv2.NORM_MINMAX)
    ent = ent.astype(np.uint8)
    
    return ent

def build_dataset_name(n, k, t, folds):
    """
    n : int -> 0 = close (c), 1 = open (o)
    k : int -> 1-4  (kernel sizes 2-5)
    t : int -> 0 = ASM, 1 = ENT
    folds : int -> k-folds
    """

    morph_map = {1: "o", 2: "c"}
    texture_map = {1: "ASM", 2: "ENT"}

    if n not in morph_map:
        raise ValueError("n must be 0 (close) or 1 (open)")
    if k not in range(1, 5):
        raise ValueError("k must be in range 1-4 (kernels 2-5)")
    if t not in texture_map:
        raise ValueError("t must be 0 (ASM) or 1 (ENT)")
    if folds <= 1:
        raise ValueError("folds must be > 1")

    return f"SynRGB_n{morph_map[n]}{k+1}_{texture_map[t]}_{folds}kFold"


def generate_multichannel_structure(dataset_name:str, r, b, folds_num:int = 5):
    name_destino = build_dataset_name(int(b[0]), int(b[1]), int(r),  folds_num)
    path_destino =f"{paths.DATA_DIR}/processed/{name_destino}"
    dataset_path = f"{paths.DATA_DIR}/processed/{dataset_name}"
    shutil.copytree(dataset_path, path_destino)
        # ---- actualizar YAMLs ----
    for root, _, files in os.walk(path_destino):
        for file in files:
            if file.endswith(".yaml"):
                yaml_path = os.path.join(root, file)
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)

                filename = os.path.basename(data['path'])
                data['path'] = f"{path_destino}/{filename}"

                with open(yaml_path, 'w') as f:
                    yaml.dump(data, f)

      # ---- CONTAMOS cuántas imágenes habrá en total (para la barra) ----
    total_images = 0
    for i in range(1, folds_num + 1):
        for subset in ['train', 'val', 'test']:
            image_dir = f"{path_destino}/split_{i}/{subset}/images"
            total_images += len(os.listdir(image_dir))

    # ================= BARRA DE PROGRESO =================
    with tqdm(total=total_images,
              desc="Generando RGB sintético",
              unit="img") as pbar:

        # ---- esta función AVANZA la barra ----
        def process_split_images(split, subset, image_list):
            for img_name in image_list:
                img_path = f"{path_destino}/split_{split}/{subset}/images/{img_name}"
                img_res = generate_synthetic_RGB_image(img_path, r, b)
                cv2.imwrite(img_path, img_res)

                pbar.update(1)   # ← AQUÍ se mueve la barra

        # ---- aquí se recorren folds y subsets ----
        for i in range(1, folds_num + 1):
            for subset in ['train', 'val', 'test']:
                image_dir = f"{path_destino}/split_{i}/{subset}/images"
                image_list = os.listdir(image_dir)
                process_split_images(i, subset, image_list)
    # ====================================================

if __name__ == "__main__":
    generate_multichannel_structure('./dataset/2026-01-18_5-Fold_Cross-val')