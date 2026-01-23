
import os
import shutil
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
import yaml

DESTINO = 'multichannel'

def generate_synthetic_RGB_image(img_path):

    img = cv2.imread(img_path)
    
    # Transformamos la imagen que puede ser en color a escala de grises
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    processed_rgb = cv2.merge([gray, otsu_binarization(gray), entropy_norm(gray)])

    return processed_rgb    


def otsu_binarization(gray_img):
    _, binary_otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_otsu


def entropy_norm(gray_img):
    entropy_img = entropy(gray_img, disk(5))

    entropy_norm = ((entropy_img - entropy_img.min()) /
                    (entropy_img.max() - entropy_img.min()) * 255).astype(np.uint8)
    
    return entropy_norm

def generate_processed_dataset():
    for root, dirs, files in os.walk(DESTINO):
        for file in files:
            path = os.path.join(root, file)

            img_proc = generate_synthetic_RGB_image(path)

            cv2.imwrite(path, img_proc)


def generate_multichannel_structure(dataset_path:str, folds_num:int = 5):

    shutil.copytree(dataset_path, DESTINO)
    for root, _, files in os.walk(DESTINO):
        for file in files:
            if file.endswith(".yaml"):
                yaml_path = os.path.join(root, file)
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                filename = os.path.basename(data['path'])
                data['path'] = f"{DESTINO}/{filename}"
                with open(yaml_path, 'w') as f:
                    yaml.dump(data, f)

        def process_split_images(split, subset, image_list):
            """
            Procesa las imágenes de un split específico y guarda los resultados.
            Los argumentos que recibe son:
            - split: Número del split (1, 2, ..., nF)
            - subset: Subconjunto de datos (train, val, test)
            - image_list: Lista de nombres de imágenes a procesar
            """
            for img_name in image_list:
                img_path = f"{DESTINO}/split_{split}/{subset}/images/{img_name}"
                rel_path = f"split_{split}/{subset}/images/{img_name.split('.')[0]}"
                img_res = generate_synthetic_RGB_image(img_path)
                cv2.imwrite(img_path, img_res)

    for i in range(1, int(folds_num) + 1):
        for subset in ['train', 'val', 'test']:
            image_dir = f"{DESTINO}/split_{i}/{subset}/images"
            image_list = os.listdir(image_dir)
            process_split_images(i, subset, image_list)

if __name__ == "__main__":
    generate_multichannel_structure('./dataset/2026-01-18_5-Fold_Cross-val')