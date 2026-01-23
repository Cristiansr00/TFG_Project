import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np

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