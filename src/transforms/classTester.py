from Binarization.BinarizationTransforms import Otsu_binarization, Spline_binarization
from Texture.TextureTransforms import Mean_texture, Std_texture, Contrast_texture, Dissimilarity_texture, Homogeneity_texture, Asm_texture, Max_texture, Entropy_texture
import cv2

def main():
    img = cv2.imread(r'C:\Users\Usuario\Desktop\PROYECTOS\TFG_Project\src\transforms\2Gy-004.JPG')
    
    if img is None:
        print("Error: No se pudo cargar la imagen")
        return
    
    # Transformamos la imagen que puede ser en color a escala de grises
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # g_img = Spline_binarization(kernel=2,morph=2)

    r_img = Spline_binarization(kernel=2, morph=2)

    b_img = Mean_texture()

    processed_rgb = cv2.merge([gray, r_img(gray), b_img(gray)])

    
    # print(f"Imagen procesada. Forma: {img_res.shape}")
    # print(f"Nombre de transformación: {binarizacion.get_name()}")
    
    # Opcional: guardar o mostrar resultado
    cv2.imwrite("resultado.jpg", processed_rgb)


if __name__ == "__main__":
    main()