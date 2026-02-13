import cv2
import numpy as np

# Cargar la imagen
img_path = "data/processed/SynRGB_nc3_ASM_5kFold/split_1/test/images/2Gy-011.JPG"
img = cv2.imread(img_path)

if img is None:
    print("[ERROR] No se pudo cargar la imagen")
else:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Imagen (original):")
    print(f"  Min: {img.min()}, Max: {img.max()}, Dtype: {img.dtype}, Shape: {img.shape}")

    if len(img.shape) == 3:
        for i, channel_name in enumerate(['Blue', 'Green', 'Red']):
            channel = img[:, :, i]
            print(f"\nCanal {channel_name}:")
            print(f"  Min: {channel.min()}, Max: {channel.max()}, Mean: {channel.mean():.2f}")
            
            # Mostrar cada canal
            cv2.imshow(f"Canal {channel_name}", channel)
    
    cv2.imshow("Imagen Original", img)
    cv2.imshow("Escala de Grises", img_gray)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()