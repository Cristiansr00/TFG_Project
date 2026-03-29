from processing import generate_synthetic_RGB_image
import cv2

def main():
    result = generate_synthetic_RGB_image("src\services\2Gy-004.JPG", 1, 0, 20, transform_params={'kernel': 2, 'morph': 1})
    cv2.imread(result)



if __name__ == "__main__":
    main()