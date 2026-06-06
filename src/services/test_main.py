from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.processing import generate_synthetic_RGB_image
from utils.TransformType import TransformType


def main():
    image_path = Path(__file__).with_name("2Gy-004.JPG")
    image, dataset_name = generate_synthetic_RGB_image(
        str(image_path),
        r={"type": TransformType.ORIGINAL},
        g={"type": TransformType.ASM},
        b={"type": TransformType.OTSU, "kernel": 2, "morph": 1},
    )

    output_path = image_path.with_name(f"{dataset_name}.jpg")
    cv2.imwrite(str(output_path), image)
    print(f"Imagen sintética guardada en {output_path}")


if __name__ == "__main__":
    main()
