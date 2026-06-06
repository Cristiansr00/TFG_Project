from enum import IntEnum
from utils.tools import mostrar_seccion

class TransformType(IntEnum):
    # Original
    ORIGINAL = 0

    # Texturas
    MEAN = 1
    STD = 2
    CONTRAST = 3
    DISSIMILARITY = 4
    HOMOGENEITY = 5
    ASM = 6
    MAX = 7
    ENTROPY = 8
    ENERGY = 9

    # Binarización
    OTSU = 20
    SPLINE = 21

def get_transformType(transform):
    """Retorna la categoría de la transformación: 'Original', 'Textura' o 'Binary'
    Acepta tanto strings (números) como TransformType values"""
    try:
        # Si es un string, convertir a int primero
        if isinstance(transform, str):
            transform = int(transform)
        # Convertir a TransformType si es un int
        if isinstance(transform, int):
            transform = TransformType(transform)
        
        if transform == TransformType.ORIGINAL:
            return "Original"
        elif transform in [TransformType.MEAN, TransformType.STD, TransformType.CONTRAST, 
                           TransformType.DISSIMILARITY, TransformType.HOMOGENEITY, 
                           TransformType.ASM, TransformType.MAX, TransformType.ENTROPY, TransformType.ENERGY]:
            return "Textura"
        elif transform in [TransformType.OTSU, TransformType.SPLINE]:
            return "Binary"
    except (ValueError, KeyError):
        return None
    return None

def print_transform_types():
    """Imprime todas las transformaciones disponibles organizadas por categoría"""
    mostrar_seccion("Transformaciones disponibles")
    print(f"  {TransformType.ORIGINAL.value}. {TransformType.ORIGINAL.name}")

    texturas = [
        TransformType.MEAN, TransformType.STD, TransformType.CONTRAST,
        TransformType.DISSIMILARITY, TransformType.HOMOGENEITY,
        TransformType.ASM, TransformType.MAX, TransformType.ENTROPY, TransformType.ENERGY
    ]
    for transform in texturas:
        print(f"  {transform.value}. {transform.name}")

    binarizaciones = [TransformType.OTSU, TransformType.SPLINE]
    for transform in binarizaciones:
        print(f"  {transform.value}. {transform.name}")
