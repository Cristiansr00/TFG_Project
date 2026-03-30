from enum import IntEnum

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

    # Binarización
    OTSU = 20
    SPLINE = 21