import cv2
import numpy as np
from scipy import interpolate, signal
from transforms import Transform

KERNELS = {                                   # Kernel size
    1: np.ones((2, 2), np.uint8),    # 2x2
    2: np.ones((3, 3), np.uint8),    # 3x3
    3: np.ones((4, 4), np.uint8),    # 4x4
    4: np.ones((5, 5), np.uint8)     # 5x5
}
MORPHS = {                                    # Kernel morphological filter
    1: cv2.MORPH_CLOSE,                    # Close
    2: cv2.MORPH_OPEN                      # Open
}
MORPH_NAMES = {1: "c", 2: "o"}


class Otsu_binarization(Transform):

    def __init__(self, morph=None, kernel=None):
        self.morph = morph
        self.kernel = kernel
    
    def __call__(self, img):
        _, binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if self.morph in MORPHS and self.kernel in KERNELS:
            morph_type = MORPHS[self.morph]
            kernel = KERNELS[self.kernel]
            binary_otsu = cv2.morphologyEx(binary_otsu, morph_type, kernel)

        return binary_otsu.astype(np.uint8)
    
    def get_name(self):
        morph_str = ""
        kernel_str = ""

        if self.morph in MORPH_NAMES:
           morph_str = MORPH_NAMES[self.morph]

        if self.kernel in KERNELS:
            size = KERNELS[self.kernel].shape[0]
            kernel_str = f"{size}"

        return f"n{morph_str}{kernel_str}"
    
class Spline_binarization(Transform):
    """
    Binarización automática usando spline cúbico sobre el histograma,
    con opción de aplicar morfología y kernel.
    """

    def __init__(self, morph=None, kernel=None):
        """
        Args:
            morph (int or None): tipo de operación morfológica (0=open, 1=close, 2=erode, 3=dilate)
            kernel (np.ndarray or None): kernel para morfología
        """
        self.morph = morph
        self.kernel = kernel

    def get_threshold(self, img):
        """Devuelve el umbral óptimo usando spline sobre el histograma"""
        if len(img.shape) != 2:
            raise ValueError("get_threshold solo acepta imágenes 2D en escala de grises")

        # Histograma
        img_his = cv2.calcHist([img], [0], None, [256], [0, 256])
        x_his = np.linspace(0, 255, 256)
        y_his = img_his[:, 0]

        # Spline cúbico
        spl = interpolate.splrep(x_his, y_his, k=3, s=0)
        n_samples = 256 * 20
        x_spl = np.linspace(0, 255, n_samples)
        y_spl = interpolate.splev(x_spl, spl)

        # Detectar picos
        d = n_samples * 30 / 256
        ind_peaks = signal.find_peaks(y_spl, height=0, distance=d)[0]

        x_floors, y_peaks = [], []

        for ind in ind_peaks:
            y_peaks.append(y_spl[ind])
            # Floor
            isfloor = ind
            while isfloor < len(y_spl)-1 and y_spl[isfloor] > 0:
                isfloor += 1
            if isfloor - ind < n_samples * 10 / 256:
                isfloor += int(n_samples * 10 / 256)
                if isfloor >= len(y_spl):
                    isfloor = len(y_spl) - 1
            x_floors.append(x_spl[isfloor])

        # Umbral óptimo
        if len(y_peaks) > 1:
            threshold = x_floors[y_peaks.index(max(y_peaks[:-1]))]
        else:
            threshold = x_floors[0]

        return threshold

    def __call__(self, img):
        """Aplica binarización y opcionalmente morfología"""
        threshold = self.get_threshold(img)
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        # Si se define morph y kernel, aplicar morfología
        if self.morph in MORPHS and self.kernel in KERNELS:
            morph_type = MORPHS[self.morph]
            kernel = KERNELS[self.kernel]
            binary = cv2.morphologyEx(binary, morph_type, kernel)

        return binary.astype(np.uint8)
    
    def get_name(self):
        morph_str = ""
        kernel_str = ""

        if self.morph in MORPH_NAMES:
           morph_str = MORPH_NAMES[self.morph]

        if self.kernel in KERNELS:
            size = KERNELS[self.kernel].shape[0]
            kernel_str = f"{size}"

        return f"s{morph_str}{kernel_str}"