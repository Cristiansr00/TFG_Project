import cv2
import numpy as np
from scipy import interpolate, signal
from ..Transform import Transform

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

# Función helper para crear el nombre de las binarizaciones
def _build_binarization_name(morph, kernel, prefix):
    """Helper para construir el nombre de la transformación"""
    morph_str = ""
    kernel_str = ""

    if morph in MORPH_NAMES:
       morph_str = MORPH_NAMES[morph]

    if kernel in KERNELS:
        size = KERNELS[kernel].shape[0]
        kernel_str = f"{size}"

    return f"{prefix}{morph_str}{kernel_str}"


# Clases de Transformaciones realizadas mediante binarización
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
        return _build_binarization_name(self.morph, self.kernel, "n")
    
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
        img_his = cv2.calcHist(img, [0], None, [256], [0, 256])

        x_his = np.transpose(np.linspace(0, 255, 256))  # Intensity values
        y_his = img_his[:, 0]                                           # Number of pixel with intensity x

        spl = interpolate.splrep(x_his, y_his, k=3, s=1000)

        n_samples = 256 * 20                                            # Resampling (get more x values for spline plotting)
        x_spl = np.linspace(0, 255, n_samples)               # Resampling x
        y_spl = interpolate.splev(x_spl, spl).tolist()                  # Resampling y

        x_peaks = []    # This array will store x indexes where max peaks in the spline curve are reached
        x_floors = []   # This array will store x indexes where floors after peaks in the spline curve are reached

        y_peaks = []    # This array will store x indexes where max peaks in the spline curve are reached

        d = n_samples * 30 / 256  # Rule of 3: if n_samples == 256 then d_samples == 30. d=30u intensity to consider another peak
        ind_peaks = signal.find_peaks(y_spl, height=5., distance=d)[0]  # x indexes where peaks can be found
        for i in range(len(ind_peaks)):
            ind = ind_peaks[i]          # index where i_th peak can be found
            y_peaks.append(y_spl[ind])  # Add peak on spline curve into y_spl
            x_peaks.append(x_spl[ind])

            isfloor = ind  # x index where i_th floor can be found

            while isfloor < len(y_spl) - 1 and y_spl[isfloor] > 0:  # is floor (end of the mountain whose peak is y_spl[ind])
                isfloor += 1

            if isfloor - ind < (n_samples * 10 / 256):
                isfloor += (n_samples * 10 / 256)
                isfloor = int(isfloor)
            
            # Asegurar que isfloor no excede los límites del array
            isfloor = min(isfloor, len(x_spl) - 1)
            x_floors.append(x_spl[isfloor])

        goodthd = x_floors[y_peaks.index(
            max(y_peaks[:-1]))]  # Returns the x_floor corresponding to the highest peak (excluding white peak, i.e [:-1])
        return [goodthd, [x_his, y_his, x_spl, y_spl, x_peaks, y_peaks, x_floors]]

    def __call__(self, img):
        """Aplica binarización y opcionalmente morfología"""
        threshold = self.get_threshold(img)[0]
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        # Si se define morph y kernel, aplicar morfología
        if self.morph in MORPHS and self.kernel in KERNELS:
            morph_type = MORPHS[self.morph]
            kernel = KERNELS[self.kernel]
            binary = cv2.morphologyEx(binary, morph_type, kernel)

        return binary.astype(np.uint8)
        # return binary
    
    def get_name(self):
        return _build_binarization_name(self.morph, self.kernel, "s")