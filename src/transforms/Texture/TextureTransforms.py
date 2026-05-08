import cv2
import numpy as np
from ..Transform import Transform

def fast_glcm(img, vmin=0, vmax=255, levels=8, kernel_size=5, distance=1.0, angle=0.0):
    '''
    Parameters
    ----------
    img: array_like, shape=(h,w), dtype=np.uint8
        input image
    vmin: int
        minimum value of input image
    vmax: int
        maximum value of input image
    levels: int
        number of grey-levels of GLCM
    kernel_size: int
        Patch size to calculate GLCM around the target pixel
    distance: float
        pixel pair distance offsets [pixel] (1.0, 2.0, and etc.)
    angle: float
        pixel pair angles [degree] (0.0, 30.0, 45.0, 90.0, and etc.)

    Returns
    -------
    Grey-level co-occurrence matrix for each pixels
    shape = (levels, levels, h, w)
    '''

    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, levels+1)
    gl1 = np.digitize(img, bins) - 1

    # make shifted image
    dx = distance*np.cos(np.deg2rad(angle))
    dy = distance*np.sin(np.deg2rad(-angle))
    mat = np.array([[1.0,0.0,-dx], [0.0,1.0,-dy]], dtype=np.float32)
    gl2 = cv2.warpAffine(gl1, mat, (w,h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REPLICATE)

    # make glcm
    glcm = np.zeros((levels, levels, h, w), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm


class Mean_texture(Transform):
    
    def __call__(self, img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
        h,w = img.shape
        glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
        mean = np.zeros((h,w), dtype=np.float32)
        for i in range(levels):
            for j in range(levels):
                mean += glcm[i,j] * i / (levels)**2

        mean = cv2.normalize(mean, None, 0, 255, cv2.NORM_MINMAX)
        mean = mean.astype(np.uint8)

        return mean
    
    def get_name(self):
        return "mean"


class Std_texture(Transform):

    def __call__(self, img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
        h,w = img.shape
        glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
        mean = np.zeros((h,w), dtype=np.float32)
        for i in range(levels):
            for j in range(levels):
                mean += glcm[i,j] * i / (levels)**2

        std2 = np.zeros((h,w), dtype=np.float32)
        for i in range(levels):
            for j in range(levels):
                std2 += (glcm[i,j] * i - mean)**2

        std = np.sqrt(std2)

        std = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX)
        std = std.astype(np.uint8)

        return std
    
    def get_name(self):
        return "std"


class Contrast_texture(Transform):

    def __call__(self, img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
        h,w = img.shape
        glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
        cont = np.zeros((h,w), dtype=np.float32)
        for i in range(levels):
            for j in range(levels):
                cont += glcm[i,j] * (i-j)**2

        cont = cv2.normalize(cont, None, 0, 255, cv2.NORM_MINMAX)
        cont = cont.astype(np.uint8)

        return cont
    
    def get_name(self):
        return "contrast"
    

class Dissimilarity_texture(Transform):

    def __call__(self, img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
        h,w = img.shape
        glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
        diss = np.zeros((h,w), dtype=np.float32)
        for i in range(levels):
            for j in range(levels):
                diss += glcm[i,j] * np.abs(i-j)

        diss = cv2.normalize(diss, None, 0, 255, cv2.NORM_MINMAX)
        diss = diss.astype(np.uint8)

        return diss

    def get_name(self):
        return "dissimilarity"


class Homogeneity_texture(Transform):

    def __call__(self, img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
        h,w = img.shape
        glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
        homo = np.zeros((h,w), dtype=np.float32)
        for i in range(levels):
            for j in range(levels):
                homo += glcm[i,j] / (1.+(i-j)**2)

        homo = cv2.normalize(homo, None, 0, 255, cv2.NORM_MINMAX)
        homo = homo.astype(np.uint8)

        return homo

    def get_name(self):
        return "homogeneity"


class Asm_texture(Transform):

    def __call__(self, img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
        h,w = img.shape
        glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
        asm = np.zeros((h,w), dtype=np.float32)
        for i in range(levels):
            for j in range(levels):
                asm  += glcm[i,j]**2

        asm = cv2.normalize(asm, None, 0, 255, cv2.NORM_MINMAX)
        asm = asm.astype(np.uint8)

        return asm

    def get_name(self):
        return "asm"


class Max_texture(Transform):
    
    def __call__(self, img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
        glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
        max_  = np.max(glcm, axis=(0,1))
        max_ = cv2.normalize(max_, None, 0, 255, cv2.NORM_MINMAX)
        max_ = max_.astype(np.uint8)

        return max_
    
    def get_name(self):
        return "max"


class Entropy_texture(Transform):

    def __call__(self, img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
        glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
        pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
        ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
        ent = cv2.normalize(ent, None, 0, 255, cv2.NORM_MINMAX)
        ent = ent.astype(np.uint8)
        
        return ent
    
    def get_name(self): 
        return "entropy"


class Energy_texture(Transform):

    def __call__(self, img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
        glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
        pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
        energy = np.sum(pnorm**2, axis=(0,1))
        energy = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX)
        energy = energy.astype(np.uint8)
        
        return energy
    
    def get_name(self):
        return "energy"
