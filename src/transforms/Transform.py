from abc import ABC, abstractmethod

class Transform(ABC):
    @abstractmethod
    def __call__ (self, img):
        pass

    @abstractmethod
    def get_name(self):
        """Retorna el nombre de la transformación"""
        pass 

class Original_transform(Transform):

    def __call__(self, img):
        return img
    
    def get_name(self):
        return "orig"