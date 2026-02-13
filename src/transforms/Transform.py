from abc import ABC, abstractmethod

class Transform(ABC):
    @abstractmethod
    def __call__ (self, img):
        pass