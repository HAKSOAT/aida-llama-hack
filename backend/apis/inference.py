from abc import ABC, abstractmethod
from typing import List
from PIL import Image


class Inference(ABC):
    @abstractmethod
    def image_to_text(self, input: List[Image.Image]) -> List[str]:
        pass

    @abstractmethod
    def text_to_image(self, input: List[str]) -> List[Image.Image]:
        pass
