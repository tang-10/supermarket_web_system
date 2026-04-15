from abc import ABC, abstractmethod
import numpy as np
import torch
from src.entities.schemas import DetectResult


class BaseModel(ABC):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_model()

    @abstractmethod
    def load_model(self):
        pass


class BaseSegmentationModel(BaseModel):
    """分割模型抽象类"""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[DetectResult]:
        pass


class BaseClassificationModel(BaseModel):
    """分类/特征提取模型抽象类"""

    @abstractmethod
    def predict(self, crop_img: np.ndarray) -> np.ndarray:
        pass
