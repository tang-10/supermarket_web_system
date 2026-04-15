import torch
from torch.nn import functional as F
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image

from src.models.base import BaseClassificationModel
from src.utils.config_utils import cfg


class ConvNeXtFeatureModel(BaseClassificationModel):
    def load_model(self):
        print(f"正在加载 ConvNeXt 特征提取模型:{self.model_path}")
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=True
        )
        self.class_to_idx = checkpoint["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}  # 反向映射
        num_classes = len(self.class_to_idx)
        self.model = None
        # 创建模型
        self.model = models.convnext_tiny(weights=None)
        self.model.classifier[2] = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.model.classifier[2].in_features, num_classes),
        )
        # 加载训练好的权重
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(" ConvNeXt 特征提取模型加载成功！")
        print(f" 类别数量: {num_classes} | 特征维度: 768")
        print(f" 示例类别: {list(self.class_to_idx.keys())[:5]}...")

        self.transform = transforms.Compose(
            [
                transforms.Resize(cfg.CLS_INPUT_SIZE + 32),
                transforms.CenterCrop(cfg.CLS_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, crop_img: np.ndarray) -> np.ndarray:
        """输入分割小图 返回商品的768维特征向量"""
        if isinstance(crop_img, np.ndarray):
            image_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        else:
            image_tensor = crop_img.to(self.device)
        with torch.no_grad():
            x = self.model.features(image_tensor)  # [B, 768, 12, 12]
            x = self.model.avgpool(x)  # [B, 768, 1, 1]
            x = torch.flatten(x, 1)  # [B, 768]
            x = F.normalize(x, p=2, dim=1)  # L2 归一化
            x = x.squeeze(0).cpu().numpy() if x.is_cuda else x.squeeze(0).numpy()
        return x
