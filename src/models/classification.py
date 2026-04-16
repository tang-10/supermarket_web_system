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

    @torch.no_grad()
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
            x_np = x.detach().cpu().numpy().reshape(-1).copy()

            # 释放不需要的 tensor 引用
            del image_tensor
            del x
        return x_np

    @torch.no_grad()
    def predict_batch(self, crop_imgs: list[np.ndarray]) -> np.ndarray:
        """
        输入分割小图列表，返回商品的 [N, 768] 维特征矩阵
        :param crop_imgs: list[np.ndarray] 每一项为 BGR 格式的 numpy 数组
        :return: np.ndarray 形状为 (N, 768)
        """
        if not crop_imgs:
            return np.array([], dtype=np.float32).reshape(0, 768)

        # 1. 批量预处理
        tensors = []
        for img in crop_imgs:
            # BGR -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # ndarray -> PIL
            img_pil = Image.fromarray(img_rgb)
            # Transform 并存入列表
            tensors.append(self.transform(img_pil))

        # 2. 堆叠成一个 Batch Tensor (Shape: [N, 3, 384, 384]) 并送入设备
        image_batch = torch.stack(tensors).to(self.device)

        # 3. 批量推理
        # 使用 self.model.features -> avgpool -> flatten 的结构
        x = self.model.features(image_batch)  # [N, 768, H', W']
        x = self.model.avgpool(x)  # [N, 768, 1, 1]
        x = torch.flatten(x, 1)  # [N, 768]

        # 4. L2 归一化 (保持 dim=1)
        x = F.normalize(x, p=2, dim=1)

        # 5. 转为 Numpy 副本并断开显存联系
        # 保留 (N, 768) 的结构
        x_np = x.detach().cpu().numpy().astype(np.float32).copy()

        # 6. 显式释放显存资源
        del image_batch
        del x

        return x_np
