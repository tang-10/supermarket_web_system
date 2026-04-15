import numpy as np

from src.models.classification import ConvNeXtFeatureModel
from src.models.segmentation import YoloSegmentationModel
from src.utils.config_utils import cfg
from src.entities.schemas import DetectResult


class ModelManager:
    def __init__(self):
        print(">>> 开始初始化模型管家，加载权重...")
        # 实例化分割模型
        self.seg_model = YoloSegmentationModel(cfg.SEG_MODEL_PATH)
        # 批量实例化特征提取模型
        self.cls_models = {}
        for big_category, path in cfg.BIG_TO_MODEL_MAP.items():
            self.cls_models[big_category] = ConvNeXtFeatureModel(path)
        print(">>> 所有模型加载完毕！")

    def detect_and_segment(self, frame: np.ndarray) -> list[DetectResult]:
        """
        调用分割模型检测大类并生成切图
        :param frame: 视频原帧
        :return: 包含框、大类名、裁切小图的结果列表
        """
        return self.seg_model.predict(frame)

    def extract_feature(self, big_category: str, crop_img: np.ndarray) -> np.ndarray:
        """
        根据大类名，自动路由到对应的细分类模型，提取商品特征
        :param big_category: 大类名称 (如 'bagged', 'bottled')
        :param crop_img: 商品的局部裁切图像 (BGR Numpy)
        :return: 提取出的 1D 特征向量
        """
        target_model = self.cls_models.get(big_category)
        if not target_model:
            print(f"[警告] 未找到大类 {big_category} 对于的特征提取模型！")
            return np.zeros(cfg.FEATURES_DIM, dtype=np.float32)

        return target_model.predict(crop_img)
