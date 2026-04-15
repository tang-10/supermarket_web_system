import yaml
from pathlib import Path

# 动态获取项目根目录 (假设该文件位于 src/utils/config_utils.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class AppConfig:
    """全局配置单例管理器"""

    _instance = None

    def __new__(cls, config_path: str = "configs/config.yaml"):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._initialize(config_path)
        return cls._instance

    def _initialize(self, config_path: str):
        config_file = PROJECT_ROOT / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件未找到: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # 1. 数据库配置
        self.DB_CONFIG = self._config.get("database", {})

        # 2. 向量库索引路径设定 (自动转换为绝对路径并确保父目录存在)
        self.FAISS_INDEX_PATH = PROJECT_ROOT / self._config["paths"]["faiss_index"]
        self.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

        # 3. 模型参数
        models_cfg = self._config["models"]
        self.FEATURES_DIM = models_cfg["features_dim"]
        self.SEG_INPUT_SIZE = models_cfg["seg_input_size"]
        self.SEG_CONF_THRESH = models_cfg["seg_conf_thresh"]
        self.SEG_IOU_THRESH = models_cfg["seg_iou_thresh"]
        self.CLS_INPUT_SIZE = models_cfg["cls_input_size"]
        self.UNKONWN_CLS_CONF_THRESH = models_cfg["unkonwn_cls_conf_thresh"]

        # 4. 模型权重路径 (全部转换为绝对路径)
        self.SEG_MODEL_PATH = str(PROJECT_ROOT / models_cfg["segmentation"]["weights"])

        self.BIG_TO_MODEL_MAP = {
            category: str(PROJECT_ROOT / path)
            for category, path in models_cfg["classification"].items()
        }

        # 5. 商品及分类信息解析
        self.BIG_CATEGORIES = list(
            self.BIG_TO_MODEL_MAP.keys()
        )  # ["bagged", "bottled", "boxed", "canned"]
        self.PRODUCTS_INFO_DICT = self._config.get("products_info", {})

        # 6. 细分类列表与映射
        self.FINE_CATEGORIES = list(self.PRODUCTS_INFO_DICT.keys())
        self.FINE_TO_BIG_MAP = {
            fine_class: info["big_category"]
            for fine_class, info in self.PRODUCTS_INFO_DICT.items()
        }


# 实例化全局配置对象，供其他模块直接 import
cfg = AppConfig()
