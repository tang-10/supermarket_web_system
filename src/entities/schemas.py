from pydantic import BaseModel
from dataclasses import dataclass


# ==========================================
# 1. 内部 DTO (Data Transfer Objects)
# 使用 dataclass，适合内部传递包含图像(numpy)的复杂对象
# ==========================================
@dataclass
class DetectResult:
    """单帧图像中的单个商品检测结果"""

    bbox: list  # 边界框[x1, y1, x2, y2]
    big_category: str  # 四大类（袋、瓶、罐、盒）(bagged, bottled...)
    crop_img: object | None = None  # 裁剪出的商品小图 (BGR numpy array)
    seg_conf: float = 0.0  # 大品类置信度


@dataclass
class RecognizeResult(DetectResult):
    """继承检测结果，追加细分类和置信度"""

    fine_class: str = "unknown"  # 细分类名称（如 'coke_500ml'）
    product_name: str = "未注册商品"  # 商品中文名
    score: float = 0.0  # 向量检索相似度得分


# ==========================================
# 2. Web API 数据模型 (FastAPI 专用)
# 使用 Pydantic BaseModel，自动进行 HTTP 请求体校验
# ==========================================
class ProductRegisterRequest(BaseModel):
    """前端发起的注册请求参数"""

    product_name: str
    price: float
    sku: str
    big_category: str | None = None


class APIResponse(BaseModel):
    """统一的 HTTP 响应结构"""

    status: int
    message: str
    data: dict | None
