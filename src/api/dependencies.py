from src.models.manager import ModelManager
from src.db.vector_db import VectorDBManager
from src.db.product_db import ProductDBManager


class AppContext:
    """全局应用上下文，用于存储单例对象"""

    model_mgr: ModelManager = None
    vector_db: VectorDBManager = None
    product_db: ProductDBManager = None


def init_app_context():
    """在 FastAPI 启动时调用，一次性加载所有资源"""
    AppContext.model_mgr = ModelManager()
    AppContext.vector_db = VectorDBManager()
    AppContext.product_db = ProductDBManager()
