import faiss
import numpy as np
from pathlib import Path

from src.utils.config_utils import cfg


class VectorDBManager:
    """faiss 向量数据库管理"""

    def __init__(self):
        self.dim = cfg.FEATURES_DIM
        self.index_path = Path(cfg.FAISS_INDEX_PATH)
        self.index = None
        self._load_or_create_index()

    def _load_or_create_index(self):
        if self.index_path.exists():
            print(f"正在加载向量库: {self.index_path}")
            self.index = faiss.read_index(self.index_path.as_posix())
            print(f"已加载现有FAISS索引 当前向量总数: {self.index.ntotal}")
        else:
            print("未发现现有向量库 正在创建全新索引 (余弦相似度索引)...")
            # 使用 L2 欧式距离 (IndexFlatL2) 或内积相似度 (IndexFlatIP)
            base_index = faiss.IndexFlatIP(self.dim)  # Inner Product + 归一化 = 余弦
            self.index = faiss.IndexIDMap2(base_index)  # 支持同一个ID多次添加向量

    def insert(self, features: np.ndarray, mysql_id: int):
        """插入新的商品特征
        :param feature: (N,dim) 1D或者2D numpy 数组
        :param mysql_id: 商品ID 和product DB 一致
        """
        # 处理向量（float32）
        if features.ndim == 1:
            features = features.reshape(1, -1)
        features = features.astype(np.float32)

        if self.index is None:
            self._load_or_create_index()

        # 同一个ID添加多条向量
        ids = np.full(features.shape[0], mysql_id, dtype=np.int64)
        self.index.add_with_ids(features, ids)

        # 保存索引
        faiss.write_index(self.index, self.index_path.as_posix())
        print(f"已为商品 ID={mysql_id} 追加 {features.shape[0]} 条向量")

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        在库中检索最相似的商品
        :return: list[{"id":id, "score":score}] 若库为空则返回 [{"id": None, "score": 0.0}]
        """
        if self.index.ntotal == 0:
            return [{"id": None, "score": 0.0}]

        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector)

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1).astype(np.float32)
        elif query_vector.ndim == 2:
            query_vector = query_vector.astype(np.float32)
        else:
            raise ValueError(
                f"query_vector 维度错误: {query_vector.shape}，必须是 1D 或 2D"
            )

        # D=相似度分数，I=ID列表
        D, I = self.index.search(query_vector, top_k)  # noqa: E741
        topk_results = [{"id": int(i), "score": float(d)} for i, d in zip(I[0], D[0])]
        return topk_results

    def search_batch(
        self, query_vectors: np.ndarray, top_k: int = 1
    ) -> list[list[dict]]:
        """
        高度鲁棒的批量检索版
        """
        # 1. 基础存在性检查
        if self.index is None or self.index.ntotal == 0:
            num_queries = query_vectors.shape[0] if query_vectors.ndim > 0 else 1
            return [[{"id": None, "score": 0.0}] * top_k for _ in range(num_queries)]

        # 2. 确保是 Numpy 数组且重新赋值
        vectors = np.array(query_vectors)

        # 3. 强制转换为 2D 矩阵 (N, dim)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # 4. 【核心修复】：校验维度是否与索引一致
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"FAISS维度不匹配: 索引要求 {self.dim}, 输入为 {vectors.shape[1]}"
            )

        # 5. 【核心修复】：确保内存连续性 (C_CONTIGUOUS)
        # 这一步能解决 90% 的 AssertionError
        if not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors)

        # 6. 【核心修复】：强制 float32 并处理 NaN
        vectors = vectors.astype(np.float32)
        if np.any(np.isnan(vectors)):
            vectors = np.nan_to_num(vectors)  # 将 NaN 替换为 0

        # 7. 执行检索
        try:
            D, I = self.index.search(vectors, top_k)
        except Exception as e:
            print(f"[FAISS底层错误] 检索失败: {e}")
            return [
                [{"id": None, "score": 0.0}] * top_k for _ in range(vectors.shape[0])
            ]

        # 8. 组装结果
        all_results = []
        for row_i, row_d in zip(I, D):
            item_top_k = []
            for i, d in zip(row_i, row_d):
                item_top_k.append(
                    {"id": int(i) if i != -1 else None, "score": float(d)}
                )
            all_results.append(item_top_k)

        return all_results
