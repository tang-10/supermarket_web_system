import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


class FeatureProcessor:
    @staticmethod
    def get_representative_feature(feature_list, n_clusters=8):
        """
        使用K-Means聚类从特征向量列表中提取最具代表性的向量。

        :param feature_list: 特征向量列表 (list of np.ndarray)，每个向量维度一致
        :param n_clusters: 希望保留的特征数量
        :return: 选出的代表性向量列表
        """
        # 1. 转换为矩阵格式
        data = np.array(feature_list)

        # 防止数据量小于设定的聚类数
        num_samples = data.shape[0]
        n_clusters = min(n_clusters, num_samples)

        # 2. 运行 K-Means
        # 使用 'k-means++' 优化中心初始化，n_init='auto' 自动选择最佳尝试次数
        kmeans = KMeans(
            n_clusters=n_clusters, init="k-means++", n_init="auto", random_state=42
        )
        kmeans.fit(data)

        # 3. 找到距离每个簇中心最近的实际特征向量
        # cluster_centers_ 是聚类生成的中心点，不一定存在于原数据集中
        # 从原数据集中找到距离这些中心点最近的真实向量
        closest_indices, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, data
        )

        # 4. 获取对应的原始向量
        representative_features = [data[i] for i in closest_indices]

        return representative_features
