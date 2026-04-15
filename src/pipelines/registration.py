from collections import defaultdict
import cv2
import numpy as np

from src.models.manager import ModelManager
from src.db.vector_db import VectorDBManager
from src.db.product_db import ProductDBManager
from src.entities.schemas import ProductRegisterRequest
from src.utils.image_utils import ImageProcessor
from src.utils.feature_utils import FeatureProcessor
from src.utils.config_utils import cfg


class ProductRegistrationPipeline:
    def __init__(
        self,
        model_mgr: ModelManager,
        vector_db: VectorDBManager,
        product_db: ProductDBManager,
    ):
        # 依赖注入
        self.model_mgr = model_mgr
        self.vector_db = vector_db
        self.product_db = product_db

    def run(self, video_path: str, req: ProductRegisterRequest) -> bool:
        """
        执行商品注册全流程
        :param video_path: 前端上传暂存到本地的视频路径
        :param req: 前端传来的表单数据 (sku, product_name, big_category)
        """
        print(f"=== 开始后台注册任务: {req.product_name} ({req.sku}) ===")
        cap = cv2.VideoCapture(video_path)
        raw_frames = []
        raw_crops = []
        frame_count = 0
        # ---------------------------------------------------------
        # 阶段 1：视频抽帧与大类侦测
        # ---------------------------------------------------------
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 每隔 2 帧抽一次，避免数据量过大且极度重复
            if frame_count % 2 == 0:
                raw_frames.append(frame)
            frame_count += 1
        cap.release()

        # 对抽出的帧进行哈希去重
        dedup_frames = ImageProcessor.deduplicate_images(raw_frames)
        print(f"抽帧完成，原始帧数: {len(raw_frames)}, 去重后帧数: {len(dedup_frames)}")

        # 用于置信度聚合的字典
        category_conf_sum = defaultdict(float)
        # 用于按大类分组保存切图的字典
        category_crops = defaultdict(list)

        for frame in dedup_frames:
            detect_results = self.model_mgr.detect_and_segment(frame)

            for detect in detect_results:
                confidence = getattr(detect, "seg_conf", 1.0)
                # 排除手的识别结果
                if detect.big_category != "hand":
                    # 1. 累加该大类的置信度得分
                    category_conf_sum[detect.big_category] += confidence
                    # 2. 将切图放入对应大类的列表中
                    category_crops[detect.big_category].append(detect.crop_img)

        if not category_conf_sum:
            print("注册失败：视频中未检测到任何大类的商品。")
            return

        # 置信度聚合，找出累积得分最高的大类
        best_big_category = max(category_conf_sum, key=category_conf_sum.get)

        print("置信度聚合结果：")
        for cls_name, score in category_conf_sum.items():
            print(f" - {cls_name}: {score:.2f}")
        print(f"==> 最终决定送入分类模型的大类为: {best_big_category}")

        if best_big_category not in cfg.BIG_CATEGORIES:
            print(f"注册失败：{best_big_category}没有对应的分类模型。")
            return

        # 只保留最终大类的切图，过滤掉视频中的误检噪点
        raw_crops = category_crops[best_big_category]

        if not raw_crops:
            print("注册失败：视频中未检测到任何大类的商品。")
            return

        # ---------------------------------------------------------
        # 阶段 2：数据去重与特征提取
        # ---------------------------------------------------------
        # 对切图进行哈希去重
        # unique_crops = ImageProcessor.deduplicate_images(raw_crops)
        # print(
        #     f"切图去重完成，初始切图数量: {len(raw_crops)}, 去重后切图数量: {len(unique_crops)}"
        # )
        unique_crops = raw_crops

        raw_features = []
        for crop in unique_crops:
            feat = self.model_mgr.extract_feature(best_big_category, crop)
            raw_features.append(feat)

        # ---------------------------------------------------------
        # 阶段 3：特征聚类与入库
        # ---------------------------------------------------------
        # 使用K-Means聚类从特征向量列表中提取最具代表性的向量
        # 剩余特征向量数大于8的时候再筛选
        if len(raw_features) > 8:
            representative_features = FeatureProcessor.get_representative_feature(
                raw_features
            )
            print(
                f"K-Means聚类完成,聚类前数量: {len(unique_crops)}, 聚类后切图数量: {len(representative_features)}"
            )
        else:
            representative_features = raw_features
        # 存入 MySQL，得到商品ID
        req.big_category = best_big_category
        product_id = self.product_db.insert_product(req)

        if product_id:
            # 存入 FAISS 向量库
            self.vector_db.insert(np.stack(representative_features, axis=0), product_id)
        else:
            print("没有product_id无法录入向量数据库。")

        print(f"=== 注册任务结束: {req.product_name} | MySQL ID:{product_id} ===")
        return True
