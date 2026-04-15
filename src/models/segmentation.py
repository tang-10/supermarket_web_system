from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np

from src.models.base import BaseSegmentationModel
from src.entities.schemas import DetectResult
from src.utils.config_utils import cfg
from src.utils.image_utils import ImageProcessor


class YoloSegmentationModel(BaseSegmentationModel):
    def load_model(self):
        print(f"正在加载 YOLO 分割模型:{self.model_path}")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        print(" YOLO 分割模型加载成功！")

    def predict(self, frame: np.ndarray) -> list[DetectResult]:
        """对输入帧进行目标侦测和分割，输出包含边界框，大分类，分割图，置信度的侦测结果列表"""
        # 预处理frame:Resize
        if (
            frame.shape[0] == cfg.SEG_INPUT_SIZE
            and frame.shape[1] == cfg.SEG_INPUT_SIZE
        ):
            resized_frame = frame
            scale, pad_top, pad_left = 1, 0, 0
        else:
            resized_frame, scale, pad_top, pad_left = ImageProcessor.letterbox_resize(
                frame, cfg.SEG_INPUT_SIZE
            )
        # 使用YOLO自带的tracker侦测目标
        results: list[Results] = self.model.track(
            resized_frame,
            persist=True,  # 保持跨帧 ID 连续
            imgsz=cfg.SEG_INPUT_SIZE,
            conf=cfg.SEG_CONF_THRESH,
            iou=cfg.SEG_IOU_THRESH,
            verbose=False,  # 关闭详细日志输出
        )
        detect_results: list[DetectResult] = []
        for result in results:
            big_category_names = result.names
            if len(result.boxes) == 0:
                return None
            for bbox, mask, track_id, conf, cls_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.masks.data.cpu().numpy(),
                result.boxes.id.cpu().numpy()
                if result.boxes.id is not None
                else np.full(len(result.boxes), -1),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                if track_id is None or track_id < 0:
                    continue
                # 还原坐标
                bbox[0] = np.maximum((bbox[0] - pad_left) / scale, 0)
                bbox[1] = np.maximum((bbox[1] - pad_top) / scale, 0)
                bbox[2] = np.maximum((bbox[2] - pad_left) / scale, 0)
                bbox[3] = np.maximum((bbox[3] - pad_top) / scale, 0)
                # 得到分割后的商品小图
                crop_img = ImageProcessor.crop_with_mask(
                    frame, mask, bbox, scale, pad_top, pad_left
                )
                big_category = big_category_names[int(cls_id)]
                detect_ret = DetectResult(
                    bbox=bbox.astype(int).tolist(),
                    big_category=big_category,
                    crop_img=crop_img,
                    seg_conf=conf.item(),
                )
                detect_results.append(detect_ret)
        return detect_results
