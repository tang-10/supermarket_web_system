from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np
import torch

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

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> list[DetectResult]:
        """对输入帧进行目标侦测和分割，输出包含边界框，大分类，分割图，置信度的侦测结果列表"""
        detect_results: list[DetectResult] = []
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
            stream=True,  # 启用生成器模式，处理完立即释放
        )
        detect_results: list[DetectResult] = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            # 【关键优化】：一次性将所有需要的数据搬运到 CPU 并断开梯度
            # 必须使用 .copy()，否则这些 Numpy 数组会一直引用显存中的大块数据
            boxes_all = result.boxes.xyxy.cpu().numpy().copy()
            conf_all = result.boxes.conf.cpu().numpy().copy()
            cls_all = result.boxes.cls.cpu().numpy().copy()

            # 处理 track_id
            if result.boxes.id is not None:
                ids_all = result.boxes.id.cpu().numpy().copy()
            else:
                ids_all = np.full(len(boxes_all), -1)

            # 处理 Masks
            if result.masks is not None:
                masks_all = result.masks.data.cpu().numpy().copy()
            else:
                continue  # 没有 Mask 则无法分割图片，跳过

            big_category_names = result.names

            for i in range(len(boxes_all)):
                bbox = boxes_all[i]
                mask = masks_all[i]
                track_id = ids_all[i]
                conf = conf_all[i]
                cls_id = cls_all[i]

                if track_id is None or track_id < 0:
                    continue
                # 还原坐标
                x1 = max((bbox[0] - pad_left) / scale, 0)
                y1 = max((bbox[1] - pad_top) / scale, 0)
                x2 = max((bbox[2] - pad_left) / scale, 0)
                y2 = max((bbox[3] - pad_top) / scale, 0)

                bbox = [int(x1), int(y1), int(x2), int(y2)]

                # 得到分割后的商品小图
                crop_img = ImageProcessor.crop_with_mask(
                    frame, mask, bbox, scale, pad_top, pad_left
                )
                if crop_img is None:
                    continue
                big_category = big_category_names[int(cls_id)]
                detect_ret = DetectResult(
                    bbox=bbox,
                    big_category=big_category,
                    crop_img=crop_img,
                    seg_conf=float(conf.item()),
                )
                detect_results.append(detect_ret)
        del results

        return detect_results
