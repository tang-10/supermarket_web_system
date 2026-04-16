import cv2
import numpy as np
from PIL import Image
import imagehash

from src.utils.config_utils import cfg


class ImageProcessor:
    @staticmethod
    def letterbox_resize(
        image: np.ndarray,
        target_size: int | tuple[int, int],
        pad_color: tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """
        等比例缩放 + 居中 letterbox(保持长宽比)

        - 原图比目标大 → 等比例缩小
        - 原图比目标小 → 等比例放大后居中粘贴到黑色（或指定颜色）背景
        - 最终输出固定尺寸的图像

        参数:
            image: 输入图像 (H, W, 3) BGR 或 RGB 均可
            target_size: 目标尺寸 (int 表示正方形，或 (height, width) 元组)
            pad_color: 填充颜色，默认纯黑 (0,0,0)

        返回:
            (target_h, target_w, 3) 的 np.ndarray
        """
        if isinstance(target_size, int):
            target_h = target_w = target_size
        else:
            target_h, target_w = target_size

        h, w = image.shape[:2]

        # 计算等比例缩放因子
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建目标尺寸的画布
        canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)

        # 计算居中位置
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2

        # 粘贴
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        return canvas, scale, pad_top, pad_left

    @staticmethod
    def crop_with_mask(
        frame: np.ndarray,
        mask: np.ndarray,
        box: list,
        frame_scale,
        frame_pad_top,
        frame_pad_left,
        target_size=cfg.CLS_INPUT_SIZE,
    ) -> np.ndarray:
        """根据分割掩码和边界框裁切出商品图像
        优化版：只在 BBox 范围内进行 Mask 运算和 Resize"""

        orig_h, orig_w = frame.shape[:2]
        x1, y1, x2, y2 = box

        # 1. 增加 Padding 并防止越界
        pad = 20
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(orig_w, x2 + pad), min(orig_h, y2 + pad)

        # 2. 直接截取原图 ROI (杜绝 frame.copy()，只引用内存)
        roi_img = frame[cy1:cy2, cx1:cx2]
        if roi_img.size == 0:
            return None

        # 3. 将 ROI 坐标映射回 Mask 的坐标空间 (1024x1024)
        # 公式：Mask_coord = Image_coord * scale + padding
        mx1 = int(cx1 * frame_scale + frame_pad_left)
        my1 = int(cy1 * frame_scale + frame_pad_top)
        mx2 = int(cx2 * frame_scale + frame_pad_left)
        my2 = int(cy2 * frame_scale + frame_pad_top)

        # 4. 截取局部 Mask 并进行二值化
        local_mask = mask[my1:my2, mx1:mx2]
        if local_mask.size == 0:
            return None

        # 5. 只对局部 Mask 进行 Resize (还原到 ROI 的大小)
        local_mask_resized = cv2.resize(
            local_mask,
            (roi_img.shape[1], roi_img.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # 生成二进制掩码
        binary_mask = (local_mask_resized > 0.5).astype(np.uint8)

        # 6. 局部掩码运算 (只在 BBox 区域内进行)
        masked_roi = cv2.bitwise_and(roi_img, roi_img, mask=binary_mask)

        # 7. 最终 Resize 到分类模型输入尺寸 (Letterbox 居中处理)
        h, w = masked_roi.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_roi = cv2.resize(masked_roi, (new_w, new_h))

        # 创建背景并居中贴图
        final_crop = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        top = (target_size - new_h) // 2
        left = (target_size - new_w) // 2
        final_crop[top : top + new_h, left : left + new_w] = resized_roi

        return final_crop

    @staticmethod
    def draw_box_with_label(img, box, text, color, alpha=0.4):
        """
        in-place 原地绘制
        img: 原图
        box: [x1, y1, x2, y2]
        text: id+类别名称+置信度
        color: (B, G, R)背景色+文字色
        alpha: 透明度
        """
        x1, y1, x2, y2 = box

        # 1. 半透明填充
        roi = img[y1:y2, x1:x2]

        if roi.size > 0:  # 防止越界
            overlay = np.full_like(roi, color[0], dtype=np.uint8)
            cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)

        # 2. 边框
        cv2.rectangle(img, (x1, y1), (x2, y2), color[0], 2)

        # 3. 文字 + 背景
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # 防止文字超出顶部
        y_text_top = max(0, y1 - h - 10)

        # 文字背景
        cv2.rectangle(img, (x1, y_text_top), (x1 + w + 6, y1), color[0], -1)

        # 文字
        cv2.putText(
            img,
            text,
            (x1 + 2, y1 - 5),
            font,
            font_scale,
            color[1],
            thickness - 2,
            cv2.LINE_AA,
        )

    @staticmethod
    def deduplicate_images(frame_list, hash_size=8, threshold=5):
        """
        使用dHash算法对图片列表进行去重
        :param frame_list: cv2读取的ndarray列表
        :param hash_size: 哈希尺寸，越大越精确，但计算越慢
        :param threshold: 汉明距离阈值，0表示完全相同，值越大越宽松
        :return: 去重后的ndarray列表
        """
        unique_frames = []
        seen_hashes = []

        for frame in frame_list:
            # 1. 将cv2的BGR图像转换为PIL图像 (imagehash库支持PIL格式)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # 2. 计算dHash
            current_hash = imagehash.dhash(pil_img, hash_size=hash_size)

            # 3. 检查是否重复
            is_duplicate = False
            for h in seen_hashes:
                # 计算汉明距离
                if current_hash - h <= threshold:
                    is_duplicate = True
                    break

            # 4. 如果不重复，添加到结果列表
            if not is_duplicate:
                unique_frames.append(frame)
                seen_hashes.append(current_hash)

        return unique_frames
