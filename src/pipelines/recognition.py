import numpy as np
import cv2
import time
from collections import defaultdict

from src.models.manager import ModelManager
from src.db.vector_db import VectorDBManager
from src.db.product_db import ProductDBManager
from src.entities.schemas import RecognizeResult
from src.utils.image_utils import ImageProcessor
from src.utils.config_utils import cfg


class RealtimeRecognitionPipeline:
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
        self.unkown_cls_conf_thresh = cfg.UNKONWN_CLS_CONF_THRESH

        # 特征缓存字典
        self.track_cache = {}

    def process_frame(self, frame) -> list[RecognizeResult]:
        """处理输入帧，输出识别结果列表"""
        t_start = time.time()

        # --- YOLO 跟踪侦测 ---
        raw_detect_results = self.model_mgr.detect_and_segment(frame)
        t_yolo = time.time()

        if not raw_detect_results:
            self.track_cache.clear()
            return []

        # 过滤无效大类并确保 track_id 存在
        detect_results = [
            det for det in raw_detect_results if det.big_category in cfg.BIG_CATEGORIES
        ]

        if not detect_results:
            return []

        current_track_ids = set()
        final_results = [None] * len(detect_results)  # 用于存放最终结果，保持顺序

        pending_indices = []  # 存储需要重新推理的索引

        # --- 查缓存分流 ---
        for i, det in enumerate(detect_results):
            tid = det.track_id
            if tid != -1:
                current_track_ids.add(tid)

            # 命中缓存判断
            if tid != -1 and tid in self.track_cache:
                cached_data = self.track_cache[tid]

                if (
                    cached_data["fine_class"] == "unknown"
                    or cached_data["score"] < 0.85
                ):
                    pending_indices.append(i)
                else:
                    # 命中缓存：提取识别信息，但使用当前帧的 bbox 和 seg_conf
                    final_results[i] = RecognizeResult(
                        bbox=det.bbox,
                        big_category=det.big_category,
                        seg_conf=det.seg_conf,
                        track_id=tid,
                        fine_class=cached_data["fine_class"],
                        product_name=cached_data["product_name"],
                        sku=cached_data["sku"],
                        price=cached_data["price"],
                        score=cached_data["score"],
                    )
            else:
                pending_indices.append(i)

        t_convnext_total = 0
        t_db_total = 0

        # --- 仅对 "新目标" 进行重度批量推理 ---
        if pending_indices:
            # 提取需要推理的 crops
            pending_detects = [detect_results[idx] for idx in pending_indices]

            # 按大类分组
            category_groups = defaultdict(list)
            for i, det in enumerate(pending_detects):
                category_groups[det.big_category].append(i)

            all_features = [None] * len(pending_detects)

            t0 = time.time()
            for big_category, group_indices in category_groups.items():
                crops = [pending_detects[idx].crop_img for idx in group_indices]
                batch_feats = self.model_mgr.extract_features_batch(big_category, crops)
                for i, idx in enumerate(group_indices):
                    all_features[idx] = batch_feats[i]
            t_convnext_total = time.time() - t0

            t1 = time.time()
            # 批量检索
            valid_features = np.stack(all_features)
            search_results = self.vector_db.search_batch(valid_features, top_k=1)

            # 批量查库
            valid_pids = [
                res[0]["id"]
                for res in search_results
                if res[0]["id"] is not None
                and res[0]["score"] >= self.unkown_cls_conf_thresh
            ]
            product_map = self.product_db.get_product_by_ids_batch(valid_pids)

            # 将新推理的结果填入 final_results 并更新缓存
            for i, idx_in_detect in enumerate(pending_indices):
                det = detect_results[idx_in_detect]
                score = float(search_results[i][0]["score"])
                pid = search_results[i][0]["id"]

                info = product_map.get(pid) if pid is not None else None

                if info and score >= self.unkown_cls_conf_thresh:
                    res_obj = RecognizeResult(
                        bbox=det.bbox,
                        big_category=det.big_category,
                        seg_conf=det.seg_conf,
                        track_id=det.track_id,
                        fine_class=info["fine_class"],
                        product_name=info["product_name"],
                        sku=info["sku"],
                        price=float(info.get("unit_price", 0.0)),
                        score=score,
                    )
                else:
                    res_obj = RecognizeResult(
                        bbox=det.bbox,
                        big_category=det.big_category,
                        seg_conf=det.seg_conf,
                        track_id=det.track_id,
                        fine_class="unknown",
                        sku="unknown",
                        product_name="未注册商品",
                        price=0.0,
                        score=0.0,
                    )

                final_results[idx_in_detect] = res_obj

                # 更新缓存 (只存关键识别信息，不存 bbox 以免混淆)
                if det.track_id != -1:
                    self.track_cache[det.track_id] = {
                        "fine_class": res_obj.fine_class,
                        "product_name": res_obj.product_name,
                        "sku": res_obj.sku,
                        "price": res_obj.price,
                        "score": res_obj.score,
                    }
            t_db_total = time.time() - t1

        # --- 清理过期缓存 (防止内存泄漏) ---
        # 如果 track_id 不在当前帧里，就从缓存删掉
        expired_ids = [tid for tid in self.track_cache if tid not in current_track_ids]
        for tid in expired_ids:
            del self.track_cache[tid]

        t_end = time.time()
        print(
            f"[耗时拆解] 总:{(t_end - t_start) * 1000:.1f}ms | YOLO:{(t_yolo - t_start) * 1000:.1f}ms | ConvNeXt:{t_convnext_total * 1000:.1f}ms | DB&Faiss:{t_db_total * 1000:.1f}ms"
        )

        return final_results

    def run_video_file(self, video_path: str, output_path: str = "output_result.mp4"):
        """
        模式 1：处理现成的测试视频文件
        :param video_path: 原始视频路径
        :param output_path: 处理后的视频保存路径
        """
        import time
        import os

        print(f"[*] 开始读取视频源: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[错误] 无法打开视频源: {video_path}，请检查路径。")
            return

        # 获取原视频的宽度、高度和帧率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 防止获取不到 fps 导致后续报错
        if fps == 0 or fps != fps:
            fps = 30.0

        print(
            f"[*] 视频信息 - 分辨率: {width}x{height}, 帧率: {fps}, 总帧数: {total_frames}"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # 确保输出目录存在
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        # 初始化 VideoWriter
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"[*] 正在处理视频并保存至: {output_path} (请耐心等待...)")

        frame_idx = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # 调用流水线的单帧处理方法
            recognize_results = self.process_frame(frame)
            # 将识别框画在帧图像上
            self.update_frame(frame, recognize_results)
            # 将画好框的帧写入新视频
            writer.write(frame)

            # 终端打印进度条 (每 10 帧打印一次)
            if frame_idx % 10 == 0 or frame_idx == total_frames:
                elapsed = time.time() - start_time
                fps_processing = frame_idx / elapsed
                print(
                    f"\r[进度] {frame_idx}/{total_frames} 帧 | 处理速度: {fps_processing:.1f} fps",
                    end="",
                )

        print("\n[*] 视频播放结束，正在释放资源...")

        # 释放所有资源
        cap.release()
        writer.release()
        print(f"[*] 成功！带检测框的视频已保存至: {output_path}")

    def run_camera(
        self,
        camera_id: int = 0,
        output_path: str = "camera_record.mp4",
        show_gui: bool = False,
    ):
        """
        模式 2：处理无限时的实时摄像头流
        :param camera_id: 摄像头设备号 (通常为 0)
        :param output_path: 运行期间的视频将录制到此路径
        :param show_gui: 是否开启 cv2 弹窗预览 (Ubuntu 纯命令行环境下必须为 False)
        """
        import cv2
        import os

        print(f"[*] 尝试唤醒摄像头 (设备号 ID: {camera_id})...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(
                f"[错误] 无法获取摄像头 {camera_id} 的画面，请检查硬件连接或 Linux 权限(如 /dev/video0)。"
            )
            return

        # 获取摄像头的默认分辨率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 25.0

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("[*] 摄像头实时识别中...")
        print(f"[*] 侦测结果将同步录制到: {output_path}")
        print("[!] 请在终端按下 'Ctrl + C' 停止测试并保存视频。")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[*] 摄像头流断开。")
                    break

                # 调用流水线的单帧处理方法
                recognize_results = self.process_frame(frame)
                # 将识别框画在帧图像上
                self.update_frame(frame, recognize_results)

                # 边识别边录制写入文件
                writer.write(frame)

                # 如果有 GUI 环境且开启了 show_gui
                if show_gui:
                    cv2.imshow(f"Camera {camera_id} - Live", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[*] 接收到按键退出指令。")
                        break

        except KeyboardInterrupt:
            # Ctrl+C 中断
            print("\n[*] 接收到中止指令(Ctrl+C)，正在安全结束录制...")

        finally:
            cap.release()
            writer.release()
            if show_gui:
                cv2.destroyAllWindows()
            print("[*] 摄像头已关闭，录像保存成功。")

    def update_frame(self, frame, recognize_results):
        color_map = {
            "bagged": [(255, 200, 0), (255, 255, 255)],  # 浅蓝，白
            "bottled": [(200, 200, 200), (60, 122, 86)],  # 灰白，褐色
            "boxed": [(0, 255, 200), (64, 64, 64)],  # 青绿，深灰
            "canned": [(255, 0, 0), (255, 255, 255)],  # 蓝，白
        }
        for result in recognize_results:
            if result.big_category in cfg.BIG_CATEGORIES:
                x1, y1, x2, y2 = result.bbox
                text = f"{result.big_category} {result.fine_class} {result.score:.2f}"
                color = color_map[result.big_category]
                ImageProcessor.draw_box_with_label(frame, (x1, y1, x2, y2), text, color)
