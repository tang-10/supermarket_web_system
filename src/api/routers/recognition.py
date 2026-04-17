import struct
import json
import cv2
import asyncio
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pathlib import Path

from src.api.dependencies import AppContext
from src.pipelines.recognition import RealtimeRecognitionPipeline
from src.api.ws_manager import ws_manager

router = APIRouter()


@router.websocket("/recognition")
async def websocket_endpoint(websocket: WebSocket, video: str = "0"):
    await ws_manager.connect(websocket)

    if video.startswith("http://") or video.startswith("https://"):
        source = video  # 手机 IP 摄像头地址
    elif not video.isdigit():
        source = str(Path("data/test_videos") / video)
    else:
        source = int(video)

    pipeline = RealtimeRecognitionPipeline(
        AppContext.model_mgr, AppContext.vector_db, AppContext.product_db
    )
    cap = cv2.VideoCapture(source)

    # 尽可能减小底层缓冲区，防止延迟积累
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    is_paused = False

    print(f"[*] WebSocket 视频流开启: 源={source}")

    try:
        while cap.isOpened():
            # 1. 以极短的超时检查前端是否发来了暂停/恢复指令
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(), timeout=0.005
                )
                if message == "pause":
                    is_paused = True
                    print("[*] 收到暂停指令")
                elif message == "resume":
                    is_paused = False
                    print("[*] 收到恢复指令")
            except asyncio.TimeoutError:
                pass  # 没有消息，正常流转

            # 2. 暂停状态处理
            if is_paused:
                await asyncio.sleep(0.1)
                continue

            loop_start = time.time()

            # 3. 清空积压的缓冲区，永远只取最新的一帧
            if "http" in str(source):
                # 连抓几次丢弃旧图，保持画面绝对实时
                cap.grab()
                cap.grab()

            ret, frame = cap.read()
            if not ret:
                # 文件视频播放结束后停止，不循环重播
                if isinstance(source, str) and not source.startswith("http"):
                    break
                # 对摄像头或网络流，继续等待下一帧
                await asyncio.sleep(0.05)
                continue

            # 4. 后台线程执行推理，不阻塞主异步循环
            process_start = time.time()
            results = await asyncio.to_thread(pipeline.process_frame, frame)
            process_time = time.time() - process_start

            # 5. 编码与打包二进制数据
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            img_bytes = buffer.tobytes()

            res_data = [res.to_dict() for res in results]
            json_bytes = json.dumps({"results": res_data}).encode("utf-8")
            header = struct.pack("!I", len(json_bytes))

            # 6. 发送数据包
            await websocket.send_bytes(header + json_bytes + img_bytes)

            total_time = time.time() - loop_start
            actual_fps = 1.0 / total_time if total_time > 0 else 0

            print(
                f"[流控] 推理:{process_time * 1000:.1f}ms | 总耗时:{total_time * 1000:.1f}ms | FPS: {actual_fps:.1f}"
            )

    except WebSocketDisconnect:
        print("[*] 前端断开 WebSocket 连接")
    except Exception as e:
        print(f"[!] WS 异常: {e}")
    finally:
        cap.release()
        ws_manager.disconnect(websocket)
