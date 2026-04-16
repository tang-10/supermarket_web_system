import struct
import json
import cv2
import asyncio
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
        from pathlib import Path

        source = str(Path("data/test_videos") / video)
    else:
        source = int(video)

    pipeline = RealtimeRecognitionPipeline(
        AppContext.model_mgr, AppContext.vector_db, AppContext.product_db
    )
    cap = cv2.VideoCapture(source)
    is_paused = False

    try:
        while cap.isOpened():
            # 检查暂停状态或接收前端消息
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                if message == "pause":
                    is_paused = True
                elif message == "resume":
                    is_paused = False
            except asyncio.TimeoutError:
                pass  # 没有消息，继续处理

            if is_paused:
                await asyncio.sleep(0.1)  # 暂停时降低CPU使用
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            results = pipeline.process_frame(frame)

            # 编码图片为原始字节 (不使用 Base64)
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            img_bytes = buffer.tobytes()

            # 准备 JSON 数据
            res_data = [res.to_dict() for res in results]
            json_bytes = json.dumps({"results": res_data}).encode("utf-8")

            # 构造二进制包：Header(4字节JSON长度) + JSON + Image
            header = struct.pack("!I", len(json_bytes))

            # 发送二进制数据包
            await websocket.send_bytes(header + json_bytes + img_bytes)

            # 控制频率，防止发送堆积
            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"WS Error: {e}")
    finally:
        cap.release()
