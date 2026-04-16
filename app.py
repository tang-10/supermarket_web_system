import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from src.api.dependencies import init_app_context
from src.api.routers import recognition, registration, videos


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理器"""
    print(">>> 启动阶段: 加载全局 AI 模型与数据库 (请耐心等待...)")
    init_app_context()
    print(">>> 启动阶段: 资源加载完毕，Web 服务已就绪！")
    yield
    print(">>> 关闭阶段: 正在释放显存与连接资源...")


app = FastAPI(title="商超CV识别系统", lifespan=lifespan)

# 跨域配置 (CORS)，允许前端 localhost:5173 访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应替换为真实的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos.router, prefix="/api/videos", tags=["视频管理"])
# 挂载静态视频目录，前端可以直接回放 data/static_videos/ 里的 MP4
os.makedirs("data/static_videos", exist_ok=True)
app.mount("/static", StaticFiles(directory="data/static_videos"), name="static")

# 挂载路由，前缀对应了前端 Axios 的请求路径
app.include_router(recognition.router, prefix="/api/recognition", tags=["实时识别"])
app.include_router(registration.router, prefix="/api/registration", tags=["商品注册"])

# 挂载 WebSocket (注意：这里的路径要和前端 new WebSocket 的地址对应)
# /ws/recognition
app.include_router(recognition.router, prefix="/ws", tags=["WebSocket"])

if __name__ == "__main__":
    import uvicorn

    # uvicorn 启动
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
