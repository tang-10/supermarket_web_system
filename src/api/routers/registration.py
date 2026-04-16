import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks

from src.api.dependencies import AppContext
from src.pipelines.registration import ProductRegistrationPipeline
from src.entities.schemas import ProductRegisterRequest

router = APIRouter()


def background_registration_task(video_path: str, req: ProductRegisterRequest):
    """运行在后台的超长耗时任务"""
    pipeline = ProductRegistrationPipeline(
        AppContext.model_mgr, AppContext.vector_db, AppContext.product_db
    )
    pipeline.run(video_path, req)


@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    sku: str = Form(...),
    product_name: str = Form(...),
    price: float = Form(...),
    big_category: str = Form(None),  # 可为空，由 AI 自动决定
):
    # 1. 确保临时目录存在
    temp_dir = Path("data/test_videos")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 2. 将前端传来的视频存入本地硬盘
    temp_file_path = temp_dir / video.filename
    with open(temp_file_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # 3. 构造请求实体
    req = ProductRegisterRequest(
        sku=sku, product_name=product_name, price=price, big_category=big_category
    )

    # 4. 把任务塞给后台异步执行，不阻塞前端！
    background_tasks.add_task(background_registration_task, str(temp_file_path), req)

    return {
        "status": "processing",
        "message": "视频已接收，后台 AI 正在疯狂提取特征中...",
    }
