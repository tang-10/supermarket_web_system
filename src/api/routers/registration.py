import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException

from src.api.dependencies import AppContext
from src.pipelines.registration import ProductRegistrationPipeline
from src.entities.schemas import ProductRegisterRequest

router = APIRouter()

# 任务状态存储 (内存中，生产环境建议用Redis或数据库)
task_status = {}


def background_registration_task(
    task_id: str, video_path: str, req: ProductRegisterRequest
):
    """运行在后台的超长耗时任务"""
    try:
        # 更新任务状态为处理中
        task_status[task_id] = {"status": "processing", "message": "正在提取特征..."}

        pipeline = ProductRegistrationPipeline(
            AppContext.model_mgr, AppContext.vector_db, AppContext.product_db
        )
        ret = pipeline.run(video_path, req)

        if ret:
            task_status[task_id] = {"status": "completed", "message": "注册成功"}
        else:
            task_status[task_id] = {"status": "failed", "message": "注册失败"}

    except Exception as e:
        task_status[task_id] = {"status": "failed", "message": f"注册失败: {str(e)}"}
    finally:
        # 清理临时文件
        try:
            Path(video_path).unlink(missing_ok=True)
        except:
            pass


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
    temp_dir = Path("data/temp_videos")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 2. 将前端传来的视频存入本地硬盘
    temp_file_path = temp_dir / f"{uuid.uuid4()}_{video.filename}"
    with open(temp_file_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # 3. 构造请求实体
    req = ProductRegisterRequest(
        sku=sku, product_name=product_name, price=price, big_category=big_category
    )

    # 4. 生成任务ID并初始化状态
    task_id = str(uuid.uuid4())
    task_status[task_id] = {"status": "uploading", "message": "视频上传成功"}

    # 5. 把任务塞给后台异步执行，不阻塞前端！
    background_tasks.add_task(
        background_registration_task, task_id, str(temp_file_path), req
    )

    return {
        "task_id": task_id,
        "status": "uploading",
        "message": "视频上传成功，等待后台提取特征...",
    }


@router.get("/status/{task_id}")
async def get_registration_status(task_id: str):
    """查询注册任务状态"""
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="任务不存在")

    return task_status[task_id]
