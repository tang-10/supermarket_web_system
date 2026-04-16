import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File

router = APIRouter()


@router.post("/upload_test")
async def upload_test_video(video: UploadFile = File(...)):
    # 将视频存放到 data/test_videos 目录下
    save_dir = Path("data/test_videos")
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = save_dir / video.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    return {"filename": video.filename}
