import argparse
from src.models.manager import ModelManager
from src.db.vector_db import VectorDBManager
from src.db.product_db import ProductDBManager
from src.pipelines.recognition import RealtimeRecognitionPipeline
from src.pipelines.registration import ProductRegistrationPipeline
from src.entities.schemas import ProductRegisterRequest


def test_recognition(video_input: str, show_gui: bool):
    print(">>> 启动测试模式: 实时侦测识别")
    model_mgr = ModelManager()
    vector_db = VectorDBManager()
    product_db = ProductDBManager()

    pipeline = RealtimeRecognitionPipeline(model_mgr, vector_db, product_db)

    # 自动识别是摄像头流还是视频文件
    if video_input.isdigit():
        # 是数字，说明调用的是摄像头
        camera_id = int(video_input)
        output_file = f"data/test_videos/camera_{camera_id}_record.mp4"
        pipeline.run_camera(
            camera_id=camera_id, output_path=output_file, show_gui=show_gui
        )
    else:
        # 是字符串，说明是本地的 mp4 视频文件
        output_file = "data/test_videos/recognition_result.mp4"
        # 视频文件模式默认不开启预览，专为 Ubuntu 服务器输出
        pipeline.run_video_file(video_path=video_input, output_path=output_file)


def test_registration(video_path: str, sku: str, name: str, price: float):
    print(f">>> 启动测试模式: 商品注册 [{name}]")
    # 1. 基础组件初始化
    model_mgr = ModelManager()
    vector_db = VectorDBManager()
    product_db = ProductDBManager()

    # 2. 实例化流水线
    pipeline = ProductRegistrationPipeline(model_mgr, vector_db, product_db)

    # 3. 组装前端发来的请求结构体
    req = ProductRegisterRequest(sku=sku, product_name=name, price=price)

    # 4. 运行后台注册流
    pipeline.run(video_path, req)


if __name__ == "__main__":
    """
    --------------------------------------------------------------------------------
    1. 模式：实时识别 (Recognize)
    --------------------------------------------------------------------------------
    描述：处理输入源并输出带有识别框与分类信息的视频文件。

        A. 处理本地视频文件 ：
        python main.py --mode recognize --video data/test_videos/demo.mp4
        [说明] 系统将自动处理该视频并保存为 data/test_videos/recognition_result.mp4

        B. 处理实时摄像头流：
        python main.py --mode recognize --video 0
        [说明] 实时读取设备 ID 为 0 的摄像头，并将检测流录制为 
                data/test_videos/camera_0_record.mp4，按下 Ctrl+C 停止。

        C. 本地调试预览 (仅限拥有图形桌面的环境)：
        python main.py --mode recognize --video 0 --show
        [说明] 开启 OpenCV 窗口进行实时可视化预览。

    --------------------------------------------------------------------------------
    2. 模式：商品注册 (Register)
    --------------------------------------------------------------------------------
    描述：从指定视频中抽帧、去重、聚类提取特征，将商品信息存入 MySQL 并建立向量库索引。

        命令示例：
        python main.py --mode register \
                        --video data/test_videos/coca_cola.mp4 \
                        --sku cola_330 \
                        --name "可口可乐330ml" \
                        --price 5.0

        参数说明：
        --video      : 注册用素材视频路径
        --sku        : 商品的唯一编码 (唯一主键)
        --name       : 商品中文展示名称
        --price      : 商品单价
    """

    parser = argparse.ArgumentParser(description="商超系统本地测试脚本")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["recognize", "register"],
        help="测试模式：recognize(识别) 或 register(注册)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="0",
        help="填 0~9 调用对应的摄像头，或填视频的本地路径(如 data/abc.mp4)",
    )

    # 图形界面开关参数
    parser.add_argument(
        "--show",
        action="store_true",
        help="开启本地弹窗预览 (注意: Ubuntu 无桌面环境下请勿加此参数！)",
    )

    # 注册专用参数
    parser.add_argument("--sku", type=str, default="test_sku_001")
    parser.add_argument("--name", type=str, default="测试商品")
    parser.add_argument(
        "--price",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    if args.mode == "recognize":
        test_recognition(args.video, args.show)
    elif args.mode == "register":
        if args.video.isdigit():
            print("[错误] 注册商品必须提供视频文件路径，不能使用实时摄像头！")
            exit(1)
        test_registration(args.video, args.sku, args.name, args.price)
