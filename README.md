--项目目录结构
supermarket_web_system/
├── .gitignore                    # Git忽略文件（忽略 data/、weights/ 等大文件和缓存）
├── app.py                        # 【Web主入口】FastAPI 服务实例、生命周期管理、启动配置
├── requirements.txt              # Python 环境依赖表 (fastapi, uvicorn, opencv-python, torch, faiss-cpu 等)
├── README.md                     # 项目说明文档
│
├── configs/                      # 【配置层】
│   └── config.yaml               # 全局配置文件 (包含模型路径、DB参数、阈值、四大类映射等)
│
├── data/                         # 【数据存储层】(本地运行时的各类数据文件，不提交到Git)
│   ├── temp_crops/               # 临时文件夹：存放商品注册时抽帧并分割出的小图，用于后续去重
│   ├── db_storage/               # 关系型数据库目录：如 SQLite 的 product_info.db
│   ├── vector_indexes/           # 向量数据库目录：如 FAISS 的 .index 索引文件存放处
│   └── test_videos/              # 测试数据源：存放前端上传的测试视频或监控录像片段
│
├── weights/                      # 【模型权重层】(存放所有训练好的 AI 模型权重)
│   ├── segmenter/   
│       ├── segmentation.pt        # YOLO等分割模型权重 (用于识别四大类并获取掩码/边界框)
│   └── classifiers/              # 细分类特征提取模型权重
│       ├── bag_weights.pt        # 袋装商品特征模型
│       ├── bottle_weights.pt     # 瓶装商品特征模型
│       ├── can_weights.pt        # 罐装商品特征模型
│       └── box_weights.pt        # 盒装商品特征模型
│
└── src/                          # 【核心源码】(严格遵循单向依赖与单一职责原则)
    ├── __init__.py
    │
    ├── api/                      # 1. 【Web API 层】(负责 HTTP 请求/响应，绝对不写业务逻辑)
    │   ├── __init__.py
    │   ├── dependencies.py       # 依赖注入机制：管理常驻内存的单例对象 (模型管家、数据库连接)
    │   └── routers/              # 路由分发
    │       ├── __init__.py
    │       ├── recognition.py    # 路由：生成实时识别的视频流 (MJPEG StreamingResponse)
    │       └── registration.py   # 路由：接收前端上传视频/图片及表单参数，触发后台注册任务
    │
    ├── entities/                 # 2. 【实体层】(数据传输对象 DTO，各层间传递的标准数据结构)
    │   ├── __init__.py
    │   └── schemas.py            # 基于 Pydantic 的类 (DetectResult, RecognizeResult, ProductInfo)
    │
    ├── pipelines/                # 3. 【业务流水线层】(负责业务流程调度，串联模型和数据库)
    │   ├── __init__.py
    │   ├── recognition.py        # 实时识别流：读帧 -> 调分割 -> 抽小图 -> 调分类取特征 -> 查向量库 -> 画框
    │   └── registration.py       # 商品注册流：读视频 -> 抽小图存 temp -> 去重 -> 取特征 -> 聚类 -> 存向量库与DB
    │
    ├── models/                   # 4. 【算法模型层】(只负责加载权重、预处理、前向推理、后处理)
    │   ├── __init__.py
    │   ├── base.py               # 抽象基类 BaseModel (定义 load_model, predict 接口)
    │   ├── segmentation.py       # 封装大类分割模型 (继承 BaseModel，处理图像并返回四大类及 crop 图像)
    │   ├── classification.py     # 封装特征提取模型 (继承 BaseModel，输入 crop 图像，返回 N 维特征向量)
    │   └── manager.py            # ModelManager 模型管家 (工厂模式，根据传入的大类自动路由到对应的小类模型)
    │
    ├── db/                       # 5. 【数据访问层】(封装所有数据持久化操作，屏蔽底层实现细节)
    │   ├── __init__.py
    │   ├── vector_db.py          # 封装向量数据库操作 (FAISS/Milvus，提供 insert_vectors, search_vector 接口)
    │   └── product_db.py         # 封装商品业务数据库操作 (SQLite/MySQL，提供 insert_product, get_product_by_id 接口)
    │
    └── utils/                    # 6. 【辅助工具层】(无状态的纯函数集合，可以随时复用)
        ├── __init__.py
        ├── image_utils.py        # 图像处理工具：在视频帧上画框绘制文本(draw_results)，图像去重算法(基于哈希/SSIM等)
        ├── feature_utils.py      # 特征处理工具：特征向量降维、聚类算法(clustering)提取代表性特征
        └── config_utils.py       # 配置工具：读取和解析 configs/config.yaml 文件