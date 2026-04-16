supermarket_web_system/
├── .gitignore                    # 忽略 data/、weights/、node_modules/、dist/ 等
├── app.py                        # 【Web主入口】FastAPI 实例，挂载静态目录、生命周期、WebSocket 路由
├── main.py                       # 【本地调试入口】原有的本地脚本执行逻辑，用于快速测试 Pipeline
├── requirements.txt              # 后端依赖 (fastapi, uvicorn, pymysql, opencv-python, torch, faiss-cpu等)
├── README.md                     # 项目架构与前后端运行说明
│
├── configs/                      # 【配置层】
│   └── config.yaml               # 全局配置 (含模型路径、MySQL连接、识别阈值、静态资源路径)
│
├── data/                         # 【数据存储层】
│   ├── temp_crops/               # 注册时暂存的商品切图 (去重前)
│   ├── db_storage/               # 若使用 SQLite 存放此处
│   ├── vector_indexes/           # FAISS 索引文件 (.index)
│   ├── test_videos/              # 前端上传用于注册的原始视频
│   └── static_videos/            # 【新增】侦测结果视频：Pipeline 生成后存放于此，供前端 Web 回放
│
├── weights/                      # 【模型权重层】
│   ├── segmenter/   
│   │   └── segmentation.pt       # YOLO 实例分割权重
│   └── classifiers/              
│       ├── bag_weights.pt        # 袋装特征模型
│       ├── bottle_weights.pt     # 瓶装特征模型
│       ├── can_weights.pt        # 罐装特征模型
│       └── box_weights.pt        # 盒装特征模型
│
├── frontend/                     # 【前端项目层】(Vite + Vue3 + Zustand + Tailwind)
│   ├── src/
│   │   ├── api/                  # Axios 封装：调用后端注册、视频列表等接口
│   │   ├── store/                # 【新增】Zustand 状态管理 (useStore.js)
│   │   ├── components/           # UI 组件 (识别结果列表、状态提示框)
│   │   ├── views/                # 页面 (Monitor.vue 监控, Register.vue 注册, Playback.vue 回放)
│   │   ├── App.vue               # 入口布局
│   │   └── main.js               # 初始化 Vue
│   ├── index.html
│   ├── package.json              # 前端依赖 (zustand-vue, tailwindcss, axios等)
│   ├── tailwind.config.js
│   └── vite.config.js            # 配置反向代理，解决开发环境跨域
│
└── src/                          # 【后端核心源码】
    ├── __init__.py
    ├── api/                      # 1. 【Web API 层】
    │   ├── __init__.py
    │   ├── dependencies.py       # 依赖注入：管理全局唯一的 ModelManager, VectorDB, ProductDB 实例
    │   ├── ws_manager.py         # 【新增】WebSocket 管理器：负责心跳及识别结果 JSON 的实时推送
    │   └── routers/              # 路由分发
    │       ├── __init__.py
    │       ├── recognition.py    # 路由：MJPEG 视频流 + WebSocket 接口
    │       ├── registration.py   # 路由：接收视频上传，触发 BackgroundTasks 异步注册流程
    │       └── videos.py         # 【新增】视频路由：获取 static_videos 目录下的文件列表
    │
    ├── entities/                 # 2. 【实体层】
    │   ├── __init__.py
    │   └── schemas.py            # Pydantic 类 (用于 API 校验) & Dataclasses (用于内部 Pipeline)
    │
    ├── pipelines/                # 3. 【业务流水线层】
    │   ├── __init__.py
    │   ├── recognition.py        # 实时识别流：支持单帧处理 & 本地/Webcam 循环调用
    │   └── registration.py       # 商品注册流：整合视频去重、置信度聚合、特征聚类、DB入库
    │
    ├── models/                   # 4. 【算法模型层】
    │   ├── __init__.py
    │   ├── base.py               # 抽象基类 (BaseModel)
    │   ├── segmentation.py       # 分割模型实现
    │   ├── classification.py     # 特征提取模型实现
    │   └── manager.py            # ModelManager：模型路由与显存管理
    │
    ├── db/                       # 5. 【数据访问层】
    │   ├── __init__.py
    │   ├── vector_db.py          # 向量库：FAISS 增删改查
    │   └── product_db.py         # 业务库：MySQL 商品元数据操作
    │
    └── utils/                    # 6. 【辅助工具层】
        ├── __init__.py
        ├── image_utils.py        # 图像处理：画框渲染、哈希去重
        ├── feature_utils.py      # 特征处理：聚类融合 (Clustering)
        └── config_utils.py       # 配置工具：YAML 解析与路径绝对化