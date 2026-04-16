# 超市产品识别系统

## 项目简介

这是一个基于计算机视觉技术的智能超市产品识别与结算系统。通过深度学习模型实现对超市商品的实时自动识别，支持商品注册、实时监控和结算功能。系统采用前后端分离架构，后端基于 Python FastAPI 提供高性能 API 服务，前端使用 Vue.js 构建现代化用户界面。

## 核心功能

- **商品注册**：上传商品视频，通过 AI 模型自动提取特征并注册到数据库
- **实时识别**：支持摄像头、视频文件和 IP 摄像头输入，实现毫秒级商品识别
- **监控界面**：实时显示识别结果，支持暂停/恢复操作
- **WebSocket 通信**：实时推送识别结果到前端界面

## 技术栈

### 后端
- Python 3.8+, FastAPI, Uvicorn
- OpenCV, PyTorch, FAISS, PyMySQL, NumPy

### 前端
- Vue 3, Vite, Tailwind CSS
- Axios, Pinia, Vue Router

### AI 模型
- YOLOv11 (实例分割)
- ConvNeXt Tiny (特征提取)
- FAISS (向量检索)

### 数据库
- MySQL (产品信息)
- FAISS (特征向量)

## 快速开始

### 环境要求
- Python 3.8+
- Node.js 16+
- MySQL 8.0+

### 安装依赖

```bash
# 后端依赖
pip install -r requirements.txt

# 前端依赖
cd frontend
npm install
```

### 配置数据库

1. 创建 MySQL 数据库：
```sql
CREATE DATABASE supermarket_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

2. 修改 `configs/config.yaml` 中的数据库连接信息

### 运行项目

```bash
# 启动后端服务
python app.py

# 启动前端服务 (新终端)
cd frontend
npm run dev
```

访问 http://localhost:5173 即可使用系统。

## 项目结构

```
supermarket_web_system/
├── app.py                        # FastAPI 主应用
├── main.py                       # 本地调试入口
├── requirements.txt              # 后端依赖
├── configs/
│   └── config.yaml               # 系统配置
├── data/
│   ├── vector_indexes/           # FAISS 向量索引
│   ├── static_videos/            # 生成的视频文件
│   └── temp_*/                   # 临时文件目录
├── weights/                      # AI 模型权重
│   ├── segmenter/                # 分割模型
│   └── classifiers/              # 分类模型
├── frontend/                     # Vue.js 前端项目
│   ├── src/
│   │   ├── views/                # 页面组件
│   │   ├── components/           # UI 组件
│   │   └── store/                # 状态管理
│   └── package.json
└── src/                          # 后端核心代码
    ├── api/                      # API 层
    ├── models/                   # AI 模型管理
    ├── db/                       # 数据库管理
    ├── pipelines/                # 业务流程
    └── utils/                    # 工具函数
```

## API 接口

### 商品注册
- `POST /api/registration/register` - 上传视频注册商品

### 实时识别
- `GET /api/recognition/stream` - 获取 MJPEG 视频流
- `WebSocket /ws/recognition` - 实时识别结果推送

### 视频管理
- `GET /api/videos/list` - 获取视频文件列表

## 详细文档

请查看 [项目报告](./project_report.md) 获取完整的系统架构、业务流程和技术实现详情。

## 许可证

MIT License
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