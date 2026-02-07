# AutoFacial

> 影视人脸识别自动化系统 — 自动从视频中提取人脸、聚类发现角色、匹配演员信息

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18%2B-blue)](https://react.dev/)

## 为什么需要 AutoFacial？

在影视后期制作、内容分析和版权管理中，人物识别是一个常见但耗时的工作。传统方式需要人工逐帧标注，效率低下且容易出错。

**AutoFacial** 通过深度学习和智能聚类算法，将这一过程自动化：

- **从零发现角色**：无需预先标注，系统自动从视频中识别出所有角色
- **演员关联**：将识别出的角色与真实演员信息匹配
- **批量处理**：支持多剧集、多视频的统一管理
- **可视化操作**：现代化 Web 界面，操作直观简单

### 应用场景

| 场景 | 价值 |
|------|------|
| **影视后期制作** | 自动生成角色集，减少人工标注工作量 90% 以上 |
| **内容检索** | 基于人物快速定位视频片段，支持「某角色在哪些场景出现」的查询 |
| **数据分析** | 统计角色出场时长、频次，为剪辑和宣发提供数据支持 |
| **版权监控** | 监控影视内容中特定人物的使用情况 |

### 技术优势

- **高准确率**：三阶段聚类策略，当前参数下达到 **80.5%** 聚类率
- **智能采样**：结合场景变化检测的帧采样，减少计算量同时保证准确性
- **可扩展**：模块化设计，易于集成到现有工作流

---

## 快速开始

### 环境要求

- **Python**: 3.8 或更高版本
- **Node.js**: 16.0 或更高版本
- **内存**: 建议 8GB 以上（处理大型视频文件时）
- **操作系统**: macOS / Linux / Windows

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/your-username/auto_facial.git
cd auto_facial
```

#### 2. 创建 Python 虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

#### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

> **注意**：首次运行时，InsightFace 会自动下载所需模型（约 200MB）到 `~/.insightface/models/` 目录。请确保网络连接正常。

#### 4. 安装前端依赖

```bash
cd web_app
npm install
cd ..
```

---

## 运行项目

### 方式一：Web 界面（推荐）

#### 启动后端服务

```bash
python api_server.py
```

后端服务将运行在：**http://localhost:8000**

#### 启动前端界面

新开一个终端窗口：

```bash
cd web_app
npm run dev
```

前端界面将运行在：**http://localhost:3000**

访问 **http://localhost:3000** 即可开始使用。

### 方式二：命令行工具

如果只需要处理单个视频，可以使用命令行工具：

```bash
# 处理视频（提取帧、检测人脸）
python scripts/process_video.py /path/to/your/video.mp4

# 聚类人脸（自动发现角色）
python scripts/cluster_faces.py <video_id>

# 导入演员照片
python scripts/import_actors.py

# 匹配聚类结果与演员
python scripts/match_clusters_to_actors.py
```

---

## 使用流程

AutoFacial 的工作流程如下：

```mermaid
graph LR
    A[上传视频] --> B[提取帧 & 人脸检测]
    B --> C[特征提取]
    C --> D[智能聚类]
    D --> E[导入演员]
    E --> F[演员匹配]
    F --> G[结果查看 & 导出]
```

### 详细步骤

1. **创建剧集**
   - 在「剧集管理」页面创建新的电视剧/电影条目

2. **上传视频**
   - 将视频文件上传并关联到对应剧集
   - 系统自动进行帧提取和人脸检测

3. **自动聚类**
   - 系统使用 DBSCAN 算法自动发现角色
   - 当前参数（eps=0.80）达到 80.5% 聚类率

4. **导入演员**
   - 上传该剧集演员的照片
   - 系统自动提取演员特征

5. **演员匹配**
   - 将聚类结果与演员库进行匹配
   - 可手动调整匹配结果

6. **查看结果**
   - 在「聚类标注」页面查看和导出结果

---

## 配置说明

配置文件位于 `config/settings.py`，可根据需求调整：

```python
# 数据存储路径
DATA_ROOT = "./data"

# 视频处理参数
VIDEO_CONFIG = {
    "sample_interval": 1,  # 采样间隔（秒）
    "max_frames": 10000,   # 最大帧数
}

# 聚类参数
CLUSTERING_CONFIG = {
    "eps": 0.80,           # 邻域半径（当前最优值）
    "min_samples": 3,      # 最小样本数
}
```

---

## 技术栈

### 后端

| 组件 | 技术 | 说明 |
|------|------|------|
| 人脸检测 | InsightFace (RetinaFace) | 高精度人脸检测 |
| 特征提取 | InsightFace (ArcFace) | 512 维特征向量 |
| 聚类算法 | DBSCAN + Agglomerative | 三阶段聚类策略 |
| API 框架 | FastAPI | 高性能异步 API |
| 数据库 | SQLite + SQLAlchemy | 轻量级数据持久化 |

### 前端

| 组件 | 技术 |
|------|------|
| 框架 | React 18 + TypeScript |
| 构建 | Vite |
| UI 库 | Tailwind CSS + Radix UI |
| 图表 | Recharts |

---

## API 文档

启动后端服务后，访问以下地址查看完整 API 文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 项目结构

```
auto_facial/
├── config/                 # 配置文件
│   ├── model_config.yaml   # 模型配置
│   └── settings.py         # 系统设置
├── src/
│   ├── core/              # 核心业务逻辑
│   │   ├── video_processor.py    # 视频处理
│   │   ├── face_detector.py      # 人脸检测
│   │   ├── face_embedder.py      # 特征提取
│   │   ├── cluster_engine.py     # 聚类引擎
│   │   └── recognition_engine.py # 识别引擎
│   ├── models/            # 数据模型
│   ├── storage/           # 数据存储层
│   └── utils/             # 工具函数
├── scripts/               # 命令行脚本
├── web_app/               # Web 前端（React + Vite）
├── api_server.py          # FastAPI 服务器
├── requirements.txt       # Python 依赖
└── README.md
```

---

## 常见问题

### Q: 首次运行很慢？
A: 首次运行需要下载 InsightFace 模型（约 200MB），请耐心等待。后续运行会使用缓存。

### Q: 如何提高聚类准确率？
A: 可以在 `config/settings.py` 中调整 `CLUSTERING_CONFIG` 的 `eps` 参数。当前最优值为 0.80。

### Q: 支持哪些视频格式？
A: 支持 OpenCV 能够读取的所有格式，包括 MP4、AVI、MOV 等。

### Q: 内存不足怎么办？
A: 可以在配置中减少 `VIDEO_CONFIG["max_frames"]` 或处理较短的视频片段。

---

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 贡献

欢迎提交 Issue 和 Pull Request！

---

## 更新日志

### v1.0.0 (最新)
- 添加演员管理功能和 UI 优化
- 添加剧集管理和演员匹配功能
- 优化聚类参数：eps=0.80 达到 80.5% 聚类率
- 实现智能帧采样策略
