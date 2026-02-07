# AutoFacial

影视人脸识别自动化系统 - 自动从视频中提取人脸、聚类、识别角色。

## 功能特点

- **智能视频采样**: 结合场景变化检测的智能帧采样
- **自动人脸聚类**: 三阶段聚类策略自动发现角色
- **Web可视化界面**: React + FastAPI 现代化Web界面
- **演员管理**: 支持导入演员照片进行角色匹配
- **批量识别**: 基于角色库的批量人脸识别
- **剧集管理**: 支持多剧集视频管理

## 技术栈

| 组件 | 选择 |
|------|------|
| 人脸检测 | InsightFace (RetinaFace) |
| 特征提取 | InsightFace (ArcFace) |
| 聚类算法 | scikit-learn (DBSCAN + Agglomerative) |
| 后端 | FastAPI + Python |
| 前端 | React + TypeScript + Vite |
| 数据存储 | SQLite |

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
│   │   ├── database.py    # 数据库操作
│   │   └── character_store.py # 角色存储
│   └── utils/             # 工具函数
├── scripts/               # 命令行脚本
│   ├── process_video.py   # 处理视频
│   ├── cluster_faces.py   # 聚类人脸
│   ├── recognize.py       # 识别
│   ├── import_actors.py   # 导入演员
│   └── match_clusters_to_actors.py # 匹配聚类
├── web_app/               # Web前端
│   ├── src/               # React源码
│   │   ├── components/    # UI组件
│   │   ├── pages/         # 页面
│   │   └── services/      # API服务
│   └── public/            # 静态资源
├── api_server.py          # FastAPI服务器
├── requirements.txt       # Python依赖
└── README.md
```

## 安装

### 1. 克隆项目

```bash
git clone <repository_url>
cd auto_facial
```

### 2. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 3. 安装前端依赖

```bash
cd web_app
npm install
cd ..
```

### 4. 下载模型

首次运行时，InsightFace会自动下载所需模型到 `~/.insightface/models/` 目录。

## 使用方法

### 启动服务

#### 启动后端

```bash
python api_server.py
```

后端运行在 http://localhost:8000

#### 启动前端

```bash
cd web_app
npm run dev
```

前端运行在 http://localhost:3000

### 使用流程

1. **剧集管理**: 创建电视剧/电影条目
2. **上传视频**: 上传视频文件并关联到对应剧集
3. **导入演员**: 导入该剧集的演员照片
4. **自动聚类**: 系统自动对人脸进行聚类
5. **演员匹配**: 将聚类结果与演员进行匹配
6. **查看结果**: 在聚类标注页面查看和调整

### 命令行工具

#### 处理视频

```bash
python scripts/process_video.py /path/to/video.mp4
```

#### 聚类人脸

```bash
python scripts/cluster_faces.py <video_id>
```

#### 导入演员照片

```bash
python scripts/import_actors.py
```

#### 匹配聚类与演员

```bash
python scripts/match_clusters_to_actors.py
```

## 配置

编辑 `config/settings.py` 修改配置参数：

- `DATA_ROOT`: 数据存储路径
- `MODELS_ROOT`: 模型存储路径
- `VIDEO_CONFIG`: 视频处理参数
- `FACE_DETECTION_CONFIG`: 人脸检测参数
- `CLUSTERING_CONFIG`: 聚类参数
- `RECOGNITION_CONFIG`: 识别参数

## API文档

启动后端服务后访问 http://localhost:8000/docs 查看完整API文档。

## 许可证

MIT License
