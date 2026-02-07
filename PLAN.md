# 影视人脸识别自动化系统 - 实施计划

## 项目概述
构建一个半自动的影视人脸识别系统，用于电影/电视剧的角色识别和分析。系统自动从视频中提取人脸并进行聚类，人工只需审核和调整角色标注。

## 技术栈选择

| 组件 | 选择 | 理由 |
|------|------|------|
| 人脸识别 | InsightFace | 高精度、鲁棒性强、支持遮挡处理 |
| 人脸检测 | MediaPipe Face Detection | 速度快、准确率高、轻量级 |
| 聚类算法 | scikit-learn (DBSCAN + Agglomerative) | 自动发现角色数量，便于合并/拆分 |
| Web界面 | Streamlit | 纯Python、快速开发、适合数据应用 |
| 视频处理 | OpenCV + FFmpeg | 成熟稳定、功能完善 |

## 项目目录结构

```
auto_facial/
├── config/
│   ├── __init__.py
│   ├── settings.py                  # 全局配置
│   └── model_config.yaml            # 模型配置
│
├── src/
│   ├── core/                        # 核心业务逻辑
│   │   ├── video_processor.py       # 视频处理引擎
│   │   ├── face_detector.py         # 人脸检测
│   │   ├── face_embedder.py         # 人脸特征提取
│   │   ├── cluster_engine.py        # 聚类引擎
│   │   └── recognition_engine.py    # 识别引擎
│   │
│   ├── models/                      # 数据模型
│   │   ├── character.py             # 角色数据模型
│   │   ├── face_sample.py           # 人脸样本模型
│   │   └── video_frame.py           # 视频帧模型
│   │
│   ├── storage/                     # 数据存储层
│   │   ├── database.py              # SQLite数据库操作
│   │   ├── character_store.py       # 角色存储
│   │   └── annotation_store.py      # 标注存储
│   │
│   ├── analysis/                    # 分析模块
│   │   ├── screen_time.py           # 出镜时长统计
│   │   ├── co_occurrence.py         # 同框分析
│   │   └── relationship_graph.py    # 关系图谱生成
│   │
│   ├── utils/                       # 工具函数
│   │   ├── image_utils.py           # 图像处理工具
│   │   ├── video_utils.py           # 视频处理工具
│   │   └── visualization.py         # 可视化工具
│   │
│   └── ui/                          # 用户界面
│       ├── clustering_ui.py         # 聚类标注界面
│       ├── recognition_ui.py        # 识别结果界面
│       └── analysis_ui.py           # 分析报告界面
│
├── data/                            # 数据目录
│   ├── raw/                         # 原始视频
│   ├── processed/
│   │   ├── frames/                  # 提取的帧
│   │   ├── faces/                   # 提取的人脸
│   │   └── embeddings/              # 特征向量
│   ├── characters/                  # 角色库
│   └── output/                      # 输出结果
│
├── models/                          # 预训练模型
├── tests/                           # 测试代码
├── scripts/                         # 脚本工具
├── app.py                           # Streamlit主应用
├── requirements.txt                 # 依赖列表
├── setup.py                         # 安装配置
└── README.md
```

## 核心模块设计

### 1. 视频处理引擎 (video_processor.py)
```python
class VideoProcessor:
    """智能视频采样和帧提取"""
    - extract_frames(): 智能采样（场景变化检测 + 人脸密度检测）
    - process_video(): 完整视频处理流程
    - 支持多线程处理
```

### 2. 人脸检测模块 (face_detector.py)
```python
class FaceDetector:
    """人脸检测和质量评估"""
    - detect_faces(): 多尺度人脸检测
    - assess_quality(): 评估人脸质量（清晰度、正面度、光照）
    - filter_high_quality(): 过滤低质量样本
    - deduplicate_faces(): 去重相似帧
```

### 3. 聚类引擎 (cluster_engine.py)
```python
class ClusterEngine:
    """三阶段聚类策略"""
    - phase1_clustering(): DBSCAN粗聚类
    - phase2_refinement(): 层次聚类细化
    - phase3_merge(): 合并小簇
    - discover_characters(): 自动识别主要角色
```

### 4. 识别引擎 (recognition_engine.py)
```python
class RecognitionEngine:
    """多策略人脸识别"""
    - recognize(): 基于特征向量识别
    - apply_temporal_smoothing(): 时序平滑避免误识别
    - handle_appearance_changes(): 处理妆容/造型变化
```

### 5. 标注界面 (clustering_ui.py)
```python
# Streamlit界面功能
- 展示聚类结果概览
- 样本网格展示（每簇最多20个样本）
- 角色命名输入
- 簇操作：合并、拆分、删除
- 添加新角色
- 样本移除功能
```

## 关键配置参数

```python
# 视频处理
VIDEO_CONFIG = {
    'sample_fps': 1.0,              # 每秒采样帧数
    'scene_threshold': 30.0,        # 场景变化阈值
}

# 人脸检测
FACE_DETECTION_CONFIG = {
    'min_face_size': 40,            # 最小人脸尺寸（像素）
    'confidence_threshold': 0.9,    # 检测置信度
    'quality_threshold': 0.6,       # 质量分数阈值
}

# 聚类配置
CLUSTERING_CONFIG = {
    'eps': 0.5,                     # 相似度阈值
    'min_samples': 5,               # 最小簇大小
    'merge_threshold': 0.85,        # 小簇合并阈值
}

# 识别配置
RECOGNITION_CONFIG = {
    'similarity_threshold': 0.5,    # 高置信度阈值
    'low_confidence_threshold': 0.4, # 低置信度阈值
    'temporal_window': 3.0,         # 时序平滑窗口（秒）
}
```

## 数据格式

### 角色库格式 (JSON)
```json
{
  "version": "1.0",
  "video_info": {...},
  "characters": [
    {
      "id": "char_001",
      "name": "角色名",
      "prototypes": [{"embedding": [...], "image_path": "..."}],
      "samples": [{"frame_path": "...", "bbox": [...], "timestamp": 123.45}],
      "statistics": {"total_samples": 150, "avg_quality": 0.89}
    }
  ]
}
```

### SQLite数据库Schema
```sql
CREATE TABLE characters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    aliases TEXT,
    created_at TIMESTAMP
);

CREATE TABLE face_samples (
    id INTEGER PRIMARY KEY,
    character_id TEXT,
    embedding BLOB,
    image_path TEXT,
    timestamp REAL,
    quality_score REAL,
    FOREIGN KEY (character_id) REFERENCES characters(id)
);

CREATE TABLE recognition_results (
    id INTEGER PRIMARY KEY,
    video_id TEXT,
    frame_id TEXT,
    timestamp REAL,
    character_id TEXT,
    bbox TEXT,
    confidence REAL
);
```

## 完整工作流程

```
1. 视频处理
   ↓
   智能采样 → 人脸检测 → 特征提取 → 质量过滤

2. 自动聚类
   ↓
   粗聚类(DBSCAN) → 细聚类(Agglomerative) → 合并小簇 → 自动发现主要角色

3. 人工标注
   ↓
   查看聚类结果 → 命名角色 → 合并/拆分簇 → 添加遗漏角色 → 保存角色库

4. 批量识别
   ↓
   逐帧识别 → 时序平滑 → 处理造型变化 → 生成标注结果

5. 分析报告
   ↓
   出镜时长统计 → 同框分析 → 关系图谱生成
```

## 分阶段实施计划

### 第一阶段 (MVP) - 核心功能
1. 项目初始化（目录结构、依赖安装）
2. 视频处理引擎（帧提取、人脸检测）
3. 人脸特征提取（InsightFace集成）
4. 基础聚类算法（DBSCAN）
5. 简单的Streamlit界面
6. 手动标注功能

### 第二阶段 - 识别增强
1. 优化聚类算法（多阶段策略）
2. 实现识别引擎
3. 时序平滑功能
4. 处理妆容/造型变化
5. 完善标注界面

### 第三阶段 - 分析功能
1. 出镜时长统计
2. 同框分析
3. 关系图谱生成
4. 可视化报告

### 第四阶段 - 优化
1. 性能优化（并行处理、缓存）
2. 超分辨率增强
3. 批处理多个视频
4. 导出功能

## 关键文件清单

| 文件路径 | 说明 |
|---------|------|
| `src/core/face_detector.py` | 人脸检测核心模块，系统基础 |
| `src/core/cluster_engine.py` | 聚类引擎，自动角色发现 |
| `src/core/recognition_engine.py` | 识别引擎，核心输出模块 |
| `src/ui/clustering_ui.py` | 聚类标注界面，用户交互入口 |
| `src/models/character.py` | 角色数据模型，核心数据结构 |

## 主要依赖

```
# 核心依赖
numpy>=1.23.0
opencv-python>=4.8.0
insightface>=0.7.3
onnxruntime>=1.15.0
scikit-learn>=1.3.0

# Web界面
streamlit>=1.28.0

# 可视化
matplotlib>=3.7.0
networkx>=3.1.0

# 工具
tqdm>=4.66.0
pillow>=10.0.0
```

## 验证测试计划

1. **单元测试**：各模块功能测试
2. **集成测试**：完整流程测试
3. **性能测试**：处理时长视频的性能
4. **准确性测试**：识别准确率评估
5. **用户测试**：界面交互体验测试
