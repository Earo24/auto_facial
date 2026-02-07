"""
全局配置文件
影视人脸识别自动化系统
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"

# 目录配置
RAW_VIDEO_DIR = DATA_ROOT / "raw"
FRAMES_DIR = DATA_ROOT / "processed" / "frames"
FACES_DIR = DATA_ROOT / "processed" / "faces"
EMBEDDINGS_DIR = DATA_ROOT / "processed" / "embeddings"
CHARACTERS_DIR = DATA_ROOT / "characters"
OUTPUT_DIR = DATA_ROOT / "output"

# 确保目录存在
for dir_path in [RAW_VIDEO_DIR, FRAMES_DIR, FACES_DIR, EMBEDDINGS_DIR, CHARACTERS_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 视频处理配置
VIDEO_CONFIG = {
    'sample_fps': 0.2,              # 每秒采样帧数（0.2 = 5秒一帧）
    'scene_threshold': 30.0,        # 场景变化阈值
    'max_frames': 10000,            # 最大提取帧数
}

# 人脸检测配置（优化后，提高召回率）
FACE_DETECTION_CONFIG = {
    'min_face_size_ratio': 0.15,    # 最小人脸尺寸占画面高度的比例（15%过滤背景小人脸）
    'min_face_size_absolute': 80,    # 绝对最小值（低分辨率视频备用）
    'confidence_threshold': 0.7,     # 检测置信度（降低，包含侧脸）
    'quality_threshold': 0.4,        # 质量分数阈值（降低，包含模糊人脸）
    'max_faces_per_frame': 10,       # 每帧最多检测人脸数
    'dedup_threshold': 0.95,         # 去重相似度阈值
}

# 聚类配置
CLUSTERING_CONFIG = {
    'eps': 0.5,                     # DBSCAN距离阈值（增大以发现更多簇，对应相似度约87%）
    'min_samples': 2,               # 最小簇大小
    'merge_threshold': 0.75,        # 小簇合并阈值（降低以合并更多相似簇）
    'max_clusters': 50,             # 最大簇数量
}

# 识别配置
RECOGNITION_CONFIG = {
    'similarity_threshold': 0.5,    # 高置信度阈值
    'low_confidence_threshold': 0.4, # 低置信度阈值
    'temporal_window': 3.0,         # 时序平滑窗口（秒）
}

# UI配置
UI_CONFIG = {
    'max_preview_samples': 20,      # 每簇最大预览样本数
    'thumbnail_size': (100, 100),   # 缩略图尺寸
}

# 数据库配置
DATABASE_PATH = DATA_ROOT / "auto_facial.db"

# 模型配置
MODEL_CONFIG = {
    'detector_name': 'retinaface_r50_v1',
    'embedder_name': 'arcface_r100_v1',
    'use_gpu': True,
}
