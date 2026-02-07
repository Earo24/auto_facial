"""
人脸样本数据模型
存储单个人脸检测样本的所有信息
"""
from dataclasses import dataclass, field
from typing import Optional, List, Any
import numpy as np
from pathlib import Path


@dataclass
class FaceSample:
    """人脸样本数据模型"""

    # 基本信息
    sample_id: str
    frame_id: str
    video_id: str

    # 人脸位置信息
    bbox: List[float]  # [x1, y1, x2, y2] 归一化坐标或像素坐标
    landmarks: Optional[List[List[float]]] = None  # 5点或68点关键点

    # 特征信息
    embedding: Optional[np.ndarray] = None  # 人脸特征向量
    quality_score: float = 0.0  # 质量分数 0-1

    # 元数据
    timestamp: float = 0.0  # 在视频中的时间戳（秒）
    frame_number: int = 0  # 帧序号
    image_path: Optional[str] = None  # 人脸图像保存路径
    face_size: Optional[tuple] = None  # 人脸尺寸 (width, height)

    # 聚类信息
    cluster_id: Optional[int] = None  # 聚类分配的簇ID
    character_id: Optional[str] = None  # 人工标注的角色ID

    # 识别信息
    recognition_confidence: Optional[float] = None  # 识别置信度

    def __post_init__(self):
        """初始化后处理"""
        if self.embedding is not None:
            self.embedding = np.array(self.embedding)

    @property
    def face_width(self) -> int:
        """获取人脸宽度"""
        if self.face_size:
            return self.face_size[0]
        if len(self.bbox) >= 4:
            return int(self.bbox[2] - self.bbox[0])
        return 0

    @property
    def face_height(self) -> int:
        """获取人脸高度"""
        if self.face_size:
            return self.face_size[1]
        if len(self.bbox) >= 4:
            return int(self.bbox[3] - self.bbox[1])
        return 0

    @property
    def is_high_quality(self) -> bool:
        """判断是否为高质量样本"""
        return self.quality_score >= 0.6

    @property
    def has_embedding(self) -> bool:
        """判断是否已提取特征"""
        return self.embedding is not None and len(self.embedding) > 0

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'sample_id': self.sample_id,
            'frame_id': self.frame_id,
            'video_id': self.video_id,
            'bbox': self.bbox,
            'landmarks': self.landmarks,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'quality_score': self.quality_score,
            'timestamp': self.timestamp,
            'frame_number': self.frame_number,
            'image_path': self.image_path,
            'face_size': self.face_size,
            'cluster_id': self.cluster_id,
            'character_id': self.character_id,
            'recognition_confidence': self.recognition_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FaceSample':
        """从字典创建实例"""
        return cls(
            sample_id=data['sample_id'],
            frame_id=data['frame_id'],
            video_id=data['video_id'],
            bbox=data['bbox'],
            landmarks=data.get('landmarks'),
            embedding=np.array(data['embedding']) if data.get('embedding') else None,
            quality_score=data.get('quality_score', 0.0),
            timestamp=data.get('timestamp', 0.0),
            frame_number=data.get('frame_number', 0),
            image_path=data.get('image_path'),
            face_size=data.get('face_size'),
            cluster_id=data.get('cluster_id'),
            character_id=data.get('character_id'),
            recognition_confidence=data.get('recognition_confidence'),
        )

    def compute_similarity(self, other: 'FaceSample') -> float:
        """计算与另一个人脸样本的相似度（余弦相似度）"""
        if not self.has_embedding or not other.has_embedding:
            return 0.0

        # 余弦相似度
        dot_product = np.dot(self.embedding, other.embedding)
        norm_a = np.linalg.norm(self.embedding)
        norm_b = np.linalg.norm(other.embedding)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def is_duplicate(self, other: 'FaceSample', threshold: float = 0.95) -> bool:
        """判断是否为重复样本（时间接近且相似度高）"""
        # 时间接近（5秒内）
        time_diff = abs(self.timestamp - other.timestamp)
        if time_diff > 5.0:
            return False

        # 相似度高
        similarity = self.compute_similarity(other)
        return similarity >= threshold


@dataclass
class FaceCluster:
    """人脸簇数据模型"""

    cluster_id: int
    samples: List[FaceSample] = field(default_factory=list)

    @property
    def size(self) -> int:
        """簇中样本数量"""
        return len(self.samples)

    @property
    def avg_quality(self) -> float:
        """平均质量分数"""
        if not self.samples:
            return 0.0
        return sum(s.quality_score for s in self.samples) / len(self.samples)

    @property
    def representative_embedding(self) -> Optional[np.ndarray]:
        """获取代表性特征向量（平均）"""
        if not self.samples:
            return None

        embeddings = [s.embedding for s in self.samples if s.has_embedding]
        if not embeddings:
            return None

        return np.mean(embeddings, axis=0)

    def add_sample(self, sample: FaceSample):
        """添加样本"""
        self.samples.append(sample)
        sample.cluster_id = self.cluster_id

    def remove_sample(self, sample: FaceSample):
        """移除样本"""
        if sample in self.samples:
            self.samples.remove(sample)
            sample.cluster_id = None

    def get_high_quality_samples(self, min_quality: float = 0.7, limit: int = 20) -> List[FaceSample]:
        """获取高质量样本"""
        high_quality = [s for s in self.samples if s.quality_score >= min_quality]
        # 按质量排序
        high_quality.sort(key=lambda s: s.quality_score, reverse=True)
        return high_quality[:limit]

    def merge(self, other: 'FaceCluster'):
        """合并另一个簇"""
        for sample in other.samples:
            sample.cluster_id = self.cluster_id
            self.samples.append(sample)
        other.samples.clear()

    def split(self, sample_indices: List[int]) -> 'FaceCluster':
        """拆分簇"""
        new_cluster = FaceCluster(cluster_id=-1)  # 临时ID
        samples_to_move = [self.samples[i] for i in sorted(sample_indices, reverse=True)]

        for sample in samples_to_move:
            self.samples.remove(sample)
            sample.cluster_id = None
            new_cluster.add_sample(sample)

        return new_cluster
