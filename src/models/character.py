"""
角色数据模型
存储角色的所有信息，包括原型样本、统计数据等
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np
import json
from datetime import datetime


@dataclass
class Prototype:
    """角色原型样本（代表性人脸样本）"""

    embedding: np.ndarray  # 特征向量
    image_path: str  # 人脸图像路径
    quality_score: float  # 质量分数
    timestamp: float  # 在视频中的时间戳

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'embedding': self.embedding.tolist(),
            'image_path': self.image_path,
            'quality_score': self.quality_score,
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Prototype':
        """从字典创建实例"""
        return cls(
            embedding=np.array(data['embedding']),
            image_path=data['image_path'],
            quality_score=data['quality_score'],
            timestamp=data['timestamp'],
        )


@dataclass
class CharacterStatistics:
    """角色统计数据"""

    total_samples: int = 0  # 总样本数
    avg_quality: float = 0.0  # 平均质量分数
    first_appearance: float = 0.0  # 首次出现时间（秒）
    last_appearance: float = 0.0  # 最后出现时间（秒）
    total_screen_time: float = 0.0  # 总出镜时长（秒）

    # 出镜分布
    appearances_by_scene: Dict[int, int] = field(default_factory=dict)  # 每场景出现次数

    def update_with_sample(self, timestamp: float, quality: float, scene_id: Optional[int] = None):
        """更新统计数据"""
        self.total_samples += 1

        # 更新平均质量
        if self.avg_quality == 0:
            self.avg_quality = quality
        else:
            self.avg_quality = (self.avg_quality * (self.total_samples - 1) + quality) / self.total_samples

        # 更新首次/最后出现时间
        if self.first_appearance == 0 or timestamp < self.first_appearance:
            self.first_appearance = timestamp
        if timestamp > self.last_appearance:
            self.last_appearance = timestamp

        # 更新场景统计
        if scene_id is not None:
            self.appearances_by_scene[scene_id] = self.appearances_by_scene.get(scene_id, 0) + 1

    @property
    def appearance_duration(self) -> float:
        """出镜持续时间"""
        if self.last_appearance == 0:
            return 0.0
        return self.last_appearance - self.first_appearance


@dataclass
class Character:
    """角色数据模型"""

    # 基本信息
    character_id: str  # 角色ID（如：char_001）
    name: str  # 角色名称
    video_id: str  # 所属视频ID

    # 人脸原型（高质量代表性样本）
    prototypes: List[Prototype] = field(default_factory=list)

    # 所有样本
    samples: List[Dict[str, Any]] = field(default_factory=list)

    # 统计数据
    statistics: CharacterStatistics = field(default_factory=CharacterStatistics)

    # 元数据
    aliases: List[str] = field(default_factory=list)  # 别名
    description: str = ""  # 描述
    color: Optional[str] = None  # 可视化颜色
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # 识别配置
    recognition_threshold: float = 0.5  # 识别阈值
    appearance_variance: float = 0.3  # 造型变化容忍度

    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.prototypes, list) and self.prototypes:
            if isinstance(self.prototypes[0], dict):
                self.prototypes = [Prototype.from_dict(p) for p in self.prototypes]

    @property
    def prototype_embeddings(self) -> np.ndarray:
        """获取所有原型特征向量"""
        if not self.prototypes:
            return np.array([])
        return np.array([p.embedding for p in self.prototypes])

    @property
    def avg_prototype_embedding(self) -> Optional[np.ndarray]:
        """获取平均原型特征向量"""
        if not self.prototypes:
            return None
        embeddings = self.prototype_embeddings
        return np.mean(embeddings, axis=0)

    @property
    def sample_count(self) -> int:
        """获取样本数量"""
        return len(self.samples)

    @property
    def is_valid(self) -> bool:
        """判断角色是否有效（至少有原型样本）"""
        return len(self.prototypes) > 0

    def add_prototype(self, embedding: np.ndarray, image_path: str, quality_score: float, timestamp: float):
        """添加原型样本"""
        prototype = Prototype(
            embedding=embedding,
            image_path=image_path,
            quality_score=quality_score,
            timestamp=timestamp,
        )
        self.prototypes.append(prototype)
        self.updated_at = datetime.now().isoformat()

    def add_sample(self, frame_path: str, bbox: List[float], timestamp: float, quality: float,
                   embedding: Optional[np.ndarray] = None, scene_id: Optional[int] = None):
        """添加样本"""
        sample = {
            'frame_path': frame_path,
            'bbox': bbox,
            'timestamp': timestamp,
            'quality': quality,
        }
        if embedding is not None:
            sample['embedding'] = embedding.tolist()
        if scene_id is not None:
            sample['scene_id'] = scene_id

        self.samples.append(sample)
        self.statistics.update_with_sample(timestamp, quality, scene_id)
        self.updated_at = datetime.now().isoformat()

    def compute_similarity(self, embedding: np.ndarray) -> float:
        """计算特征向量与角色的相似度"""
        if not self.prototypes:
            return 0.0

        # 计算与所有原型的相似度，取最大值
        max_similarity = 0.0
        for prototype in self.prototypes:
            similarity = self._cosine_similarity(embedding, prototype.embedding)
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(emb1, emb2)
        norm_a = np.linalg.norm(emb1)
        norm_b = np.linalg.norm(emb2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def merge(self, other: 'Character'):
        """合并另一个角色"""
        # 合并原型
        for prototype in other.prototypes:
            self.prototypes.append(prototype)

        # 合并样本
        self.samples.extend(other.samples)

        # 更新统计
        self.statistics.total_samples += other.statistics.total_samples

        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'character_id': self.character_id,
            'name': self.name,
            'video_id': self.video_id,
            'prototypes': [p.to_dict() for p in self.prototypes],
            'samples': self.samples,
            'statistics': {
                'total_samples': self.statistics.total_samples,
                'avg_quality': self.statistics.avg_quality,
                'first_appearance': self.statistics.first_appearance,
                'last_appearance': self.statistics.last_appearance,
                'total_screen_time': self.statistics.total_screen_time,
                'appearances_by_scene': self.statistics.appearances_by_scene,
            },
            'aliases': self.aliases,
            'description': self.description,
            'color': self.color,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'recognition_threshold': self.recognition_threshold,
            'appearance_variance': self.appearance_variance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Character':
        """从字典创建实例"""
        statistics = CharacterStatistics(
            total_samples=data.get('statistics', {}).get('total_samples', 0),
            avg_quality=data.get('statistics', {}).get('avg_quality', 0.0),
            first_appearance=data.get('statistics', {}).get('first_appearance', 0.0),
            last_appearance=data.get('statistics', {}).get('last_appearance', 0.0),
            total_screen_time=data.get('statistics', {}).get('total_screen_time', 0.0),
            appearances_by_scene=data.get('statistics', {}).get('appearances_by_scene', {}),
        )

        return cls(
            character_id=data['character_id'],
            name=data['name'],
            video_id=data['video_id'],
            prototypes=[Prototype.from_dict(p) for p in data.get('prototypes', [])],
            samples=data.get('samples', []),
            statistics=statistics,
            aliases=data.get('aliases', []),
            description=data.get('description', ''),
            color=data.get('color'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            recognition_threshold=data.get('recognition_threshold', 0.5),
            appearance_variance=data.get('appearance_variance', 0.3),
        )


@dataclass
class CharacterLibrary:
    """角色库"""

    version: str = "1.0"
    video_info: Optional[Dict[str, Any]] = None
    characters: List[Character] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def character_count(self) -> int:
        """获取角色数量"""
        return len(self.characters)

    def add_character(self, character: Character):
        """添加角色"""
        self.characters.append(character)
        self.updated_at = datetime.now().isoformat()

    def get_character(self, character_id: str) -> Optional[Character]:
        """获取角色"""
        for char in self.characters:
            if char.character_id == character_id:
                return char
        return None

    def get_character_by_name(self, name: str) -> Optional[Character]:
        """按名称获取角色"""
        for char in self.characters:
            if char.name == name or name in char.aliases:
                return char
        return None

    def remove_character(self, character_id: str) -> bool:
        """移除角色"""
        for i, char in enumerate(self.characters):
            if char.character_id == character_id:
                self.characters.pop(i)
                self.updated_at = datetime.now().isoformat()
                return True
        return False

    def save(self, file_path: str):
        """保存到文件"""
        data = {
            'version': self.version,
            'video_info': self.video_info,
            'characters': [char.to_dict() for char in self.characters],
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: str) -> 'CharacterLibrary':
        """从文件加载"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        library = cls(
            version=data.get('version', '1.0'),
            video_info=data.get('video_info'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
        )

        for char_data in data.get('characters', []):
            library.characters.append(Character.from_dict(char_data))

        return library

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'version': self.version,
            'video_info': self.video_info,
            'characters': [char.to_dict() for char in self.characters],
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }
