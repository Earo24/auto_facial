"""
识别引擎
基于角色库进行人脸识别
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
import logging

from config.settings import RECOGNITION_CONFIG
from src.models.character import Character
from src.models.face_sample import FaceSample
from src.core.face_embedder import FaceEmbedder, FaceMatcher

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """识别结果"""
    sample_id: str
    character_id: Optional[str]
    character_name: Optional[str]
    confidence: float
    timestamp: float
    bbox: List[float]


class TemporalSmoother:
    """时序平滑器 - 减少识别抖动"""

    def __init__(self, window_size: int = 10):
        """
        初始化时序平滑器

        Args:
            window_size: 时间窗口大小（帧数）
        """
        self.window_size = window_size
        self.history: Dict[int, deque] = {}  # {face_id: deque of (character_id, confidence)}

    def add_result(self, face_id: int, character_id: str, confidence: float):
        """添加识别结果到历史"""
        if face_id not in self.history:
            self.history[face_id] = deque(maxlen=self.window_size)

        self.history[face_id].append((character_id, confidence))

    def get_smoothed_result(self, face_id: int) -> Optional[Tuple[str, float]]:
        """
        获取平滑后的识别结果

        Args:
            face_id: 人脸ID

        Returns:
            (character_id, avg_confidence) 或 None
        """
        if face_id not in self.history or not self.history[face_id]:
            return None

        # 统计窗口内各角色出现次数
        char_counts = defaultdict(list)
        for char_id, conf in self.history[face_id]:
            char_counts[char_id].append(conf)

        # 找出出现最多次的角色
        best_char = max(char_counts.items(), key=lambda x: len(x[1]))
        char_id, confidences = best_char

        # 计算平均置信度
        avg_confidence = sum(confidences) / len(confidences)

        # 如果出现次数占多数，返回该角色
        if len(confidences) >= len(self.history[face_id]) * 0.6:
            return (char_id, avg_confidence)

        return None

    def reset(self):
        """重置历史"""
        self.history.clear()


class RecognitionEngine:
    """识别引擎 - 多策略人脸识别"""

    def __init__(self, embedder: FaceEmbedder):
        """
        初始化识别引擎

        Args:
            embedder: 人脸特征提取器
        """
        self.embedder = embedder
        self.matcher = FaceMatcher(embedder)
        self.config = RECOGNITION_CONFIG
        self.characters: Dict[str, Character] = {}  # {character_id: Character}
        self.smoother = TemporalSmoother(window_size=30)

    def load_characters(self, characters: List[Character]):
        """
        加载角色库

        Args:
            characters: 角色列表
        """
        self.characters.clear()
        self.matcher.character_embeddings.clear()

        for char in characters:
            self.characters[char.character_id] = char

            # 提取原型特征
            embeddings = [p.embedding for p in char.prototypes]
            self.matcher.add_character(char.character_id, embeddings)

        logger.info(f"加载角色库: {len(characters)} 个角色")

    def add_character(self, character: Character):
        """
        添加角色

        Args:
            character: 角色对象
        """
        self.characters[character.character_id] = character

        embeddings = [p.embedding for p in character.prototypes]
        self.matcher.add_character(character.character_id, embeddings)

    def recognize(self, sample: FaceSample,
                 use_temporal_smoothing: bool = False,
                 face_id: Optional[int] = None) -> Optional[RecognitionResult]:
        """
        识别人脸样本

        Args:
            sample: 人脸样本
            use_temporal_smoothing: 是否使用时序平滑
            face_id: 人脸ID（用于时序平滑）

        Returns:
            识别结果
        """
        if not sample.has_embedding:
            return None

        # 基础识别
        result = self.matcher.match(
            sample.embedding,
            min_similarity=self.config['low_confidence_threshold']
        )

        if result is None:
            return RecognitionResult(
                sample_id=sample.sample_id,
                character_id=None,
                character_name=None,
                confidence=0.0,
                timestamp=sample.timestamp,
                bbox=sample.bbox,
            )

        character_id, confidence = result
        character = self.characters.get(character_id)

        # 应用时序平滑
        if use_temporal_smoothing and face_id is not None:
            self.smoother.add_result(face_id, character_id, confidence)
            smoothed = self.smoother.get_smoothed_result(face_id)

            if smoothed:
                character_id, confidence = smoothed
                character = self.characters.get(character_id)
            else:
                # 平滑结果不可靠，返回未知
                character_id = None
                character = None
                confidence = 0.0

        return RecognitionResult(
            sample_id=sample.sample_id,
            character_id=character_id,
            character_name=character.name if character else None,
            confidence=confidence,
            timestamp=sample.timestamp,
            bbox=sample.bbox,
        )

    def recognize_batch(self, samples: List[FaceSample],
                       use_temporal_smoothing: bool = False) -> List[RecognitionResult]:
        """
        批量识别

        Args:
            samples: 人脸样本列表
            use_temporal_smoothing: 是否使用时序平滑

        Returns:
            识别结果列表
        """
        results = []

        # 如果使用时序平滑，需要按时间排序并跟踪人脸
        if use_temporal_smoothing:
            self.smoother.reset()
            samples = sorted(samples, key=lambda s: s.timestamp)

            # 简单的人脸跟踪（基于位置和连续性）
            face_tracker = {}

        for i, sample in enumerate(samples):
            face_id = i if use_temporal_smoothing else None
            result = self.recognize(sample, use_temporal_smoothing, face_id)
            if result:
                results.append(result)

        return results

    def handle_appearance_changes(self, character: Character,
                                 new_samples: List[FaceSample],
                                 threshold: float = 0.4) -> bool:
        """
        处理角色造型变化（如妆容、发型变化）

        Args:
            character: 角色对象
            new_samples: 新的人脸样本
            threshold: 相似度阈值

        Returns:
            是否成功处理
        """
        if not character.prototypes:
            return False

        # 找出与现有原型相似度低但可能属于同一角色的样本
        avg_prototype = character.avg_prototype_embedding
        if avg_prototype is None:
            return False

        new_prototypes = []
        for sample in new_samples:
            if not sample.has_embedding:
                continue

            similarity = self.embedder.compute_similarity(avg_prototype, sample.embedding)

            # 如果相似度在合理范围内，可能是造型变化
            if threshold <= similarity < self.config['similarity_threshold']:
                new_prototypes.append(sample)

        # 添加新的原型样本
        for sample in new_prototypes:
            character.add_prototype(
                embedding=sample.embedding,
                image_path=sample.image_path or "",
                quality_score=sample.quality_score,
                timestamp=sample.timestamp,
            )

        if new_prototypes:
            # 重新加载角色到匹配器
            embeddings = [p.embedding for p in character.prototypes]
            self.matcher.add_character(character.character_id, embeddings)
            logger.info(f"为角色 {character.name} 添加了 {len(new_prototypes)} 个新原型样本")

        return len(new_prototypes) > 0

    def update_character_from_samples(self, character_id: str,
                                     high_confidence_samples: List[Tuple[FaceSample, float]]):
        """
        使用高置信度识别结果更新角色

        Args:
            character_id: 角色ID
            high_confidence_samples: (样本, 置信度) 列表
        """
        character = self.characters.get(character_id)
        if not character:
            return

        for sample, confidence in high_confidence_samples:
            if confidence >= self.config['similarity_threshold'] and sample.has_embedding:
                # 检查是否可以作为新原型
                should_add = True
                for prototype in character.prototypes:
                    sim = self.embedder.compute_similarity(prototype.embedding, sample.embedding)
                    if sim > 0.95:  # 太相似，不需要添加
                        should_add = False
                        break

                if should_add and sample.image_path:
                    character.add_prototype(
                        embedding=sample.embedding,
                        image_path=sample.image_path,
                        quality_score=sample.quality_score,
                        timestamp=sample.timestamp,
                    )

        # 更新匹配器
        embeddings = [p.embedding for p in character.prototypes]
        self.matcher.add_character(character_id, embeddings)

    def get_recognition_statistics(self, results: List[RecognitionResult]) -> Dict:
        """
        获取识别统计信息

        Args:
            results: 识别结果列表

        Returns:
            统计信息字典
        """
        total = len(results)
        if total == 0:
            return {}

        high_conf = sum(1 for r in results if r.confidence >= self.config['similarity_threshold'])
        low_conf = sum(1 for r in results if self.config['low_confidence_threshold'] <= r.confidence < self.config['similarity_threshold'])
        unknown = sum(1 for r in results if r.character_id is None)

        char_counts = defaultdict(int)
        for result in results:
            if result.character_id:
                char_counts[result.character_id] += 1

        return {
            'total_samples': total,
            'high_confidence': high_conf,
            'low_confidence': low_conf,
            'unknown': unknown,
            'character_counts': dict(char_counts),
        }
