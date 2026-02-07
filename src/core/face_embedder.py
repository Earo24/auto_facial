"""
人脸特征提取模块
使用InsightFace提取人脸特征向量
"""
import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from config.settings import MODEL_CONFIG
from src.models.face_sample import FaceSample

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """人脸特征提取器 - 基于InsightFace"""

    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        """
        初始化特征提取器

        Args:
            model_path: 模型路径（可选）
            use_gpu: 是否使用GPU
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.embedder = None
        self.embedding_dim = 512  # ArcFace默认维度

    def initialize(self):
        """初始化特征提取模型"""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            import onnxruntime as ort

            # 根据可用执行提供器选择最佳配置
            available_providers = ort.get_available_providers()

            if self.use_gpu:
                if 'CoreMLExecutionProvider' in available_providers:
                    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
                    ctx_id = 0
                    logger.info("Using CoreML GPU acceleration for M2 chip (embedder)")
                elif 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    ctx_id = 0
                    logger.info("Using CUDA GPU acceleration (embedder)")
                else:
                    providers = ['CPUExecutionProvider']
                    ctx_id = -1
                    logger.warning("GPU not available, using CPU (embedder)")
            else:
                providers = ['CPUExecutionProvider']
                ctx_id = -1

            self.embedder = FaceAnalysis(
                roots=self.model_path,
                providers=providers
            )
            # 准备模型（包括recognition模型）
            self.embedder.prepare(ctx_id=ctx_id, det_size=(640, 640))

            logger.info(f"Face embedder initialized with providers: {providers}")
        except Exception as e:
            logger.error(f"Failed to initialize face embedder: {e}")
            raise

    def compute_embedding(self, image: np.ndarray, face_bbox: Optional[List[float]] = None,
                         face_landmarks: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        计算单张人脸的特征向量

        Args:
            image: 输入图像（BGR格式）
            face_bbox: 人脸边界框 [x1, y1, x2, y2]（可选）
            face_landmarks: 5点关键点（可选）

        Returns:
            特征向量 (512维)，失败返回None
        """
        if self.embedder is None:
            self.initialize()

        try:
            # 使用InsightFace获取特征
            faces = self.embedder.get(image)

            if not faces:
                return None

            # 如果指定了边界框，找到最匹配的人脸
            if face_bbox is not None:
                best_face = None
                best_iou = 0

                x1, y1, x2, y2 = face_bbox
                bbox_area = (x2 - x1) * (y2 - y1)

                for face in faces:
                    fx1, fy1, fx2, fy2 = face.bbox
                    # 计算IoU
                    inter_x1 = max(x1, fx1)
                    inter_y1 = max(y1, fy1)
                    inter_x2 = min(x2, fx2)
                    inter_y2 = min(y2, fy2)

                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        union_area = bbox_area + ((fx2 - fx1) * (fy2 - fy1)) - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0

                        if iou > best_iou:
                            best_iou = iou
                            best_face = face

                if best_face is not None and best_iou > 0.5 and hasattr(best_face, 'embedding'):
                    return best_face.embedding
            else:
                # 返回检测到的第一个人脸的特征
                if faces[0].embedding is not None:
                    return faces[0].embedding

            return None

        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return None

    def compute_embeddings_batch(self, images: List[np.ndarray],
                                 face_bboxes: Optional[List[List[float]]] = None) -> List[Optional[np.ndarray]]:
        """
        批量计算人脸特征向量

        Args:
            images: 图像列表
            face_bboxes: 人脸边界框列表（可选）

        Returns:
            特征向量列表
        """
        embeddings = []

        for i, image in enumerate(images):
            bbox = face_bboxes[i] if face_bboxes and i < len(face_bboxes) else None
            embedding = self.compute_embedding(image, bbox)
            embeddings.append(embedding)

        return embeddings

    def extract_embeddings(self, samples: List[FaceSample],
                          images: Dict[str, np.ndarray]) -> List[FaceSample]:
        """
        为人脸样本提取特征向量

        Args:
            samples: 人脸样本列表
            images: 图像字典 {frame_id: image}

        Returns:
            更新后的人脸样本列表
        """
        for sample in samples:
            if sample.frame_id not in images:
                continue

            image = images[sample.frame_id]
            embedding = self.compute_embedding(image, sample.bbox, sample.landmarks)

            if embedding is not None:
                sample.embedding = embedding

        return samples

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        计算两个特征向量的余弦相似度

        Args:
            embedding1: 特征向量1
            embedding2: 特征向量2

        Returns:
            相似度分数 (0-1)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0

        # 确保是numpy数组
        emb1 = np.array(embedding1).flatten()
        emb2 = np.array(embedding2).flatten()

        # 余弦相似度
        dot_product = np.dot(emb1, emb2)
        norm_a = np.linalg.norm(emb1)
        norm_b = np.linalg.norm(emb2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        计算两个特征向量的欧氏距离

        Args:
            embedding1: 特征向量1
            embedding2: 特征向量2

        Returns:
            距离值
        """
        if embedding1 is None or embedding2 is None:
            return float('inf')

        emb1 = np.array(embedding1).flatten()
        emb2 = np.array(embedding2).flatten()

        return float(np.linalg.norm(emb1 - emb2))

    def find_similar_faces(self, query_embedding: np.ndarray,
                          embedding_list: List[np.ndarray],
                          threshold: float = 0.5,
                          top_k: int = 5) -> List[tuple]:
        """
        查找相似的人脸

        Args:
            query_embedding: 查询特征向量
            embedding_list: 候选特征向量列表
            threshold: 相似度阈值
            top_k: 返回前k个最相似的结果

        Returns:
            [(index, similarity), ...] 按相似度降序排列
        """
        similarities = []

        for i, emb in enumerate(embedding_list):
            if emb is not None:
                sim = self.compute_similarity(query_embedding, emb)
                if sim >= threshold:
                    similarities.append((i, sim))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def compute_average_embedding(self, embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        计算平均特征向量

        Args:
            embeddings: 特征向量列表

        Returns:
            平均特征向量
        """
        valid_embeddings = [e for e in embeddings if e is not None]

        if not valid_embeddings:
            return None

        return np.mean(valid_embeddings, axis=0)

    def save_embeddings(self, samples: List[FaceSample], output_path: str):
        """
        保存特征向量到文件

        Args:
            samples: 人脸样本列表
            output_path: 输出文件路径（.npy格式）
        """
        embeddings = []

        for sample in samples:
            if sample.has_embedding:
                embeddings.append({
                    'sample_id': sample.sample_id,
                    'embedding': sample.embedding,
                    'character_id': sample.character_id,
                })

        np.save(output_path, embeddings)
        logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")

    def load_embeddings(self, input_path: str) -> Dict[str, np.ndarray]:
        """
        从文件加载特征向量

        Args:
            input_path: 输入文件路径

        Returns:
            {sample_id: embedding} 字典
        """
        data = np.load(input_path, allow_pickle=True)

        embeddings = {}
        for item in data:
            embeddings[item['sample_id']] = item['embedding']

        logger.info(f"Loaded {len(embeddings)} embeddings from {input_path}")
        return embeddings


class FaceMatcher:
    """人脸匹配器 - 用于识别阶段"""

    def __init__(self, embedder: FaceEmbedder):
        """
        初始化匹配器

        Args:
            embedder: 人脸特征提取器实例
        """
        self.embedder = embedder
        self.character_embeddings: Dict[str, List[np.ndarray]] = {}  # {character_id: [embeddings]}

    def add_character(self, character_id: str, embeddings: List[np.ndarray]):
        """
        添加角色的特征向量

        Args:
            character_id: 角色ID
            embeddings: 特征向量列表
        """
        self.character_embeddings[character_id] = embeddings

    def match(self, embedding: np.ndarray, min_similarity: float = 0.4) -> Optional[tuple]:
        """
        匹配人脸到角色

        Args:
            embedding: 待匹配的特征向量
            min_similarity: 最小相似度阈值

        Returns:
            (character_id, similarity) 或 None
        """
        best_match = None
        best_similarity = min_similarity

        for char_id, embeddings in self.character_embeddings.items():
            for emb in embeddings:
                sim = self.embedder.compute_similarity(embedding, emb)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = char_id

        if best_match:
            return (best_match, best_similarity)

        return None

    def match_top_k(self, embedding: np.ndarray, top_k: int = 3,
                   min_similarity: float = 0.4) -> List[tuple]:
        """
        匹配人脸到角色（返回前k个结果）

        Args:
            embedding: 待匹配的特征向量
            top_k: 返回前k个结果
            min_similarity: 最小相似度阈值

        Returns:
            [(character_id, similarity), ...] 按相似度降序
        """
        results = []

        for char_id, embeddings in self.character_embeddings.items():
            # 计算与该角色所有原型的最大相似度
            max_sim = 0.0
            for emb in embeddings:
                sim = self.embedder.compute_similarity(embedding, emb)
                max_sim = max(max_sim, sim)

            if max_sim >= min_similarity:
                results.append((char_id, max_sim))

        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]
