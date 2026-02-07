"""
人脸检测模块
使用InsightFace进行人脸检测和质量评估
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging

from config.settings import FACE_DETECTION_CONFIG
from src.models.face_sample import FaceSample
from src.utils.image_utils import calculate_image_quality, extract_face_region, align_face

logger = logging.getLogger(__name__)


class FaceDetector:
    """人脸检测器 - 基于InsightFace"""

    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        """
        初始化人脸检测器

        Args:
            model_path: 模型路径（可选，默认使用预训练模型）
            use_gpu: 是否使用GPU
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.detector = None
        self.config = FACE_DETECTION_CONFIG
        self.video_height = None  # 视频高度，用于动态计算最小人脸尺寸

    def set_video_resolution(self, width: int, height: int):
        """设置视频分辨率，用于动态计算最小人脸尺寸"""
        self.video_width = width
        self.video_height = height
        # 根据视频高度动态计算最小人脸尺寸
        self.min_face_size = int(height * self.config['min_face_size_ratio'])
        # 确保不小于绝对最小值
        self.min_face_size = max(self.min_face_size, self.config['min_face_size_absolute'])
        logger.info(f"视频分辨率: {width}x{height}, 动态最小人脸尺寸: {self.min_face_size}px")

    def initialize(self):
        """初始化检测模型"""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            import onnxruntime as ort

            # 根据可用执行提供器选择最佳配置
            available_providers = ort.get_available_providers()

            if self.use_gpu:
                # M2 Mac使用CoreML，其他GPU使用CUDA
                if 'CoreMLExecutionProvider' in available_providers:
                    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
                    ctx_id = 0
                    logger.info("Using CoreML GPU acceleration for M2 chip")
                elif 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    ctx_id = 0
                    logger.info("Using CUDA GPU acceleration")
                else:
                    providers = ['CPUExecutionProvider']
                    ctx_id = -1
                    logger.warning("GPU not available, using CPU")
            else:
                providers = ['CPUExecutionProvider']
                ctx_id = -1

            self.detector = FaceAnalysis(
                roots=self.model_path,
                providers=providers
            )
            self.detector.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info(f"Face detector initialized with providers: {providers}")
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            raise

    def detect_faces(self, image: np.ndarray, frame_id: str = "",
                     video_id: str = "", timestamp: float = 0.0,
                     frame_number: int = 0) -> List[FaceSample]:
        """
        检测图像中的人脸

        Args:
            image: 输入图像（BGR格式）
            frame_id: 帧ID
            video_id: 视频ID
            timestamp: 时间戳
            frame_number: 帧序号

        Returns:
            检测到的人脸样本列表
        """
        if self.detector is None:
            self.initialize()

        if image is None or image.size == 0:
            return []

        # 执行检测
        faces = self.detector.get(image)

        samples = []
        for i, face in enumerate(faces):
            # 过滤低置信度检测
            if face.det_score < self.config['confidence_threshold']:
                continue

            # 获取边界框
            bbox = face.bbox.astype(float).tolist()  # [x1, y1, x2, y2]

            # 计算人脸尺寸
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]

            # 动态计算最小人脸尺寸（如果还没设置则使用默认值）
            min_size = self.min_face_size if hasattr(self, 'min_face_size') else 80

            # 过滤小脸（使用动态阈值）
            if face_width < min_size or face_height < min_size:
                continue

            # 获取关键点
            landmarks = face.landmark.astype(float).tolist() if face.landmark is not None else None

            # 计算质量分数
            quality_score = self.assess_quality(image, bbox, face)

            # 过滤低质量样本
            if quality_score < self.config['quality_threshold']:
                continue

            # 创建人脸样本
            sample = FaceSample(
                sample_id=f"{frame_id}_face_{i}",
                frame_id=frame_id,
                video_id=video_id,
                bbox=bbox,
                landmarks=landmarks,
                quality_score=quality_score,
                timestamp=timestamp,
                frame_number=frame_number,
                face_size=(int(face_width), int(face_height)),
            )

            samples.append(sample)

            # 限制每帧最多检测人脸数
            if len(samples) >= self.config['max_faces_per_frame']:
                break

        return samples

    def detect_single(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        检测单张图像中的人脸（简化版）

        Args:
            image: 输入图像

        Returns:
            人脸信息字典，如果没有检测到则返回None
        """
        if self.detector is None:
            self.initialize()

        faces = self.detector.get(image)

        if not faces:
            return None

        # 返回置信度最高的人脸
        face = max(faces, key=lambda f: f.det_score)

        return {
            'bbox': face.bbox.astype(float).tolist(),
            'landmarks': face.landmark.astype(float).tolist() if face.landmark is not None else None,
            'confidence': float(face.det_score),
            'embedding': face.embedding.astype(float).tolist() if hasattr(face, 'embedding') and face.embedding is not None else None,
        }

    def assess_quality(self, image: np.ndarray, bbox: List[float],
                      face_obj: Optional[Any] = None) -> float:
        """
        评估人脸质量

        Args:
            image: 输入图像
            bbox: 人脸边界框
            face_obj: InsightFace人脸对象（可选）

        Returns:
            质量分数 (0-1)
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        # 基本图像质量
        image_quality = calculate_image_quality(image, bbox)

        # 如果有InsightFace对象，使用其质量评估
        if face_obj is not None and hasattr(face_obj, 'det_score'):
            detection_confidence = float(face_obj.det_score)
            # 综合图像质量和检测置信度
            return (image_quality + detection_confidence) / 2

        return image_quality

    def filter_high_quality(self, samples: List[FaceSample],
                           min_quality: Optional[float] = None) -> List[FaceSample]:
        """
        过滤高质量样本

        Args:
            samples: 人脸样本列表
            min_quality: 最小质量阈值（默认使用配置值）

        Returns:
            高质量样本列表
        """
        threshold = min_quality or self.config['quality_threshold']
        return [s for s in samples if s.quality_score >= threshold]

    def deduplicate_faces(self, samples: List[FaceSample],
                         threshold: Optional[float] = None) -> List[FaceSample]:
        """
        去除重复的人脸样本（基于时间和相似度）

        Args:
            samples: 人脸样本列表
            threshold: 相似度阈值（默认使用配置值）

        Returns:
            去重后的样本列表
        """
        if threshold is None:
            threshold = self.config['dedup_threshold']

        # 按时间戳排序
        sorted_samples = sorted(samples, key=lambda s: s.timestamp)

        unique_samples = []
        for sample in sorted_samples:
            is_duplicate = False
            for existing in unique_samples:
                if sample.is_duplicate(existing, threshold):
                    # 保留质量更高的样本
                    if sample.quality_score > existing.quality_score:
                        unique_samples.remove(existing)
                        unique_samples.append(sample)
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_samples.append(sample)

        return unique_samples

    def extract_and_save_faces(self, image: np.ndarray, samples: List[FaceSample],
                               output_dir: Path, align: bool = True) -> List[FaceSample]:
        """
        提取并保存人脸图像

        Args:
            image: 原始图像
            samples: 人脸样本列表
            output_dir: 输出目录
            align: 是否对齐人脸

        Returns:
            更新后的人脸样本列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            # 提取人脸区域
            if align and sample.landmarks:
                landmarks_array = np.array(sample.landmarks)
                face_img = align_face(image, landmarks_array)
            else:
                face_img = extract_face_region(image, sample.bbox)

            # 保存人脸图像
            face_filename = f"{sample.sample_id}.jpg"
            face_path = output_dir / face_filename
            cv2.imwrite(str(face_path), face_img)

            # 更新样本路径
            sample.image_path = str(face_path)

        return samples

    def draw_detections(self, image: np.ndarray, samples: List[FaceSample]) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            image: 输入图像
            samples: 人脸样本列表

        Returns:
            绘制后的图像
        """
        result = image.copy()

        for sample in samples:
            x1, y1, x2, y2 = [int(coord) for coord in sample.bbox]

            # 根据质量选择颜色
            if sample.quality_score >= 0.8:
                color = (0, 255, 0)  # 绿色 - 高质量
            elif sample.quality_score >= 0.6:
                color = (0, 165, 255)  # 橙色 - 中等质量
            else:
                color = (0, 0, 255)  # 红色 - 低质量

            # 绘制边界框
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # 绘制关键点
            if sample.landmarks:
                for point in sample.landmarks:
                    cv2.circle(result, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)

            # 绘制质量分数
            label = f"Q: {sample.quality_score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 绘制样本ID
            if sample.character_id:
                cv2.putText(result, sample.character_id, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return result
