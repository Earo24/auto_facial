"""
视频处理引擎
处理视频文件，提取帧并检测人脸
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Callable
from pathlib import Path
from tqdm import tqdm
import logging

from config.settings import VIDEO_CONFIG, FRAMES_DIR, FACES_DIR
from src.models.video_frame import VideoFrame, VideoInfo
from src.models.face_sample import FaceSample
from src.core.face_detector import FaceDetector
from src.core.face_embedder import FaceEmbedder
from src.utils.video_utils import smart_sample_frames

logger = logging.getLogger(__name__)


class VideoProcessor:
    """视频处理引擎 - 处理视频提取和人脸检测"""

    def __init__(self, detector: FaceDetector, embedder: FaceEmbedder):
        """
        初始化视频处理器

        Args:
            detector: 人脸检测器
            embedder: 特征提取器
        """
        self.detector = detector
        self.embedder = embedder
        self.config = VIDEO_CONFIG

    def process_video(self, video_path: str, video_id: str,
                     output_dir: Optional[Path] = None,
                     progress_callback: Optional[Callable] = None) -> dict:
        """
        完整处理视频流程

        Args:
            video_path: 视频文件路径
            video_id: 视频ID
            output_dir: 输出目录
            progress_callback: 进度回调

        Returns:
            处理结果字典
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # 设置输出目录
        if output_dir is None:
            output_dir = FRAMES_DIR / video_id
        else:
            output_dir = Path(output_dir) / video_id

        # 获取视频信息
        video_info = VideoInfo.from_file(str(video_path))

        # 设置检测器的动态阈值（根据视频分辨率）
        self.detector.set_video_resolution(video_info.width, video_info.height)

        # 智能采样提取帧
        frames_data = smart_sample_frames(
            str(video_path),
            output_dir,
            base_sample_fps=self.config['sample_fps'],
            scene_threshold=self.config['scene_threshold'],
            max_frames=self.config['max_frames'],
            progress_callback=progress_callback
        )

        # 处理每一帧
        all_samples = []
        frame_objects = []

        for frame_number, frame_path, timestamp, is_scene_change in tqdm(frames_data, desc="处理帧"):
            # 读取帧
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            # 创建帧对象
            frame_id = f"{video_id}_frame_{frame_number:06d}"
            video_frame = VideoFrame(
                frame_id=frame_id,
                video_id=video_id,
                frame_number=frame_number,
                timestamp=timestamp,
                image_path=frame_path,
                width=frame.shape[1],
                height=frame.shape[0],
                is_scene_change=is_scene_change,
            )

            # 检测人脸
            samples = self.detector.detect_faces(
                frame,
                frame_id=frame_id,
                video_id=video_id,
                timestamp=timestamp,
                frame_number=frame_number,
            )

            # 提取特征
            if samples:
                samples = self.embedder.extract_embeddings(samples, {frame_id: frame})

                # 保存人脸图像
                face_output_dir = FACES_DIR / video_id
                samples = self.detector.extract_and_save_faces(frame, samples, face_output_dir)

                # 更新帧对象
                video_frame.face_count = len(samples)
                video_frame.face_ids = [s.sample_id for s in samples]

                all_samples.extend(samples)

            frame_objects.append(video_frame)

            if progress_callback:
                progress = (len(frame_objects) / len(frames_data)) * 100
                progress_callback(progress, f"处理帧: {len(frame_objects)}/{len(frames_data)}")

        # 更新视频信息
        video_info.processed_frames = len(frame_objects)
        video_info.detected_faces = len(all_samples)

        # 去重
        unique_samples = self.detector.deduplicate_faces(all_samples)

        logger.info(f"视频处理完成: {video_id}")
        logger.info(f"  - 提取帧数: {len(frame_objects)}")
        logger.info(f"  - 检测人脸: {len(all_samples)}")
        logger.info(f"  - 去重后: {len(unique_samples)}")

        return {
            'video_info': video_info,
            'frames': frame_objects,
            'face_samples': unique_samples,
            'all_samples': all_samples,
        }

    def extract_frames_only(self, video_path: str, video_id: str,
                           output_dir: Optional[Path] = None) -> List[VideoFrame]:
        """
        仅提取帧（不进行人脸检测）

        Args:
            video_path: 视频文件路径
            video_id: 视频ID
            output_dir: 输出目录

        Returns:
            视频帧对象列表
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if output_dir is None:
            output_dir = FRAMES_DIR / video_id
        else:
            output_dir = Path(output_dir) / video_id

        # 智能采样
        frames_data = smart_sample_frames(
            str(video_path),
            output_dir,
            base_sample_fps=self.config['sample_fps'],
            scene_threshold=self.config['scene_threshold'],
            max_frames=self.config['max_frames'],
        )

        # 创建帧对象
        frame_objects = []
        for frame_number, frame_path, timestamp, is_scene_change in frames_data:
            frame_id = f"{video_id}_frame_{frame_number:06d}"

            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            video_frame = VideoFrame(
                frame_id=frame_id,
                video_id=video_id,
                frame_number=frame_number,
                timestamp=timestamp,
                image_path=frame_path,
                width=frame.shape[1],
                height=frame.shape[0],
                is_scene_change=is_scene_change,
            )

            frame_objects.append(video_frame)

        logger.info(f"提取帧完成: {len(frame_objects)} 帧")
        return frame_objects

    def detect_faces_batch(self, frames: List[VideoFrame],
                          progress_callback: Optional[Callable] = None) -> List[FaceSample]:
        """
        批量检测人脸

        Args:
            frames: 视频帧对象列表
            progress_callback: 进度回调

        Returns:
            人脸样本列表
        """
        all_samples = []

        for i, frame in enumerate(tqdm(frames, desc="检测人脸")):
            # 读取帧
            image = frame.load_image()
            if image is None:
                continue

            # 检测人脸
            samples = self.detector.detect_faces(
                image,
                frame_id=frame.frame_id,
                video_id=frame.video_id,
                timestamp=frame.timestamp,
                frame_number=frame.frame_number,
            )

            # 更新帧信息
            frame.face_count = len(samples)
            frame.face_ids = [s.sample_id for s in samples]

            all_samples.extend(samples)

            if progress_callback:
                progress = (i + 1) / len(frames) * 100
                progress_callback(progress, f"检测: {i + 1}/{len(frames)}")

        return all_samples

    def extract_embeddings_batch(self, samples: List[FaceSample],
                                progress_callback: Optional[Callable] = None) -> List[FaceSample]:
        """
        批量提取特征向量

        Args:
            samples: 人脸样本列表
            progress_callback: 进度回调

        Returns:
            更新后的人脸样本列表
        """
        # 按帧分组
        frames_dict: Dict[str, List[FaceSample]] = {}
        for sample in samples:
            if sample.frame_id not in frames_dict:
                frames_dict[sample.frame_id] = []
            frames_dict[sample.frame_id].append(sample)

        # 加载图像
        images: Dict[str, np.ndarray] = {}
        for frame_id, frame_samples in tqdm(frames_dict.items(), desc="加载帧图像"):
            if frame_samples and frame_samples[0].image_path:
                image = cv2.imread(frame_samples[0].image_path)
                if image is not None:
                    images[frame_id] = image

        # 提取特征
        samples_with_embeddings = self.embedder.extract_embeddings(samples, images)

        if progress_callback:
            progress_callback(100, f"特征提取完成")

        return samples_with_embeddings

    def create_video_summary(self, video_path: str,
                            output_path: str,
                            num_frames: int = 10) -> bool:
        """
        创建视频摘要（关键帧缩略图）

        Args:
            video_path: 视频路径
            output_path: 输出图像路径
            num_frames: 关键帧数量

        Returns:
            是否成功
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = total_frames // num_frames

        frames = []
        for i in range(num_frames):
            frame_number = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                # 缩放
                frame = cv2.resize(frame, (320, 180))
                frames.append(frame)

        cap.release()

        if not frames:
            return False

        # 拼接图像
        rows = []
        for i in range(0, len(frames), 5):
            row = frames[i:i+5]
            if len(row) < 5:
                row.extend([np.zeros_like(row[0])] * (5 - len(row)))
            rows.append(np.hstack(row))

        result = np.vstack(rows)
        cv2.imwrite(output_path, result)

        return True
