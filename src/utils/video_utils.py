"""
视频处理工具函数
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


def get_video_info(video_path: str) -> dict:
    """
    获取视频信息

    Args:
        video_path: 视频文件路径

    Returns:
        视频信息字典
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': 0.0,
    }

    info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0

    cap.release()

    return info


def extract_frames(video_path: str, output_dir: Path,
                   sample_fps: float = 1.0,
                   max_frames: int = 10000,
                   progress_callback=None) -> List[Tuple[int, str, float]]:
    """
    从视频中提取帧

    Args:
        video_path: 视频路径
        output_dir: 输出目录
        sample_fps: 采样帧率
        max_frames: 最大提取帧数
        progress_callback: 进度回调函数

    Returns:
        [(frame_number, image_path, timestamp), ...]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(video_fps / sample_fps) if sample_fps > 0 else 1

    extracted = []
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 跳帧采样
        if frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps
            frame_filename = f"frame_{frame_count:06d}.jpg"
            frame_path = output_dir / frame_filename

            cv2.imwrite(str(frame_path), frame)
            extracted.append((frame_count, str(frame_path), timestamp))

            extracted_count += 1

            if progress_callback:
                progress = frame_count / total_frames
                progress_callback(progress, f"提取帧: {extracted_count}/{max_frames}")

            if extracted_count >= max_frames:
                break

        frame_count += 1

    cap.release()

    return extracted


def detect_scene_change(frame1: np.ndarray, frame2: np.ndarray,
                       threshold: float = 30.0) -> bool:
    """
    检测场景变化

    Args:
        frame1: 前一帧
        frame2: 当前帧
        threshold: 变化阈值

    Returns:
        是否为场景切换
    """
    # 转换为灰度图
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 计算差异
    diff = cv2.absdiff(gray1, gray2)
    mean_diff = np.mean(diff)

    return mean_diff > threshold


def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    计算两帧之间的差异程度

    Args:
        frame1: 第一帧
        frame2: 第二帧

    Returns:
        差异值 (0-255)
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    return float(np.mean(diff))


def smart_sample_frames(video_path: str, output_dir: Path,
                        base_sample_fps: float = 1.0,
                        scene_threshold: float = 30.0,
                        max_frames: int = 10000,
                        progress_callback=None) -> List[Tuple[int, str, float, bool]]:
    """
    智能帧采样（结合场景变化检测）

    Args:
        video_path: 视频路径
        output_dir: 输出目录
        base_sample_fps: 基础采样帧率
        scene_threshold: 场景变化阈值
        max_frames: 最大提取帧数
        progress_callback: 进度回调

    Returns:
        [(frame_number, image_path, timestamp, is_scene_change), ...]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(video_fps / base_sample_fps) if base_sample_fps > 0 else 1

    extracted = []
    frame_count = 0
    extracted_count = 0
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / video_fps
        is_scene_change = False
        should_extract = False

        # 场景变化检测
        if prev_frame is not None:
            diff = calculate_frame_difference(prev_frame, frame)
            if diff > scene_threshold:
                is_scene_change = True
                should_extract = True

        # 定时采样
        if frame_count % frame_interval == 0:
            should_extract = True

        if should_extract and extracted_count < max_frames:
            frame_filename = f"frame_{frame_count:06d}"
            if is_scene_change:
                frame_filename += "_scene"
            frame_filename += ".jpg"

            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)

            extracted.append((frame_count, str(frame_path), timestamp, is_scene_change))
            extracted_count += 1

            if progress_callback:
                progress = frame_count / total_frames
                progress_callback(progress, f"智能采样: {extracted_count}/{max_frames}")

        prev_frame = frame.copy() if prev_frame is not None or frame_count % 10 == 0 else None
        frame_count += 1

    cap.release()

    return extracted


def resize_frame(frame: np.ndarray, max_size: int = 1920) -> np.ndarray:
    """
    缩放帧到合适大小

    Args:
        frame: 输入帧
        max_size: 最大尺寸

    Returns:
        缩放后的帧
    """
    h, w = frame.shape[:2]

    if max(h, w) <= max_size:
        return frame

    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)

    return cv2.resize(frame, (new_w, new_h))


def get_video_thumbnail(video_path: str, timestamp: float = 5.0,
                       size: Tuple[int, int] = (320, 180)) -> Optional[np.ndarray]:
    """
    获取视频缩略图

    Args:
        video_path: 视频路径
        timestamp: 时间戳位置
        size: 输出尺寸

    Returns:
        缩略图图像
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, size)
        cap.release()
        return frame

    cap.release()
    return None
