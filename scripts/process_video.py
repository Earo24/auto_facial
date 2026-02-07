#!/usr/bin/env python3
"""
视频处理脚本
处理视频文件，提取帧并检测人脸
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.core.video_processor import VideoProcessor
from src.core.face_detector import FaceDetector
from src.core.face_embedder import FaceEmbedder
from src.storage.database import Database
from config.settings import MODELS_ROOT


def process_video(video_path: str, video_id: str = None, use_gpu: bool = True):
    """
    处理视频

    Args:
        video_path: 视频文件路径
        video_id: 视频ID（可选，默认使用文件名）
        use_gpu: 是否使用GPU
    """
    video_path = Path(video_path)

    if not video_path.exists():
        print(f"错误: 视频文件不存在: {video_path}")
        return

    if video_id is None:
        video_id = video_path.stem

    print(f"开始处理视频: {video_path.name}")
    print(f"视频ID: {video_id}")

    # 初始化组件
    print("初始化检测器和特征提取器...")
    detector = FaceDetector(model_path=str(MODELS_ROOT), use_gpu=use_gpu)
    detector.initialize()

    embedder = FaceEmbedder(model_path=str(MODELS_ROOT), use_gpu=use_gpu)
    embedder.initialize()

    processor = VideoProcessor(detector, embedder)
    database = Database()

    # 处理视频
    def progress_callback(progress, message):
        print(f"[{progress*100:.1f}%] {message}")

    print("开始视频处理...")
    result = processor.process_video(str(video_path), video_id, progress_callback=progress_callback)

    # 保存到数据库
    print("保存数据到数据库...")
    database.save_video_info(result['video_info'])
    database.save_frames(result['frames'])
    database.save_face_samples(result['face_samples'])

    print(f"\n处理完成!")
    print(f"  - 提取帧数: {result['video_info'].processed_frames}")
    print(f"  - 检测人脸: {result['video_info'].detected_faces}")
    print(f"  - 去重后: {len(result['face_samples'])}")

    return result


def main():
    parser = argparse.ArgumentParser(description="处理视频并检测人脸")
    parser.add_argument("video_path", help="视频文件路径")
    parser.add_argument("--video-id", help="视频ID（可选）")
    parser.add_argument("--cpu", action="store_true", help="使用CPU而非GPU")

    args = parser.parse_args()

    process_video(args.video_path, args.video_id, use_gpu=not args.cpu)


if __name__ == "__main__":
    main()
