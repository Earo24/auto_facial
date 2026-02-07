"""
视频帧数据模型
存储从视频中提取的帧信息
"""
from dataclasses import dataclass, field
from typing import Optional, List, Any
from pathlib import Path
import numpy as np


@dataclass
class VideoFrame:
    """视频帧数据模型"""

    # 基本信息
    frame_id: str
    video_id: str

    # 帧信息
    frame_number: int  # 帧序号
    timestamp: float  # 时间戳（秒）

    # 图像信息
    image_path: Optional[str] = None  # 帧图像保存路径
    width: int = 0
    height: int = 0

    # 场景信息
    is_scene_change: bool = False  # 是否为场景切换帧
    scene_id: Optional[int] = None  # 场景ID

    # 人脸检测信息
    face_count: int = 0  # 检测到的人脸数量
    face_ids: List[str] = field(default_factory=list)  # 人脸样本ID列表

    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.face_ids, str):
            self.face_ids = []

    @property
    def has_faces(self) -> bool:
        """是否检测到人脸"""
        return self.face_count > 0

    @property
    def resolution(self) -> tuple:
        """获取分辨率"""
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """获取宽高比"""
        if self.height == 0:
            return 0.0
        return self.width / self.height

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'frame_id': self.frame_id,
            'video_id': self.video_id,
            'frame_number': self.frame_number,
            'timestamp': self.timestamp,
            'image_path': self.image_path,
            'width': self.width,
            'height': self.height,
            'is_scene_change': self.is_scene_change,
            'scene_id': self.scene_id,
            'face_count': self.face_count,
            'face_ids': self.face_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'VideoFrame':
        """从字典创建实例"""
        return cls(
            frame_id=data['frame_id'],
            video_id=data['video_id'],
            frame_number=data['frame_number'],
            timestamp=data['timestamp'],
            image_path=data.get('image_path'),
            width=data.get('width', 0),
            height=data.get('height', 0),
            is_scene_change=data.get('is_scene_change', False),
            scene_id=data.get('scene_id'),
            face_count=data.get('face_count', 0),
            face_ids=data.get('face_ids', []),
        )

    def load_image(self) -> Optional[np.ndarray]:
        """加载帧图像"""
        if self.image_path is None or not Path(self.image_path).exists():
            return None

        import cv2
        image = cv2.imread(self.image_path)
        if image is not None:
            self.width = image.shape[1]
            self.height = image.shape[0]
        return image


@dataclass
class VideoInfo:
    """视频元信息"""

    video_id: str
    file_path: str
    filename: str

    # 视频属性
    duration: float = 0.0  # 时长（秒）
    fps: float = 0.0  # 帧率
    total_frames: int = 0  # 总帧数
    width: int = 0  # 视频宽度
    height: int = 0  # 视频高度

    # 处理信息
    processed_frames: int = 0  # 已处理帧数
    detected_faces: int = 0  # 检测到的人脸总数
    characters_found: int = 0  # 发现的角色数

    # 音频信息
    has_audio: bool = False
    audio_channels: int = 0

    @property
    def resolution(self) -> tuple:
        """获取分辨率"""
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """获取宽高比"""
        if self.height == 0:
            return 0.0
        return self.width / self.height

    @property
    def format_resolution(self) -> str:
        """格式化分辨率字符串"""
        return f"{self.width}x{self.height}"

    @property
    def format_duration(self) -> str:
        """格式化时长字符串"""
        hours = int(self.duration // 3600)
        minutes = int((self.duration % 3600) // 60)
        seconds = int(self.duration % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @classmethod
    def from_file(cls, file_path: str) -> 'VideoInfo':
        """从视频文件创建信息"""
        import cv2
        from pathlib import Path

        path = Path(file_path)
        cap = cv2.VideoCapture(file_path)

        info = cls(
            video_id=path.stem,
            file_path=str(path.absolute()),
            filename=path.name,
        )

        if cap.isOpened():
            info.fps = cap.get(cv2.CAP_PROP_FPS)
            info.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info.duration = info.total_frames / info.fps if info.fps > 0 else 0

        cap.release()
        return info

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'video_id': self.video_id,
            'file_path': self.file_path,
            'filename': self.filename,
            'duration': self.duration,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'width': self.width,
            'height': self.height,
            'processed_frames': self.processed_frames,
            'detected_faces': self.detected_faces,
            'characters_found': self.characters_found,
            'has_audio': self.has_audio,
            'audio_channels': self.audio_channels,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'VideoInfo':
        """从字典创建实例"""
        return cls(
            video_id=data['video_id'],
            file_path=data['file_path'],
            filename=data['filename'],
            duration=data.get('duration', 0.0),
            fps=data.get('fps', 0.0),
            total_frames=data.get('total_frames', 0),
            width=data.get('width', 0),
            height=data.get('height', 0),
            processed_frames=data.get('processed_frames', 0),
            detected_faces=data.get('detected_faces', 0),
            characters_found=data.get('characters_found', 0),
            has_audio=data.get('has_audio', False),
            audio_channels=data.get('audio_channels', 0),
        )
