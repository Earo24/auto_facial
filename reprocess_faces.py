"""
快速重新处理人脸检测（使用动态阈值）
"""
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.face_detector import FaceDetector
from src.core.face_embedder import FaceEmbedder
from src.storage.database import Database

# 配置
video_id = "太平年_1"
frames_dir = Path("data/processed/frames") / video_id
faces_dir = Path("data/processed/faces") / video_id

# 视频分辨率 1920x804，最小人脸尺寸 = 804 * 0.15 = 120px
min_face_size = 120

print(f"=== 重新处理人脸检测 ===")
print(f"视频分辨率: 1920x804")
print(f"动态最小人脸尺寸: {min_face_size}px")
print(f"已有帧数: {len(list(frames_dir.glob('*.jpg')))}")

# 初始化检测器
print("\n初始化检测器...")
detector = FaceDetector(use_gpu=True)
detector.initialize()
detector.min_face_size = min_face_size

embedder = FaceEmbedder(use_gpu=True)
embedder.initialize()

# 清理旧数据
if faces_dir.exists():
    import shutil
    shutil.rmtree(faces_dir)
faces_dir.mkdir(parents=True, exist_ok=True)

# 获取所有帧
frame_files = sorted(frames_dir.glob("*.jpg"))
print(f"开始处理 {len(frame_files)} 帧...\n")

all_samples = []
processed = 0

# 处理每一帧
for i, frame_path in enumerate(frame_files):
    # 读取帧
    frame = cv2.imread(str(frame_path))
    if frame is None:
        continue

    # 提取帧号
    frame_name = frame_path.stem
    parts = frame_name.split('_')
    if parts[0] == 'frame':
        frame_number = int(parts[1])
    else:
        continue

    # 检测人脸
    samples = detector.detect_faces(
        frame,
        frame_id=frame_name,
        video_id=video_id,
        frame_number=frame_number,
        timestamp=float(frame_number)
    )

    # 提取embedding
    for sample in samples:
        if sample.embedding is None:
            face_region = frame[
                int(sample.bbox[1]):int(sample.bbox[3]),
                int(sample.bbox[0]):int(sample.bbox[2])
            ]
            sample.embedding = embedder.compute_embedding(
                face_region,
                face_bbox=sample.bbox,
                face_landmarks=sample.landmarks
            )

        # 保存人脸图像
        face_filename = f"{frame_name}_face_{len(all_samples)}.jpg"
        face_path = faces_dir / face_filename

        face_region = frame[
            int(sample.bbox[1]):int(sample.bbox[3]),
            int(sample.bbox[0]):int(sample.bbox[2])
        ]
        cv2.imwrite(str(face_path), face_region)
        sample.image_path = str(face_path)

        all_samples.append(sample)

    processed += 1
    all_samples.extend(samples)

    # 每500帧输出一次进度
    if processed % 500 == 0:
        print(f"[进度] 已处理 {processed}/{len(frame_files)} 帧，检测到 {len(all_samples)} 张人脸")

# 保存到数据库
print(f"\n=== 处理完成 ===")
print(f"处理帧数: {processed}")
print(f"检测人脸: {len(all_samples)}")

if all_samples:
    db = Database()
    db.save_face_samples(all_samples)
    print("数据已保存到数据库")
else:
    print("警告: 未检测到任何人脸")
