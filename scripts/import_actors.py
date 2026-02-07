#!/usr/bin/env python3
"""
批量导入演员照片到太平年剧集
"""
import sys
from pathlib import Path
import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.core.face_detector import FaceDetector
from src.core.face_embedder import FaceEmbedder
from src.storage.database import Database
from config.settings import DATA_ROOT, MODELS_ROOT

# 目标剧集ID
SERIES_ID = "series_2893"
ACTORS_DIR = Path("/Users/easonpeng/code/earo/vibecoding/tmp/演员照片")

# 存储演员照片的目录
ACTOR_PHOTOS_DIR = DATA_ROOT / "actor_photos"
ACTOR_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


def parse_filename(filename):
    """
    解析文件名，提取演员名和角色名
    格式: 演员名_饰角色名.jpg
    """
    name = filename.stem
    if "_饰" in name:
        parts = name.split("_饰")
        actor_name = parts[0]
        character_name = parts[1] if len(parts) > 1 else ""
        # 移除角色名中的空格和下划线（如"钱弘侑___孙本"）
        character_name = character_name.replace("___", "、").replace("__", "、").strip("_")
        return actor_name, character_name
    return None, None


def extract_face_embedding(image_path, detector, embedder):
    """提取照片中的人脸特征"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  无法读取图片: {image_path}")
        return None, None, None

    # 检测人脸
    face_info = detector.detect_single(img)

    if face_info is None:
        print(f"  未检测到人脸: {image_path}")
        return None, None, None

    bbox = face_info['bbox']

    # 提取特征
    embedding = embedder.compute_embedding(img, bbox)

    if embedding is None:
        print(f"  无法提取人脸特征: {image_path}")
        return None, None, None

    # 裁剪人脸区域
    x1, y1, x2, y2 = [int(x) for x in bbox]
    face_img = img[y1:y2, x1:x2]

    print(f"  提取特征成功, 置信度: {face_info['confidence']:.2f}")

    return embedding, face_img, face_info


def main():
    print("初始化模型...")
    detector = FaceDetector(model_path=str(MODELS_ROOT), use_gpu=True)
    detector.initialize()

    embedder = FaceEmbedder(model_path=str(MODELS_ROOT), use_gpu=True)
    embedder.initialize()

    db = Database()

    print(f"\n处理演员照片目录: {ACTORS_DIR}")
    print(f"目标剧集: {SERIES_ID}")
    print("-" * 60)

    success_count = 0
    skip_count = 0
    error_count = 0

    # 获取所有jpg文件
    photo_files = sorted(ACTORS_DIR.glob("*.jpg"))

    for photo_file in photo_files:
        if photo_file.name.startswith("."):
            continue

        actor_name, character_name = parse_filename(photo_file)

        if not actor_name:
            print(f"跳过: {photo_file.name} (无法解析文件名)")
            skip_count += 1
            continue

        print(f"\n处理: {photo_file.name}")
        print(f"  演员: {actor_name}")
        print(f"  角色: {character_name}")

        # 检查演员是否已存在
        existing_actors = db.get_actors()
        actor_id = None

        for actor in existing_actors:
            if actor['name'] == actor_name:
                actor_id = actor['actor_id']
                print(f"  演员已存在: {actor_id}")
                break

        # 提取人脸特征
        embedding, face_img, face_info = extract_face_embedding(photo_file, detector, embedder)

        if embedding is None:
            error_count += 1
            continue

        # 保存人脸照片
        actor_photo_path = None
        if face_img is not None:
            if actor_id is None:
                actor_id = f"actor_{hash(actor_name) % 10000:04d}"

            actor_photo_path = str(ACTOR_PHOTOS_DIR / f"{actor_id}.jpg")
            cv2.imwrite(actor_photo_path, face_img)
            print(f"  保存照片: {actor_photo_path}")

        # 如果演员不存在，创建新演员
        if actor_id and not any(a['actor_id'] == actor_id for a in existing_actors):
            db.create_actor(
                actor_id=actor_id,
                name=actor_name,
                photo_path=actor_photo_path,
                embedding=embedding
            )
            print(f"  创建演员: {actor_id}")
        elif actor_id:
            # 更新已有演员的特征和照片
            db.update_actor_embedding(actor_id, embedding)
            print(f"  更新演员特征: {actor_id}")

        # 添加到剧集
        if actor_id and character_name:
            # 检查是否已经添加
            series_actors = db.get_series_actors(SERIES_ID)
            already_added = any(
                sa['actor_id'] == actor_id and sa['character_name'] == character_name
                for sa in series_actors
            )

            if not already_added:
                db.add_series_actor(
                    series_id=SERIES_ID,
                    actor_id=actor_id,
                    character_name=character_name,
                    role_order=success_count,
                    is_main_character=True
                )
                print(f"  添加到剧集: {character_name}")
                success_count += 1
            else:
                print(f"  已在剧中: {character_name}")
                skip_count += 1

    print("\n" + "=" * 60)
    print(f"处理完成!")
    print(f"  成功添加: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  错误: {error_count}")
    print("=" * 60)

    # 显示最终结果
    print("\n太平年剧集演员列表:")
    actors = db.get_series_actors(SERIES_ID)
    for i, actor in enumerate(actors, 1):
        print(f"  {i}. {actor['actor_name']} 饰 {actor['character_name']}")


if __name__ == "__main__":
    main()
