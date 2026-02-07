#!/usr/bin/env python3
"""
人脸识别脚本
基于角色库进行批量人脸识别
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.core.recognition_engine import RecognitionEngine
from src.core.face_embedder import FaceEmbedder
from src.storage.database import Database
from src.storage.character_store import CharacterStore
from config.settings import MODELS_ROOT, CHARACTERS_DIR


def recognize(video_id: str, use_gpu: bool = True, use_temporal_smoothing: bool = True):
    """
    运行人脸识别

    Args:
        video_id: 视频ID
        use_gpu: 是否使用GPU
        use_temporal_smoothing: 是否使用时序平滑
    """
    print(f"开始识别: {video_id}")

    database = Database()

    # 初始化
    print("初始化识别引擎...")
    embedder = FaceEmbedder(model_path=str(MODELS_ROOT), use_gpu=use_gpu)
    embedder.initialize()

    engine = RecognitionEngine(embedder)

    # 加载角色库
    print("加载角色库...")
    character_store = CharacterStore(CHARACTERS_DIR, database)
    library = character_store.load_library(video_id)

    if not library:
        library = character_store.load_library_from_db(video_id)

    if not library or not library.characters:
        print("错误: 未找到角色库，请先完成聚类标注")
        return

    print(f"加载了 {len(library.characters)} 个角色")

    # 加载角色到引擎
    engine.load_characters(library.characters)

    # 加载待识别的样本
    print("加载人脸样本...")
    all_samples = database.get_face_samples(video_id)
    unassigned_samples = [s for s in all_samples if s.character_id is None]

    print(f"待识别样本数: {len(unassigned_samples)}")

    if not unassigned_samples:
        print("所有样本已标注，无需识别")
        return

    # 批量识别
    print("开始识别...")
    results = []

    for sample in tqdm(unassigned_samples, desc="识别中"):
        result = engine.recognize(sample, use_temporal_smoothing=use_temporal_smoothing)
        if result:
            results.append(result)

            # 更新数据库
            if result.character_id:
                from src.storage.database import Database
                db = Database()
                db.update_sample_character(sample.sample_id, result.character_id)

    # 保存识别结果
    print("保存识别结果...")
    database.save_recognition_results(video_id, results)

    # 统计信息
    stats = engine.get_recognition_statistics(results)

    print(f"\n识别完成!")
    print(f"  - 总样本数: {stats.get('total_samples', 0)}")
    print(f"  - 高置信度: {stats.get('high_confidence', 0)}")
    print(f"  - 低置信度: {stats.get('low_confidence', 0)}")
    print(f"  - 未知: {stats.get('unknown', 0)}")

    if 'character_counts' in stats:
        print(f"\n角色识别统计:")
        for char_id, count in stats['character_counts'].items():
            char = next((c for c in library.characters if c.character_id == char_id), None)
            char_name = char.name if char else char_id
            print(f"  - {char_name}: {count}")

    return results


def main():
    parser = argparse.ArgumentParser(description="人脸识别")
    parser.add_argument("video_id", help="视频ID")
    parser.add_argument("--cpu", action="store_true", help="使用CPU而非GPU")
    parser.add_argument("--no-smoothing", action="store_true", help="禁用时序平滑")

    args = parser.parse_args()

    recognize(args.video_id, use_gpu=not args.cpu, use_temporal_smoothing=not args.no_smoothing)


if __name__ == "__main__":
    main()
