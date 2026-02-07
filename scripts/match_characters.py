#!/usr/bin/env python3
"""
角色匹配脚本
将聚类结果与演员照片库匹配，自动识别角色
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.character_matcher import CharacterMatcher
from src.storage.database import Database


def setup_actors_template():
    """创建演员照片目录模板"""
    print("创建演员照片目录模板...")

    matcher = CharacterMatcher()
    template_path = matcher.export_actor_template()

    print(f"\n模板已创建: {template_path}")
    print("\n下一步:")
    print("1. 在 data/actors/ 目录下为每个角色创建文件夹")
    print("2. 文件夹命名格式: 演员名_角色名")
    print("3. 放入演员的清晰照片")
    print("4. 重新运行本脚本进行匹配")


def match_characters(video_id: str, min_similarity: float = 0.45):
    """
    匹配视频中的角色

    Args:
        video_id: 视频ID
        min_similarity: 最小相似度阈值
    """
    print(f"开始匹配角色: {video_id}")

    # 初始化匹配器
    matcher = CharacterMatcher()

    # 加载演员信息
    print("\n加载演员照片库...")
    actors = matcher.load_actors_from_directory()

    if not actors:
        print("\n错误: 没有找到演员照片")
        print("请先运行: python scripts/match_characters.py --setup")
        return

    print(f"找到 {len(actors)} 个演员照片")

    # 计算演员embedding
    print("\n计算演员人脸特征...")
    matcher.compute_actor_embeddings()

    # 加载聚类结果
    print(f"\n加载聚类结果...")
    database = Database()
    samples = database.get_face_samples(video_id)

    # 按簇分组
    from collections import defaultdict
    clusters_dict = defaultdict(list)
    for s in samples:
        if s.cluster_id is not None:
            clusters_dict[s.cluster_id].append(s)

    print(f"找到 {len(clusters_dict)} 个簇")

    # 匹配每个簇
    print("\n开始匹配...")
    results = {}

    for cluster_id in sorted(clusters_dict.keys()):
        cluster_samples = clusters_dict[cluster_id]
        print(f"\n簇 {cluster_id} ({len(cluster_samples)} 样本):")

        match = matcher.match_cluster_to_actor(cluster_samples, min_similarity)

        if match:
            print(f"  ✅ 匹配到: {match.actor_name} 饰演 {match.character_name}")
            results[cluster_id] = {
                'actor_name': match.actor_name,
                'character_name': match.character_name,
                'photo_path': match.photo_path
            }

            # 更新数据库
            for sample in cluster_samples:
                database.update_sample_character(
                    sample.sample_id,
                    f"{match.actor_name} 饰演 {match.character_name}"
                )
        else:
            print(f"  ❌ 未找到匹配")

    # 总结
    print(f"\n" + "="*50)
    print(f"匹配完成!")
    print(f"成功匹配: {len(results)}/{len(clusters_dict)} 个簇")
    print(f"="*50)

    if results:
        print("\n匹配结果:")
        for cluster_id, info in sorted(results.items()):
            actor = info['actor_name']
            character = info['character_name']
            print(f"  簇{cluster_id}: {actor} 饰演 {character}")


def main():
    parser = argparse.ArgumentParser(description="角色匹配")
    parser.add_argument("--setup", action="store_true", help="创建演员照片目录模板")
    parser.add_argument("--video-id", help="视频ID")
    parser.add_argument("--min-similarity", type=float, default=0.45, help="最小相似度阈值")

    args = parser.parse_args()

    if args.setup:
        setup_actors_template()
    elif args.video_id:
        match_characters(args.video_id, args.min_similarity)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
