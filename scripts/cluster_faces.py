#!/usr/bin/env python3
"""
人脸聚类脚本
对检测到的人脸进行聚类，自动发现角色
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.cluster_engine import ClusterEngine
from src.storage.database import Database


def cluster_faces(video_id: str, min_cluster_size: int = 5):
    """
    聚类人脸

    Args:
        video_id: 视频ID
        min_cluster_size: 最小簇大小
    """
    print(f"开始聚类: {video_id}")

    database = Database()

    # 加载人脸样本
    print("加载人脸样本...")
    samples = database.get_face_samples(video_id)
    print(f"加载了 {len(samples)} 个人脸样本")

    if not samples:
        print("错误: 没有找到人脸样本")
        return

    # 过滤有特征的样本
    valid_samples = [s for s in samples if s.has_embedding]
    print(f"有效样本数: {len(valid_samples)}")

    if not valid_samples:
        print("错误: 没有包含特征向量的样本")
        return

    # 聚类
    print("开始聚类...")
    engine = ClusterEngine()
    clusters = engine.discover_characters(valid_samples, min_cluster_size=min_cluster_size)

    print(f"\n聚类完成: 发现 {len(clusters)} 个簇")

    # 打印聚类摘要
    print("\n聚类摘要:")
    for cluster in clusters:
        time_range = (
            min(s.timestamp for s in cluster.samples),
            max(s.timestamp for s in cluster.samples)
        )
        print(f"  簇 {cluster.cluster_id}: {cluster.size} 样本, "
              f"平均质量 {cluster.avg_quality:.3f}, "
              f"时间 {time_range[0]:.1f}s - {time_range[1]:.1f}s")

    # 保存聚类结果到数据库
    print("\n保存聚类结果...")
    for sample in valid_samples:
        if sample.cluster_id is not None:
            database.update_sample_cluster(sample.sample_id, sample.cluster_id)

    print("聚类完成!")

    return clusters


def main():
    parser = argparse.ArgumentParser(description="人脸聚类")
    parser.add_argument("video_id", help="视频ID")
    parser.add_argument("--min-size", type=int, default=5, help="最小簇大小")

    args = parser.parse_args()

    cluster_faces(args.video_id, args.min_size)


if __name__ == "__main__":
    main()
