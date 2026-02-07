#!/usr/bin/env python3
"""
将聚类的人脸簇与演员表进行自动匹配
"""
import sys
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.storage.database import Database
from src.core.face_embedder import FaceEmbedder
from config.settings import DATA_ROOT, MODELS_ROOT

API_BASE = "http://localhost:8000/api"
SERIES_ID = "series_2893"
VIDEO_ID = "太平年_1"

# 相似度阈值
SIMILARITY_THRESHOLD = 0.45


def get_series_actors_with_embeddings(db: Database, series_id: str) -> List[Dict]:
    """获取剧集的所有演员（带embedding）"""
    actors = db.get_series_actors(series_id)

    for actor in actors:
        # 获取演员的embedding
        embedding = db.get_actor_embedding(actor['actor_id'])
        if embedding is not None:
            actor['embedding'] = embedding
        else:
            actor['embedding'] = None
            print(f"警告: 演员 {actor['actor_name']} 没有embedding数据")

    return actors


def get_cluster_samples(db: Database, video_id: str) -> Dict[int, List[Dict]]:
    """获取所有簇的样本"""
    samples = db.get_face_samples(video_id)

    # 按cluster_id分组
    clusters = {}
    for sample in samples:
        if sample.cluster_id is not None and sample.embedding is not None:
            if sample.cluster_id not in clusters:
                clusters[sample.cluster_id] = []
            clusters[sample.cluster_id].append(sample)

    return clusters


def compute_cluster_embedding(samples: List) -> np.ndarray:
    """计算簇的平均embedding（只使用高质量样本）"""
    # 按质量排序，取前50%
    sorted_samples = sorted(samples, key=lambda s: s.quality_score, reverse=True)
    top_samples = sorted_samples[:max(5, len(sorted_samples) // 2)]

    embeddings = [s.embedding for s in top_samples if s.embedding is not None]

    if not embeddings:
        return None

    # 计算平均embedding
    return np.mean(embeddings, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度"""
    if a is None or b is None:
        return 0.0

    a = np.array(a).flatten()
    b = np.array(b).flatten()

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def match_cluster_to_actors(cluster_embedding: np.ndarray,
                            actors: List[Dict],
                            min_similarity: float = SIMILARITY_THRESHOLD) -> List[Tuple]:
    """将簇匹配到演员"""
    matches = []

    for actor in actors:
        if actor['embedding'] is None:
            continue

        sim = cosine_similarity(cluster_embedding, actor['embedding'])

        if sim >= min_similarity:
            matches.append((actor, sim))

    # 按相似度降序排序
    matches.sort(key=lambda x: x[1], reverse=True)

    return matches


def main():
    print("=" * 60)
    print("簇与演员匹配程序")
    print("=" * 60)

    db = Database()

    # 1. 获取演员列表
    print(f"\n1. 获取剧集 {SERIES_ID} 的演员列表...")
    actors = get_series_actors_with_embeddings(db, SERIES_ID)

    valid_actors = [a for a in actors if a['embedding'] is not None]
    print(f"   共 {len(actors)} 位演员，其中 {len(valid_actors)} 位有人脸特征")

    # 2. 获取聚类结果
    print(f"\n2. 获取视频 {VIDEO_ID} 的聚类结果...")
    clusters = get_cluster_samples(db, VIDEO_ID)

    if not clusters:
        print("   错误: 没有找到聚类结果！")
        print("   请先运行聚类: POST /api/videos/{video_id}/cluster")
        return

    print(f"   共 {len(clusters)} 个簇")

    # 3. 计算每个簇的embedding并匹配演员
    print(f"\n3. 匹配簇与演员...")

    results = []
    matched_actor_ids = set()

    for cluster_id in sorted(clusters.keys()):
        samples = clusters[cluster_id]
        cluster_embedding = compute_cluster_embedding(samples)

        if cluster_embedding is None:
            continue

        # 匹配演员
        matches = match_cluster_to_actors(cluster_embedding, valid_actors)

        # 找到最佳且未使用的匹配
        best_match = None
        for actor, sim in matches:
            if actor['actor_id'] not in matched_actor_ids:
                best_match = (actor, sim)
                matched_actor_ids.add(actor['actor_id'])
                break

        if best_match:
            actor, sim = best_match
            results.append({
                'cluster_id': cluster_id,
                'sample_count': len(samples),
                'actor_id': actor['actor_id'],
                'actor_name': actor['actor_name'],
                'character_name': actor['character_name'],
                'similarity': sim,
                'quality': np.mean([s.quality_score for s in samples])
            })
            print(f"   簇 {cluster_id:2d} ({len(samples):3d} 样本, 质量{np.mean([s.quality_score for s in samples]):.2f}) -> "
                  f"{actor['actor_name']} 饰 {actor['character_name']} (相似度: {sim:.3f})")
        else:
            results.append({
                'cluster_id': cluster_id,
                'sample_count': len(samples),
                'actor_id': None,
                'actor_name': None,
                'character_name': None,
                'similarity': 0,
                'quality': np.mean([s.quality_score for s in samples])
            })
            print(f"   簇 {cluster_id:2d} ({len(samples):3d} 样本) -> 未匹配")

    # 4. 保存匹配结果到数据库
    print(f"\n4. 保存匹配结果...")
    embedder = FaceEmbedder(model_path=str(MODELS_ROOT), use_gpu=True)

    for result in results:
        if result['actor_id'] is None:
            continue

        cluster_id = result['cluster_id']
        actor_id = result['actor_id']
        character_name = result['character_name']

        # 获取该簇的所有样本
        cluster_samples = clusters[cluster_id]

        # 创建character_id（使用actor_id作为character_id）
        character_id = actor_id

        # 更新每个样本的character_id
        updated_count = 0
        for sample in cluster_samples:
            db.update_sample_character(sample.sample_id, character_id)
            updated_count += 1

        print(f"   簇 {cluster_id} -> {character_id} ({result['actor_name']} 饰 {character_name}): "
              f"更新了 {updated_count} 个样本")

    # 5. 汇总报告
    print("\n" + "=" * 60)
    print("匹配结果汇总")
    print("=" * 60)

    matched_clusters = [r for r in results if r['actor_id'] is not None]
    unmatched_clusters = [r for r in results if r['actor_id'] is None]

    print(f"\n成功匹配: {len(matched_clusters)} 个簇")
    print(f"未匹配:   {len(unmatched_clusters)} 个簇")

    if matched_clusters:
        print(f"\n已匹配的角色:")
        for r in sorted(matched_clusters, key=lambda x: x['similarity'], reverse=True):
            print(f"  - {r['actor_name']} 饰 {r['character_name']} "
                  f"(簇{r['cluster_id']}, {r['sample_count']}样本, 相似度:{r['similarity']:.3f})")

    if unmatched_clusters:
        print(f"\n未匹配的簇:")
        for r in unmatched_clusters:
            print(f"  - 簇{r['cluster_id']} ({r['sample_count']}样本, 平均质量:{r['quality']:.2f})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
