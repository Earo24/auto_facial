"""
聚类引擎
使用多阶段策略对人脸进行聚类，自动发现角色
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import logging

from config.settings import CLUSTERING_CONFIG
from src.models.face_sample import FaceSample, FaceCluster

logger = logging.getLogger(__name__)


class ClusterEngine:
    """聚类引擎 - 多阶段人脸聚类"""

    def __init__(self):
        """初始化聚类引擎"""
        self.config = CLUSTERING_CONFIG
        self.clusters: List[FaceCluster] = []

    def compute_similarity_matrix(self, samples: List[FaceSample]) -> np.ndarray:
        """
        计算样本间的相似度矩阵

        Args:
            samples: 人脸样本列表

        Returns:
            相似度矩阵
        """
        # 过滤有特征的样本
        valid_samples = [s for s in samples if s.has_embedding]

        if not valid_samples:
            return np.array([])

        embeddings = np.array([s.embedding for s in valid_samples])

        # 计算余弦相似度
        similarity_matrix = cosine_similarity(embeddings)

        return similarity_matrix

    def compute_distance_matrix(self, samples: List[FaceSample]) -> np.ndarray:
        """
        计算样本间的距离矩阵

        Args:
            samples: 人脸样本列表

        Returns:
            距离矩阵
        """
        # 过滤有特征的样本
        valid_samples = [s for s in samples if s.has_embedding]

        if not valid_samples:
            return np.array([])

        embeddings = np.array([s.embedding for s in valid_samples])

        # 计算欧氏距离
        from sklearn.metrics.pairwise import euclidean_distances
        distance_matrix = euclidean_distances(embeddings)

        return distance_matrix

    def phase1_dbscan(self, samples: List[FaceSample]) -> List[FaceCluster]:
        """
        第一阶段：DBSCAN粗聚类

        Args:
            samples: 人脸样本列表

        Returns:
            粗聚类结果
        """
        logger.info("开始第一阶段：DBSCAN粗聚类")

        # 过滤有特征的样本
        valid_samples = [s for s in samples if s.has_embedding]

        if len(valid_samples) < self.config['min_samples']:
            logger.warning(f"样本数量不足 ({len(valid_samples)} < {self.config['min_samples']})")
            return []

        embeddings = np.array([s.embedding for s in valid_samples])

        # DBSCAN聚类
        # 使用余弦距离的近似：将向量归一化后使用欧氏距离
        from sklearn.preprocessing import normalize
        normalized_embeddings = normalize(embeddings)

        dbscan = DBSCAN(
            eps=self.config['eps'],
            min_samples=self.config['min_samples'],
            metric='euclidean',
            n_jobs=-1
        )

        labels = dbscan.fit_predict(normalized_embeddings)

        # 创建簇
        clusters_dict: Dict[int, List[FaceSample]] = defaultdict(list)
        for sample, label in zip(valid_samples, labels):
            if label >= 0:  # 忽略噪声点
                clusters_dict[label].append(sample)

        # 转换为FaceCluster对象
        clusters = []
        for label, cluster_samples in clusters_dict.items():
            cluster = FaceCluster(cluster_id=label)
            for sample in cluster_samples:
                cluster.add_sample(sample)
            clusters.append(cluster)

        logger.info(f"DBSCAN聚类完成: 发现 {len(clusters)} 个簇")

        return clusters

    def phase2_refinement(self, clusters: List[FaceCluster],
                         samples: List[FaceSample]) -> List[FaceCluster]:
        """
        第二阶段：层次聚类细化

        Args:
            clusters: 第一阶段的簇
            samples: 所有样本（包括未分配的）

        Returns:
            细化后的簇
        """
        logger.info("开始第二阶段：层次聚类细化")

        if not clusters:
            return []

        refined_clusters = []

        # 对每个簇进行层次聚类细化
        for cluster in clusters:
            if cluster.size <= 3:
                # 小簇不需要细化
                refined_clusters.append(cluster)
                continue

            embeddings = np.array([s.embedding for s in cluster.samples])

            # 层次聚类
            n_clusters = max(2, cluster.size // 10)  # 动态确定子簇数量

            agg = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='average',
                metric='cosine'
            )

            labels = agg.fit_predict(embeddings)

            # 创建子簇
            sub_clusters_dict: Dict[int, List[FaceSample]] = defaultdict(list)
            for sample, label in zip(cluster.samples, labels):
                sub_clusters_dict[label].append(sample)

            # 如果只有一个子簇，保留原簇
            if len(sub_clusters_dict) == 1:
                refined_clusters.append(cluster)
            else:
                # 创建新的簇对象
                for i, sub_samples in sub_clusters_dict.items():
                    new_cluster = FaceCluster(cluster_id=len(refined_clusters) + len(sub_clusters_dict))
                    for sample in sub_samples:
                        new_cluster.add_sample(sample)
                    refined_clusters.append(new_cluster)

        logger.info(f"层次聚类细化完成: {len(refined_clusters)} 个簇")

        return refined_clusters

    def phase3_merge_small_clusters(self, clusters: List[FaceCluster],
                                    merge_threshold: Optional[float] = None) -> List[FaceCluster]:
        """
        第三阶段：合并小簇

        Args:
            clusters: 当前的簇列表
            merge_threshold: 合并相似度阈值

        Returns:
            合并后的簇列表
        """
        logger.info("开始第三阶段：合并小簇")

        if merge_threshold is None:
            merge_threshold = self.config['merge_threshold']

        # 按大小排序
        sorted_clusters = sorted(clusters, key=lambda c: c.size, reverse=True)

        # 找出小簇
        avg_size = sum(c.size for c in sorted_clusters) / len(sorted_clusters) if sorted_clusters else 0
        small_clusters = [c for c in sorted_clusters if c.size < avg_size * 0.5]
        large_clusters = [c for c in sorted_clusters if c.size >= avg_size * 0.5]

        merged_clusters = large_clusters.copy()
        removed_clusters = []  # 用列表代替集合，避免 FaceCluster 哈希问题

        for small_cluster in small_clusters:
            if small_cluster in removed_clusters:
                continue

            # 计算与大簇的相似度
            best_match = None
            best_similarity = merge_threshold

            small_embedding = small_cluster.representative_embedding

            if small_embedding is None:
                continue

            for large_cluster in large_clusters:
                if large_cluster in removed_clusters:
                    continue

                large_embedding = large_cluster.representative_embedding
                if large_embedding is None:
                    continue

                # 计算余弦相似度
                similarity = np.dot(small_embedding, large_embedding) / (
                    np.linalg.norm(small_embedding) * np.linalg.norm(large_embedding)
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = large_cluster

            # 合并到最相似的大簇
            if best_match:
                best_match.merge(small_cluster)
                removed_clusters.append(small_cluster)
            else:
                # 没有合适的合并对象，保留原簇
                merged_clusters.append(small_cluster)

        # 重新分配cluster_id
        for i, cluster in enumerate(merged_clusters):
            cluster.cluster_id = i

        logger.info(f"小簇合并完成: {len(merged_clusters)} 个簇")

        return merged_clusters

    def discover_characters(self, samples: List[FaceSample],
                           min_cluster_size: int = 5) -> List[FaceCluster]:
        """
        自动发现角色（完整三阶段聚类）

        Args:
            samples: 人脸样本列表
            min_cluster_size: 最小簇大小

        Returns:
            发现的角色簇列表
        """
        logger.info("开始自动角色发现")

        # 第一阶段：DBSCAN粗聚类
        clusters = self.phase1_dbscan(samples)

        if not clusters:
            logger.warning("未发现任何簇")
            return []

        # 第二阶段：层次聚类细化
        clusters = self.phase2_refinement(clusters, samples)

        # 第三阶段：合并小簇
        clusters = self.phase3_merge_small_clusters(clusters)

        # 过滤小簇
        valid_clusters = [c for c in clusters if c.size >= min_cluster_size]

        # 重新分配cluster_id
        for i, cluster in enumerate(valid_clusters):
            cluster.cluster_id = i

        self.clusters = valid_clusters

        logger.info(f"角色发现完成: {len(valid_clusters)} 个角色")

        return valid_clusters

    def assign_remaining_samples(self, clusters: List[FaceCluster],
                                 samples: List[FaceSample],
                                 threshold: float = 0.6) -> List[FaceCluster]:
        """
        将未分配的样本分配到最近的簇

        Args:
            clusters: 已有的簇
            samples: 未分配的样本
            threshold: 相似度阈值

        Returns:
            更新后的簇列表
        """
        unassigned = [s for s in samples if s.cluster_id is None and s.has_embedding]

        if not unassigned:
            return clusters

        logger.info(f"分配 {len(unassigned)} 个未分配样本")

        for sample in unassigned:
            best_cluster = None
            best_similarity = threshold

            for cluster in clusters:
                rep_embedding = cluster.representative_embedding
                if rep_embedding is None:
                    continue

                similarity = sample.compute_similarity(FaceSample(
                    sample_id="", frame_id="", video_id="",
                    bbox=[], embedding=rep_embedding
                ))

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster

            if best_cluster:
                best_cluster.add_sample(sample)

        return clusters

    def merge_clusters(self, cluster_id1: int, cluster_id2: int) -> bool:
        """
        合并两个簇

        Args:
            cluster_id1: 第一个簇ID
            cluster_id2: 第二个簇ID

        Returns:
            是否成功
        """
        cluster1 = next((c for c in self.clusters if c.cluster_id == cluster_id1), None)
        cluster2 = next((c for c in self.clusters if c.cluster_id == cluster_id2), None)

        if cluster1 is None or cluster2 is None:
            return False

        cluster1.merge(cluster2)
        self.clusters.remove(cluster2)

        return True

    def split_cluster(self, cluster_id: int,
                     sample_indices: List[int]) -> Optional[FaceCluster]:
        """
        拆分簇

        Args:
            cluster_id: 簇ID
            sample_indices: 要拆分出去的样本索引

        Returns:
            新创建的簇
        """
        cluster = next((c for c in self.clusters if c.cluster_id == cluster_id), None)

        if cluster is None:
            return None

        new_cluster = cluster.split(sample_indices)
        new_cluster.cluster_id = len(self.clusters)
        self.clusters.append(new_cluster)

        return new_cluster

    def get_cluster_summary(self) -> List[Dict]:
        """
        获取聚类摘要

        Returns:
            簇摘要列表
        """
        summary = []

        for cluster in self.clusters:
            summary.append({
                'cluster_id': cluster.cluster_id,
                'size': cluster.size,
                'avg_quality': cluster.avg_quality,
                'time_range': (
                    min(s.timestamp for s in cluster.samples),
                    max(s.timestamp for s in cluster.samples)
                ) if cluster.samples else (0, 0),
            })

        return summary
