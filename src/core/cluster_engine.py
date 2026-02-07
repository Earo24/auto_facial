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
        logger.info(f"聚类引擎初始化配置: eps={self.config['eps']}, min_samples={self.config['min_samples']}, merge_threshold={self.config['merge_threshold']}")

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

        # 找出小簇（包括单样本簇）
        avg_size = sum(c.size for c in sorted_clusters) / len(sorted_clusters) if sorted_clusters else 0
        small_clusters = [c for c in sorted_clusters if c.size < avg_size * 0.5]
        large_clusters = [c for c in sorted_clusters if c.size >= avg_size * 0.5]

        merged_clusters = large_clusters.copy()
        removed_clusters = []  # 用列表代替集合，避免 FaceCluster 哈希问题

        for small_cluster in small_clusters:
            if small_cluster in removed_clusters:
                continue

            # 计算与所有大簇的相似度
            best_match = None
            # 对于单样本簇，使用更宽松的阈值
            # 使用统一的相似度阈值，不再根据簇大小调整
            best_similarity = merge_threshold

            # 获取小簇的时间范围
            small_start = min(s.timestamp for s in small_cluster.samples)
            small_end = max(s.timestamp for s in small_cluster.samples)

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

                # 不再使用时间重叠度提升相似度，避免错误合并
                # 相似度完全由人脸embedding决定

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = large_cluster

            # 合并到最相似的大簇
            if best_match:
                # 计算时间关系
                large_start = min(s.timestamp for s in best_match.samples)
                large_end = max(s.timestamp for s in best_match.samples)

                # 判断时间关系类型
                is_contained = (small_start >= large_start and small_end <= large_end)  # 小在大内
                is_overlapping = (small_end > large_start and small_start < large_end)  # 有重叠
                is_adjacent = (abs(small_end - large_start) <= 60 or abs(large_end - small_start) <= 60)  # 相邻（60秒内）

                combined_start = min(small_start, large_start)
                combined_end = max(small_end, large_end)
                combined_span = combined_end - combined_start

                # 影视剧友好的时间跨度保护（平衡召回率和精确率）
                skip_merge = False
                if combined_span > 1200:  # 超过20分钟需要检查
                    # 如果时间不重叠也不相邻，禁止合并
                    if not (is_overlapping or is_adjacent):
                        skip_merge = True
                    elif combined_span > 1800:  # 超过30分钟基本不允许
                        skip_merge = True

                if not skip_merge:
                    best_match.merge(small_cluster)
                    removed_clusters.append(small_cluster)
                else:
                    merged_clusters.append(small_cluster)
            else:
                # 没有合适的合并对象，保留原簇
                merged_clusters.append(small_cluster)

        # 重新分配cluster_id（同时更新簇和样本的cluster_id）
        for i, cluster in enumerate(merged_clusters):
            cluster.cluster_id = i
            for sample in cluster.samples:
                sample.cluster_id = i

        logger.info(f"小簇合并完成: {len(merged_clusters)} 个簇")

        # 第四阶段：合并剩余的相似簇（处理时间重叠/相邻的中等大小簇）
        merged_clusters = self.phase4_merge_similar_clusters(merged_clusters, merge_threshold)

        # 第五阶段：拆分有大时间间隔的簇
        merged_clusters = self.phase5_split_large_gap_clusters(merged_clusters)

        return merged_clusters

    def phase5_split_large_gap_clusters(self, clusters: List[FaceCluster]) -> List[FaceCluster]:
        """
        第五阶段：拆分有大时间间隔的簇
        如果一个簇内样本之间有大的时间间隔（>600秒），则拆分成多个簇

        Args:
            clusters: 当前的簇列表

        Returns:
            拆分后的簇列表
        """
        logger.info("开始第五阶段：拆分大时间间隔簇")

        # 先记录所有簇的时间跨度
        for cluster in clusters:
            if cluster.size > 1:
                times = sorted([s.timestamp for s in cluster.samples])
                span = times[-1] - times[0]
                # 检查是否有大间隔
                max_gap = 0
                for i in range(1, len(times)):
                    gap = times[i] - times[i-1]
                    if gap > max_gap:
                        max_gap = gap
                if max_gap > 1200 or span > 2000:  # 放宽时间跨度检查
                    logger.info(f"  簇{cluster.cluster_id}: {cluster.size}样本, 时间跨度{span:.1f}秒, 最大间隔{max_gap:.1f}秒")

        result_clusters = []

        for cluster in clusters:
            if cluster.size <= 1:
                result_clusters.append(cluster)
                continue

            # 按时间排序样本
            sorted_samples = sorted(cluster.samples, key=lambda s: s.timestamp)

            # 检查时间间隔并分组
            sub_groups = []
            current_group = [sorted_samples[0]]

            for i in range(1, len(sorted_samples)):
                prev_time = current_group[-1].timestamp
                curr_time = sorted_samples[i].timestamp
                gap = curr_time - prev_time

                if gap > 600:  # 间隔超过10分钟才拆分（影视剧角色可能长时间缺席后重现）
                    logger.info(f"  在时间 {prev_time:.1f}s 和 {curr_time:.1f}s 之间拆分 (间隔{gap:.1f}秒)")
                    sub_groups.append(current_group)
                    current_group = [sorted_samples[i]]
                else:
                    current_group.append(sorted_samples[i])

            # 添加最后一组
            if current_group:
                sub_groups.append(current_group)

            if len(sub_groups) > 1:
                # 有拆分，创建新簇
                logger.info(f"  簇 {cluster.cluster_id} 被拆分为 {len(sub_groups)} 个子簇")
                for group in sub_groups:
                    new_cluster = FaceCluster(cluster.cluster_id)
                    for sample in group:
                        new_cluster.add_sample(sample)
                    result_clusters.append(new_cluster)
            else:
                # 没有拆分，保留原簇
                result_clusters.append(cluster)

        # 重新分配cluster_id（同时更新簇和样本的cluster_id）
        for i, cluster in enumerate(result_clusters):
            cluster.cluster_id = i
            for sample in cluster.samples:
                sample.cluster_id = i

        logger.info(f"大间隔拆分完成: {len(result_clusters)} 个簇")

        return result_clusters

    def phase4_merge_similar_clusters(self, clusters: List[FaceCluster],
                                       merge_threshold: float) -> List[FaceCluster]:
        """
        第四阶段：合并剩余的相似簇
        处理时间重叠或相邻的簇之间的合并

        Args:
            clusters: 当前的簇列表
            merge_threshold: 合并相似度阈值

        Returns:
            合并后的簇列表
        """
        logger.info("开始第四阶段：合并相似簇")

        if not clusters:
            return clusters

        removed_clusters = []
        result_clusters = []

        for i, cluster1 in enumerate(clusters):
            if cluster1 in removed_clusters:
                continue

            # 尝试与其他簇合并
            merged = False
            c1_start = min(s.timestamp for s in cluster1.samples)
            c1_end = max(s.timestamp for s in cluster1.samples)
            c1_embedding = cluster1.representative_embedding

            if c1_embedding is None:
                result_clusters.append(cluster1)
                continue

            for j, cluster2 in enumerate(clusters):
                if j <= i or cluster2 in removed_clusters:
                    continue

                c2_start = min(s.timestamp for s in cluster2.samples)
                c2_end = max(s.timestamp for s in cluster2.samples)
                c2_embedding = cluster2.representative_embedding

                if c2_embedding is None:
                    continue

                # 计算相似度
                similarity = np.dot(c1_embedding, c2_embedding) / (
                    np.linalg.norm(c1_embedding) * np.linalg.norm(c2_embedding)
                )

                # 判断时间关系
                is_overlapping = (c1_end > c2_start and c1_start < c2_end)
                is_adjacent = (abs(c1_end - c2_start) <= 60 or abs(c2_end - c1_start) <= 60)

                # 计算合并后的时间跨度
                combined_start = min(c1_start, c2_start)
                combined_end = max(c1_end, c2_end)
                combined_span = combined_end - combined_start

                # 决定是否合并
                should_merge = False

                # 平衡的合并条件
                min_similarity = merge_threshold  # 使用配置的阈值

                if is_overlapping or is_adjacent:
                    # 时间重叠或相邻，使用基础相似度要求
                    if similarity > min_similarity and combined_span < 1200:
                        should_merge = True
                else:
                    # 时间不相交不相邻，要求更高相似度
                    if similarity > min_similarity + 0.04 and combined_span < 600:
                        should_merge = True

                if should_merge:
                    # 合并cluster2到cluster1
                    cluster1.merge(cluster2)
                    removed_clusters.append(cluster2)
                    merged = True

            if not merged:
                result_clusters.append(cluster1)
            else:
                result_clusters.append(cluster1)

        # 重新分配cluster_id（同时更新簇和样本的cluster_id）
        for i, cluster in enumerate(result_clusters):
            cluster.cluster_id = i
            for sample in cluster.samples:
                sample.cluster_id = i

        logger.info(f"相似簇合并完成: {len(result_clusters)} 个簇")

        return result_clusters

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

        # 先重置所有样本的cluster_id为None
        for sample in samples:
            sample.cluster_id = None

        # 重新分配cluster_id（只更新保留簇中的样本）
        for i, cluster in enumerate(valid_clusters):
            cluster.cluster_id = i
            for sample in cluster.samples:
                sample.cluster_id = i

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
