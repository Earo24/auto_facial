"""
角色匹配器
通过本地演员照片库匹配聚类结果中的角色
"""
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ActorInfo:
    """演员信息"""
    actor_name: str      # 演员名
    character_name: str  # 角色名
    photo_path: str      # 照片路径
    embedding: Optional[np.ndarray] = None  # 人脸embedding


class CharacterMatcher:
    """角色匹配器 - 将聚类结果与演员照片库匹配"""

    def __init__(self, actors_dir: str = "data/actors"):
        """
        初始化角色匹配器

        Args:
            actors_dir: 演员照片目录，结构应为:
                        data/actors/
                        ├── 演员名_角色名/
                        │   ├── photo1.jpg
                        │   ├── photo2.jpg
                        │   └── ...
        """
        self.actors_dir = Path(actors_dir)
        self.actors: List[ActorInfo] = []
        self.face_analysis = None

        # 创建演员目录
        self.actors_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"角色匹配器初始化，演员目录: {self.actors_dir}")

    def init_face_model(self):
        """初始化人脸模型（延迟加载）"""
        if self.face_analysis is not None:
            return

        if not INSIGHTFACE_AVAILABLE:
            logger.warning("InsightFace未安装，无法进行人脸识别")
            return

        try:
            # 使用与主系统相同的配置
            self.face_analysis = FaceAnalysis(
                providers=['CUDAPath', 'CPUExecutionProvider']
            )
            self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("人脸模型初始化成功")
        except Exception as e:
            logger.error(f"人脸模型初始化失败: {e}")
            self.face_analysis = None

    def load_actors_from_directory(self) -> List[ActorInfo]:
        """
        从目录加载演员信息

        目录结构:
        data/actors/
        ├── 张三_李四/
        │   ├── photo1.jpg
        │   ├── photo2.jpg
        └── 王五_赵六/
            ├── photo1.jpg

        文件夹名格式: 演员名_角色名
        """
        self.actors = []

        if not self.actors_dir.exists():
            logger.warning(f"演员目录不存在: {self.actors_dir}")
            return self.actors

        # 遍历演员文件夹
        for actor_folder in self.actors_dir.iterdir():
            if not actor_folder.is_dir():
                continue

            # 解析文件夹名: "演员名_角色名"
            folder_name = actor_folder.name
            if '_' not in folder_name:
                logger.warning(f"跳过格式错误的文件夹: {folder_name}")
                continue

            parts = folder_name.split('_', 1)
            actor_name = parts[0].strip()
            character_name = parts[1].strip() if len(parts) > 1 else actor_name

            # 获取照片文件
            photo_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                photo_files.extend(actor_folder.glob(ext))

            if not photo_files:
                logger.warning(f"演员 {actor_name} 没有找到照片")
                continue

            # 为每张照片创建ActorInfo
            for photo_path in photo_files:
                self.actors.append(ActorInfo(
                    actor_name=actor_name,
                    character_name=character_name,
                    photo_path=str(photo_path)
                ))

        logger.info(f"加载了 {len(self.actors)} 个演员照片")

        return self.actors

    def compute_actor_embeddings(self) -> Dict[str, np.ndarray]:
        """
        计算所有演员照片的人脸embedding

        Returns:
            Dict[photo_path, embedding]
        """
        if not self.actors:
            logger.warning("没有加载演员信息")
            return {}

        self.init_face_model()
        if self.face_analysis is None:
            logger.error("人脸模型未初始化")
            return {}

        embeddings = {}

        for actor_info in self.actors:
            try:
                import cv2
                img = cv2.imread(actor_info.photo_path)
                if img is None:
                    logger.warning(f"无法读取照片: {actor_info.photo_path}")
                    continue

                # 检测人脸
                faces = self.face_analysis.get(img)
                if not faces:
                    logger.warning(f"照片中未检测到人脸: {actor_info.photo_path}")
                    continue

                # 取第一个人脸（假设照片中只有一个主要人物）
                face = faces[0]
                embedding = face.embedding / np.linalg.norm(face.embedding)

                actor_info.embedding = embedding
                embeddings[actor_info.photo_path] = embedding

                logger.debug(f"计算embedding: {actor_info.actor_name} - {actor_info.photo_path}")

            except Exception as e:
                logger.error(f"处理照片失败 {actor_info.photo_path}: {e}")

        logger.info(f"成功计算 {len(embeddings)} 个演员embedding")

        return embeddings

    def match_cluster_to_actor(self, cluster_samples: List,
                              min_similarity: float = 0.45) -> Optional[ActorInfo]:
        """
        将聚类结果匹配到演员

        Args:
            cluster_samples: 簇中的人脸样本列表
            min_similarity: 最小相似度阈值

        Returns:
            匹配到的ActorInfo，如果没有匹配则返回None
        """
        if not self.actors:
            return None

        # 确保演员embedding已计算
        if not hasattr(self, 'actors') or not self.actors:
            self.load_actors_from_directory()

        # 计算演员embedding（如果还没计算）
        if self.actors and self.actors[0].embedding is None:
            self.compute_actor_embeddings()

        # 获取簇的代表embedding（使用质心）
        from src.models.face_sample import FaceSample
        valid_samples = [s for s in cluster_samples if hasattr(s, 'embedding') and s.embedding is not None]

        if not valid_samples:
            return None

        # 计算簇质心
        embeddings = np.array([s.embedding for s in valid_samples])
        centroid = embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        # 计算与所有演员的相似度
        best_match = None
        best_similarity = min_similarity

        for actor_info in self.actors:
            if actor_info.embedding is None:
                continue

            # 计算余弦相似度
            similarity = float(np.dot(centroid, actor_info.embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = actor_info

        if best_match:
            logger.info(f"匹配成功: {best_match.actor_name} ({best_match.character_name}), 相似度: {best_similarity:.3f}")

        return best_match

    def match_all_clusters(self, clusters: List[List]) -> Dict[int, ActorInfo]:
        """
        匹配所有簇到演员

        Args:
            clusters: 簇列表

        Returns:
            Dict[cluster_id, ActorInfo]
        """
        # 加载演员信息
        self.load_actors_from_directory()

        # 计算演员embedding
        self.compute_actor_embeddings()

        results = {}

        for cluster in clusters:
            cluster_id = cluster.get('cluster_id') if isinstance(cluster, dict) else cluster.cluster_id
            samples = cluster.get('samples') if isinstance(cluster, dict) else cluster.samples

            match = self.match_cluster_to_actor(samples)
            if match:
                results[cluster_id] = match

        logger.info(f"匹配完成: {len(results)}/{len(clusters)} 个簇找到对应演员")

        return results

    def export_actor_template(self):
        """导出演员照片目录模板"""
        readme_content = """# 演员照片目录

## 目录结构
```
data/actors/
├── 张三_李四/        # 文件夹名格式: 演员名_角色名
│   ├── photo1.jpg    # 演员的照片
│   ├── photo2.jpg
│   └── ...
└── 王五_赵六/
    ├── photo1.jpg
    └── ...
```

## 使用说明

1. 为每个主要角色创建一个文件夹
2. 文件夹命名格式: `演员名_角色名`
3. 放入该演员的1-5张清晰照片
4. 运行角色匹配即可

## 示例

例如《太平年》的主要角色:
- 刘钧_乾隆
- 李乃文_富察皇后
- 王雷_ ...
- ...

## 照片要求

- 正面清晰照片
- 尽量使用剧照
- 避免模糊、遮挡
- 建议多张不同角度的照片
"""

        readme_path = self.actors_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        logger.info(f"已创建模板说明: {readme_path}")

        return str(readme_path)
