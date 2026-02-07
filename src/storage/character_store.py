"""
角色库存储模块
管理角色库的保存和加载
"""
import json
import logging
from pathlib import Path
from typing import List, Optional

from src.models.character import Character, CharacterLibrary
from src.storage.database import Database

logger = logging.getLogger(__name__)


class CharacterStore:
    """角色库存储管理"""

    def __init__(self, library_dir: Path, database: Optional[Database] = None):
        """
        初始化角色库存储

        Args:
            library_dir: 角色库目录
            database: 数据库实例（可选）
        """
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.database = database

    def save_library(self, library: CharacterLibrary, video_id: str) -> str:
        """
        保存角色库到文件

        Args:
            library: 角色库对象
            video_id: 视频ID

        Returns:
            保存的文件路径
        """
        file_path = self.library_dir / f"{video_id}_characters.json"
        library.save(str(file_path))

        # 同时保存到数据库
        if self.database:
            for character in library.characters:
                self.database.save_character(character)

        logger.info(f"角色库已保存: {file_path}")
        return str(file_path)

    def load_library(self, video_id: str) -> Optional[CharacterLibrary]:
        """
        从文件加载角色库

        Args:
            video_id: 视频ID

        Returns:
            角色库对象，如果不存在返回None
        """
        file_path = self.library_dir / f"{video_id}_characters.json"

        if not file_path.exists():
            return None

        library = CharacterLibrary.load(str(file_path))
        logger.info(f"角色库已加载: {file_path}, {library.character_count} 个角色")

        return library

    def load_library_from_db(self, video_id: str) -> Optional[CharacterLibrary]:
        """
        从数据库加载角色库

        Args:
            video_id: 视频ID

        Returns:
            角色库对象
        """
        if not self.database:
            return None

        characters = self.database.get_characters(video_id)

        if not characters:
            return None

        library = CharacterLibrary(
            video_info={'video_id': video_id},
            characters=characters
        )

        logger.info(f"从数据库加载角色库: {library.character_count} 个角色")

        return library

    def export_library(self, library: CharacterLibrary, output_path: str,
                      format: str = 'json'):
        """
        导出角色库

        Args:
            library: 角色库对象
            output_path: 输出路径
            format: 导出格式 (json, csv)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            library.save(str(output_path))

        elif format == 'csv':
            # 导出为CSV格式
            import csv

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # 写入表头
                writer.writerow([
                    '角色ID', '角色名称', '别名', '样本数量',
                    '平均质量', '首次出现', '最后出现'
                ])

                # 写入数据
                for char in library.characters:
                    writer.writerow([
                        char.character_id,
                        char.name,
                        ','.join(char.aliases),
                        char.sample_count,
                        char.statistics.avg_quality,
                        char.statistics.first_appearance,
                        char.statistics.last_appearance,
                    ])

        logger.info(f"角色库已导出: {output_path}")

    def import_library(self, input_path: str) -> Optional[CharacterLibrary]:
        """
        导入角色库

        Args:
            input_path: 输入文件路径

        Returns:
            角色库对象
        """
        input_path = Path(input_path)

        if not input_path.exists():
            logger.error(f"文件不存在: {input_path}")
            return None

        if input_path.suffix == '.json':
            library = CharacterLibrary.load(str(input_path))
            logger.info(f"角色库已导入: {library.character_count} 个角色")
            return library

        else:
            logger.error(f"不支持的文件格式: {input_path.suffix}")
            return None

    def create_library_from_clusters(self, video_id: str,
                                     clusters: List,
                                     samples: List) -> CharacterLibrary:
        """
        从聚类结果创建角色库

        Args:
            video_id: 视频ID
            clusters: 聚类结果列表
            samples: 所有人脸样本

        Returns:
            角色库对象
        """
        from src.models.face_sample import FaceCluster

        library = CharacterLibrary(
            video_info={'video_id': video_id},
            characters=[]
        )

        for cluster in clusters:
            if not isinstance(cluster, FaceCluster):
                continue

            # 创建角色
            character_id = f"char_{cluster.cluster_id:03d}"

            # 获取高质量样本作为原型
            high_quality = cluster.get_high_quality_samples(min_quality=0.7, limit=5)

            character = Character(
                character_id=character_id,
                name=f"角色 {cluster.cluster_id + 1}",
                video_id=video_id,
                recognition_threshold=0.5,
            )

            # 添加原型
            for sample in high_quality:
                if sample.has_embedding and sample.image_path:
                    character.add_prototype(
                        embedding=sample.embedding,
                        image_path=sample.image_path,
                        quality_score=sample.quality_score,
                        timestamp=sample.timestamp,
                    )

            # 添加所有样本
            for sample in cluster.samples:
                character.add_sample(
                    frame_path=sample.image_path or "",
                    bbox=sample.bbox,
                    timestamp=sample.timestamp,
                    quality=sample.quality_score,
                    embedding=sample.embedding if sample.has_embedding else None,
                )

            library.add_character(character)

        return library

    def merge_libraries(self, libraries: List[CharacterLibrary],
                       output_video_id: str) -> CharacterLibrary:
        """
        合并多个角色库

        Args:
            libraries: 角色库列表
            output_video_id: 输出视频ID

        Returns:
            合并后的角色库
        """
        merged = CharacterLibrary(
            video_info={'video_id': output_video_id, 'merged_from': [lib.video_info for lib in libraries]},
            characters=[]
        )

        char_id_counter = 0

        for library in libraries:
            for char in library.characters:
                # 重新分配ID避免冲突
                new_char = Character(
                    character_id=f"char_{char_id_counter:03d}",
                    name=char.name,
                    video_id=output_video_id,
                    prototypes=char.prototypes,
                    samples=char.samples,
                    description=char.description,
                    recognition_threshold=char.recognition_threshold,
                )
                merged.add_character(new_char)
                char_id_counter += 1

        logger.info(f"合并角色库: {len(libraries)} 个库 -> {merged.character_count} 个角色")

        return merged

    def get_character_statistics(self, video_id: str) -> dict:
        """
        获取角色统计信息

        Args:
            video_id: 视频ID

        Returns:
            统计信息字典
        """
        library = self.load_library(video_id)

        if not library:
            library = self.load_library_from_db(video_id)

        if not library:
            return {}

        stats = {
            'total_characters': library.character_count,
            'characters': []
        }

        for char in library.characters:
            char_stats = {
                'id': char.character_id,
                'name': char.name,
                'sample_count': char.sample_count,
                'avg_quality': char.statistics.avg_quality,
                'first_appearance': char.statistics.first_appearance,
                'last_appearance': char.statistics.last_appearance,
                'duration': char.statistics.appearance_duration,
            }
            stats['characters'].append(char_stats)

        return stats
