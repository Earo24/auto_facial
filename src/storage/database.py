"""
数据库操作模块
使用SQLite存储所有数据
"""
import sqlite3
import json
import numpy as np
from typing import List, Dict, Optional, Any
from pathlib import Path
from contextlib import contextmanager
import logging

from config.settings import DATABASE_PATH
from src.models.character import Character, CharacterLibrary
from src.models.face_sample import FaceSample
from src.models.video_frame import VideoFrame, VideoInfo

logger = logging.getLogger(__name__)


class Database:
    """数据库操作类"""

    def __init__(self, db_path: Optional[Path] = None):
        """
        初始化数据库

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path or DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        """初始化数据库表结构"""
        with self.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    series_id TEXT,
                    file_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    duration REAL DEFAULT 0,
                    fps REAL DEFAULT 0,
                    total_frames INTEGER DEFAULT 0,
                    width INTEGER DEFAULT 0,
                    height INTEGER DEFAULT 0,
                    processed_frames INTEGER DEFAULT 0,
                    detected_faces INTEGER DEFAULT 0,
                    characters_found INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (series_id) REFERENCES tv_series(series_id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS frames (
                    frame_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    frame_number INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    image_path TEXT,
                    width INTEGER DEFAULT 0,
                    height INTEGER DEFAULT 0,
                    is_scene_change BOOLEAN DEFAULT 0,
                    scene_id INTEGER,
                    face_count INTEGER DEFAULT 0,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS face_samples (
                    sample_id TEXT PRIMARY KEY,
                    frame_id TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    bbox TEXT NOT NULL,
                    embedding BLOB,
                    quality_score REAL DEFAULT 0,
                    timestamp REAL NOT NULL,
                    frame_number INTEGER NOT NULL,
                    image_path TEXT,
                    face_size TEXT,
                    cluster_id INTEGER,
                    character_id TEXT,
                    recognition_confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS characters (
                    character_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    aliases TEXT,
                    description TEXT,
                    color TEXT,
                    recognition_threshold REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS character_prototypes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    image_path TEXT NOT NULL,
                    quality_score REAL DEFAULT 0,
                    timestamp REAL DEFAULT 0,
                    FOREIGN KEY (character_id) REFERENCES characters(character_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS character_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id TEXT NOT NULL,
                    frame_path TEXT NOT NULL,
                    bbox TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    quality REAL DEFAULT 0,
                    scene_id INTEGER,
                    FOREIGN KEY (character_id) REFERENCES characters(character_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS recognition_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    frame_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    character_id TEXT,
                    bbox TEXT NOT NULL,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_face_samples_cluster ON face_samples(cluster_id);
                CREATE INDEX IF NOT EXISTS idx_face_samples_character ON face_samples(character_id);

                CREATE TABLE IF NOT EXISTS tv_series (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    series_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    year INTEGER,
                    description TEXT,
                    poster_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS actors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    actor_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    photo_path TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS series_actors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    series_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    character_name TEXT NOT NULL,
                    role_order INTEGER DEFAULT 0,
                    is_main_character BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (series_id) REFERENCES tv_series(series_id) ON DELETE CASCADE,
                    FOREIGN KEY (actor_id) REFERENCES actors(actor_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_series_actors_series ON series_actors(series_id);
                CREATE INDEX IF NOT EXISTS idx_series_actors_actor ON series_actors(actor_id);
                CREATE INDEX IF NOT EXISTS idx_recognition_video ON recognition_results(video_id);
                CREATE INDEX IF NOT EXISTS idx_recognition_character ON recognition_results(character_id);
            """)

            logger.info("Database initialized")

    def save_video_info(self, video_info: VideoInfo):
        """保存视频信息"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO videos
                (video_id, file_path, filename, duration, fps, total_frames,
                 width, height, processed_frames, detected_faces, characters_found, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                video_info.video_id,
                video_info.file_path,
                video_info.filename,
                video_info.duration,
                video_info.fps,
                video_info.total_frames,
                video_info.width,
                video_info.height,
                video_info.processed_frames,
                video_info.detected_faces,
                video_info.characters_found,
            ))

    def get_video_info(self, video_id: str) -> Optional[VideoInfo]:
        """获取视频信息"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE video_id = ?",
                (video_id,)
            ).fetchone()

            if row is None:
                return None

            return VideoInfo(
                video_id=row['video_id'],
                file_path=row['file_path'],
                filename=row['filename'],
                duration=row['duration'],
                fps=row['fps'],
                total_frames=row['total_frames'],
                width=row['width'],
                height=row['height'],
                processed_frames=row['processed_frames'],
                detected_faces=row['detected_faces'],
                characters_found=row['characters_found'],
            )

    def save_frames(self, frames: List[VideoFrame]):
        """批量保存帧信息"""
        with self.get_connection() as conn:
            for frame in frames:
                conn.execute("""
                    INSERT OR REPLACE INTO frames
                    (frame_id, video_id, frame_number, timestamp, image_path,
                     width, height, is_scene_change, scene_id, face_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    frame.frame_id,
                    frame.video_id,
                    frame.frame_number,
                    frame.timestamp,
                    frame.image_path,
                    frame.width,
                    frame.height,
                    int(frame.is_scene_change),
                    frame.scene_id,
                    frame.face_count,
                ))

    def save_face_samples(self, samples: List[FaceSample]):
        """批量保存人脸样本"""
        with self.get_connection() as conn:
            for sample in samples:
                embedding_blob = None
                if sample.embedding is not None:
                    embedding_blob = sample.embedding.tobytes()

                face_size_json = json.dumps(sample.face_size) if sample.face_size else None

                conn.execute("""
                    INSERT OR REPLACE INTO face_samples
                    (sample_id, frame_id, video_id, bbox, embedding, quality_score,
                     timestamp, frame_number, image_path, face_size, cluster_id, character_id,
                     recognition_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sample.sample_id,
                    sample.frame_id,
                    sample.video_id,
                    json.dumps(sample.bbox),
                    embedding_blob,
                    sample.quality_score,
                    sample.timestamp,
                    sample.frame_number,
                    sample.image_path,
                    face_size_json,
                    sample.cluster_id,
                    sample.character_id,
                    sample.recognition_confidence,
                ))

    def get_face_samples(self, video_id: str,
                        cluster_id: Optional[int] = None,
                        character_id: Optional[str] = None) -> List[FaceSample]:
        """获取人脸样本"""
        with self.get_connection() as conn:
            query = "SELECT * FROM face_samples WHERE video_id = ?"
            params = [video_id]

            if cluster_id is not None:
                query += " AND cluster_id = ?"
                params.append(cluster_id)

            if character_id is not None:
                query += " AND character_id = ?"
                params.append(character_id)

            rows = conn.execute(query, params).fetchall()

            samples = []
            for row in rows:
                embedding = None
                if row['embedding']:
                    embedding = np.frombuffer(row['embedding'], dtype=np.float32)

                face_size = json.loads(row['face_size']) if row['face_size'] else None

                # 处理 cluster_id 可能是 bytes 的情况
                cluster_id = row['cluster_id']
                if isinstance(cluster_id, bytes):
                    cluster_id = int.from_bytes(cluster_id, byteorder='little')

                sample = FaceSample(
                    sample_id=row['sample_id'],
                    frame_id=row['frame_id'],
                    video_id=row['video_id'],
                    bbox=json.loads(row['bbox']),
                    embedding=embedding,
                    quality_score=row['quality_score'],
                    timestamp=row['timestamp'],
                    frame_number=row['frame_number'],
                    image_path=row['image_path'],
                    face_size=face_size,
                    cluster_id=cluster_id,
                    character_id=row['character_id'],
                    recognition_confidence=row['recognition_confidence'],
                )
                samples.append(sample)

            return samples

    def save_character(self, character: Character):
        """保存角色信息"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO characters
                (character_id, video_id, name, aliases, description, color,
                 recognition_threshold, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                character.character_id,
                character.video_id,
                character.name,
                json.dumps(character.aliases),
                character.description,
                character.color,
                character.recognition_threshold,
            ))

            # 删除旧的原型数据
            conn.execute(
                "DELETE FROM character_prototypes WHERE character_id = ?",
                (character.character_id,)
            )

            # 插入原型数据
            for prototype in character.prototypes:
                conn.execute("""
                    INSERT INTO character_prototypes
                    (character_id, embedding, image_path, quality_score, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    character.character_id,
                    prototype.embedding.tobytes(),
                    prototype.image_path,
                    prototype.quality_score,
                    prototype.timestamp,
                ))

    def get_characters(self, video_id: str) -> List[Character]:
        """获取角色列表"""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM characters WHERE video_id = ?
                ORDER BY created_at
            """, (video_id,)).fetchall()

            characters = []
            for row in rows:
                # 获取原型数据
                proto_rows = conn.execute("""
                    SELECT * FROM character_prototypes WHERE character_id = ?
                """, (row['character_id'],)).fetchall()

                prototypes = []
                for proto_row in proto_rows:
                    from src.models.character import Prototype
                    prototype = Prototype(
                        embedding=np.frombuffer(proto_row['embedding'], dtype=np.float32),
                        image_path=proto_row['image_path'],
                        quality_score=proto_row['quality_score'],
                        timestamp=proto_row['timestamp'],
                    )
                    prototypes.append(prototype)

                character = Character(
                    character_id=row['character_id'],
                    name=row['name'],
                    video_id=row['video_id'],
                    prototypes=prototypes,
                    aliases=json.loads(row['aliases']) if row['aliases'] else [],
                    description=row['description'] or '',
                    color=row['color'],
                    recognition_threshold=row['recognition_threshold'],
                )
                characters.append(character)

            return characters

    def update_sample_character(self, sample_id: str, character_id: Optional[str]):
        """更新样本的角色标注"""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE face_samples
                SET character_id = ?
                WHERE sample_id = ?
            """, (character_id, sample_id))

    def update_sample_cluster(self, sample_id: str, cluster_id: Optional[int]):
        """更新样本的簇标注"""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE face_samples
                SET cluster_id = ?
                WHERE sample_id = ?
            """, (cluster_id, sample_id))

    def clear_video_clusters(self, video_id: str):
        """清除视频的所有聚类标注"""
        logger.info(f"清除视频 {video_id} 的所有聚类标注")
        with self.get_connection() as conn:
            result = conn.execute("""
                UPDATE face_samples
                SET cluster_id = NULL
                WHERE video_id = ?
            """, (video_id,))
            logger.info(f"已清除 {result.rowcount} 个样本的聚类标注")

    def get_cluster_summary(self, video_id: str) -> List[Dict]:
        """获取聚类摘要"""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT cluster_id,
                       COUNT(*) as sample_count,
                       AVG(quality_score) as avg_quality,
                       MIN(timestamp) as first_appearance,
                       MAX(timestamp) as last_appearance
                FROM face_samples
                WHERE video_id = ? AND cluster_id IS NOT NULL
                GROUP BY cluster_id
                ORDER BY cluster_id
            """, (video_id,)).fetchall()

            summary = []
            for row in rows:
                summary.append({
                    'cluster_id': row['cluster_id'],
                    'sample_count': row['sample_count'],
                    'avg_quality': row['avg_quality'],
                    'first_appearance': row['first_appearance'],
                    'last_appearance': row['last_appearance'],
                })

            return summary

    def save_recognition_results(self, video_id: str, results: List[Any]):
        """保存识别结果"""
        with self.get_connection() as conn:
            for result in results:
                conn.execute("""
                    INSERT INTO recognition_results
                    (video_id, frame_id, timestamp, character_id, bbox, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    video_id,
                    result.sample_id,
                    result.timestamp,
                    result.character_id,
                    json.dumps(result.bbox),
                    result.confidence,
                ))

    def delete_video(self, video_id: str):
        """删除视频相关数据"""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM recognition_results WHERE video_id = ?", (video_id,))
            conn.execute("DELETE FROM face_samples WHERE video_id = ?", (video_id,))
            conn.execute("DELETE FROM frames WHERE video_id = ?", (video_id,))
            conn.execute("DELETE FROM characters WHERE video_id = ?", (video_id,))
            conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))

    def update_video_series(self, video_id: str, series_id: str) -> bool:
        """更新视频所属电视剧"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    UPDATE videos SET series_id = ? WHERE video_id = ?
                """, (series_id, video_id))
            logger.info(f"更新视频 {video_id} 所属电视剧为 {series_id}")
            return True
        except Exception as e:
            logger.error(f"更新视频电视剧失败: {e}")
            return False

    # ========== 电视剧管理 ==========

    def create_tv_series(self, series_id: str, name: str, year: int = None,
                        description: str = None, poster_path: str = None) -> bool:
        """创建电视剧"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO tv_series (series_id, name, year, description, poster_path)
                    VALUES (?, ?, ?, ?, ?)
                """, (series_id, name, year, description, poster_path))
            logger.info(f"创建电视剧: {series_id} - {name}")
            return True
        except Exception as e:
            logger.error(f"创建电视剧失败: {e}")
            return False

    def get_tv_series(self, series_id: str = None) -> List[Dict]:
        """获取电视剧列表"""
        with self.get_connection() as conn:
            if series_id:
                rows = conn.execute("""
                    SELECT * FROM tv_series WHERE series_id = ?
                """, (series_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM tv_series ORDER BY created_at DESC").fetchall()

            return [dict(row) for row in rows]

    def delete_tv_series(self, series_id: str) -> bool:
        """删除电视剧"""
        try:
            with self.get_connection() as conn:
                conn.execute("DELETE FROM tv_series WHERE series_id = ?", (series_id,))
            logger.info(f"删除电视剧: {series_id}")
            return True
        except Exception as e:
            logger.error(f"删除电视剧失败: {e}")
            return False

    # ========== 演员管理 ==========

    def create_actor(self, actor_id: str, name: str, photo_path: str = None,
                    embedding: np.ndarray = None) -> bool:
        """创建演员"""
        try:
            embedding_blob = embedding.tobytes() if embedding is not None else None
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO actors (actor_id, name, photo_path, embedding)
                    VALUES (?, ?, ?, ?)
                """, (actor_id, name, photo_path, embedding_blob))
            logger.info(f"创建演员: {actor_id} - {name}")
            return True
        except Exception as e:
            logger.error(f"创建演员失败: {e}")
            return False

    def get_actors(self, actor_id: str = None) -> List[Dict]:
        """获取演员列表"""
        with self.get_connection() as conn:
            if actor_id:
                rows = conn.execute("""
                    SELECT * FROM actors WHERE actor_id = ?
                """, (actor_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM actors ORDER BY name").fetchall()

            return [dict(row) for row in rows]

    def update_actor_embedding(self, actor_id: str, embedding: np.ndarray) -> bool:
        """更新演员人脸embedding"""
        try:
            embedding_blob = embedding.tobytes()
            with self.get_connection() as conn:
                conn.execute("""
                    UPDATE actors SET embedding = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE actor_id = ?
                """, (embedding_blob, actor_id))
            logger.info(f"更新演员embedding: {actor_id}")
            return True
        except Exception as e:
            logger.error(f"更新演员embedding失败: {e}")
            return False

    def delete_actor(self, actor_id: str) -> bool:
        """删除演员"""
        try:
            with self.get_connection() as conn:
                conn.execute("DELETE FROM actors WHERE actor_id = ?", (actor_id,))
            logger.info(f"删除演员: {actor_id}")
            return True
        except Exception as e:
            logger.error(f"删除演员失败: {e}")
            return False

    # ========== 电视剧演员关联管理 ==========

    def add_series_actor(self, series_id: str, actor_id: str, character_name: str,
                        role_order: int = 0, is_main_character: bool = True) -> bool:
        """添加电视剧演员关联"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO series_actors
                    (series_id, actor_id, character_name, role_order, is_main_character)
                    VALUES (?, ?, ?, ?, ?)
                """, (series_id, actor_id, character_name, role_order, is_main_character))
            logger.info(f"添加电视剧演员: {series_id} - {actor_id} 饰演 {character_name}")
            return True
        except Exception as e:
            logger.error(f"添加电视剧演员失败: {e}")
            return False

    def get_series_actors(self, series_id: str) -> List[Dict]:
        """获取电视剧的演员列表"""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT sa.*, a.name as actor_name, a.photo_path
                FROM series_actors sa
                LEFT JOIN actors a ON sa.actor_id = a.actor_id
                WHERE sa.series_id = ?
                ORDER BY sa.role_order, sa.id
            """, (series_id,)).fetchall()

            return [dict(row) for row in rows]

    def remove_series_actor(self, series_id: str, actor_id: str) -> bool:
        """移除电视剧演员"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    DELETE FROM series_actors
                    WHERE series_id = ? AND actor_id = ?
                """, (series_id, actor_id))
            logger.info(f"移除电视剧演员: {series_id} - {actor_id}")
            return True
        except Exception as e:
            logger.error(f"移除电视剧演员失败: {e}")
            return False

    def get_actor_embedding(self, actor_id: str) -> Optional[np.ndarray]:
        """获取演员的人脸embedding"""
        with self.get_connection() as conn:
            row = conn.execute("""
                SELECT embedding FROM actors WHERE actor_id = ?
            """, (actor_id,)).fetchone()

            if row and row['embedding']:
                return np.frombuffer(row['embedding'], dtype=np.float32)
            return None

