"""
影视人脸识别系统 - FastAPI 服务器
提供Web前端调用的API接口
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import shutil
from pathlib import Path
import sys
import threading
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.video_processor import VideoProcessor
from src.core.face_detector import FaceDetector
from src.core.face_embedder import FaceEmbedder
from src.core.cluster_engine import ClusterEngine
from src.core.recognition_engine import RecognitionEngine
from src.storage.database import Database
from src.storage.character_store import CharacterStore
from config.settings import DATA_ROOT, MODELS_ROOT

app = FastAPI(
    title="影视人脸识别系统 API",
    description="自动人脸检测、聚类和识别API",
    version="0.1.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局组件
database = Database()
detector = None
embedder = None
processor = None
cluster_engine = ClusterEngine()
recognition_engine = None

# 视频处理任务状态
processing_tasks: Dict[str, Dict[str, Any]] = {}


# 数据模型
class ClusterNameRequest(BaseModel):
    video_id: str
    name: str


class MergeClustersRequest(BaseModel):
    video_id: str
    source_cluster_id: int
    target_cluster_id: int


class RemoveSampleRequest(BaseModel):
    video_id: str
    sample_id: str


# 启动时初始化
@app.on_event("startup")
async def startup_event():
    """初始化模型"""
    global detector, embedder, processor, recognition_engine

    print("初始化人脸检测模型...")
    try:
        detector = FaceDetector(model_path=str(MODELS_ROOT), use_gpu=True)
        detector.initialize()
    except Exception as e:
        print(f"检测器初始化失败: {e}")

    print("初始化特征提取模型...")
    try:
        embedder = FaceEmbedder(model_path=str(MODELS_ROOT), use_gpu=True)
        embedder.initialize()
    except Exception as e:
        print(f"特征提取器初始化失败: {e}")

    if detector and embedder:
        processor = VideoProcessor(detector, embedder)
        recognition_engine = RecognitionEngine(embedder)
        print("模型初始化完成！")
    else:
        print("警告: 模型未完全初始化，某些功能可能不可用")


# ========== 视频处理相关 API ==========

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        'status': 'ok',
        'models_loaded': detector is not None and embedder is not None,
    }


@app.get("/api/videos")
async def get_videos():
    """获取所有视频列表"""
    try:
        with database.get_connection() as conn:
            videos = conn.execute("""
                SELECT
                    v.*,
                    s.name as series_name,
                    s.year as series_year
                FROM videos v
                LEFT JOIN tv_series s ON v.series_id = s.series_id
                ORDER BY v.created_at DESC
            """).fetchall()

        result = []
        for v in videos:
            result.append({
                'video_id': v['video_id'],
                'filename': v['filename'],
                'duration': v['duration'],
                'processed_frames': v['processed_frames'],
                'detected_faces': v['detected_faces'],
                'characters_found': v['characters_found'],
                'series_id': v['series_id'],
                'series_name': v['series_name'],
                'series_year': v['series_year'],
                'created_at': str(v['created_at']) if v['created_at'] else None,
            })
        return result
    except Exception as e:
        return {'error': str(e), 'videos': []}


@app.put("/api/videos/{video_id}/series")
async def update_video_series_api(video_id: str, request: Dict[str, Any]):
    """更新视频所属电视剧"""
    try:
        series_id = request.get('series_id')
        success = database.update_video_series(video_id, series_id)
        if success:
            return {'success': True, 'message': '视频关联已更新'}
        else:
            return {'success': False, 'error': '视频不存在'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...), series_id: str = Form(None)):
    """上传视频"""
    # 保存文件
    upload_dir = DATA_ROOT / "raw"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 从文件名获取video_id（去掉扩展名）
    video_id = Path(file.filename).stem

    # 保存到数据库
    with database.get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO videos (video_id, file_path, filename, series_id)
            VALUES (?, ?, ?, ?)
        """, (video_id, str(file_path), file.filename, series_id))

    # 创建处理任务
    processing_tasks[video_id] = {
        'status': 'pending',
        'progress': 0,
        'message': '等待处理',
        'series_id': series_id,
        'video_id': video_id,
        'file_path': str(file_path),
    }

    # 使用线程处理视频（不阻塞API响应）
    thread = threading.Thread(target=process_video_task, args=(video_id, str(file_path)))
    thread.daemon = True
    thread.start()

    return {'video_id': video_id, 'status': 'pending'}


def process_video_task(video_id: str, file_path: str):
    """后台处理视频任务"""
    print(f"[DEBUG] Starting video processing: {video_id}")
    processing_tasks[video_id]['status'] = 'processing'
    processing_tasks[video_id]['progress'] = 10
    processing_tasks[video_id]['message'] = '正在处理视频'

    try:
        if processor is None:
            print("[DEBUG] Processor not initialized!")
            raise Exception("处理器未初始化")

        print(f"[DEBUG] Processor OK, starting to process: {file_path}")
        result = processor.process_video(file_path, video_id)
        print(f"[DEBUG] Processing complete: {result}")

        # 保存到数据库
        database.save_video_info(result['video_info'])
        database.save_frames(result['frames'])
        database.save_face_samples(result['face_samples'])

        processing_tasks[video_id]['status'] = 'completed'
        processing_tasks[video_id]['progress'] = 100
        processing_tasks[video_id]['message'] = '处理完成'
        processing_tasks[video_id]['result'] = {
            'processed_frames': result['video_info'].processed_frames,
            'detected_faces': result['video_info'].detected_faces,
        }
        print(f"[DEBUG] Video processing completed successfully: {video_id}")
    except Exception as e:
        print(f"[DEBUG] Error processing video: {e}")
        processing_tasks[video_id]['status'] = 'error'
        processing_tasks[video_id]['message'] = f'处理失败: {str(e)}'
        import traceback
        traceback.print_exc()


@app.get("/api/videos/{video_id}/status")
async def get_video_status(video_id: str):
    """获取视频处理状态"""
    if video_id not in processing_tasks:
        # 检查数据库
        try:
            video_info = database.get_video_info(video_id)
            if video_info:
                return {
                    'video_id': video_id,
                    'status': 'completed',
                    'progress': 100,
                    'message': '处理完成',
                    'result': {
                        'processed_frames': video_info.processed_frames,
                        'detected_faces': video_info.detected_faces,
                    }
                }
        except:
            pass
        raise HTTPException(status_code=404, detail="视频不存在")

    return processing_tasks[video_id]


@app.post("/api/videos/{video_id}/reprocess_faces")
async def reprocess_faces(video_id: str):
    """重新处理人脸检测（从已有帧开始，不重新抽帧）- 后台任务"""
    import cv2
    from pathlib import Path

    # 检查帧目录是否存在
    frames_dir = Path(DATA_ROOT) / "processed" / "frames" / video_id
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="未找到已提取的帧，请先进行完整处理")

    # 获取所有帧文件
    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        raise HTTPException(status_code=404, detail="帧目录为空")

    # 创建处理任务
    task_id = f"reprocess_{video_id}"
    processing_tasks[task_id] = {
        'status': 'pending',
        'progress': 0,
        'message': '等待处理',
        'video_id': video_id,
        'task_type': 'reprocess_faces',
    }

    # 使用线程处理（不阻塞API响应）
    thread = threading.Thread(target=reprocess_faces_task, args=(task_id, video_id))
    thread.daemon = True
    thread.start()

    return {'task_id': task_id, 'video_id': video_id, 'status': 'pending'}


def reprocess_faces_task(task_id: str, video_id: str):
    """后台重新处理人脸任务"""
    import cv2
    from pathlib import Path

    print(f"[DEBUG] Starting reprocess_faces: {video_id}")
    processing_tasks[task_id]['status'] = 'processing'
    processing_tasks[task_id]['progress'] = 10
    processing_tasks[task_id]['message'] = '正在重新检测人脸'

    try:
        if detector is None or embedder is None:
            print("[DEBUG] Detector or embedder not initialized!")
            raise Exception("检测器或特征提取器未初始化")

        frames_dir = Path(DATA_ROOT) / "processed" / "frames" / video_id
        frame_files = sorted(frames_dir.glob("*.jpg"))
        total_frames = len(frame_files)

        # 清理旧的人脸数据
        faces_dir = Path(DATA_ROOT) / "processed" / "faces" / video_id
        if faces_dir.exists():
            import shutil
            shutil.rmtree(faces_dir)
        faces_dir.mkdir(parents=True, exist_ok=True)

        # 获取第一帧来设置分辨率
        first_frame = cv2.imread(str(frame_files[0]))
        height, width = first_frame.shape[:2]
        detector.set_video_resolution(width, height)

        # 处理每一帧
        all_samples = []
        processed_count = 0

        for idx, frame_path in enumerate(frame_files):
            # 从文件名提取信息
            frame_name = frame_path.stem
            parts = frame_name.split('_')

            # 提取帧号
            if 'frame_' in frame_name:
                if len(parts) >= 3 and parts[-2] == 'frame':
                    frame_number = int(parts[-1])
                elif parts[0] == 'frame':
                    frame_number = int(parts[1])
                else:
                    continue
            else:
                continue

            # 读取帧
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            # 检测人脸
            samples = detector.detect_faces(
                frame,
                frame_id=frame_name,
                video_id=video_id,
                frame_number=frame_number,
                timestamp=frame_number
            )

            # 提取embedding并保存人脸图像
            for sample in samples:
                if sample.embedding is None:
                    face_region = frame[
                        int(sample.bbox[1]):int(sample.bbox[3]),
                        int(sample.bbox[0]):int(sample.bbox[2])
                    ]
                    sample.embedding = embedder.compute_embedding(
                        face_region,
                        face_bbox=sample.bbox,
                        face_landmarks=sample.landmarks
                    )

                # 保存人脸图像
                face_filename = f"{frame_name}_face_{samples.index(sample)}.jpg"
                face_path = faces_dir / face_filename

                face_region = frame[
                    int(sample.bbox[1]):int(sample.bbox[3]),
                    int(sample.bbox[0]):int(sample.bbox[2])
                ]
                cv2.imwrite(str(face_path), face_region)
                sample.image_path = str(face_path)

            all_samples.extend(samples)
            processed_count += 1

            # 更新进度
            progress = 10 + int((processed_count / total_frames) * 80)
            processing_tasks[task_id]['progress'] = progress
            processing_tasks[task_id]['message'] = f'已处理 {processed_count}/{total_frames} 帧'

            if processed_count % 50 == 0:
                print(f"[DEBUG] 已处理 {processed_count}/{total_frames} 帧")

        # 保存到数据库
        database.save_face_samples(all_samples)

        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['progress'] = 100
        processing_tasks[task_id]['message'] = f'重新处理完成，检测到 {len(all_samples)} 张人脸'
        processing_tasks[task_id]['result'] = {
            'processed_frames': processed_count,
            'total_faces': len(all_samples),
        }
        print(f"[DEBUG] Reprocess faces completed: {video_id}, {len(all_samples)} faces")

    except Exception as e:
        print(f"[DEBUG] Error reprocessing faces: {e}")
        processing_tasks[task_id]['status'] = 'error'
        processing_tasks[task_id]['message'] = f'处理失败: {str(e)}'
        import traceback
        traceback.print_exc()


# ========== 聚类相关 API ==========

@app.post("/api/videos/{video_id}/cluster")
async def cluster_faces(video_id: str, min_cluster_size: int = 5):
    """对视频的人脸进行聚类"""
    try:
        # 清除之前的聚类结果
        database.clear_video_clusters(video_id)

        samples = database.get_face_samples(video_id)

        if not samples:
            raise HTTPException(status_code=404, detail="未找到人脸样本")

        # 过滤有效样本
        valid_samples = [s for s in samples if s.has_embedding]

        if not valid_samples:
            raise HTTPException(status_code=400, detail="没有人脸特征数据")

        # 执行聚类
        clusters = cluster_engine.discover_characters(valid_samples, min_cluster_size)

        # 更新数据库
        for sample in valid_samples:
            if sample.cluster_id is not None:
                database.update_sample_cluster(sample.sample_id, sample.cluster_id)

        # 返回聚类结果
        result = []
        for cluster in clusters:
            result.append({
                'cluster_id': cluster.cluster_id,
                'sample_count': cluster.size,
                'avg_quality': cluster.avg_quality,
                'first_appearance': min(s.timestamp for s in cluster.samples),
                'last_appearance': max(s.timestamp for s in cluster.samples),
                'samples': [
                    {
                        'sample_id': s.sample_id,
                        'image_path': s.image_path,
                        'quality_score': s.quality_score,
                        'timestamp': s.timestamp,
                    }
                    for s in cluster.samples[:20]
                ]
            })

        return {'clusters': result, 'total': len(result)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/videos/{video_id}/clusters")
async def get_clusters(video_id: str):
    """获取聚类结果"""
    try:
        samples = database.get_face_samples(video_id)

        # 按簇分组
        clusters_dict: Dict[int, List] = {}
        for sample in samples:
            if sample.cluster_id is not None:
                if sample.cluster_id not in clusters_dict:
                    clusters_dict[sample.cluster_id] = []
                clusters_dict[sample.cluster_id].append(sample)

        result = []
        for cluster_id, cluster_samples in clusters_dict.items():
            # 确保 timestamp 是 float 类型
            timestamps = [float(s.timestamp) for s in cluster_samples]

            # 转换图片路径为 API URL
            api_samples = []
            for s in cluster_samples[:50]:  # 增加返回样本数
                # 将本地路径转换为 API URL
                image_url = s.image_path
                if s.image_path and str(DATA_ROOT) in s.image_path:
                    try:
                        # 提取相对路径并转换为 API URL
                        rel_path = Path(s.image_path).relative_to(DATA_ROOT)
                        if 'faces' in rel_path.parts:
                            # 格式: processed/faces/{video_id}/{filename}
                            parts = rel_path.parts
                            if len(parts) >= 4 and parts[0] == 'processed' and parts[1] == 'faces':
                                vid = parts[2]
                                filename = rel_path.name
                                image_url = f"http://localhost:8000/api/face_images/{vid}/{filename}"
                    except Exception as e:
                        print(f"[DEBUG] Path conversion error for {s.image_path}: {e}")

                api_samples.append({
                    'sample_id': s.sample_id,
                    'image_path': image_url,
                    'quality_score': s.quality_score,
                    'timestamp': s.timestamp,
                    'cluster_id': s.cluster_id,
                    'character_id': s.character_id,
                })

            result.append({
                'cluster_id': cluster_id,
                'sample_count': len(cluster_samples),
                'avg_quality': sum(s.quality_score for s in cluster_samples) / len(cluster_samples),
                'first_appearance': min(timestamps),
                'last_appearance': max(timestamps),
                'samples': api_samples
            })

        return {'clusters': sorted(result, key=lambda x: x['cluster_id'])}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'clusters': [], 'error': str(e)}


@app.put("/api/clusters/{cluster_id}/name")
async def name_cluster(cluster_id: int, request: ClusterNameRequest):
    """为簇命名角色"""
    try:
        # 更新角色库
        character_store = CharacterStore(DATA_ROOT / "characters", database)
        library = character_store.load_library(request.video_id)

        if not library:
            library = character_store.load_library_from_db(request.video_id)

        if not library:
            from src.models.character import CharacterLibrary, Character
            library = CharacterLibrary(video_info={'video_id': request.video_id}, characters=[])

        # 检查角色是否存在
        char_id = f"char_{cluster_id:03d}"
        character = library.get_character(char_id)

        if character:
            character.name = request.name
        else:
            from src.models.character import Character
            character = Character(
                character_id=char_id,
                name=request.name,
                video_id=request.video_id,
            )
            library.add_character(character)

        # 保存
        character_store.save_library(library, request.video_id)
        database.save_character(character)

        return {'success': True, 'character_id': char_id, 'name': request.name}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


@app.post("/api/clusters/merge")
async def merge_clusters(request: MergeClustersRequest):
    """合并两个簇"""
    try:
        # 获取所有样本
        samples = database.get_face_samples(request.video_id)

        # 过滤出两个簇的样本
        source_samples = [s for s in samples if s.cluster_id == request.source_cluster_id]
        target_samples = [s for s in samples if s.cluster_id == request.target_cluster_id]

        if not source_samples:
            raise HTTPException(status_code=404, detail=f"源簇 {request.source_cluster_id} 不存在")
        if not target_samples:
            raise HTTPException(status_code=404, detail=f"目标簇 {request.target_cluster_id} 不存在")

        # 将源簇的所有样本的cluster_id更新为目标簇的ID
        for sample in source_samples:
            database.update_sample_cluster(sample.sample_id, request.target_cluster_id)

        return {
            'success': True,
            'message': f"已将簇 {request.source_cluster_id} 合并到簇 {request.target_cluster_id}",
            'merged_count': len(source_samples),
            'target_cluster_id': request.target_cluster_id
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


@app.post("/api/samples/remove")
async def remove_sample_from_cluster(request: RemoveSampleRequest):
    """从簇中移除样本（设置cluster_id为NULL）"""
    try:
        # 更新样本的cluster_id为NULL
        database.update_sample_cluster(request.sample_id, None)

        return {
            'success': True,
            'message': f"样本 {request.sample_id} 已从簇中移除"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


# ========== 识别相关 API ==========

@app.post("/api/videos/{video_id}/recognize")
async def recognize_video(video_id: str, use_temporal_smoothing: bool = True):
    """运行人脸识别"""
    try:
        # 加载角色库
        character_store = CharacterStore(DATA_ROOT / "characters", database)
        library = character_store.load_library(video_id)

        if not library:
            library = character_store.load_library_from_db(video_id)

        if not library or not library.characters:
            raise HTTPException(status_code=400, detail="请先完成角色标注")

        # 加载角色到识别引擎
        recognition_engine.load_characters(library.characters)

        # 获取未识别的样本
        all_samples = database.get_face_samples(video_id)
        unassigned_samples = [s for s in all_samples if s.character_id is None]

        results = []
        for sample in unassigned_samples[:100]:  # 限制数量
            if sample.has_embedding:
                result = recognition_engine.recognize(sample, use_temporal_smoothing)
                if result and result.character_id:
                    results.append({
                        'sample_id': result.sample_id,
                        'character_id': result.character_id,
                        'character_name': result.character_name,
                        'confidence': result.confidence,
                        'timestamp': result.timestamp,
                        'bbox': str(result.bbox),
                    })

                    # 更新数据库
                    database.update_sample_character(sample.sample_id, result.character_id)

        # 保存识别结果
        from src.core.recognition_engine import RecognitionResult as RecResult
        rec_results = [
            type('obj', (object,), {
                'sample_id': r['sample_id'],
                'character_id': r['character_id'],
                'confidence': r['confidence'],
                'timestamp': r['timestamp'],
                'bbox': r['bbox'],
            }) for r in results
        ]
        database.save_recognition_results(video_id, rec_results)

        return {
            'video_id': video_id,
            'total_samples': len(unassigned_samples),
            'recognized': len(results),
            'results': results,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/videos/{video_id}/recognition")
async def get_recognition_results(video_id: str):
    """获取识别结果"""
    try:
        with database.get_connection() as conn:
            rows = conn.execute("""
                SELECT rr.*, c.name as character_name
                FROM recognition_results rr
                LEFT JOIN characters c ON rr.character_id = c.character_id
                WHERE rr.video_id = ?
                ORDER BY rr.timestamp
                LIMIT 200
            """, (video_id,)).fetchall()

        result = []
        for r in rows:
            result.append({
                'sample_id': r['sample_id'],
                'frame_id': r['frame_id'],
                'timestamp': r['timestamp'],
                'character_id': r['character_id'],
                'character_name': r['character_name'],
                'confidence': r['confidence'],
                'bbox': r['bbox'],
            })

        return result
    except Exception as e:
        return []


# ========== 分析相关 API ==========

@app.get("/api/videos/{video_id}/analysis")
async def get_analysis(video_id: str):
    """获取分析数据"""
    try:
        character_store = CharacterStore(DATA_ROOT / "characters", database)
        stats = character_store.get_character_statistics(video_id)

        # 获取识别统计
        with database.get_connection() as conn:
            rows = conn.execute("""
                SELECT
                    character_id,
                    COUNT(*) as count,
                    MIN(timestamp) as first_seen,
                    MAX(timestamp) as last_seen
                FROM recognition_results
                WHERE video_id = ? AND character_id IS NOT NULL
                GROUP BY character_id
            """, (video_id,)).fetchall()

        character_stats = []
        for r in rows:
            char = database.get_characters(video_id)
            char_name = next((c.name for c in char if c.character_id == r['character_id']), r['character_id'])
            character_stats.append({
                'character_id': r['character_id'],
                'character_name': char_name,
                'count': r['count'],
                'first_seen': r['first_seen'],
                'last_seen': r['last_seen'],
                'screen_time': r['last_seen'] - r['first_seen'],
            })

        return {
            'video_id': video_id,
            'characters': stats.get('characters', []),
            'recognition_stats': character_stats,
        }
    except Exception as e:
        return {'video_id': video_id, 'characters': [], 'recognition_stats': [], 'error': str(e)}


# ========== 人脸样本 API ==========

@app.get("/api/videos/{video_id}/samples")
async def get_samples(video_id: str, cluster_id: Optional[int] = None, character_id: Optional[str] = None):
    """获取人脸样本"""
    try:
        samples = database.get_face_samples(video_id, cluster_id, character_id)

        result = []
        for sample in samples:
            result.append({
                'sample_id': sample.sample_id,
                'frame_id': sample.frame_id,
                'timestamp': sample.timestamp,
                'quality_score': sample.quality_score,
                'image_path': sample.image_path,
                'cluster_id': sample.cluster_id,
                'character_id': sample.character_id,
            })

        return {'samples': result, 'total': len(result)}
    except Exception as e:
        return {'samples': [], 'total': 0, 'error': str(e)}


@app.delete("/api/samples/{sample_id}")
async def remove_sample(sample_id: str):
    """移除样本"""
    try:
        database.update_sample_cluster(sample_id, None)
        database.update_sample_character(sample_id, None)
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ========== 角色库 API ==========

@app.get("/api/videos/{video_id}/characters")
async def get_characters(video_id: str):
    """获取角色列表"""
    try:
        characters = database.get_characters(video_id)

        result = []
        for char in characters:
            result.append({
                'character_id': char.character_id,
                'name': char.name,
                'video_id': char.video_id,
                'prototypes': [],
            })

        return {'characters': result}
    except Exception as e:
        return {'characters': [], 'error': str(e)}


# ========== 电视剧管理 API ==========

@app.get("/api/series")
async def get_series():
    """获取所有电视剧列表"""
    try:
        series_list = database.get_tv_series()
        return {'series': series_list}
    except Exception as e:
        return {'series': [], 'error': str(e)}


@app.post("/api/series")
async def create_series(request: Dict[str, Any]):
    """创建电视剧"""
    try:
        series_id = request.get('series_id') or f"series_{int(hash(str(request)) % 10000)}"
        success = database.create_tv_series(
            series_id=series_id,
            name=request.get('name', ''),
            year=request.get('year'),
            description=request.get('description'),
            poster_path=request.get('poster_path')
        )
        if success:
            return {'success': True, 'series_id': series_id}
        return {'success': False, 'error': '创建失败'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@app.delete("/api/series/{series_id}")
async def delete_series(series_id: str):
    """删除电视剧"""
    try:
        success = database.delete_tv_series(series_id)
        return {'success': success}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@app.get("/api/series/{series_id}/actors")
async def get_series_actors_api(series_id: str):
    """获取电视剧的演员列表"""
    try:
        actors = database.get_series_actors(series_id)
        return {'actors': actors}
    except Exception as e:
        return {'actors': [], 'error': str(e)}


@app.post("/api/series/{series_id}/actors")
async def add_series_actor(series_id: str, request: Dict[str, Any]):
    """为电视剧添加演员"""
    try:
        success = database.add_series_actor(
            series_id=series_id,
            actor_id=request.get('actor_id', ''),
            character_name=request.get('character_name', ''),
            role_order=request.get('role_order', 0),
            is_main_character=request.get('is_main_character', True)
        )
        if success:
            return {'success': True}
        return {'success': False, 'error': '添加失败'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@app.delete("/api/series/{series_id}/actors/{actor_id}")
async def remove_series_actor(series_id: str, actor_id: str):
    """从电视剧中移除演员"""
    try:
        success = database.remove_series_actor(series_id, actor_id)
        return {'success': success}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@app.get("/api/series/{series_id}/characters")
async def get_series_characters(series_id: str):
    """获取电视剧的角色统计信息（聚合所有视频的标注结果）"""
    try:
        with database.get_connection() as conn:
            # 获取剧集信息
            series = conn.execute("SELECT * FROM tv_series WHERE series_id = ?", (series_id,)).fetchone()
            if not series:
                raise HTTPException(status_code=404, detail="剧集不存在")

            # 获取剧集的演员列表
            actors = conn.execute("""
                SELECT sa.*, a.name as actor_name, a.photo_path
                FROM series_actors sa
                LEFT JOIN actors a ON sa.actor_id = a.actor_id
                WHERE sa.series_id = ?
                ORDER BY sa.role_order, sa.id
            """, (series_id,)).fetchall()

            # 获取该剧集的所有视频
            videos = conn.execute("SELECT video_id, filename FROM videos WHERE series_id = ?", (series_id,)).fetchall()

            video_ids = [v['video_id'] for v in videos]

            # 为每个演员统计样本信息
            result = []
            for actor in actors:
                actor_id = actor['actor_id']

                # 统计该演员在该剧所有视频中的样本数
                if video_ids:
                    placeholders = ','.join(['?' for _ in video_ids])
                    sample_stats = conn.execute(f"""
                        SELECT
                            COUNT(*) as total_samples,
                            COUNT(DISTINCT video_id) as videos_appeared,
                            MIN(timestamp) as first_appearance,
                            MAX(timestamp) as last_appearance
                        FROM face_samples
                        WHERE video_id IN ({placeholders})
                        AND character_id = ?
                    """, video_ids + [actor_id]).fetchone()
                else:
                    sample_stats = {'total_samples': 0, 'videos_appeared': 0, 'first_appearance': None, 'last_appearance': None}

                result.append({
                    'actor_id': actor_id,
                    'actor_name': actor['actor_name'],
                    'photo_path': actor['photo_path'],
                    'character_name': actor['character_name'],
                    'role_order': actor['role_order'],
                    'is_main_character': bool(actor['is_main_character']),
                    'total_samples': sample_stats['total_samples'] if sample_stats else 0,
                    'videos_appeared': sample_stats['videos_appeared'] if sample_stats else 0,
                    'first_appearance': float(sample_stats['first_appearance']) if sample_stats and sample_stats['first_appearance'] else None,
                    'last_appearance': float(sample_stats['last_appearance']) if sample_stats and sample_stats['last_appearance'] else None,
                })

            return {
                'series_id': series_id,
                'series_name': series['name'],
                'series_year': series['year'],
                'description': series['description'],
                'total_actors': len(result),
                'total_videos': len(videos),
                'characters': result
            }
    except HTTPException:
        raise
    except Exception as e:
        return {'error': str(e), 'characters': []}


@app.get("/api/series/{series_id}/faces")
async def get_series_face_samples(series_id: str, actor_id: str = None, limit: int = 100):
    """获取剧集的人脸样本（按演员聚合）"""
    try:
        with database.get_connection() as conn:
            # 获取该剧集的所有视频
            videos = conn.execute("SELECT video_id, filename FROM videos WHERE series_id = ?", (series_id,)).fetchall()
            video_ids = [v['video_id'] for v in videos]

            if not video_ids:
                return {'samples': [], 'error': '该剧集暂无视频'}

            placeholders = ','.join(['?' for _ in video_ids])

            # 构建查询条件
            where_clause = f"WHERE fs.video_id IN ({placeholders})"
            params = video_ids[:]

            if actor_id:
                where_clause += " AND fs.character_id = ?"
                params.append(actor_id)

            # 添加限制
            params.append(limit)

            # 查询人脸样本
            samples = conn.execute(f"""
                SELECT
                    fs.sample_id,
                    fs.video_id as video_id,
                    fs.character_id,
                    fs.image_path,
                    fs.timestamp,
                    fs.frame_number,
                    fs.quality_score,
                    v.filename as video_filename
                FROM face_samples fs
                LEFT JOIN videos v ON fs.video_id = v.video_id
                {where_clause}
                ORDER BY fs.quality_score DESC
                LIMIT ?
            """, params).fetchall()

            result = []
            for sample in samples:
                # 转换图片路径为相对路径供API访问
                image_path = sample['image_path']
                if image_path:
                    # 提取相对路径用于API
                    rel_path = image_path.split('/processed/faces/')[-1] if '/processed/faces/' in image_path else image_path
                    api_path = f"/api/face_images/{rel_path}"
                else:
                    api_path = None

                result.append({
                    'sample_id': sample['sample_id'],
                    'video_id': sample['video_id'],
                    'video_filename': sample['video_filename'],
                    'character_id': sample['character_id'],
                    'image_path': api_path,
                    'timestamp': float(sample['timestamp']) if sample['timestamp'] else None,
                    'frame_number': sample['frame_number'],
                    'quality_score': sample['quality_score'],
                })

            return {'samples': result}
    except Exception as e:
        return {'samples': [], 'error': str(e)}


# ========== 演员管理 API ==========

@app.get("/api/actors")
async def get_actors_api():
    """获取所有演员列表"""
    try:
        actors = database.get_actors()
        return {'actors': actors}
    except Exception as e:
        return {'actors': [], 'error': str(e)}


@app.post("/api/actors")
async def create_actor(request: Dict[str, Any]):
    """创建演员"""
    try:
        actor_id = request.get('actor_id') or f"actor_{int(hash(str(request)) % 10000)}"
        success = database.create_actor(
            actor_id=actor_id,
            name=request.get('name', ''),
            photo_path=request.get('photo_path')
        )
        if success:
            return {'success': True, 'actor_id': actor_id}
        return {'success': False, 'error': '创建失败'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@app.post("/api/actors/{actor_id}/photo")
async def upload_actor_photo(actor_id: str, file: UploadFile = File(...)):
    """上传演员照片"""
    try:
        # 创建演员照片目录
        actors_dir = DATA_ROOT / "actors"
        actors_dir.mkdir(parents=True, exist_ok=True)

        # 保存照片
        file_ext = Path(file.filename).suffix or '.jpg'
        photo_path = actors_dir / f"{actor_id}{file_ext}"

        with open(photo_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        # 提取人脸embedding
        import cv2
        import numpy as np

        img = cv2.imread(str(photo_path))
        if img is not None and embedder:
            faces = detector.detect(img)
            if faces:
                embedding = embedder.compute_embedding(img, faces[0])
                database.update_actor_embedding(actor_id, embedding)

        # 更新演员照片路径
        import sqlite3
        with database.get_connection() as conn:
            conn.execute(
                "UPDATE actors SET photo_path = ? WHERE actor_id = ?",
                (str(photo_path), actor_id)
            )

        return {'success': True, 'photo_path': str(photo_path)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


@app.delete("/api/actors/{actor_id}")
async def delete_actor(actor_id: str):
    """删除演员"""
    try:
        success = database.delete_actor(actor_id)
        return {'success': success}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@app.get("/api/actor_photos/{filename}")
async def get_actor_photo(filename: str):
    """获取演员照片"""
    try:
        actors_dir = DATA_ROOT / "actor_photos"
        image_path = actors_dir / filename
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="照片不存在")
        return FileResponse(str(image_path), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/processed/faces/{video_id}/{filename}")
async def get_face_sample_photo(video_id: str, filename: str):
    """获取人脸样本照片"""
    try:
        faces_dir = DATA_ROOT / "processed" / "faces" / video_id
        image_path = faces_dir / filename
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="照片不存在")
        return FileResponse(str(image_path), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/face_images/{path:path}")
async def get_face_image(path: str):
    """获取人脸样本图片"""
    try:
        # 路径格式: video_name/frame_xxx_face_y.jpg
        faces_dir = DATA_ROOT / "processed" / "faces"
        image_path = faces_dir / path
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="图片不存在")
        return FileResponse(str(image_path), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/series/{series_id}/recluster")
async def recluster_series(series_id: str, request: Dict[str, Any]):
    """重新匹配剧集的人脸聚类到演员"""
    try:
        import numpy as np
        from src.core.face_embedder import FaceEmbedder

        video_id = request.get('video_id')
        if not video_id:
            return {'success': False, 'error': '需要指定 video_id'}

        with database.get_connection() as conn:
            # 1. 获取剧集的所有演员
            actors = conn.execute("""
                SELECT sa.*, a.name as actor_name, a.embedding as actor_embedding
                FROM series_actors sa
                LEFT JOIN actors a ON sa.actor_id = a.actor_id
                WHERE sa.series_id = ?
                ORDER BY sa.role_order, sa.id
            """, (series_id,)).fetchall()

            valid_actors = []
            for actor in actors:
                if actor['actor_embedding']:
                    try:
                        import pickle
                        actor_emb = pickle.loads(actor['actor_embedding'])
                        actor['actor_embedding'] = actor_emb
                        valid_actors.append(actor)
                    except Exception as e:
                        print(f"Warning: Failed to load embedding for actor {actor['actor_name']}: {e}")
                        continue

            if not valid_actors:
                return {'success': False, 'error': '该剧集没有演员特征数据，请先上传演员照片'}

            # 2. 获取视频的所有聚类结果
            samples = conn.execute("""
                SELECT sample_id, cluster_id, embedding, quality_score
                FROM face_samples
                WHERE video_id = ? AND cluster_id IS NOT NULL AND embedding IS NOT NULL
                ORDER BY cluster_id, quality_score DESC
            """, (video_id,)).fetchall()

            if not samples:
                return {'success': False, 'error': '该视频没有聚类结果'}

            # 3. 按簇分组
            clusters = {}
            for sample in samples:
                cluster_id = sample['cluster_id']
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(sample)

            # 4. 匹配每个簇到演员
            results = []
            matched_actor_ids = set()

            for cluster_id in sorted(clusters.keys()):
                cluster_samples = clusters[cluster_id]

                # 计算簇的平均embedding（使用高质量样本）
                sorted_samples = sorted(cluster_samples, key=lambda s: s['quality_score'], reverse=True)
                top_n = max(5, len(sorted_samples) // 2)
                top_samples = sorted_samples[:top_n]

                embeddings = []
                for s in top_samples:
                    emb = s['embedding']
                    if emb:
                        import pickle
                        embeddings.append(pickle.loads(emb))

                if not embeddings:
                    continue

                cluster_embedding = np.mean(embeddings, axis=0)

                # 匹配演员
                best_match = None
                best_similarity = 0

                for actor in valid_actors:
                    actor_emb = actor['actor_embedding']

                    # 计算余弦相似度
                    dot = np.dot(cluster_embedding, actor_emb)
                    norm_cluster = np.linalg.norm(cluster_embedding)
                    norm_actor = np.linalg.norm(actor_emb)

                    if norm_cluster == 0 or norm_actor == 0:
                        similarity = 0
                    else:
                        similarity = float(dot / (norm_cluster * norm_actor))

                    if similarity >= 0.45 and similarity > best_similarity and actor['actor_id'] not in matched_actor_ids:
                        best_similarity = similarity
                        best_match = actor

                # 保存匹配结果
                if best_match:
                    matched_actor_ids.add(best_match['actor_id'])

                    # 更新所有样本的character_id
                    character_id = best_match['actor_id']
                    for sample in cluster_samples:
                        conn.execute("""
                            UPDATE face_samples
                            SET character_id = ?
                            WHERE sample_id = ?
                        """, (character_id, sample['sample_id']))

                    results.append({
                        'cluster_id': cluster_id,
                        'actor_id': best_match['actor_id'],
                        'actor_name': best_match['actor_name'],
                        'character_name': best_match['character_name'],
                        'sample_count': len(cluster_samples),
                        'similarity': best_similarity
                    })

            # 更新characters_found统计
            conn.execute("""
                UPDATE videos
                SET characters_found = ?
                WHERE video_id = ?
            """, (len(matched_actor_ids), video_id))

            return {
                'success': True,
                'matched_clusters': len(results),
                'total_clusters': len(clusters),
                'results': results
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


# ========== 静态文件 API ==========


if __name__ == "__main__":
    import uvicorn
    print("启动API服务器 http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
