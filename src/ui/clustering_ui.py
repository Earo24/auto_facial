"""
聚类标注界面
Streamlit界面用于审核和调整聚类结果
"""
import streamlit as st
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import cv2
from io import BytesIO

from config.settings import UI_CONFIG, CLUSTERING_CONFIG, CHARACTERS_DIR
from src.models.character import CharacterLibrary, Character
from src.models.face_sample import FaceCluster
from src.storage.database import Database
from src.storage.character_store import CharacterStore


def init_session_state():
    """初始化会话状态"""
    if 'current_video_id' not in st.session_state:
        st.session_state.current_video_id = None
    if 'clusters' not in st.session_state:
        st.session_state.clusters = []
    if 'samples' not in st.session_state:
        st.session_state.samples = []
    if 'character_library' not in st.session_state:
        st.session_state.character_library = None
    if 'selected_cluster_id' not in st.session_state:
        st.session_state.selected_cluster_id = None
    if 'editing_mode' not in st.session_state:
        st.session_state.editing_mode = False


def render_header():
    """渲染页面标题"""
    st.title("🎬 影视人脸识别系统 - 聚类标注")
    st.markdown("---")


def render_video_selector(database: Database):
    """渲染视频选择器"""
    st.subheader("选择视频")

    # 获取所有视频
    conn = database.get_connection().__enter__()
    videos = conn.execute("SELECT video_id, filename, processed_frames, detected_faces FROM videos").fetchall()
    conn.__exit__(None, None, None)

    if not videos:
        st.info("暂无视频数据，请先处理视频")
        return None

    video_options = {f"{v['filename']} ({v['video_id']})": v['video_id'] for v in videos}

    selected = st.selectbox("选择要标注的视频", options=list(video_options.keys()))

    if selected:
        video_id = video_options[selected]
        st.session_state.current_video_id = video_id

        # 显示视频信息
        video_info = database.get_video_info(video_id)
        if video_info:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("处理帧数", video_info.processed_frames)
            col2.metric("检测人脸", video_info.detected_faces)
            col3.metric("时长", f"{video_info.duration:.1f}秒")
            col4.metric("分辨率", video_info.format_resolution)

        return video_id

    return None


def load_clustering_data(database: Database, video_id: str):
    """加载聚类数据"""
    # 从数据库加载样本
    samples = database.get_face_samples(video_id)

    if not samples:
        st.warning("未找到人脸样本数据")
        return

    # 按簇分组
    clusters_dict: Dict[int, List] = {}
    unassigned = []

    for sample in samples:
        if sample.cluster_id is not None:
            if sample.cluster_id not in clusters_dict:
                clusters_dict[sample.cluster_id] = []
            clusters_dict[sample.cluster_id].append(sample)
        else:
            unassigned.append(sample)

    # 创建簇对象
    clusters = []
    for cluster_id, cluster_samples in clusters_dict.items():
        cluster = FaceCluster(cluster_id=cluster_id)
        for sample in cluster_samples:
            cluster.add_sample(sample)
        clusters.append(cluster)

    st.session_state.clusters = clusters
    st.session_state.samples = samples

    # 尝试加载角色库
    character_store = CharacterStore(CHARACTERS_DIR, database)
    library = character_store.load_library(video_id)
    if not library:
        library = character_store.load_library_from_db(video_id)

    st.session_state.character_library = library

    st.success(f"加载完成: {len(clusters)} 个簇, {len(samples)} 个样本")


def render_cluster_list():
    """渲染簇列表"""
    st.subheader("聚类结果")

    if not st.session_state.clusters:
        st.info("暂无聚类数据")
        return

    # 按列显示簇
    cols_per_row = 4
    clusters = st.session_state.clusters

    for i in range(0, len(clusters), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(clusters):
                cluster = clusters[idx]
                library = st.session_state.character_library

                # 获取角色名称
                char_name = f"簇 {cluster.cluster_id}"
                if library:
                    char = next((c for c in library.characters if c.character_id == f"char_{cluster.cluster_id:03d}"), None)
                    if char and char.name != f"角色 {cluster.cluster_id + 1}":
                        char_name = char.name

                with col:
                    # 显示统计信息
                    st.metric(char_name, f"{cluster.size} 样本", f"质量: {cluster.avg_quality:.2f}")

                    # 选择按钮
                    if st.button(f"查看详情", key=f"select_{cluster.cluster_id}"):
                        st.session_state.selected_cluster_id = cluster.cluster_id

                    # 编辑按钮
                    if library:
                        if st.button(f"编辑", key=f"edit_{cluster.cluster_id}"):
                            st.session_state.editing_mode = True
                            st.session_state.selected_cluster_id = cluster.cluster_id


def render_cluster_detail(database: Database):
    """渲染簇详情"""
    if st.session_state.selected_cluster_id is None:
        return

    cluster_id = st.session_state.selected_cluster_id
    cluster = next((c for c in st.session_state.clusters if c.cluster_id == cluster_id), None)

    if not cluster:
        st.warning(f"未找到簇 {cluster_id}")
        return

    st.markdown("---")
    st.subheader(f"簇 {cluster_id} 详情")

    # 显示统计信息
    col1, col2, col3 = st.columns(3)
    col1.metric("样本数量", cluster.size)
    col2.metric("平均质量", f"{cluster.avg_quality:.3f}")
    col3.metric("质量范围", f"{min(s.quality_score for s in cluster.samples):.2f} - {max(s.quality_score for s in cluster.samples):.2f}")

    # 显示时间范围
    time_range = (
        min(s.timestamp for s in cluster.samples),
        max(s.timestamp for s in cluster.samples)
    )
    st.metric("出现时间", f"{time_range[0]:.1f}s - {time_range[1]:.1f}s")

    # 显示样本网格
    st.markdown("### 人脸样本")

    # 获取高质量样本
    display_samples = cluster.get_high_quality_samples(min_quality=0.5, limit=UI_CONFIG['max_preview_samples'])

    # 按网格显示
    cols_per_row = 5
    for i in range(0, len(display_samples), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(display_samples):
                sample = display_samples[idx]
                with col:
                    # 显示图像
                    if sample.image_path and Path(sample.image_path).exists():
                        img = cv2.imread(sample.image_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, UI_CONFIG['thumbnail_size'])
                            st.image(img, use_column_width=True)

                    # 显示信息
                    st.caption(f"Q: {sample.quality_score:.2f} | T: {sample.timestamp:.1f}s")

                    # 移除按钮
                    if st.button("移除", key=f"remove_{sample.sample_id}"):
                        cluster.remove_sample(sample)
                        database.update_sample_cluster(sample.sample_id, None)
                        st.rerun()


def render_character_editing(database: Database):
    """渲染角色编辑界面"""
    if st.session_state.selected_cluster_id is None:
        return

    cluster_id = st.session_state.selected_cluster_id
    cluster = next((c for c in st.session_state.clusters if c.cluster_id == cluster_id), None)

    if not cluster:
        return

    st.markdown("---")
    st.subheader("角色标注")

    library = st.session_state.character_library
    if not library:
        library = CharacterLibrary(video_info={'video_id': st.session_state.current_video_id}, characters=[])

    # 查找或创建角色
    character_id = f"char_{cluster_id:03d}"
    character = next((c for c in library.characters if c.character_id == character_id), None)

    if not character:
        character = Character(
            character_id=character_id,
            name=f"角色 {cluster_id + 1}",
            video_id=st.session_state.current_video_id,
        )
        library.add_character(character)

    # 角色名称输入
    name = st.text_input("角色名称", value=character.name, key=f"name_{cluster_id}")

    # 别名
    aliases_str = st.text_input("别名 (逗号分隔)", value=",".join(character.aliases), key=f"aliases_{cluster_id}")

    # 描述
    description = st.text_area("角色描述", value=character.description, key=f"desc_{cluster_id}")

    # 保存按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("保存角色信息", key=f"save_char_{cluster_id}"):
            character.name = name
            character.aliases = [a.strip() for a in aliases_str.split(',') if a.strip()]
            character.description = description

            # 添加原型样本
            high_quality = cluster.get_high_quality_samples(min_quality=0.7, limit=5)
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

            # 保存角色
            database.save_character(character)

            # 保存角色库
            character_store = CharacterStore(CHARACTERS_DIR, database)
            character_store.save_library(library, st.session_state.current_video_id)

            st.session_state.character_library = library
            st.success("角色信息已保存")

    with col2:
        if st.button("返回", key=f"back_{cluster_id}"):
            st.session_state.editing_mode = False
            st.rerun()


def render_cluster_operations(database: Database):
    """渲染簇操作"""
    st.markdown("---")
    st.subheader("簇操作")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 合并簇")
        cluster_ids = [c.cluster_id for c in st.session_state.clusters]
        merge_from = st.selectbox("从簇", cluster_ids, key="merge_from")
        merge_to = st.selectbox("到簇", cluster_ids, key="merge_to")

        if st.button("合并") and merge_from != merge_to:
            from_cluster = next((c for c in st.session_state.clusters if c.cluster_id == merge_from), None)
            to_cluster = next((c for c in st.session_state.clusters if c.cluster_id == merge_to), None)

            if from_cluster and to_cluster:
                to_cluster.merge(from_cluster)
                st.session_state.clusters.remove(from_cluster)

                # 更新数据库
                for sample in to_cluster.samples:
                    database.update_sample_cluster(sample.sample_id, to_cluster.cluster_id)

                st.success(f"已合并簇 {merge_from} 到 {merge_to}")
                st.rerun()

    with col2:
        st.markdown("##### 保存角色库")
        if st.button("保存全部角色"):
            character_store = CharacterStore(CHARACTERS_DIR, database)
            library = character_store.create_library_from_clusters(
                st.session_state.current_video_id,
                st.session_state.clusters,
                st.session_state.samples
            )
            character_store.save_library(library, st.session_state.current_video_id)
            st.session_state.character_library = library
            st.success("角色库已保存")

    with col3:
        st.markdown("##### 导出")
        export_format = st.selectbox("格式", ["JSON", "CSV"])
        if st.button("导出角色库"):
            character_store = CharacterStore(CHARACTERS_DIR, database)
            library = st.session_state.character_library

            if not library:
                library = character_store.create_library_from_clusters(
                    st.session_state.current_video_id,
                    st.session_state.clusters,
                    st.session_state.samples
                )

            output_path = f"export_{st.session_state.current_video_id}.{export_format.lower()}"
            character_store.export_library(library, output_path, format=export_format.lower())
            st.success(f"已导出到 {output_path}")


def render_clustering_ui():
    """渲染聚类标注主界面"""
    init_session_state()
    render_header()

    # 初始化数据库
    database = Database()

    # 视频选择
    video_id = render_video_selector(database)

    if video_id:
        # 加载数据按钮
        if st.button("加载聚类数据", use_container_width=True):
            load_clustering_data(database, video_id)

        # 渲染界面
        if st.session_state.clusters:
            if st.session_state.editing_mode:
                render_character_editing(database)
            else:
                render_cluster_list()
                render_cluster_detail(database)
                render_cluster_operations(database)
        else:
            st.info("请点击上方按钮加载聚类数据")


if __name__ == "__main__":
    render_clustering_ui()
