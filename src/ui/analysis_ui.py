"""
分析报告界面
显示角色出镜时长、同框分析等
"""
import streamlit as st
import numpy as np
from typing import List, Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.settings import DATABASE_PATH
from src.storage.database import Database
from src.storage.character_store import CharacterStore
from src.models.character import CharacterLibrary


def render_analysis_ui():
    """渲染分析报告界面"""
    st.title("📊 角色分析报告")
    st.markdown("---")

    database = Database()

    # 视频选择
    st.subheader("选择视频")
    conn = database.get_connection().__enter__()
    videos = conn.execute("SELECT video_id, filename FROM videos").fetchall()
    conn.__exit__(None, None, None)

    if not videos:
        st.info("暂无视频数据")
        return

    video_options = {v['filename']: v['video_id'] for v in videos}
    selected = st.selectbox("选择视频", options=list(video_options.keys()))

    if not selected:
        return

    video_id = video_options[selected]

    # 加载角色库
    character_store = CharacterStore(DATABASE_PATH.parent / "characters", database)
    library = character_store.load_library(video_id)

    if not library:
        library = character_store.load_library_from_db(video_id)

    if not library or not library.characters:
        st.info("暂无角色数据")
        return

    # 渲染统计卡片
    render_statistics_cards(library)

    # 角色出镜时长
    render_screen_time_analysis(library)

    # 角色时间线
    render_timeline_analysis(library)

    # 角色质量分析
    render_quality_analysis(library)


def render_statistics_cards(library: CharacterLibrary):
    """渲染统计卡片"""
    st.markdown("### 总览")

    total_samples = sum(char.sample_count for char in library.characters)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("角色数量", len(library.characters))
    col2.metric("总样本数", total_samples)
    col3.metric("平均质量", f"{np.mean([char.statistics.avg_quality for char in library.characters]):.3f}")
    col4.metric("视频时长", f"{library.characters[0].statistics.last_appearance:.1f}s" if library.characters else "0s")


def render_screen_time_analysis(library: CharacterLibrary):
    """渲染出镜时长分析"""
    st.markdown("### 角色出镜时长")

    # 准备数据
    data = []
    for char in library.characters:
        data.append({
            '角色': char.name,
            '样本数': char.sample_count,
            '出镜时长': char.statistics.appearance_duration,
            '平均质量': char.statistics.avg_quality,
        })

    df = pd.DataFrame(data)
    df = df.sort_values('样本数', ascending=False)

    # 显示表格
    st.dataframe(df, use_container_width=True)

    # 绘制条形图
    fig = px.bar(df, x='角色', y='样本数', title='角色样本数分布',
                 color='平均质量', color_continuous_scale='viridis')
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def render_timeline_analysis(library: CharacterLibrary):
    """渲染时间线分析"""
    st.markdown("### 角色时间线")

    # 创建时间线数据
    fig = go.Figure()

    for char in library.characters:
        if char.sample_count > 0:
            fig.add_trace(go.Scatter(
                x=[char.statistics.first_appearance, char.statistics.last_appearance],
                y=[char.name, char.name],
                mode='lines+markers',
                name=char.name,
                line=dict(width=10),
            ))

    fig.update_layout(
        title='角色出现时间线',
        xaxis_title='时间 (秒)',
        yaxis_title='角色',
        height=max(400, len(library.characters) * 30),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_quality_analysis(library: CharacterLibrary):
    """渲染质量分析"""
    st.markdown("### 人脸质量分析")

    # 准备数据
    all_qualities = []
    for char in library.characters:
        all_qualities.extend([
            {'角色': char.name, '质量': s.get('quality', 0)}
            for s in char.samples
            if 'quality' in s
        ])

    if not all_qualities:
        st.info("暂无质量数据")
        return

    df = pd.DataFrame(all_qualities)

    # 质量分布直方图
    fig = px.histogram(df, x='质量', color='角色',
                       title='人脸质量分布',
                       nbins=50,
                       barmode='overlay')
    st.plotly_chart(fig, use_container_width=True)

    # 箱线图
    fig = px.box(df, x='角色', y='质量', title='角色质量对比')
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render_analysis_ui()
