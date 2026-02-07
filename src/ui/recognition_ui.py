"""
识别结果界面
显示人脸识别结果
"""
import streamlit as st
import cv2
from pathlib import Path
from typing import List, Dict
import numpy as np

from config.settings import DATABASE_PATH
from src.storage.database import Database
from src.core.recognition_engine import RecognitionResult


def render_recognition_ui():
    """渲染识别结果界面"""
    st.title("🔍 人脸识别结果")
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

    # 渲染识别结果
    render_recognition_results(database, video_id)


def render_recognition_results(database: Database, video_id: str):
    """渲染识别结果"""
    # 获取识别结果
    conn = database.get_connection().__enter__()
    results = conn.execute("""
        SELECT rr.*, c.name as character_name
        FROM recognition_results rr
        LEFT JOIN characters c ON rr.character_id = c.character_id
        WHERE rr.video_id = ?
        ORDER BY rr.timestamp
    """, (video_id,)).fetchall()
    conn.__exit__(None, None, None)

    if not results:
        st.info("暂无识别结果，请先运行识别")
        return

    # 统计信息
    total = len(results)
    high_conf = sum(1 for r in results if r['confidence'] and r['confidence'] > 0.5)
    known = sum(1 for r in results if r['character_id'])

    col1, col2, col3 = st.columns(3)
    col1.metric("总检测数", total)
    col2.metric("已识别", known)
    col3.metric("高置信度", high_conf)

    # 按角色统计
    char_stats = {}
    for r in results:
        char_name = r['character_name'] or '未知'
        char_stats[char_name] = char_stats.get(char_name, 0) + 1

    st.markdown("### 角色识别统计")
    st.bar_chart(char_stats)

    # 显示结果列表
    st.markdown("### 识别结果详情")

    # 分页显示
    page_size = 20
    total_pages = (len(results) + page_size - 1) // page_size

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.number_input("页码", min_value=1, max_value=total_pages, value=1)

    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(results))

    # 显示结果
    for i in range(start_idx, end_idx):
        r = results[i]

        with st.expander(f"[{r['timestamp']:.2f}s] {r['character_name'] or '未知'} - 置信度: {r['confidence']:.2f}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**帧ID**: {r['frame_id']}")
                st.write(f"**时间**: {r['timestamp']:.2f}秒")
                st.write(f"**角色**: {r['character_name'] or '未知'}")
                st.write(f"**置信度**: {r['confidence']:.3f}")

            with col2:
                bbox = eval(r['bbox']) if isinstance(r['bbox'], str) else r['bbox']
                st.write(f"**位置**: ({bbox[0]:.0f}, {bbox[1]:.0f}) - ({bbox[2]:.0f}, {bbox[3]:.0f})")


if __name__ == "__main__":
    render_recognition_ui()
