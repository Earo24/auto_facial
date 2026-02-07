"""
影视人脸识别自动化系统 - 主应用
Streamlit多页面应用
"""
import streamlit as st
from streamlit_option_menu import option_menu
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.ui.clustering_ui import render_clustering_ui
from src.ui.recognition_ui import render_recognition_ui
from src.ui.analysis_ui import render_analysis_ui


def render_home():
    """渲染首页"""
    st.title("🎬 影视人脸识别自动化系统")
    st.markdown("---")

    st.markdown("""
    ## 系统功能

    本系统提供以下功能：

    ### 1. 视频处理
    - 自动从视频中提取关键帧
    - 检测人脸并提取特征
    - 智能采样（场景变化检测）

    ### 2. 人脸聚类
    - 自动发现视频中的角色
    - 三阶段聚类策略（DBSCAN -> 层次聚类 -> 合并小簇）
    - 支持人工审核和调整

    ### 3. 角色标注
    - 为聚类结果命名角色
    - 管理角色库
    - 合并、拆分簇

    ### 4. 人脸识别
    - 基于角色库批量识别
    - 时序平滑减少误识别
    - 处理造型变化

    ### 5. 分析报告
    - 角色出镜时长统计
    - 同框分析
    - 质量评估报告

    ## 使用流程

    1. **上传视频**: 在"视频处理"页面上传并处理视频
    2. **自动聚类**: 系统自动对人脸进行聚类
    3. **角色标注**: 在"聚类标注"页面审核并命名角色
    4. **批量识别**: 运行批量识别获取完整结果
    5. **查看报告**: 在"分析报告"页面查看统计信息

    ## 技术栈

    - **人脸检测**: InsightFace (RetinaFace)
    - **特征提取**: InsightFace (ArcFace)
    - **聚类算法**: DBSCAN + Agglomerative Clustering
    - **Web界面**: Streamlit
    - **数据存储**: SQLite
    """)

    # 显示系统状态
    st.markdown("---")
    st.subheader("系统状态")

    from config.settings import DATABASE_PATH

    if DATABASE_PATH.exists():
        st.success("✅ 数据库已初始化")

        from src.storage.database import Database
        db = Database()

        conn = db.get_connection().__enter__()
        video_count = conn.execute("SELECT COUNT(*) as count FROM videos").fetchone()['count']
        face_count = conn.execute("SELECT COUNT(*) as count FROM face_samples").fetchone()['count']
        char_count = conn.execute("SELECT COUNT(*) as count FROM characters").fetchone()['count']
        conn.__exit__(None, None, None)

        col1, col2, col3 = st.columns(3)
        col1.metric("处理视频", video_count)
        col2.metric("人脸样本", face_count)
        col3.metric("已标注角色", char_count)
    else:
        st.info("📊 数据库尚未初始化")


def render_video_processing():
    """渲染视频处理页面"""
    st.title("📹 视频处理")
    st.markdown("---")

    st.info("此功能正在开发中，请使用命令行工具处理视频")

    st.markdown("""
    ### 命令行使用方法

    ```bash
    # 处理视频（检测人脸）
    python scripts/process_video.py /path/to/video.mp4

    # 运行聚类
    python scripts/cluster_faces.py video_id

    # 运行识别
    python scripts/recognize.py video_id
    ```
    """)


def main():
    """主应用入口"""
    st.set_page_config(
        page_title="影视人脸识别系统",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 侧边栏导航
    with st.sidebar:
        st.title("🎬 影视人脸识别")
        st.markdown("---")

        page = option_menu(
            menu_title="导航菜单",
            options=["首页", "视频处理", "聚类标注", "识别结果", "分析报告"],
            icons=["house", "camera-video", "people", "search", "bar-chart"],
            menu_icon="cast",
            default_index=0,
        )

        st.markdown("---")
        st.markdown("""
        ### 快速帮助

        1. 首先在"视频处理"中上传视频
        2. 然后在"聚类标注"中命名角色
        3. 最后在"识别结果"中查看结果
        """)

    # 渲染选中的页面
    if page == "首页":
        render_home()
    elif page == "视频处理":
        render_video_processing()
    elif page == "聚类标注":
        render_clustering_ui()
    elif page == "识别结果":
        render_recognition_ui()
    elif page == "分析报告":
        render_analysis_ui()


if __name__ == "__main__":
    main()
