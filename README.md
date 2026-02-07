# 影视人脸识别自动化系统

一个半自动的影视人脸识别系统，用于电影/电视剧的角色识别和分析。系统自动从视频中提取人脸并进行聚类，人工只需审核和调整角色标注。

## 功能特点

- **智能视频采样**: 结合场景变化检测的智能帧采样
- **自动人脸聚类**: 三阶段聚类策略自动发现角色
- **可视化标注**: Streamlit界面进行角色命名和标注
- **批量识别**: 基于角色库的批量人脸识别
- **分析报告**: 角色出镜时长、同框分析等

## 技术栈

| 组件 | 选择 |
|------|------|
| 人脸检测 | InsightFace (RetinaFace) |
| 特征提取 | InsightFace (ArcFace) |
| 聚类算法 | scikit-learn (DBSCAN + Agglomerative) |
| Web界面 | Streamlit |
| 数据存储 | SQLite |

## 安装

### 1. 克隆项目

```bash
git clone <repository_url>
cd auto_facial
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型

首次运行时，InsightFace会自动下载所需模型到 `models/` 目录。

## 使用方法

### 方式一：命令行

#### 1. 处理视频

```bash
python scripts/process_video.py /path/to/video.mp4
```

#### 2. 聚类人脸

```bash
python scripts/cluster_faces.py <video_id>
```

#### 3. 运行识别

```bash
python scripts/recognize.py <video_id>
```

### 方式二：Web界面

#### 启动应用

```bash
streamlit run app.py
```

然后在浏览器中打开 `http://localhost:8501`

#### 使用流程

1. **视频处理**: 上传并处理视频（自动检测人脸）
2. **聚类标注**: 审核自动聚类结果，命名角色
3. **识别结果**: 查看批量识别结果
4. **分析报告**: 查看角色统计和分析

## 项目结构

```
auto_facial/
├── config/                 # 配置文件
├── src/
│   ├── core/              # 核心业务逻辑
│   ├── models/            # 数据模型
│   ├── storage/           # 数据存储层
│   ├── utils/             # 工具函数
│   └── ui/                # 用户界面
├── data/                  # 数据目录
├── scripts/               # 命令行脚本
├── tests/                 # 测试代码
└── app.py                 # 主应用入口
```

## 配置

编辑 `config/settings.py` 修改配置参数：

- `VIDEO_CONFIG`: 视频处理参数
- `FACE_DETECTION_CONFIG`: 人脸检测参数
- `CLUSTERING_CONFIG`: 聚类参数
- `RECOGNITION_CONFIG`: 识别参数

## 许可证

MIT License
