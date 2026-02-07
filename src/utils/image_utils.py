"""
图像处理工具函数
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List


def calculate_image_quality(image: np.ndarray, face_bbox: List[float]) -> float:
    """
    计算人脸图像的质量分数

    Args:
        image: 输入图像
        face_bbox: 人脸边界框 [x1, y1, x2, y2]

    Returns:
        质量分数 (0-1)
    """
    x1, y1, x2, y2 = [int(coord) for coord in face_bbox]
    face_img = image[y1:y2, x1:x2]

    if face_img.size == 0:
        return 0.0

    scores = []

    # 1. 清晰度评分（拉普拉斯方差）
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 500.0, 1.0)
    scores.append(sharpness_score)

    # 2. 亮度评分
    brightness = np.mean(gray)
    if brightness < 20:
        brightness_score = brightness / 20.0
    elif brightness > 220:
        brightness_score = (255 - brightness) / 35.0
    else:
        brightness_score = 1.0
    scores.append(brightness_score)

    # 3. 对比度评分
    contrast = gray.std()
    contrast_score = min(contrast / 100.0, 1.0)
    scores.append(contrast_score)

    # 4. 人脸尺寸评分
    face_area = (x2 - x1) * (y2 - y1)
    size_score = min(face_area / 10000.0, 1.0)
    scores.append(size_score)

    # 综合评分
    return sum(scores) / len(scores)


def normalize_face(image: np.ndarray, landmarks: Optional[np.ndarray] = None,
                   target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    归一化人脸图像

    Args:
        image: 输入人脸图像
        landmarks: 5点关键点 (可选)
        target_size: 目标尺寸

    Returns:
        归一化后的人脸图像
    """
    # 调整大小
    if image.shape[:2] != target_size:
        face = cv2.resize(image, target_size)
    else:
        face = image.copy()

    # 归一化像素值
    face = face.astype(np.float32) / 255.0

    # 标准化（ImageNet均值）
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    face = (face - mean) / std

    return face


def align_face(image: np.ndarray, landmarks: np.ndarray,
               target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    基于5点关键点对齐人脸

    Args:
        image: 输入图像
        landmarks: 5点关键点 [[x,y], ...]
        target_size: 目标尺寸

    Returns:
        对齐后的人脸图像
    """
    # InsightFace的5点关键点顺序：左眼、右眼、鼻尖、左嘴角、右嘴角
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]

    # 计算眼睛角度
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # 计算旋转中心（两眼中心）
    eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                   (left_eye[1] + right_eye[1]) / 2)

    # 旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    # 旋转图像
    rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_CUBIC)

    # 裁剪人脸区域
    face_width = right_eye[0] - left_eye[0]
    face_height = face_width * 1.2

    x1 = int(max(0, eyes_center[0] - face_width * 0.8))
    y1 = int(max(0, eyes_center[1] - face_height * 0.5))
    x2 = int(min(image.shape[1], eyes_center[0] + face_width * 0.8))
    y2 = int(min(image.shape[0], eyes_center[1] + face_height * 0.7))

    face = rotated[y1:y2, x1:x2]

    if face.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # 调整到目标尺寸
    face = cv2.resize(face, target_size)

    return face


def extract_face_region(image: np.ndarray, bbox: List[float],
                        margin: float = 0.2) -> np.ndarray:
    """
    提取人脸区域（带边距）

    Args:
        image: 输入图像
        bbox: 边界框 [x1, y1, x2, y2]
        margin: 边距比例

    Returns:
        人脸区域图像
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]

    # 计算边距
    width = x2 - x1
    height = y2 - y1
    margin_x = int(width * margin)
    margin_y = int(height * margin)

    # 添加边距
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(image.shape[1], x2 + margin_x)
    y2 = min(image.shape[0], y2 + margin_y)

    face = image[y1:y2, x1:x2]

    return face


def resize_with_aspect_ratio(image: np.ndarray, max_size: int = 800) -> np.ndarray:
    """
    按比例缩放图像

    Args:
        image: 输入图像
        max_size: 最大尺寸

    Returns:
        缩放后的图像
    """
    h, w = image.shape[:2]

    if max(h, w) <= max_size:
        return image

    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)

    return cv2.resize(image, (new_w, new_h))


def draw_face_detection(image: np.ndarray, bboxes: List[List[float]],
                        landmarks: Optional[List[np.ndarray]] = None,
                        names: Optional[List[str]] = None,
                        scores: Optional[List[float]] = None) -> np.ndarray:
    """
    在图像上绘制人脸检测结果

    Args:
        image: 输入图像
        bboxes: 边界框列表
        landmarks: 关键点列表（可选）
        names: 名称列表（可选）
        scores: 置信度列表（可选）

    Returns:
        绘制后的图像
    """
    result = image.copy()

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        # 绘制边界框
        color = (0, 255, 0)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # 绘制关键点
        if landmarks and i < len(landmarks):
            for point in landmarks[i]:
                cv2.circle(result, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

        # 绘制标签
        label = ""
        if names and i < len(names):
            label = names[i]
        if scores and i < len(scores):
            if label:
                label += f" ({scores[i]:.2f})"
            else:
                label = f"{scores[i]:.2f}"

        if label:
            # 获取文本大小
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # 绘制背景
            cv2.rectangle(result, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)

            # 绘制文本
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result
