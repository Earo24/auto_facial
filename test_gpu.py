"""
测试M2 GPU加速是否启用
"""
import logging
logging.basicConfig(level=logging.INFO)

print("=" * 50)
print("测试 M2 GPU 加速")
print("=" * 50)

# 检查ONNX Runtime
import onnxruntime as ort
print(f"\n1. ONNX Runtime 版本: {ort.__version__}")
print(f"   可用执行提供器: {ort.get_available_providers()}")

# 测试FaceDetector
print("\n2. 测试人脸检测器初始化...")
from src.core.face_detector import FaceDetector

detector = FaceDetector(use_gpu=True)
detector.initialize()
print(f"   检测器已初始化")
print(f"   使用的执行提供器: {detector.detector.session.providers if hasattr(detector.detector, 'session') else 'N/A'}")

# 测试FaceEmbedder
print("\n3. 测试特征提取器初始化...")
from src.core.face_embedder import FaceEmbedder

embedder = FaceEmbedder(use_gpu=True)
embedder.initialize()
print(f"   特征提取器已初始化")

# 简单性能测试
print("\n4. 性能测试...")
import cv2
import numpy as np
import time

# 创建测试图像
test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# 测试人脸检测
start = time.time()
faces = detector.detect_faces(test_image, frame_id="test")
elapsed = time.time() - start
print(f"   人脸检测耗时: {elapsed:.3f}秒")
print(f"   检测到人脸数: {len(faces)}")

print("\n" + "=" * 50)
print("✅ GPU测试完成！")
print("=" * 50)
