"""
安装配置
"""
from setuptools import setup, find_packages

setup(
    name="auto-facial",
    version="0.1.0",
    description="影视人脸识别自动化系统",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.23.0',
        'opencv-python>=4.8.0',
        'insightface>=0.7.3',
        'onnxruntime>=1.15.0',
        'scikit-learn>=1.3.0',
        'streamlit>=1.28.0',
        'matplotlib>=3.7.0',
        'networkx>=3.1.0',
        'tqdm>=4.66.0',
        'pillow>=10.0.0',
        'pyyaml>=6.0',
        'pandas>=2.0.0',
        'sqlalchemy>=2.0.0',
        'plotly>=5.17.0',
        'streamlit-option-menu>=0.3.6',
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'auto-facial=app:main',
        ],
    },
)
