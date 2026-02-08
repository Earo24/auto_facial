#!/usr/bin/env python3
"""
AutoFacial 自动演示视频录制
使用Playwright自动化演示 + ffmpeg录制屏幕
"""
from playwright.sync_api import sync_playwright
import time
import subprocess
import signal
import os
import sys

# 演示脚本配置
DEMO_CONFIG = {
    'output': '/tmp/auto_facial_demo.mp4',
    'duration': 120,  # 2分钟
    'fps': 30,
    'width': 1920,
    'height': 1080
}

def start_recording(config):
    """使用ffmpeg开始录制屏幕"""
    output = config['output']

    # ffmpeg命令（macOS）
    cmd = [
        'ffmpeg',
        '-f', 'avfoundation',
        '-framerate', str(config['fps']),
        '-i', '1',  # 屏幕设备
        '-r', str(config['fps']),
        '-t', str(config['duration']),
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        '-preset', 'slow',
        output
    ]

    print(f"开始录制屏幕: {output}")
    print(f"录制时长: {config['duration']}秒")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def run_demo():
    """运行自动演示流程"""

    print("\n=== AutoFacial 自动演示开始 ===\n")

    with sync_playwright() as p:
        # 启动浏览器（窗口大小固定）
        browser = p.chromium.launch(
            headless=False,
            args=['--window-size=1600,900']
        )
        context = browser.new_context(viewport={'width': 1600, 'height': 900})
        page = context.new_page()

        # 阶段1: 仪表板 (20秒)
        print("[1/6] 仪表板展示 (20秒)")
        page.goto('http://localhost:3001')
        page.wait_for_load_state('networkidle')
        page.screenshot(path='/tmp/demo_01_dashboard.png')
        time.sleep(3)  # 展示3秒

        # 滚动查看更多信息
        page.evaluate('window.scrollBy(0, 300)')
        time.sleep(2)
        page.evaluate('window.scrollBy(0, -300)')

        # 阶段2: 视频列表 (15秒)
        print("[2/6] 视频列表 (15秒)")
        page.click('a:has-text("视频列表")')
        page.wait_for_load_state('networkidle')
        page.screenshot(path='/tmp/demo_02_videos.png')
        time.sleep(3)

        # 阶段3: 进入聚类页面 (25秒)
        print("[3/6] 聚类标注页面 (25秒)")
        page.goto('http://localhost:3001/clustering?video=老舅14集_片段')
        page.wait_for_load_state('networkidle')
        page.screenshot(path='/tmp/demo_03_clustering.png')
        time.sleep(3)

        # 展示聚类列表
        page.evaluate('window.scrollBy(0, 400)')
        time.sleep(2)

        # 选择第一个簇
        print("  - 选择簇查看详情...")
        page.click('div:has-text("张海宇")')
        time.sleep(4)

        # 阶段4: 查看人脸样本 (20秒)
        print("[4/6] 簇详情和样本 (20秒)")
        page.screenshot(path='/tmp/demo_04_cluster_detail.png')
        page.evaluate('window.scrollBy(0, 300)')
        time.sleep(3)

        # 阶段5: 剧集管理 (15秒)
        print("[5/6] 剧集管理 (15秒)")
        page.click('a:has-text("剧集管理")')
        page.wait_for_load_state('networkidle')
        page.screenshot(path='/tmp/demo_05_series.png')
        time.sleep(3)

        # 阶段6: 视频处理 (15秒)
        print("[6/6] 视频处理页面 (15秒)")
        page.click('a:has-text("视频处理")')
        page.wait_for_load_state('networkidle')
        page.screenshot(path='/tmp/demo_06_processing.png')
        time.sleep(3)

        # 结束
        browser.close()

    print("\n=== 演示完成 ===")
    print(f"截图已保存到 /tmp/demo_*.png")

def main():
    """主函数"""
    import fcntl
    import termios

    print("=" * 60)
    print("AutoFacial 自动演示录制")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  输出文件: {DEMO_CONFIG['output']}")
    print(f"  录制时长: {DEMO_CONFIG['duration']}秒")
    print(f"  分辨率: {DEMO_CONFIG['width']}x{DEMO_CONFIG['height']}")

    print("\n请确保:")
    print("  1. 后端服务运行在 http://localhost:8000")
    print("  2. 前端服务运行在 http://localhost:3001")
    print("  3. 已安装 ffmpeg")

    print("\n录屏选项:")
    print("  1. 自动录制 (需要ffmpeg)")
    print("  2. 仅演示 (手动录制)")

    choice = input("\n请选择 (1/2): ").strip()

    if choice == '1':
        # 自动录制
        try:
            # 启动ffmpeg录制
            record_process = start_recording(DEMO_CONFIG)

            # 等待2秒让录制开始
            time.sleep(2)

            # 运行演示
            run_demo()

            # 等待录制完成
            print("\n等待录制完成...")
            record_process.wait()

            print(f"\n✓ 视频已保存: {DEMO_CONFIG['output']}")

        except KeyboardInterrupt:
            print("\n\n录制已中断")
            record_process.terminate()
        except FileNotFoundError:
            print("\n错误: 未找到ffmpeg，请先安装:")
            print("  brew install ffmpeg")
    else:
        # 仅演示
        print("\n请手动开始屏幕录制，然后按回车继续...")
        input()
        run_demo()
        print("\n请停止屏幕录制!")

if __name__ == "__main__":
    main()
