#!/usr/bin/env python3
"""
AutoFacial 自动演示录制（无需用户输入）
"""
from playwright.sync_api import sync_playwright
import time
import subprocess
import signal
import os

# 配置
OUTPUT_FILE = "/tmp/auto_facial_demo.mp4"
DURATION = 120  # 2分钟

def start_ffmpeg():
    """启动ffmpeg录制"""
    cmd = [
        'ffmpeg',
        '-f', 'avfoundation',
        '-framerate', '30',
        '-i', '1',  # 主屏幕
        '-r', '30',
        '-t', str(DURATION),
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        '-preset', 'fast',
        OUTPUT_FILE
    ]

    print(f"开始录制: {OUTPUT_FILE}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def run_demo():
    """运行演示流程"""
    print("开始演示...")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=['--window-size=1400,900']
        )
        page = browser.new_page()

        # 1. 仪表板 (20秒)
        print("1/6 仪表板")
        page.goto('http://localhost:3001')
        page.wait_for_load_state('networkidle')
        time.sleep(5)

        # 2. 视频列表 (15秒)
        print("2/6 视频列表")
        page.click('a:has-text("视频列表")')
        page.wait_for_load_state('networkidle')
        time.sleep(5)

        # 3. 聚类页面 (30秒)
        print("3/6 聚类标注")
        page.goto('http://localhost:3001/clustering?video=老舅14集_片段')
        page.wait_for_load_state('networkidle')
        time.sleep(5)

        # 选择簇
        page.click('div:has-text("张海宇")')
        time.sleep(10)

        # 4. 剧集管理 (20秒)
        print("4/6 剧集管理")
        page.click('a:has-text("剧集管理")')
        page.wait_for_load_state('networkidle')
        time.sleep(5)

        # 5. 视频处理 (15秒)
        print("5/6 视频处理")
        page.click('a:has-text("视频处理")')
        page.wait_for_load_state('networkidle')
        time.sleep(5)

        # 6. 返回仪表板 (15秒)
        print("6/6 返回仪表板")
        page.click('a:has-text("仪表板")')
        page.wait_for_load_state('networkidle')
        time.sleep(5)

        browser.close()

    print("演示完成!")

if __name__ == "__main__":
    # 启动录制
    ffmpeg_process = start_ffmpeg()
    time.sleep(2)  # 等待录制启动

    try:
        # 运行演示
        run_demo()

        # 等待录制完成
        print("等待录制完成...")
        ffmpeg_process.wait()

        print(f"✓ 视频已保存: {OUTPUT_FILE}")

        # 复制到项目目录
        import shutil
        demo_file = "demo_video.mp4"
        shutil.copy(OUTPUT_FILE, demo_file)
        print(f"✓ 已复制到: {demo_file}")

    except KeyboardInterrupt:
        print("\n录制已中断")
        ffmpeg_process.terminate()
