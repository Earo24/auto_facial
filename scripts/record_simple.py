#!/usr/bin/env python3
"""
简化版演示录制 - 不依赖页面元素
"""
from playwright.sync_api import sync_playwright
import time
import subprocess
import shutil
import os

OUTPUT_FILE = "/tmp/auto_facial_demo.mp4"
DURATION = 120

def start_ffmpeg():
    cmd = [
        'ffmpeg', '-y',  # 覆盖已存在的文件
        '-f', 'avfoundation',
        '-framerate', '30',
        '-i', '1',
        '-r', '30',
        '-t', str(DURATION),
        '-pix_fmt', 'yuv420p',
        '-crf', '20',
        '-preset', 'fast',
        OUTPUT_FILE
    ]
    print(f"开始录制: {OUTPUT_FILE} ({DURATION}秒)")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def run_demo():
    print("演示开始...")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=['--window-size=1440,900']
        )
        page = browser.new_page()

        # 1. 仪表板 (25秒)
        page.goto('http://localhost:3001')
        page.wait_for_load_state('networkidle')
        print("1/5 仪表板 (25秒)")
        for i in range(5):
            time.sleep(5)
            print(f"  {i+1}/5...")

        # 2. 视频列表 (20秒)
        page.evaluate('window.location.href = "/videos"')
        page.wait_for_load_state('networkidle')
        print("2/5 视频列表 (20秒)")
        time.sleep(20)

        # 3. 聚类页面 (35秒)
        page.goto('http://localhost:3001/clustering?video=老舅14集_片段')
        page.wait_for_load_state('networkidle')
        print("3/5 聚类标注 (35秒)")
        time.sleep(20)
        page.evaluate('window.scrollBy(0, 400)')
        time.sleep(15)

        # 4. 剧集管理 (20秒)
        page.evaluate('window.location.href = "/series"')
        page.wait_for_load_state('networkidle')
        print("4/5 剧集管理 (20秒)")
        time.sleep(20)

        # 5. 视频处理 (20秒)
        page.evaluate('window.location.href = "/processing"')
        page.wait_for_load_state('networkidle')
        print("5/5 视频处理 (20秒)")
        time.sleep(20)

        browser.close()

    print("演示完成!")

if __name__ == "__main__":
    ffmpeg = start_ffmpeg()
    time.sleep(3)

    try:
        run_demo()
        print("\n等待录制完成...")
        time.sleep(5)
        if ffmpeg.poll() is None:
            ffmpeg.wait()

        size = os.path.getsize(OUTPUT_FILE) / (1024*1024)
        print(f"✓ 视频已保存: {OUTPUT_FILE} ({size:.1f} MB)")

        shutil.copy(OUTPUT_FILE, "demo_video.mp4")
        print(f"✓ 已复制到: demo_video.mp4")

    except KeyboardInterrupt:
        print("\n录制已中断")
        ffmpeg.terminate()
