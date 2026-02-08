#!/usr/bin/env python3
"""
AutoFacial 演示视频录制脚本
自动化演示流程并录制屏幕
"""
from playwright.sync_api import sync_playwright
import time
import subprocess
import sys

def start_recording(output_file="/tmp/auto_facial_demo.mov"):
    """使用macOS内置屏幕录制"""
    print("请按以下步骤手动录制：")
    print("1. 按 Cmd+Shift+5 打开屏幕录制")
    print("2. 选择录制整个屏幕或选区")
    print("3. 点击'录制'按钮")
    print("4. 演示将自动开始")
    print("5. 演示结束后停止录制")
    print("\n按回车键开始自动演示...")
    input()

def run_demo():
    """运行自动演示流程"""

    print("=== AutoFacial 演示开始 ===\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # 1. 访问仪表板
        print("1. 访问仪表板...")
        page.goto('http://localhost:3001')
        page.wait_for_load_state('networkidle')
        time.sleep(3)

        # 2. 展示仪表板统计
        print("2. 展示仪表板统计...")
        page.screenshot(path='/tmp/demo_1_dashboard.png')
        time.sleep(2)

        # 3. 访问视频列表
        print("3. 访问视频列表...")
        page.click('text=视频列表')
        page.wait_for_load_state('networkidle')
        time.sleep(2)

        # 4. 选择视频
        print("4. 选择视频...")
        page.select_option('select', '老舅14集_片段')
        time.sleep(2)

        # 5. 进入聚类页面
        print("5. 进入聚类标注页面...")
        page.goto('http://localhost:3001/clustering?video=老舅14集_片段')
        page.wait_for_load_state('networkidle')
        time.sleep(3)

        # 6. 展示聚类结果
        print("6. 展示聚类结果...")
        page.screenshot(path='/tmp/demo_2_clustering.png')
        time.sleep(2)

        # 7. 选择一个簇查看详情
        print("7. 选择簇查看详情...")
        page.click('text=张海宇')
        time.sleep(2)

        # 8. 展示演员匹配结果
        print("8. 展示演员匹配结果...")
        page.screenshot(path='/tmp/demo_3_actor.png')
        time.sleep(2)

        # 9. 访问剧集管理
        print("9. 访问剧集管理...")
        page.click('text=剧集管理')
        page.wait_for_load_state('networkidle')
        time.sleep(2)

        # 10. 展示剧集信息
        print("10. 展示剧集信息...")
        page.screenshot(path='/tmp/demo_4_series.png')
        time.sleep(2)

        browser.close()

    print("\n=== 演示完成 ===")
    print(f"截图已保存到 /tmp/demo_*.png")

if __name__ == "__main__":
    # 提示用户开始录制
    start_recording()

    # 运行演示
    run_demo()

    print("\n请停止屏幕录制！")
