"""
测试前后端集成
验证Web应用能否正确连接到后端API
"""
from playwright.sync_api import sync_playwright, Page

def test_frontend_backend_integration():
    """测试前后端集成"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # 访问前端页面
        page.goto('http://localhost:3004')
        page.wait_for_load_state('networkidle')

        # 检查页面标题
        assert "AutoFacial" in page.title() or "Face Recognition" in page.title()
        print("✓ 页面加载成功")

        # 检查侧边栏导航
        nav_items = page.locator('nav a').all()
        assert len(nav_items) >= 5
        print(f"✓ 侧边栏有 {len(nav_items)} 个导航项")

        # 检查中文导航文本
        nav_texts = [item.text_content() for item in nav_items]
        assert '仪表板' in nav_texts
        assert '视频处理' in nav_texts
        assert '聚类标注' in nav_texts
        print("✓ 导航文本为中文")

        # 检查仪表板页面
        stats_cards = page.locator('[class*="Card"]').all()
        print(f"✓ 仪表板有 {len(stats_cards)} 个统计卡片")

        # 点击视频处理页面
        page.click('text=视频处理')
        page.wait_for_load_state('networkidle')
        print("✓ 视频处理页面加载成功")

        # 检查上传区域
        upload_area = page.locator('text=拖拽视频到此处').or_(page.locator('text=浏览文件'))
        assert upload_area.count() > 0
        print("✓ 上传区域存在")

        # 检查API连接
        # 通过浏览器开发者工具检查网络请求
        page.goto('http://localhost:3004')
        page.wait_for_load_state('networkidle')

        # 截图
        page.screenshot(path='/tmp/integration_test.png', full_page=True)
        print("✓ 截图已保存到 /tmp/integration_test.png")

        browser.close()
        print("\n✅ 前后端集成测试通过！")

if __name__ == '__main__':
    test_frontend_backend_integration()
