"""
Test script for AutoFacial Web Application
Tests the UI, navigation, and components
"""
from playwright.sync_api import sync_playwright
import time

def test_web_app():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={'width': 1440, 'height': 900})
        page = context.new_page()

        # Enable console logging
        console_messages = []
        def on_console(msg):
            console_messages.append(f"{msg.type}: {msg.text}")
        page.on('console', on_console)

        print("=" * 60)
        print("Testing AutoFacial Web Application")
        print("=" * 60)

        # Test 1: Navigate to home page
        print("\n[TEST 1] Navigating to http://localhost:3000")
        try:
            page.goto('http://localhost:3000', timeout=10000)
            page.wait_for_load_state('networkidle', timeout=10000)
            print("✅ Page loaded successfully")

            # Take screenshot
            page.screenshot(path='/tmp/dashboard.png', full_page=True)
            print("📸 Screenshot saved to /tmp/dashboard.png")

        except Exception as e:
            print(f"❌ Failed to load page: {e}")
            browser.close()
            return

        # Test 2: Check page title
        print("\n[TEST 2] Checking page title")
        title = page.title()
        print(f"📄 Page title: {title}")
        if "AutoFacial" in title or "Facial" in title:
            print("✅ Page title contains expected text")
        else:
            print("⚠️  Page title might be incorrect")

        # Test 3: Check for sidebar
        print("\n[TEST 3] Checking sidebar navigation")
        try:
            sidebar = page.locator('aside').first
            if sidebar.count() > 0:
                print("✅ Sidebar found")

                # Check navigation items
                nav_items = page.locator('nav a').all()
                print(f"📍 Found {len(nav_items)} navigation items:")
                for i, item in enumerate(nav_items[:6]):
                    text = item.inner_text()
                    print(f"   {i+1}. {text}")
            else:
                print("❌ Sidebar not found")
        except Exception as e:
            print(f"❌ Error checking sidebar: {e}")

        # Test 4: Check dashboard content
        print("\n[TEST 4] Checking dashboard content")
        try:
            # Check for stats cards
            stats = page.locator('.card').all()
            print(f"📊 Found {len(stats)} cards on dashboard")
            if len(stats) >= 4:
                print("✅ Stats cards are present")
            else:
                print("⚠️  Expected at least 4 stats cards")

            # Check for sections
            sections = page.locator('h1, h2, h3').all()
            print(f"📝 Found {len(sections)} headings")
        except Exception as e:
            print(f"❌ Error checking dashboard: {e}")

        # Test 5: Test navigation to each page
        pages_to_test = [
            ('Dashboard', '/'),
            ('Video Processing', '/processing'),
            ('Clustering', '/clustering'),
            ('Recognition', '/recognition'),
            ('Analysis', '/analysis'),
            ('Settings', '/settings'),
        ]

        for page_name, path in pages_to_test:
            print(f"\n[TEST] Navigating to {page_name} ({path})")
            try:
                page.goto(f'http://localhost:3000{path}', timeout=10000)
                page.wait_for_load_state('networkidle', timeout=10000)

                # Check if page loaded
                h1 = page.locator('h1').first
                if h1.count() > 0:
                    title_text = h1.inner_text()
                    print(f"✅ {page_name} page loaded: {title_text}")

                    # Take screenshot
                    safe_name = page_name.lower().replace(' ', '_')
                    page.screenshot(path=f'/tmp/{safe_name}.png', full_page=True)
                    print(f"📸 Screenshot saved to /tmp/{safe_name}.png")
                else:
                    print(f"⚠️  {page_name} page might not have loaded correctly")

                # Check for errors
                time.sleep(0.5)

            except Exception as e:
                print(f"❌ Error navigating to {page_name}: {e}")

        # Test 6: Check dark mode theme
        print("\n[TEST 6] Verifying dark mode theme")
        page.goto('http://localhost:3000', timeout=10000)
        page.wait_for_load_state('networkidle')

        try:
            body = page.locator('body').first
            bg_color = body.evaluate('el => getComputedStyle(el).backgroundColor')
            print(f"🎨 Background color: {bg_color}")

            text_color = body.evaluate('el => getComputedStyle(el).color')
            print(f"🎨 Text color: {text_color}")

            if 'rgb(10, 10, 10)' in bg_color or '#0a0a0a' in bg_color or 'rgba(0, 0, 0' in bg_color:
                print("✅ Dark mode background applied")
            else:
                print(f"⚠️  Background might not be dark: {bg_color}")

        except Exception as e:
            print(f"❌ Error checking theme: {e}")

        # Test 7: Check console errors
        print("\n[TEST 7] Checking console for errors")
        errors = [msg for msg in console_messages if 'error' in msg.lower()]
        if errors:
            print(f"⚠️  Found {len(errors)} console errors:")
            for err in errors[:5]:
                print(f"   {err}")
        else:
            print("✅ No console errors found")

        # Test 8: Test interactive elements
        print("\n[TEST 8] Testing interactive elements")
        page.goto('http://localhost:3000/processing', timeout=10000)
        page.wait_for_load_state('networkidle')

        try:
            # Check for buttons
            buttons = page.locator('button').all()
            print(f"🔘 Found {len(buttons)} buttons")

            # Check for inputs
            inputs = page.locator('input').all()
            print(f"📝 Found {len(inputs)} input fields")

            # Try clicking a button
            if buttons:
                buttons[0].click()
                print("✅ Successfully clicked a button")

        except Exception as e:
            print(f"❌ Error testing interactions: {e}")

        # Final screenshot
        print("\n[FINAL] Taking final screenshot")
        page.goto('http://localhost:3000', timeout=10000)
        page.wait_for_load_state('networkidle')
        page.screenshot(path='/tmp/final_state.png', full_page=True)
        print("📸 Final screenshot saved to /tmp/final_state.png")

        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print("Screenshots saved to /tmp/:")
        print("  - dashboard.png")
        print("  - video_processing.png")
        print("  - clustering.png")
        print("  - recognition.png")
        print("  - analysis.png")
        print("  - settings.png")
        print("  - final_state.png")
        print("=" * 60)

        # Keep browser open for inspection
        input("\nPress Enter to close browser...")

        browser.close()
        print("\n✅ Testing complete!")

if __name__ == '__main__':
    test_web_app()
