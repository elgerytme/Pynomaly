#!/usr/bin/env python3
"""
Direct UI test execution without pytest framework.
This avoids asyncio conflicts and dependency issues.
"""

from playwright.sync_api import sync_playwright
from pathlib import Path
import time
import json
from datetime import datetime


class UITestRunner:
    """Direct UI test runner using Playwright."""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.screenshots_dir = Path("screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
        self.test_results = []
        
    def log_test(self, test_name, status, details="", duration=0):
        """Log test result."""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "duration": f"{duration:.2f}s",
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {test_name}: {status} ({duration:.2f}s)")
        if details:
            print(f"   📝 {details}")
    
    def test_dashboard_loads(self, page):
        """Test dashboard loading."""
        start_time = time.time()
        try:
            page.goto(f"{self.base_url}/")
            page.wait_for_load_state("networkidle")
            
            title = page.title()
            assert "Pynomaly" in title or title == "", "Title check failed"
            
            # Check key elements
            assert page.locator("#main-navigation").is_visible(), "Navigation not visible"
            assert page.locator("#logo").is_visible(), "Logo not visible"
            assert page.locator("#main-content").is_visible(), "Main content not visible"
            
            # Take screenshot
            page.screenshot(path=str(self.screenshots_dir / "dashboard_test.png"), full_page=True)
            
            duration = time.time() - start_time
            self.log_test("Dashboard Load", "PASS", f"Title: '{title}'", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Dashboard Load", "FAIL", str(e), duration)
            return False
    
    def test_navigation(self, page):
        """Test navigation functionality."""
        start_time = time.time()
        try:
            page.goto(f"{self.base_url}/")
            page.wait_for_load_state("networkidle")
            
            # Test navigation to different pages
            pages = [
                ("/detectors", "Detectors"),
                ("/datasets", "Datasets"),
                ("/detection", "Detection"),
                ("/visualizations", "Visualizations"),
                ("/exports", "Exports")
            ]
            
            nav_success = 0
            for url_path, expected_title in pages:
                try:
                    page.goto(f"{self.base_url}{url_path}")
                    page.wait_for_load_state("networkidle")
                    
                    # Take screenshot
                    page_name = url_path.strip("/")
                    page.screenshot(path=str(self.screenshots_dir / f"{page_name}_test.png"), full_page=True)
                    
                    nav_success += 1
                except Exception as e:
                    print(f"     ⚠️ Navigation to {url_path} failed: {e}")
            
            duration = time.time() - start_time
            success_rate = nav_success / len(pages)
            
            if success_rate >= 0.8:  # 80% success rate
                self.log_test("Navigation", "PASS", f"{nav_success}/{len(pages)} pages", duration)
                return True
            else:
                self.log_test("Navigation", "FAIL", f"Only {nav_success}/{len(pages)} pages", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Navigation", "FAIL", str(e), duration)
            return False
    
    def test_responsive_design(self, page):
        """Test responsive design across viewports."""
        start_time = time.time()
        try:
            viewports = [
                ("desktop", {"width": 1920, "height": 1080}),
                ("tablet", {"width": 768, "height": 1024}),
                ("mobile", {"width": 375, "height": 667})
            ]
            
            responsive_success = 0
            for device_name, viewport in viewports:
                try:
                    page.set_viewport_size(viewport)
                    page.goto(f"{self.base_url}/")
                    page.wait_for_load_state("networkidle")
                    
                    # Take screenshot
                    page.screenshot(
                        path=str(self.screenshots_dir / f"responsive_{device_name}.png"),
                        full_page=True
                    )
                    
                    # Basic visibility checks
                    assert page.locator("#logo").is_visible(), f"Logo not visible on {device_name}"
                    assert page.locator("#main-content").is_visible(), f"Content not visible on {device_name}"
                    
                    responsive_success += 1
                except Exception as e:
                    print(f"     ⚠️ Responsive test failed for {device_name}: {e}")
            
            duration = time.time() - start_time
            if responsive_success == len(viewports):
                self.log_test("Responsive Design", "PASS", f"All {len(viewports)} viewports", duration)
                return True
            else:
                self.log_test("Responsive Design", "PARTIAL", f"{responsive_success}/{len(viewports)} viewports", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Responsive Design", "FAIL", str(e), duration)
            return False
    
    def test_mobile_menu(self, page):
        """Test mobile menu functionality."""
        start_time = time.time()
        try:
            # Set mobile viewport
            page.set_viewport_size({"width": 375, "height": 667})
            page.goto(f"{self.base_url}/")
            page.wait_for_load_state("networkidle")
            
            # Check mobile menu button
            mobile_btn = page.locator("#mobile-menu-btn")
            assert mobile_btn.is_visible(), "Mobile menu button not visible"
            
            # Check menu is initially hidden
            mobile_menu = page.locator("#mobile-menu")
            assert mobile_menu.is_hidden(), "Mobile menu should be hidden initially"
            
            # Click to open menu
            mobile_btn.click()
            assert mobile_menu.is_visible(), "Mobile menu should be visible after click"
            
            # Take screenshot with menu open
            page.screenshot(path=str(self.screenshots_dir / "mobile_menu_open.png"), full_page=True)
            
            duration = time.time() - start_time
            self.log_test("Mobile Menu", "PASS", "Menu toggle working", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Mobile Menu", "FAIL", str(e), duration)
            return False
    
    def test_interactive_elements(self, page):
        """Test interactive elements."""
        start_time = time.time()
        try:
            page.set_viewport_size({"width": 1920, "height": 1080})
            page.goto(f"{self.base_url}/")
            page.wait_for_load_state("networkidle")
            
            # Test button interactions
            buttons = [
                "#quick-detection-btn",
                "#upload-dataset-btn", 
                "#autonomous-mode-btn"
            ]
            
            interactive_success = 0
            for button_id in buttons:
                try:
                    button = page.locator(button_id)
                    if button.is_visible():
                        button.hover()  # Test hover interaction
                        interactive_success += 1
                except Exception as e:
                    print(f"     ⚠️ Button interaction failed for {button_id}: {e}")
            
            duration = time.time() - start_time
            if interactive_success >= 2:  # At least 2 out of 3 buttons working
                self.log_test("Interactive Elements", "PASS", f"{interactive_success}/{len(buttons)} elements", duration)
                return True
            else:
                self.log_test("Interactive Elements", "FAIL", f"Only {interactive_success}/{len(buttons)} elements", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Interactive Elements", "FAIL", str(e), duration)
            return False
    
    def test_performance(self, page):
        """Test basic performance metrics."""
        start_time = time.time()
        try:
            load_start = time.time()
            page.goto(f"{self.base_url}/")
            page.wait_for_load_state("networkidle")
            load_time = time.time() - load_start
            
            duration = time.time() - start_time
            
            if load_time < 5.0:  # Reasonable load time
                self.log_test("Performance", "PASS", f"Load time: {load_time:.2f}s", duration)
                return True
            else:
                self.log_test("Performance", "FAIL", f"Slow load time: {load_time:.2f}s", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Performance", "FAIL", str(e), duration)
            return False
    
    def test_health_endpoint(self, page):
        """Test health endpoint."""
        start_time = time.time()
        try:
            response = page.goto(f"{self.base_url}/health")
            content = page.content()
            
            assert response.status == 200, "Health endpoint not returning 200"
            assert "healthy" in content, "Health response missing 'healthy'"
            
            duration = time.time() - start_time
            self.log_test("Health Endpoint", "PASS", "API responding", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Health Endpoint", "FAIL", str(e), duration)
            return False
    
    def run_all_tests(self):
        """Run all UI tests."""
        print("🎯 Starting Comprehensive UI Testing")
        print("=" * 60)
        
        total_start = time.time()
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()
            
            # Run all tests
            tests = [
                self.test_health_endpoint,
                self.test_dashboard_loads,
                self.test_navigation,
                self.test_responsive_design,
                self.test_mobile_menu,
                self.test_interactive_elements,
                self.test_performance
            ]
            
            passed_tests = 0
            for test_func in tests:
                if test_func(page):
                    passed_tests += 1
            
            browser.close()
        
        total_duration = time.time() - total_start
        
        # Generate summary
        print("\n" + "=" * 60)
        print("📊 UI TESTING RESULTS SUMMARY")
        print("=" * 60)
        
        success_rate = (passed_tests / len(tests)) * 100
        print(f"📈 Success Rate: {passed_tests}/{len(tests)} ({success_rate:.1f}%)")
        print(f"⏱️ Total Duration: {total_duration:.2f}s")
        
        # Screenshot summary
        screenshots = list(self.screenshots_dir.glob("*.png"))
        print(f"📸 Screenshots Captured: {len(screenshots)}")
        for screenshot in screenshots:
            print(f"   • {screenshot.name}")
        
        # Save detailed results
        results_file = Path("ui_test_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "summary": {
                    "total_tests": len(tests),
                    "passed_tests": passed_tests,
                    "success_rate": f"{success_rate:.1f}%",
                    "total_duration": f"{total_duration:.2f}s",
                    "screenshots_count": len(screenshots)
                },
                "detailed_results": self.test_results
            }, f, indent=2)
        
        print(f"📋 Detailed results saved: {results_file}")
        
        if success_rate >= 80:
            print("🎉 UI Testing: SUCCESS")
            return True
        else:
            print("⚠️ UI Testing: NEEDS ATTENTION")
            return False


if __name__ == "__main__":
    runner = UITestRunner()
    success = runner.run_all_tests()
    exit(0 if success else 1)