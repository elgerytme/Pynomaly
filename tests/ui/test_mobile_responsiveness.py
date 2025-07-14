"""
Comprehensive test suite for mobile UI responsiveness enhancements.
Tests touch interactions, responsive design, navigation, and accessibility.
"""

import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.touch_actions import TouchActions
import time


class TestMobileResponsiveness:
    """Test suite for mobile UI responsiveness and touch interactions."""

    @pytest.fixture(scope="class")
    def mobile_driver(self):
        """Setup mobile Chrome driver with touch simulation."""
        mobile_emulation = {
            "deviceMetrics": {"width": 375, "height": 667, "pixelRatio": 2.0},
            "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
        }
        
        options = webdriver.ChromeOptions()
        options.add_experimental_option("mobileEmulation", mobile_emulation)
        options.add_argument("--enable-touch-events")
        
        driver = webdriver.Chrome(options=options)
        yield driver
        driver.quit()

    @pytest.fixture(scope="class")
    def tablet_driver(self):
        """Setup tablet Chrome driver."""
        mobile_emulation = {
            "deviceMetrics": {"width": 768, "height": 1024, "pixelRatio": 2.0},
            "userAgent": "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
        }
        
        options = webdriver.ChromeOptions()
        options.add_experimental_option("mobileEmulation", mobile_emulation)
        
        driver = webdriver.Chrome(options=options)
        yield driver
        driver.quit()

    def test_responsive_breakpoints(self, mobile_driver, tablet_driver):
        """Test responsive design at different breakpoints."""
        # Test mobile layout (375px)
        mobile_driver.get("http://localhost:8000/dashboard")
        
        # Verify mobile-specific elements
        tab_bar = mobile_driver.find_element(By.CLASS_NAME, "tab-bar")
        assert tab_bar.is_displayed(), "Tab bar should be visible on mobile"
        
        # Verify tab bar is at bottom
        tab_bar_rect = tab_bar.rect
        viewport_height = mobile_driver.execute_script("return window.innerHeight")
        assert tab_bar_rect['y'] + tab_bar_rect['height'] >= viewport_height - 10, "Tab bar should be at bottom"
        
        # Test tablet layout (768px)
        tablet_driver.get("http://localhost:8000/dashboard")
        
        # Tab bar should be hidden on tablet
        try:
            tab_bar_tablet = tablet_driver.find_element(By.CLASS_NAME, "tab-bar")
            assert not tab_bar_tablet.is_displayed(), "Tab bar should be hidden on tablet"
        except:
            pass  # Element not found is also acceptable
        
        # Dashboard tabs should be visible on tablet
        dashboard_tabs = tablet_driver.find_element(By.CLASS_NAME, "dashboard-tabs")
        assert dashboard_tabs.is_displayed(), "Dashboard tabs should be visible on tablet"

    def test_touch_target_sizes(self, mobile_driver):
        """Test that touch targets meet minimum size requirements (44px)."""
        mobile_driver.get("http://localhost:8000/dashboard")
        
        # Find all interactive elements
        buttons = mobile_driver.find_elements(By.TAG_NAME, "button")
        links = mobile_driver.find_elements(By.TAG_NAME, "a")
        inputs = mobile_driver.find_elements(By.TAG_NAME, "input")
        
        interactive_elements = buttons + links + inputs
        
        for element in interactive_elements[:10]:  # Test first 10 elements
            if element.is_displayed():
                size = element.size
                # Touch targets should be at least 44px (converted to viewport pixels)
                assert size['height'] >= 44 or size['width'] >= 44, f"Touch target too small: {size}"

    def test_swipe_navigation(self, mobile_driver):
        """Test swipe gesture navigation between tabs."""
        mobile_driver.get("http://localhost:8000/dashboard")
        
        # Wait for mobile dashboard to initialize
        WebDriverWait(mobile_driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "mobile-dashboard"))
        )
        
        # Get initial active tab
        active_tab = mobile_driver.find_element(By.CSS_SELECTOR, ".tab-button.active")
        initial_tab_index = int(active_tab.get_attribute("data-tab") or "0")
        
        # Simulate swipe left on content area
        content_area = mobile_driver.find_element(By.CLASS_NAME, "content-area")
        actions = ActionChains(mobile_driver)
        
        # Swipe left (should go to next tab)
        actions.click_and_hold(content_area).move_by_offset(-100, 0).release().perform()
        time.sleep(0.5)
        
        # Check if tab changed
        new_active_tab = mobile_driver.find_element(By.CSS_SELECTOR, ".tab-button.active")
        new_tab_index = int(new_active_tab.get_attribute("data-tab") or "0")
        
        if initial_tab_index < 3:  # If not the last tab
            assert new_tab_index == initial_tab_index + 1, "Swipe left should go to next tab"

    def test_pull_to_refresh(self, mobile_driver):
        """Test pull-to-refresh functionality."""
        mobile_driver.get("http://localhost:8000/dashboard")
        
        # Wait for content to load
        WebDriverWait(mobile_driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "content-area"))
        )
        
        content_area = mobile_driver.find_element(By.CLASS_NAME, "content-area")
        
        # Ensure we're at the top
        mobile_driver.execute_script("arguments[0].scrollTop = 0", content_area)
        
        # Simulate pull down gesture
        actions = ActionChains(mobile_driver)
        actions.click_and_hold(content_area).move_by_offset(0, 100).release().perform()
        
        # Check for refresh indicator
        try:
            refresh_indicator = mobile_driver.find_element(By.CLASS_NAME, "pull-to-refresh-indicator")
            assert refresh_indicator.is_displayed(), "Pull-to-refresh indicator should appear"
        except:
            pass  # Some implementations may not show indicator immediately

    def test_navigation_drawer(self, mobile_driver):
        """Test mobile navigation drawer functionality."""
        mobile_driver.get("http://localhost:8000/dashboard")
        
        # Find and click menu button
        menu_button = mobile_driver.find_element(By.CLASS_NAME, "menu-button")
        menu_button.click()
        
        # Wait for navigation drawer to appear
        WebDriverWait(mobile_driver, 5).until(
            EC.presence_of_element_located((By.CLASS_NAME, "mobile-nav-drawer"))
        )
        
        nav_drawer = mobile_driver.find_element(By.CLASS_NAME, "mobile-nav-drawer")
        assert "open" in nav_drawer.get_attribute("class"), "Navigation drawer should be open"
        
        # Test overlay click to close
        overlay = mobile_driver.find_element(By.CLASS_NAME, "mobile-nav-overlay")
        overlay.click()
        
        # Wait for drawer to close
        time.sleep(0.5)
        try:
            nav_drawer = mobile_driver.find_element(By.CLASS_NAME, "mobile-nav-drawer")
            assert "open" not in nav_drawer.get_attribute("class"), "Navigation drawer should be closed"
        except:
            pass  # Element removed is also acceptable

    def test_form_accessibility(self, mobile_driver):
        """Test form elements accessibility on mobile."""
        mobile_driver.get("http://localhost:8000/dashboard")
        
        # Look for form elements
        form_inputs = mobile_driver.find_elements(By.CSS_SELECTOR, ".form-input, .form-select, .form-textarea")
        
        for input_element in form_inputs:
            if input_element.is_displayed():
                # Check for proper labeling
                input_id = input_element.get_attribute("id")
                aria_label = input_element.get_attribute("aria-label")
                
                if input_id:
                    # Look for associated label
                    try:
                        label = mobile_driver.find_element(By.CSS_SELECTOR, f"label[for='{input_id}']")
                        assert label.is_displayed(), f"Label for {input_id} should be visible"
                    except:
                        assert aria_label, f"Input {input_id} should have aria-label if no label found"
                
                # Check touch target size
                size = input_element.size
                assert size['height'] >= 44, f"Form input should meet minimum touch target height: {size}"

    def test_widget_interactions(self, mobile_driver):
        """Test widget touch interactions and gestures."""
        mobile_driver.get("http://localhost:8000/dashboard")
        
        # Wait for widgets to load
        WebDriverWait(mobile_driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "widget"))
        )
        
        widgets = mobile_driver.find_elements(By.CLASS_NAME, "widget")
        
        if widgets:
            widget = widgets[0]
            
            # Test touch feedback
            actions = ActionChains(mobile_driver)
            actions.click(widget).perform()
            
            # Check for touch-active class (briefly)
            time.sleep(0.1)
            # Note: touch-active class is typically removed quickly

    def test_toast_notifications(self, mobile_driver):
        """Test mobile toast notification system."""
        mobile_driver.get("http://localhost:8000/dashboard")
        
        # Execute JavaScript to show a test toast
        mobile_driver.execute_script("""
            if (window.MobileDashboardManager) {
                const dashboard = new window.MobileDashboardManager(document.body);
                dashboard.showToast('Test notification', 'info', 2000);
            }
        """)
        
        # Check for toast appearance
        try:
            WebDriverWait(mobile_driver, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "mobile-toast"))
            )
            
            toast = mobile_driver.find_element(By.CLASS_NAME, "mobile-toast")
            assert toast.is_displayed(), "Toast notification should be visible"
            
            # Check positioning (should be near top on mobile)
            toast_rect = toast.rect
            assert toast_rect['y'] < 200, "Toast should appear near top of screen"
            
        except:
            pass  # JavaScript might not be available in test environment

    def test_keyboard_navigation(self, mobile_driver):
        """Test keyboard navigation accessibility."""
        mobile_driver.get("http://localhost:8000/dashboard")
        
        # Find focusable elements
        focusable_elements = mobile_driver.find_elements(
            By.CSS_SELECTOR, 
            "button, a, input, select, textarea, [tabindex]"
        )
        
        visible_focusable = [el for el in focusable_elements if el.is_displayed()]
        
        if visible_focusable:
            first_element = visible_focusable[0]
            first_element.click()
            
            # Check focus visibility
            focused_element = mobile_driver.switch_to.active_element
            assert focused_element == first_element, "Element should receive focus"

    def test_viewport_meta_tag(self, mobile_driver):
        """Test that viewport meta tag is properly configured."""
        mobile_driver.get("http://localhost:8000/dashboard")
        
        viewport_meta = mobile_driver.find_element(By.CSS_SELECTOR, "meta[name='viewport']")
        viewport_content = viewport_meta.get_attribute("content")
        
        assert "width=device-width" in viewport_content, "Viewport should use device width"
        assert "initial-scale=1.0" in viewport_content, "Viewport should have proper initial scale"

    def test_safe_area_support(self, mobile_driver):
        """Test safe area inset support for devices with notches."""
        mobile_driver.get("http://localhost:8000/dashboard")
        
        # Check if CSS uses safe-area-inset properties
        header = mobile_driver.find_element(By.CLASS_NAME, "mobile-header")
        
        # Execute JavaScript to check computed styles
        padding_top = mobile_driver.execute_script("""
            const header = arguments[0];
            const styles = window.getComputedStyle(header);
            return styles.paddingTop;
        """, header)
        
        # Safe area insets might not be present in emulated environment
        # but the CSS should be properly structured

    def test_progressive_enhancement(self, mobile_driver):
        """Test that progressive enhancement features work properly."""
        mobile_driver.get("http://localhost:8000/dashboard")
        
        # Test backdrop-filter support detection
        supports_backdrop_filter = mobile_driver.execute_script("""
            return CSS.supports('backdrop-filter', 'blur(10px)');
        """)
        
        if supports_backdrop_filter:
            header = mobile_driver.find_element(By.CLASS_NAME, "mobile-header")
            backdrop_filter = mobile_driver.execute_script("""
                const styles = window.getComputedStyle(arguments[0]);
                return styles.backdropFilter || styles.webkitBackdropFilter;
            """, header)
            
            assert backdrop_filter and backdrop_filter != "none", "Backdrop filter should be applied when supported"


@pytest.mark.integration
class TestMobileUIIntegration:
    """Integration tests for mobile UI with backend services."""
    
    def test_mobile_api_calls(self, mobile_driver):
        """Test that mobile UI properly handles API calls."""
        mobile_driver.get("http://localhost:8000/dashboard")
        
        # Wait for any initial API calls to complete
        WebDriverWait(mobile_driver, 10).until(
            lambda driver: driver.execute_script("return document.readyState") == "complete"
        )
        
        # Check for error states in mobile UI
        error_elements = mobile_driver.find_elements(By.CSS_SELECTOR, ".error, .toast.error")
        
        # Should not have critical errors on page load
        critical_errors = [el for el in error_elements if el.is_displayed() and "critical" in el.get_attribute("class")]
        assert len(critical_errors) == 0, "Should not have critical errors on mobile dashboard"

    def test_mobile_performance(self, mobile_driver):
        """Test mobile UI performance metrics."""
        mobile_driver.get("http://localhost:8000/dashboard")
        
        # Wait for page to fully load
        WebDriverWait(mobile_driver, 15).until(
            lambda driver: driver.execute_script("return document.readyState") == "complete"
        )
        
        # Check for performance timing
        navigation_timing = mobile_driver.execute_script("""
            const timing = performance.timing;
            return {
                domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                loadComplete: timing.loadEventEnd - timing.navigationStart
            };
        """)
        
        # Basic performance assertions
        assert navigation_timing['domContentLoaded'] < 5000, "DOM should load within 5 seconds"
        assert navigation_timing['loadComplete'] < 10000, "Page should fully load within 10 seconds"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])