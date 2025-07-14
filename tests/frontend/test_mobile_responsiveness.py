"""
Comprehensive Mobile Responsiveness Test Suite
Issue #18: Mobile-Responsive UI Enhancements

Tests for mobile UI components, touch interactions, and responsive behavior
using Selenium WebDriver with mobile device emulation
"""

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.touch_actions import TouchActions
import time


class TestMobileResponsiveness:
    """Test suite for mobile responsive design and touch interactions."""

    @pytest.fixture(scope="class")
    def mobile_driver(self):
        """Chrome driver configured for mobile testing."""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Mobile device emulation (iPhone 12)
        mobile_emulation = {
            "deviceMetrics": {
                "width": 390,
                "height": 844,
                "pixelRatio": 3.0
            },
            "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
        }
        options.add_experimental_option("mobileEmulation", mobile_emulation)
        
        driver = webdriver.Chrome(options=options)
        driver.implicitly_wait(10)
        yield driver
        driver.quit()

    @pytest.fixture(scope="class")
    def tablet_driver(self):
        """Chrome driver configured for tablet testing."""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Tablet device emulation (iPad)
        mobile_emulation = {
            "deviceMetrics": {
                "width": 768,
                "height": 1024,
                "pixelRatio": 2.0
            },
            "userAgent": "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
        }
        options.add_experimental_option("mobileEmulation", mobile_emulation)
        
        driver = webdriver.Chrome(options=options)
        driver.implicitly_wait(10)
        yield driver
        driver.quit()

    def test_mobile_navigation_toggle(self, mobile_driver):
        """Test mobile navigation menu toggle functionality."""
        mobile_driver.get("http://localhost:8000")
        
        # Check mobile nav toggle is visible
        nav_toggle = mobile_driver.find_element(By.ID, "mobile-menu-toggle")
        assert nav_toggle.is_displayed()
        
        # Check sidebar is initially hidden
        sidebar = mobile_driver.find_element(By.ID, "mobile-navigation")
        assert "open" not in sidebar.get_attribute("class")
        
        # Click toggle to open menu
        nav_toggle.click()
        time.sleep(0.5)
        
        # Check sidebar is now visible
        assert "open" in sidebar.get_attribute("class")
        
        # Check overlay is visible
        overlay = mobile_driver.find_element(By.ID, "mobile-sidebar-overlay")
        assert "open" in overlay.get_attribute("class")

    def test_mobile_search_functionality(self, mobile_driver):
        """Test mobile search overlay and functionality."""
        mobile_driver.get("http://localhost:8000")
        
        # Find and click search button
        search_button = mobile_driver.find_element(By.CSS_SELECTOR, "[aria-label='Search']")
        search_button.click()
        
        # Check search overlay appears
        search_overlay = mobile_driver.find_element(By.ID, "mobile-search-overlay")
        assert "hidden" not in search_overlay.get_attribute("class")
        
        # Test search input
        search_input = mobile_driver.find_element(By.ID, "mobile-search-input")
        search_input.send_keys("test query")
        
        # Wait for search results
        time.sleep(1)
        results = mobile_driver.find_element(By.ID, "mobile-search-results")
        assert results.text != "Start typing to search..."

    def test_responsive_grid_layout(self, mobile_driver):
        """Test responsive grid layout adjusts correctly on mobile."""
        mobile_driver.get("http://localhost:8000")
        
        # Check metric cards use single column on mobile
        metric_cards = mobile_driver.find_elements(By.CLASS_NAME, "metric-card")
        
        if metric_cards:
            # Get the container width
            container = mobile_driver.find_element(By.CSS_SELECTOR, ".grid")
            container_width = container.size['width']
            
            # Check cards are in single column (approximately full width)
            for card in metric_cards[:3]:  # Check first 3 cards
                card_width = card.size['width']
                width_ratio = card_width / container_width
                assert width_ratio > 0.8, f"Card width ratio {width_ratio} should be > 0.8 for single column"

    def test_touch_targets_minimum_size(self, mobile_driver):
        """Test all touch targets meet minimum 44px requirement."""
        mobile_driver.get("http://localhost:8000")
        
        # Find all interactive elements
        touch_elements = mobile_driver.find_elements(By.CSS_SELECTOR, 
            "button, .btn, .mobile-touch-target, a, input[type='submit'], input[type='button']")
        
        for element in touch_elements:
            if element.is_displayed():
                size = element.size
                assert size['height'] >= 44, f"Touch target height {size['height']} should be >= 44px"
                assert size['width'] >= 44, f"Touch target width {size['width']} should be >= 44px"

    def test_mobile_form_elements(self, mobile_driver):
        """Test form elements are mobile-friendly."""
        mobile_driver.get("http://localhost:8000/detection")
        
        # Check input fields have appropriate size
        inputs = mobile_driver.find_elements(By.CSS_SELECTOR, "input, select, textarea")
        
        for input_elem in inputs:
            if input_elem.is_displayed():
                height = input_elem.size['height']
                assert height >= 44, f"Form input height {height} should be >= 44px"
                
                # Check font size is at least 16px to prevent zoom on iOS
                font_size = mobile_driver.execute_script(
                    "return window.getComputedStyle(arguments[0]).fontSize;", input_elem
                )
                font_size_value = float(font_size.replace('px', ''))
                assert font_size_value >= 16, f"Font size {font_size_value} should be >= 16px"

    def test_mobile_table_scrolling(self, mobile_driver):
        """Test tables have horizontal scrolling on mobile."""
        mobile_driver.get("http://localhost:8000/datasets")
        
        # Find table containers
        table_containers = mobile_driver.find_elements(By.CLASS_NAME, "table-container")
        
        for container in table_containers:
            if container.is_displayed():
                # Check container has overflow-x auto or scroll
                overflow_x = mobile_driver.execute_script(
                    "return window.getComputedStyle(arguments[0]).overflowX;", container
                )
                assert overflow_x in ['auto', 'scroll'], f"Table container should have overflow-x: auto or scroll"

    def test_mobile_chart_responsiveness(self, mobile_driver):
        """Test charts are responsive and touch-friendly."""
        mobile_driver.get("http://localhost:8000")
        
        # Find chart containers
        chart_containers = mobile_driver.find_elements(By.CLASS_NAME, "chart-container")
        
        for chart in chart_containers:
            if chart.is_displayed():
                # Check chart takes full width
                width = chart.size['width']
                parent_width = mobile_driver.execute_script(
                    "return arguments[0].parentElement.offsetWidth;", chart
                )
                width_ratio = width / parent_width
                assert width_ratio > 0.9, f"Chart should take nearly full width"
                
                # Check minimum height for touch interaction
                height = chart.size['height']
                assert height >= 200, f"Chart height {height} should be >= 200px for mobile"

    def test_tablet_layout_differences(self, tablet_driver):
        """Test layout differences on tablet vs mobile."""
        tablet_driver.get("http://localhost:8000")
        
        # Check grid uses 2+ columns on tablet
        grid_container = tablet_driver.find_element(By.CSS_SELECTOR, ".grid")
        grid_template_columns = tablet_driver.execute_script(
            "return window.getComputedStyle(arguments[0]).gridTemplateColumns;", grid_container
        )
        
        # Should have multiple columns (not just '1fr')
        column_count = len(grid_template_columns.split(' '))
        assert column_count >= 2, f"Tablet should have {column_count} >= 2 grid columns"

    def test_mobile_modal_behavior(self, mobile_driver):
        """Test modals behave correctly on mobile."""
        mobile_driver.get("http://localhost:8000")
        
        # Try to find and trigger a modal
        modal_triggers = mobile_driver.find_elements(By.CSS_SELECTOR, "[data-modal], .modal-trigger")
        
        if modal_triggers:
            modal_triggers[0].click()
            time.sleep(0.5)
            
            # Check modal appears at bottom of screen (mobile style)
            modal = mobile_driver.find_element(By.CLASS_NAME, "modal")
            if modal.is_displayed():
                modal_content = modal.find_element(By.CLASS_NAME, "modal-content")
                position = modal_content.location
                screen_height = mobile_driver.get_window_size()['height']
                
                # Modal should be positioned near bottom
                assert position['y'] > screen_height * 0.5, "Mobile modal should appear from bottom"

    def test_mobile_gesture_classes(self, mobile_driver):
        """Test mobile-specific CSS classes are applied."""
        mobile_driver.get("http://localhost:8000")
        
        # Check mobile-specific classes exist
        body_classes = mobile_driver.find_element(By.TAG_NAME, "body").get_attribute("class")
        
        # Should have touch capability classes
        assert any(cls in body_classes for cls in ['supports-touch', 'touch-points-']), \
            "Body should have touch capability classes"

    def test_performance_optimizations(self, mobile_driver):
        """Test performance optimizations are active on mobile."""
        mobile_driver.get("http://localhost:8000")
        
        # Check for performance optimization classes
        body_classes = mobile_driver.find_element(By.TAG_NAME, "body").get_attribute("class")
        
        # Should have device-specific optimizations
        assert any(cls in body_classes for cls in ['low-end-device', 'high-end-device']), \
            "Body should have device performance classes"
        
        # Check CSS custom properties are defined
        mobile_transition = mobile_driver.execute_script(
            "return getComputedStyle(document.documentElement).getPropertyValue('--mobile-transition');"
        )
        assert mobile_transition, "Mobile transition custom property should be defined"

    def test_viewport_meta_tag(self, mobile_driver):
        """Test viewport meta tag is correctly configured."""
        mobile_driver.get("http://localhost:8000")
        
        viewport_meta = mobile_driver.find_element(By.CSS_SELECTOR, "meta[name='viewport']")
        viewport_content = viewport_meta.get_attribute("content")
        
        assert "width=device-width" in viewport_content, "Viewport should set width=device-width"
        assert "initial-scale=1.0" in viewport_content, "Viewport should set initial-scale=1.0"

    def test_mobile_accessibility_features(self, mobile_driver):
        """Test mobile accessibility features."""
        mobile_driver.get("http://localhost:8000")
        
        # Check skip links are present
        skip_links = mobile_driver.find_elements(By.CLASS_NAME, "skip-link")
        assert len(skip_links) > 0, "Skip links should be present for accessibility"
        
        # Check ARIA labels on interactive elements
        nav_toggle = mobile_driver.find_element(By.ID, "mobile-menu-toggle")
        assert nav_toggle.get_attribute("aria-label"), "Nav toggle should have aria-label"
        assert nav_toggle.get_attribute("aria-expanded"), "Nav toggle should have aria-expanded"

    def test_touch_feedback(self, mobile_driver):
        """Test touch feedback is implemented."""
        mobile_driver.get("http://localhost:8000")
        
        # Check buttons have touch feedback styles
        buttons = mobile_driver.find_elements(By.CSS_SELECTOR, "button, .btn")
        
        for button in buttons[:3]:  # Test first 3 buttons
            if button.is_displayed():
                # Check for transition styles
                transition = mobile_driver.execute_script(
                    "return window.getComputedStyle(arguments[0]).transition;", button
                )
                assert transition and transition != "none", "Buttons should have transition for touch feedback"

    def test_mobile_spacing_utilities(self, mobile_driver):
        """Test mobile-specific spacing utilities."""
        mobile_driver.get("http://localhost:8000")
        
        # Check mobile spacing classes are available
        mobile_spaced_elements = mobile_driver.find_elements(By.CSS_SELECTOR, 
            ".mobile-p-4, .mobile-px-4, .mobile-py-4, .mobile-m-4")
        
        # If elements with mobile spacing exist, verify they have appropriate spacing
        for element in mobile_spaced_elements:
            padding = mobile_driver.execute_script(
                "return window.getComputedStyle(arguments[0]).padding;", element
            )
            assert padding and padding != "0px", "Mobile spacing utilities should apply padding"


class TestTouchGestures:
    """Test suite for touch gesture functionality."""

    @pytest.fixture
    def touch_driver(self):
        """Chrome driver with touch events enabled."""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--touch-events")
        
        mobile_emulation = {
            "deviceMetrics": {"width": 390, "height": 844, "pixelRatio": 3.0},
            "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
        }
        options.add_experimental_option("mobileEmulation", mobile_emulation)
        
        driver = webdriver.Chrome(options=options)
        yield driver
        driver.quit()

    def test_swipe_gesture_setup(self, touch_driver):
        """Test swipe gesture event listeners are set up."""
        touch_driver.get("http://localhost:8000")
        
        # Check if swipe navigation elements exist
        swipe_elements = touch_driver.find_elements(By.CLASS_NAME, "swipe-navigation")
        
        # Verify touch gesture manager is initialized
        gesture_manager_exists = touch_driver.execute_script(
            "return typeof window.EnhancedTouchGestureManager !== 'undefined';"
        )
        assert gesture_manager_exists, "Touch gesture manager should be initialized"

    def test_pull_to_refresh_setup(self, touch_driver):
        """Test pull-to-refresh functionality setup."""
        touch_driver.get("http://localhost:8000")
        
        # Check for pull-to-refresh elements
        refresh_elements = touch_driver.find_elements(By.CLASS_NAME, "pull-to-refresh")
        
        if refresh_elements:
            # Check refresh indicator exists
            indicators = touch_driver.find_elements(By.CLASS_NAME, "pull-to-refresh-indicator")
            assert len(indicators) > 0, "Pull-to-refresh indicator should exist"

    def test_long_press_detection(self, touch_driver):
        """Test long press gesture detection."""
        touch_driver.get("http://localhost:8000")
        
        # Find elements with long press data attribute
        long_press_elements = touch_driver.find_elements(By.CSS_SELECTOR, "[data-long-press]")
        
        if long_press_elements:
            # Verify long press event listener is set up
            has_listener = touch_driver.execute_script(
                "return typeof window.EnhancedTouchGestureManager !== 'undefined';"
            )
            assert has_listener, "Long press detection should be set up"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])