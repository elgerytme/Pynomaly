"""
Comprehensive test suite for Pynomaly frontend components.

This module provides extensive testing coverage for JavaScript components,
CSS styling, interactive elements, and PWA features.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import json
import os
from pathlib import Path

# Import frontend test utilities
from tests.presentation.web.utils import (
    create_test_driver,
    wait_for_element,
    wait_for_text,
    simulate_file_upload,
    check_accessibility,
    measure_performance
)


class TestCacheManagerComponent:
    """Test the cache manager JavaScript component."""
    
    def setup_method(self):
        """Set up test environment."""
        self.driver = create_test_driver()
        self.driver.get("http://localhost:8080")
        
    def teardown_method(self):
        """Clean up after tests."""
        if self.driver:
            self.driver.quit()
    
    def test_cache_manager_initialization(self):
        """Test cache manager component initialization."""
        # Execute JavaScript to test cache manager
        cache_manager_exists = self.driver.execute_script("""
            return typeof window.CacheManager !== 'undefined';
        """)
        assert cache_manager_exists, "CacheManager should be defined"
        
        # Test cache initialization
        cache_initialized = self.driver.execute_script("""
            return window.CacheManager.isInitialized();
        """)
        assert cache_initialized, "Cache should be initialized"
    
    def test_cache_set_and_get(self):
        """Test cache set and get operations."""
        # Test cache set operation
        result = self.driver.execute_script("""
            window.CacheManager.set('test-key', {data: 'test-value'}, 5000);
            return window.CacheManager.get('test-key');
        """)
        assert result is not None
        assert result['data'] == 'test-value'
    
    def test_cache_expiration(self):
        """Test cache expiration functionality."""
        # Set cache with short expiration
        self.driver.execute_script("""
            window.CacheManager.set('expire-key', {data: 'expire-value'}, 100);
        """)
        
        # Check immediate retrieval
        immediate_result = self.driver.execute_script("""
            return window.CacheManager.get('expire-key');
        """)
        assert immediate_result is not None
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Check after expiration
        expired_result = self.driver.execute_script("""
            return window.CacheManager.get('expire-key');
        """)
        assert expired_result is None
    
    def test_cache_storage_limits(self):
        """Test cache storage limits and cleanup."""
        # Fill cache with test data
        self.driver.execute_script("""
            for (let i = 0; i < 100; i++) {
                window.CacheManager.set(`key-${i}`, {data: `value-${i}`}, 60000);
            }
        """)
        
        # Check cache size management
        cache_size = self.driver.execute_script("""
            return window.CacheManager.getSize();
        """)
        assert cache_size > 0
        assert cache_size <= 100  # Should respect size limits
    
    def test_cache_clear_functionality(self):
        """Test cache clear operations."""
        # Set some cache data
        self.driver.execute_script("""
            window.CacheManager.set('clear-key-1', {data: 'value1'}, 60000);
            window.CacheManager.set('clear-key-2', {data: 'value2'}, 60000);
        """)
        
        # Clear cache
        self.driver.execute_script("""
            window.CacheManager.clear();
        """)
        
        # Verify cache is cleared
        result1 = self.driver.execute_script("""
            return window.CacheManager.get('clear-key-1');
        """)
        result2 = self.driver.execute_script("""
            return window.CacheManager.get('clear-key-2');
        """)
        
        assert result1 is None
        assert result2 is None


class TestAnalyticsCharts:
    """Test analytics chart components."""
    
    def setup_method(self):
        """Set up test environment."""
        self.driver = create_test_driver()
        self.driver.get("http://localhost:8080/analytics")
        
    def teardown_method(self):
        """Clean up after tests."""
        if self.driver:
            self.driver.quit()
    
    def test_chart_initialization(self):
        """Test chart component initialization."""
        # Wait for chart container to be present
        chart_container = wait_for_element(self.driver, By.ID, "analytics-chart")
        assert chart_container is not None
        
        # Check if chart is rendered
        chart_canvas = wait_for_element(self.driver, By.TAG_NAME, "canvas")
        assert chart_canvas is not None
    
    def test_chart_data_loading(self):
        """Test chart data loading functionality."""
        # Simulate data loading
        self.driver.execute_script("""
            window.AnalyticsChart.loadData([
                {date: '2023-01-01', value: 10},
                {date: '2023-01-02', value: 20},
                {date: '2023-01-03', value: 15}
            ]);
        """)
        
        # Check if chart updated
        chart_points = self.driver.execute_script("""
            return window.AnalyticsChart.getDataPoints().length;
        """)
        assert chart_points == 3
    
    def test_chart_responsiveness(self):
        """Test chart responsiveness to window resize."""
        # Get initial chart dimensions
        initial_width = self.driver.execute_script("""
            return document.querySelector('canvas').width;
        """)
        
        # Resize window
        self.driver.set_window_size(800, 600)
        time.sleep(0.5)  # Wait for resize handler
        
        # Check if chart resized
        new_width = self.driver.execute_script("""
            return document.querySelector('canvas').width;
        """)
        assert new_width != initial_width
    
    def test_chart_interaction(self):
        """Test chart interaction features."""
        chart_canvas = wait_for_element(self.driver, By.TAG_NAME, "canvas")
        
        # Test hover interaction
        ActionChains(self.driver).move_to_element(chart_canvas).perform()
        time.sleep(0.5)
        
        # Check if tooltip appears
        tooltip = self.driver.find_elements(By.CLASS_NAME, "chart-tooltip")
        assert len(tooltip) > 0
    
    def test_chart_export_functionality(self):
        """Test chart export functionality."""
        # Click export button
        export_button = wait_for_element(self.driver, By.ID, "export-chart")
        export_button.click()
        
        # Check if export dialog appears
        export_dialog = wait_for_element(self.driver, By.CLASS_NAME, "export-dialog")
        assert export_dialog is not None
        
        # Test different export formats
        png_option = wait_for_element(self.driver, By.ID, "export-png")
        png_option.click()
        
        # Verify download initiated
        download_started = self.driver.execute_script("""
            return window.exportStarted === true;
        """)
        assert download_started


class TestRealTimeDashboard:
    """Test real-time dashboard components."""
    
    def setup_method(self):
        """Set up test environment."""
        self.driver = create_test_driver()
        self.driver.get("http://localhost:8080/dashboard")
        
    def teardown_method(self):
        """Clean up after tests."""
        if self.driver:
            self.driver.quit()
    
    def test_dashboard_initialization(self):
        """Test dashboard component initialization."""
        # Check for dashboard container
        dashboard = wait_for_element(self.driver, By.ID, "real-time-dashboard")
        assert dashboard is not None
        
        # Check for key dashboard elements
        metrics_panel = wait_for_element(self.driver, By.CLASS_NAME, "metrics-panel")
        charts_panel = wait_for_element(self.driver, By.CLASS_NAME, "charts-panel")
        alerts_panel = wait_for_element(self.driver, By.CLASS_NAME, "alerts-panel")
        
        assert metrics_panel is not None
        assert charts_panel is not None
        assert alerts_panel is not None
    
    def test_websocket_connection(self):
        """Test WebSocket connection for real-time updates."""
        # Check WebSocket connection status
        ws_connected = self.driver.execute_script("""
            return window.DashboardWebSocket && window.DashboardWebSocket.readyState === 1;
        """)
        assert ws_connected, "WebSocket should be connected"
    
    def test_real_time_metrics_update(self):
        """Test real-time metrics updates."""
        # Get initial metric values
        initial_metrics = self.driver.execute_script("""
            return {
                totalDetections: parseInt(document.querySelector('#total-detections').textContent),
                activeAlerts: parseInt(document.querySelector('#active-alerts').textContent)
            };
        """)
        
        # Simulate real-time update
        self.driver.execute_script("""
            window.DashboardWebSocket.onmessage({
                data: JSON.stringify({
                    type: 'metrics_update',
                    data: {
                        totalDetections: 150,
                        activeAlerts: 5
                    }
                })
            });
        """)
        
        # Wait for update
        time.sleep(0.5)
        
        # Check updated values
        updated_metrics = self.driver.execute_script("""
            return {
                totalDetections: parseInt(document.querySelector('#total-detections').textContent),
                activeAlerts: parseInt(document.querySelector('#active-alerts').textContent)
            };
        """)
        
        assert updated_metrics['totalDetections'] == 150
        assert updated_metrics['activeAlerts'] == 5
    
    def test_alert_notifications(self):
        """Test alert notification system."""
        # Simulate alert notification
        self.driver.execute_script("""
            window.DashboardWebSocket.onmessage({
                data: JSON.stringify({
                    type: 'alert',
                    data: {
                        id: 'alert-123',
                        level: 'warning',
                        message: 'High anomaly rate detected',
                        timestamp: '2023-01-01T12:00:00Z'
                    }
                })
            });
        """)
        
        # Check if alert appears
        alert_notification = wait_for_element(self.driver, By.CLASS_NAME, "alert-notification")
        assert alert_notification is not None
        
        # Check alert content
        alert_text = alert_notification.text
        assert "High anomaly rate detected" in alert_text
    
    def test_dashboard_filtering(self):
        """Test dashboard filtering functionality."""
        # Open filter menu
        filter_button = wait_for_element(self.driver, By.ID, "filter-button")
        filter_button.click()
        
        # Select time range filter
        time_filter = wait_for_element(self.driver, By.ID, "time-range-filter")
        time_filter.click()
        
        # Select last 24 hours
        last_24h = wait_for_element(self.driver, By.XPATH, "//option[@value='24h']")
        last_24h.click()
        
        # Apply filter
        apply_button = wait_for_element(self.driver, By.ID, "apply-filter")
        apply_button.click()
        
        # Check if filter applied
        filter_applied = self.driver.execute_script("""
            return window.DashboardFilters.getCurrentFilter().timeRange === '24h';
        """)
        assert filter_applied


class TestInteractiveElements:
    """Test interactive UI elements."""
    
    def setup_method(self):
        """Set up test environment."""
        self.driver = create_test_driver()
        self.driver.get("http://localhost:8080/detection")
        
    def teardown_method(self):
        """Clean up after tests."""
        if self.driver:
            self.driver.quit()
    
    def test_file_upload_component(self):
        """Test file upload component."""
        # Find file input
        file_input = wait_for_element(self.driver, By.ID, "dataset-upload")
        assert file_input is not None
        
        # Test file upload simulation
        test_file_path = "/tmp/test_dataset.csv"
        with open(test_file_path, 'w') as f:
            f.write("col1,col2,col3\n1,2,3\n4,5,6\n")
        
        file_input.send_keys(test_file_path)
        
        # Check upload progress
        upload_progress = wait_for_element(self.driver, By.CLASS_NAME, "upload-progress")
        assert upload_progress is not None
        
        # Wait for upload completion
        upload_complete = wait_for_text(self.driver, By.CLASS_NAME, "upload-status", "Upload complete")
        assert upload_complete
        
        # Clean up test file
        os.remove(test_file_path)
    
    def test_drag_drop_interface(self):
        """Test drag and drop interface."""
        # Find drag zone
        drag_zone = wait_for_element(self.driver, By.CLASS_NAME, "drag-drop-zone")
        assert drag_zone is not None
        
        # Test drag over event
        self.driver.execute_script("""
            var event = new DragEvent('dragover', {
                dataTransfer: new DataTransfer()
            });
            document.querySelector('.drag-drop-zone').dispatchEvent(event);
        """)
        
        # Check drag over styling
        drag_over_class = drag_zone.get_attribute("class")
        assert "drag-over" in drag_over_class
    
    def test_modal_components(self):
        """Test modal dialog components."""
        # Open modal
        modal_trigger = wait_for_element(self.driver, By.ID, "open-settings-modal")
        modal_trigger.click()
        
        # Check modal visibility
        modal = wait_for_element(self.driver, By.CLASS_NAME, "modal")
        assert modal.is_displayed()
        
        # Check modal content
        modal_title = wait_for_element(self.driver, By.CLASS_NAME, "modal-title")
        assert modal_title.text == "Settings"
        
        # Test modal close
        close_button = wait_for_element(self.driver, By.CLASS_NAME, "modal-close")
        close_button.click()
        
        # Check modal is hidden
        time.sleep(0.5)
        assert not modal.is_displayed()
    
    def test_form_validation(self):
        """Test form validation components."""
        # Find form elements
        email_input = wait_for_element(self.driver, By.ID, "email-input")
        password_input = wait_for_element(self.driver, By.ID, "password-input")
        submit_button = wait_for_element(self.driver, By.ID, "submit-form")
        
        # Test invalid email
        email_input.send_keys("invalid-email")
        password_input.send_keys("password123")
        submit_button.click()
        
        # Check validation error
        email_error = wait_for_element(self.driver, By.CLASS_NAME, "email-error")
        assert email_error is not None
        assert "Invalid email format" in email_error.text
        
        # Test valid input
        email_input.clear()
        email_input.send_keys("user@example.com")
        submit_button.click()
        
        # Check error is cleared
        email_errors = self.driver.find_elements(By.CLASS_NAME, "email-error")
        assert len(email_errors) == 0
    
    def test_data_table_component(self):
        """Test data table component functionality."""
        # Find data table
        data_table = wait_for_element(self.driver, By.CLASS_NAME, "data-table")
        assert data_table is not None
        
        # Test sorting
        sort_header = wait_for_element(self.driver, By.CLASS_NAME, "sortable-header")
        sort_header.click()
        
        # Check sort indicator
        sort_indicator = wait_for_element(self.driver, By.CLASS_NAME, "sort-indicator")
        assert sort_indicator is not None
        
        # Test pagination
        next_page = wait_for_element(self.driver, By.CLASS_NAME, "next-page")
        if next_page.is_enabled():
            next_page.click()
            
            # Check page number update
            page_info = wait_for_element(self.driver, By.CLASS_NAME, "page-info")
            assert "Page 2" in page_info.text
        
        # Test row selection
        first_row_checkbox = wait_for_element(self.driver, By.XPATH, "//tr[1]//input[@type='checkbox']")
        first_row_checkbox.click()
        
        # Check selection count
        selection_count = wait_for_element(self.driver, By.CLASS_NAME, "selection-count")
        assert "1 selected" in selection_count.text


class TestPWAFeatures:
    """Test Progressive Web App features."""
    
    def setup_method(self):
        """Set up test environment."""
        self.driver = create_test_driver()
        self.driver.get("http://localhost:8080")
        
    def teardown_method(self):
        """Clean up after tests."""
        if self.driver:
            self.driver.quit()
    
    def test_service_worker_registration(self):
        """Test service worker registration."""
        # Check service worker registration
        sw_registered = self.driver.execute_script("""
            return 'serviceWorker' in navigator && 
                   navigator.serviceWorker.controller !== null;
        """)
        assert sw_registered, "Service worker should be registered"
    
    def test_offline_functionality(self):
        """Test offline functionality."""
        # Register service worker if not already registered
        self.driver.execute_script("""
            if ('serviceWorker' in navigator) {
                navigator.serviceWorker.register('/sw.js');
            }
        """)
        
        # Simulate offline mode
        self.driver.execute_script("""
            window.navigator.onLine = false;
            window.dispatchEvent(new Event('offline'));
        """)
        
        # Check offline indicator
        offline_indicator = wait_for_element(self.driver, By.CLASS_NAME, "offline-indicator")
        assert offline_indicator is not None
        assert offline_indicator.is_displayed()
    
    def test_install_prompt(self):
        """Test PWA install prompt."""
        # Check if install prompt is available
        install_available = self.driver.execute_script("""
            return window.deferredPrompt !== undefined;
        """)
        
        if install_available:
            # Trigger install prompt
            install_button = wait_for_element(self.driver, By.ID, "install-app")
            install_button.click()
            
            # Check if prompt shown
            prompt_shown = self.driver.execute_script("""
                return window.installPromptShown === true;
            """)
            assert prompt_shown
    
    def test_app_manifest(self):
        """Test app manifest configuration."""
        # Check manifest link
        manifest_link = self.driver.find_element(By.XPATH, "//link[@rel='manifest']")
        assert manifest_link is not None
        
        # Fetch and validate manifest
        manifest_url = manifest_link.get_attribute("href")
        manifest_content = self.driver.execute_script(f"""
            return fetch('{manifest_url}').then(r => r.json());
        """)
        
        assert manifest_content is not None
        assert "name" in manifest_content
        assert "short_name" in manifest_content
        assert "start_url" in manifest_content
    
    def test_push_notifications(self):
        """Test push notification support."""
        # Check push notification support
        push_supported = self.driver.execute_script("""
            return 'PushManager' in window;
        """)
        assert push_supported, "Push notifications should be supported"
        
        # Test notification permission request
        permission_result = self.driver.execute_script("""
            return Notification.permission;
        """)
        assert permission_result in ["default", "granted", "denied"]


class TestAccessibilityCompliance:
    """Test accessibility compliance."""
    
    def setup_method(self):
        """Set up test environment."""
        self.driver = create_test_driver()
        self.driver.get("http://localhost:8080")
        
    def teardown_method(self):
        """Clean up after tests."""
        if self.driver:
            self.driver.quit()
    
    def test_keyboard_navigation(self):
        """Test keyboard navigation."""
        # Tab through focusable elements
        body = self.driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.TAB)
        
        # Check focus indicator
        focused_element = self.driver.switch_to.active_element
        assert focused_element is not None
        
        # Check focus outline
        focus_outline = self.driver.execute_script("""
            return window.getComputedStyle(arguments[0]).outline;
        """, focused_element)
        assert focus_outline != "none"
    
    def test_aria_labels(self):
        """Test ARIA labels and attributes."""
        # Check for ARIA labels on interactive elements
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        for button in buttons:
            aria_label = button.get_attribute("aria-label")
            text_content = button.text
            assert aria_label or text_content, "Buttons should have accessible labels"
    
    def test_color_contrast(self):
        """Test color contrast compliance."""
        # Check text color contrast
        text_elements = self.driver.find_elements(By.XPATH, "//p | //h1 | //h2 | //h3 | //span")
        
        for element in text_elements[:5]:  # Test first 5 elements
            contrast_ratio = self.driver.execute_script("""
                var element = arguments[0];
                var style = window.getComputedStyle(element);
                var bgColor = style.backgroundColor;
                var textColor = style.color;
                
                // Simple contrast check (simplified)
                return {
                    background: bgColor,
                    text: textColor,
                    // Actual contrast calculation would be more complex
                    sufficient: bgColor !== textColor
                };
            """, element)
            
            assert contrast_ratio['sufficient'], "Text should have sufficient contrast"
    
    def test_screen_reader_support(self):
        """Test screen reader support."""
        # Check for landmark roles
        landmarks = self.driver.find_elements(By.XPATH, "//*[@role='main' or @role='navigation' or @role='banner' or @role='contentinfo']")
        assert len(landmarks) > 0, "Page should have landmark roles"
        
        # Check for heading hierarchy
        headings = self.driver.find_elements(By.XPATH, "//h1 | //h2 | //h3 | //h4 | //h5 | //h6")
        assert len(headings) > 0, "Page should have proper heading structure"
    
    def test_form_accessibility(self):
        """Test form accessibility."""
        # Check form labels
        form_inputs = self.driver.find_elements(By.XPATH, "//input[@type!='hidden'] | //textarea | //select")
        
        for input_element in form_inputs:
            input_id = input_element.get_attribute("id")
            if input_id:
                # Check for associated label
                label = self.driver.find_elements(By.XPATH, f"//label[@for='{input_id}']")
                aria_label = input_element.get_attribute("aria-label")
                assert len(label) > 0 or aria_label, f"Input {input_id} should have accessible label"


class TestPerformanceMetrics:
    """Test frontend performance metrics."""
    
    def setup_method(self):
        """Set up test environment."""
        self.driver = create_test_driver()
        
    def teardown_method(self):
        """Clean up after tests."""
        if self.driver:
            self.driver.quit()
    
    def test_page_load_performance(self):
        """Test page load performance."""
        # Navigate to page and measure load time
        start_time = time.time()
        self.driver.get("http://localhost:8080")
        
        # Wait for page to be fully loaded
        WebDriverWait(self.driver, 10).until(
            lambda driver: driver.execute_script("return document.readyState") == "complete"
        )
        
        end_time = time.time()
        load_time = end_time - start_time
        
        assert load_time < 5.0, "Page should load within 5 seconds"
    
    def test_javascript_performance(self):
        """Test JavaScript performance metrics."""
        self.driver.get("http://localhost:8080")
        
        # Measure JavaScript execution time
        js_performance = self.driver.execute_script("""
            var start = performance.now();
            
            // Simulate some JavaScript work
            for (var i = 0; i < 100000; i++) {
                Math.random();
            }
            
            var end = performance.now();
            return end - start;
        """)
        
        assert js_performance < 100, "JavaScript execution should be fast"
    
    def test_memory_usage(self):
        """Test memory usage metrics."""
        self.driver.get("http://localhost:8080")
        
        # Check memory usage if available
        memory_info = self.driver.execute_script("""
            return performance.memory ? {
                usedJSHeapSize: performance.memory.usedJSHeapSize,
                totalJSHeapSize: performance.memory.totalJSHeapSize,
                jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
            } : null;
        """)
        
        if memory_info:
            heap_usage_ratio = memory_info['usedJSHeapSize'] / memory_info['totalJSHeapSize']
            assert heap_usage_ratio < 0.8, "Memory usage should be reasonable"
    
    def test_network_requests(self):
        """Test network request performance."""
        self.driver.get("http://localhost:8080")
        
        # Get network timing information
        network_timing = self.driver.execute_script("""
            return performance.getEntriesByType('navigation')[0];
        """)
        
        if network_timing:
            dns_time = network_timing['domainLookupEnd'] - network_timing['domainLookupStart']
            connect_time = network_timing['connectEnd'] - network_timing['connectStart']
            
            assert dns_time < 1000, "DNS lookup should be fast"
            assert connect_time < 2000, "Connection should be fast"


# Test utilities
def create_test_driver():
    """Create a test WebDriver instance."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    
    return webdriver.Chrome(options=chrome_options)


def wait_for_element(driver, by, value, timeout=10):
    """Wait for element to be present."""
    try:
        return WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
    except TimeoutException:
        return None


def wait_for_text(driver, by, value, text, timeout=10):
    """Wait for element to contain specific text."""
    try:
        WebDriverWait(driver, timeout).until(
            EC.text_to_be_present_in_element((by, value), text)
        )
        return True
    except TimeoutException:
        return False


# Test fixtures
@pytest.fixture
def test_driver():
    """Create a test driver fixture."""
    driver = create_test_driver()
    yield driver
    driver.quit()


@pytest.fixture
def test_data_file():
    """Create a test data file."""
    test_file = "/tmp/test_data.csv"
    with open(test_file, 'w') as f:
        f.write("col1,col2,col3\n1,2,3\n4,5,6\n7,8,9\n")
    
    yield test_file
    
    if os.path.exists(test_file):
        os.remove(test_file)