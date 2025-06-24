#!/usr/bin/env python3
"""
Standalone UI automation tests that don't depend on main Pynomaly conftest.
This allows testing the UI infrastructure without complex dependencies.
"""

import pytest
from playwright.sync_api import sync_playwright, Page, Browser
from pathlib import Path
import time


class TestPynomalaUIStandalone:
    """Standalone UI tests for Pynomaly web application."""
    
    @pytest.fixture(scope="class")
    def browser_setup(self):
        """Set up browser for testing."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            yield browser
            browser.close()
    
    @pytest.fixture
    def page_setup(self, browser_setup):
        """Set up page for each test."""
        context = browser_setup.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        yield page
        context.close()
    
    def test_dashboard_loads(self, page_setup):
        """Test that the dashboard loads successfully."""
        page = page_setup
        page.goto("http://localhost:8000/")
        page.wait_for_load_state("networkidle")
        
        # Check title
        title = page.title()
        assert "Pynomaly" in title
        
        # Check main elements
        assert page.locator("#main-navigation").is_visible()
        assert page.locator("#logo").is_visible()
        assert page.locator("#main-content").is_visible()
        
        # Take screenshot
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        page.screenshot(path=str(screenshots_dir / "test_dashboard_loads.png"), full_page=True)
    
    def test_navigation_menu(self, page_setup):
        """Test navigation menu functionality."""
        page = page_setup
        page.goto("http://localhost:8000/")
        page.wait_for_load_state("networkidle")
        
        # Check navigation links
        nav_links = [
            ("Detectors", "/detectors"),
            ("Datasets", "/datasets"), 
            ("Detection", "/detection"),
            ("Visualizations", "/visualizations"),
            ("Exports", "/exports")
        ]
        
        for link_text, expected_url in nav_links:
            link = page.locator(f'nav a:has-text("{link_text}")')
            assert link.is_visible()
            
            # Click and verify navigation
            link.click()
            page.wait_for_load_state("networkidle")
            assert expected_url in page.url
    
    def test_mobile_responsiveness(self, page_setup):
        """Test mobile responsiveness."""
        page = page_setup
        
        # Test mobile viewport
        page.set_viewport_size({"width": 375, "height": 667})
        page.goto("http://localhost:8000/")
        page.wait_for_load_state("networkidle")
        
        # Check mobile menu button exists
        mobile_menu_btn = page.locator("#mobile-menu-btn")
        assert mobile_menu_btn.is_visible()
        
        # Test mobile menu functionality
        mobile_menu = page.locator("#mobile-menu")
        assert mobile_menu.is_hidden()
        
        mobile_menu_btn.click()
        assert mobile_menu.is_visible()
        
        # Take mobile screenshot
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        page.screenshot(path=str(screenshots_dir / "test_mobile_menu.png"), full_page=True)
    
    def test_page_elements(self, page_setup):
        """Test key page elements on dashboard."""
        page = page_setup
        page.goto("http://localhost:8000/")
        page.wait_for_load_state("networkidle")
        
        # Check statistics cards
        assert page.locator("#stats-cards").is_visible()
        assert page.locator("#detector-count").is_visible()
        assert page.locator("#dataset-count").is_visible()
        assert page.locator("#results-count").is_visible()
        
        # Check recent results section
        assert page.locator("#recent-results").is_visible()
        
        # Check quick actions
        assert page.locator("#quick-actions").is_visible()
        assert page.locator("#quick-detection-btn").is_visible()
        assert page.locator("#upload-dataset-btn").is_visible()
        assert page.locator("#autonomous-mode-btn").is_visible()
    
    def test_detectors_page(self, page_setup):
        """Test detectors page functionality."""
        page = page_setup
        page.goto("http://localhost:8000/detectors")
        page.wait_for_load_state("networkidle")
        
        # Check page title
        page_title = page.locator("#page-title")
        assert page_title.is_visible()
        assert "Detectors" in page_title.text_content()
        
        # Check detectors section
        assert page.locator("#detectors-section").is_visible()
        
        # Take screenshot
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        page.screenshot(path=str(screenshots_dir / "test_detectors_page.png"), full_page=True)
    
    def test_datasets_page(self, page_setup):
        """Test datasets page functionality."""
        page = page_setup
        page.goto("http://localhost:8000/datasets")
        page.wait_for_load_state("networkidle")
        
        # Check page elements
        page_title = page.locator("#page-title")
        assert page_title.is_visible()
        assert "Datasets" in page_title.text_content()
        
        # Check upload functionality
        assert page.locator("#datasets-section").is_visible()
        assert page.locator("#upload-btn").is_visible()
        
        # Take screenshot
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        page.screenshot(path=str(screenshots_dir / "test_datasets_page.png"), full_page=True)
    
    def test_detection_page(self, page_setup):
        """Test detection page functionality."""
        page = page_setup
        page.goto("http://localhost:8000/detection")
        page.wait_for_load_state("networkidle")
        
        # Check form elements
        assert page.locator("#detection-form").is_visible()
        assert page.locator("#detector-select").is_visible()
        assert page.locator("#dataset-select").is_visible()
        assert page.locator("#run-detection-btn").is_visible()
        
        # Take screenshot
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        page.screenshot(path=str(screenshots_dir / "test_detection_page.png"), full_page=True)
    
    def test_visualizations_page(self, page_setup):
        """Test visualizations page."""
        page = page_setup
        page.goto("http://localhost:8000/visualizations")
        page.wait_for_load_state("networkidle")
        
        # Check visualization elements
        assert page.locator("#visualizations-section").is_visible()
        
        # Take screenshot
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        page.screenshot(path=str(screenshots_dir / "test_visualizations_page.png"), full_page=True)
    
    def test_exports_page(self, page_setup):
        """Test exports page functionality."""
        page = page_setup
        page.goto("http://localhost:8000/exports")
        page.wait_for_load_state("networkidle")
        
        # Check export elements
        assert page.locator("#exports-section").is_visible()
        assert page.locator("#export-format").is_visible()
        assert page.locator("#export-btn").is_visible()
        
        # Take screenshot
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        page.screenshot(path=str(screenshots_dir / "test_exports_page.png"), full_page=True)
    
    def test_health_endpoint(self, page_setup):
        """Test health endpoint."""
        page = page_setup
        response = page.goto("http://localhost:8000/health")
        assert response.status == 200
        
        # Check JSON response
        content = page.content()
        assert "healthy" in content
        assert "Pynomaly UI Test Server" in content
    
    def test_responsive_design_comprehensive(self, page_setup):
        """Test responsive design across multiple viewports."""
        page = page_setup
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        
        viewports = [
            ("desktop", {"width": 1920, "height": 1080}),
            ("laptop", {"width": 1366, "height": 768}),
            ("tablet", {"width": 768, "height": 1024}),
            ("mobile", {"width": 375, "height": 667})
        ]
        
        for device_name, viewport in viewports:
            page.set_viewport_size(viewport)
            page.goto("http://localhost:8000/")
            page.wait_for_load_state("networkidle")
            
            # Take screenshot for each viewport
            page.screenshot(
                path=str(screenshots_dir / f"responsive_{device_name}.png"),
                full_page=True
            )
            
            # Verify essential elements are visible
            assert page.locator("#logo").is_visible()
            assert page.locator("#main-content").is_visible()
    
    def test_performance_basic(self, page_setup):
        """Basic performance test - page load time."""
        page = page_setup
        
        start_time = time.time()
        page.goto("http://localhost:8000/")
        page.wait_for_load_state("networkidle")
        load_time = time.time() - start_time
        
        # Assert reasonable load time (less than 5 seconds)
        assert load_time < 5.0, f"Page load time too slow: {load_time:.2f}s"
        
        print(f"Dashboard load time: {load_time:.2f}s")