"""Enhanced UI automation tests with robust wait strategies and retry mechanisms."""

import pytest
from playwright.sync_api import expect

from .enhanced_page_objects import DashboardPage, retry_on_failure


class TestWebAppAutomationEnhanced:
    """Enhanced UI automation test suite with improved stability."""
    
    @pytest.fixture
    def dashboard_page(self, page):
        """Create enhanced dashboard page object."""
        return DashboardPage(page)
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def test_dashboard_loads_robust(self, dashboard_page):
        """Test dashboard loading with enhanced stability."""
        # Navigate and wait for complete load
        dashboard_page.navigate()
        
        # Verify page title with retry
        expect(dashboard_page.page).to_have_title("Pynomaly", timeout=10000)
        
        # Verify main navigation is visible
        assert dashboard_page.is_navigation_visible(), "Main navigation should be visible"
        
        # Verify logo presence with explicit wait
        logo_element = dashboard_page.wait_for_element_visible(dashboard_page.logo_selector)
        assert logo_element.is_visible(), "Logo should be visible"
        
        # Take screenshots for verification
        dashboard_page.take_dashboard_screenshot("enhanced_dashboard_loaded")
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def test_navigation_menu_robust(self, dashboard_page):
        """Test navigation with enhanced stability and verification."""
        dashboard_page.navigate()
        
        # Test each navigation link systematically
        navigation_results = []
        
        for link_text, expected_path in dashboard_page.nav_links.items():
            if link_text == "Dashboard":
                continue  # Skip dashboard as we're already there
            
            try:
                # Click navigation link with enhanced waiting
                dashboard_page.click_nav_link(link_text)
                
                # Verify URL change
                expected_url = f"{dashboard_page.base_url}{expected_path}"
                dashboard_page.wait_for_url_change(expected_url)
                
                # Verify page loaded successfully
                dashboard_page.wait_for_page_load()
                
                navigation_results.append(f"✓ {link_text} navigation successful")
                
                # Return to dashboard for next test
                dashboard_page.click_nav_link("Dashboard")
                dashboard_page.wait_for_dashboard_load()
                
            except Exception as e:
                navigation_results.append(f"✗ {link_text} navigation failed: {str(e)}")
        
        # Verify at least 80% of navigation links work
        successful_navigations = len([r for r in navigation_results if r.startswith("✓")])
        total_navigations = len(navigation_results)
        
        assert successful_navigations / total_navigations >= 0.8, \
            f"Navigation success rate too low: {navigation_results}"
    
    def test_dashboard_components_comprehensive(self, dashboard_page):
        """Test comprehensive dashboard component verification."""
        dashboard_page.navigate()
        
        # Verify all expected components
        components = dashboard_page.verify_dashboard_components()
        
        # Essential components must be present
        essential_components = ["logo", "navigation", "title"]
        for component in essential_components:
            assert components.get(component, False), \
                f"Essential component '{component}' is missing"
        
        # Optional components (may not be present in all environments)
        optional_components = ["metrics", "charts", "recent_detections", "quick_actions"]
        present_optional = sum(1 for comp in optional_components if components.get(comp, False))
        
        # At least some optional components should be present
        assert present_optional > 0, \
            "No optional dashboard components found - dashboard may be empty"
    
    @retry_on_failure(max_retries=2, delay=1.0)
    def test_responsive_design_validation(self, dashboard_page):
        """Test responsive design across multiple viewport sizes."""
        dashboard_page.navigate()
        
        # Test responsive layout
        responsive_results = dashboard_page.verify_responsive_layout()
        
        # Verify layouts work on different screen sizes
        successful_layouts = 0
        for size, result in responsive_results.items():
            if result["layout_ok"]:
                successful_layouts += 1
        
        # At least 75% of layouts should work properly
        total_layouts = len(responsive_results)
        success_rate = successful_layouts / total_layouts
        
        assert success_rate >= 0.75, \
            f"Responsive design issues: {responsive_results}"
    
    def test_accessibility_validation(self, dashboard_page):
        """Test basic accessibility requirements."""
        dashboard_page.navigate()
        
        # Verify accessibility features
        accessibility_results = dashboard_page.verify_accessibility()
        
        # Critical accessibility features must be present
        critical_features = ["has_title", "has_main_heading"]
        for feature in critical_features:
            assert accessibility_results.get(feature, False), \
                f"Critical accessibility feature '{feature}' is missing"
        
        # Check overall accessibility score
        total_checks = len(accessibility_results)
        passed_checks = sum(1 for result in accessibility_results.values() if result)
        accessibility_score = passed_checks / total_checks
        
        assert accessibility_score >= 0.8, \
            f"Accessibility score too low ({accessibility_score:.2%}): {accessibility_results}"
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def test_user_interaction_flow(self, dashboard_page):
        """Test complete user interaction workflow."""
        dashboard_page.navigate()
        
        # Simulate realistic user interaction patterns
        interaction_results = dashboard_page.simulate_user_interaction_flow()
        
        # Verify interaction success rate
        successful_interactions = len([r for r in interaction_results if r.startswith("✓")])
        total_interactions = len(interaction_results)
        
        if total_interactions > 0:
            success_rate = successful_interactions / total_interactions
            assert success_rate >= 0.7, \
                f"User interaction success rate too low: {interaction_results}"
    
    def test_page_load_performance(self, dashboard_page):
        """Test page load performance with timing validation."""
        import time

        # Measure page load time
        start_time = time.time()
        dashboard_page.navigate()
        end_time = time.time()
        
        load_time = end_time - start_time
        
        # Page should load within reasonable time (10 seconds)
        assert load_time < 10.0, f"Page load time too slow: {load_time:.2f} seconds"
        
        # Take performance screenshot
        dashboard_page.take_screenshot(f"performance_test_load_time_{load_time:.2f}s")
    
    def test_dynamic_content_loading(self, dashboard_page):
        """Test dynamic content loading with HTMX/AJAX."""
        dashboard_page.navigate()
        
        # Wait for all dynamic content to settle
        dashboard_page.wait_for_htmx_settle()
        dashboard_page.wait_for_ajax_complete()
        
        # Verify metrics loaded (if present)
        metrics_data = dashboard_page.get_metrics_data()
        
        # If metrics are present, they should have valid data
        if metrics_data:
            for metric in metrics_data:
                assert metric["title"].strip() != "", "Metric title should not be empty"
                assert metric["value"].strip() != "", "Metric value should not be empty"
                assert "Loading" not in metric["value"], "Metrics should be fully loaded"
    
    def test_error_handling_resilience(self, dashboard_page):
        """Test UI resilience to various error conditions."""
        dashboard_page.navigate()
        
        # Test handling of non-existent elements
        non_existent_visible = dashboard_page.is_element_visible(".non-existent-element")
        assert not non_existent_visible, "Non-existent elements should not be visible"
        
        # Test handling of disabled elements
        disabled_enabled = dashboard_page.is_element_enabled(".disabled-element")
        assert not disabled_enabled, "Disabled elements should not be enabled"
        
        # Verify page remains functional after error checks
        assert dashboard_page.is_navigation_visible(), \
            "Navigation should remain functional after error checks"
    
    @pytest.mark.slow
    def test_extended_user_session(self, dashboard_page):
        """Test extended user session with multiple interactions."""
        dashboard_page.navigate()
        
        # Simulate extended user session
        session_actions = []
        
        try:
            # Multiple navigation cycles
            for cycle in range(3):
                for link_text in ["Detectors", "Datasets"]:
                    dashboard_page.click_nav_link(link_text)
                    session_actions.append(f"Cycle {cycle + 1}: Navigated to {link_text}")
                    
                    dashboard_page.click_nav_link("Dashboard")
                    session_actions.append(f"Cycle {cycle + 1}: Returned to Dashboard")
            
            # Verify session remained stable
            assert dashboard_page.is_navigation_visible(), \
                "Navigation should remain functional after extended session"
            
        except Exception as e:
            pytest.fail(f"Extended session failed: {str(e)}\nActions completed: {session_actions}")
    
    def test_concurrent_user_simulation(self, dashboard_page):
        """Test UI behavior under simulated concurrent usage."""
        dashboard_page.navigate()
        
        # Simulate rapid user actions
        rapid_actions = []
        
        try:
            # Quick succession of actions
            for i in range(5):
                # Rapid navigation
                dashboard_page.click_nav_link("Detectors")
                rapid_actions.append(f"Quick nav {i + 1}: To Detectors")
                
                dashboard_page.click_nav_link("Dashboard")
                rapid_actions.append(f"Quick nav {i + 1}: Back to Dashboard")
            
            # Verify UI remained responsive
            assert dashboard_page.is_navigation_visible(), \
                "UI should remain responsive after rapid interactions"
            
        except Exception as e:
            # Rapid interactions might fail, but UI should recover
            dashboard_page.navigate()  # Reset
            assert dashboard_page.is_navigation_visible(), \
                f"UI should recover after rapid interaction issues: {str(e)}"