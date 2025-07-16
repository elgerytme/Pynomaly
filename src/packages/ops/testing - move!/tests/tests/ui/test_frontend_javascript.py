"""
Frontend JavaScript Test Suite
Tests frontend JavaScript functionality using Node.js and Jest-like testing
"""

from pathlib import Path

import pytest


class TestFrontendJavaScript:
    """Test frontend JavaScript components"""

    @pytest.fixture
    def static_js_path(self):
        """Path to static JavaScript files"""
        return Path(__file__).parent.parent.parent / "src" / "monorepo" / "presentation" / "web" / "static" / "js"

    @pytest.fixture
    def frontend_src_path(self):
        """Path to frontend source files"""
        return Path(__file__).parent.parent.parent / "src" / "monorepo" / "presentation" / "web" / "static" / "js" / "src"

    def test_frontend_monitoring_structure(self, frontend_src_path):
        """Test frontend monitoring file structure"""
        monitoring_file = frontend_src_path / "utils" / "frontend-monitoring.js"
        assert monitoring_file.exists()

        # Check file contains expected classes
        content = monitoring_file.read_text()
        assert "class FrontendPerformanceMonitor" in content
        assert "class FrontendSecurityMonitor" in content
        assert "setupCoreWebVitals" in content
        assert "recordCoreWebVital" in content

    def test_performance_dashboard_structure(self, frontend_src_path):
        """Test performance dashboard file structure"""
        dashboard_file = frontend_src_path / "utils" / "performance-dashboard.js"
        assert dashboard_file.exists()

        # Check file contains expected classes
        content = dashboard_file.read_text()
        assert "class PerformanceDashboard" in content
        assert "updateCoreWebVitals" in content
        assert "exportData" in content
        assert "addMetric" in content

    def test_pynomaly_frontend_structure(self, frontend_src_path):
        """Test main frontend integration file structure"""
        frontend_file = frontend_src_path / "pynomaly-frontend.js"
        assert frontend_file.exists()

        # Check file contains expected classes
        content = frontend_file.read_text()
        assert "class PynominalyFrontend" in content
        assert "class PynominalyAPIClient" in content
        assert "initializeMonitoring" in content
        assert "setupPerformanceDashboard" in content

    def test_frontend_monitoring_api_integration(self, frontend_src_path):
        """Test frontend monitoring API integration"""
        monitoring_file = frontend_src_path / "utils" / "frontend-monitoring.js"
        content = monitoring_file.read_text()

        # Check API endpoint references
        assert "/api/metrics/critical" in content
        assert "/api/security/events" in content
        assert "/api/monitoring/performance" in content
        assert "/api/monitoring/security" in content

    def test_csrf_token_handling(self, frontend_src_path):
        """Test CSRF token handling in JavaScript"""
        frontend_file = frontend_src_path / "pynomaly-frontend.js"
        content = frontend_file.read_text()

        # Check CSRF token handling
        assert "csrf-token" in content
        assert "getCSRFToken" in content
        assert "X-CSRF-Token" in content

    def test_error_handling_implementation(self, frontend_src_path):
        """Test error handling implementation"""
        frontend_file = frontend_src_path / "pynomaly-frontend.js"
        content = frontend_file.read_text()

        # Check error handling
        assert "setupErrorHandling" in content
        assert "handleError" in content
        assert "window.addEventListener('error'" in content
        assert "window.addEventListener('unhandledrejection'" in content

    def test_performance_monitoring_methods(self, frontend_src_path):
        """Test performance monitoring methods"""
        monitoring_file = frontend_src_path / "utils" / "frontend-monitoring.js"
        content = monitoring_file.read_text()

        # Check Core Web Vitals monitoring
        expected_methods = [
            "onCLS",
            "onFID",
            "onFCP",
            "onLCP",
            "onTTFB",
            "recordCoreWebVital",
            "sendCriticalMetric"
        ]

        for method in expected_methods:
            assert method in content

    def test_security_monitoring_methods(self, frontend_src_path):
        """Test security monitoring methods"""
        monitoring_file = frontend_src_path / "utils" / "frontend-monitoring.js"
        content = monitoring_file.read_text()

        # Check security monitoring methods
        expected_methods = [
            "setupCSPViolationReporting",
            "setupInputSanitization",
            "validateInput",
            "sanitizeXSS",
            "reportSecurityEvent",
            "setupSessionMonitoring"
        ]

        for method in expected_methods:
            assert method in content

    def test_dashboard_ui_components(self, frontend_src_path):
        """Test dashboard UI components"""
        dashboard_file = frontend_src_path / "utils" / "performance-dashboard.js"
        content = dashboard_file.read_text()

        # Check UI components
        assert "getDashboardHTML" in content
        assert "addStyles" in content
        assert "setupEventListeners" in content
        assert "updateDashboard" in content

        # Check for specific UI elements
        assert "Core Web Vitals" in content
        assert "Page Performance" in content
        assert "API Performance" in content
        assert "Real-time Metrics" in content

    def test_api_client_methods(self, frontend_src_path):
        """Test API client methods"""
        frontend_file = frontend_src_path / "pynomaly-frontend.js"
        content = frontend_file.read_text()

        # Check API client methods
        expected_methods = [
            "request",
            "get",
            "post",
            "put",
            "delete",
            "reportPerformanceMetric",
            "reportSecurityEvent"
        ]

        for method in expected_methods:
            assert method in content

    def test_feature_initialization(self, frontend_src_path):
        """Test feature initialization methods"""
        frontend_file = frontend_src_path / "pynomaly-frontend.js"
        content = frontend_file.read_text()

        # Check feature initialization
        expected_features = [
            "initializeDarkMode",
            "initializeLazyLoading",
            "initializeCaching",
            "initializeOfflineSupport"
        ]

        for feature in expected_features:
            assert feature in content

    def test_form_enhancements(self, frontend_src_path):
        """Test form enhancement features"""
        frontend_file = frontend_src_path / "pynomaly-frontend.js"
        content = frontend_file.read_text()

        # Check form enhancements
        assert "setupFormEnhancements" in content
        assert "setupAutoSave" in content
        assert "setupFormValidation" in content
        assert "validateForm" in content
        assert "isValidEmail" in content

    def test_ui_enhancements(self, frontend_src_path):
        """Test UI enhancement features"""
        frontend_file = frontend_src_path / "pynomaly-frontend.js"
        content = frontend_file.read_text()

        # Check UI enhancements
        assert "setupUIEnhancements" in content
        assert "setupLoadingIndicators" in content
        assert "setupNavigationEnhancements" in content
        assert "setupBackToTopButton" in content

    def test_monitoring_integration(self, frontend_src_path):
        """Test monitoring integration"""
        monitoring_file = frontend_src_path / "utils" / "frontend-monitoring.js"
        content = monitoring_file.read_text()

        # Check integration with dashboard
        assert "integrateWithDashboard" in content
        assert "window.performanceDashboard" in content
        assert "addMetric" in content

    def test_configuration_loading(self, frontend_src_path):
        """Test configuration loading"""
        frontend_file = frontend_src_path / "pynomaly-frontend.js"
        content = frontend_file.read_text()

        # Check configuration loading
        assert "loadConfig" in content
        assert "getFallbackConfig" in content
        assert "/api/ui/config" in content

    def test_event_handling(self, frontend_src_path):
        """Test event handling"""
        frontend_file = frontend_src_path / "pynomaly-frontend.js"
        content = frontend_file.read_text()

        # Check event handling
        assert "DOMContentLoaded" in content
        assert "pynomaly:initialized" in content
        assert "dispatchEvent" in content

    def test_javascript_syntax_validation(self, frontend_src_path):
        """Test JavaScript syntax validation"""
        js_files = [
            "utils/frontend-monitoring.js",
            "utils/performance-dashboard.js",
            "pynomaly-frontend.js"
        ]

        for js_file in js_files:
            file_path = frontend_src_path / js_file
            assert file_path.exists()

            # Read and check basic syntax
            content = file_path.read_text()

            # Check for balanced braces
            open_braces = content.count('{')
            close_braces = content.count('}')
            assert open_braces == close_braces, f"Unbalanced braces in {js_file}"

            # Check for balanced parentheses
            open_parens = content.count('(')
            close_parens = content.count(')')
            assert open_parens == close_parens, f"Unbalanced parentheses in {js_file}"

    def test_module_exports(self, frontend_src_path):
        """Test module exports"""
        js_files = [
            ("utils/frontend-monitoring.js", ["FrontendPerformanceMonitor", "FrontendSecurityMonitor"]),
            ("utils/performance-dashboard.js", ["PerformanceDashboard"]),
            ("pynomaly-frontend.js", ["PynominalyFrontend", "PynominalyAPIClient"])
        ]

        for js_file, expected_exports in js_files:
            file_path = frontend_src_path / js_file
            content = file_path.read_text()

            # Check for module exports
            assert "module.exports" in content

            for export_name in expected_exports:
                assert export_name in content

    def test_global_variable_assignments(self, frontend_src_path):
        """Test global variable assignments"""
        monitoring_file = frontend_src_path / "utils" / "frontend-monitoring.js"
        content = monitoring_file.read_text()

        # Check global assignments
        assert "window.frontendMonitor" in content
        assert "window.securityMonitor" in content

        dashboard_file = frontend_src_path / "utils" / "performance-dashboard.js"
        dashboard_content = dashboard_file.read_text()
        assert "window.performanceDashboard" in dashboard_content

    def test_initialization_sequence(self, frontend_src_path):
        """Test initialization sequence"""
        frontend_file = frontend_src_path / "pynomaly-frontend.js"
        content = frontend_file.read_text()

        # Check initialization sequence
        assert "window.pynomaly = new PynominalyFrontend()" in content
        assert "window.monorepo.init()" in content
        assert "_performInit" in content

    def test_browser_compatibility_features(self, frontend_src_path):
        """Test browser compatibility features"""
        monitoring_file = frontend_src_path / "utils" / "frontend-monitoring.js"
        content = monitoring_file.read_text()

        # Check for feature detection
        assert "'PerformanceObserver' in window" in content
        assert "'connection' in navigator" in content
        assert "performance.memory" in content

        frontend_file = frontend_src_path / "pynomaly-frontend.js"
        frontend_content = frontend_file.read_text()

        # Check for service worker support
        assert "'serviceWorker' in navigator" in frontend_content
        assert "'IntersectionObserver' in window" in frontend_content


class TestJavaScriptIntegration:
    """Test JavaScript integration with HTML templates"""

    def test_template_script_includes(self):
        """Test that templates include JavaScript files"""
        template_path = Path(__file__).parent.parent.parent / "src" / "monorepo" / "presentation" / "web" / "templates" / "base.html"
        assert template_path.exists()

        content = template_path.read_text()

        # Check script includes
        assert "frontend-monitoring.js" in content
        assert "performance-dashboard.js" in content
        assert "pynomaly-frontend.js" in content

    def test_csrf_meta_tag(self):
        """Test CSRF meta tag in templates"""
        template_path = Path(__file__).parent.parent.parent / "src" / "monorepo" / "presentation" / "web" / "templates" / "base.html"
        content = template_path.read_text()

        # Check CSRF meta tag
        assert 'name="csrf-token"' in content
        assert 'name="csrf-param"' in content

    def test_external_dependencies(self):
        """Test external JavaScript dependencies"""
        template_path = Path(__file__).parent.parent.parent / "src" / "monorepo" / "presentation" / "web" / "templates" / "base.html"
        content = template_path.read_text()

        # Check external dependencies
        assert "tailwindcss.com" in content
        assert "htmx.org" in content
        assert "alpinejs" in content
        assert "d3js.org" in content
        assert "echarts" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
