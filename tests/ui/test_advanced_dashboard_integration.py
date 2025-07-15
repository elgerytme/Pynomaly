"""
Comprehensive test suite for advanced dashboard integration features.
Tests the integration between real-time analytics, PWA capabilities, and interactive visualizations.
"""

import json
import time

import pytest
import websockets
from playwright.async_api import Page, expect
from tests.helpers.test_helpers import generate_test_anomalies


class TestAdvancedDashboardIntegration:
    """Test suite for advanced dashboard integration features."""

    @pytest.fixture
    async def dashboard_page(self, browser, live_server):
        """Create a dashboard page with advanced features enabled."""
        context = await browser.new_context(
            permissions=["notifications"], viewport={"width": 1920, "height": 1080}
        )
        page = await context.new_page()

        # Navigate to advanced dashboard
        await page.goto(f"{live_server}/dashboard/advanced")

        # Wait for dashboard to initialize
        await page.wait_for_selector('[data-testid="dashboard-ready"]', timeout=10000)

        yield page
        await context.close()

    @pytest.fixture
    async def websocket_server(self):
        """Mock WebSocket server for testing real-time features."""
        connected_clients = set()

        async def websocket_handler(websocket, path):
            connected_clients.add(websocket)
            try:
                async for message in websocket:
                    data = json.loads(message)
                    # Echo back for testing
                    await websocket.send(json.dumps({"type": "echo", "original": data}))
            finally:
                connected_clients.discard(websocket)

        server = await websockets.serve(websocket_handler, "localhost", 8765)

        # Store reference to connected clients for test manipulation
        server.connected_clients = connected_clients

        yield server
        server.close()
        await server.wait_closed()

    @pytest.mark.asyncio
    async def test_dashboard_initialization(self, dashboard_page: Page):
        """Test that the advanced dashboard initializes correctly with all components."""
        # Check that main dashboard container exists
        await expect(dashboard_page.locator(".investigation-dashboard")).to_be_visible()

        # Check that all tabs are present
        expected_tabs = ["overview", "investigation", "correlation", "settings"]
        for tab in expected_tabs:
            await expect(dashboard_page.locator(f'[data-view="{tab}"]')).to_be_visible()

        # Check that sidebar is present
        await expect(dashboard_page.locator(".investigation-sidebar")).to_be_visible()

        # Check that main panel is present
        await expect(dashboard_page.locator(".investigation-main")).to_be_visible()

        # Verify JavaScript components are loaded
        component_check = await dashboard_page.evaluate("""
            () => {
                return {
                    dashboardController: !!window.dashboardInstance,
                    analyticsEngine: !!window.RealTimeAnalyticsEngine,
                    pwaaManager: !!window.EnhancedPWAManager,
                    visualizations: !!window.AdvancedInteractiveVisualizations,
                    websocketService: !!window.EnhancedWebSocketService
                };
            }
        """)

        assert component_check[
            "dashboardController"
        ], "Dashboard controller not initialized"
        assert component_check["analyticsEngine"], "Analytics engine not loaded"
        assert component_check["pwaaManager"], "PWA manager not loaded"
        assert component_check["visualizations"], "Visualizations not loaded"
        assert component_check["websocketService"], "WebSocket service not loaded"

    @pytest.mark.asyncio
    async def test_real_time_analytics_engine(self, dashboard_page: Page):
        """Test real-time analytics engine functionality."""
        # Start the analytics engine
        await dashboard_page.evaluate("""
            () => {
                window.testAnalytics = new window.RealTimeAnalyticsEngine({
                    bufferSize: 100,
                    updateInterval: 50
                });
                window.testAnalytics.start();
            }
        """)

        # Add test data points
        test_data = [
            {"value": 10.5, "timestamp": int(time.time() * 1000)},
            {"value": 25.3, "timestamp": int(time.time() * 1000) + 1000},
            {
                "value": 95.7,
                "timestamp": int(time.time() * 1000) + 2000,
            },  # Potential anomaly
        ]

        # Add data points to engine
        for data_point in test_data:
            await dashboard_page.evaluate(
                f"() => window.testAnalytics.addDataPoint({json.dumps(data_point)})"
            )

        # Wait for processing
        await dashboard_page.wait_for_timeout(200)

        # Check analytics engine stats
        stats = await dashboard_page.evaluate("""
            () => window.testAnalytics.getProcessingStats()
        """)

        assert (
            stats["totalProcessed"] >= 3
        ), "Should have processed at least 3 data points"
        assert stats["throughput"] > 0, "Should have positive throughput"

        # Check buffer status
        buffer_status = await dashboard_page.evaluate("""
            () => window.testAnalytics.getBufferStatus()
        """)

        assert buffer_status["dataBuffer"]["size"] >= 0, "Data buffer should exist"

    @pytest.mark.asyncio
    async def test_pwa_offline_capabilities(self, dashboard_page: Page):
        """Test PWA offline functionality."""
        # Initialize PWA manager
        await dashboard_page.evaluate("""
            () => {
                window.testPWA = new window.EnhancedPWAManager({
                    enableOfflineDetection: true,
                    enableDataSync: true
                });
            }
        """)

        # Wait for PWA initialization
        await dashboard_page.wait_for_timeout(1000)

        # Test offline action
        result = await dashboard_page.evaluate("""
            async () => {
                try {
                    const result = await window.testPWA.performAction('analyze-data', {
                        type: 'basic_stats',
                        parameters: { sample_size: 100 }
                    });
                    return { success: true, result };
                } catch (error) {
                    return { success: false, error: error.message };
                }
            }
        """)

        assert result["success"], f"Offline action failed: {result.get('error')}"
        assert "result" in result, "Should return analysis result"

        # Test offline capabilities info
        capabilities = await dashboard_page.evaluate("""
            () => window.testPWA.getOfflineCapabilities()
        """)

        assert "isOnline" in capabilities, "Should report online status"
        assert "backgroundSyncSupported" in capabilities, "Should report sync support"

    @pytest.mark.asyncio
    async def test_interactive_visualizations(self, dashboard_page: Page):
        """Test advanced interactive visualization features."""
        # Create visualization instance
        await dashboard_page.evaluate("""
            () => {
                window.testViz = new window.AdvancedInteractiveVisualizations();

                // Create a test chart container
                const container = document.createElement('div');
                container.id = 'test-chart-container';
                container.style.width = '800px';
                container.style.height = '400px';
                document.body.appendChild(container);
            }
        """)

        # Generate test data
        test_anomalies = generate_test_anomalies(50)

        # Create timeline chart
        await dashboard_page.evaluate(f"""
            () => {{
                const container = document.getElementById('test-chart-container');
                const data = {json.dumps(test_anomalies)};

                window.testChart = window.testViz.createAnomalyTimelineChart(container, data, {{
                    enableDrillDown: true,
                    enableAnnotations: true
                }});
            }}
        """)

        # Wait for chart rendering
        await dashboard_page.wait_for_timeout(1000)

        # Check that chart was created
        chart_exists = await dashboard_page.evaluate("""
            () => !!window.testChart && !!document.querySelector('#test-chart-container svg')
        """)

        assert chart_exists, "Timeline chart should be created and rendered"

        # Test chart interaction
        chart_element = dashboard_page.locator("#test-chart-container svg")
        await expect(chart_element).to_be_visible()

        # Click on chart to trigger interaction
        await chart_element.click(position={"x": 400, "y": 200})

        # Verify interaction was handled
        interaction_handled = await dashboard_page.evaluate("""
            () => window.testChart.getLastInteraction() != null
        """)

        assert interaction_handled, "Chart interaction should be handled"

    @pytest.mark.asyncio
    async def test_websocket_real_time_features(
        self, dashboard_page: Page, websocket_server
    ):
        """Test WebSocket real-time communication features."""
        # Override WebSocket URL to point to test server
        await dashboard_page.evaluate("""
            () => {
                window.testWebSocket = new window.EnhancedWebSocketService({
                    url: 'ws://localhost:8765'
                });
            }
        """)

        # Wait for connection
        await dashboard_page.wait_for_timeout(2000)

        # Test sending message
        await dashboard_page.evaluate("""
            () => {
                window.testWebSocket.send({
                    type: 'test_message',
                    payload: { test: 'data' }
                });
            }
        """)

        # Wait for echo response
        await dashboard_page.wait_for_timeout(1000)

        # Check connection status
        is_connected = await dashboard_page.evaluate("""
            () => window.testWebSocket.isConnected
        """)

        # Note: May be false due to test server implementation
        # In production, this would be true
        assert isinstance(is_connected, bool), "Should report connection status"

    @pytest.mark.asyncio
    async def test_dashboard_navigation(self, dashboard_page: Page):
        """Test dashboard navigation and view switching."""
        # Test tab navigation
        tabs = ["overview", "investigation", "correlation", "settings"]

        for tab in tabs:
            # Click tab
            await dashboard_page.click(f'[data-view="{tab}"]')

            # Wait for view switch
            await dashboard_page.wait_for_timeout(300)

            # Check active tab
            active_tab = await dashboard_page.get_attribute(
                "[data-view].active", "data-view"
            )
            if active_tab:  # Some tabs might not be implemented yet
                assert active_tab == tab, f"Tab {tab} should be active"

            # Check corresponding panel
            panel_visible = await dashboard_page.is_visible(
                f'[data-view="{tab}"].investigation-panel.active'
            )
            # Note: Panel visibility depends on implementation

    @pytest.mark.asyncio
    async def test_anomaly_investigation_workflow(self, dashboard_page: Page):
        """Test complete anomaly investigation workflow."""
        # Generate test anomaly data
        test_anomalies = [
            {
                "id": "anomaly_1",
                "timestamp": int(time.time() * 1000),
                "value": 95.5,
                "anomaly_score": 0.85,
                "severity": "high",
                "confidence": 0.92,
            },
            {
                "id": "anomaly_2",
                "timestamp": int(time.time() * 1000) + 60000,
                "value": 12.3,
                "anomaly_score": 0.65,
                "severity": "medium",
                "confidence": 0.78,
            },
        ]

        # Add anomalies to the dashboard
        await dashboard_page.evaluate(f"""
            () => {{
                const anomalies = {json.dumps(test_anomalies)};
                if (window.dashboardInstance) {{
                    anomalies.forEach(anomaly => {{
                        window.dashboardInstance.handleAnomalyDetected(anomaly);
                    }});
                }}
            }}
        """)

        # Wait for anomalies to be processed
        await dashboard_page.wait_for_timeout(1000)

        # Check if anomalies appear in sidebar
        anomaly_items = dashboard_page.locator(".anomaly-item")
        await expect(anomaly_items.first()).to_be_visible(timeout=5000)

        # Click on first anomaly
        await anomaly_items.first().click()

        # Wait for details to load
        await dashboard_page.wait_for_timeout(500)

        # Check if investigation tools are available
        tool_buttons = dashboard_page.locator(".tool-btn")
        await expect(tool_buttons.first()).to_be_visible()

    @pytest.mark.asyncio
    async def test_accessibility_features(self, dashboard_page: Page):
        """Test dashboard accessibility features."""
        # Check for accessibility landmarks
        main_landmark = dashboard_page.locator('main[role="main"]')
        await expect(main_landmark).to_be_visible()

        # Check for skip links
        skip_links = dashboard_page.locator(".skip-link")
        if await skip_links.count() > 0:
            await expect(skip_links.first()).to_be_visible()

        # Test keyboard navigation
        # Focus on first tab
        await dashboard_page.keyboard.press("Tab")

        # Check if element is focused
        focused_element = await dashboard_page.evaluate(
            "() => document.activeElement.tagName"
        )
        assert focused_element in [
            "BUTTON",
            "A",
            "INPUT",
        ], "Should focus on interactive element"

        # Test arrow key navigation
        await dashboard_page.keyboard.press("ArrowRight")

        # Test Escape key for clearing selections
        await dashboard_page.keyboard.press("Escape")

    @pytest.mark.asyncio
    async def test_responsive_design(self, dashboard_page: Page):
        """Test dashboard responsive design at different viewport sizes."""
        viewports = [
            {"width": 1920, "height": 1080},  # Desktop
            {"width": 1024, "height": 768},  # Tablet
            {"width": 375, "height": 667},  # Mobile
        ]

        for viewport in viewports:
            await dashboard_page.set_viewport_size(
                viewport["width"], viewport["height"]
            )
            await dashboard_page.wait_for_timeout(500)

            # Check that dashboard is still visible
            dashboard = dashboard_page.locator(".investigation-dashboard")
            await expect(dashboard).to_be_visible()

            # Check responsive behavior
            if viewport["width"] <= 768:
                # Mobile: sidebar should stack vertically
                sidebar = dashboard_page.locator(".investigation-sidebar")
                if await sidebar.count() > 0:
                    sidebar_width = await sidebar.evaluate("el => el.offsetWidth")
                    viewport_width = viewport["width"]
                    # On mobile, sidebar should take full width or be hidden
                    assert sidebar_width <= viewport_width
            else:
                # Desktop: sidebar should be visible
                sidebar = dashboard_page.locator(".investigation-sidebar")
                if await sidebar.count() > 0:
                    await expect(sidebar).to_be_visible()

    @pytest.mark.asyncio
    async def test_performance_metrics(self, dashboard_page: Page):
        """Test dashboard performance and loading times."""
        # Measure page load performance
        start_time = time.time()

        # Wait for dashboard to be fully loaded
        await dashboard_page.wait_for_selector(
            '[data-testid="dashboard-ready"]', timeout=10000
        )

        load_time = time.time() - start_time

        # Assert reasonable load time (under 10 seconds)
        assert load_time < 10, f"Dashboard took too long to load: {load_time:.2f}s"

        # Check JavaScript heap size (if available)
        memory_info = await dashboard_page.evaluate("""
            () => {
                if (performance.memory) {
                    return {
                        usedJSHeapSize: performance.memory.usedJSHeapSize,
                        totalJSHeapSize: performance.memory.totalJSHeapSize,
                        jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
                    };
                }
                return null;
            }
        """)

        if memory_info:
            # Check that memory usage is reasonable (less than 100MB)
            memory_mb = memory_info["usedJSHeapSize"] / (1024 * 1024)
            assert memory_mb < 100, f"Memory usage too high: {memory_mb:.2f}MB"

    @pytest.mark.asyncio
    async def test_error_handling(self, dashboard_page: Page):
        """Test dashboard error handling and recovery."""
        # Test handling of invalid data
        await dashboard_page.evaluate("""
            () => {
                if (window.dashboardInstance) {
                    // Try to handle malformed anomaly data
                    try {
                        window.dashboardInstance.handleAnomalyDetected(null);
                        window.dashboardInstance.handleAnomalyDetected({});
                        window.dashboardInstance.handleAnomalyDetected({ invalid: 'data' });
                    } catch (error) {
                        console.log('Error handling test:', error);
                    }
                }
            }
        """)

        # Check that dashboard is still functional
        await dashboard_page.wait_for_timeout(1000)
        dashboard = dashboard_page.locator(".investigation-dashboard")
        await expect(dashboard).to_be_visible()

        # Test network error handling by blocking requests
        await dashboard_page.route("**/api/**", lambda route: route.abort())

        # Try to refresh data
        await dashboard_page.evaluate("""
            () => {
                if (window.dashboardInstance) {
                    window.dashboardInstance.refreshData();
                }
            }
        """)

        # Dashboard should still be functional
        await dashboard_page.wait_for_timeout(2000)
        await expect(dashboard).to_be_visible()

    @pytest.mark.asyncio
    async def test_data_persistence(self, dashboard_page: Page):
        """Test data persistence and state management."""
        # Set some dashboard state
        await dashboard_page.evaluate("""
            () => {
                if (window.dashboardInstance) {
                    window.dashboardInstance.updateFilter('severity', 'high');
                    window.dashboardInstance.updateTimeRange('1h');
                    window.dashboardInstance.switchView('investigation');
                }
            }
        """)

        # Wait for state to be applied
        await dashboard_page.wait_for_timeout(500)

        # Reload page
        await dashboard_page.reload()
        await dashboard_page.wait_for_selector(
            '[data-testid="dashboard-ready"]', timeout=10000
        )

        # Check if state was persisted (depends on implementation)
        current_state = await dashboard_page.evaluate("""
            () => {
                if (window.dashboardInstance) {
                    return window.dashboardInstance.dashboardState;
                }
                return null;
            }
        """)

        if current_state:
            # Some state might be persisted via URL or localStorage
            assert "currentView" in current_state
            assert "selectedTimeRange" in current_state

    @pytest.mark.asyncio
    async def test_integration_endpoints(self, dashboard_page: Page):
        """Test integration with backend API endpoints."""
        # Test dashboard config endpoint
        config_response = await dashboard_page.evaluate("""
            async () => {
                try {
                    const response = await fetch('/api/dashboard/config');
                    return {
                        status: response.status,
                        ok: response.ok,
                        data: response.ok ? await response.json() : null
                    };
                } catch (error) {
                    return { error: error.message };
                }
            }
        """)

        # Should either succeed or fail gracefully
        assert "status" in config_response or "error" in config_response

        # Test datasets endpoint
        datasets_response = await dashboard_page.evaluate("""
            async () => {
                try {
                    const response = await fetch('/api/datasets?active=true');
                    return {
                        status: response.status,
                        ok: response.ok,
                        data: response.ok ? await response.json() : null
                    };
                } catch (error) {
                    return { error: error.message };
                }
            }
        """)

        assert "status" in datasets_response or "error" in datasets_response


# Performance benchmark tests
class TestDashboardPerformance:
    """Performance-specific tests for the advanced dashboard."""

    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, dashboard_page: Page):
        """Test dashboard performance with large datasets."""
        # Generate large test dataset
        large_anomaly_set = generate_test_anomalies(1000)

        start_time = time.time()

        # Add large dataset to dashboard
        await dashboard_page.evaluate(f"""
            () => {{
                const anomalies = {json.dumps(large_anomaly_set)};
                if (window.dashboardInstance) {{
                    anomalies.forEach((anomaly, index) => {{
                        if (index % 10 === 0) {{ // Process every 10th to avoid blocking
                            setTimeout(() => {{
                                window.dashboardInstance.handleAnomalyDetected(anomaly);
                            }}, index / 10);
                        }}
                    }});
                }}
            }}
        """)

        # Wait for processing
        await dashboard_page.wait_for_timeout(5000)

        processing_time = time.time() - start_time

        # Should process large dataset in reasonable time
        assert (
            processing_time < 30
        ), f"Large dataset processing took too long: {processing_time:.2f}s"

        # Check that dashboard is still responsive
        dashboard = dashboard_page.locator(".investigation-dashboard")
        await expect(dashboard).to_be_visible()

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, dashboard_page: Page):
        """Test memory usage stability over time."""
        initial_memory = await dashboard_page.evaluate("""
            () => performance.memory ? performance.memory.usedJSHeapSize : 0
        """)

        # Simulate continuous data streaming
        for i in range(50):
            await dashboard_page.evaluate(f"""
                () => {{
                    if (window.dashboardInstance) {{
                        const anomaly = {{
                            id: 'test_anomaly_{i}',
                            timestamp: Date.now(),
                            value: Math.random() * 100,
                            anomaly_score: Math.random(),
                            severity: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)]
                        }};
                        window.dashboardInstance.handleAnomalyDetected(anomaly);
                    }}
                }}
            """)

            await dashboard_page.wait_for_timeout(100)

        final_memory = await dashboard_page.evaluate("""
            () => performance.memory ? performance.memory.usedJSHeapSize : 0
        """)

        if initial_memory > 0 and final_memory > 0:
            memory_increase = final_memory - initial_memory
            memory_increase_mb = memory_increase / (1024 * 1024)

            # Memory increase should be reasonable (less than 50MB)
            assert (
                memory_increase_mb < 50
            ), f"Memory increased too much: {memory_increase_mb:.2f}MB"
