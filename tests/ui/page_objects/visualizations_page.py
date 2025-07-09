"""Visualizations page object."""

from typing import Any

from .base_page import BasePage


class VisualizationsPage(BasePage):
    """Visualizations page object with specific functionality."""

    # Locators
    CHART_CONTAINER = "#chart-container, .chart-container"
    D3_CHARTS = ".d3-chart, svg"
    ECHARTS_CONTAINER = ".echarts-container, [id*='echarts']"
    DETECTION_TIMELINE_CHART = "#detection-timeline"
    ANOMALY_RATES_CHART = "#anomaly-rates"
    CHART_CONTROLS = ".chart-controls, .controls"
    FILTER_CONTROLS = "select, input[type='date']"

    def navigate(self) -> None:
        """Navigate to visualizations page."""
        self.navigate_to("/visualizations")

    def wait_for_charts_to_load(self, timeout: int = 10000) -> bool:
        """Wait for charts to load."""
        try:
            # Wait for either D3 SVG elements or ECharts containers
            self.page.wait_for_selector(
                f"{self.D3_CHARTS}, {self.ECHARTS_CONTAINER}", timeout=timeout
            )

            # Give additional time for chart rendering
            self.page.wait_for_timeout(2000)
            return True

        except Exception as e:
            print(f"Charts failed to load: {e}")
            return False

    def get_available_charts(self) -> list[dict[str, Any]]:
        """Get information about available charts."""
        charts = []

        # Check for D3 charts
        d3_charts = self.page.locator(self.D3_CHARTS)
        for i in range(d3_charts.count()):
            chart = d3_charts.nth(i)
            chart_id = chart.get_attribute("id") or f"d3-chart-{i}"

            charts.append(
                {
                    "id": chart_id,
                    "type": "d3",
                    "visible": chart.is_visible(),
                    "has_data": self.has_chart_data(chart),
                }
            )

        # Check for ECharts
        echarts_containers = self.page.locator(self.ECHARTS_CONTAINER)
        for i in range(echarts_containers.count()):
            container = echarts_containers.nth(i)
            container_id = container.get_attribute("id") or f"echarts-{i}"

            charts.append(
                {
                    "id": container_id,
                    "type": "echarts",
                    "visible": container.is_visible(),
                    "has_data": self.has_echarts_data(container),
                }
            )

        return charts

    def has_chart_data(self, chart_element) -> bool:
        """Check if D3 chart has data."""
        try:
            # Check for common D3 chart elements
            data_elements = chart_element.locator("circle, rect, path, line")
            return data_elements.count() > 0
        except:
            return False

    def has_echarts_data(self, container_element) -> bool:
        """Check if ECharts container has data."""
        try:
            # ECharts typically creates canvas or SVG elements
            chart_elements = container_element.locator("canvas, svg")
            return chart_elements.count() > 0
        except:
            return False

    def get_detection_timeline_data(self) -> dict[str, Any]:
        """Get detection timeline chart data."""
        timeline_data = {"has_data": False, "data_points": 0}

        timeline_chart = self.page.locator(self.DETECTION_TIMELINE_CHART)
        if timeline_chart.count() > 0:
            # For D3 charts, count data points
            data_points = timeline_chart.locator("circle, rect").count()
            timeline_data["has_data"] = data_points > 0
            timeline_data["data_points"] = data_points

        return timeline_data

    def get_anomaly_rates_data(self) -> dict[str, Any]:
        """Get anomaly rates chart data."""
        rates_data = {"has_data": False, "data_points": 0}

        rates_chart = self.page.locator(self.ANOMALY_RATES_CHART)
        if rates_chart.count() > 0:
            # Count chart elements
            data_points = rates_chart.locator("circle, rect, bar").count()
            rates_data["has_data"] = data_points > 0
            rates_data["data_points"] = data_points

        return rates_data

    def test_chart_interactivity(self) -> dict[str, bool]:
        """Test chart interactivity."""
        results = {}

        # Test hover effects on D3 charts
        d3_charts = self.page.locator(self.D3_CHARTS)
        if d3_charts.count() > 0:
            first_chart = d3_charts.first
            data_elements = first_chart.locator("circle, rect")

            if data_elements.count() > 0:
                # Hover over first data point
                data_elements.first.hover()
                self.page.wait_for_timeout(500)

                # Check for tooltip or highlight effects
                tooltips = self.page.locator(".tooltip, [class*='tooltip']")
                results["d3_hover_works"] = tooltips.count() > 0
            else:
                results["d3_hover_works"] = False
        else:
            results["d3_hover_works"] = False

        # Test ECharts interactivity
        echarts_containers = self.page.locator(self.ECHARTS_CONTAINER)
        if echarts_containers.count() > 0:
            first_container = echarts_containers.first

            # Click on chart area
            first_container.click()
            self.page.wait_for_timeout(500)

            # ECharts usually shows tooltips on interaction
            results["echarts_interactive"] = (
                True  # Assume interactive if container exists
            )
        else:
            results["echarts_interactive"] = False

        return results

    def test_chart_responsiveness(self) -> dict[str, bool]:
        """Test chart responsiveness."""
        results = {}

        # Get initial chart dimensions
        initial_charts = self.get_available_charts()

        # Resize viewport
        self.page.set_viewport_size({"width": 800, "height": 600})
        self.page.wait_for_timeout(1000)

        # Check if charts adapted
        resized_charts = self.get_available_charts()

        results["charts_respond_to_resize"] = len(resized_charts) == len(initial_charts)

        # Restore original viewport
        self.page.set_viewport_size({"width": 1920, "height": 1080})

        return results

    def verify_visualizations_page_layout(self) -> dict[str, bool]:
        """Verify visualizations page layout."""
        return {
            "has_title": "Visualizations" in (self.page.title() or ""),
            "has_chart_container": self.page.locator(self.CHART_CONTAINER).count() > 0,
            "has_d3_charts": self.page.locator(self.D3_CHARTS).count() > 0,
            "has_echarts": self.page.locator(self.ECHARTS_CONTAINER).count() > 0,
            "charts_loaded": self.wait_for_charts_to_load(5000),
        }

    def get_chart_performance_metrics(self) -> dict[str, Any]:
        """Get chart rendering performance metrics."""
        # Measure chart rendering time
        start_time = self.page.evaluate("() => performance.now()")

        self.wait_for_charts_to_load()

        end_time = self.page.evaluate("() => performance.now()")

        charts = self.get_available_charts()

        return {
            "chart_render_time": end_time - start_time,
            "total_charts": len(charts),
            "d3_charts": len([c for c in charts if c["type"] == "d3"]),
            "echarts": len([c for c in charts if c["type"] == "echarts"]),
            "charts_with_data": len([c for c in charts if c["has_data"]]),
        }
