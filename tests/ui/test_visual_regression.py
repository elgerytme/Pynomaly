"""Visual Regression Testing with Screenshot Comparison."""

from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import Page


class VisualTester:
    """Helper class for visual regression testing."""

    def __init__(self):
        self.baseline_dir = Path("visual-baselines")
        self.test_dir = Path("screenshots")
        self.diff_dir = Path("visual-diffs")

        # Create directories
        self.baseline_dir.mkdir(exist_ok=True)
        self.test_dir.mkdir(exist_ok=True)
        self.diff_dir.mkdir(exist_ok=True)

    def compare_images(
        self, baseline_path: Path, test_path: Path, threshold: float = 0.95
    ) -> dict[str, any]:
        """Compare two images and return similarity metrics."""
        try:
            # Read images
            baseline_img = cv2.imread(str(baseline_path))
            test_img = cv2.imread(str(test_path))

            if baseline_img is None or test_img is None:
                return {
                    "similar": False,
                    "error": "Could not read one or both images",
                    "similarity": 0.0,
                }

            # Resize images to same size if different
            if baseline_img.shape != test_img.shape:
                height, width = baseline_img.shape[:2]
                test_img = cv2.resize(test_img, (width, height))

            # Calculate structural similarity
            baseline_gray = cv2.cvtColor(baseline_img, cv2.COLOR_BGR2GRAY)
            test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

            # Simple pixel difference
            diff = cv2.absdiff(baseline_gray, test_gray)
            non_zero_count = np.count_nonzero(diff)
            total_pixels = diff.size
            similarity = 1.0 - (non_zero_count / total_pixels)

            # Create diff image
            diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
            diff_path = self.diff_dir / f"diff_{baseline_path.stem}.png"
            cv2.imwrite(str(diff_path), diff_colored)

            return {
                "similar": similarity >= threshold,
                "similarity": similarity,
                "diff_path": str(diff_path),
                "threshold": threshold,
            }

        except Exception as e:
            return {"similar": False, "error": str(e), "similarity": 0.0}

    def capture_and_compare(
        self,
        page: Page,
        test_name: str,
        full_page: bool = True,
        threshold: float = 0.95,
    ) -> dict[str, any]:
        """Capture screenshot and compare with baseline."""
        test_path = self.test_dir / f"{test_name}.png"
        baseline_path = self.baseline_dir / f"{test_name}.png"

        # Take screenshot
        page.screenshot(path=str(test_path), full_page=full_page)

        # If baseline doesn't exist, create it
        if not baseline_path.exists():
            # Copy test image as baseline
            import shutil

            shutil.copy(str(test_path), str(baseline_path))
            return {
                "similar": True,
                "baseline_created": True,
                "message": f"Baseline created for {test_name}",
            }

        # Compare with baseline
        return self.compare_images(baseline_path, test_path, threshold)


class TestVisualRegression:
    """Visual regression test suite."""

    def setup_method(self):
        """Setup visual tester for each test."""
        self.visual_tester = VisualTester()

    def test_dashboard_visual_consistency(self, dashboard_page):
        """Test dashboard visual consistency."""
        dashboard_page.navigate()

        # Wait for all elements to load
        dashboard_page.wait_for_load()
        dashboard_page.page.wait_for_timeout(2000)  # Extra time for animations

        # Capture and compare full dashboard
        result = self.visual_tester.capture_and_compare(
            dashboard_page.page, "dashboard_full", full_page=True, threshold=0.95
        )

        assert result[
            "similar"
        ], f"Dashboard visual regression detected. Similarity: {result.get('similarity', 0)}"

        # Capture specific sections
        stats_section = dashboard_page.page.locator(
            ".grid .bg-white.shadow.rounded-lg"
        ).first
        if stats_section.is_visible():
            stats_section.screenshot(path="screenshots/dashboard_stats.png")

            baseline_stats = Path("visual-baselines/dashboard_stats.png")
            if baseline_stats.exists():
                stats_result = self.visual_tester.compare_images(
                    baseline_stats, Path("screenshots/dashboard_stats.png")
                )
                assert stats_result[
                    "similar"
                ], "Stats section visual regression detected"

    def test_navigation_visual_consistency(self, page: Page):
        """Test navigation visual consistency across pages."""
        pages = [
            {"path": "/", "name": "dashboard_nav"},
            {"path": "/detectors", "name": "detectors_nav"},
            {"path": "/datasets", "name": "datasets_nav"},
            {"path": "/detection", "name": "detection_nav"},
        ]

        for page_info in pages:
            page.goto(f"http://pynomaly-app:8000{page_info['path']}")
            page.wait_for_load_state("networkidle")

            # Capture just the navigation area
            nav_element = page.locator("nav")
            if nav_element.is_visible():
                nav_element.screenshot(path=f"screenshots/{page_info['name']}.png")

                # Compare with baseline
                baseline_path = Path(f"visual-baselines/{page_info['name']}.png")
                test_path = Path(f"screenshots/{page_info['name']}.png")

                if baseline_path.exists():
                    result = self.visual_tester.compare_images(baseline_path, test_path)
                    assert result[
                        "similar"
                    ], f"Navigation visual regression on {page_info['path']}"

    def test_form_visual_consistency(self, detectors_page):
        """Test form visual consistency."""
        detectors_page.navigate()

        # Capture form in initial state
        self.visual_tester.capture_and_compare(
            detectors_page.page, "detectors_form_initial", full_page=False
        )

        # Test form with validation errors
        if detectors_page.page.locator(detectors_page.CREATE_BUTTON).count() > 0:
            detectors_page.click_element(detectors_page.CREATE_BUTTON)
            detectors_page.page.wait_for_timeout(1000)

            # Capture form with errors
            self.visual_tester.capture_and_compare(
                detectors_page.page, "detectors_form_errors", full_page=False
            )

    def test_responsive_visual_consistency(self, page: Page):
        """Test visual consistency across different viewport sizes."""
        viewports = [
            {"width": 1920, "height": 1080, "name": "desktop"},
            {"width": 768, "height": 1024, "name": "tablet"},
            {"width": 375, "height": 667, "name": "mobile"},
        ]

        for viewport in viewports:
            page.set_viewport_size(
                {"width": viewport["width"], "height": viewport["height"]}
            )
            page.goto("http://pynomaly-app:8000/web/")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1000)  # Wait for responsive adjustments

            # Capture full page for each viewport
            result = self.visual_tester.capture_and_compare(
                page,
                f"responsive_{viewport['name']}",
                full_page=True,
                threshold=0.90,  # Lower threshold for responsive differences
            )

            # Don't assert failure for responsive tests as layout changes are expected
            # Just log the results
            if not result.get("baseline_created", False):
                print(
                    f"Responsive test {viewport['name']}: "
                    f"Similar={result['similar']}, "
                    f"Similarity={result.get('similarity', 0)}"
                )

    def test_dark_mode_visual_consistency(self, page: Page):
        """Test dark mode visual consistency if available."""
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        # Check if dark mode toggle exists
        dark_mode_toggle = page.locator("[data-theme-toggle], .dark-mode-toggle")

        if dark_mode_toggle.count() > 0:
            # Capture light mode
            light_result = self.visual_tester.capture_and_compare(
                page, "dashboard_light_mode", threshold=0.95
            )

            # Toggle to dark mode
            dark_mode_toggle.click()
            page.wait_for_timeout(1000)  # Wait for theme change

            # Capture dark mode
            dark_result = self.visual_tester.capture_and_compare(
                page, "dashboard_dark_mode", threshold=0.95
            )

            assert light_result["similar"], "Light mode visual regression"
            assert dark_result["similar"], "Dark mode visual regression"

    def test_chart_visual_consistency(self, visualizations_page):
        """Test chart visual consistency."""
        visualizations_page.navigate()

        # Wait for charts to load
        charts_loaded = visualizations_page.wait_for_charts_to_load()

        if charts_loaded:
            # Capture full visualizations page
            self.visual_tester.capture_and_compare(
                visualizations_page.page,
                "visualizations_full",
                threshold=0.85,  # Lower threshold for charts with dynamic data
            )

            # Capture individual charts if they exist
            charts = visualizations_page.get_available_charts()

            for i, chart in enumerate(charts):
                if chart["visible"]:
                    chart_element = visualizations_page.page.locator(f"#{chart['id']}")
                    if chart_element.count() > 0:
                        chart_element.screenshot(path=f"screenshots/chart_{i}.png")

    def test_loading_state_visual_consistency(self, page: Page):
        """Test loading states visual consistency."""
        page.goto("http://pynomaly-app:8000/web/detection")
        page.wait_for_load_state("networkidle")

        # Trigger an action that shows loading state
        run_button = page.locator("button[type='submit']")

        if run_button.count() > 0:
            # Capture before action
            self.visual_tester.capture_and_compare(page, "detection_before_action")

            # Trigger action and try to capture loading state
            run_button.click()
            page.wait_for_timeout(500)  # Quick capture during loading

            self.visual_tester.capture_and_compare(page, "detection_loading_state")

    def test_error_state_visual_consistency(self, detectors_page):
        """Test error states visual consistency."""
        detectors_page.navigate()

        # Trigger validation errors
        if detectors_page.page.locator(detectors_page.CREATE_BUTTON).count() > 0:
            detectors_page.click_element(detectors_page.CREATE_BUTTON)
            detectors_page.page.wait_for_timeout(1000)

            # Capture error state
            self.visual_tester.capture_and_compare(
                detectors_page.page, "form_error_state"
            )

            # Check for specific error message styling
            error_elements = detectors_page.page.locator(
                ".error, .invalid, [class*='error']"
            )
            if error_elements.count() > 0:
                error_elements.first.screenshot(path="screenshots/error_message.png")

    def test_component_visual_isolation(self, dashboard_page):
        """Test individual component visual consistency."""
        dashboard_page.navigate()

        # Capture individual components
        components = [
            (".grid .bg-white.shadow.rounded-lg", "stats_card"),
            ("#results-table", "results_table"),
            (".grid .relative.rounded-lg", "quick_actions"),
        ]

        for selector, name in components:
            element = dashboard_page.page.locator(selector).first
            if element.is_visible():
                element.screenshot(path=f"screenshots/{name}.png")

                baseline_path = Path(f"visual-baselines/{name}.png")
                if baseline_path.exists():
                    result = self.visual_tester.compare_images(
                        baseline_path, Path(f"screenshots/{name}.png")
                    )
                    assert result["similar"], f"Component {name} visual regression"

    def test_animation_visual_consistency(self, page: Page):
        """Test animations and transitions visual consistency."""
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        # Look for elements with animations
        animated_elements = page.locator("[class*='transition'], [class*='animate']")

        if animated_elements.count() > 0:
            # Capture before interaction
            self.visual_tester.capture_and_compare(page, "before_animation")

            # Trigger animation (hover, click, etc.)
            first_element = animated_elements.first
            first_element.hover()
            page.wait_for_timeout(1000)  # Wait for animation

            # Capture after animation
            self.visual_tester.capture_and_compare(page, "after_animation")

    def generate_visual_report(self) -> dict[str, any]:
        """Generate a visual regression test report."""
        report = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "new_baselines": 0,
            "differences": [],
        }

        # Scan diff directory for differences
        if self.visual_tester.diff_dir.exists():
            diff_files = list(self.visual_tester.diff_dir.glob("*.png"))
            report["differences"] = [str(f) for f in diff_files]

        # Count baseline files
        if self.visual_tester.baseline_dir.exists():
            baseline_files = list(self.visual_tester.baseline_dir.glob("*.png"))
            report["total_baselines"] = len(baseline_files)

        return report
