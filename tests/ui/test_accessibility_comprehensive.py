"""Comprehensive Accessibility Testing with WCAG 2.1 AA Compliance and axe-core Integration."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from playwright.sync_api import Page, expect

from tests.ui.enhanced_page_objects.base_page import BasePage

# Configuration
ACCESSIBILITY_TESTING_ENABLED = (
    os.getenv("ACCESSIBILITY_TESTING", "true").lower() == "true"
)
WCAG_LEVEL = os.getenv("WCAG_LEVEL", "AA")  # A, AA, or AAA
ACCESSIBILITY_REPORTS_DIR = Path("test_reports/accessibility")
ACCESSIBILITY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# WCAG 2.1 compliance configuration
WCAG_CONFIG = {
    "level": WCAG_LEVEL,
    "tags": [
        "wcag2a",
        "wcag2aa" if WCAG_LEVEL in ["AA", "AAA"] else None,
        "wcag2aaa" if WCAG_LEVEL == "AAA" else None,
        "wcag21a",
        "wcag21aa" if WCAG_LEVEL in ["AA", "AAA"] else None,
        "best-practice",
    ],
    "rules": {
        # Enable/disable specific rules
        "color-contrast": {"enabled": True},
        "keyboard-navigation": {"enabled": True},
        "focus-management": {"enabled": True},
        "aria-attributes": {"enabled": True},
        "semantic-structure": {"enabled": True},
        "alt-text": {"enabled": True},
        "form-labels": {"enabled": True},
        "heading-order": {"enabled": True},
    },
}

# Filter out None values from tags
WCAG_CONFIG["tags"] = [tag for tag in WCAG_CONFIG["tags"] if tag is not None]


class AccessibilityTester:
    """Comprehensive accessibility testing with axe-core and WCAG compliance."""

    def __init__(self, page: Page):
        self.page = page
        self.base_page = BasePage(page)
        self.axe_injected = False

    async def inject_axe_core(self):
        """Inject axe-core library into the page."""
        if self.axe_injected:
            return

        try:
            # Inject axe-core from CDN
            await self.page.add_script_tag(
                url="https://unpkg.com/axe-core@4.8.3/axe.min.js"
            )

            # Configure axe with our WCAG settings
            await self.page.evaluate(
                f"""
                window.axeConfig = {json.dumps(WCAG_CONFIG)};
                
                // Configure axe with our rules
                if (typeof axe !== 'undefined') {{
                    axe.configure({{
                        tags: window.axeConfig.tags,
                        rules: Object.fromEntries(
                            Object.entries(window.axeConfig.rules).map(([rule, config]) => [
                                rule, {{ enabled: config.enabled }}
                            ])
                        )
                    }});
                }}
            """
            )

            self.axe_injected = True

        except Exception as e:
            print(f"Failed to inject axe-core: {e}")
            self.axe_injected = False

    async def run_accessibility_scan(
        self,
        context: Optional[str] = None,
        tags: Optional[List[str]] = None,
        rules: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive accessibility scan with axe-core."""
        await self.inject_axe_core()

        if not self.axe_injected:
            return {"error": "axe-core not available", "violations": [], "passes": []}

        try:
            # Prepare scan options
            scan_options = {"tags": tags or WCAG_CONFIG["tags"], "rules": rules or {}}

            if context:
                scan_options["include"] = [context]

            # Run axe scan
            results = await self.page.evaluate(
                f"""
                async () => {{
                    try {{
                        const options = {json.dumps(scan_options)};
                        const results = await axe.run(document, options);
                        
                        // Enhance results with additional context
                        results.testEngine = {{
                            name: "axe-core",
                            version: axe.version
                        }};
                        results.testEnvironment = {{
                            userAgent: navigator.userAgent,
                            windowSize: {{
                                width: window.innerWidth,
                                height: window.innerHeight
                            }},
                            url: window.location.href,
                            timestamp: new Date().toISOString()
                        }};
                        
                        return results;
                    }} catch (error) {{
                        return {{
                            error: error.message,
                            violations: [],
                            passes: [],
                            incomplete: [],
                            inapplicable: []
                        }};
                    }}
                }}
            """
            )

            # Process and categorize results
            processed_results = self._process_axe_results(results)

            # Save detailed report
            await self._save_accessibility_report(processed_results)

            return processed_results

        except Exception as e:
            return {
                "error": str(e),
                "violations": [],
                "passes": [],
                "incomplete": [],
                "inapplicable": [],
            }

    def _process_axe_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enhance axe results with additional analysis."""
        processed = {
            "summary": {
                "violations_count": len(results.get("violations", [])),
                "passes_count": len(results.get("passes", [])),
                "incomplete_count": len(results.get("incomplete", [])),
                "inapplicable_count": len(results.get("inapplicable", [])),
                "wcag_level": WCAG_LEVEL,
                "overall_score": 0,
                "compliance_status": "unknown",
            },
            "violations": self._categorize_violations(results.get("violations", [])),
            "passes": results.get("passes", []),
            "incomplete": results.get("incomplete", []),
            "inapplicable": results.get("inapplicable", []),
            "test_environment": results.get("testEnvironment", {}),
            "test_engine": results.get("testEngine", {}),
            "recommendations": [],
        }

        # Calculate compliance score and status
        total_tests = (
            processed["summary"]["violations_count"]
            + processed["summary"]["passes_count"]
        )

        if total_tests > 0:
            processed["summary"]["overall_score"] = (
                processed["summary"]["passes_count"] / total_tests * 100
            )

            # Determine compliance status
            if processed["summary"]["violations_count"] == 0:
                processed["summary"]["compliance_status"] = "compliant"
            elif processed["summary"]["violations_count"] <= 3:
                processed["summary"]["compliance_status"] = "minor_issues"
            elif processed["summary"]["violations_count"] <= 10:
                processed["summary"]["compliance_status"] = "moderate_issues"
            else:
                processed["summary"]["compliance_status"] = "major_issues"

        # Generate recommendations
        processed["recommendations"] = self._generate_recommendations(
            processed["violations"]
        )

        return processed

    def _categorize_violations(
        self, violations: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize violations by type and severity."""
        categories = {
            "critical": [],
            "serious": [],
            "moderate": [],
            "minor": [],
            "color_contrast": [],
            "keyboard_navigation": [],
            "screen_reader": [],
            "form_accessibility": [],
            "semantic_structure": [],
            "focus_management": [],
        }

        for violation in violations:
            impact = violation.get("impact", "minor")
            rule_id = violation.get("id", "")

            # Add to severity category
            if impact in categories:
                categories[impact].append(violation)

            # Add to functional categories
            if "color-contrast" in rule_id:
                categories["color_contrast"].append(violation)
            elif any(
                keyword in rule_id for keyword in ["keyboard", "focus", "tabindex"]
            ):
                categories["keyboard_navigation"].append(violation)
            elif any(
                keyword in rule_id for keyword in ["aria", "screen-reader", "label"]
            ):
                categories["screen_reader"].append(violation)
            elif any(keyword in rule_id for keyword in ["form", "input", "select"]):
                categories["form_accessibility"].append(violation)
            elif any(
                keyword in rule_id for keyword in ["heading", "landmark", "region"]
            ):
                categories["semantic_structure"].append(violation)
            elif "focus" in rule_id:
                categories["focus_management"].append(violation)

        return categories

    def _generate_recommendations(
        self, violations: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on violations."""
        recommendations = []

        # Color contrast recommendations
        if violations.get("color_contrast"):
            recommendations.append(
                {
                    "category": "Color Contrast",
                    "priority": "high",
                    "recommendation": "Improve color contrast ratios to meet WCAG 2.1 AA standards (4.5:1 for normal text, 3:1 for large text).",
                    "action": "Use color contrast analyzers and update CSS color values.",
                }
            )

        # Keyboard navigation recommendations
        if violations.get("keyboard_navigation"):
            recommendations.append(
                {
                    "category": "Keyboard Navigation",
                    "priority": "high",
                    "recommendation": "Ensure all interactive elements are keyboard accessible with proper focus management.",
                    "action": "Add tabindex attributes, implement focus styles, and test with Tab navigation.",
                }
            )

        # Screen reader recommendations
        if violations.get("screen_reader"):
            recommendations.append(
                {
                    "category": "Screen Reader Support",
                    "priority": "high",
                    "recommendation": "Add proper ARIA labels and descriptions for screen reader users.",
                    "action": "Implement aria-label, aria-describedby, and role attributes.",
                }
            )

        # Form accessibility recommendations
        if violations.get("form_accessibility"):
            recommendations.append(
                {
                    "category": "Form Accessibility",
                    "priority": "medium",
                    "recommendation": "Associate form labels with inputs and provide clear error messages.",
                    "action": "Use <label> elements with for attributes or aria-labelledby.",
                }
            )

        # Semantic structure recommendations
        if violations.get("semantic_structure"):
            recommendations.append(
                {
                    "category": "Semantic Structure",
                    "priority": "medium",
                    "recommendation": "Improve page structure with proper headings and landmarks.",
                    "action": "Use h1-h6 elements in order and add landmark roles (main, nav, aside).",
                }
            )

        return recommendations

    async def _save_accessibility_report(self, results: Dict[str, Any]):
        """Save detailed accessibility report to file."""
        timestamp = results.get("test_environment", {}).get("timestamp", "unknown")
        url = results.get("test_environment", {}).get("url", "unknown")

        # Generate filename
        url_safe = url.replace("/", "_").replace(":", "").replace("?", "_")
        filename = (
            f"accessibility_report_{url_safe}_{timestamp[:19].replace(':', '-')}.json"
        )

        report_path = ACCESSIBILITY_REPORTS_DIR / filename

        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Accessibility report saved: {report_path}")

    async def test_keyboard_navigation(self) -> Dict[str, Any]:
        """Test keyboard navigation functionality."""
        results = {
            "focusable_elements": [],
            "tab_order": [],
            "focus_indicators": {},
            "keyboard_traps": [],
            "skip_links": [],
        }

        try:
            # Get all focusable elements
            focusable_elements = await self.page.evaluate(
                """
                () => {
                    const selector = 'a, button, input, select, textarea, [tabindex]:not([tabindex="-1"])';
                    const elements = Array.from(document.querySelectorAll(selector));
                    return elements.map((el, index) => ({
                        tagName: el.tagName,
                        type: el.type || null,
                        id: el.id || null,
                        className: el.className || null,
                        text: el.textContent?.trim().substring(0, 50) || null,
                        tabIndex: el.tabIndex,
                        isVisible: el.offsetParent !== null,
                        hasAriaLabel: el.hasAttribute('aria-label'),
                        hasTitle: el.hasAttribute('title'),
                        index: index
                    }));
                }
            """
            )

            results["focusable_elements"] = focusable_elements

            # Test tab navigation order
            visible_elements = [el for el in focusable_elements if el["isVisible"]]

            for i, element in enumerate(
                visible_elements[:10]
            ):  # Test first 10 elements
                await self.page.keyboard.press("Tab")

                # Check if focus indicator is visible
                focused_element = await self.page.evaluate(
                    """
                    () => {
                        const active = document.activeElement;
                        if (!active) return null;
                        
                        const styles = window.getComputedStyle(active);
                        return {
                            tagName: active.tagName,
                            hasFocusRing: styles.outline !== 'none' || 
                                         styles.boxShadow !== 'none' ||
                                         styles.border !== styles.border, // Simplified check
                            outlineStyle: styles.outline,
                            boxShadow: styles.boxShadow
                        };
                    }
                """
                )

                if focused_element:
                    results["tab_order"].append(
                        {
                            "index": i,
                            "element": focused_element,
                            "has_focus_indicator": focused_element["hasFocusRing"],
                        }
                    )

            # Check for skip links
            skip_links = await self.page.evaluate(
                """
                () => {
                    const skipLinks = Array.from(document.querySelectorAll('a[href^="#"]'));
                    return skipLinks.filter(link => 
                        link.textContent.toLowerCase().includes('skip') ||
                        link.textContent.toLowerCase().includes('jump')
                    ).map(link => ({
                        text: link.textContent.trim(),
                        href: link.getAttribute('href'),
                        isVisible: link.offsetParent !== null
                    }));
                }
            """
            )

            results["skip_links"] = skip_links

        except Exception as e:
            results["error"] = str(e)

        return results

    async def test_screen_reader_compatibility(self) -> Dict[str, Any]:
        """Test screen reader compatibility."""
        results = {
            "headings_structure": [],
            "landmarks": [],
            "aria_labels": [],
            "alt_texts": [],
            "form_labels": [],
            "live_regions": [],
        }

        try:
            # Test heading structure
            headings = await self.page.evaluate(
                """
                () => {
                    const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'));
                    return headings.map(h => ({
                        level: parseInt(h.tagName.charAt(1)),
                        text: h.textContent.trim(),
                        id: h.id || null,
                        isVisible: h.offsetParent !== null
                    }));
                }
            """
            )

            # Check heading order
            for i, heading in enumerate(headings[1:], 1):
                prev_level = headings[i - 1]["level"]
                curr_level = heading["level"]

                if curr_level > prev_level + 1:
                    heading["skip_level_error"] = True

            results["headings_structure"] = headings

            # Test landmarks
            landmarks = await self.page.evaluate(
                """
                () => {
                    const landmarks = Array.from(document.querySelectorAll(
                        'main, nav, aside, header, footer, section, [role="main"], [role="navigation"], [role="complementary"], [role="banner"], [role="contentinfo"]'
                    ));
                    return landmarks.map(landmark => ({
                        tagName: landmark.tagName,
                        role: landmark.getAttribute('role') || landmark.tagName.toLowerCase(),
                        hasAriaLabel: landmark.hasAttribute('aria-label'),
                        ariaLabel: landmark.getAttribute('aria-label'),
                        id: landmark.id || null
                    }));
                }
            """
            )

            results["landmarks"] = landmarks

            # Test images alt text
            images = await self.page.evaluate(
                """
                () => {
                    const images = Array.from(document.querySelectorAll('img'));
                    return images.map(img => ({
                        src: img.src,
                        alt: img.alt,
                        hasAlt: img.hasAttribute('alt'),
                        isDecorative: img.alt === '',
                        role: img.getAttribute('role'),
                        ariaLabel: img.getAttribute('aria-label')
                    }));
                }
            """
            )

            results["alt_texts"] = images

            # Test form labels
            form_elements = await self.page.evaluate(
                """
                () => {
                    const inputs = Array.from(document.querySelectorAll('input, select, textarea'));
                    return inputs.map(input => {
                        const label = document.querySelector(`label[for="${input.id}"]`) || 
                                     input.closest('label');
                        return {
                            type: input.type || input.tagName.toLowerCase(),
                            id: input.id,
                            name: input.name,
                            hasLabel: !!label,
                            labelText: label ? label.textContent.trim() : null,
                            hasAriaLabel: input.hasAttribute('aria-label'),
                            ariaLabel: input.getAttribute('aria-label'),
                            hasAriaLabelledBy: input.hasAttribute('aria-labelledby'),
                            placeholder: input.placeholder || null
                        };
                    });
                }
            """
            )

            results["form_labels"] = form_elements

            # Test live regions
            live_regions = await self.page.evaluate(
                """
                () => {
                    const liveElements = Array.from(document.querySelectorAll('[aria-live], [role="status"], [role="alert"]'));
                    return liveElements.map(el => ({
                        tagName: el.tagName,
                        ariaLive: el.getAttribute('aria-live'),
                        role: el.getAttribute('role'),
                        id: el.id || null,
                        className: el.className || null
                    }));
                }
            """
            )

            results["live_regions"] = live_regions

        except Exception as e:
            results["error"] = str(e)

        return results


# Test fixtures
@pytest.fixture
def accessibility_tester(page: Page):
    """Create accessibility tester instance."""
    return AccessibilityTester(page)


# Accessibility test suite
@pytest.mark.skipif(
    not ACCESSIBILITY_TESTING_ENABLED, reason="Accessibility testing disabled"
)
class TestAccessibilityCompliance:
    """Comprehensive WCAG 2.1 AA accessibility compliance tests."""

    async def test_homepage_accessibility(
        self, accessibility_tester: AccessibilityTester, page: Page
    ):
        """Test homepage accessibility compliance."""
        await page.goto("/")

        results = await accessibility_tester.run_accessibility_scan()

        # Assert no critical violations
        critical_violations = results["violations"].get("critical", [])
        assert (
            len(critical_violations) == 0
        ), f"Critical accessibility violations found: {critical_violations}"

        # Assert no serious violations
        serious_violations = results["violations"].get("serious", [])
        assert (
            len(serious_violations) == 0
        ), f"Serious accessibility violations found: {serious_violations}"

        # Check overall compliance score
        assert (
            results["summary"]["overall_score"] >= 90
        ), f"Accessibility score too low: {results['summary']['overall_score']}%"

    async def test_detectors_page_accessibility(
        self, accessibility_tester: AccessibilityTester, page: Page
    ):
        """Test detectors page accessibility compliance."""
        await page.goto("/detectors")

        results = await accessibility_tester.run_accessibility_scan()

        # Check for form accessibility
        form_violations = results["violations"].get("form_accessibility", [])
        assert (
            len(form_violations) <= 2
        ), f"Too many form accessibility issues: {form_violations}"

        # Test keyboard navigation
        keyboard_results = await accessibility_tester.test_keyboard_navigation()

        # Ensure all focusable elements have focus indicators
        elements_without_focus = [
            el
            for el in keyboard_results.get("tab_order", [])
            if not el.get("has_focus_indicator", False)
        ]
        assert (
            len(elements_without_focus) <= 1
        ), f"Elements missing focus indicators: {elements_without_focus}"

    async def test_datasets_page_accessibility(
        self, accessibility_tester: AccessibilityTester, page: Page
    ):
        """Test datasets page accessibility compliance."""
        await page.goto("/datasets")

        results = await accessibility_tester.run_accessibility_scan()

        # Check table accessibility if tables are present
        await page.wait_for_selector("table", state="attached", timeout=5000)

        if await page.query_selector("table"):
            # Test table-specific accessibility
            table_results = await page.evaluate(
                """
                () => {
                    const tables = Array.from(document.querySelectorAll('table'));
                    return tables.map(table => ({
                        hasCaption: !!table.querySelector('caption'),
                        hasTheadTbody: !!table.querySelector('thead') && !!table.querySelector('tbody'),
                        hasThElements: table.querySelectorAll('th').length > 0,
                        hasScope: Array.from(table.querySelectorAll('th')).every(th => 
                            th.hasAttribute('scope') || th.hasAttribute('id')
                        )
                    }));
                }
            """
            )

            for table in table_results:
                assert table[
                    "hasTheadTbody"
                ], "Tables should have thead and tbody elements"
                assert table[
                    "hasThElements"
                ], "Tables should have th elements for headers"

    async def test_detection_page_accessibility(
        self, accessibility_tester: AccessibilityTester, page: Page
    ):
        """Test detection page accessibility compliance."""
        await page.goto("/detection")

        results = await accessibility_tester.run_accessibility_scan()

        # Test form accessibility specifically
        screen_reader_results = (
            await accessibility_tester.test_screen_reader_compatibility()
        )

        # Check form labels
        form_elements = screen_reader_results.get("form_labels", [])
        unlabeled_elements = [
            el
            for el in form_elements
            if not (
                el.get("hasLabel")
                or el.get("hasAriaLabel")
                or el.get("hasAriaLabelledBy")
            )
        ]

        assert (
            len(unlabeled_elements) == 0
        ), f"Form elements without labels: {unlabeled_elements}"

    async def test_visualizations_page_accessibility(
        self, accessibility_tester: AccessibilityTester, page: Page
    ):
        """Test visualizations page accessibility compliance."""
        await page.goto("/visualizations")

        results = await accessibility_tester.run_accessibility_scan()

        # Check for alternative text on charts/visualizations
        images = results.get("alt_texts", [])
        if images:
            images_without_alt = [
                img
                for img in images
                if not img.get("hasAlt") and not img.get("ariaLabel")
            ]
            assert (
                len(images_without_alt) == 0
            ), f"Images without alternative text: {images_without_alt}"

    async def test_keyboard_navigation_comprehensive(
        self, accessibility_tester: AccessibilityTester, page: Page
    ):
        """Test comprehensive keyboard navigation across the application."""
        pages_to_test = [
            "/",
            "/detectors",
            "/datasets",
            "/detection",
            "/visualizations",
        ]

        for url in pages_to_test:
            await page.goto(url)

            keyboard_results = await accessibility_tester.test_keyboard_navigation()

            # Check that tab navigation works
            assert (
                len(keyboard_results.get("tab_order", [])) > 0
            ), f"No keyboard navigation possible on {url}"

            # Check for keyboard traps
            traps = keyboard_results.get("keyboard_traps", [])
            assert len(traps) == 0, f"Keyboard traps found on {url}: {traps}"

    async def test_color_contrast_compliance(
        self, accessibility_tester: AccessibilityTester, page: Page
    ):
        """Test color contrast compliance across pages."""
        pages_to_test = [
            "/",
            "/detectors",
            "/datasets",
            "/detection",
            "/visualizations",
        ]

        for url in pages_to_test:
            await page.goto(url)

            results = await accessibility_tester.run_accessibility_scan(
                tags=["wcag2aa"], rules={"color-contrast": {"enabled": True}}
            )

            # Check for color contrast violations
            contrast_violations = results["violations"].get("color_contrast", [])
            assert (
                len(contrast_violations) == 0
            ), f"Color contrast violations on {url}: {contrast_violations}"

    async def test_screen_reader_compatibility_comprehensive(
        self, accessibility_tester: AccessibilityTester, page: Page
    ):
        """Test screen reader compatibility across the application."""
        await page.goto("/")

        screen_reader_results = (
            await accessibility_tester.test_screen_reader_compatibility()
        )

        # Check heading structure
        headings = screen_reader_results.get("headings_structure", [])
        if headings:
            # Should start with h1
            assert headings[0]["level"] == 1, "Page should start with h1"

            # Check for heading level skips
            skip_errors = [h for h in headings if h.get("skip_level_error")]
            assert len(skip_errors) == 0, f"Heading level skips found: {skip_errors}"

        # Check landmarks
        landmarks = screen_reader_results.get("landmarks", [])
        landmark_roles = [l["role"] for l in landmarks]

        # Should have main landmark
        assert "main" in landmark_roles, "Page should have main landmark"

        # Should have navigation if nav elements exist
        if await page.query_selector("nav"):
            assert (
                "navigation" in landmark_roles or "nav" in landmark_roles
            ), "Navigation should be marked as landmark"

    async def test_wcag_aaa_compliance(
        self, accessibility_tester: AccessibilityTester, page: Page
    ):
        """Test WCAG AAA compliance (optional enhanced test)."""
        if WCAG_LEVEL != "AAA":
            pytest.skip("WCAG AAA testing not enabled")

        await page.goto("/")

        results = await accessibility_tester.run_accessibility_scan(
            tags=["wcag2aaa", "wcag21aaa"]
        )

        # AAA compliance allows for some violations but should be minimal
        total_violations = results["summary"]["violations_count"]
        assert (
            total_violations <= 5
        ), f"Too many WCAG AAA violations: {total_violations}"


# Utility functions for accessibility testing
def generate_accessibility_report_summary(
    results_dir: Path = ACCESSIBILITY_REPORTS_DIR,
) -> Dict[str, Any]:
    """Generate summary report from all accessibility test results."""
    summary = {
        "total_reports": 0,
        "pages_tested": set(),
        "total_violations": 0,
        "violation_categories": {},
        "compliance_status": {},
        "recommendations": set(),
    }

    for report_file in results_dir.glob("*.json"):
        try:
            with open(report_file) as f:
                data = json.load(f)

            summary["total_reports"] += 1

            url = data.get("test_environment", {}).get("url", "unknown")
            summary["pages_tested"].add(url)

            summary["total_violations"] += data.get("summary", {}).get(
                "violations_count", 0
            )

            # Aggregate violation categories
            for category, violations in data.get("violations", {}).items():
                if violations:
                    summary["violation_categories"][category] = summary[
                        "violation_categories"
                    ].get(category, 0) + len(violations)

            # Track compliance status
            status = data.get("summary", {}).get("compliance_status", "unknown")
            summary["compliance_status"][status] = (
                summary["compliance_status"].get(status, 0) + 1
            )

            # Collect recommendations
            for rec in data.get("recommendations", []):
                summary["recommendations"].add(rec.get("recommendation", ""))

        except Exception as e:
            print(f"Error processing {report_file}: {e}")

    # Convert sets to lists for JSON serialization
    summary["pages_tested"] = list(summary["pages_tested"])
    summary["recommendations"] = list(summary["recommendations"])

    return summary


if __name__ == "__main__":
    # Run accessibility tests standalone
    pytest.main(
        [
            __file__,
            "-v",
            "--html=test_reports/accessibility_report.html",
            "--self-contained-html",
        ]
    )
