"""
Automated Accessibility Testing Procedures
Comprehensive accessibility test suite with CI/CD integration
"""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pytest
from playwright.async_api import Page, Browser, BrowserContext, expect
from tests.ui.accessibility.wcag_validation_framework import WCAGValidationFramework, WCAGLevel


class AccessibilityTestRunner:
    """Automated accessibility test runner with comprehensive reporting"""
    
    def __init__(self):
        self.results = []
        self.reports_dir = Path("test_reports/accessibility")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Test configurations for different scenarios
        self.test_scenarios = {
            "smoke": {
                "pages": ["/", "/dashboard"],
                "browsers": ["chromium"],
                "viewports": [{"width": 1920, "height": 1080}],
                "timeout": 300
            },
            "comprehensive": {
                "pages": ["/", "/dashboard", "/datasets", "/datasets/upload", "/models", "/analysis", "/settings"],
                "browsers": ["chromium", "firefox", "webkit"],
                "viewports": [
                    {"width": 320, "height": 568},   # Mobile
                    {"width": 768, "height": 1024},  # Tablet
                    {"width": 1920, "height": 1080}  # Desktop
                ],
                "timeout": 1800
            },
            "critical": {
                "pages": ["/", "/dashboard", "/datasets", "/models"],
                "browsers": ["chromium", "firefox"],
                "viewports": [{"width": 1920, "height": 1080}],
                "timeout": 600
            }
        }
    
    async def run_accessibility_test_suite(self, scenario: str = "comprehensive", base_url: str = "http://localhost:8000") -> Dict[str, Any]:
        """Run complete accessibility test suite"""
        print(f"🚀 Starting accessibility test suite: {scenario}")
        start_time = datetime.now()
        
        config = self.test_scenarios.get(scenario, self.test_scenarios["comprehensive"])
        
        suite_results = {
            "scenario": scenario,
            "start_time": start_time.isoformat(),
            "base_url": base_url,
            "configuration": config,
            "browser_results": {},
            "summary": {},
            "critical_issues": [],
            "recommendations": [],
            "compliance_scores": {},
            "execution_time": 0.0
        }
        
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            for browser_name in config["browsers"]:
                print(f"  🌐 Testing with {browser_name}")
                
                browser = await getattr(p, browser_name).launch(
                    headless=True,
                    args=['--disable-web-security', '--disable-features=VizDisplayCompositor']
                )
                
                browser_results = await self._test_browser(browser, browser_name, config, base_url)
                suite_results["browser_results"][browser_name] = browser_results
                
                await browser.close()
        
        # Generate comprehensive summary
        suite_results["summary"] = self._generate_suite_summary(suite_results["browser_results"])
        suite_results["compliance_scores"] = self._calculate_suite_compliance(suite_results["browser_results"])
        suite_results["critical_issues"] = self._extract_critical_issues(suite_results["browser_results"])
        suite_results["recommendations"] = self._generate_suite_recommendations(suite_results["browser_results"])
        suite_results["execution_time"] = (datetime.now() - start_time).total_seconds()
        
        # Save comprehensive report
        await self._save_suite_report(suite_results)
        
        print(f"✅ Accessibility test suite completed in {suite_results['execution_time']:.1f}s")
        print(f"📊 Overall Compliance: {suite_results['compliance_scores']['average']:.1f}%")
        
        return suite_results
    
    async def _test_browser(self, browser: Browser, browser_name: str, config: Dict[str, Any], base_url: str) -> Dict[str, Any]:
        """Test accessibility in a specific browser"""
        browser_results = {
            "browser": browser_name,
            "viewport_results": {},
            "page_results": {},
            "summary": {},
            "compliance_score": 0.0
        }
        
        for viewport in config["viewports"]:
            viewport_key = f"{viewport['width']}x{viewport['height']}"
            print(f"    📱 Testing viewport: {viewport_key}")
            
            context = await browser.new_context(
                viewport=viewport,
                user_agent=f"Pynomaly-AccessibilityTester/{browser_name}"
            )
            
            page = await context.new_page()
            
            # Enable accessibility tree
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => false,
                });
            """)
            
            viewport_results = await self._test_viewport(page, config["pages"], base_url)
            browser_results["viewport_results"][viewport_key] = viewport_results
            
            await context.close()
        
        # Aggregate browser results
        browser_results["summary"] = self._aggregate_browser_results(browser_results["viewport_results"])
        browser_results["compliance_score"] = self._calculate_browser_compliance(browser_results["viewport_results"])
        
        return browser_results
    
    async def _test_viewport(self, page: Page, pages: List[str], base_url: str) -> Dict[str, Any]:
        """Test accessibility at a specific viewport"""
        viewport_results = {
            "viewport": page.viewport_size,
            "page_results": [],
            "summary": {},
            "compliance_score": 0.0
        }
        
        wcag_validator = WCAGValidationFramework(target_level=WCAGLevel.AA)
        
        for page_path in pages:
            page_url = f"{base_url}{page_path}"
            print(f"      📄 Testing: {page_path}")
            
            try:
                # Navigate with comprehensive error handling
                await page.goto(page_url, wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(2000)  # Allow dynamic content
                
                # Run WCAG validation
                wcag_result = await wcag_validator.run_axe_scan(page, page_path)
                
                # Run additional Pynomaly-specific tests
                pynomaly_result = await self._run_pynomaly_specific_tests(page, page_path)
                
                # Combine results
                combined_result = self._combine_test_results(wcag_result, pynomaly_result)
                viewport_results["page_results"].append(combined_result)
                
            except Exception as e:
                print(f"        ❌ Error testing {page_path}: {str(e)}")
                error_result = {
                    "page_url": page_url,
                    "page_path": page_path,
                    "error": str(e),
                    "violations": [],
                    "compliance_score": 0.0
                }
                viewport_results["page_results"].append(error_result)
        
        # Calculate viewport summary
        viewport_results["summary"] = self._calculate_viewport_summary(viewport_results["page_results"])
        viewport_results["compliance_score"] = self._calculate_viewport_compliance(viewport_results["page_results"])
        
        return viewport_results
    
    async def _run_pynomaly_specific_tests(self, page: Page, page_path: str) -> Dict[str, Any]:
        """Run Pynomaly-specific accessibility tests"""
        specific_tests = {
            "page_path": page_path,
            "tests": [],
            "violations": [],
            "passes": []
        }
        
        # Test 1: Data Table Accessibility
        if page_path in ["/datasets", "/models", "/analysis"]:
            table_result = await self._test_data_table_accessibility(page)
            specific_tests["tests"].append(table_result)
            if table_result["violations"]:
                specific_tests["violations"].extend(table_result["violations"])
        
        # Test 2: Chart/Visualization Accessibility
        if page_path in ["/dashboard", "/analysis"]:
            chart_result = await self._test_chart_accessibility(page)
            specific_tests["tests"].append(chart_result)
            if chart_result["violations"]:
                specific_tests["violations"].extend(chart_result["violations"])
        
        # Test 3: Form Accessibility
        if page_path in ["/datasets/upload", "/settings"]:
            form_result = await self._test_form_accessibility(page)
            specific_tests["tests"].append(form_result)
            if form_result["violations"]:
                specific_tests["violations"].extend(form_result["violations"])
        
        # Test 4: Navigation Accessibility
        nav_result = await self._test_navigation_accessibility(page)
        specific_tests["tests"].append(nav_result)
        if nav_result["violations"]:
            specific_tests["violations"].extend(nav_result["violations"])
        
        # Test 5: PWA Accessibility
        pwa_result = await self._test_pwa_accessibility(page)
        specific_tests["tests"].append(pwa_result)
        if pwa_result["violations"]:
            specific_tests["violations"].extend(pwa_result["violations"])
        
        return specific_tests
    
    async def _test_data_table_accessibility(self, page: Page) -> Dict[str, Any]:
        """Test data table accessibility"""
        test_result = {
            "test_name": "data_table_accessibility",
            "violations": [],
            "passes": []
        }
        
        # Find tables
        tables = await page.query_selector_all("table, [data-component='data-table']")
        
        for table in tables:
            # Check for table headers
            has_headers = await table.query_selector("thead, th")
            if not has_headers:
                test_result["violations"].append({
                    "id": "table-missing-headers",
                    "impact": "serious",
                    "description": "Data table missing proper headers",
                    "help": "Use thead and th elements for table headers"
                })
            else:
                test_result["passes"].append({
                    "id": "table-has-headers",
                    "description": "Table has proper header structure"
                })
            
            # Check for table caption or aria-label
            has_caption = await table.evaluate('''
                el => el.querySelector('caption') || 
                      el.hasAttribute('aria-label') || 
                      el.hasAttribute('aria-labelledby')
            ''')
            
            if not has_caption:
                test_result["violations"].append({
                    "id": "table-missing-caption",
                    "impact": "moderate",
                    "description": "Data table missing caption or label",
                    "help": "Provide a caption or aria-label for data tables"
                })
            else:
                test_result["passes"].append({
                    "id": "table-has-caption",
                    "description": "Table has proper caption or label"
                })
        
        return test_result
    
    async def _test_chart_accessibility(self, page: Page) -> Dict[str, Any]:
        """Test chart/visualization accessibility"""
        test_result = {
            "test_name": "chart_accessibility",
            "violations": [],
            "passes": []
        }
        
        # Find charts
        charts = await page.query_selector_all("svg, canvas, [data-component='anomaly-chart'], .chart")
        
        for chart in charts:
            # Check for alternative text
            has_alt = await chart.evaluate('''
                el => {
                    if (el.tagName === 'SVG') {
                        return el.querySelector('title, desc') || 
                               el.hasAttribute('aria-label') || 
                               el.hasAttribute('aria-labelledby');
                    }
                    return el.hasAttribute('aria-label') || 
                           el.hasAttribute('aria-labelledby') ||
                           el.hasAttribute('role');
                }
            ''')
            
            if not has_alt:
                test_result["violations"].append({
                    "id": "chart-missing-alt",
                    "impact": "serious",
                    "description": "Chart missing alternative text or description",
                    "help": "Provide alternative text for charts and visualizations"
                })
            else:
                test_result["passes"].append({
                    "id": "chart-has-alt",
                    "description": "Chart has proper alternative text"
                })
            
            # Check for data table alternative (if complex chart)
            is_complex = await chart.evaluate('''
                el => {
                    const rect = el.getBoundingClientRect();
                    return rect.width > 400 && rect.height > 300;
                }
            ''')
            
            if is_complex:
                # Look for associated data table
                has_table = await page.query_selector("table")
                if not has_table:
                    test_result["violations"].append({
                        "id": "complex-chart-missing-table",
                        "impact": "moderate",
                        "description": "Complex chart missing data table alternative",
                        "help": "Provide a data table for complex visualizations"
                    })
        
        return test_result
    
    async def _test_form_accessibility(self, page: Page) -> Dict[str, Any]:
        """Test form accessibility"""
        test_result = {
            "test_name": "form_accessibility",
            "violations": [],
            "passes": []
        }
        
        # Find form elements
        forms = await page.query_selector_all("form")
        inputs = await page.query_selector_all("input, select, textarea")
        
        # Test form structure
        for form in forms:
            # Check for fieldsets in complex forms
            fieldsets = await form.query_selector_all("fieldset")
            inputs_count = len(await form.query_selector_all("input, select, textarea"))
            
            if inputs_count > 5 and len(fieldsets) == 0:
                test_result["violations"].append({
                    "id": "complex-form-missing-fieldset",
                    "impact": "moderate",
                    "description": "Complex form missing fieldset grouping",
                    "help": "Group related form controls with fieldset elements"
                })
        
        # Test individual inputs
        for input_elem in inputs:
            # Check for labels
            has_label = await input_elem.evaluate('''
                el => {
                    const id = el.id;
                    const hasExplicitLabel = id && document.querySelector(`label[for="${id}"]`);
                    const hasImplicitLabel = el.closest('label');
                    const hasAriaLabel = el.hasAttribute('aria-label') || el.hasAttribute('aria-labelledby');
                    return hasExplicitLabel || hasImplicitLabel || hasAriaLabel;
                }
            ''')
            
            if not has_label:
                test_result["violations"].append({
                    "id": "input-missing-label",
                    "impact": "serious",
                    "description": "Form input missing accessible label",
                    "help": "Associate labels with form controls"
                })
            else:
                test_result["passes"].append({
                    "id": "input-has-label",
                    "description": "Form input has proper label"
                })
            
            # Check for required field indication
            is_required = await input_elem.get_attribute("required")
            if is_required:
                has_required_indication = await input_elem.evaluate('''
                    el => {
                        const label = el.closest('label') || 
                                     (el.id && document.querySelector(`label[for="${el.id}"]`));
                        return label && (label.textContent.includes('*') || 
                                        label.textContent.includes('required') ||
                                        el.hasAttribute('aria-required'));
                    }
                ''')
                
                if not has_required_indication:
                    test_result["violations"].append({
                        "id": "required-field-not-indicated",
                        "impact": "moderate",
                        "description": "Required field not clearly indicated",
                        "help": "Clearly mark required form fields"
                    })
        
        return test_result
    
    async def _test_navigation_accessibility(self, page: Page) -> Dict[str, Any]:
        """Test navigation accessibility"""
        test_result = {
            "test_name": "navigation_accessibility",
            "violations": [],
            "passes": []
        }
        
        # Test skip links
        skip_links = await page.query_selector_all("a[href^='#'], .skip-link")
        if len(skip_links) == 0:
            test_result["violations"].append({
                "id": "missing-skip-links",
                "impact": "moderate",
                "description": "Page missing skip links for keyboard navigation",
                "help": "Provide skip links to main content"
            })
        else:
            test_result["passes"].append({
                "id": "has-skip-links",
                "description": "Page has skip links for keyboard navigation"
            })
        
        # Test navigation landmarks
        nav_elements = await page.query_selector_all("nav, [role='navigation']")
        main_elements = await page.query_selector_all("main, [role='main']")
        
        if len(nav_elements) == 0:
            test_result["violations"].append({
                "id": "missing-nav-landmark",
                "impact": "moderate",
                "description": "Page missing navigation landmark",
                "help": "Use nav element or role='navigation'"
            })
        
        if len(main_elements) == 0:
            test_result["violations"].append({
                "id": "missing-main-landmark",
                "impact": "serious",
                "description": "Page missing main content landmark",
                "help": "Use main element or role='main'"
            })
        else:
            test_result["passes"].append({
                "id": "has-main-landmark",
                "description": "Page has main content landmark"
            })
        
        # Test heading hierarchy
        headings = await page.query_selector_all("h1, h2, h3, h4, h5, h6")
        if len(headings) > 0:
            heading_levels = []
            for heading in headings:
                tag_name = await heading.evaluate("el => el.tagName.toLowerCase()")
                level = int(tag_name[1])
                heading_levels.append(level)
            
            # Check for proper hierarchy (no skipped levels)
            for i in range(1, len(heading_levels)):
                if heading_levels[i] > heading_levels[i-1] + 1:
                    test_result["violations"].append({
                        "id": "heading-hierarchy-skipped",
                        "impact": "moderate",
                        "description": "Heading hierarchy skips levels",
                        "help": "Use headings in proper hierarchical order"
                    })
                    break
            else:
                test_result["passes"].append({
                    "id": "proper-heading-hierarchy",
                    "description": "Headings follow proper hierarchy"
                })
        
        return test_result
    
    async def _test_pwa_accessibility(self, page: Page) -> Dict[str, Any]:
        """Test PWA-specific accessibility features"""
        test_result = {
            "test_name": "pwa_accessibility",
            "violations": [],
            "passes": []
        }
        
        # Test offline page accessibility
        if page.url.endswith("/offline"):
            # Check for proper offline messaging
            offline_message = await page.query_selector(".offline-message, .offline-animation")
            if offline_message:
                test_result["passes"].append({
                    "id": "offline-message-present",
                    "description": "Offline page has clear messaging"
                })
            
            # Check for keyboard navigation in offline mode
            focusable_elements = await page.query_selector_all("button:not([disabled]), a:not([disabled]), [tabindex]:not([tabindex='-1'])")
            if len(focusable_elements) > 0:
                test_result["passes"].append({
                    "id": "offline-keyboard-navigation",
                    "description": "Offline page supports keyboard navigation"
                })
        
        # Test installation prompts accessibility
        install_buttons = await page.query_selector_all("[data-action='install'], .install-button")
        for button in install_buttons:
            has_accessible_name = await button.evaluate('''
                el => el.textContent.trim().length > 0 || 
                      el.hasAttribute('aria-label') || 
                      el.hasAttribute('aria-labelledby')
            ''')
            
            if not has_accessible_name:
                test_result["violations"].append({
                    "id": "install-button-missing-name",
                    "impact": "moderate",
                    "description": "Install button missing accessible name",
                    "help": "Provide clear labeling for installation buttons"
                })
        
        # Test notification accessibility
        notifications = await page.query_selector_all(".notification, .toast, [role='alert']")
        for notification in notifications:
            has_live_region = await notification.evaluate('''
                el => el.hasAttribute('aria-live') || 
                      el.hasAttribute('role') && ['alert', 'status'].includes(el.getAttribute('role'))
            ''')
            
            if not has_live_region:
                test_result["violations"].append({
                    "id": "notification-missing-live-region",
                    "impact": "moderate",
                    "description": "Notification missing ARIA live region",
                    "help": "Use aria-live or role='alert' for notifications"
                })
        
        return test_result
    
    def _combine_test_results(self, wcag_result, pynomaly_result) -> Dict[str, Any]:
        """Combine WCAG and Pynomaly-specific test results"""
        combined_violations = list(wcag_result.violations)
        
        # Convert Pynomaly violations to WCAG format
        for violation in pynomaly_result.get("violations", []):
            combined_violations.append(violation)
        
        return {
            "page_url": wcag_result.page_url,
            "page_path": pynomaly_result.get("page_path", ""),
            "total_violations": len(combined_violations),
            "wcag_violations": len(wcag_result.violations),
            "pynomaly_violations": len(pynomaly_result.get("violations", [])),
            "violations": combined_violations,
            "wcag_passes": wcag_result.passes,
            "pynomaly_passes": pynomaly_result.get("passes", []),
            "compliance_score": max(0, 100 - (len(combined_violations) * 5)),  # Simple scoring
            "execution_time": wcag_result.execution_time,
            "timestamp": wcag_result.timestamp
        }
    
    def _calculate_viewport_summary(self, page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary for viewport results"""
        total_pages = len(page_results)
        total_violations = sum(r.get("total_violations", 0) for r in page_results)
        pages_with_errors = len([r for r in page_results if "error" in r])
        
        return {
            "total_pages": total_pages,
            "total_violations": total_violations,
            "pages_with_errors": pages_with_errors,
            "average_violations_per_page": total_violations / max(total_pages, 1)
        }
    
    def _calculate_viewport_compliance(self, page_results: List[Dict[str, Any]]) -> float:
        """Calculate compliance score for viewport"""
        if not page_results:
            return 0.0
        
        total_score = sum(r.get("compliance_score", 0) for r in page_results)
        return total_score / len(page_results)
    
    def _aggregate_browser_results(self, viewport_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results across viewports for a browser"""
        all_violations = 0
        all_pages = 0
        
        for viewport_data in viewport_results.values():
            all_violations += viewport_data["summary"]["total_violations"]
            all_pages += viewport_data["summary"]["total_pages"]
        
        return {
            "total_violations": all_violations,
            "total_pages": all_pages,
            "viewports_tested": len(viewport_results)
        }
    
    def _calculate_browser_compliance(self, viewport_results: Dict[str, Any]) -> float:
        """Calculate overall compliance for browser"""
        if not viewport_results:
            return 0.0
        
        total_score = sum(vr["compliance_score"] for vr in viewport_results.values())
        return total_score / len(viewport_results)
    
    def _generate_suite_summary(self, browser_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary across all browsers"""
        total_violations = 0
        total_pages = 0
        
        for browser_data in browser_results.values():
            total_violations += browser_data["summary"]["total_violations"]
            total_pages += browser_data["summary"]["total_pages"]
        
        return {
            "browsers_tested": len(browser_results),
            "total_violations": total_violations,
            "total_pages": total_pages,
            "average_violations_per_page": total_violations / max(total_pages, 1)
        }
    
    def _calculate_suite_compliance(self, browser_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate compliance scores across suite"""
        browser_scores = {browser: data["compliance_score"] 
                         for browser, data in browser_results.items()}
        
        average_score = sum(browser_scores.values()) / len(browser_scores) if browser_scores else 0
        
        return {
            "by_browser": browser_scores,
            "average": average_score,
            "minimum": min(browser_scores.values()) if browser_scores else 0,
            "maximum": max(browser_scores.values()) if browser_scores else 0
        }
    
    def _extract_critical_issues(self, browser_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract critical accessibility issues"""
        critical_issues = []
        
        for browser, browser_data in browser_results.items():
            for viewport, viewport_data in browser_data.get("viewport_results", {}).items():
                for page_result in viewport_data.get("page_results", []):
                    for violation in page_result.get("violations", []):
                        if hasattr(violation, 'impact') and violation.impact == "critical":
                            critical_issues.append({
                                "browser": browser,
                                "viewport": viewport,
                                "page": page_result.get("page_url", ""),
                                "violation": violation
                            })
        
        return critical_issues
    
    def _generate_suite_recommendations(self, browser_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate accessibility improvement recommendations"""
        recommendations = [
            {
                "priority": "high",
                "category": "WCAG Compliance",
                "recommendation": "Address all critical and serious accessibility violations",
                "details": "Focus on violations that prevent users from accessing content"
            },
            {
                "priority": "medium", 
                "category": "Keyboard Navigation",
                "recommendation": "Ensure all interactive elements are keyboard accessible",
                "details": "Test tab navigation and provide visible focus indicators"
            },
            {
                "priority": "medium",
                "category": "Screen Reader Support",
                "recommendation": "Improve semantic markup and ARIA labels",
                "details": "Use proper heading hierarchy and descriptive labels"
            },
            {
                "priority": "low",
                "category": "Progressive Enhancement",
                "recommendation": "Test accessibility across different browsers and devices",
                "details": "Ensure consistent experience across all platforms"
            }
        ]
        
        return recommendations
    
    async def _save_suite_report(self, suite_results: Dict[str, Any]):
        """Save comprehensive suite report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario = suite_results["scenario"]
        
        # Save JSON report
        json_path = self.reports_dir / f"accessibility_suite_{scenario}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(suite_results, f, indent=2, default=str)
        
        # Save HTML report
        html_path = self.reports_dir / f"accessibility_suite_{scenario}_{timestamp}.html"
        html_content = self._generate_suite_html_report(suite_results)
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        # Save CI-friendly summary
        summary_path = self.reports_dir / f"accessibility_summary_{scenario}_{timestamp}.txt"
        summary_content = self._generate_ci_summary(suite_results)
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        
        print(f"📊 Suite reports saved:")
        print(f"  JSON: {json_path}")
        print(f"  HTML: {html_path}")
        print(f"  Summary: {summary_path}")
    
    def _generate_suite_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report for test suite"""
        compliance_score = results["compliance_scores"]["average"]
        score_color = "#27ae60" if compliance_score >= 80 else "#f39c12" if compliance_score >= 60 else "#e74c3c"
        
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Pynomaly Accessibility Test Suite Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 40px; }}
                .score {{ font-size: 3em; font-weight: bold; color: {score_color}; }}
                .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
                .metric {{ background: #ecf0f1; padding: 20px; border-radius: 6px; text-align: center; }}
                .browser-results {{ margin: 20px 0; }}
                .browser {{ background: #fff; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; padding: 15px; }}
                .recommendations {{ background: #fff8dc; border-left: 4px solid #f39c12; padding: 20px; margin: 20px 0; }}
                .critical {{ color: #e74c3c; }}
                .high {{ color: #f39c12; }}
                .medium {{ color: #3498db; }}
                .low {{ color: #95a5a6; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 Accessibility Test Suite Report</h1>
                    <h2>Scenario: {results['scenario'].title()}</h2>
                    <div class="score">{compliance_score:.1f}%</div>
                    <p>Average Compliance Score</p>
                    <p><small>Generated: {results['start_time']}</small></p>
                </div>
                
                <div class="summary">
                    <div class="metric">
                        <div style="font-size: 2em; font-weight: bold;">{results['summary']['browsers_tested']}</div>
                        <div>Browsers Tested</div>
                    </div>
                    <div class="metric">
                        <div style="font-size: 2em; font-weight: bold;">{results['summary']['total_pages']}</div>
                        <div>Total Pages</div>
                    </div>
                    <div class="metric">
                        <div style="font-size: 2em; font-weight: bold;">{results['summary']['total_violations']}</div>
                        <div>Total Violations</div>
                    </div>
                    <div class="metric">
                        <div style="font-size: 2em; font-weight: bold;">{len(results['critical_issues'])}</div>
                        <div>Critical Issues</div>
                    </div>
                </div>
                
                <div class="browser-results">
                    <h3>📊 Browser Results</h3>
                    {''.join([f'''
                    <div class="browser">
                        <h4>{browser_name.title()}</h4>
                        <p>Compliance Score: {browser_data['compliance_score']:.1f}%</p>
                        <p>Violations: {browser_data['summary']['total_violations']}</p>
                        <p>Viewports Tested: {browser_data['summary']['viewports_tested']}</p>
                    </div>
                    ''' for browser_name, browser_data in results['browser_results'].items()])}
                </div>
                
                <div class="recommendations">
                    <h3>💡 Recommendations</h3>
                    <ul>
                        {''.join([f"<li class='{rec['priority']}'><strong>{rec['category']}</strong> - {rec['recommendation']}</li>" for rec in results['recommendations']])}
                    </ul>
                </div>
                
                <div style="margin-top: 40px; text-align: center; color: #666;">
                    <p>Execution Time: {results['execution_time']:.1f} seconds</p>
                    <p>Generated by Pynomaly Accessibility Test Suite</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _generate_ci_summary(self, results: Dict[str, Any]) -> str:
        """Generate CI-friendly summary"""
        compliance_score = results["compliance_scores"]["average"]
        status = "PASS" if compliance_score >= 80 else "WARN" if compliance_score >= 60 else "FAIL"
        
        summary = f"""
ACCESSIBILITY TEST SUITE SUMMARY
================================

Status: {status}
Scenario: {results['scenario']}
Overall Compliance: {compliance_score:.1f}%

Summary:
- Browsers Tested: {results['summary']['browsers_tested']}
- Total Pages: {results['summary']['total_pages']} 
- Total Violations: {results['summary']['total_violations']}
- Critical Issues: {len(results['critical_issues'])}
- Execution Time: {results['execution_time']:.1f}s

Browser Scores:
"""
        
        for browser, score in results["compliance_scores"]["by_browser"].items():
            summary += f"- {browser.title()}: {score:.1f}%\n"
        
        if results['critical_issues']:
            summary += f"\nCritical Issues Found: {len(results['critical_issues'])}\n"
            summary += "RECOMMENDATION: Address critical accessibility issues before deployment.\n"
        
        return summary


# Pytest integration
@pytest.mark.accessibility
@pytest.mark.integration
class TestAutomatedAccessibility:
    """Automated accessibility test class for CI/CD integration"""
    
    @pytest.mark.smoke
    async def test_accessibility_smoke(self, page: Page):
        """Quick accessibility smoke test"""
        runner = AccessibilityTestRunner()
        results = await runner.run_accessibility_test_suite("smoke")
        
        assert results["compliance_scores"]["average"] >= 70.0, f"Accessibility smoke test failed with {results['compliance_scores']['average']:.1f}% compliance"
        assert len(results["critical_issues"]) == 0, f"Found {len(results['critical_issues'])} critical accessibility issues"
    
    @pytest.mark.comprehensive
    async def test_accessibility_comprehensive(self, page: Page):
        """Comprehensive accessibility test suite"""
        runner = AccessibilityTestRunner()
        results = await runner.run_accessibility_test_suite("comprehensive")
        
        assert results["compliance_scores"]["average"] >= 80.0, f"Comprehensive accessibility test failed with {results['compliance_scores']['average']:.1f}% compliance"
        assert len(results["critical_issues"]) == 0, "Critical accessibility issues found"
        assert results["summary"]["total_violations"] <= 20, f"Too many accessibility violations: {results['summary']['total_violations']}"


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Pynomaly accessibility tests")
    parser.add_argument("--scenario", choices=["smoke", "comprehensive", "critical"], 
                       default="comprehensive", help="Test scenario to run")
    parser.add_argument("--base-url", default="http://localhost:8000", 
                       help="Base URL to test")
    parser.add_argument("--ci", action="store_true", 
                       help="Run in CI mode (exit with error code on failure)")
    
    args = parser.parse_args()
    
    async def main():
        runner = AccessibilityTestRunner()
        results = await runner.run_accessibility_test_suite(args.scenario, args.base_url)
        
        compliance_score = results["compliance_scores"]["average"]
        critical_issues = len(results["critical_issues"])
        
        print(f"\n🎯 Accessibility Test Results:")
        print(f"Scenario: {args.scenario}")
        print(f"Compliance Score: {compliance_score:.1f}%")
        print(f"Critical Issues: {critical_issues}")
        print(f"Total Violations: {results['summary']['total_violations']}")
        
        if args.ci:
            if compliance_score < 80.0 or critical_issues > 0:
                print("❌ Accessibility tests failed")
                sys.exit(1)
            else:
                print("✅ Accessibility tests passed")
                sys.exit(0)
    
    asyncio.run(main())