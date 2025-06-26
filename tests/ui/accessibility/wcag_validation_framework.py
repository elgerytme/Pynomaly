"""
WCAG 2.1 AA Compliance Validation Framework
Comprehensive accessibility testing and validation for Pynomaly UI
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import pytest
from playwright.async_api import Page, Browser, BrowserContext
from tests.ui.enhanced_page_objects.base_page import BasePage


class WCAGLevel(Enum):
    """WCAG Conformance Levels"""
    A = "A"
    AA = "AA"
    AAA = "AAA"


class WCAGPrinciple(Enum):
    """WCAG 2.1 Principles"""
    PERCEIVABLE = "1_perceivable"
    OPERABLE = "2_operable"
    UNDERSTANDABLE = "3_understandable"
    ROBUST = "4_robust"


@dataclass
class WCAGViolation:
    """WCAG Violation Details"""
    id: str
    impact: str  # critical, serious, moderate, minor
    tags: List[str]
    description: str
    help: str
    help_url: str
    nodes: List[Dict[str, Any]]
    principle: WCAGPrinciple
    guideline: str
    success_criterion: str
    level: WCAGLevel
    page_url: str
    timestamp: str


@dataclass
class AccessibilityTestResult:
    """Accessibility Test Result"""
    page_url: str
    test_name: str
    total_violations: int
    critical_violations: int
    serious_violations: int
    moderate_violations: int
    minor_violations: int
    violations: List[WCAGViolation]
    passes: List[Dict[str, Any]]
    inapplicable: List[Dict[str, Any]]
    incomplete: List[Dict[str, Any]]
    execution_time: float
    timestamp: str
    browser: str
    viewport: Dict[str, int]
    user_agent: str


class WCAGValidationFramework:
    """Comprehensive WCAG 2.1 AA Validation Framework"""
    
    def __init__(self, target_level: WCAGLevel = WCAGLevel.AA):
        self.target_level = target_level
        self.results: List[AccessibilityTestResult] = []
        self.reports_dir = Path("test_reports/accessibility")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # WCAG 2.1 AA Success Criteria
        self.wcag_aa_criteria = {
            "1.1.1": "Non-text Content",
            "1.2.1": "Audio-only and Video-only (Prerecorded)",
            "1.2.2": "Captions (Prerecorded)",
            "1.2.3": "Audio Description or Media Alternative (Prerecorded)",
            "1.3.1": "Info and Relationships",
            "1.3.2": "Meaningful Sequence",
            "1.3.3": "Sensory Characteristics",
            "1.3.4": "Orientation",
            "1.3.5": "Identify Input Purpose",
            "1.4.1": "Use of Color",
            "1.4.2": "Audio Control",
            "1.4.3": "Contrast (Minimum)",
            "1.4.4": "Resize text",
            "1.4.5": "Images of Text",
            "1.4.10": "Reflow",
            "1.4.11": "Non-text Contrast",
            "1.4.12": "Text Spacing",
            "1.4.13": "Content on Hover or Focus",
            "2.1.1": "Keyboard",
            "2.1.2": "No Keyboard Trap",
            "2.1.4": "Character Key Shortcuts",
            "2.2.1": "Timing Adjustable",
            "2.2.2": "Pause, Stop, Hide",
            "2.3.1": "Three Flashes or Below Threshold",
            "2.4.1": "Bypass Blocks",
            "2.4.2": "Page Titled",
            "2.4.3": "Focus Order",
            "2.4.4": "Link Purpose (In Context)",
            "2.4.5": "Multiple Ways",
            "2.4.6": "Headings and Labels",
            "2.4.7": "Focus Visible",
            "2.5.1": "Pointer Gestures",
            "2.5.2": "Pointer Cancellation",
            "2.5.3": "Label in Name",
            "2.5.4": "Motion Actuation",
            "3.1.1": "Language of Page",
            "3.1.2": "Language of Parts",
            "3.2.1": "On Focus",
            "3.2.2": "On Input",
            "3.2.3": "Consistent Navigation",
            "3.2.4": "Consistent Identification",
            "3.3.1": "Error Identification",
            "3.3.2": "Labels or Instructions",
            "3.3.3": "Error Suggestion",
            "3.3.4": "Error Prevention (Legal, Financial, Data)",
            "4.1.1": "Parsing",
            "4.1.2": "Name, Role, Value",
            "4.1.3": "Status Messages"
        }
        
        # Priority test pages for Pynomaly
        self.test_pages = [
            "/",
            "/dashboard",
            "/datasets",
            "/datasets/upload",
            "/models",
            "/analysis",
            "/settings",
            "/docs",
            "/offline"
        ]
    
    async def run_comprehensive_validation(self, page: Page, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
        """Run comprehensive WCAG 2.1 AA validation across all test pages"""
        print(f"🔍 Starting comprehensive WCAG 2.1 AA validation...")
        start_time = time.time()
        
        validation_results = {
            "summary": {},
            "page_results": [],
            "critical_issues": [],
            "recommendations": [],
            "compliance_score": 0.0,
            "execution_time": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        for page_path in self.test_pages:
            page_url = f"{base_url}{page_path}"
            print(f"  📄 Testing page: {page_path}")
            
            try:
                # Navigate to page
                await page.goto(page_url, wait_until="networkidle")
                await page.wait_for_timeout(1000)  # Allow dynamic content to load
                
                # Run accessibility scan
                result = await self.run_axe_scan(page, page_path)
                validation_results["page_results"].append(result)
                
                # Additional manual checks
                manual_result = await self.run_manual_checks(page, page_path)
                result.violations.extend(manual_result.violations)
                
                # Check for critical issues
                critical_violations = [v for v in result.violations if v.impact == "critical"]
                validation_results["critical_issues"].extend(critical_violations)
                
            except Exception as e:
                print(f"  ❌ Error testing {page_path}: {str(e)}")
                continue
        
        # Generate summary and compliance score
        validation_results["summary"] = self._generate_summary(validation_results["page_results"])
        validation_results["compliance_score"] = self._calculate_compliance_score(validation_results["page_results"])
        validation_results["recommendations"] = self._generate_recommendations(validation_results["page_results"])
        validation_results["execution_time"] = time.time() - start_time
        
        # Save comprehensive report
        await self._save_comprehensive_report(validation_results)
        
        print(f"✅ WCAG validation completed in {validation_results['execution_time']:.2f}s")
        print(f"📊 Compliance Score: {validation_results['compliance_score']:.1f}%")
        
        return validation_results
    
    async def run_axe_scan(self, page: Page, page_name: str) -> AccessibilityTestResult:
        """Run axe-core accessibility scan"""
        start_time = time.time()
        
        # Inject axe-core
        await self._inject_axe_core(page)
        
        # Configure axe for WCAG 2.1 AA
        axe_config = {
            "rules": {},
            "tags": ["wcag2a", "wcag2aa", "wcag21aa"],
            "reporter": "v2"
        }
        
        # Run axe scan
        results = await page.evaluate(f"""
            async () => {{
                try {{
                    const results = await axe.run(document, {json.dumps(axe_config)});
                    return results;
                }} catch (error) {{
                    return {{ error: error.message }};
                }}
            }}
        """)
        
        if "error" in results:
            raise Exception(f"Axe scan failed: {results['error']}")
        
        # Process violations
        violations = []
        for violation in results.get("violations", []):
            wcag_violation = self._create_wcag_violation(violation, page.url)
            violations.append(wcag_violation)
        
        # Get browser info
        browser_info = await self._get_browser_info(page)
        
        # Create test result
        test_result = AccessibilityTestResult(
            page_url=page.url,
            test_name=f"WCAG_2.1_AA_Scan_{page_name}",
            total_violations=len(violations),
            critical_violations=len([v for v in violations if v.impact == "critical"]),
            serious_violations=len([v for v in violations if v.impact == "serious"]),
            moderate_violations=len([v for v in violations if v.impact == "moderate"]),
            minor_violations=len([v for v in violations if v.impact == "minor"]),
            violations=violations,
            passes=results.get("passes", []),
            inapplicable=results.get("inapplicable", []),
            incomplete=results.get("incomplete", []),
            execution_time=time.time() - start_time,
            timestamp=datetime.now().isoformat(),
            browser=browser_info["name"],
            viewport=browser_info["viewport"],
            user_agent=browser_info["user_agent"]
        )
        
        self.results.append(test_result)
        return test_result
    
    async def run_manual_checks(self, page: Page, page_name: str) -> AccessibilityTestResult:
        """Run manual accessibility checks for Pynomaly-specific requirements"""
        violations = []
        
        # Check 1: Keyboard Navigation
        keyboard_violations = await self._check_keyboard_navigation(page)
        violations.extend(keyboard_violations)
        
        # Check 2: Color Contrast (visual verification)
        contrast_violations = await self._check_color_contrast(page)
        violations.extend(contrast_violations)
        
        # Check 3: Form Labels and Instructions
        form_violations = await self._check_form_accessibility(page)
        violations.extend(form_violations)
        
        # Check 4: Data Visualization Accessibility
        chart_violations = await self._check_chart_accessibility(page)
        violations.extend(chart_violations)
        
        # Check 5: Responsive Design Accessibility
        responsive_violations = await self._check_responsive_accessibility(page)
        violations.extend(responsive_violations)
        
        # Check 6: Dynamic Content Accessibility
        dynamic_violations = await self._check_dynamic_content_accessibility(page)
        violations.extend(dynamic_violations)
        
        return AccessibilityTestResult(
            page_url=page.url,
            test_name=f"Manual_Checks_{page_name}",
            total_violations=len(violations),
            critical_violations=len([v for v in violations if v.impact == "critical"]),
            serious_violations=len([v for v in violations if v.impact == "serious"]),
            moderate_violations=len([v for v in violations if v.impact == "moderate"]),
            minor_violations=len([v for v in violations if v.impact == "minor"]),
            violations=violations,
            passes=[],
            inapplicable=[],
            incomplete=[],
            execution_time=0.0,
            timestamp=datetime.now().isoformat(),
            browser="manual",
            viewport={},
            user_agent=""
        )
    
    async def _inject_axe_core(self, page: Page):
        """Inject axe-core library"""
        try:
            # Try to load from CDN first
            await page.add_script_tag(url="https://cdn.jsdelivr.net/npm/axe-core@4.8.2/axe.min.js")
        except:
            # Fallback to local file
            axe_script = (Path(__file__).parent / "axe-core.min.js").read_text()
            await page.add_script_tag(content=axe_script)
    
    async def _check_keyboard_navigation(self, page: Page) -> List[WCAGViolation]:
        """Check keyboard navigation accessibility"""
        violations = []
        
        # Find all interactive elements
        interactive_elements = await page.query_selector_all(
            'button, a, input, select, textarea, [tabindex], [role="button"], [role="link"]'
        )
        
        for element in interactive_elements:
            # Check if element is focusable
            is_focusable = await element.evaluate('el => el.tabIndex >= 0 || el.matches("a, button, input, select, textarea")')
            
            if not is_focusable:
                # Check if element has proper ARIA attributes
                has_aria = await element.evaluate('el => el.hasAttribute("role") || el.hasAttribute("aria-label")')
                
                if not has_aria:
                    violations.append(WCAGViolation(
                        id="keyboard-navigation-missing",
                        impact="serious",
                        tags=["keyboard", "wcag2aa"],
                        description="Interactive element is not keyboard accessible",
                        help="Ensure all interactive elements are keyboard accessible",
                        help_url="https://www.w3.org/WAI/WCAG21/Understanding/keyboard.html",
                        nodes=[{"html": await element.inner_html()}],
                        principle=WCAGPrinciple.OPERABLE,
                        guideline="2.1",
                        success_criterion="2.1.1",
                        level=WCAGLevel.A,
                        page_url=page.url,
                        timestamp=datetime.now().isoformat()
                    ))
        
        return violations
    
    async def _check_color_contrast(self, page: Page) -> List[WCAGViolation]:
        """Check color contrast ratios"""
        violations = []
        
        # This would typically use a more sophisticated color contrast analyzer
        # For now, we'll check for common patterns that might have contrast issues
        
        # Check if design system CSS is loaded (indicates proper contrast should be in place)
        has_design_system = await page.evaluate('''
            () => {
                const links = Array.from(document.querySelectorAll('link[rel="stylesheet"]'));
                return links.some(link => link.href.includes('design-system.css'));
            }
        ''')
        
        if not has_design_system:
            violations.append(WCAGViolation(
                id="design-system-missing",
                impact="moderate",
                tags=["color-contrast", "wcag2aa"],
                description="Design system CSS not loaded, color contrast may not be compliant",
                help="Ensure design system CSS is loaded for proper color contrast",
                help_url="https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html",
                nodes=[],
                principle=WCAGPrinciple.PERCEIVABLE,
                guideline="1.4",
                success_criterion="1.4.3",
                level=WCAGLevel.AA,
                page_url=page.url,
                timestamp=datetime.now().isoformat()
            ))
        
        return violations
    
    async def _check_form_accessibility(self, page: Page) -> List[WCAGViolation]:
        """Check form accessibility"""
        violations = []
        
        # Find all form inputs
        form_inputs = await page.query_selector_all('input, select, textarea')
        
        for input_element in form_inputs:
            # Check for associated label
            has_label = await input_element.evaluate('''
                el => {
                    const id = el.id;
                    const hasLabel = id && document.querySelector(`label[for="${id}"]`);
                    const hasAriaLabel = el.hasAttribute('aria-label') || el.hasAttribute('aria-labelledby');
                    return hasLabel || hasAriaLabel;
                }
            ''')
            
            if not has_label:
                violations.append(WCAGViolation(
                    id="form-label-missing",
                    impact="serious",
                    tags=["forms", "wcag2aa"],
                    description="Form input missing accessible label",
                    help="Provide labels or instructions for form inputs",
                    help_url="https://www.w3.org/WAI/WCAG21/Understanding/labels-or-instructions.html",
                    nodes=[{"html": await input_element.outer_html()}],
                    principle=WCAGPrinciple.UNDERSTANDABLE,
                    guideline="3.3",
                    success_criterion="3.3.2",
                    level=WCAGLevel.A,
                    page_url=page.url,
                    timestamp=datetime.now().isoformat()
                ))
        
        return violations
    
    async def _check_chart_accessibility(self, page: Page) -> List[WCAGViolation]:
        """Check data visualization accessibility"""
        violations = []
        
        # Look for chart containers
        chart_elements = await page.query_selector_all('[data-component="anomaly-chart"], .chart, svg')
        
        for chart in chart_elements:
            # Check for alternative text or description
            has_alt = await chart.evaluate('''
                el => {
                    return el.hasAttribute('aria-label') || 
                           el.hasAttribute('aria-labelledby') || 
                           el.hasAttribute('aria-describedby') ||
                           el.querySelector('title, desc');
                }
            ''')
            
            if not has_alt:
                violations.append(WCAGViolation(
                    id="chart-alt-missing",
                    impact="serious",
                    tags=["images", "wcag2aa"],
                    description="Data visualization missing alternative text",
                    help="Provide alternative text for charts and graphs",
                    help_url="https://www.w3.org/WAI/WCAG21/Understanding/non-text-content.html",
                    nodes=[{"html": await chart.outer_html()}],
                    principle=WCAGPrinciple.PERCEIVABLE,
                    guideline="1.1",
                    success_criterion="1.1.1",
                    level=WCAGLevel.A,
                    page_url=page.url,
                    timestamp=datetime.now().isoformat()
                ))
        
        return violations
    
    async def _check_responsive_accessibility(self, page: Page) -> List[WCAGViolation]:
        """Check responsive design accessibility"""
        violations = []
        
        # Test different viewport sizes
        viewports = [
            {"width": 320, "height": 568},  # Mobile
            {"width": 768, "height": 1024}, # Tablet
            {"width": 1920, "height": 1080} # Desktop
        ]
        
        original_viewport = page.viewport_size
        
        for viewport in viewports:
            await page.set_viewport_size(viewport)
            await page.wait_for_timeout(500)
            
            # Check for horizontal scrolling
            has_horizontal_scroll = await page.evaluate('''
                () => document.documentElement.scrollWidth > document.documentElement.clientWidth
            ''')
            
            if has_horizontal_scroll and viewport["width"] >= 320:
                violations.append(WCAGViolation(
                    id="responsive-scroll-issue",
                    impact="moderate",
                    tags=["responsive", "wcag2aa"],
                    description=f"Horizontal scrolling detected at {viewport['width']}px width",
                    help="Ensure content reflows without horizontal scrolling",
                    help_url="https://www.w3.org/WAI/WCAG21/Understanding/reflow.html",
                    nodes=[],
                    principle=WCAGPrinciple.PERCEIVABLE,
                    guideline="1.4",
                    success_criterion="1.4.10",
                    level=WCAGLevel.AA,
                    page_url=page.url,
                    timestamp=datetime.now().isoformat()
                ))
        
        # Restore original viewport
        if original_viewport:
            await page.set_viewport_size(original_viewport)
        
        return violations
    
    async def _check_dynamic_content_accessibility(self, page: Page) -> List[WCAGViolation]:
        """Check dynamic content accessibility (ARIA live regions, etc.)"""
        violations = []
        
        # Look for dynamic content areas that might need ARIA live regions
        dynamic_selectors = [
            '[data-component="notification-center"]',
            '[data-component="detection-status"]',
            '.alert',
            '.toast',
            '.status',
            '.loading'
        ]
        
        for selector in dynamic_selectors:
            elements = await page.query_selector_all(selector)
            
            for element in elements:
                has_live_region = await element.evaluate('''
                    el => el.hasAttribute('aria-live') || 
                          el.hasAttribute('aria-atomic') || 
                          el.hasAttribute('role') && ['alert', 'status', 'log'].includes(el.getAttribute('role'))
                ''')
                
                if not has_live_region:
                    violations.append(WCAGViolation(
                        id="dynamic-content-missing-aria",
                        impact="moderate",
                        tags=["aria", "wcag2aa"],
                        description="Dynamic content missing ARIA live region",
                        help="Use ARIA live regions for dynamic content updates",
                        help_url="https://www.w3.org/WAI/WCAG21/Understanding/status-messages.html",
                        nodes=[{"html": await element.outer_html()}],
                        principle=WCAGPrinciple.ROBUST,
                        guideline="4.1",
                        success_criterion="4.1.3",
                        level=WCAGLevel.AA,
                        page_url=page.url,
                        timestamp=datetime.now().isoformat()
                    ))
        
        return violations
    
    def _create_wcag_violation(self, axe_violation: Dict[str, Any], page_url: str) -> WCAGViolation:
        """Create WCAGViolation from axe-core violation"""
        # Map axe tags to WCAG principles
        principle = WCAGPrinciple.ROBUST  # Default
        for tag in axe_violation.get("tags", []):
            if "wcag1" in tag:
                principle = WCAGPrinciple.PERCEIVABLE
            elif "wcag2" in tag:
                principle = WCAGPrinciple.OPERABLE
            elif "wcag3" in tag:
                principle = WCAGPrinciple.UNDERSTANDABLE
            elif "wcag4" in tag:
                principle = WCAGPrinciple.ROBUST
        
        # Extract success criterion from tags
        success_criterion = "Unknown"
        guideline = "Unknown"
        for tag in axe_violation.get("tags", []):
            if tag.startswith("wcag") and len(tag) > 4:
                # Try to extract criterion like "wcag111" -> "1.1.1"
                try:
                    numbers = tag[4:]
                    if len(numbers) >= 3:
                        success_criterion = f"{numbers[0]}.{numbers[1]}.{numbers[2:]}"
                        guideline = f"{numbers[0]}.{numbers[1]}"
                except:
                    pass
        
        return WCAGViolation(
            id=axe_violation.get("id", "unknown"),
            impact=axe_violation.get("impact", "unknown"),
            tags=axe_violation.get("tags", []),
            description=axe_violation.get("description", ""),
            help=axe_violation.get("help", ""),
            help_url=axe_violation.get("helpUrl", ""),
            nodes=axe_violation.get("nodes", []),
            principle=principle,
            guideline=guideline,
            success_criterion=success_criterion,
            level=WCAGLevel.AA,  # Assuming AA level for now
            page_url=page_url,
            timestamp=datetime.now().isoformat()
        )
    
    async def _get_browser_info(self, page: Page) -> Dict[str, Any]:
        """Get browser information"""
        return await page.evaluate('''
            () => ({
                name: navigator.userAgent.includes('Chrome') ? 'Chrome' : 
                      navigator.userAgent.includes('Firefox') ? 'Firefox' : 
                      navigator.userAgent.includes('Safari') ? 'Safari' : 'Unknown',
                user_agent: navigator.userAgent,
                viewport: {
                    width: window.innerWidth,
                    height: window.innerHeight
                }
            })
        ''')
    
    def _generate_summary(self, results: List[AccessibilityTestResult]) -> Dict[str, Any]:
        """Generate validation summary"""
        total_violations = sum(r.total_violations for r in results)
        total_critical = sum(r.critical_violations for r in results)
        total_serious = sum(r.serious_violations for r in results)
        total_moderate = sum(r.moderate_violations for r in results)
        total_minor = sum(r.minor_violations for r in results)
        
        return {
            "total_pages_tested": len(results),
            "total_violations": total_violations,
            "critical_violations": total_critical,
            "serious_violations": total_serious,
            "moderate_violations": total_moderate,
            "minor_violations": total_minor,
            "pages_with_violations": len([r for r in results if r.total_violations > 0]),
            "pages_without_violations": len([r for r in results if r.total_violations == 0])
        }
    
    def _calculate_compliance_score(self, results: List[AccessibilityTestResult]) -> float:
        """Calculate WCAG compliance score"""
        if not results:
            return 0.0
        
        total_checks = len(results) * len(self.wcag_aa_criteria)
        total_violations = sum(r.total_violations for r in results)
        
        # Weight violations by impact
        weighted_violations = 0
        for result in results:
            weighted_violations += (
                result.critical_violations * 4 +
                result.serious_violations * 3 +
                result.moderate_violations * 2 +
                result.minor_violations * 1
            )
        
        # Calculate score (0-100)
        max_possible_weight = total_checks * 4  # All critical
        compliance_score = max(0, 100 - (weighted_violations / max_possible_weight * 100))
        
        return min(100, compliance_score)
    
    def _generate_recommendations(self, results: List[AccessibilityTestResult]) -> List[Dict[str, Any]]:
        """Generate accessibility improvement recommendations"""
        recommendations = []
        
        # Analyze violation patterns
        violation_counts = {}
        for result in results:
            for violation in result.violations:
                key = f"{violation.principle.value}_{violation.success_criterion}"
                violation_counts[key] = violation_counts.get(key, 0) + 1
        
        # Sort by frequency
        sorted_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)
        
        for violation_key, count in sorted_violations[:5]:  # Top 5 issues
            principle, criterion = violation_key.split("_", 1)
            criterion_name = self.wcag_aa_criteria.get(criterion, "Unknown")
            
            recommendations.append({
                "priority": "high" if count >= 3 else "medium" if count >= 2 else "low",
                "principle": principle,
                "success_criterion": criterion,
                "criterion_name": criterion_name,
                "affected_pages": count,
                "recommendation": f"Address {criterion_name} violations across {count} pages",
                "resources": [
                    f"https://www.w3.org/WAI/WCAG21/Understanding/{criterion.replace('.', '-')}.html"
                ]
            })
        
        return recommendations
    
    async def _save_comprehensive_report(self, validation_results: Dict[str, Any]):
        """Save comprehensive accessibility validation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_report_path = self.reports_dir / f"wcag_validation_report_{timestamp}.json"
        with open(json_report_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        # Save HTML report
        html_report_path = self.reports_dir / f"wcag_validation_report_{timestamp}.html"
        html_content = self._generate_html_report(validation_results)
        with open(html_report_path, 'w') as f:
            f.write(html_content)
        
        print(f"📊 Reports saved:")
        print(f"  JSON: {json_report_path}")
        print(f"  HTML: {html_report_path}")
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML accessibility report"""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Pynomaly WCAG 2.1 AA Validation Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 40px; }}
                .score {{ font-size: 3em; font-weight: bold; color: {'#27ae60' if results['compliance_score'] >= 80 else '#f39c12' if results['compliance_score'] >= 60 else '#e74c3c'}; }}
                .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
                .metric {{ background: #ecf0f1; padding: 20px; border-radius: 6px; text-align: center; }}
                .critical {{ color: #e74c3c; }}
                .serious {{ color: #f39c12; }}
                .moderate {{ color: #3498db; }}
                .minor {{ color: #95a5a6; }}
                .violations {{ margin: 20px 0; }}
                .violation {{ background: #fff; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; padding: 15px; }}
                .recommendations {{ background: #fff8dc; border-left: 4px solid #f39c12; padding: 20px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 WCAG 2.1 AA Validation Report</h1>
                    <div class="score">{results['compliance_score']:.1f}%</div>
                    <p>Compliance Score</p>
                    <p><small>Generated: {results['timestamp']}</small></p>
                </div>
                
                <div class="summary">
                    <div class="metric">
                        <div style="font-size: 2em; font-weight: bold;">{results['summary']['total_pages_tested']}</div>
                        <div>Pages Tested</div>
                    </div>
                    <div class="metric">
                        <div style="font-size: 2em; font-weight: bold;">{results['summary']['total_violations']}</div>
                        <div>Total Violations</div>
                    </div>
                    <div class="metric">
                        <div style="font-size: 2em; font-weight: bold;" class="critical">{results['summary']['critical_violations']}</div>
                        <div>Critical</div>
                    </div>
                    <div class="metric">
                        <div style="font-size: 2em; font-weight: bold;" class="serious">{results['summary']['serious_violations']}</div>
                        <div>Serious</div>
                    </div>
                </div>
                
                <div class="recommendations">
                    <h3>💡 Top Recommendations</h3>
                    <ul>
                        {''.join([f"<li><strong>{rec['criterion_name']}</strong> - {rec['recommendation']}</li>" for rec in results['recommendations'][:5]])}
                    </ul>
                </div>
                
                <h3>📄 Page Results</h3>
                {''.join([f'''
                <div class="violation">
                    <h4>{page['page_url']}</h4>
                    <p>Violations: {page['total_violations']} 
                       (Critical: {page['critical_violations']}, 
                        Serious: {page['serious_violations']}, 
                        Moderate: {page['moderate_violations']}, 
                        Minor: {page['minor_violations']})</p>
                </div>
                ''' for page in results['page_results']])}
            </div>
        </body>
        </html>
        """


# Pytest fixtures and test functions
@pytest.fixture(scope="session")
def wcag_validator():
    """Create WCAG validation framework instance"""
    return WCAGValidationFramework(target_level=WCAGLevel.AA)


@pytest.mark.accessibility
@pytest.mark.wcag
class TestWCAGCompliance:
    """WCAG 2.1 AA Compliance Test Suite"""
    
    @pytest.mark.critical
    async def test_comprehensive_wcag_validation(self, page: Page, wcag_validator: WCAGValidationFramework):
        """Run comprehensive WCAG 2.1 AA validation"""
        results = await wcag_validator.run_comprehensive_validation(page)
        
        # Assert compliance score
        assert results["compliance_score"] >= 80.0, f"WCAG compliance score {results['compliance_score']:.1f}% is below 80%"
        
        # Assert no critical violations
        assert results["summary"]["critical_violations"] == 0, f"Found {results['summary']['critical_violations']} critical accessibility violations"
        
        # Assert serious violations are minimal
        assert results["summary"]["serious_violations"] <= 5, f"Found {results['summary']['serious_violations']} serious accessibility violations"
    
    @pytest.mark.smoke
    async def test_main_pages_accessibility(self, page: Page, wcag_validator: WCAGValidationFramework):
        """Test accessibility of main application pages"""
        main_pages = ["/", "/dashboard", "/datasets"]
        
        for page_path in main_pages:
            await page.goto(f"http://localhost:8000{page_path}")
            result = await wcag_validator.run_axe_scan(page, page_path.replace("/", "home") if page_path == "/" else page_path[1:])
            
            # Each main page should have minimal violations
            assert result.critical_violations == 0, f"Critical violations found on {page_path}"
            assert result.serious_violations <= 2, f"Too many serious violations on {page_path}"


if __name__ == "__main__":
    # Run standalone validation
    import asyncio
    from playwright.async_api import async_playwright
    
    async def main():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            validator = WCAGValidationFramework()
            results = await validator.run_comprehensive_validation(page)
            
            print(f"\n🎯 WCAG 2.1 AA Validation Complete")
            print(f"Compliance Score: {results['compliance_score']:.1f}%")
            print(f"Critical Issues: {results['summary']['critical_violations']}")
            print(f"Total Violations: {results['summary']['total_violations']}")
            
            await browser.close()
    
    asyncio.run(main())