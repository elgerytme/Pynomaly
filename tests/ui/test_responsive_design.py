"""Responsive Design Testing Suite."""

from typing import Any, Dict, List, Tuple

import pytest
from playwright.sync_api import Page


class TestResponsiveDesign:
    """Comprehensive responsive design testing suite."""
    
    # Standard viewport sizes for testing
    VIEWPORTS = [
        {"width": 320, "height": 568, "name": "mobile_small", "device": "iPhone SE"},
        {"width": 375, "height": 667, "name": "mobile_medium", "device": "iPhone 8"},
        {"width": 414, "height": 896, "name": "mobile_large", "device": "iPhone XR"},
        {"width": 768, "height": 1024, "name": "tablet_portrait", "device": "iPad"},
        {"width": 1024, "height": 768, "name": "tablet_landscape", "device": "iPad Landscape"},
        {"width": 1280, "height": 720, "name": "desktop_small", "device": "Small Desktop"},
        {"width": 1920, "height": 1080, "name": "desktop_large", "device": "Large Desktop"},
    ]
    
    def test_viewport_responsiveness(self, page: Page):
        """Test layout responsiveness across different viewport sizes."""
        results = {}
        
        for viewport in self.VIEWPORTS:
            page.set_viewport_size({"width": viewport["width"], "height": viewport["height"]})
            page.goto("http://pynomaly-app:8000/web/")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1000)  # Wait for responsive adjustments
            
            # Check key layout elements
            layout_check = self._check_layout_elements(page, viewport)
            results[viewport["name"]] = layout_check
            
            # Take screenshot for visual verification
            page.screenshot(
                path=f"screenshots/responsive_{viewport['name']}.png",
                full_page=True
            )
            
        # Verify critical elements are present across all viewports
        for viewport_name, result in results.items():
            assert result["navigation_visible"], \
                f"Navigation should be visible on {viewport_name}"
            assert result["main_content_visible"], \
                f"Main content should be visible on {viewport_name}"
                
    def _check_layout_elements(self, page: Page, viewport: Dict) -> Dict[str, Any]:
        """Check key layout elements for a specific viewport."""
        return {
            "viewport": viewport,
            "navigation_visible": page.locator("nav").is_visible(),
            "main_content_visible": page.locator("main").is_visible(),
            "footer_visible": page.locator("footer").is_visible(),
            "mobile_menu_visible": self._is_mobile_menu_visible(page, viewport),
            "desktop_menu_visible": self._is_desktop_menu_visible(page, viewport),
            "content_readable": self._check_content_readability(page),
            "no_horizontal_scroll": self._check_horizontal_scroll(page)
        }
        
    def _is_mobile_menu_visible(self, page: Page, viewport: Dict) -> bool:
        """Check if mobile menu is appropriately visible."""
        mobile_button = page.locator("nav .sm\\:hidden button, .mobile-menu-button")
        
        if viewport["width"] < 768:  # Mobile breakpoint
            return mobile_button.is_visible()
        else:
            return True  # Desktop doesn't need mobile menu visible
            
    def _is_desktop_menu_visible(self, page: Page, viewport: Dict) -> bool:
        """Check if desktop menu is appropriately visible."""
        desktop_nav = page.locator("nav .hidden.sm\\:flex, .desktop-menu")
        
        if viewport["width"] >= 768:  # Desktop breakpoint
            return desktop_nav.is_visible()
        else:
            return True  # Mobile doesn't need desktop menu visible
            
    def _check_content_readability(self, page: Page) -> bool:
        """Check if content is readable (not overlapping, proper sizing)."""
        # Check for text elements that might be too small
        small_text = page.evaluate("""
            () => {
                const elements = document.querySelectorAll('p, span, div, a, button');
                let tooSmall = 0;
                
                elements.forEach(el => {
                    const styles = window.getComputedStyle(el);
                    const fontSize = parseFloat(styles.fontSize);
                    if (fontSize < 14) tooSmall++;
                });
                
                return tooSmall < elements.length * 0.1; // Less than 10% too small
            }
        """)
        
        return small_text
        
    def _check_horizontal_scroll(self, page: Page) -> bool:
        """Check if there's unwanted horizontal scroll."""
        has_horizontal_scroll = page.evaluate("""
            () => {
                return document.documentElement.scrollWidth <= window.innerWidth;
            }
        """)
        
        return has_horizontal_scroll
        
    def test_mobile_navigation_functionality(self, page: Page):
        """Test mobile navigation functionality."""
        # Test on mobile viewport
        page.set_viewport_size({"width": 375, "height": 667})
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")
        
        # Check mobile menu button
        mobile_button = page.locator("nav button").filter(has_text="")
        if mobile_button.count() == 0:
            # Look for hamburger menu button with different selector
            mobile_button = page.locator("nav .sm\\:hidden button")
            
        if mobile_button.count() > 0:
            # Test menu toggle
            mobile_button.click()
            page.wait_for_timeout(500)
            
            # Check if mobile menu items are accessible
            mobile_nav_items = page.locator("nav .sm\\:hidden a, nav .mobile-menu a")
            
            # Should have navigation items
            assert mobile_nav_items.count() > 0, "Mobile menu should have navigation items"
            
            # Test navigation
            if mobile_nav_items.count() > 0:
                first_nav_item = mobile_nav_items.first
                href = first_nav_item.get_attribute("href")
                
                if href and href != "#":
                    first_nav_item.click()
                    page.wait_for_load_state("networkidle")
                    
                    # Should navigate successfully
                    assert href in page.url, "Mobile navigation should work"
                    
    def test_touch_targets(self, page: Page):
        """Test touch target sizes for mobile devices."""
        page.set_viewport_size({"width": 375, "height": 667})
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")
        
        # Check interactive elements for appropriate touch target size
        interactive_elements = page.locator("button, a, input, select")
        
        touch_target_results = []
        
        for i in range(interactive_elements.count()):
            element = interactive_elements.nth(i)
            
            # Get element dimensions
            box = element.bounding_box()
            if box:
                touch_target_results.append({
                    "element": element.get_attribute("tagName") or "unknown",
                    "width": box["width"],
                    "height": box["height"],
                    "min_dimension": min(box["width"], box["height"]),
                    "adequate_size": min(box["width"], box["height"]) >= 44  # iOS/Android guideline
                })
                
        # Most interactive elements should have adequate touch target size
        adequate_targets = [r for r in touch_target_results if r["adequate_size"]]
        total_targets = len(touch_target_results)
        
        if total_targets > 0:
            adequate_percentage = len(adequate_targets) / total_targets
            assert adequate_percentage >= 0.8, \
                f"At least 80% of touch targets should be adequate size. Got {adequate_percentage:.1%}"
                
    def test_text_scaling(self, page: Page):
        """Test text scaling and readability."""
        viewports_to_test = [
            {"width": 320, "height": 568, "name": "small_mobile"},
            {"width": 768, "height": 1024, "name": "tablet"},
            {"width": 1920, "height": 1080, "name": "desktop"}
        ]
        
        for viewport in viewports_to_test:
            page.set_viewport_size({"width": viewport["width"], "height": viewport["height"]})
            page.goto("http://pynomaly-app:8000/web/")
            page.wait_for_load_state("networkidle")
            
            # Check font sizes
            font_size_check = page.evaluate("""
                () => {
                    const textElements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, span, div, a, button');
                    const fontSizes = [];
                    
                    textElements.forEach(el => {
                        const styles = window.getComputedStyle(el);
                        const fontSize = parseFloat(styles.fontSize);
                        if (fontSize > 0) {
                            fontSizes.push(fontSize);
                        }
                    });
                    
                    return {
                        minFontSize: Math.min(...fontSizes),
                        maxFontSize: Math.max(...fontSizes),
                        avgFontSize: fontSizes.reduce((a, b) => a + b, 0) / fontSizes.length
                    };
                }
            """)
            
            # Minimum font size should be readable
            assert font_size_check["minFontSize"] >= 12, \
                f"Minimum font size should be at least 12px on {viewport['name']}"
                
            # Should have reasonable font size range
            assert font_size_check["maxFontSize"] <= 72, \
                f"Maximum font size should be reasonable on {viewport['name']}"
                
    def test_image_responsiveness(self, page: Page):
        """Test image responsiveness."""
        viewports_to_test = [
            {"width": 375, "height": 667},
            {"width": 768, "height": 1024},
            {"width": 1920, "height": 1080}
        ]
        
        for viewport in viewports_to_test:
            page.set_viewport_size({"width": viewport["width"], "height": viewport["height"]})
            page.goto("http://pynomaly-app:8000/web/")
            page.wait_for_load_state("networkidle")
            
            # Check images
            images = page.locator("img")
            
            for i in range(images.count()):
                img = images.nth(i)
                
                # Get image dimensions
                img_info = page.evaluate("""
                    (img) => {
                        const rect = img.getBoundingClientRect();
                        return {
                            naturalWidth: img.naturalWidth,
                            naturalHeight: img.naturalHeight,
                            displayWidth: rect.width,
                            displayHeight: rect.height,
                            maxWidth: window.getComputedStyle(img).maxWidth
                        };
                    }
                """, img)
                
                # Image should not overflow viewport
                viewport_width = viewport["width"]
                assert img_info["displayWidth"] <= viewport_width + 1, \
                    f"Image {i} should not overflow viewport on {viewport_width}px width"
                    
    def test_layout_grid_responsiveness(self, page: Page):
        """Test CSS Grid and Flexbox responsiveness."""
        page.goto("http://pynomaly-app:8000/web/")
        
        viewports_to_test = [
            {"width": 375, "height": 667, "expected_columns": 1},
            {"width": 768, "height": 1024, "expected_columns": 2},
            {"width": 1920, "height": 1080, "expected_columns": 3}
        ]
        
        for viewport in viewports_to_test:
            page.set_viewport_size({"width": viewport["width"], "height": viewport["height"]})
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(500)
            
            # Check grid layouts (like stats cards)
            grid_containers = page.locator(".grid, [class*='grid-cols']")
            
            for i in range(grid_containers.count()):
                container = grid_containers.nth(i)
                
                # Get grid information
                grid_info = page.evaluate("""
                    (container) => {
                        const styles = window.getComputedStyle(container);
                        const children = container.children;
                        const rect = container.getBoundingClientRect();
                        
                        // Count visible children in first row
                        let columnsInFirstRow = 0;
                        if (children.length > 0) {
                            const firstChildTop = children[0].getBoundingClientRect().top;
                            for (let child of children) {
                                if (Math.abs(child.getBoundingClientRect().top - firstChildTop) < 5) {
                                    columnsInFirstRow++;
                                } else {
                                    break;
                                }
                            }
                        }
                        
                        return {
                            display: styles.display,
                            gridTemplateColumns: styles.gridTemplateColumns,
                            columnsInFirstRow: columnsInFirstRow,
                            totalChildren: children.length
                        };
                    }
                """, container)
                
                # Check if layout adapts appropriately
                if grid_info["display"] === "grid" and grid_info["totalChildren"] > 0:
                    columns_in_row = grid_info["columnsInFirstRow"]
                    
                    # Should adapt to viewport
                    if viewport["width"] <= 640:  # Mobile
                        assert columns_in_row <= 2, \
                            f"Mobile should have at most 2 columns, got {columns_in_row}"
                    elif viewport["width"] <= 1024:  # Tablet
                        assert columns_in_row <= 3, \
                            f"Tablet should have at most 3 columns, got {columns_in_row}"
                            
    def test_form_responsiveness(self, page: Page):
        """Test form responsiveness."""
        page.goto("http://pynomaly-app:8000/web/detectors")
        
        viewports_to_test = [
            {"width": 375, "height": 667, "name": "mobile"},
            {"width": 768, "height": 1024, "name": "tablet"},
            {"width": 1920, "height": 1080, "name": "desktop"}
        ]
        
        for viewport in viewports_to_test:
            page.set_viewport_size({"width": viewport["width"], "height": viewport["height"]})
            page.wait_for_load_state("networkidle")
            
            # Check form elements
            forms = page.locator("form")
            
            if forms.count() > 0:
                form = forms.first
                
                # Check form width
                form_info = page.evaluate("""
                    (form) => {
                        const rect = form.getBoundingClientRect();
                        const styles = window.getComputedStyle(form);
                        return {
                            width: rect.width,
                            maxWidth: styles.maxWidth,
                            padding: styles.padding
                        };
                    }
                """, form)
                
                # Form should not be wider than viewport
                assert form_info["width"] <= viewport["width"], \
                    f"Form should fit in viewport on {viewport['name']}"
                    
                # Check input elements
                inputs = form.locator("input, select, textarea")
                
                for i in range(inputs.count()):
                    input_elem = inputs.nth(i)
                    
                    input_info = page.evaluate("""
                        (input) => {
                            const rect = input.getBoundingClientRect();
                            return {
                                width: rect.width,
                                height: rect.height
                            };
                        }
                    """, input_elem)
                    
                    # Input should be appropriately sized
                    if viewport["width"] <= 640:  # Mobile
                        assert input_info["height"] >= 40, \
                            f"Input {i} should be at least 40px tall on mobile"
                            
    def test_content_reflow(self, page: Page):
        """Test content reflow and overflow handling."""
        page.goto("http://pynomaly-app:8000/web/")
        
        # Test with very narrow viewport
        page.set_viewport_size({"width": 320, "height": 568})
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1000)
        
        # Check for horizontal overflow
        overflow_check = page.evaluate("""
            () => {
                const body = document.body;
                const html = document.documentElement;
                
                const scrollWidth = Math.max(
                    body.scrollWidth, body.offsetWidth,
                    html.clientWidth, html.scrollWidth, html.offsetWidth
                );
                
                const clientWidth = html.clientWidth;
                
                // Find elements that might cause overflow
                const wideElements = [];
                const allElements = document.querySelectorAll('*');
                
                allElements.forEach((el, index) => {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > clientWidth + 5) { // 5px tolerance
                        wideElements.push({
                            tagName: el.tagName,
                            className: el.className,
                            width: rect.width,
                            index: index
                        });
                    }
                });
                
                return {
                    hasHorizontalOverflow: scrollWidth > clientWidth,
                    scrollWidth: scrollWidth,
                    clientWidth: clientWidth,
                    wideElements: wideElements.slice(0, 5) // First 5 problematic elements
                };
            }
        """)
        
        assert not overflow_check["hasHorizontalOverflow"], \
            f"Page should not have horizontal overflow. Wide elements: {overflow_check['wideElements']}"
            
    def test_breakpoint_consistency(self, page: Page):
        """Test consistency at CSS breakpoints."""
        # Common Tailwind CSS breakpoints
        breakpoints = [
            640,   # sm
            768,   # md
            1024,  # lg
            1280,  # xl
            1536   # 2xl
        ]
        
        for breakpoint in breakpoints:
            # Test just below and just above breakpoint
            for width in [breakpoint - 1, breakpoint + 1]:
                page.set_viewport_size({"width": width, "height": 800})
                page.goto("http://pynomaly-app:8000/web/")
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(500)
                
                # Check that layout is stable
                layout_stable = page.evaluate("""
                    () => {
                        // Check for any layout shift indicators
                        const main = document.querySelector('main');
                        const nav = document.querySelector('nav');
                        
                        return {
                            mainVisible: main && main.offsetHeight > 0,
                            navVisible: nav && nav.offsetHeight > 0,
                            bodyWidth: document.body.offsetWidth,
                            noErrors: !document.querySelector('.error, [class*="error"]')
                        };
                    }
                """)
                
                assert layout_stable["mainVisible"], \
                    f"Main content should be visible at {width}px"
                assert layout_stable["navVisible"], \
                    f"Navigation should be visible at {width}px"
                    
    def generate_responsive_report(self, page: Page) -> Dict[str, Any]:
        """Generate comprehensive responsive design report."""
        report = {
            "viewports_tested": [],
            "breakpoint_issues": [],
            "touch_target_issues": [],
            "overflow_issues": []
        }
        
        for viewport in self.VIEWPORTS:
            page.set_viewport_size({"width": viewport["width"], "height": viewport["height"]})
            page.goto("http://pynomaly-app:8000/web/")
            page.wait_for_load_state("networkidle")
            
            layout_check = self._check_layout_elements(page, viewport)
            
            report["viewports_tested"].append({
                "viewport": viewport,
                "layout_check": layout_check,
                "passed": all([
                    layout_check["navigation_visible"],
                    layout_check["main_content_visible"],
                    layout_check["content_readable"],
                    layout_check["no_horizontal_scroll"]
                ])
            })
            
        return report