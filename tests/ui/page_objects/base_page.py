"""Base page object for UI tests."""

from typing import Optional, List, Dict, Any
from playwright.sync_api import Page, Locator, expect
from pathlib import Path


class BasePage:
    """Base page object with common functionality."""
    
    def __init__(self, page: Page, base_url: str):
        self.page = page
        self.base_url = base_url
        self.timeout = 30000  # 30 seconds
        
    def navigate_to(self, path: str = "") -> None:
        """Navigate to a specific path."""
        url = f"{self.base_url}/web{path}"
        self.page.goto(url, wait_until="networkidle")
        
    def wait_for_load(self) -> None:
        """Wait for page to fully load."""
        self.page.wait_for_load_state("networkidle")
        
    def take_screenshot(self, name: str, full_page: bool = True) -> Path:
        """Take a screenshot."""
        screenshot_path = Path("screenshots") / f"{name}.png"
        self.page.screenshot(path=str(screenshot_path), full_page=full_page)
        return screenshot_path
        
    def get_element(self, selector: str) -> Locator:
        """Get element by selector."""
        return self.page.locator(selector)
        
    def wait_for_element(self, selector: str, timeout: Optional[int] = None) -> Locator:
        """Wait for element and return it."""
        timeout = timeout or self.timeout
        element = self.get_element(selector)
        element.wait_for(timeout=timeout)
        return element
        
    def click_element(self, selector: str) -> None:
        """Click an element."""
        element = self.wait_for_element(selector)
        element.click()
        
    def fill_input(self, selector: str, value: str) -> None:
        """Fill an input field."""
        element = self.wait_for_element(selector)
        element.fill(value)
        
    def select_option(self, selector: str, value: str) -> None:
        """Select option from dropdown."""
        element = self.wait_for_element(selector)
        element.select_option(value)
        
    def get_text(self, selector: str) -> str:
        """Get text content of element."""
        element = self.wait_for_element(selector)
        return element.text_content() or ""
        
    def get_attribute(self, selector: str, attribute: str) -> Optional[str]:
        """Get attribute value of element."""
        element = self.wait_for_element(selector)
        return element.get_attribute(attribute)
        
    def is_visible(self, selector: str) -> bool:
        """Check if element is visible."""
        try:
            element = self.get_element(selector)
            return element.is_visible()
        except:
            return False
            
    def is_enabled(self, selector: str) -> bool:
        """Check if element is enabled."""
        try:
            element = self.get_element(selector)
            return element.is_enabled()
        except:
            return False
            
    def wait_for_url(self, url_pattern: str, timeout: Optional[int] = None) -> None:
        """Wait for URL to match pattern."""
        timeout = timeout or self.timeout
        self.page.wait_for_url(url_pattern, timeout=timeout)
        
    def wait_for_response(self, url_pattern: str, timeout: Optional[int] = None):
        """Wait for response matching URL pattern."""
        timeout = timeout or self.timeout
        with self.page.expect_response(url_pattern, timeout=timeout) as response_info:
            pass
        return response_info.value
        
    def get_navigation_links(self) -> List[Dict[str, str]]:
        """Get all navigation links."""
        nav_links = []
        desktop_nav = self.page.locator("nav .hidden.sm\\:flex a")
        
        for i in range(desktop_nav.count()):
            link = desktop_nav.nth(i)
            nav_links.append({
                "text": link.text_content() or "",
                "href": link.get_attribute("href") or "",
                "is_active": "border-primary" in (link.get_attribute("class") or "")
            })
            
        return nav_links
        
    def check_responsive_navigation(self) -> Dict[str, bool]:
        """Check if responsive navigation works."""
        # Check mobile menu button
        mobile_button = self.page.locator("button[\\@click=\"mobileMenuOpen = !mobileMenuOpen\"]")
        mobile_menu = self.page.locator("[x-show=\"mobileMenuOpen\"]")
        
        results = {
            "mobile_button_visible": mobile_button.is_visible(),
            "mobile_menu_hidden": not mobile_menu.is_visible(),
        }
        
        # Test mobile menu toggle
        if results["mobile_button_visible"]:
            mobile_button.click()
            self.page.wait_for_timeout(500)  # Wait for animation
            results["mobile_menu_toggles"] = mobile_menu.is_visible()
        else:
            results["mobile_menu_toggles"] = False
            
        return results
        
    def check_accessibility_basics(self) -> Dict[str, Any]:
        """Check basic accessibility features."""
        results = {
            "has_lang_attribute": bool(self.page.locator("html[lang]").count()),
            "has_viewport_meta": bool(self.page.locator("meta[name=\"viewport\"]").count()),
            "has_title": bool(self.page.title()),
            "images_with_alt": [],
            "buttons_with_labels": [],
            "form_inputs_with_labels": []
        }
        
        # Check images
        images = self.page.locator("img")
        for i in range(images.count()):
            img = images.nth(i)
            alt_text = img.get_attribute("alt")
            results["images_with_alt"].append({
                "src": img.get_attribute("src"),
                "has_alt": bool(alt_text),
                "alt_text": alt_text
            })
            
        # Check buttons
        buttons = self.page.locator("button")
        for i in range(buttons.count()):
            btn = buttons.nth(i)
            text = btn.text_content()
            aria_label = btn.get_attribute("aria-label")
            results["buttons_with_labels"].append({
                "has_text_or_label": bool(text or aria_label),
                "text": text,
                "aria_label": aria_label
            })
            
        # Check form inputs
        inputs = self.page.locator("input, select, textarea")
        for i in range(inputs.count()):
            input_elem = inputs.nth(i)
            input_id = input_elem.get_attribute("id")
            name = input_elem.get_attribute("name")
            placeholder = input_elem.get_attribute("placeholder")
            
            # Check for associated label
            has_label = False
            if input_id:
                label = self.page.locator(f"label[for=\"{input_id}\"]")
                has_label = label.count() > 0
                
            results["form_inputs_with_labels"].append({
                "id": input_id,
                "name": name,
                "placeholder": placeholder,
                "has_label": has_label
            })
            
        return results
        
    def measure_performance(self) -> Dict[str, Any]:
        """Measure basic performance metrics."""
        # Get performance metrics using JavaScript
        metrics = self.page.evaluate("""
            () => {
                const navigation = performance.getEntriesByType('navigation')[0];
                const paint = performance.getEntriesByType('paint');
                
                const result = {
                    dom_content_loaded: navigation ? navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart : 0,
                    load_complete: navigation ? navigation.loadEventEnd - navigation.loadEventStart : 0,
                    first_paint: 0,
                    first_contentful_paint: 0
                };
                
                paint.forEach(entry => {
                    if (entry.name === 'first-paint') {
                        result.first_paint = entry.startTime;
                    } else if (entry.name === 'first-contentful-paint') {
                        result.first_contentful_paint = entry.startTime;
                    }
                });
                
                return result;
            }
        """)
        
        return metrics
        
    def check_console_errors(self) -> List[str]:
        """Get console errors from the page."""
        errors = []
        
        def handle_console(msg):
            if msg.type == "error":
                errors.append(msg.text)
                
        self.page.on("console", handle_console)
        return errors
        
    def validate_html_structure(self) -> Dict[str, bool]:
        """Validate basic HTML structure."""
        return {
            "has_doctype": self.page.evaluate("() => document.doctype !== null"),
            "has_html_element": bool(self.page.locator("html").count()),
            "has_head_element": bool(self.page.locator("head").count()),
            "has_body_element": bool(self.page.locator("body").count()),
            "has_main_element": bool(self.page.locator("main").count()),
            "has_nav_element": bool(self.page.locator("nav").count()),
            "has_footer_element": bool(self.page.locator("footer").count())
        }