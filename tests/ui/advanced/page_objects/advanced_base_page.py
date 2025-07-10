"""
Advanced Base Page Object for Pynomaly UI Testing

Provides comprehensive page object functionality including:
- Enhanced wait strategies
- Performance monitoring
- Accessibility helpers
- Error handling and recovery
- Advanced screenshot capabilities
"""

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from playwright.async_api import Page, Locator, expect


class AdvancedBasePage:
    """
    Advanced base page object with comprehensive testing capabilities
    """

    def __init__(self, page: Page, base_url: str = "http://localhost:8000"):
        self.page = page
        self.base_url = base_url
        self.load_start_time: Optional[float] = None
        self.interactions_log: List[Dict[str, Any]] = []
        
        # Common selectors
        self.selectors = {
            "navigation": "nav, .navigation, [role='navigation']",
            "main_content": "main, .main-content, [role='main']",
            "footer": "footer, .footer",
            "loading_spinner": ".spinner, .loading, [aria-label*='loading']",
            "error_message": ".error, .alert-error, [role='alert']",
            "success_message": ".success, .alert-success",
            "modal": ".modal, [role='dialog']",
            "tooltip": ".tooltip, [role='tooltip']"
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            "page_load_time": 3000,  # 3 seconds
            "interaction_response": 1000,  # 1 second
            "animation_duration": 500,  # 500ms
        }

    async def navigate_to(self, path: str = "/", wait_for_load: bool = True) -> None:
        """
        Navigate to a specific path with performance monitoring
        
        Args:
            path: URL path to navigate to
            wait_for_load: Whether to wait for complete page load
        """
        self.load_start_time = time.time()
        full_url = f"{self.base_url}{path}"
        
        await self.page.goto(full_url)
        
        if wait_for_load:
            await self.wait_for_page_load()
        
        load_time = (time.time() - self.load_start_time) * 1000
        
        self._log_interaction({
            "action": "navigate",
            "url": full_url,
            "load_time_ms": load_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    async def wait_for_page_load(self, timeout: int = 30000) -> None:
        """
        Wait for complete page load with multiple strategies
        
        Args:
            timeout: Maximum wait time in milliseconds
        """
        # Wait for network to be idle
        await self.page.wait_for_load_state("networkidle", timeout=timeout)
        
        # Wait for any loading spinners to disappear
        await self.wait_for_element_hidden(self.selectors["loading_spinner"], timeout=5000)
        
        # Wait for main content to be visible
        try:
            await self.wait_for_element_visible(self.selectors["main_content"], timeout=5000)
        except:
            pass  # Main content selector might not exist on all pages

    async def wait_for_element_visible(self, selector: str, timeout: int = 10000) -> Locator:
        """
        Wait for element to be visible with enhanced error handling
        
        Args:
            selector: CSS selector or text
            timeout: Maximum wait time in milliseconds
            
        Returns:
            Locator object for the element
            
        Raises:
            TimeoutError: If element doesn't become visible within timeout
        """
        try:
            locator = self.page.locator(selector)
            await expect(locator).to_be_visible(timeout=timeout)
            return locator
        except Exception as e:
            await self._capture_debug_screenshot(f"element_not_visible_{selector.replace(' ', '_')}")
            raise TimeoutError(f"Element '{selector}' not visible within {timeout}ms: {e}")

    async def wait_for_element_hidden(self, selector: str, timeout: int = 10000) -> None:
        """
        Wait for element to be hidden or not exist
        
        Args:
            selector: CSS selector
            timeout: Maximum wait time in milliseconds
        """
        try:
            locator = self.page.locator(selector)
            await expect(locator).to_be_hidden(timeout=timeout)
        except:
            # Element might not exist, which is also considered "hidden"
            pass

    async def click_element(self, selector: str, timeout: int = 10000) -> None:
        """
        Click element with performance monitoring and error handling
        
        Args:
            selector: CSS selector
            timeout: Maximum wait time in milliseconds
        """
        start_time = time.time()
        
        try:
            locator = await self.wait_for_element_visible(selector, timeout)
            await locator.click()
            
            # Wait for any animations or state changes
            await self.page.wait_for_timeout(100)
            
            interaction_time = (time.time() - start_time) * 1000
            
            self._log_interaction({
                "action": "click",
                "selector": selector,
                "response_time_ms": interaction_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            if interaction_time > self.performance_thresholds["interaction_response"]:
                print(f"⚠️  Slow interaction: {selector} took {interaction_time:.2f}ms")
                
        except Exception as e:
            await self._capture_debug_screenshot(f"click_failed_{selector.replace(' ', '_')}")
            raise Exception(f"Failed to click element '{selector}': {e}")

    async def fill_input(self, selector: str, value: str, timeout: int = 10000) -> None:
        """
        Fill input field with validation
        
        Args:
            selector: CSS selector for input
            value: Value to enter
            timeout: Maximum wait time in milliseconds
        """
        start_time = time.time()
        
        try:
            locator = await self.wait_for_element_visible(selector, timeout)
            await locator.fill(value)
            
            # Verify value was entered correctly
            await expect(locator).to_have_value(value)
            
            interaction_time = (time.time() - start_time) * 1000
            
            self._log_interaction({
                "action": "fill_input",
                "selector": selector,
                "value_length": len(value),
                "response_time_ms": interaction_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            await self._capture_debug_screenshot(f"fill_failed_{selector.replace(' ', '_')}")
            raise Exception(f"Failed to fill input '{selector}': {e}")

    async def select_option(self, selector: str, value: str, timeout: int = 10000) -> None:
        """
        Select option from dropdown
        
        Args:
            selector: CSS selector for select element
            value: Value to select
            timeout: Maximum wait time in milliseconds
        """
        try:
            locator = await self.wait_for_element_visible(selector, timeout)
            await locator.select_option(value)
            
            self._log_interaction({
                "action": "select_option",
                "selector": selector,
                "value": value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            await self._capture_debug_screenshot(f"select_failed_{selector.replace(' ', '_')}")
            raise Exception(f"Failed to select option '{value}' in '{selector}': {e}")

    async def hover_element(self, selector: str, timeout: int = 10000) -> None:
        """
        Hover over element
        
        Args:
            selector: CSS selector
            timeout: Maximum wait time in milliseconds
        """
        try:
            locator = await self.wait_for_element_visible(selector, timeout)
            await locator.hover()
            
            # Wait for hover effects
            await self.page.wait_for_timeout(300)
            
            self._log_interaction({
                "action": "hover",
                "selector": selector,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            raise Exception(f"Failed to hover over element '{selector}': {e}")

    async def get_element_text(self, selector: str, timeout: int = 10000) -> str:
        """
        Get text content of element
        
        Args:
            selector: CSS selector
            timeout: Maximum wait time in milliseconds
            
        Returns:
            Text content of the element
        """
        try:
            locator = await self.wait_for_element_visible(selector, timeout)
            return await locator.inner_text()
        except Exception as e:
            raise Exception(f"Failed to get text from element '{selector}': {e}")

    async def get_element_attribute(self, selector: str, attribute: str, timeout: int = 10000) -> Optional[str]:
        """
        Get attribute value of element
        
        Args:
            selector: CSS selector
            attribute: Attribute name
            timeout: Maximum wait time in milliseconds
            
        Returns:
            Attribute value or None if not found
        """
        try:
            locator = await self.wait_for_element_visible(selector, timeout)
            return await locator.get_attribute(attribute)
        except Exception as e:
            raise Exception(f"Failed to get attribute '{attribute}' from element '{selector}': {e}")

    async def is_element_visible(self, selector: str) -> bool:
        """
        Check if element is visible without waiting
        
        Args:
            selector: CSS selector
            
        Returns:
            True if element is visible, False otherwise
        """
        try:
            locator = self.page.locator(selector)
            return await locator.is_visible()
        except:
            return False

    async def is_element_enabled(self, selector: str) -> bool:
        """
        Check if element is enabled
        
        Args:
            selector: CSS selector
            
        Returns:
            True if element is enabled, False otherwise
        """
        try:
            locator = self.page.locator(selector)
            return await locator.is_enabled()
        except:
            return False

    async def wait_for_url_change(self, expected_url: str = None, timeout: int = 10000) -> str:
        """
        Wait for URL to change
        
        Args:
            expected_url: Expected URL pattern (optional)
            timeout: Maximum wait time in milliseconds
            
        Returns:
            Current URL after change
        """
        if expected_url:
            await self.page.wait_for_url(expected_url, timeout=timeout)
        else:
            # Wait for any URL change
            current_url = self.page.url
            await self.page.wait_for_function(
                f"() => window.location.href !== '{current_url}'",
                timeout=timeout
            )
        
        return self.page.url

    async def wait_for_ajax_requests(self, timeout: int = 10000) -> None:
        """
        Wait for all AJAX requests to complete
        
        Args:
            timeout: Maximum wait time in milliseconds
        """
        try:
            await self.page.wait_for_function(
                """
                () => {
                    // Check for jQuery AJAX
                    if (window.jQuery && jQuery.active > 0) return false;
                    
                    // Check for fetch requests (modern approach)
                    if (window.fetchPendingRequests && window.fetchPendingRequests > 0) return false;
                    
                    // Check for XMLHttpRequest
                    if (window.activeXHRRequests && window.activeXHRRequests > 0) return false;
                    
                    return true;
                }
                """,
                timeout=timeout
            )
        except:
            # If we can't detect AJAX, wait a reasonable amount of time
            await self.page.wait_for_timeout(1000)

    async def wait_for_htmx_requests(self, timeout: int = 10000) -> None:
        """
        Wait for HTMX requests to complete
        
        Args:
            timeout: Maximum wait time in milliseconds
        """
        try:
            await self.page.wait_for_function(
                """
                () => {
                    // Check if htmx is available and has no pending requests
                    if (window.htmx && window.htmx.config) {
                        return !document.querySelector('[hx-request]');
                    }
                    return true;
                }
                """,
                timeout=timeout
            )
        except:
            await self.page.wait_for_timeout(500)

    async def capture_screenshot(self, name: str, full_page: bool = True) -> Path:
        """
        Capture screenshot with metadata
        
        Args:
            name: Screenshot name
            full_page: Whether to capture full page
            
        Returns:
            Path to saved screenshot
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        screenshot_name = f"{name}_{timestamp}.png"
        screenshot_path = Path("test_artifacts/screenshots") / screenshot_name
        
        # Ensure directory exists
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        
        await self.page.screenshot(
            path=str(screenshot_path),
            full_page=full_page,
            type="png"
        )
        
        self._log_interaction({
            "action": "screenshot",
            "name": name,
            "path": str(screenshot_path),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return screenshot_path

    async def _capture_debug_screenshot(self, name: str) -> Path:
        """
        Capture debug screenshot for failures
        
        Args:
            name: Debug screenshot name
            
        Returns:
            Path to saved screenshot
        """
        debug_name = f"debug_{name}"
        return await self.capture_screenshot(debug_name)

    async def get_page_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive page performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        try:
            metrics = await self.page.evaluate("""
                () => {
                    const timing = performance.timing;
                    const navigation = performance.getEntriesByType('navigation')[0] || {};
                    const paint = performance.getEntriesByType('paint') || [];
                    
                    const firstPaint = paint.find(p => p.name === 'first-paint');
                    const firstContentfulPaint = paint.find(p => p.name === 'first-contentful-paint');
                    
                    return {
                        // Basic timing
                        domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                        loadComplete: timing.loadEventEnd - timing.navigationStart,
                        domInteractive: timing.domInteractive - timing.navigationStart,
                        
                        // Navigation timing
                        dnsLookup: timing.domainLookupEnd - timing.domainLookupStart,
                        tcpConnect: timing.connectEnd - timing.connectStart,
                        serverResponse: timing.responseEnd - timing.requestStart,
                        
                        // Paint timing
                        firstPaint: firstPaint ? firstPaint.startTime : null,
                        firstContentfulPaint: firstContentfulPaint ? firstContentfulPaint.startTime : null,
                        
                        // Resource timing
                        resourceCount: performance.getEntriesByType('resource').length,
                        
                        // Memory (if available)
                        usedJSHeapSize: performance.memory ? performance.memory.usedJSHeapSize : null,
                        totalJSHeapSize: performance.memory ? performance.memory.totalJSHeapSize : null
                    };
                }
            """)
            
            return metrics
        except Exception as e:
            print(f"Failed to get performance metrics: {e}")
            return {}

    async def get_accessibility_tree(self) -> Dict[str, Any]:
        """
        Get accessibility tree information
        
        Returns:
            Accessibility tree data
        """
        try:
            snapshot = await self.page.accessibility.snapshot()
            return snapshot or {}
        except Exception as e:
            print(f"Failed to get accessibility tree: {e}")
            return {}

    async def check_console_errors(self) -> List[Dict[str, Any]]:
        """
        Check for console errors and warnings
        
        Returns:
            List of console messages
        """
        console_messages = []
        
        def handle_console(msg):
            if msg.type in ['error', 'warning']:
                console_messages.append({
                    "type": msg.type,
                    "text": msg.text,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        self.page.on("console", handle_console)
        
        return console_messages

    async def validate_html(self) -> Dict[str, Any]:
        """
        Basic HTML validation
        
        Returns:
            Validation results
        """
        try:
            validation_results = await self.page.evaluate("""
                () => {
                    const issues = [];
                    
                    // Check for missing alt attributes on images
                    const imagesWithoutAlt = document.querySelectorAll('img:not([alt])');
                    if (imagesWithoutAlt.length > 0) {
                        issues.push({
                            type: 'accessibility',
                            message: `${imagesWithoutAlt.length} images missing alt attributes`
                        });
                    }
                    
                    // Check for missing form labels
                    const inputsWithoutLabels = document.querySelectorAll('input:not([aria-label]):not([aria-labelledby])');
                    const unlabeledInputs = Array.from(inputsWithoutLabels).filter(input => {
                        const label = document.querySelector(`label[for="${input.id}"]`);
                        return !label && input.type !== 'hidden' && input.type !== 'submit';
                    });
                    
                    if (unlabeledInputs.length > 0) {
                        issues.push({
                            type: 'accessibility',
                            message: `${unlabeledInputs.length} inputs without proper labels`
                        });
                    }
                    
                    // Check for heading hierarchy
                    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
                    const headingLevels = Array.from(headings).map(h => parseInt(h.tagName.charAt(1)));
                    
                    for (let i = 1; i < headingLevels.length; i++) {
                        if (headingLevels[i] > headingLevels[i-1] + 1) {
                            issues.push({
                                type: 'accessibility',
                                message: 'Heading hierarchy skips levels'
                            });
                            break;
                        }
                    }
                    
                    return {
                        valid: issues.length === 0,
                        issues: issues,
                        stats: {
                            images: document.querySelectorAll('img').length,
                            images_with_alt: document.querySelectorAll('img[alt]').length,
                            forms: document.querySelectorAll('form').length,
                            inputs: document.querySelectorAll('input').length,
                            headings: headings.length
                        }
                    };
                }
            """)
            
            return validation_results
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "issues": []
            }

    def _log_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """
        Log user interaction for debugging and analysis
        
        Args:
            interaction_data: Interaction details
        """
        self.interactions_log.append(interaction_data)

    def get_interactions_log(self) -> List[Dict[str, Any]]:
        """
        Get complete interactions log
        
        Returns:
            List of logged interactions
        """
        return self.interactions_log.copy()

    async def scroll_to_element(self, selector: str) -> None:
        """
        Scroll element into view
        
        Args:
            selector: CSS selector
        """
        try:
            locator = self.page.locator(selector)
            await locator.scroll_into_view_if_needed()
        except Exception as e:
            raise Exception(f"Failed to scroll to element '{selector}': {e}")

    async def drag_and_drop(self, source_selector: str, target_selector: str) -> None:
        """
        Perform drag and drop operation
        
        Args:
            source_selector: Source element CSS selector
            target_selector: Target element CSS selector
        """
        try:
            source = await self.wait_for_element_visible(source_selector)
            target = await self.wait_for_element_visible(target_selector)
            
            await source.drag_to(target)
            
            self._log_interaction({
                "action": "drag_and_drop",
                "source": source_selector,
                "target": target_selector,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            raise Exception(f"Failed to drag from '{source_selector}' to '{target_selector}': {e}")

    async def wait_for_animation_complete(self, selector: str = None, timeout: int = 5000) -> None:
        """
        Wait for CSS animations/transitions to complete
        
        Args:
            selector: Element selector (if None, waits for all animations)
            timeout: Maximum wait time in milliseconds
        """
        try:
            if selector:
                await self.page.wait_for_function(
                    f"""
                    () => {{
                        const element = document.querySelector('{selector}');
                        if (!element) return true;
                        
                        const computedStyle = getComputedStyle(element);
                        const animationDuration = computedStyle.animationDuration;
                        const transitionDuration = computedStyle.transitionDuration;
                        
                        return animationDuration === '0s' && transitionDuration === '0s';
                    }}
                    """,
                    timeout=timeout
                )
            else:
                # Wait for all animations to complete
                await self.page.wait_for_function(
                    """
                    () => {
                        const elements = document.querySelectorAll('*');
                        for (let element of elements) {
                            const computedStyle = getComputedStyle(element);
                            if (computedStyle.animationDuration !== '0s' || computedStyle.transitionDuration !== '0s') {
                                return false;
                            }
                        }
                        return true;
                    }
                    """,
                    timeout=timeout
                )
        except:
            # If we can't detect animations, wait a reasonable amount of time
            await self.page.wait_for_timeout(self.performance_thresholds["animation_duration"])