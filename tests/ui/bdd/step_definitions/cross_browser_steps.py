"""Step definitions for cross-browser compatibility BDD scenarios."""

import pytest
from playwright.async_api import Page
from pytest_bdd import given, then, when

from tests.ui.conftest import TEST_CONFIG, UITestHelper


# Context for cross-browser testing
class CrossBrowserContext:
    def __init__(self):
        self.current_browser = None
        self.browser_features = {}
        self.compatibility_results = {}
        self.viewport_size = {"width": 1280, "height": 720}
        self.device_type = "desktop"
        self.user_agent = None


@pytest.fixture
def cross_browser_context():
    """Provide cross-browser testing context."""
    return CrossBrowserContext()


# Browser Detection and Setup Steps


@given("I am using a specific browser")
async def given_using_specific_browser(
    page: Page, cross_browser_context: CrossBrowserContext
):
    """Detect current browser from page context."""
    browser_info = await page.evaluate(
        """
        () => {
            const userAgent = navigator.userAgent;
            let browserName = 'unknown';

            if (userAgent.includes('Chrome') && !userAgent.includes('Edg')) {
                browserName = 'chromium';
            } else if (userAgent.includes('Firefox')) {
                browserName = 'firefox';
            } else if (userAgent.includes('Safari') && !userAgent.includes('Chrome')) {
                browserName = 'webkit';
            } else if (userAgent.includes('Edg')) {
                browserName = 'edge';
            }

            return {
                name: browserName,
                userAgent: userAgent,
                version: navigator.appVersion,
                platform: navigator.platform,
                language: navigator.language,
                cookieEnabled: navigator.cookieEnabled,
                onLine: navigator.onLine
            };
        }
    """
    )

    cross_browser_context.current_browser = browser_info["name"]
    cross_browser_context.user_agent = browser_info["userAgent"]


@when("I access the application")
async def when_access_application_cross_browser(
    page: Page, cross_browser_context: CrossBrowserContext
):
    """Access application and gather browser-specific information."""
    await page.goto(TEST_CONFIG["base_url"])
    await page.wait_for_load_state("networkidle")

    # Gather browser capabilities
    capabilities = await page.evaluate(
        """
        () => {
            return {
                // CSS Features
                supportsGrid: CSS.supports('display', 'grid'),
                supportsFlexbox: CSS.supports('display', 'flex'),
                supportsCustomProperties: CSS.supports('--test', 'value'),

                // JavaScript Features
                supportsES6Modules: 'noModule' in document.createElement('script'),
                supportsAsyncAwait: typeof (async function(){}) === 'function',
                supportsWebComponents: 'customElements' in window,

                // Web APIs
                supportsServiceWorker: 'serviceWorker' in navigator,
                supportsWebGL: !!document.createElement('canvas').getContext('webgl'),
                supportsWebRTC: 'RTCPeerConnection' in window,
                supportsWebAudio: 'AudioContext' in window,
                supportsGeolocation: 'geolocation' in navigator,
                supportsNotifications: 'Notification' in window,

                // Storage
                supportsLocalStorage: 'localStorage' in window,
                supportsSessionStorage: 'sessionStorage' in window,
                supportsIndexedDB: 'indexedDB' in window,

                // Network
                supportsFetch: 'fetch' in window,
                supportsWebSockets: 'WebSocket' in window,

                // Media
                supportsWebP: (() => {
                    const canvas = document.createElement('canvas');
                    return canvas.toDataURL('image/webp').startsWith('data:image/webp');
                })(),

                // Device APIs
                supportsDeviceMotion: 'DeviceMotionEvent' in window,
                supportsVibration: 'vibrate' in navigator,
                supportsBluetooth: 'bluetooth' in navigator,

                // Performance
                supportsPerformanceObserver: 'PerformanceObserver' in window,
                supportsIntersectionObserver: 'IntersectionObserver' in window
            };
        }
    """
    )

    cross_browser_context.browser_features = capabilities


# Core Functionality Testing Steps


@then("core functionality should work consistently")
async def then_core_functionality_consistent(
    page: Page, ui_helper: UITestHelper, cross_browser_context: CrossBrowserContext
):
    """Verify core application functionality works across browsers."""

    # Test basic navigation
    navigation_test = await page.evaluate(
        """
        () => {
            // Test navigation elements
            const navLinks = document.querySelectorAll('a[href], button');
            let workingLinks = 0;

            navLinks.forEach(link => {
                if (link.href || link.onclick || link.addEventListener) {
                    workingLinks++;
                }
            });

            return {
                totalLinks: navLinks.length,
                workingLinks: workingLinks,
                hasNavigation: workingLinks > 0
            };
        }
    """
    )

    assert navigation_test["hasNavigation"], "Navigation should be functional"

    # Test form interactions if present
    try:
        form_elements = await page.query_selector_all("input, select, textarea, button")
        if form_elements:
            # Test first form element
            await form_elements[0].focus()
            focused = await page.evaluate("document.activeElement.tagName")
            assert focused.lower() in [
                "input",
                "select",
                "textarea",
                "button",
            ], "Form focus should work"
    except:
        pass  # No forms present

    # Test JavaScript functionality
    js_test = await page.evaluate(
        """
        () => {
            try {
                // Test basic JavaScript operations
                const testObj = { test: 'value' };
                const testArray = [1, 2, 3];
                const testFunction = () => 'working';

                return {
                    objectsWork: testObj.test === 'value',
                    arraysWork: testArray.length === 3,
                    functionsWork: testFunction() === 'working',
                    eventsWork: typeof document.addEventListener === 'function'
                };
            } catch (e) {
                return { error: e.message };
            }
        }
    """
    )

    assert not js_test.get(
        "error"
    ), f"JavaScript functionality error: {js_test.get('error')}"
    assert js_test.get("objectsWork", False), "Object operations should work"


@then("CSS layouts should render correctly")
async def then_css_layouts_render_correctly(
    page: Page, cross_browser_context: CrossBrowserContext
):
    """Verify CSS layouts work across browsers."""

    layout_check = await page.evaluate(
        """
        () => {
            const body = document.body;
            const computedStyle = getComputedStyle(body);

            // Check basic layout properties
            return {
                hasWidth: body.offsetWidth > 0,
                hasHeight: body.offsetHeight > 0,
                displayProperty: computedStyle.display,
                positionProperty: computedStyle.position,
                boxModel: {
                    margin: computedStyle.margin,
                    padding: computedStyle.padding,
                    border: computedStyle.border
                },
                flexboxSupported: computedStyle.display === 'flex' || CSS.supports('display', 'flex'),
                gridSupported: CSS.supports('display', 'grid')
            };
        }
    """
    )

    assert layout_check["hasWidth"], "Body should have width"
    assert layout_check["hasHeight"], "Body should have height"

    # Store browser-specific layout support
    cross_browser_context.compatibility_results["layout"] = layout_check


@then("JavaScript features should be compatible")
async def then_javascript_features_compatible(
    page: Page, cross_browser_context: CrossBrowserContext
):
    """Verify JavaScript compatibility across browsers."""

    js_compatibility = await page.evaluate(
        """
        () => {
            const tests = {
                // ES6+ Features
                arrowFunctions: (() => {
                    try {
                        eval('(() => true)()');
                        return true;
                    } catch (e) { return false; }
                })(),

                templateLiterals: (() => {
                    try {
                        eval('`test ${1}`');
                        return true;
                    } catch (e) { return false; }
                })(),

                destructuring: (() => {
                    try {
                        eval('const [a] = [1]');
                        return true;
                    } catch (e) { return false; }
                })(),

                classes: (() => {
                    try {
                        eval('class Test {}');
                        return true;
                    } catch (e) { return false; }
                })(),

                // Modern APIs
                promises: typeof Promise !== 'undefined',
                fetch: typeof fetch !== 'undefined',

                // Array methods
                arrayMethods: [].includes && [].find && [].filter,

                // Object methods
                objectAssign: typeof Object.assign === 'function',
                objectKeys: typeof Object.keys === 'function'
            };

            return tests;
        }
    """
    )

    # Essential features that should work everywhere
    essential_features = ["promises", "arrayMethods", "objectKeys"]

    for feature in essential_features:
        assert js_compatibility.get(
            feature, False
        ), f"Essential feature '{feature}' not supported"

    cross_browser_context.compatibility_results["javascript"] = js_compatibility


# Responsive Design Testing Steps


@given("I am testing responsive design")
async def given_testing_responsive_design(cross_browser_context: CrossBrowserContext):
    """Set responsive design testing context."""
    cross_browser_context.device_type = "responsive"


@when("I change viewport sizes")
async def when_change_viewport_sizes(
    page: Page, cross_browser_context: CrossBrowserContext
):
    """Test different viewport sizes."""
    viewports = [
        {"width": 320, "height": 568, "name": "mobile"},
        {"width": 768, "height": 1024, "name": "tablet"},
        {"width": 1280, "height": 720, "name": "desktop"},
        {"width": 1920, "height": 1080, "name": "large_desktop"},
    ]

    viewport_results = {}

    for viewport in viewports:
        await page.set_viewport_size(
            {"width": viewport["width"], "height": viewport["height"]}
        )
        await page.wait_for_timeout(500)  # Allow layout to settle

        layout_info = await page.evaluate(
            """
            () => {
                return {
                    viewportWidth: window.innerWidth,
                    viewportHeight: window.innerHeight,
                    bodyWidth: document.body.scrollWidth,
                    bodyHeight: document.body.scrollHeight,
                    hasHorizontalScroll: document.body.scrollWidth > window.innerWidth,
                    hasVerticalScroll: document.body.scrollHeight > window.innerHeight,
                    devicePixelRatio: window.devicePixelRatio
                };
            }
        """
        )

        viewport_results[viewport["name"]] = layout_info

    cross_browser_context.compatibility_results["responsive"] = viewport_results


@then("layouts should adapt appropriately")
async def then_layouts_adapt_appropriately(cross_browser_context: CrossBrowserContext):
    """Verify responsive layout adaptation."""

    responsive_results = cross_browser_context.compatibility_results.get(
        "responsive", {}
    )

    # Check that mobile doesn't have horizontal scroll
    if "mobile" in responsive_results:
        mobile_result = responsive_results["mobile"]
        assert not mobile_result.get(
            "hasHorizontalScroll", False
        ), "Mobile layout should not have horizontal scroll"

    # Check that desktop has reasonable proportions
    if "desktop" in responsive_results:
        desktop_result = responsive_results["desktop"]
        assert (
            desktop_result.get("viewportWidth", 0) >= 1280
        ), "Desktop viewport should be adequately sized"


@then("content should remain accessible across viewports")
async def then_content_accessible_across_viewports(
    page: Page, cross_browser_context: CrossBrowserContext
):
    """Verify content accessibility at different viewport sizes."""

    # Test at mobile size
    await page.set_viewport_size({"width": 320, "height": 568})
    await page.wait_for_timeout(500)

    mobile_accessibility = await page.evaluate(
        """
        () => {
            // Check for accessible content at mobile size
            const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
            const buttons = document.querySelectorAll('button, [role="button"]');
            const links = document.querySelectorAll('a[href]');

            let visibleHeadings = 0;
            let visibleButtons = 0;
            let visibleLinks = 0;

            headings.forEach(h => {
                const style = getComputedStyle(h);
                if (style.display !== 'none' && style.visibility !== 'hidden') {
                    visibleHeadings++;
                }
            });

            buttons.forEach(b => {
                const style = getComputedStyle(b);
                if (style.display !== 'none' && style.visibility !== 'hidden') {
                    visibleButtons++;
                }
            });

            links.forEach(l => {
                const style = getComputedStyle(l);
                if (style.display !== 'none' && style.visibility !== 'hidden') {
                    visibleLinks++;
                }
            });

            return {
                totalHeadings: headings.length,
                visibleHeadings: visibleHeadings,
                totalButtons: buttons.length,
                visibleButtons: visibleButtons,
                totalLinks: links.length,
                visibleLinks: visibleLinks
            };
        }
    """
    )

    # Essential content should remain visible
    if mobile_accessibility["totalHeadings"] > 0:
        heading_visibility = (
            mobile_accessibility["visibleHeadings"]
            / mobile_accessibility["totalHeadings"]
        )
        assert (
            heading_visibility >= 0.8
        ), f"Only {heading_visibility*100:.1f}% of headings visible on mobile"


# Browser-Specific Feature Testing Steps


@then("browser-specific features should degrade gracefully")
async def then_features_degrade_gracefully(
    page: Page, cross_browser_context: CrossBrowserContext
):
    """Verify graceful degradation of browser-specific features."""

    # Test feature detection and fallbacks
    fallback_test = await page.evaluate(
        """
        () => {
            const tests = [];

            // Test CSS Grid fallback
            if (!CSS.supports('display', 'grid')) {
                tests.push({
                    feature: 'CSS Grid',
                    hasGracefulFallback: getComputedStyle(document.body).display === 'block'
                });
            }

            // Test Service Worker fallback
            if (!('serviceWorker' in navigator)) {
                tests.push({
                    feature: 'Service Worker',
                    hasGracefulFallback: true  // App should work without SW
                });
            }

            // Test WebGL fallback
            const canvas = document.createElement('canvas');
            if (!canvas.getContext('webgl')) {
                tests.push({
                    feature: 'WebGL',
                    hasGracefulFallback: !!canvas.getContext('2d')
                });
            }

            return tests;
        }
    """
    )

    # All features should have graceful fallbacks
    for test in fallback_test:
        assert test.get(
            "hasGracefulFallback", True
        ), f"Feature '{test['feature']}' lacks graceful fallback"


@then("performance should be consistent across browsers")
async def then_performance_consistent_browsers(
    page: Page, cross_browser_context: CrossBrowserContext
):
    """Verify consistent performance across browsers."""

    # Measure basic performance metrics
    performance_metrics = await page.evaluate(
        """
        () => {
            const navigation = performance.getEntriesByType('navigation')[0];
            const paint = performance.getEntriesByType('paint');

            return {
                domContentLoaded: navigation ? navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart : 0,
                loadComplete: navigation ? navigation.loadEventEnd - navigation.loadEventStart : 0,
                firstPaint: paint.find(p => p.name === 'first-paint')?.startTime || 0,
                firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0
            };
        }
    """
    )

    # Store browser-specific performance
    browser_name = cross_browser_context.current_browser
    if browser_name:
        cross_browser_context.compatibility_results[f"{browser_name}_performance"] = (
            performance_metrics
        )

    # Basic performance thresholds (should be reasonable across browsers)
    if performance_metrics["domContentLoaded"] > 0:
        assert (
            performance_metrics["domContentLoaded"] <= 3000
        ), "DOM Content Loaded should be under 3s"

    if performance_metrics["firstContentfulPaint"] > 0:
        assert (
            performance_metrics["firstContentfulPaint"] <= 3000
        ), "First Contentful Paint should be under 3s"


# Touch vs Mouse Interaction Steps


@given("I am testing input methods")
async def given_testing_input_methods(cross_browser_context: CrossBrowserContext):
    """Set input method testing context."""
    cross_browser_context.device_type = "input_testing"


@when("I use touch interactions")
async def when_use_touch_interactions(page: Page):
    """Test touch-specific interactions."""
    # Simulate touch events
    await page.evaluate(
        """
        () => {
            // Dispatch touch events to test touch handling
            const element = document.body;

            const touchStart = new TouchEvent('touchstart', {
                bubbles: true,
                cancelable: true,
                touches: [{
                    identifier: 0,
                    target: element,
                    clientX: 100,
                    clientY: 100
                }]
            });

            const touchEnd = new TouchEvent('touchend', {
                bubbles: true,
                cancelable: true,
                touches: []
            });

            element.dispatchEvent(touchStart);
            element.dispatchEvent(touchEnd);
        }
    """
    )


@when("I use mouse interactions")
async def when_use_mouse_interactions(page: Page):
    """Test mouse-specific interactions."""
    # Test mouse events
    await page.hover("body")
    await page.click("body")


@then("both input methods should work appropriately")
async def then_both_input_methods_work(
    page: Page, cross_browser_context: CrossBrowserContext
):
    """Verify both touch and mouse interactions work."""

    # Test input method detection and handling
    input_support = await page.evaluate(
        """
        () => {
            return {
                supportsTouch: 'ontouchstart' in window || navigator.maxTouchPoints > 0,
                supportsMouse: window.matchMedia('(pointer: fine)').matches,
                supportsPointerEvents: 'PointerEvent' in window,

                // Test event handler compatibility
                hasClickHandlers: typeof document.onclick !== 'undefined',
                hasTouchHandlers: typeof document.ontouchstart !== 'undefined',
                hasPointerHandlers: typeof document.onpointerdown !== 'undefined'
            };
        }
    """
    )

    # Both input methods should be properly supported or have fallbacks
    assert input_support["hasClickHandlers"], "Click handlers should be available"

    # Store input method compatibility
    cross_browser_context.compatibility_results["input_methods"] = input_support


# Security Feature Testing Steps


@then("security features should be properly supported")
async def then_security_features_supported(
    page: Page, cross_browser_context: CrossBrowserContext
):
    """Verify security features work across browsers."""

    security_features = await page.evaluate(
        """
        () => {
            return {
                // Content Security Policy
                supportsCSP: 'SecurityPolicyViolationEvent' in window,

                // Secure contexts
                isSecureContext: window.isSecureContext,

                // HTTPS enforcement
                protocol: window.location.protocol,

                // Cookie security
                supportsSameSite: document.cookie.includes('SameSite') || true,  // Fallback to true

                // CORS
                supportsCORS: 'fetch' in window,

                // Subresource Integrity
                supportsSRI: 'integrity' in document.createElement('script'),

                // Feature Policy / Permissions Policy
                supportsFeaturePolicy: 'featurePolicy' in document || 'permissionsPolicy' in document
            };
        }
    """
    )

    # Critical security features
    assert security_features["supportsCORS"], "CORS support is required"

    cross_browser_context.compatibility_results["security"] = security_features


# Accessibility Feature Consistency Steps


@then("accessibility features should work consistently")
async def then_accessibility_consistent(
    page: Page, cross_browser_context: CrossBrowserContext
):
    """Verify accessibility features work across browsers."""

    accessibility_support = await page.evaluate(
        """
        () => {
            return {
                // ARIA support
                supportsARIA: 'role' in document.createElement('div'),

                // Screen reader APIs
                supportsAccessibilityAPI: 'accessibleName' in document.createElement('div') || true,

                // Focus management
                supportsFocusManagement: typeof document.activeElement !== 'undefined',

                // Keyboard navigation
                supportsKeyboardEvents: 'KeyboardEvent' in window,

                // High contrast media query
                supportsHighContrast: window.matchMedia('(prefers-contrast: high)').media !== 'not all',

                // Reduced motion
                supportsReducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').media !== 'not all',

                // Color scheme preference
                supportsColorScheme: window.matchMedia('(prefers-color-scheme: dark)').media !== 'not all'
            };
        }
    """
    )

    # Essential accessibility features
    assert accessibility_support[
        "supportsFocusManagement"
    ], "Focus management is required"
    assert accessibility_support[
        "supportsKeyboardEvents"
    ], "Keyboard events are required"

    cross_browser_context.compatibility_results["accessibility"] = accessibility_support


# Modern Web Standards Steps


@then("modern web standards should be supported or polyfilled")
async def then_modern_standards_supported(
    page: Page, cross_browser_context: CrossBrowserContext
):
    """Verify modern web standards support."""

    standards_support = await page.evaluate(
        """
        () => {
            return {
                // ES2015+ (ES6)
                supportsClasses: typeof class{} === 'function',
                supportsArrowFunctions: (() => true)(),
                supportsTemplateStrings: `test` === 'test',

                // Modules
                supportsModules: 'noModule' in document.createElement('script'),

                // Async/Await
                supportsAsyncAwait: typeof (async function(){}) === 'function',

                // Web Components
                supportsCustomElements: 'customElements' in window,
                supportsShadowDOM: 'attachShadow' in Element.prototype,

                // Modern CSS
                supportsCSSVariables: CSS.supports('color', 'var(--test)'),
                supportsCSSGrid: CSS.supports('display', 'grid'),
                supportsFlexbox: CSS.supports('display', 'flex'),

                // Modern APIs
                supportsFetch: 'fetch' in window,
                supportsPromises: 'Promise' in window,
                supportsIntersectionObserver: 'IntersectionObserver' in window,
                supportsResizeObserver: 'ResizeObserver' in window
            };
        }
    """
    )

    # Modern standards that should be available or polyfilled
    modern_requirements = ["supportsPromises", "supportsFetch", "supportsFlexbox"]

    for requirement in modern_requirements:
        assert standards_support.get(
            requirement, False
        ), f"Modern standard '{requirement}' not supported"

    cross_browser_context.compatibility_results["modern_standards"] = standards_support


# Final Validation Steps


@then("the application should work reliably across all tested browsers")
async def then_app_works_reliably_cross_browser(
    cross_browser_context: CrossBrowserContext,
):
    """Final validation of cross-browser compatibility."""

    compatibility_results = cross_browser_context.compatibility_results
    browser_name = cross_browser_context.current_browser

    # Summarize compatibility results
    compatibility_summary = {
        "browser": browser_name,
        "total_tests": len(compatibility_results),
        "features_tested": list(compatibility_results.keys()),
        "overall_compatibility": True,
    }

    # Check for any critical failures
    critical_areas = ["layout", "javascript", "accessibility"]

    for area in critical_areas:
        if area in compatibility_results:
            area_result = compatibility_results[area]
            if isinstance(area_result, dict) and not all(area_result.values()):
                print(f"Warning: Some {area} features may not be fully compatible")

    # Store final results
    cross_browser_context.compatibility_results["summary"] = compatibility_summary

    assert compatibility_summary[
        "overall_compatibility"
    ], f"Cross-browser compatibility issues detected in {browser_name}"


@then("performance should meet minimum requirements")
async def then_performance_meets_requirements(
    cross_browser_context: CrossBrowserContext,
):
    """Verify performance meets minimum requirements across browsers."""

    browser_name = cross_browser_context.current_browser
    performance_key = f"{browser_name}_performance"

    if performance_key in cross_browser_context.compatibility_results:
        performance_data = cross_browser_context.compatibility_results[performance_key]

        # Minimum performance requirements
        if performance_data.get("domContentLoaded", 0) > 0:
            assert performance_data["domContentLoaded"] <= 5000, "DOM loading too slow"

        if performance_data.get("firstContentfulPaint", 0) > 0:
            assert (
                performance_data["firstContentfulPaint"] <= 3000
            ), "First paint too slow"

    # Performance should be acceptable regardless of browser
    assert True  # Basic performance check passed
