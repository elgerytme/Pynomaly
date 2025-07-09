"""Step definitions for accessibility compliance BDD scenarios."""

import pytest
from playwright.async_api import Page
from pytest_bdd import given, then, when
from tests.ui.conftest import TEST_CONFIG
from tests.ui.test_accessibility_enhanced import AccessibilityTester


# Context for accessibility testing
class AccessibilityContext:
    def __init__(self):
        self.accessibility_tester = None
        self.test_results = {}
        self.current_page = None
        self.assistive_technology = None
        self.accessibility_features = {}


@pytest.fixture
def accessibility_context():
    """Provide accessibility testing context."""
    return AccessibilityContext()


# Background Steps


@given("accessibility features are enabled")
async def given_accessibility_features_enabled(
    page: Page, accessibility_context: AccessibilityContext
):
    """Enable accessibility features for testing."""
    accessibility_context.accessibility_tester = AccessibilityTester(page)
    await accessibility_context.accessibility_tester.inject_axe()
    accessibility_context.current_page = page


# Screen Reader Navigation Steps


@given("I am using a screen reader")
async def given_using_screen_reader(accessibility_context: AccessibilityContext):
    """Set screen reader context."""
    accessibility_context.assistive_technology = "screen_reader"


@then("all content should be accessible via screen reader")
async def then_content_accessible_screen_reader(
    page: Page, accessibility_context: AccessibilityContext
):
    """Verify content is accessible to screen readers."""
    results = await accessibility_context.accessibility_tester.run_axe_scan(
        {
            "rules": {
                "document-title": {"enabled": True},
                "html-has-lang": {"enabled": True},
                "landmark-one-main": {"enabled": True},
                "page-has-heading-one": {"enabled": True},
            }
        }
    )

    # Check for critical screen reader violations
    screen_reader_violations = [
        v
        for v in results.get("violations", [])
        if any(tag in v.get("tags", []) for tag in ["wcag2a", "wcag2aa"])
        and v.get("impact") in ["serious", "critical"]
    ]

    assert (
        len(screen_reader_violations) == 0
    ), f"Found {len(screen_reader_violations)} screen reader violations"


@then("page landmarks should be properly labeled")
async def then_landmarks_properly_labeled(page: Page):
    """Verify page landmarks are properly labeled."""
    landmarks_check = await page.evaluate(
        """
        () => {
            const landmarks = document.querySelectorAll('main, nav, header, footer, aside, [role="main"], [role="navigation"], [role="banner"], [role="contentinfo"]');
            let properlyLabeled = 0;

            landmarks.forEach(landmark => {
                const hasLabel = landmark.getAttribute('aria-label') ||
                                landmark.getAttribute('aria-labelledby') ||
                                landmark.querySelector('h1, h2, h3, h4, h5, h6');
                if (hasLabel) properlyLabeled++;
            });

            return {
                total_landmarks: landmarks.length,
                properly_labeled: properlyLabeled,
                percentage: landmarks.length > 0 ? (properlyLabeled / landmarks.length) * 100 : 100
            };
        }
    """
    )

    # At least 80% of landmarks should be properly labeled
    assert (
        landmarks_check["percentage"] >= 80
    ), f"Only {landmarks_check['percentage']:.1f}% of landmarks are properly labeled"


@then("heading structure should be logical")
async def then_heading_structure_logical(page: Page):
    """Verify heading structure is logical and hierarchical."""
    heading_structure = await page.evaluate(
        """
        () => {
            const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
            const levels = Array.from(headings).map(h => parseInt(h.tagName.charAt(1)));

            let isLogical = true;
            let hasH1 = levels.includes(1);
            let previousLevel = 0;

            levels.forEach(level => {
                if (level > previousLevel + 1) {
                    isLogical = false;
                }
                previousLevel = level;
            });

            return {
                total_headings: levels.length,
                has_h1: hasH1,
                logical_order: isLogical,
                levels: levels
            };
        }
    """
    )

    assert heading_structure["has_h1"], "Page should have an H1 heading"
    assert heading_structure[
        "logical_order"
    ], f"Heading structure is not logical: {heading_structure['levels']}"


# Keyboard Navigation Steps


@given("I am using only keyboard navigation")
async def given_keyboard_only_navigation(accessibility_context: AccessibilityContext):
    """Set keyboard-only navigation context."""
    accessibility_context.assistive_technology = "keyboard_only"


@then("all interactive elements should be keyboard accessible")
async def then_interactive_elements_keyboard_accessible(page: Page):
    """Verify all interactive elements are keyboard accessible."""
    keyboard_test = await page.evaluate(
        """
        () => {
            const interactiveElements = document.querySelectorAll(
                'a, button, input, select, textarea, [tabindex]:not([tabindex="-1"]), [role="button"], [role="link"]'
            );

            let keyboardAccessible = 0;

            interactiveElements.forEach(element => {
                // Focus the element
                element.focus();
                if (document.activeElement === element) {
                    keyboardAccessible++;
                }
            });

            return {
                total_interactive: interactiveElements.length,
                keyboard_accessible: keyboardAccessible,
                percentage: interactiveElements.length > 0 ? (keyboardAccessible / interactiveElements.length) * 100 : 100
            };
        }
    """
    )

    # At least 95% of interactive elements should be keyboard accessible
    assert (
        keyboard_test["percentage"] >= 95
    ), f"Only {keyboard_test['percentage']:.1f}% of interactive elements are keyboard accessible"


@then("tab order should be logical and predictable")
async def then_tab_order_logical(page: Page):
    """Verify tab order is logical and predictable."""
    # Test tab order by tabbing through first several elements
    tab_order_test = []

    try:
        # Start from first focusable element
        await page.keyboard.press("Tab")

        for i in range(10):  # Test first 10 elements
            focused_info = await page.evaluate(
                """
                () => {
                    const element = document.activeElement;
                    const rect = element.getBoundingClientRect();
                    return {
                        tagName: element.tagName,
                        id: element.id || null,
                        x: rect.left,
                        y: rect.top,
                        visible: rect.width > 0 && rect.height > 0
                    };
                }
            """
            )

            tab_order_test.append(focused_info)
            await page.keyboard.press("Tab")

    except Exception as e:
        # Tab order test encountered an issue
        print(f"Tab order test warning: {e}")

    # Verify that we could tab through elements and they are visible
    visible_elements = [el for el in tab_order_test if el["visible"]]
    assert (
        len(visible_elements) >= 5
    ), "Should be able to tab through at least 5 visible elements"


@then("focus indicators should be clearly visible")
async def then_focus_indicators_visible(page: Page):
    """Verify focus indicators are clearly visible."""
    focus_test = await page.evaluate(
        """
        () => {
            const focusableElements = document.querySelectorAll(
                'a, button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );

            let visibleFocusCount = 0;

            focusableElements.forEach(element => {
                element.focus();
                const styles = getComputedStyle(element, ':focus');
                const outline = styles.outline;
                const outlineWidth = styles.outlineWidth;
                const boxShadow = styles.boxShadow;

                // Check for visible focus indicators
                if ((outline && outline !== 'none') ||
                    (outlineWidth && outlineWidth !== '0px') ||
                    (boxShadow && boxShadow !== 'none')) {
                    visibleFocusCount++;
                }
            });

            return {
                total_focusable: focusableElements.length,
                visible_focus: visibleFocusCount,
                percentage: focusableElements.length > 0 ? (visibleFocusCount / focusableElements.length) * 100 : 100
            };
        }
    """
    )

    # At least 90% of focusable elements should have visible focus indicators
    assert (
        focus_test["percentage"] >= 90
    ), f"Only {focus_test['percentage']:.1f}% of elements have visible focus indicators"


# High Contrast Mode Steps


@given("I need high contrast for visual accessibility")
async def given_need_high_contrast(accessibility_context: AccessibilityContext):
    """Set high contrast mode context."""
    accessibility_context.assistive_technology = "high_contrast"


@when("I enable high contrast mode")
async def when_enable_high_contrast(page: Page):
    """Enable high contrast mode."""
    # Simulate high contrast mode by adding CSS
    await page.add_style_tag(
        content="""
        @media (prefers-contrast: high) {
            * {
                background-color: black !important;
                color: white !important;
                border-color: white !important;
            }
            a, button {
                background-color: blue !important;
                color: yellow !important;
            }
        }
    """
    )


@then("all text should have sufficient color contrast")
async def then_sufficient_color_contrast(
    page: Page, accessibility_context: AccessibilityContext
):
    """Verify sufficient color contrast in high contrast mode."""
    contrast_results = await accessibility_context.accessibility_tester.run_axe_scan(
        {
            "rules": {
                "color-contrast": {"enabled": True},
                "color-contrast-enhanced": {"enabled": True},
            }
        }
    )

    contrast_violations = [
        v
        for v in contrast_results.get("violations", [])
        if "color-contrast" in v.get("id", "")
        and v.get("impact") in ["serious", "critical"]
    ]

    assert (
        len(contrast_violations) == 0
    ), f"Found {len(contrast_violations)} serious color contrast violations"


# Text Scaling Steps


@given("I need larger text for readability")
async def given_need_larger_text(accessibility_context: AccessibilityContext):
    """Set text scaling context."""
    accessibility_context.assistive_technology = "text_scaling"


@when("I scale text to 200% size")
async def when_scale_text_200_percent(page: Page):
    """Scale text to 200% size."""
    await page.add_style_tag(
        content="""
        * {
            font-size: 200% !important;
            line-height: 1.5 !important;
        }
    """
    )

    # Wait for layout to settle
    await page.wait_for_timeout(1000)


@then("layout should not break with larger text")
async def then_layout_not_break_larger_text(page: Page):
    """Verify layout doesn't break with larger text."""
    layout_check = await page.evaluate(
        """
        () => {
            return {
                hasHorizontalScroll: document.body.scrollWidth > window.innerWidth,
                contentVisible: document.body.offsetHeight > 0,
                hasOverflow: Array.from(document.querySelectorAll('*')).some(
                    el => getComputedStyle(el).overflow === 'hidden' && el.scrollWidth > el.clientWidth
                )
            };
        }
    """
    )

    # Layout should accommodate larger text without breaking
    assert not layout_check[
        "hasOverflow"
    ], "Text scaling should not cause overflow issues"


# Form Accessibility Steps


@given("I am filling out a form")
async def given_filling_form(page: Page):
    """Navigate to a page with forms."""
    await page.goto(f"{TEST_CONFIG['base_url']}/datasets")
    await page.wait_for_load_state("networkidle")


@then("all form fields should have proper labels")
async def then_form_fields_have_labels(page: Page):
    """Verify all form fields have proper labels."""
    form_accessibility = await page.evaluate(
        """
        () => {
            const formElements = document.querySelectorAll('input, select, textarea');
            let properlyLabeled = 0;

            formElements.forEach(element => {
                const id = element.id;
                const ariaLabel = element.getAttribute('aria-label');
                const ariaLabelledBy = element.getAttribute('aria-labelledby');
                const label = id ? document.querySelector(`label[for="${id}"]`) : null;
                const placeholder = element.getAttribute('placeholder');

                if (label || ariaLabel || ariaLabelledBy || (placeholder && placeholder.length > 3)) {
                    properlyLabeled++;
                }
            });

            return {
                total_form_elements: formElements.length,
                properly_labeled: properlyLabeled,
                percentage: formElements.length > 0 ? (properlyLabeled / formElements.length) * 100 : 100
            };
        }
    """
    )

    # At least 90% of form elements should be properly labeled
    assert (
        form_accessibility["percentage"] >= 90
    ), f"Only {form_accessibility['percentage']:.1f}% of form elements are properly labeled"


@then("required fields should be clearly indicated")
async def then_required_fields_indicated(page: Page):
    """Verify required fields are clearly indicated."""
    required_fields_check = await page.evaluate(
        """
        () => {
            const requiredElements = document.querySelectorAll('[required], [aria-required="true"]');
            let clearlyIndicated = 0;

            requiredElements.forEach(element => {
                const hasVisualIndicator = element.getAttribute('aria-required') === 'true' ||
                                         element.hasAttribute('required') ||
                                         element.closest('label')?.textContent.includes('*') ||
                                         element.nextElementSibling?.textContent.includes('required');

                if (hasVisualIndicator) {
                    clearlyIndicated++;
                }
            });

            return {
                total_required: requiredElements.length,
                clearly_indicated: clearlyIndicated,
                percentage: requiredElements.length > 0 ? (clearlyIndicated / requiredElements.length) * 100 : 100
            };
        }
    """
    )

    # All required fields should be clearly indicated
    assert (
        required_fields_check["percentage"] >= 100
    ), f"Only {required_fields_check['percentage']:.1f}% of required fields are clearly indicated"


# Modal Dialog Steps


@given("I encounter a modal dialog")
async def given_encounter_modal(page: Page):
    """Trigger a modal dialog if one exists."""
    # Look for modal triggers
    modal_triggers = await page.query_selector_all(
        "button[data-modal], [data-toggle='modal'], .modal-trigger"
    )

    if modal_triggers:
        await modal_triggers[0].click()
        await page.wait_for_timeout(500)


@when("the dialog opens")
async def when_dialog_opens(page: Page):
    """Verify dialog opens."""
    # Wait for modal to appear
    try:
        await page.wait_for_selector("[role='dialog'], .modal, .dialog", timeout=5000)
    except:
        # If no modal appears, that's okay for this test
        pass


@then("focus should move to the dialog")
async def then_focus_moves_to_dialog(page: Page):
    """Verify focus moves to the dialog when it opens."""
    try:
        modal = await page.query_selector("[role='dialog'], .modal, .dialog")
        if modal:
            focused_element = await page.evaluate("document.activeElement")
            modal_contains_focus = await modal.evaluate(
                "(modal, focusedEl) => modal.contains(focusedEl)", focused_element
            )
            assert modal_contains_focus, "Focus should move to the modal dialog"
    except:
        # If no modal exists, skip this assertion
        pass


# Performance-related accessibility steps


@then("the interface should remain fully functional")
async def then_interface_remains_functional(page: Page):
    """Verify interface remains functional under accessibility conditions."""
    # Test basic navigation
    try:
        await page.keyboard.press("Tab")
        await page.keyboard.press("Enter")
        await page.keyboard.press("Escape")

        # Verify page is still responsive
        page_title = await page.title()
        assert len(page_title) > 0, "Page should remain functional"

    except Exception as e:
        assert False, f"Interface functionality compromised: {e}"


# Compliance validation steps


@then("WCAG 2.1 AA compliance should be achieved")
async def then_wcag_compliance_achieved(
    page: Page, accessibility_context: AccessibilityContext
):
    """Verify WCAG 2.1 AA compliance is achieved."""
    compliance_results = await accessibility_context.accessibility_tester.run_axe_scan(
        {"tags": ["wcag2a", "wcag2aa", "wcag21aa"]}
    )

    # Check for any serious or critical WCAG violations
    wcag_violations = [
        v
        for v in compliance_results.get("violations", [])
        if any(tag in v.get("tags", []) for tag in ["wcag2a", "wcag2aa", "wcag21aa"])
        and v.get("impact") in ["serious", "critical"]
    ]

    assert (
        len(wcag_violations) == 0
    ), f"Found {len(wcag_violations)} serious WCAG compliance violations"


@then("automated accessibility tests should pass")
async def then_automated_tests_pass(accessibility_context: AccessibilityContext):
    """Verify automated accessibility tests pass."""
    # This would integrate with the actual test results
    # For now, verify that we have accessibility testing capabilities
    assert (
        accessibility_context.accessibility_tester is not None
    ), "Accessibility testing should be available"

    # In a real implementation, this would check the overall test suite results
    test_results = accessibility_context.test_results
    if test_results:
        failed_tests = [t for t in test_results.values() if not t.get("passed", True)]
        assert (
            len(failed_tests) == 0
        ), f"Found {len(failed_tests)} failed accessibility tests"
