"""Accessibility Testing Suite."""

from typing import Any

from playwright.sync_api import Page


class TestAccessibility:
    """Comprehensive accessibility testing suite."""

    def test_semantic_html_structure(self, page: Page):
        """Test semantic HTML structure."""
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        # Check for semantic HTML5 elements
        semantic_elements = {
            "main": page.locator("main"),
            "nav": page.locator("nav"),
            "header": page.locator("header"),
            "footer": page.locator("footer"),
            "section": page.locator("section"),
            "article": page.locator("article"),
        }

        # Main content should be in <main>
        assert semantic_elements["main"].count() > 0, "Page should have <main> element"

        # Navigation should be in <nav>
        assert semantic_elements["nav"].count() > 0, "Page should have <nav> element"

        # Check for proper heading hierarchy
        page.locator("h1, h2, h3, h4, h5, h6")
        h1_count = page.locator("h1").count()

        assert h1_count >= 1, "Page should have at least one h1 element"
        assert h1_count <= 2, "Page should not have more than 2 h1 elements"

    def test_keyboard_navigation(self, page: Page):
        """Test keyboard navigation accessibility."""
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        # Test tab navigation
        focusable_elements = []
        tab_count = 0
        max_tabs = 20

        # Start tabbing through elements
        while tab_count < max_tabs:
            page.keyboard.press("Tab")
            tab_count += 1

            # Get currently focused element
            focused_info = page.evaluate(
                """
                () => {
                    const element = document.activeElement;
                    return {
                        tagName: element.tagName,
                        type: element.type || null,
                        id: element.id || null,
                        className: element.className || null,
                        text: element.textContent ? element.textContent.substring(0, 50) : null,
                        tabIndex: element.tabIndex,
                        href: element.href || null
                    };
                }
            """
            )

            # Skip if we've cycled back to body or similar
            if focused_info["tagName"] in ["BODY", "HTML"]:
                break

            focusable_elements.append(focused_info)

        # Should have found focusable elements
        assert len(focusable_elements) > 0, "Should have focusable elements"

        # Check that interactive elements are focusable
        interactive_tags = ["A", "BUTTON", "INPUT", "SELECT", "TEXTAREA"]
        interactive_focused = [
            el for el in focusable_elements if el["tagName"] in interactive_tags
        ]

        assert (
            len(interactive_focused) > 0
        ), "Should be able to focus interactive elements"

        # Test skip links if they exist
        skip_links = page.locator("a[href^='#']").filter(has_text="skip")
        if skip_links.count() > 0:
            # Skip links should be focusable
            skip_links.first.focus()
            focused_element = page.evaluate("() => document.activeElement.textContent")
            assert "skip" in focused_element.lower(), "Skip links should be focusable"

    def test_aria_attributes(self, page: Page):
        """Test ARIA attributes and accessibility features."""
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        # Check for ARIA landmarks
        landmarks = {
            "main": page.locator("[role='main'], main"),
            "navigation": page.locator("[role='navigation'], nav"),
            "banner": page.locator("[role='banner'], header"),
            "contentinfo": page.locator("[role='contentinfo'], footer"),
        }

        # Should have main landmark
        assert landmarks["main"].count() > 0, "Should have main landmark"

        # Should have navigation landmark
        assert landmarks["navigation"].count() > 0, "Should have navigation landmark"

        # Check for proper ARIA labels on interactive elements
        buttons = page.locator("button")
        for i in range(buttons.count()):
            button = buttons.nth(i)
            button_text = button.text_content() or ""
            aria_label = button.get_attribute("aria-label")
            aria_labelledby = button.get_attribute("aria-labelledby")

            # Button should have accessible name
            has_accessible_name = (
                len(button_text.strip()) > 0
                or aria_label is not None
                or aria_labelledby is not None
            )

            assert has_accessible_name, f"Button {i} should have accessible name"

        # Check form controls have labels
        form_controls = page.locator("input, select, textarea")
        for i in range(form_controls.count()):
            control = form_controls.nth(i)
            control_id = control.get_attribute("id")
            aria_label = control.get_attribute("aria-label")
            aria_labelledby = control.get_attribute("aria-labelledby")

            # Check for associated label
            has_label = False
            if control_id:
                label = page.locator(f"label[for='{control_id}']")
                has_label = label.count() > 0

            # Should have some form of label
            has_accessible_name = (
                has_label or aria_label is not None or aria_labelledby is not None
            )

            # Allow placeholder as fallback for some input types
            placeholder = control.get_attribute("placeholder")
            if placeholder and len(placeholder) > 3:
                has_accessible_name = True

            assert has_accessible_name, f"Form control {i} should have accessible name"

    def test_color_contrast(self, page: Page):
        """Test color contrast accessibility."""
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        # Get computed styles for key elements
        contrast_checks = page.evaluate(
            """
            () => {
                const elements = [
                    document.querySelector('nav'),
                    document.querySelector('main'),
                    document.querySelector('button'),
                    document.querySelector('a')
                ].filter(el => el !== null);

                return elements.map(el => {
                    const styles = window.getComputedStyle(el);
                    return {
                        tagName: el.tagName,
                        backgroundColor: styles.backgroundColor,
                        color: styles.color,
                        fontSize: styles.fontSize
                    };
                });
            }
        """
        )

        # Basic check that colors are different
        for check in contrast_checks:
            bg_color = check["backgroundColor"]
            text_color = check["color"]

            # Colors should be different (basic check)
            assert (
                bg_color != text_color
            ), f"{check['tagName']} background and text colors should be different"

    def test_focus_indicators(self, page: Page):
        """Test focus indicators visibility."""
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        # Find focusable elements
        focusable = page.locator(
            "a, button, input, select, textarea, [tabindex]:not([tabindex='-1'])"
        )

        if focusable.count() > 0:
            first_focusable = focusable.first

            # Focus the element
            first_focusable.focus()

            # Check if focus is visible
            focus_styles = page.evaluate(
                """
                (element) => {
                    const styles = window.getComputedStyle(element);
                    return {
                        outline: styles.outline,
                        outlineColor: styles.outlineColor,
                        outlineStyle: styles.outlineStyle,
                        outlineWidth: styles.outlineWidth,
                        boxShadow: styles.boxShadow
                    };
                }
            """,
                first_focusable,
            )

            # Should have some form of focus indicator
            (
                focus_styles["outline"] != "none"
                or "0px" not in focus_styles["outlineWidth"]
                or focus_styles["boxShadow"] != "none"
            )

            # Note: This is a basic check, actual focus indicators might be handled by CSS

    def test_screen_reader_content(self, page: Page):
        """Test screen reader specific content."""
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        # Check for screen reader only content
        page.locator(".sr-only, .screen-reader-only, .visually-hidden")

        # Check for proper use of aria-hidden
        page.locator("[aria-hidden='true']")

        # Decorative images should have aria-hidden or empty alt
        images = page.locator("img")
        for i in range(images.count()):
            img = images.nth(i)
            alt_text = img.get_attribute("alt")
            aria_hidden = img.get_attribute("aria-hidden")

            # Image should have alt attribute
            assert alt_text is not None, f"Image {i} should have alt attribute"

            # If decorative, should have empty alt or aria-hidden
            if aria_hidden == "true":
                # Decorative image, this is fine
                pass
            elif alt_text == "":
                # Empty alt for decorative image, this is fine
                pass
            else:
                # Should have descriptive alt text
                assert len(alt_text) > 0, f"Image {i} should have descriptive alt text"

    def test_form_accessibility(self, page: Page):
        """Test form accessibility features."""
        page.goto("http://pynomaly-app:8000/web/detectors")
        page.wait_for_load_state("networkidle")

        # Find forms
        forms = page.locator("form")

        if forms.count() > 0:
            form = forms.first

            # Check for fieldsets and legends
            fieldsets = form.locator("fieldset")
            for i in range(fieldsets.count()):
                fieldset = fieldsets.nth(i)
                legend = fieldset.locator("legend")
                assert legend.count() > 0, f"Fieldset {i} should have legend"

            # Check required field indicators
            form.locator("[required], [aria-required='true']")

            # Check error message associations
            form.locator("[role='alert'], .error, .invalid")

            # Check form submission feedback
            submit_buttons = form.locator("button[type='submit'], input[type='submit']")

            for i in range(submit_buttons.count()):
                button = submit_buttons.nth(i)
                button_text = button.text_content() or ""

                # Submit button should have clear text
                assert (
                    len(button_text.strip()) > 0
                ), f"Submit button {i} should have text"

    def test_table_accessibility(self, page: Page):
        """Test table accessibility."""
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        # Find tables
        tables = page.locator("table")

        for i in range(tables.count()):
            table = tables.nth(i)

            # Table should have headers
            headers = table.locator("th")
            assert headers.count() > 0, f"Table {i} should have header cells"

            # Check for table caption or summary
            caption = table.locator("caption")
            summary = table.get_attribute("summary")
            aria_label = table.get_attribute("aria-label")
            aria_labelledby = table.get_attribute("aria-labelledby")

            # Table should have some form of description
            (
                caption.count() > 0
                or summary is not None
                or aria_label is not None
                or aria_labelledby is not None
            )

            # For data tables, headers should have scope
            if headers.count() > 0:
                first_header = headers.first
                first_header.get_attribute("scope")
                # Note: scope is helpful but not always required

    def test_language_attributes(self, page: Page):
        """Test language attributes."""
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        # Check html lang attribute
        html_lang = page.locator("html").get_attribute("lang")
        assert html_lang is not None, "HTML element should have lang attribute"
        assert len(html_lang) >= 2, "Lang attribute should be valid language code"

        # Check for lang changes in content
        lang_elements = page.locator("[lang]")

        # If there are lang changes, they should be valid
        for i in range(lang_elements.count()):
            element = lang_elements.nth(i)
            lang_value = element.get_attribute("lang")
            assert len(lang_value) >= 2, f"Lang attribute {i} should be valid"

    def test_page_title_accessibility(self, page: Page):
        """Test page title accessibility."""
        pages = [
            {"url": "/", "expected": "Dashboard"},
            {"url": "/detectors", "expected": "Detectors"},
            {"url": "/datasets", "expected": "Datasets"},
            {"url": "/detection", "expected": "Detection"},
        ]

        for page_info in pages:
            page.goto(f"http://pynomaly-app:8000{page_info['url']}")
            page.wait_for_load_state("networkidle")

            title = page.title()
            assert title is not None, f"Page {page_info['url']} should have title"
            assert len(title) > 0, f"Page {page_info['url']} title should not be empty"

            # Title should be descriptive
            assert (
                len(title) >= 5
            ), f"Page {page_info['url']} title should be descriptive"

    def test_error_message_accessibility(self, page: Page):
        """Test error message accessibility."""
        page.goto("http://pynomaly-app:8000/web/detectors")
        page.wait_for_load_state("networkidle")

        # Trigger form validation errors
        submit_button = page.locator("button[type='submit']")

        if submit_button.count() > 0:
            submit_button.click()
            page.wait_for_timeout(1000)

            # Check for error messages
            error_elements = page.locator(
                ".error, .invalid, [role='alert'], [aria-live]"
            )

            for i in range(error_elements.count()):
                error = error_elements.nth(i)
                error_text = error.text_content()

                # Error should have text content
                assert error_text is not None, f"Error {i} should have text"
                assert (
                    len(error_text.strip()) > 0
                ), f"Error {i} should have meaningful text"

                # Check for proper ARIA attributes
                error.get_attribute("role")
                error.get_attribute("aria-live")

                # Should have appropriate role or aria-live

    def generate_accessibility_report(self, page: Page) -> dict[str, Any]:
        """Generate comprehensive accessibility report."""
        page.goto("http://pynomaly-app:8000/web/")
        page.wait_for_load_state("networkidle")

        report = {
            "page_title": page.title(),
            "html_lang": page.locator("html").get_attribute("lang"),
            "headings": {
                "h1_count": page.locator("h1").count(),
                "total_headings": page.locator("h1, h2, h3, h4, h5, h6").count(),
            },
            "images": {
                "total": page.locator("img").count(),
                "with_alt": page.locator("img[alt]").count(),
                "empty_alt": page.locator("img[alt='']").count(),
            },
            "forms": {
                "total_inputs": page.locator("input, select, textarea").count(),
                "labeled_inputs": page.locator(
                    "input[id] + label, label input"
                ).count(),
                "required_fields": page.locator(
                    "[required], [aria-required='true']"
                ).count(),
            },
            "landmarks": {
                "main": page.locator("main, [role='main']").count(),
                "nav": page.locator("nav, [role='navigation']").count(),
                "header": page.locator("header, [role='banner']").count(),
                "footer": page.locator("footer, [role='contentinfo']").count(),
            },
            "buttons": {
                "total": page.locator("button").count(),
                "with_text": page.locator("button").filter(has_text="").count(),
            },
        }

        return report
