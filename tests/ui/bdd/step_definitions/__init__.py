"""
BDD Step Definitions Module

This module contains all the step definitions for Pynomaly's BDD testing framework.
Step definitions implement the Gherkin scenarios defined in the feature files.

Modules:
- accessibility_steps.py: WCAG compliance and accessibility testing steps
- comprehensive_workflow_steps.py: Complete user workflow step implementations
- cross_browser_steps.py: Cross-browser compatibility testing steps
- performance_steps.py: Performance testing and Core Web Vitals steps

Usage:
    These modules are automatically imported by pytest-bdd when running tests.
    Step definitions use decorators from pytest_bdd to map to Gherkin steps.

Example:
    @given("I am on the homepage")
    def given_homepage(page):
        page.goto("/")
        
    @when("I click the login button")
    def when_click_login(page):
        page.click("button[data-testid='login']")
        
    @then("I should see the dashboard")
    def then_see_dashboard(page):
        expect(page.locator("h1")).to_contain_text("Dashboard")
"""

# Import all step definitions to make them available to pytest-bdd
from .accessibility_steps import *
from .comprehensive_workflow_steps import *
from .cross_browser_steps import *
from .performance_steps import *

__all__ = [
    # Re-export all step definitions for pytest-bdd discovery
    # The actual symbols are imported via star imports above
]