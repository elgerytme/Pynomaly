"""
Simple UI Testing Framework Test

Basic test to verify the advanced UI testing infrastructure works
"""

import pytest

from tests.ui.advanced.utils.accessibility_validator import AccessibilityValidator
from tests.ui.advanced.utils.cross_browser_manager import CrossBrowserManager
from tests.ui.advanced.utils.test_data_generator import TestDataGenerator


@pytest.mark.ui
def test_accessibility_validator_creation():
    """Test that AccessibilityValidator can be created"""
    validator = AccessibilityValidator()
    assert validator is not None
    assert hasattr(validator, "wcag_rules")
    assert "color_contrast" in validator.wcag_rules


@pytest.mark.ui
def test_cross_browser_manager_creation():
    """Test that CrossBrowserManager can be created"""
    manager = CrossBrowserManager()
    assert manager is not None
    assert hasattr(manager, "browser_configs")
    assert "chromium" in manager.browser_configs


@pytest.mark.ui
def test_test_data_generator_creation():
    """Test that TestDataGenerator can be created and generate data"""
    generator = TestDataGenerator()
    assert generator is not None

    # Test user data generation
    user_data = generator.generate_user_data("data_scientist")
    assert user_data is not None
    assert "id" in user_data
    assert "username" in user_data
    assert "email" in user_data
    assert user_data["role"] == "Data Scientist"


@pytest.mark.ui
def test_form_data_generation():
    """Test form data generation capabilities"""
    generator = TestDataGenerator()

    # Test login form data
    login_data = generator.generate_form_test_data("login")
    assert "valid" in login_data
    assert "invalid_email" in login_data
    assert "username" in login_data["valid"]
    assert "password" in login_data["valid"]


@pytest.mark.ui
def test_performance_scenarios():
    """Test performance test scenario generation"""
    generator = TestDataGenerator()

    scenarios = generator.generate_performance_test_scenarios()
    assert len(scenarios) > 0

    light_load = scenarios[0]
    assert light_load["name"] == "Light Load"
    assert "concurrent_users" in light_load
    assert "duration_seconds" in light_load


@pytest.mark.ui
def test_accessibility_scenarios():
    """Test accessibility test scenario generation"""
    generator = TestDataGenerator()

    scenarios = generator.generate_accessibility_test_scenarios()
    assert len(scenarios) > 0

    keyboard_scenario = scenarios[0]
    assert keyboard_scenario["name"] == "Keyboard Only Navigation"
    assert "assistive_technology" in keyboard_scenario
    assert "test_actions" in keyboard_scenario
