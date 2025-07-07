"""Comprehensive BDD Step Definitions for All User Workflows."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from playwright.sync_api import Page, expect
from pytest_bdd import given, parsers, then, when

from tests.ui.enhanced_page_objects.base_page import BasePage

# Test data and configuration
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "test_data"
MOCK_DATA = {
    "financial_fraud": {
        "filename": "financial_fraud_detection.csv",
        "records": 50000,
        "features": 15,
        "fraud_rate": 0.02,
    },
    "network_traffic": {
        "filename": "network_traffic_security.csv",
        "records": 100000,
        "features": 25,
        "anomaly_rate": 0.05,
    },
    "production_model": {
        "name": "FraudDetection_v2.1",
        "algorithm": "Isolation Forest",
        "accuracy": 0.91,
        "version": "2.1",
    },
}

# Global state for maintaining context across steps
test_context = {}


# ================================
# Background and Setup Steps
# ================================


@given("I am a data scientist with expertise in anomaly detection")
def setup_data_scientist_persona(page: Page):
    """Setup data scientist user persona."""
    test_context["user_role"] = "data_scientist"
    test_context["permissions"] = ["read_data", "create_models", "run_analysis"]
    test_context["page"] = page


@given("I am a security analyst responsible for threat detection")
def setup_security_analyst_persona(page: Page):
    """Setup security analyst user persona."""
    test_context["user_role"] = "security_analyst"
    test_context["permissions"] = [
        "monitor_security",
        "investigate_threats",
        "manage_alerts",
    ]
    test_context["page"] = page


@given("I am an ML engineer responsible for production ML systems")
def setup_ml_engineer_persona(page: Page):
    """Setup ML engineer user persona."""
    test_context["user_role"] = "ml_engineer"
    test_context["permissions"] = [
        "deploy_models",
        "manage_infrastructure",
        "monitor_production",
    ]
    test_context["page"] = page


@given("the Pynomaly web application is running and accessible")
def verify_application_running(page: Page):
    """Verify the application is accessible."""
    page.goto("/")
    expect(page).to_have_title("Pynomaly - Anomaly Detection Platform")
    test_context["app_available"] = True


@given("I have appropriate permissions for data analysis")
@given("I have appropriate security clearance and permissions")
@given("I have appropriate deployment permissions and infrastructure access")
def verify_user_permissions():
    """Verify user has appropriate permissions."""
    assert test_context.get("permissions"), "User permissions not configured"
    test_context["permissions_verified"] = True


# ================================
# Data Scientist Workflow Steps
# ================================


@given("I have a financial transactions dataset with known fraud cases")
def setup_financial_dataset():
    """Setup financial fraud dataset context."""
    test_context["dataset"] = MOCK_DATA["financial_fraud"]
    test_context["dataset"]["has_labels"] = True


@given(
    parsers.parse(
        "the dataset contains {records:d} transactions with {fraud_rate:g}% fraud rate"
    )
)
def verify_dataset_specifications(records: int, fraud_rate: float):
    """Verify dataset meets specifications."""
    test_context["dataset"]["expected_records"] = records
    test_context["dataset"]["expected_fraud_rate"] = fraud_rate


@when("I navigate to the datasets page")
def navigate_to_datasets(page: Page):
    """Navigate to datasets page."""
    page.goto("/datasets")
    expect(page.locator("h1")).to_contain_text("Datasets")


@when(parsers.parse('I upload the dataset named "{filename}"'))
def upload_dataset(page: Page, filename: str):
    """Upload a dataset file."""
    # Mock file upload by clicking upload button and simulating success
    page.click("[data-testid='upload-dataset-btn']")

    # Simulate file selection (in real test, would use file input)
    page.fill("[data-testid='dataset-name-input']", filename)
    page.click("[data-testid='upload-confirm-btn']")

    test_context["uploaded_filename"] = filename


@then("I should see the upload progress indicator")
def verify_upload_progress(page: Page):
    """Verify upload progress is shown."""
    expect(page.locator("[data-testid='upload-progress']")).to_be_visible()


@then(parsers.parse('I should see "{message}" message within {timeout:d} seconds'))
def verify_success_message(page: Page, message: str, timeout: int):
    """Verify success message appears within timeout."""
    expect(page.locator(f"text={message}")).to_be_visible(timeout=timeout * 1000)


@then(parsers.parse('I should see the dataset "{filename}" in the datasets list'))
def verify_dataset_in_list(page: Page, filename: str):
    """Verify dataset appears in the list."""
    expect(page.locator(f"[data-testid='dataset-{filename}']")).to_be_visible()


@then(parsers.parse('I should see dataset statistics showing "{stats}"'))
def verify_dataset_statistics(page: Page, stats: str):
    """Verify dataset statistics are displayed."""
    expect(page.locator("[data-testid='dataset-stats']")).to_contain_text(stats)


@when('I click on "View Dataset Details"')
def click_view_dataset_details(page: Page):
    """Click on view dataset details."""
    page.click("[data-testid='view-dataset-details']")


@then("I should see the dataset preview with first 10 records")
def verify_dataset_preview(page: Page):
    """Verify dataset preview is shown."""
    expect(page.locator("[data-testid='dataset-preview']")).to_be_visible()
    expect(page.locator("[data-testid='preview-table'] tbody tr")).to_have_count(10)


@then("I should see feature distribution charts")
def verify_feature_charts(page: Page):
    """Verify feature distribution charts are shown."""
    expect(page.locator("[data-testid='feature-charts']")).to_be_visible()


@when("I navigate to the detectors page")
def navigate_to_detectors(page: Page):
    """Navigate to detectors page."""
    page.goto("/detectors")
    expect(page.locator("h1")).to_contain_text("Detectors")


@when('I click "Create New Detector"')
def click_create_detector(page: Page):
    """Click create new detector button."""
    page.click("[data-testid='create-detector-btn']")
    expect(page.locator("[data-testid='detector-form']")).to_be_visible()


@when(parsers.parse('I fill in the detector name as "{name}"'))
def fill_detector_name(page: Page, name: str):
    """Fill in detector name."""
    page.fill("[data-testid='detector-name-input']", name)
    test_context["detector_name"] = name


@when(parsers.parse('I select "{algorithm}" algorithm'))
def select_algorithm(page: Page, algorithm: str):
    """Select detection algorithm."""
    page.select_option("[data-testid='algorithm-select']", algorithm)
    test_context["selected_algorithm"] = algorithm


@when(parsers.parse('I select the dataset "{dataset_name}"'))
def select_dataset(page: Page, dataset_name: str):
    """Select dataset for training."""
    page.select_option("[data-testid='dataset-select']", dataset_name)


@when(parsers.parse("I configure contamination rate to {rate:g}"))
def configure_contamination_rate(page: Page, rate: float):
    """Configure contamination rate."""
    page.fill("[data-testid='contamination-input']", str(rate))


@when('I click "Create Detector"')
def click_create_detector_submit(page: Page):
    """Submit detector creation form."""
    page.click("[data-testid='create-detector-submit']")


@then(parsers.parse('I should see the detector "{name}" in the detectors list'))
def verify_detector_in_list(page: Page, name: str):
    """Verify detector appears in list."""
    expect(page.locator(f"[data-testid='detector-{name}']")).to_be_visible()


@when(parsers.parse('I click "Train Detector" for "{detector_name}"'))
def click_train_detector(page: Page, detector_name: str):
    """Click train detector button."""
    page.click(f"[data-testid='train-{detector_name}']")


@then("I should see training progress indicator")
def verify_training_progress(page: Page):
    """Verify training progress is shown."""
    expect(page.locator("[data-testid='training-progress']")).to_be_visible()


@then('I should see "Training in progress..." status')
def verify_training_status(page: Page):
    """Verify training status message."""
    expect(page.locator("[data-testid='training-status']")).to_contain_text(
        "Training in progress"
    )


@then(parsers.parse("training should complete within {timeout:d} seconds"))
def verify_training_completion(page: Page, timeout: int):
    """Verify training completes within timeout."""
    expect(page.locator("[data-testid='training-complete']")).to_be_visible(
        timeout=timeout * 1000
    )


@then("I should see training metrics including accuracy and F1-score")
def verify_training_metrics(page: Page):
    """Verify training metrics are displayed."""
    expect(page.locator("[data-testid='training-metrics']")).to_be_visible()
    expect(page.locator("[data-testid='accuracy-metric']")).to_be_visible()
    expect(page.locator("[data-testid='f1-score-metric']")).to_be_visible()


# ================================
# Security Analyst Workflow Steps
# ================================


@given("I have network traffic data streaming from security sensors")
def setup_network_traffic_stream():
    """Setup network traffic streaming context."""
    test_context["traffic_stream"] = {
        "active": True,
        "sources": ["firewall", "ids", "network_tap"],
        "data_rate": "10MB/sec",
    }


@given("I have configured baseline traffic patterns")
def setup_baseline_patterns():
    """Setup baseline traffic patterns."""
    test_context["baseline"] = {
        "established": True,
        "learning_period": "30 days",
        "patterns": ["normal_business_hours", "weekend_patterns", "holiday_patterns"],
    }


@when("I navigate to the security monitoring dashboard")
def navigate_to_security_dashboard(page: Page):
    """Navigate to security monitoring dashboard."""
    page.goto("/security/dashboard")
    expect(page.locator("h1")).to_contain_text("Security Monitoring")


@then("I should see real-time network traffic visualization")
def verify_traffic_visualization(page: Page):
    """Verify traffic visualization is shown."""
    expect(page.locator("[data-testid='traffic-viz']")).to_be_visible()


@then("I should see current threat level indicators")
def verify_threat_indicators(page: Page):
    """Verify threat level indicators."""
    expect(page.locator("[data-testid='threat-level']")).to_be_visible()


@when("I configure anomaly detection for network monitoring")
def configure_network_anomaly_detection(page: Page):
    """Configure network anomaly detection."""
    page.click("[data-testid='configure-detection-btn']")
    expect(page.locator("[data-testid='detection-config-form']")).to_be_visible()


@when("I activate real-time monitoring")
def activate_realtime_monitoring(page: Page):
    """Activate real-time monitoring."""
    page.click("[data-testid='activate-monitoring-btn']")


@then('I should see "Real-time monitoring active" status')
def verify_monitoring_active(page: Page):
    """Verify monitoring is active."""
    expect(page.locator("[data-testid='monitoring-status']")).to_contain_text(
        "Real-time monitoring active"
    )


@when("a suspicious network activity occurs")
def simulate_suspicious_activity():
    """Simulate suspicious network activity."""
    test_context["suspicious_activity"] = {
        "type": "data_exfiltration",
        "severity": "high",
        "source_ip": "192.168.1.100",
        "destination": "suspicious.external.com",
    }


@then(
    parsers.parse(
        "I should receive an immediate security alert within {timeout:d} seconds"
    )
)
def verify_security_alert(page: Page, timeout: int):
    """Verify security alert is received."""
    expect(page.locator("[data-testid='security-alert']")).to_be_visible(
        timeout=timeout * 1000
    )


@then("I should see the anomalous traffic highlighted in red")
def verify_anomalous_traffic_highlight(page: Page):
    """Verify anomalous traffic is highlighted."""
    expect(
        page.locator("[data-testid='anomalous-traffic'].highlighted")
    ).to_be_visible()


@then("I should see threat severity classification")
def verify_threat_severity(page: Page):
    """Verify threat severity is shown."""
    expect(page.locator("[data-testid='threat-severity']")).to_be_visible()


# ================================
# ML Engineer Workflow Steps
# ================================


@given("I have a trained anomaly detection model ready for production")
def setup_production_model():
    """Setup production-ready model context."""
    test_context["production_model"] = MOCK_DATA["production_model"]
    test_context["production_model"]["ready_for_deployment"] = True


@given("the model has passed validation tests")
def verify_model_validation():
    """Verify model validation status."""
    test_context["production_model"]["validation_passed"] = True


@when("I navigate to the deployment dashboard")
def navigate_to_deployment_dashboard(page: Page):
    """Navigate to deployment dashboard."""
    page.goto("/deployment/dashboard")
    expect(page.locator("h1")).to_contain_text("Model Deployment")


@then("I should see available models for deployment")
def verify_available_models(page: Page):
    """Verify available models are shown."""
    expect(page.locator("[data-testid='available-models']")).to_be_visible()


@when(parsers.parse('I select the model "{model_name}" for deployment'))
def select_model_for_deployment(page: Page, model_name: str):
    """Select model for deployment."""
    page.click(f"[data-testid='select-model-{model_name}']")
    test_context["selected_model"] = model_name


@when("I configure deployment settings")
def configure_deployment_settings(page: Page):
    """Configure deployment settings."""
    page.click("[data-testid='configure-deployment-btn']")
    expect(page.locator("[data-testid='deployment-config-form']")).to_be_visible()


@when('I click "Deploy Model"')
def click_deploy_model(page: Page):
    """Click deploy model button."""
    page.click("[data-testid='deploy-model-btn']")


@then("I should see deployment progress indicator")
def verify_deployment_progress(page: Page):
    """Verify deployment progress is shown."""
    expect(page.locator("[data-testid='deployment-progress']")).to_be_visible()


@then("I should see container building status")
def verify_container_building(page: Page):
    """Verify container building status."""
    expect(page.locator("[data-testid='container-status']")).to_contain_text("Building")


@then(parsers.parse("deployment should complete within {timeout:d} minutes"))
def verify_deployment_completion(page: Page, timeout: int):
    """Verify deployment completes within timeout."""
    expect(page.locator("[data-testid='deployment-complete']")).to_be_visible(
        timeout=timeout * 60000
    )


@when("deployment completes")
def when_deployment_completes():
    """Set deployment completion context."""
    test_context["deployment_completed"] = True


@then("I should see live model endpoint URL")
def verify_model_endpoint(page: Page):
    """Verify model endpoint URL is shown."""
    expect(page.locator("[data-testid='model-endpoint']")).to_be_visible()


# ================================
# Common Workflow Steps
# ================================


@when("I navigate to the visualizations page")
def navigate_to_visualizations(page: Page):
    """Navigate to visualizations page."""
    page.goto("/visualizations")
    expect(page.locator("h1")).to_contain_text("Visualizations")


@then("I should see scatter plot of anomaly scores")
def verify_scatter_plot(page: Page):
    """Verify scatter plot is displayed."""
    expect(page.locator("[data-testid='scatter-plot']")).to_be_visible()


@when("I navigate to the export page")
def navigate_to_export(page: Page):
    """Navigate to export page."""
    page.goto("/export")
    expect(page.locator("h1")).to_contain_text("Export Results")


@when("I select CSV format")
def select_csv_format(page: Page):
    """Select CSV export format."""
    page.check("[data-testid='export-format-csv']")


@when('I click "Export Results"')
def click_export_results(page: Page):
    """Click export results button."""
    page.click("[data-testid='export-results-btn']")


@then(parsers.parse('I should receive a download of "{filename}"'))
def verify_file_download(page: Page, filename: str):
    """Verify file download occurs."""
    # In a real test, this would check for actual file download
    expect(page.locator("[data-testid='download-success']")).to_be_visible()
    test_context["downloaded_file"] = filename


# ================================
# Table-driven Step Definitions
# ================================


@when("I create multiple detectors with different algorithms")
def create_multiple_detectors(page: Page, datatable):
    """Create multiple detectors based on data table."""
    for row in datatable:
        algorithm = row["Algorithm"]
        contamination = float(row["Contamination"])
        features = row["Features"]

        # Navigate and create detector
        page.goto("/detectors")
        page.click("[data-testid='create-detector-btn']")
        page.fill("[data-testid='detector-name-input']", f"Detector_{algorithm}")
        page.select_option("[data-testid='algorithm-select']", algorithm)
        page.fill("[data-testid='contamination-input']", str(contamination))
        page.click("[data-testid='create-detector-submit']")

        # Wait for creation to complete
        expect(page.locator("[data-testid='detector-created']")).to_be_visible()


@when("I configure streaming parameters")
def configure_streaming_parameters(page: Page, datatable):
    """Configure streaming parameters based on data table."""
    page.click("[data-testid='configure-streaming-btn']")

    for row in datatable:
        parameter = row["Parameter"]
        value = row["Value"]

        # Map parameter to form field
        field_map = {
            "Buffer Size": "buffer-size-input",
            "Update Frequency": "update-frequency-input",
            "Alert Threshold": "alert-threshold-input",
        }

        if parameter in field_map:
            page.fill(f"[data-testid='{field_map[parameter]}']", value)


@when("I configure deployment settings")
def configure_deployment_settings_table(page: Page, datatable):
    """Configure deployment settings from data table."""
    page.click("[data-testid='deployment-settings-btn']")

    for row in datatable:
        parameter = row["Parameter"]
        value = row["Value"]

        # Map parameters to form fields
        field_map = {
            "Environment": "environment-select",
            "Scaling": "scaling-config",
            "Resource Limits": "resource-limits",
            "Timeout": "timeout-input",
            "Circuit Breaker": "circuit-breaker-toggle",
            "Rate Limiting": "rate-limit-input",
        }

        if parameter in field_map:
            if "select" in field_map[parameter]:
                page.select_option(f"[data-testid='{field_map[parameter]}']", value)
            elif "toggle" in field_map[parameter]:
                if value.lower() == "enabled":
                    page.check(f"[data-testid='{field_map[parameter]}']")
            else:
                page.fill(f"[data-testid='{field_map[parameter]}']", value)


# ================================
# Error Handling and Edge Cases
# ================================


@when("I try to upload an invalid file")
def upload_invalid_file(page: Page):
    """Attempt to upload an invalid file."""
    page.click("[data-testid='upload-dataset-btn']")
    page.fill("[data-testid='dataset-name-input']", "invalid_file.txt")
    page.click("[data-testid='upload-confirm-btn']")


@then("I should see an appropriate error message")
def verify_error_message(page: Page):
    """Verify appropriate error message is shown."""
    expect(page.locator("[data-testid='error-message']")).to_be_visible()


@then("the system should remain stable")
def verify_system_stability(page: Page):
    """Verify system remains stable after error."""
    # Check that main navigation is still functional
    expect(page.locator("[data-testid='main-nav']")).to_be_visible()

    # Check that we can still navigate to other pages
    page.click("[data-testid='nav-dashboard']")
    expect(page.locator("h1")).to_contain_text("Dashboard")


# ================================
# Performance and Accessibility Steps
# ================================


@given("I am using a mobile device")
def setup_mobile_device(page: Page):
    """Setup mobile device viewport."""
    page.set_viewport_size({"width": 375, "height": 667})
    test_context["device_type"] = "mobile"


@then("all pages should be mobile-friendly")
def verify_mobile_friendly(page: Page):
    """Verify pages are mobile-friendly."""
    # Check that content fits within mobile viewport
    body_width = page.evaluate("document.body.scrollWidth")
    viewport_width = page.evaluate("window.innerWidth")
    assert body_width <= viewport_width, "Content exceeds mobile viewport width"


@given("I am using assistive technology")
def setup_assistive_technology():
    """Setup assistive technology context."""
    test_context["assistive_tech"] = True
    test_context["screen_reader"] = True


@then("all content should be accessible via screen reader")
def verify_screen_reader_accessibility(page: Page):
    """Verify content is accessible to screen readers."""
    # Check for proper heading structure
    headings = page.locator("h1, h2, h3, h4, h5, h6").all()
    assert len(headings) > 0, "No headings found for screen reader navigation"

    # Check for alt text on images
    images = page.locator("img").all()
    for img in images:
        alt_text = img.get_attribute("alt")
        assert alt_text is not None, "Image missing alt text"


@then("all interactive elements should be keyboard accessible")
def verify_keyboard_accessibility(page: Page):
    """Verify keyboard accessibility."""
    # Test tab navigation
    focusable_elements = page.locator(
        "a, button, input, select, textarea, [tabindex]:not([tabindex='-1'])"
    ).all()

    for element in focusable_elements[:5]:  # Test first 5 elements
        element.focus()
        focused_element = page.evaluate("document.activeElement")
        assert focused_element is not None, "Element not keyboard focusable"


# ================================
# Performance Testing Steps
# ================================


@given(parsers.parse("I have a large dataset ({size} records)"))
def setup_large_dataset(size: str):
    """Setup large dataset context."""
    # Parse size (e.g., "10k+", "1 million")
    if "k" in size:
        record_count = int(size.replace("k+", "000"))
    elif "million" in size:
        record_count = int(size.replace(" million", "000000"))
    else:
        record_count = int(size)

    test_context["large_dataset"] = {"size": record_count, "performance_test": True}


@then(parsers.parse("the upload should complete within reasonable time"))
def verify_upload_performance(page: Page):
    """Verify upload completes within reasonable time."""
    # For large datasets, allow up to 5 minutes
    expect(page.locator("[data-testid='upload-complete']")).to_be_visible(
        timeout=300000
    )


@then("the interface should remain responsive")
def verify_interface_responsiveness(page: Page):
    """Verify interface remains responsive during operations."""
    # Test that navigation still works
    start_time = time.time()
    page.click("[data-testid='nav-dashboard']")
    end_time = time.time()

    response_time = end_time - start_time
    assert response_time < 2.0, f"Interface response time too slow: {response_time}s"


# ================================
# Utility Functions
# ================================


def wait_for_element_with_retry(page: Page, selector: str, timeout: int = 30):
    """Wait for element with retry logic."""
    for attempt in range(3):
        try:
            expect(page.locator(selector)).to_be_visible(timeout=timeout * 1000)
            return
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(1)


def verify_data_table_structure(page: Page, expected_columns: List[str]):
    """Verify data table has expected structure."""
    for column in expected_columns:
        expect(page.locator(f"th:has-text('{column}')")).to_be_visible()


def simulate_api_response(response_data: Dict[str, Any]):
    """Simulate API response for testing."""
    test_context["mock_api_response"] = response_data


def cleanup_test_context():
    """Clean up test context between scenarios."""
    test_context.clear()


# ================================
# Hooks for Test Management
# ================================


@pytest.fixture(autouse=True)
def setup_test_context():
    """Setup test context before each test."""
    test_context.clear()
    yield
    cleanup_test_context()


@pytest.fixture
def mock_external_services():
    """Mock external services for testing."""
    # Mock threat intelligence feeds
    test_context["threat_feeds"] = {
        "virustotal": {"available": True, "api_key": "mock_key"},
        "alienqueue": {"available": True, "api_key": "mock_key"},
    }

    # Mock model registry
    test_context["model_registry"] = {
        "available": True,
        "models": [MOCK_DATA["production_model"]],
    }

    yield

    # Cleanup mocks
    test_context.pop("threat_feeds", None)
    test_context.pop("model_registry", None)
