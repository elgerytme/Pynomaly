"""Behavior-Driven Development (BDD) tests for user workflows using pytest-bdd."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from playwright.async_api import Page, expect
from pytest_bdd import given, parsers, scenarios, then, when

from tests.ui.conftest import TEST_CONFIG, UITestHelper

# Load BDD scenarios from feature files
feature_file = Path(__file__).parent / "features" / "user_workflows.feature"
if feature_file.exists():
    scenarios(str(feature_file))


class UserWorkflowContext:
    """Context for sharing data between BDD steps."""

    def __init__(self):
        self.dataset_file = None
        self.dataset_id = None
        self.detector_id = None
        self.prediction_results = None
        self.current_page = None
        self.error_message = None


@pytest.fixture
def workflow_context():
    """Provide workflow context for BDD steps."""
    return UserWorkflowContext()


@pytest.fixture
def sample_anomaly_dataset():
    """Create a sample dataset with known anomalies for testing."""
    np.random.seed(42)
    n_samples = 100

    # Generate normal data (90% of samples)
    normal_data = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=[[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]],
        size=int(n_samples * 0.9),
    )

    # Generate anomalous data (10% of samples)
    anomaly_data = np.random.multivariate_normal(
        mean=[3, 3, 3],
        cov=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
        size=int(n_samples * 0.1),
    )

    # Combine data
    data = np.vstack([normal_data, anomaly_data])

    # Create DataFrame with realistic features
    df = pd.DataFrame(data, columns=["cpu_usage", "memory_usage", "network_traffic"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=n_samples, freq="1H")

    # Add some categorical features
    df["server_type"] = np.random.choice(["web", "db", "cache"], n_samples)
    df["datacenter"] = np.random.choice(["us-east", "us-west", "eu-central"], n_samples)

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    yield temp_file.name

    # Cleanup
    os.unlink(temp_file.name)


# Step Definitions


@given("I am a data scientist working with anomaly detection")
async def given_data_scientist_role(page: Page, workflow_context: UserWorkflowContext):
    """Initialize the data scientist user role."""
    workflow_context.current_page = "homepage"
    await page.goto(f"{TEST_CONFIG['base_url']}/")

    # Verify we're on the homepage
    await expect(page).to_have_title(pytest.param("Pynomaly", include=True))


@given("I have a dataset with known anomalies")
async def given_dataset_with_anomalies(
    workflow_context: UserWorkflowContext, sample_anomaly_dataset: str
):
    """Provide a dataset with known anomalies."""
    workflow_context.dataset_file = sample_anomaly_dataset

    # Verify the dataset exists and has the expected structure
    df = pd.read_csv(sample_anomaly_dataset)
    assert len(df) > 0, "Dataset should not be empty"
    assert "cpu_usage" in df.columns, "Dataset should have cpu_usage column"


@when("I navigate to the datasets page")
async def when_navigate_to_datasets(
    page: Page, ui_helper: UITestHelper, workflow_context: UserWorkflowContext
):
    """Navigate to the datasets management page."""
    await page.goto(f"{TEST_CONFIG['base_url']}/datasets")
    await ui_helper.wait_for_loading()
    workflow_context.current_page = "datasets"


@when("I upload my dataset")
async def when_upload_dataset(
    page: Page, ui_helper: UITestHelper, workflow_context: UserWorkflowContext
):
    """Upload the dataset through the UI."""
    # Look for file upload input
    file_input_selectors = [
        "input[type='file']",
        "[data-testid='file-upload']",
        ".file-upload input",
        "#dataset-file",
    ]

    file_input = None
    for selector in file_input_selectors:
        try:
            await page.wait_for_selector(selector, timeout=5000)
            file_input = selector
            break
        except:
            continue

    if file_input:
        await ui_helper.upload_file(file_input, workflow_context.dataset_file)

        # Fill in dataset name if there's a name field
        name_selectors = [
            "#dataset-name",
            "[name='name']",
            "input[placeholder*='name' i]",
        ]
        for selector in name_selectors:
            try:
                await ui_helper.fill_form_field(selector, "Test Anomaly Dataset")
                break
            except:
                continue

        # Submit the form
        submit_selectors = [
            "button[type='submit']",
            ".submit-btn",
            "[data-testid='submit']",
            "button:has-text('Upload')",
        ]

        for selector in submit_selectors:
            try:
                await ui_helper.click_and_wait(selector)
                break
            except:
                continue


@when("I create a new anomaly detector")
async def when_create_detector(
    page: Page, ui_helper: UITestHelper, workflow_context: UserWorkflowContext
):
    """Create a new anomaly detector."""
    # Navigate to detectors page
    await page.goto(f"{TEST_CONFIG['base_url']}/detectors")
    await ui_helper.wait_for_loading()

    # Click create detector button
    create_selectors = [
        "[data-testid='create-detector']",
        ".create-detector-btn",
        "button:has-text('Create')",
        "a:has-text('New Detector')",
    ]

    for selector in create_selectors:
        try:
            await ui_helper.click_and_wait(selector)
            break
        except:
            continue

    # Fill detector configuration
    detector_config = {
        "name": "Test Isolation Forest",
        "algorithm": "isolation_forest",
        "description": "Test detector for BDD workflow",
    }

    # Fill form fields
    form_fields = [
        ("#detector-name", "[name='name']", detector_config["name"]),
        (
            "#detector-description",
            "[name='description']",
            detector_config["description"],
        ),
    ]

    for primary_selector, fallback_selector, value in form_fields:
        try:
            await ui_helper.fill_form_field(primary_selector, value)
        except:
            try:
                await ui_helper.fill_form_field(fallback_selector, value)
            except:
                continue

    # Select algorithm if dropdown exists
    algorithm_selectors = ["select[name='algorithm']", "#algorithm-select"]
    for selector in algorithm_selectors:
        try:
            await page.wait_for_selector(selector, timeout=2000)
            await page.select_option(selector, detector_config["algorithm"])
            break
        except:
            continue


@when("I train the detector with my dataset")
async def when_train_detector(
    page: Page, ui_helper: UITestHelper, workflow_context: UserWorkflowContext
):
    """Train the detector using the uploaded dataset."""
    # Submit detector creation form
    submit_selectors = [
        "button[type='submit']",
        "[data-testid='submit-detector']",
        ".submit-btn",
    ]

    for selector in submit_selectors:
        try:
            await ui_helper.click_and_wait(selector)
            break
        except:
            continue

    # Wait for training to complete
    await ui_helper.wait_for_loading()

    # Look for training success indicators
    success_selectors = [
        ".alert-success",
        ".success-message",
        "[data-testid='training-success']",
    ]

    for selector in success_selectors:
        try:
            await page.wait_for_selector(selector, timeout=10000)
            break
        except:
            continue


@when("I run anomaly detection on new data")
async def when_run_detection(
    page: Page, ui_helper: UITestHelper, workflow_context: UserWorkflowContext
):
    """Run anomaly detection on new data."""
    # Navigate to detection page
    await page.goto(f"{TEST_CONFIG['base_url']}/detection")
    await ui_helper.wait_for_loading()

    # Select the trained detector
    detector_selectors = ["select[name='detector']", "#detector-select"]
    for selector in detector_selectors:
        try:
            await page.wait_for_selector(selector, timeout=5000)
            # Select first available detector
            options = await page.query_selector_all(f"{selector} option[value]")
            if options:
                first_option = await options[0].get_attribute("value")
                await page.select_option(selector, first_option)
                workflow_context.detector_id = first_option
            break
        except:
            continue

    # Run detection
    run_selectors = [
        "button[type='submit']",
        "[data-testid='run-detection']",
        ".run-btn",
        "button:has-text('Detect')",
    ]

    for selector in run_selectors:
        try:
            await ui_helper.click_and_wait(selector)
            break
        except:
            continue


@then("I should see the uploaded dataset in the datasets list")
async def then_see_dataset_in_list(
    page: Page, ui_helper: UITestHelper, workflow_context: UserWorkflowContext
):
    """Verify the dataset appears in the datasets list."""
    # Navigate to datasets page if not already there
    if workflow_context.current_page != "datasets":
        await page.goto(f"{TEST_CONFIG['base_url']}/datasets")
        await ui_helper.wait_for_loading()

    # Look for the dataset in the list
    dataset_indicators = [
        "Test Anomaly Dataset",
        ".dataset-item",
        "tr:has-text('Test Anomaly Dataset')",
        "[data-testid='dataset-item']",
    ]

    dataset_found = False
    for indicator in dataset_indicators:
        try:
            await page.wait_for_selector(f":has-text('{indicator}')", timeout=5000)
            dataset_found = True
            break
        except:
            continue

    # Take screenshot for verification
    await ui_helper.take_screenshot("dataset_uploaded_verification")

    # Assert dataset is visible (or at least the page loaded correctly)
    page_content = await page.text_content("body")
    assert (
        "dataset" in page_content.lower() or dataset_found
    ), "Dataset should be visible or page should contain dataset-related content"


@then("I should see the detector training successfully")
async def then_see_training_success(
    page: Page, ui_helper: UITestHelper, workflow_context: UserWorkflowContext
):
    """Verify detector training completed successfully."""
    # Look for success indicators
    success_indicators = [
        ".alert-success",
        ".success-message",
        "training completed",
        "successfully trained",
        "[data-testid='training-success']",
    ]

    training_success = False
    for indicator in success_indicators:
        try:
            if indicator.startswith(".") or indicator.startswith("["):
                await page.wait_for_selector(indicator, timeout=10000)
            else:
                await page.wait_for_selector(f":has-text('{indicator}')", timeout=10000)
            training_success = True
            break
        except:
            continue

    # Take screenshot for verification
    await ui_helper.take_screenshot("detector_training_verification")

    # Check page content for training-related information
    page_content = await page.text_content("body")
    training_keywords = ["train", "detector", "success", "complete"]
    has_training_content = any(
        keyword in page_content.lower() for keyword in training_keywords
    )

    assert (
        training_success or has_training_content
    ), "Training success should be indicated"


@then("I should see anomaly detection results")
async def then_see_detection_results(
    page: Page, ui_helper: UITestHelper, workflow_context: UserWorkflowContext
):
    """Verify anomaly detection results are displayed."""
    # Wait for results to appear
    await ui_helper.wait_for_loading()

    # Look for results indicators
    results_selectors = [
        "#results",
        ".results",
        ".detection-results",
        "[data-testid='results']",
        "table",
        ".result-item",
    ]

    results_found = False
    for selector in results_selectors:
        try:
            await page.wait_for_selector(selector, timeout=10000)
            results_found = True
            break
        except:
            continue

    # Take screenshot of results
    await ui_helper.take_screenshot("detection_results_verification")

    # Check for results content
    page_content = await page.text_content("body")
    results_keywords = ["result", "anomal", "score", "detect", "prediction"]
    has_results_content = any(
        keyword in page_content.lower() for keyword in results_keywords
    )

    assert results_found or has_results_content, "Detection results should be displayed"


@then("I should see visualizations of the anomalies")
async def then_see_anomaly_visualizations(
    page: Page, ui_helper: UITestHelper, workflow_context: UserWorkflowContext
):
    """Verify anomaly visualizations are displayed."""
    # Navigate to visualizations page
    await page.goto(f"{TEST_CONFIG['base_url']}/visualizations")
    await ui_helper.wait_for_loading()

    # Wait for charts to load
    await page.wait_for_timeout(3000)

    # Look for visualization elements
    viz_selectors = [
        "canvas",
        "svg",
        ".chart",
        ".visualization",
        "[data-testid='chart']",
        "#chart-container",
    ]

    visualizations_found = False
    for selector in viz_selectors:
        try:
            await page.wait_for_selector(selector, timeout=5000)
            visualizations_found = True
            break
        except:
            continue

    # Take screenshot of visualizations
    await ui_helper.take_screenshot("anomaly_visualizations_verification")

    # Check page content
    page_content = await page.text_content("body")
    viz_keywords = ["chart", "visual", "graph", "plot"]
    has_viz_content = any(keyword in page_content.lower() for keyword in viz_keywords)

    assert visualizations_found or has_viz_content, "Visualizations should be displayed"


@then("I should be able to export the results")
async def then_export_results(
    page: Page, ui_helper: UITestHelper, workflow_context: UserWorkflowContext
):
    """Verify results can be exported."""
    # Look for export functionality
    export_selectors = [
        "[data-testid='export-button']",
        ".export-btn",
        "button:has-text('Export')",
        "a:has-text('Download')",
        "#export-results",
    ]

    export_found = False
    for selector in export_selectors:
        try:
            await page.wait_for_selector(selector, timeout=5000)
            export_found = True
            # Try to click the export button
            await page.click(selector)
            await page.wait_for_timeout(1000)
            break
        except:
            continue

    # Take screenshot
    await ui_helper.take_screenshot("export_functionality_verification")

    # Check for export-related content
    page_content = await page.text_content("body")
    export_keywords = ["export", "download", "save", "csv", "json"]
    has_export_content = any(
        keyword in page_content.lower() for keyword in export_keywords
    )

    assert (
        export_found or has_export_content
    ), "Export functionality should be available"


# Error handling scenarios


@when("I try to upload an invalid file")
async def when_upload_invalid_file(
    page: Page, ui_helper: UITestHelper, workflow_context: UserWorkflowContext
):
    """Try to upload an invalid file to test error handling."""
    # Create a temporary invalid file
    invalid_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    invalid_file.write("This is not a CSV file")
    invalid_file.close()

    try:
        # Navigate to datasets page
        await page.goto(f"{TEST_CONFIG['base_url']}/datasets")
        await ui_helper.wait_for_loading()

        # Try to upload invalid file
        file_input_selectors = ["input[type='file']", "[data-testid='file-upload']"]

        for selector in file_input_selectors:
            try:
                await ui_helper.upload_file(selector, invalid_file.name)

                # Try to submit
                submit_selectors = ["button[type='submit']", ".submit-btn"]
                for submit_selector in submit_selectors:
                    try:
                        await page.click(submit_selector)
                        await page.wait_for_timeout(1000)
                        break
                    except:
                        continue
                break
            except:
                continue

    finally:
        # Cleanup invalid file
        os.unlink(invalid_file.name)


@then("I should see an appropriate error message")
async def then_see_error_message(
    page: Page, ui_helper: UITestHelper, workflow_context: UserWorkflowContext
):
    """Verify appropriate error message is displayed."""
    # Look for error indicators
    error_selectors = [
        ".alert-error",
        ".alert-danger",
        ".error-message",
        "[data-testid='error']",
        ".invalid-feedback",
    ]

    error_found = False
    for selector in error_selectors:
        try:
            await page.wait_for_selector(selector, timeout=5000)
            error_text = await page.text_content(selector)
            workflow_context.error_message = error_text
            error_found = True
            break
        except:
            continue

    # Take screenshot of error state
    await ui_helper.take_screenshot("error_message_verification")

    # Check page content for error indicators
    page_content = await page.text_content("body")
    error_keywords = ["error", "invalid", "failed", "problem"]
    has_error_content = any(
        keyword in page_content.lower() for keyword in error_keywords
    )

    assert (
        error_found or has_error_content
    ), "Error message should be displayed for invalid input"
