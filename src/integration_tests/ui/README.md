# Enhanced UI Testing Framework

This directory contains a comprehensive UI testing framework with the following enhancements:

## Features

### 1. Page-Object Pattern
- **Location**: `tests/ui/pages/`
- **Structure**: Clean separation of page elements and actions
- **Benefits**: Improved maintainability and reusability

### 2. Enhanced Wait Strategies
- **WebDriverWait**: Robust element waiting with expected conditions
- **HTMX Wait Helper**: Custom wait helpers for HTMX requests
- **Retry Mechanisms**: `@retry_on_failure(3)` decorator for interim stability

### 3. Cross-Browser Testing
- **Chrome**: Headless Chrome testing in Docker
- **Firefox**: Headless Firefox testing in Docker
- **Configuration**: Environment-based browser selection

### 4. Failure Capture
- **Screenshots**: Automatic screenshot capture on test failure
- **Videos**: Video recording of test execution
- **Traces**: Playwright trace files for debugging

### 5. Docker Integration
- **Containerized Testing**: Tests run in isolated Docker containers
- **Multi-Browser Support**: Parallel execution across browsers
- **Artifact Collection**: Centralized test artifacts

## Directory Structure

```
tests/ui/
├── pages/                    # Page Object implementations
│   ├── __init__.py
│   ├── base_page.py         # Base page with common functionality
│   ├── dashboard_page.py    # Dashboard-specific page object
│   ├── datasets_page.py     # Datasets page object
│   ├── detectors_page.py    # Detectors page object
│   ├── detection_page.py    # Detection page object
│   └── visualizations_page.py # Visualizations page object
├── helpers/                 # Test utilities and helpers
│   ├── __init__.py
│   ├── retry_decorator.py   # Retry mechanism decorator
│   ├── wait_helpers.py      # WebDriver and HTMX wait helpers
│   ├── screenshot_helper.py # Screenshot capture utilities
│   └── video_helper.py      # Video recording utilities
├── docker/                  # Docker configuration for testing
│   ├── Dockerfile          # Docker image for UI tests
│   ├── docker-compose.yml  # Multi-browser test orchestration
│   ├── conftest.py         # Docker-specific test configuration
│   ├── test_config.py      # Enhanced test configuration
│   └── run_tests.py        # Test runner script
├── screenshots/            # Test failure screenshots
├── videos/                 # Test execution videos
├── conftest.py            # Main test configuration
└── test_*.py              # Test files
```

## Usage

### Local Development

1. **Install Dependencies**:
   ```bash
   pip install playwright pytest pytest-playwright
   playwright install chrome firefox
   ```

2. **Run Tests Locally**:
   ```bash
   # Run all UI tests
   pytest tests/ui/ -v

   # Run with specific browser
   BROWSER=chrome pytest tests/ui/ -v
   BROWSER=firefox pytest tests/ui/ -v

   # Run with screenshots and videos
   TAKE_SCREENSHOTS=true RECORD_VIDEOS=true pytest tests/ui/ -v
   ```

### Docker Testing

1. **Build Docker Image**:
   ```bash
   cd tests/ui/docker
   python run_tests.py --build
   ```

2. **Run Tests in Docker**:
   ```bash
   # Run tests with Chrome
   python run_tests.py --browser chrome

   # Run tests with Firefox
   python run_tests.py --browser firefox

   # Run tests with both browsers
   python run_tests.py --browser both

   # Use Docker Compose
   python run_tests.py --compose
   ```

3. **Docker Compose**:
   ```bash
   docker-compose up --build
   ```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYNOMALY_BASE_URL` | `http://localhost:8000` | Base URL for testing |
| `BROWSER` | `chrome` | Browser to use (chrome/firefox) |
| `HEADLESS` | `true` | Run browser in headless mode |
| `TAKE_SCREENSHOTS` | `true` | Capture screenshots on failure |
| `RECORD_VIDEOS` | `true` | Record test execution videos |
| `RECORD_TRACES` | `false` | Record Playwright traces |
| `TIMEOUT` | `30000` | Default timeout in milliseconds |
| `PARALLEL_WORKERS` | `2` | Number of parallel test workers |

## Page Objects

### BasePage

Base class for all page objects with common functionality:

```python
from tests.ui.pages import BasePage

class MyPage(BasePage):
    def __init__(self, page):
        super().__init__(page)
        self.my_element = "#my-element"
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def click_my_element(self):
        self.click_element(self.my_element)
```

### Enhanced Wait Strategies

#### HTMX Wait Helper

```python
# Wait for HTMX requests to complete
self.htmx_wait.wait_for_htmx_settle()

# Wait for specific HTMX response
self.htmx_wait.wait_for_htmx_response("/api/data")

# Wait for HTMX element swap
self.htmx_wait.wait_for_htmx_element_swap("#content")
```

#### WebDriver Wait Helper

```python
# Wait for element to be visible
element = self.webdriver_wait.wait_for_element_visible((By.ID, "my-element"))

# Wait for element to be clickable
element = self.webdriver_wait.wait_for_element_clickable((By.ID, "button"))

# Wait for text to appear
self.webdriver_wait.wait_for_text_in_element((By.ID, "status"), "Complete")
```

### Retry Decorator

```python
from tests.ui.helpers import retry_on_failure

@retry_on_failure(max_retries=3, delay=1.0, exponential_backoff=True)
def flaky_operation(self):
    # This will retry up to 3 times with exponential backoff
    pass
```

## Test Configuration

### Fixtures

- `dashboard_page`: Dashboard page object
- `datasets_page`: Datasets page object  
- `detectors_page`: Detectors page object
- `detection_page`: Detection page object
- `visualizations_page`: Visualizations page object
- `sample_csv_data`: Sample CSV data for testing

### Markers

- `@pytest.mark.browser("chrome")`: Browser-specific tests
- `@pytest.mark.timeout(60)`: Custom timeout
- `@pytest.mark.flaky`: Flaky test marker
- `@pytest.mark.slow`: Slow test marker
- `@pytest.mark.critical`: Critical test marker

## Best Practices

### 1. Page Object Design

```python
class MyPage(BasePage):
    def __init__(self, page):
        super().__init__(page)
        # Define selectors
        self.submit_button = "#submit-btn"
        self.form_input = "#form-input"
    
    @retry_on_failure(max_retries=3, delay=0.5)
    def submit_form(self, data):
        # Fill form
        self.page.fill(self.form_input, data)
        # Click submit
        self.click_element(self.submit_button)
        # Wait for HTMX
        self.htmx_wait.wait_for_htmx_settle()
```

### 2. Test Structure

```python
def test_feature_workflow(dashboard_page):
    # Navigate to page
    dashboard_page.navigate()
    
    # Perform actions
    dashboard_page.click_nav_link("Datasets")
    
    # Verify results
    assert dashboard_page.verify_navigation_successful()
    
    # Capture screenshot for documentation
    dashboard_page.capture_screenshot("feature_workflow_complete")
```

### 3. Error Handling

```python
@retry_on_failure(max_retries=3, delay=1.0)
def test_with_retry(page):
    try:
        # Test logic
        pass
    except Exception as e:
        # Capture artifacts on failure
        page.capture_screenshot_on_failure("test_failed")
        raise
```

## Reporting

### Test Reports

- **HTML Report**: `test_reports/report_[browser].html`
- **JSON Report**: `test_reports/report_[browser].json`
- **Combined Report**: `test_reports/combined_report.json`

### Artifacts

- **Screenshots**: `test_reports/screenshots/`
- **Videos**: `test_reports/videos/`
- **Traces**: `test_reports/traces/`

### Example Report Structure

```json
{
  "browsers": {
    "chrome": {
      "summary": {
        "total": 25,
        "passed": 23,
        "failed": 2,
        "skipped": 0
      }
    },
    "firefox": {
      "summary": {
        "total": 25,
        "passed": 24,
        "failed": 1,
        "skipped": 0
      }
    }
  },
  "summary": {
    "total_tests": 50,
    "passed": 47,
    "failed": 3,
    "skipped": 0,
    "duration": 120.5
  }
}
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: UI Tests

on: [push, pull_request]

jobs:
  ui-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        browser: [chrome, firefox]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Run UI Tests
      run: |
        cd tests/ui/docker
        python run_tests.py --browser ${{ matrix.browser }} --build
    
    - name: Upload Artifacts
      uses: actions/upload-artifact@v2
      if: always()
      with:
        name: test-results-${{ matrix.browser }}
        path: tests/ui/docker/test_reports/
```

## Troubleshooting

### Common Issues

1. **Element Not Found**: Use proper wait strategies
2. **HTMX Timeouts**: Increase timeout or use HTMX-specific waits
3. **Test Flakiness**: Use retry decorators and proper synchronization
4. **Docker Issues**: Ensure proper networking and volume mounting

### Debug Mode

```bash
# Run with debug output
DEBUG=true pytest tests/ui/ -v -s

# Run with trace recording
RECORD_TRACES=true pytest tests/ui/ -v

# Run single test with video
RECORD_VIDEOS=true pytest tests/ui/test_dashboard.py::test_dashboard_loads -v
```

## Contributing

1. Follow the Page Object pattern
2. Use retry decorators for flaky operations
3. Add proper wait strategies for HTMX
4. Include test documentation
5. Test on both Chrome and Firefox
6. Ensure Docker compatibility
