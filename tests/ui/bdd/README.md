# Pynomaly BDD Testing Framework

## Overview

This directory contains the Behavior-Driven Development (BDD) testing framework for Pynomaly, implementing comprehensive user workflow testing using Gherkin scenarios and pytest-bdd.

## Framework Architecture

### 🎯 **Components**

1. **Feature Files** (`features/*.feature`)
   - Gherkin-based scenario definitions
   - Human-readable test specifications
   - User story documentation

2. **Step Definitions** (`step_definitions/*.py`)
   - Python implementations of Gherkin steps
   - Reusable test actions and assertions
   - Cross-browser compatibility helpers

3. **Test Runners** (`test_*.py`)
   - pytest-bdd integration
   - Parallel execution support
   - Comprehensive reporting

4. **Configuration** (`pytest-bdd.ini`)
   - Test discovery settings
   - Marker definitions
   - Timeout and coverage configuration

## 📁 Directory Structure

```
tests/ui/bdd/
├── README.md                          # This file
├── features/                          # Gherkin feature files
│   ├── user_workflows.feature         # Complete user journey tests
│   ├── accessibility_compliance.feature # WCAG compliance scenarios
│   └── performance_optimization.feature # Performance testing scenarios
├── step_definitions/                  # Python step implementations
│   ├── __init__.py
│   ├── accessibility_steps.py         # Accessibility test steps
│   ├── performance_steps.py           # Performance test steps
│   └── cross_browser_steps.py         # Cross-browser test steps
├── test_user_workflows.py             # Main BDD test runner
└── conftest.py                        # BDD test configuration
```

## 🎪 **Test Categories**

### 1. User Workflow Tests (`user_workflows.feature`)
- **Data Scientist Research Workflow**: Complete ML pipeline testing
- **Dataset Upload and Management**: File handling and validation
- **Detector Creation and Configuration**: Algorithm setup and training
- **Real-time Anomaly Detection**: Live detection scenarios
- **Visualization and Analysis**: Chart interaction and exploration
- **Results Export and Reporting**: Data export functionality
- **Error Handling**: Invalid input and edge case testing
- **Collaborative Features**: Team sharing and project management

### 2. Accessibility Compliance Tests (`accessibility_compliance.feature`)
- **Screen Reader Support**: ARIA compliance and navigation
- **Keyboard Navigation**: Tab order and focus management
- **Color Contrast**: WCAG 2.1 AA compliance
- **Form Accessibility**: Label association and error handling
- **Responsive Design**: Mobile and tablet accessibility

### 3. Performance Optimization Tests (`performance_optimization.feature`)
- **Core Web Vitals**: LCP, FID, CLS monitoring
- **Bundle Size Optimization**: JavaScript and CSS efficiency
- **API Response Times**: Backend performance validation
- **Large Dataset Handling**: Memory and processing efficiency
- **Progressive Loading**: Lazy loading and chunking

### 4. Cross-Browser Compatibility Tests
- **Browser Support**: Chrome, Firefox, Safari, Edge testing
- **Feature Consistency**: Cross-browser behavior validation
- **Visual Regression**: Layout and styling consistency
- **JavaScript Compatibility**: API support verification

## 🚀 **Usage Guide**

### Running BDD Tests

#### All Tests
```bash
# Run all BDD scenarios
python tests/ui/run_bdd_tests.py

# With pytest directly
pytest tests/ui/bdd/ -v --html=reports/bdd_report.html
```

#### Specific Categories
```bash
# Run only user workflow tests
python tests/ui/run_bdd_tests.py --categories workflow

# Run accessibility tests
python tests/ui/run_bdd_tests.py --categories accessibility

# Run performance tests
python tests/ui/run_bdd_tests.py --categories performance
```

#### Advanced Options
```bash
# Run with custom output directory
python tests/ui/run_bdd_tests.py --output-dir custom_reports/

# Run specific scenarios with pytest markers
pytest tests/ui/bdd/ -m "accessibility and not slow"

# Run with coverage reporting
pytest tests/ui/bdd/ --cov=src --cov-report=html
```

### Development Workflow

#### 1. Writing New Feature Files
```gherkin
Feature: New Functionality
  As a [user_type]
  I want to [functionality]
  So that I can [benefit]

  Background:
    Given I am logged into the application
    And I have the necessary permissions

  Scenario: Happy Path Scenario
    Given I have [precondition]
    When I [action]
    Then I should see [expected_result]
    And I should be able to [follow_up_action]

  Scenario Outline: Multiple Data Scenarios
    Given I have <input_type> data
    When I process the data
    Then I should get <result_type> results

    Examples:
      | input_type | result_type |
      | valid      | success     |
      | invalid    | error       |
```

#### 2. Implementing Step Definitions
```python
from pytest_bdd import given, when, then, parsers

@given("I have <input_type> data")
def given_input_data(page, input_type):
    """Prepare test data based on input type."""
    # Implementation here
    pass

@when("I process the data")
def when_process_data(page, ui_helper):
    """Trigger data processing."""
    # Implementation here
    pass

@then("I should get <result_type> results")
def then_verify_results(page, result_type):
    """Verify expected results."""
    # Implementation here
    pass
```

#### 3. Adding Test Markers
```python
import pytest
from pytest_bdd import scenarios

# Load scenarios with markers
scenarios(
    'features/my_feature.feature',
    example_converters=dict(result_type=str)
)

@pytest.mark.accessibility
@pytest.mark.slow
def test_accessibility_scenario():
    pass
```

## 📊 **Reporting and Analysis**

### Report Types Generated

1. **HTML Report** (`test_reports/bdd/bdd_report.html`)
   - Interactive test results
   - Step-by-step execution details
   - Screenshot attachments
   - Timeline visualization

2. **JUnit XML** (`test_reports/bdd/bdd_junit.xml`)
   - CI/CD integration format
   - Test result summary
   - Failure details

3. **JSON Report** (`test_reports/bdd/bdd_comprehensive_report.json`)
   - Machine-readable results
   - Detailed execution metrics
   - Feature analysis data

4. **Text Summary** (`test_reports/bdd/bdd_summary.txt`)
   - Quick overview
   - Pass/fail statistics
   - Recommendations

### Metrics Tracked

- **Execution Time**: Per scenario and step timing
- **Success Rate**: Percentage of passing tests
- **Coverage Analysis**: Feature and scenario coverage
- **Browser Compatibility**: Cross-browser test results
- **Performance Metrics**: Core Web Vitals integration
- **Accessibility Scores**: WCAG compliance levels

## 🔧 **Configuration**

### Environment Variables
```bash
# Test configuration
export PYNOMALY_BASE_URL="http://localhost:8000"
export PYNOMALY_TEST_TIMEOUT="300"
export PYNOMALY_HEADLESS="true"

# Browser configuration
export BROWSER="chromium"  # chromium, firefox, webkit
export DEVICE="Desktop Chrome"

# Reporting configuration
export BDD_REPORT_DIR="test_reports/bdd"
export BDD_SCREENSHOTS="true"
```

### pytest Configuration (`pytest-bdd.ini`)
- Test discovery patterns
- Marker definitions
- Timeout settings
- Coverage requirements
- Parallel execution setup

## 🚨 **Troubleshooting**

### Common Issues

#### 1. Step Definition Not Found
```bash
# Error: Step definition for "Given I am on the homepage" not found
# Solution: Implement the missing step definition
@given("I am on the homepage")
def given_homepage(page):
    page.goto("/")
```

#### 2. Element Not Found
```bash
# Error: Selector not found
# Solution: Use more robust selectors or wait conditions
await page.wait_for_selector("button", state="visible", timeout=10000)
```

#### 3. Test Timeout
```bash
# Error: Test timed out
# Solution: Increase timeout or optimize test speed
pytest tests/ui/bdd/ --timeout=600
```

#### 4. Browser Launch Failed
```bash
# Error: Browser launch failed
# Solution: Install browser dependencies
playwright install --with-deps chromium
```

### Debugging Tips

1. **Enable Debug Mode**
   ```bash
   pytest tests/ui/bdd/ --headed --slowmo=1000
   ```

2. **Screenshot on Failure**
   ```python
   # Automatic screenshots are taken on test failures
   # Check test_reports/bdd/screenshots/
   ```

3. **Verbose Output**
   ```bash
   pytest tests/ui/bdd/ -vvv --tb=long
   ```

4. **Run Single Scenario**
   ```bash
   pytest tests/ui/bdd/ -k "test_user_workflow"
   ```

## 🎯 **Best Practices**

### 1. Writing Effective Scenarios
- Use business language, not technical terms
- Focus on user goals and outcomes
- Keep scenarios independent and atomic
- Use examples for data-driven testing
- Include both positive and negative cases

### 2. Step Definition Guidelines
- Make steps reusable across scenarios
- Use page object pattern for UI interactions
- Implement proper wait conditions
- Add meaningful assertions
- Handle errors gracefully

### 3. Maintenance Tips
- Review and update scenarios regularly
- Keep step definitions DRY (Don't Repeat Yourself)
- Use consistent naming conventions
- Document complex test logic
- Monitor test execution times

### 4. CI/CD Integration
- Run BDD tests in parallel
- Generate artifacts for analysis
- Set up failure notifications
- Track test metrics over time
- Use environment-specific configurations

## 📈 **Continuous Improvement**

### Metrics to Monitor
- Test execution time trends
- Flaky test identification
- Coverage gap analysis
- User journey completeness
- Browser compatibility scores

### Regular Reviews
- Monthly scenario relevance check
- Quarterly framework updates
- Performance optimization cycles
- Accessibility compliance audits
- Cross-browser testing updates

---

## 📞 **Support**

For questions or issues with the BDD testing framework:

1. Check this documentation first
2. Review existing scenarios for examples
3. Check the troubleshooting section
4. Create an issue with:
   - Error details
   - Steps to reproduce
   - Expected vs. actual behavior
   - Environment information

## 📚 **References**

- [pytest-bdd Documentation](https://pytest-bdd.readthedocs.io/)
- [Gherkin Syntax Reference](https://cucumber.io/docs/gherkin/)
- [Playwright Python API](https://playwright.dev/python/)
- [BDD Best Practices](https://cucumber.io/docs/bdd/)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
