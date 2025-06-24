# 🔍 Pynomaly UI Automation Testing

This comprehensive UI testing framework provides automated testing, visual regression detection, accessibility validation, and detailed critiques of the Pynomaly web interface.

## 🚀 Quick Start

### Run All UI Tests
```bash
# Run the complete UI testing suite
./scripts/run_ui_testing.sh

# Skip Docker image building (if images already exist)
./scripts/run_ui_testing.sh --skip-build

# Cleanup Docker resources only
./scripts/run_ui_testing.sh --cleanup-only
```

### Manual Docker Compose
```bash
# Start application and run UI tests
docker-compose -f docker-compose.ui-testing.yml up --build

# Run only visual regression tests
docker-compose -f docker-compose.ui-testing.yml run --rm visual-tests

# Run specific test files
docker-compose -f docker-compose.ui-testing.yml run --rm ui-tests pytest tests/ui/test_layout_validation.py -v
```

## 🎯 Test Categories

### 1. Layout Validation (`test_layout_validation.py`)
- **Navigation consistency** across all pages
- **Semantic HTML** structure validation
- **Form elements** proper structure and labeling
- **Responsive navigation** behavior
- **Button states** and interactions
- **Error message** display and accessibility

### 2. UX Flow Testing (`test_ux_flows.py`)
- **Complete user journeys** (detector creation, navigation flows)
- **Form validation** and error recovery
- **HTMX interactions** and dynamic updates
- **Mobile navigation** functionality
- **Performance** during user interactions
- **Accessibility navigation** with keyboard

### 3. Visual Regression (`test_visual_regression.py`)
- **Screenshot comparison** with OpenCV
- **Component-level** visual consistency
- **Responsive design** visual validation
- **Animation and transition** consistency
- **Cross-browser** visual compatibility
- **Baseline management** for visual tests

### 4. Accessibility Testing (`test_accessibility.py`)
- **WCAG 2.1** compliance validation
- **Keyboard navigation** functionality
- **Screen reader** compatibility
- **ARIA attributes** proper usage
- **Color contrast** validation
- **Focus indicators** visibility
- **Semantic structure** validation

### 5. Responsive Design (`test_responsive_design.py`)
- **Multi-viewport** testing (320px to 1920px)
- **Touch target** sizing validation
- **Content reflow** behavior
- **Breakpoint consistency** 
- **Mobile navigation** functionality
- **Image responsiveness**
- **Grid layout** adaptation

## 📊 Test Reports

### Generated Reports
After running tests, check these directories:

```
📁 reports/
├── ui_test_report_YYYYMMDD_HHMMSS.html    # Comprehensive HTML report
└── ui_test_summary_YYYYMMDD_HHMMSS.json   # JSON summary for CI/CD

📁 screenshots/
├── dashboard_layout.png                    # Layout screenshots
├── mobile_navigation.png                   # Mobile screenshots
├── responsive_mobile.png                   # Responsive screenshots
└── failure_test_name_timestamp.png         # Failure screenshots

📁 visual-baselines/
├── dashboard_full.png                      # Visual regression baselines
└── navigation_components.png               # Component baselines

📁 test-results/
├── videos/                                 # Test execution videos
└── traces/                                 # Playwright traces
```

### Report Features
- **Overall quality score** (0-100) with letter grade
- **Category-specific scores** for each test area
- **Detailed critiques** with actionable recommendations
- **Priority issue** ranking (Critical, Warning, Recommendation)
- **Visual comparisons** with baseline images
- **Performance metrics** and timing data

## 🛠️ Configuration

### Environment Variables
```bash
# Application URL (default: http://pynomaly-app:8000)
PYNOMALY_BASE_URL=http://localhost:8000

# Browser type (default: chromium)
BROWSER=chromium  # or firefox, webkit

# Enable video recording
RECORD_VIDEO=true

# Screenshot on failure
SCREENSHOT_ON_FAILURE=true
```

### Customizing Tests

#### Adding New Test Cases
```python
# tests/ui/test_custom.py
def test_custom_functionality(page: Page):
    """Test custom functionality."""
    page.goto("http://pynomaly-app:8000/web/custom")
    
    # Your test logic here
    assert page.locator("h1").text_content() == "Expected Title"
```

#### Updating Visual Baselines
```bash
# Delete existing baselines to recreate them
rm -rf visual-baselines/

# Run visual tests to create new baselines
docker-compose -f docker-compose.ui-testing.yml run --rm visual-tests
```

#### Custom Page Objects
```python
# tests/ui/page_objects/custom_page.py
from .base_page import BasePage

class CustomPage(BasePage):
    def navigate(self):
        self.navigate_to("/custom")
        
    def interact_with_element(self):
        self.click_element("button[data-action='custom']")
```

## 🔧 Development & Debugging

### Running Tests Locally
```bash
# Install dependencies
pip install playwright pytest-playwright opencv-python

# Install Playwright browsers
playwright install

# Run specific test
pytest tests/ui/test_layout_validation.py::test_navigation_consistency -v -s

# Run with headed browser (for debugging)
pytest tests/ui/ --headed
```

### Debugging Failed Tests
```bash
# Run with trace recording
pytest tests/ui/ --tracing=on

# View trace file
playwright show-trace test-results/trace.zip

# Run with video recording
pytest tests/ui/ --video=on

# Run with screenshots on failure
pytest tests/ui/ --screenshot=on
```

### Adding Custom Assertions
```python
def test_custom_validation(page: Page):
    """Custom validation example."""
    page.goto("http://pynomaly-app:8000/web/")
    
    # Custom assertion with detailed message
    nav_element = page.locator("nav")
    assert nav_element.is_visible(), \
        f"Navigation should be visible but was not found"
    
    # Take screenshot for manual review
    page.screenshot(path="screenshots/custom_validation.png")
```

## 📈 Performance Monitoring

### Metrics Tracked
- **Page load times** across different viewport sizes
- **Interactive element** response times
- **HTMX request/response** timing
- **Chart rendering** performance
- **Form submission** latency

### Performance Thresholds
- Page load: < 3 seconds
- Interactive elements: < 100ms response
- HTMX updates: < 1 second
- Chart rendering: < 2 seconds

## 🚨 Issue Detection & Fixes

### Common Issues & Solutions

#### Layout Issues
```yaml
Issue: Navigation inconsistent across pages
Fix: Review navigation component in base.html template
Priority: High

Issue: Mobile menu not functional
Fix: Check Alpine.js integration and mobile breakpoints
Priority: High
```

#### Accessibility Issues
```yaml
Issue: Images missing alt text
Fix: Add descriptive alt attributes to all <img> tags
Priority: High

Issue: Form inputs without labels
Fix: Associate labels with inputs using for/id attributes
Priority: High

Issue: Poor color contrast
Fix: Update CSS color values to meet WCAG AA standards
Priority: Medium
```

#### Responsive Issues
```yaml
Issue: Horizontal scroll on mobile
Fix: Check for fixed-width elements, use max-width: 100%
Priority: Medium

Issue: Touch targets too small
Fix: Ensure interactive elements are at least 44px × 44px
Priority: Medium
```

#### Performance Issues
```yaml
Issue: Slow chart rendering
Fix: Optimize D3.js/ECharts initialization and data loading
Priority: Low

Issue: Large bundle size
Fix: Implement code splitting and lazy loading
Priority: Low
```

## 🔄 CI/CD Integration

### GitHub Actions Example
```yaml
name: UI Tests
on: [push, pull_request]
jobs:
  ui-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run UI Tests
        run: ./scripts/run_ui_testing.sh
      - name: Upload Test Reports
        uses: actions/upload-artifact@v3
        with:
          name: ui-test-reports
          path: |
            reports/
            screenshots/
            test-results/
```

### Quality Gates
- **Accessibility score** must be ≥ 80/100
- **Overall UI score** must be ≥ 85/100
- **No critical issues** allowed in production
- **Visual regressions** require approval

## 📚 Best Practices

### Test Writing
1. **Use page objects** for maintainable tests
2. **Wait for elements** explicitly, avoid fixed timeouts
3. **Take screenshots** on failures for debugging
4. **Test real user scenarios**, not just happy paths
5. **Include accessibility** in every test case

### Maintenance
1. **Update baselines** when UI changes are intentional
2. **Review failing tests** promptly to avoid test debt
3. **Monitor performance** trends over time
4. **Keep dependencies** updated for security

### Team Collaboration
1. **Document test scenarios** for domain experts
2. **Share test reports** with design and product teams
3. **Use visual tests** for design review processes
4. **Involve accessibility experts** in validation

## 🆘 Troubleshooting

### Common Problems

#### Docker Issues
```bash
# Problem: Docker container won't start
# Solution: Check Docker daemon and available resources
docker system prune -f
docker-compose -f docker-compose.ui-testing.yml down -v

# Problem: Permission denied errors
# Solution: Fix directory permissions
chmod 755 test-results screenshots reports visual-baselines
```

#### Test Failures
```bash
# Problem: Tests timing out
# Solution: Increase timeout values in conftest.py

# Problem: Visual regression false positives
# Solution: Lower similarity threshold or update baselines

# Problem: Accessibility test failures
# Solution: Check HTML structure and ARIA attributes
```

#### Browser Issues
```bash
# Problem: Playwright browsers not found
# Solution: Reinstall browsers
playwright install --with-deps

# Problem: Browser crashes
# Solution: Add browser arguments for stability
--no-sandbox --disable-dev-shm-usage
```

## 📞 Support

For UI testing issues:
1. Check the test logs in `test-results/`
2. Review screenshots in `screenshots/` 
3. Consult the HTML report in `reports/`
4. Create an issue with test artifacts attached

---

**Happy Testing! 🎯**

The goal is to ensure Pynomaly provides an excellent user experience across all devices and for all users, including those using assistive technologies.