# Cross-Browser Testing Guide

## 🌐 **Overview**

This guide provides comprehensive documentation for cross-browser testing in the Pynomaly platform, covering compatibility testing, device testing, performance monitoring, and reporting across multiple browsers and devices.

## 📋 **Table of Contents**

- [🎯 Testing Strategy](#-testing-strategy)
- [🌏 Browser Support Matrix](#-browser-support-matrix)
- [📱 Device Testing](#-device-testing)
- [🧪 Test Categories](#-test-categories)
- [⚡ Performance Testing](#-performance-testing)
- [♿ Accessibility Testing](#-accessibility-testing)
- [📊 Reporting and Analysis](#-reporting-and-analysis)
- [🛠️ Tools and Configuration](#️-tools-and-configuration)
- [🚀 Running Tests](#-running-tests)
- [🔧 Troubleshooting](#-troubleshooting)

## 🎯 **Testing Strategy**

### Cross-Browser Testing Approach

Our cross-browser testing strategy follows a **risk-based testing** approach:

1. **Tier 1 (Critical)**: Chrome, Firefox, Safari, Edge - Latest 2 versions
2. **Tier 2 (Important)**: Mobile Chrome, Mobile Safari, iPad Safari
3. **Tier 3 (Nice-to-have)**: Older browser versions, less common browsers

### Test Pyramid for Cross-Browser

```
    🔺 E2E Cross-Browser Tests (Critical flows only)
   🔺🔺 Integration Tests (API + UI interactions)
  🔺🔺🔺 Unit Tests (Browser-agnostic logic)
 🔺🔺🔺🔺 Static Analysis (Linting, type checking)
```

### Testing Scope

- **Core Functionality**: Authentication, data upload, analysis, visualization
- **UI Interactions**: Forms, navigation, modals, responsive behavior
- **Performance**: Load times, Core Web Vitals, memory usage
- **Accessibility**: WCAG 2.1 AA compliance across browsers
- **Progressive Enhancement**: Graceful degradation for older browsers

## 🌏 **Browser Support Matrix**

### Desktop Browsers

| Browser | Versions | Platform | Test Priority | Notes |
|---------|----------|----------|---------------|-------|
| **Chrome** | Latest 2 | Windows, macOS, Linux | Tier 1 | Primary development browser |
| **Firefox** | Latest 2 | Windows, macOS, Linux | Tier 1 | Alternative rendering engine |
| **Safari** | Latest 2 | macOS | Tier 1 | WebKit engine, iOS consistency |
| **Edge** | Latest 2 | Windows, macOS | Tier 1 | Chromium-based, enterprise users |
| **Chrome Legacy** | 90+ | All platforms | Tier 3 | Backward compatibility |

### Mobile Browsers

| Browser | Versions | Platform | Test Priority | Notes |
|---------|----------|----------|---------------|-------|
| **Mobile Chrome** | Latest 2 | Android | Tier 2 | Dominant mobile browser |
| **Mobile Safari** | Latest 2 | iOS | Tier 2 | iOS default browser |
| **Mobile Firefox** | Latest | Android | Tier 3 | Alternative mobile option |
| **Samsung Internet** | Latest | Android | Tier 3 | Popular on Samsung devices |

### Tablet Browsers

| Device | Browser | Test Priority | Notes |
|--------|---------|---------------|-------|
| **iPad Pro** | Safari | Tier 2 | Large tablet interface |
| **iPad Air** | Safari | Tier 2 | Standard tablet size |
| **Galaxy Tab** | Chrome | Tier 3 | Android tablet |
| **Surface Pro** | Edge | Tier 3 | Windows tablet |

## 📱 **Device Testing**

### Device Categories

#### Mobile Devices (320px - 767px)
- **iPhone 12/13/14 series**: Primary iOS testing
- **Pixel 5/6/7**: Primary Android testing
- **Galaxy S21/S22**: Samsung-specific testing
- **iPhone SE**: Small screen testing

#### Tablet Devices (768px - 1023px)
- **iPad Pro 12.9"**: Large tablet interface
- **iPad Air**: Standard tablet size
- **Galaxy Tab S4**: Android tablet
- **Surface Pro**: Windows tablet/laptop hybrid

#### Desktop Devices (1024px+)
- **1080p (1920x1080)**: Standard desktop resolution
- **1440p (2560x1440)**: High-resolution desktop
- **4K (3840x2160)**: Ultra-high resolution
- **Ultrawide (3440x1440)**: Widescreen monitors

### Responsive Breakpoints

```css
/* Mobile First Approach */
/* Base: 320px - 767px (Mobile) */

@media (min-width: 768px) {
  /* Tablet: 768px - 1023px */
}

@media (min-width: 1024px) {
  /* Desktop: 1024px - 1439px */
}

@media (min-width: 1440px) {
  /* Large Desktop: 1440px+ */
}
```

### Touch Interaction Testing

#### Touch Targets
- **Minimum Size**: 44x44px (WCAG 2.1 AA)
- **Recommended Size**: 48x48px
- **Spacing**: 8px minimum between targets

#### Gesture Support
- **Tap**: Primary interaction
- **Long Press**: Context menus
- **Swipe**: Navigation, carousels
- **Pinch Zoom**: Accessibility requirement
- **Scroll**: Smooth scrolling behavior

## 🧪 **Test Categories**

### 1. Core Functionality Tests

```typescript
// Example: Cross-browser navigation test
test.describe('Navigation Compatibility', () => {
  test('should navigate consistently across browsers', async ({ page, browserName }) => {
    await page.goto('/');
    
    // Test main navigation
    await page.click('nav a[href="/dashboard"]');
    await expect(page).toHaveURL(/dashboard/);
    
    // Verify browser-specific behavior
    if (browserName === 'webkit') {
      // Safari-specific assertions
    }
  });
});
```

### 2. JavaScript API Compatibility

Tests for modern JavaScript features and APIs:

- **ES6+ Features**: Arrow functions, classes, modules
- **Web APIs**: Fetch, localStorage, IndexedDB
- **Performance APIs**: Navigation Timing, Observer APIs
- **Modern Features**: IntersectionObserver, ResizeObserver

### 3. CSS Feature Support

Tests for CSS compatibility:

- **Flexbox and Grid**: Layout consistency
- **Custom Properties**: CSS variables support
- **Modern Units**: vw, vh, calc()
- **Transforms and Animations**: Cross-browser consistency

### 4. Form Compatibility

Tests for form behavior across browsers:

- **Input Types**: email, number, date, etc.
- **Validation**: HTML5 validation consistency
- **File Upload**: Multi-file, drag-and-drop
- **Autofill**: Browser autofill behavior

## ⚡ **Performance Testing**

### Core Web Vitals Monitoring

```typescript
test('should meet Core Web Vitals thresholds', async ({ page, browserName }) => {
  await page.goto('/dashboard');
  
  const metrics = await page.evaluate(() => {
    return new Promise((resolve) => {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const vitals = {};
        
        entries.forEach((entry) => {
          if (entry.entryType === 'largest-contentful-paint') {
            vitals.LCP = entry.startTime;
          }
          if (entry.entryType === 'first-input') {
            vitals.FID = entry.processingStart - entry.startTime;
          }
          if (entry.entryType === 'layout-shift' && !entry.hadRecentInput) {
            vitals.CLS = (vitals.CLS || 0) + entry.value;
          }
        });
        
        setTimeout(() => resolve(vitals), 3000);
      });
      
      observer.observe({ entryTypes: ['largest-contentful-paint', 'first-input', 'layout-shift'] });
    });
  });
  
  // Browser-specific thresholds
  const thresholds = {
    'chromium': { LCP: 2500, FID: 100, CLS: 0.1 },
    'firefox': { LCP: 3000, FID: 150, CLS: 0.15 },
    'webkit': { LCP: 3500, FID: 200, CLS: 0.2 }
  };
  
  const threshold = thresholds[browserName] || thresholds['chromium'];
  
  if (metrics.LCP) expect(metrics.LCP).toBeLessThan(threshold.LCP);
  if (metrics.FID) expect(metrics.FID).toBeLessThan(threshold.FID);
  if (metrics.CLS) expect(metrics.CLS).toBeLessThan(threshold.CLS);
});
```

### Performance Metrics by Browser

| Metric | Chrome | Firefox | Safari | Edge | Notes |
|--------|--------|---------|--------|------|-------|
| **LCP** | <2.5s | <3.0s | <3.5s | <2.5s | Largest Contentful Paint |
| **FID** | <100ms | <150ms | <200ms | <100ms | First Input Delay |
| **CLS** | <0.1 | <0.15 | <0.2 | <0.1 | Cumulative Layout Shift |
| **TTI** | <5s | <6s | <7s | <5s | Time to Interactive |

### Memory Usage Testing

```typescript
test('should handle memory constraints efficiently', async ({ page, browserName }) => {
  await page.goto('/datasets');
  
  const memoryUsage = await page.evaluate(() => {
    if (performance.memory) {
      return {
        used: performance.memory.usedJSHeapSize,
        total: performance.memory.totalJSHeapSize,
        limit: performance.memory.jsHeapSizeLimit
      };
    }
    return null;
  });
  
  if (memoryUsage) {
    // Memory usage should be reasonable
    expect(memoryUsage.used / memoryUsage.limit).toBeLessThan(0.8);
  }
});
```

## ♿ **Accessibility Testing**

### Cross-Browser Accessibility Tests

```typescript
test('should maintain accessibility across browsers', async ({ page }) => {
  await page.goto('/');
  
  // Test keyboard navigation
  await page.keyboard.press('Tab');
  const firstFocusable = await page.locator(':focus').first();
  await expect(firstFocusable).toBeVisible();
  
  // Test screen reader landmarks
  const landmarks = await page.locator('[role="main"], [role="navigation"], [role="banner"]').count();
  expect(landmarks).toBeGreaterThan(0);
  
  // Test color contrast
  const contrastIssues = await page.evaluate(() => {
    // Simplified contrast check
    const elements = document.querySelectorAll('*');
    let issues = 0;
    
    elements.forEach(el => {
      const styles = window.getComputedStyle(el);
      const bgColor = styles.backgroundColor;
      const textColor = styles.color;
      
      // Check if colors are set and contrasting
      if (bgColor !== 'rgba(0, 0, 0, 0)' && textColor !== 'rgba(0, 0, 0, 0)') {
        // Simplified contrast calculation
        // In real implementation, use proper contrast ratio calculation
      }
    });
    
    return issues;
  });
  
  expect(contrastIssues).toBe(0);
});
```

### Browser-Specific Accessibility Features

| Browser | Screen Reader | Keyboard Nav | High Contrast | Zoom |
|---------|---------------|--------------|---------------|------|
| **Chrome** | ✅ NVDA, JAWS | ✅ Full support | ✅ Windows HC | ✅ 500% |
| **Firefox** | ✅ NVDA, JAWS | ✅ Full support | ✅ Windows HC | ✅ 300% |
| **Safari** | ✅ VoiceOver | ✅ Full support | ✅ macOS HC | ✅ 500% |
| **Edge** | ✅ Narrator | ✅ Full support | ✅ Windows HC | ✅ 500% |

## 📊 **Reporting and Analysis**

### Test Reports Generated

1. **HTML Report** (`test_reports/cross-browser/cross-browser-report.html`)
   - Interactive compatibility matrix
   - Browser performance metrics
   - Issue severity analysis
   - Visual screenshots comparison

2. **JSON Report** (`test_reports/cross-browser/cross-browser-report.json`)
   - Machine-readable test results
   - Detailed metrics and timings
   - Compatibility scores
   - Performance data

3. **CSV Export** (`test_reports/cross-browser/cross-browser-results.csv`)
   - Test results for data analysis
   - Browser compatibility matrix
   - Performance metrics export

4. **Compatibility Matrix** (`test_reports/cross-browser/compatibility-matrix.json`)
   - Browser support matrix
   - Feature compatibility scores
   - Regression tracking

### Report Sections

#### Executive Summary
- Overall compatibility percentage
- Browser coverage statistics
- Critical issues count
- Performance summary

#### Browser Metrics
- Pass/fail rates per browser
- Average test duration
- Error frequency analysis
- Performance comparison

#### Compatibility Issues
- Issue severity classification
- Browser-specific problems
- Recommended fixes
- Impact assessment

#### Performance Analysis
- Core Web Vitals by browser
- Load time comparisons
- Memory usage patterns
- Regression detection

## 🛠️ **Tools and Configuration**

### Playwright Configuration

```typescript
// playwright.config.ts
export default defineConfig({
  projects: [
    // Desktop browsers
    {
      name: 'Desktop Chrome',
      use: { ...devices['Desktop Chrome'] }
    },
    {
      name: 'Desktop Firefox', 
      use: { ...devices['Desktop Firefox'] }
    },
    {
      name: 'Desktop Safari',
      use: { ...devices['Desktop Safari'] }
    },
    {
      name: 'Desktop Edge',
      use: { ...devices['Desktop Edge'] }
    },
    
    // Mobile browsers
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] }
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] }
    },
    
    // Tablets
    {
      name: 'iPad',
      use: { ...devices['iPad Pro'] }
    }
  ],
  
  reporter: [
    ['html'],
    ['./tests/ui/reporters/cross-browser-reporter.ts']
  ]
});
```

### Test Organization

```
tests/ui/
├── cross-browser/
│   ├── test_core_functionality.spec.ts
│   ├── test_javascript_compatibility.spec.ts
│   ├── test_css_features.spec.ts
│   └── test_performance.spec.ts
├── device/
│   ├── test_device_compatibility.spec.ts
│   ├── test_touch_interactions.spec.ts
│   └── test_responsive_design.spec.ts
├── accessibility/
│   ├── test_keyboard_navigation.spec.ts
│   ├── test_screen_readers.spec.ts
│   └── test_color_contrast.spec.ts
└── reporters/
    └── cross-browser-reporter.ts
```

### CI/CD Integration

```yaml
# .github/workflows/cross-browser-tests.yml
name: Cross-Browser Testing
on: [push, pull_request]

jobs:
  cross-browser-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        browser: [chromium, firefox, webkit]
    
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Install Playwright browsers
        run: npx playwright install --with-deps ${{ matrix.browser }}
      
      - name: Run cross-browser tests
        run: npx playwright test --project="${{ matrix.browser }}"
      
      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: cross-browser-results-${{ matrix.browser }}
          path: test_reports/
```

## 🚀 **Running Tests**

### Basic Commands

```bash
# Run all cross-browser tests
npm run test:cross-browser

# Run specific browser tests
npm run test:desktop
npm run test:mobile
npm run test:device-compatibility

# Run with visual comparison
npm run test:visual

# Run performance tests
npm run test:performance

# Generate compatibility matrix
npm run test:browser-matrix
```

### Advanced Test Execution

```bash
# Run tests in headed mode (see browser windows)
npx playwright test --project=chromium --headed

# Run specific test file across all browsers
npx playwright test tests/ui/cross-browser/test_core_functionality.spec.ts

# Run tests with custom timeout
npx playwright test --timeout=60000

# Run tests with retry on failure
npx playwright test --retries=2

# Debug specific test
npx playwright test --debug tests/ui/device/test_device_compatibility.spec.ts
```

### Test Filtering and Tags

```bash
# Run only Tier 1 browser tests
npx playwright test --grep="@tier1"

# Skip slow tests
npx playwright test --grep-invert="@slow"

# Run only accessibility tests
npx playwright test tests/ui/accessibility/

# Run responsive design tests
npx playwright test tests/ui/responsive/
```

## 🔧 **Troubleshooting**

### Common Issues and Solutions

#### Browser Launch Failures

```bash
# Error: Browser not found
# Solution: Install browsers
npx playwright install

# Error: Dependencies missing
# Solution: Install system dependencies  
npx playwright install-deps
```

#### Test Timeouts

```bash
# Error: Test timeout
# Solution: Increase timeout in config
test.setTimeout(60000);

# Or use per-test timeout
test('slow test', async ({ page }) => {
  test.setTimeout(120000);
  // test code
});
```

#### Flaky Tests

```typescript
// Solution: Add proper waits
await page.waitForLoadState('networkidle');
await page.waitForSelector('[data-testid="content"]');

// Solution: Use retry mechanism
test.describe.configure({ retries: 2 });
```

#### Browser-Specific Issues

```typescript
// Handle Safari-specific behavior
if (browserName === 'webkit') {
  await page.waitForTimeout(500); // Safari needs extra time
}

// Handle mobile-specific issues
if (browserName.includes('Mobile')) {
  await page.tap('button'); // Use tap instead of click
}
```

### Performance Issues

```bash
# Reduce parallel workers for better performance
npx playwright test --workers=1

# Use specific browser for faster debugging
npx playwright test --project=chromium

# Run tests in Docker for consistency
docker run -it --rm mcr.microsoft.com/playwright:latest npx playwright test
```

### Debugging Tools

```bash
# Generate trace files for debugging
npx playwright test --trace=on

# Use Playwright Inspector
npx playwright test --debug

# Generate screenshots on failure
npx playwright test --screenshot=only-on-failure

# Record video on failure
npx playwright test --video=retain-on-failure
```

### CI/CD Troubleshooting

```yaml
# Increase timeout for CI
- name: Run tests
  run: npx playwright test
  timeout-minutes: 30

# Use xvfb for headless testing
- name: Run tests
  run: xvfb-run -a npx playwright test

# Capture artifacts on failure
- name: Upload artifacts
  uses: actions/upload-artifact@v3
  if: failure()
  with:
    name: test-artifacts
    path: |
      test-results/
      playwright-report/
```

## 📈 **Best Practices**

### Test Design

1. **Progressive Enhancement Testing**
   - Test core functionality without JavaScript
   - Verify graceful degradation
   - Test with slow network conditions

2. **Mobile-First Testing**
   - Start with mobile viewport
   - Test touch interactions
   - Verify performance on mobile

3. **Accessibility-First**
   - Test with keyboard only
   - Verify screen reader compatibility
   - Check color contrast

### Performance Optimization

1. **Test Parallelization**
   - Run browsers in parallel
   - Use worker threads efficiently
   - Optimize test data setup

2. **Smart Test Selection**
   - Focus on critical paths
   - Use risk-based testing
   - Skip redundant tests

3. **Resource Management**
   - Close browsers properly
   - Clean up test data
   - Monitor memory usage

### Maintenance

1. **Regular Updates**
   - Update browser versions
   - Review compatibility matrix
   - Update performance thresholds

2. **Issue Tracking**
   - Document known issues
   - Track regression patterns
   - Maintain fix history

3. **Reporting Enhancement**
   - Improve report clarity
   - Add trend analysis
   - Include actionable insights

---

## 📚 **Resources**

- [Playwright Documentation](https://playwright.dev/)
- [Can I Use - Browser Compatibility](https://caniuse.com/)
- [MDN Browser Compatibility Data](https://github.com/mdn/browser-compat-data)
- [Web Platform Tests](https://web-platform-tests.org/)
- [Browser Market Share](https://gs.statcounter.com/)

## 🎯 **Next Steps**

1. **Expand Device Coverage**: Add more mobile devices and screen sizes
2. **Enhance Reporting**: Include trend analysis and historical data
3. **Automate Regression Detection**: Implement automatic baseline comparison
4. **Performance Monitoring**: Add real-time performance tracking
5. **Visual Testing**: Implement comprehensive visual regression testing

For questions or improvements to this testing framework, please refer to the project documentation or create an issue in the repository.