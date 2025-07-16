# Memory Leak Detection and Monitoring

This directory contains comprehensive memory leak detection and performance monitoring tests for the Pynomaly web UI. The testing suite is designed to identify memory leaks, monitor resource usage, and ensure optimal performance under various usage patterns.

## Overview

Memory leaks in web applications can cause performance degradation, browser crashes, and poor user experience. This testing suite provides:

- **Automated Memory Leak Detection**: Identifies memory leaks in JavaScript heap, DOM nodes, and event listeners
- **Performance Monitoring**: Tracks memory usage patterns during user interactions
- **Resource Usage Analysis**: Monitors WebSocket connections, timers, and other resources
- **Comprehensive Reporting**: Generates detailed reports with recommendations

## Test Structure

```
memory-leak-testing/
├── memory-leak-detection.spec.ts     # Core memory leak detection tests
├── performance-monitoring.spec.ts    # Performance monitoring and profiling
├── memory-leak.config.ts            # Playwright configuration for memory tests
├── memory-global-setup.ts           # Global test environment setup
├── memory-global-teardown.ts        # Global cleanup and reporting
├── run-memory-tests.sh              # Comprehensive test execution script
└── README.md                        # This documentation
```

## Features

### 1. Memory Leak Detection

- **Dashboard Navigation**: Tests memory usage during extensive page navigation
- **Chart Rendering**: Monitors memory during chart creation/destruction cycles
- **WebSocket Connections**: Validates proper cleanup of WebSocket connections
- **Form Handling**: Tests memory usage in dynamic form operations
- **Data Table Operations**: Monitors memory during table sorting, filtering, and pagination

### 2. Performance Monitoring

- **Heavy Operations**: Monitors performance during intensive dashboard operations
- **Data Visualization**: Tracks memory patterns during visualization creation
- **Performance Degradation**: Detects performance decline over extended sessions
- **Component Cleanup**: Validates proper resource cleanup after component unmount

### 3. Resource Monitoring

- **DOM Node Growth**: Tracks DOM tree size changes
- **Event Listener Management**: Monitors event listener lifecycle
- **Timer and Interval Tracking**: Detects hanging timers and intervals
- **WebSocket Connection Management**: Validates connection cleanup

### 4. Advanced Analysis

- **Memory Growth Patterns**: Analyzes memory growth over time
- **Resource Leak Detection**: Identifies various types of resource leaks
- **Performance Trend Analysis**: Calculates performance degradation trends
- **Risk Assessment**: Provides risk levels based on memory usage patterns

## Configuration

### Browser Configuration

Tests are configured to run with specific Chrome flags for memory monitoring:

- `--enable-precise-memory-info`: Enables detailed memory API
- `--js-flags="--expose-gc"`: Exposes garbage collection to JavaScript
- `--max-old-space-size`: Limits memory to detect leaks faster

### Test Thresholds

- **Memory Leak Threshold**: 10MB growth during test execution
- **DOM Node Leak**: 100+ nodes remaining after cleanup
- **Event Listener Leak**: Listeners not properly removed
- **Performance Degradation**: >50MB growth in extended sessions

## Running Tests

### Quick Start

```bash
# Navigate to the memory testing directory
cd tests/ui/memory-leak-testing

# Run all memory tests
./run-memory-tests.sh
```

### Individual Test Suites

```bash
# Run only memory leak detection
npx playwright test --config=memory-leak.config.ts memory-leak-detection.spec.ts

# Run only performance monitoring
npx playwright test --config=memory-leak.config.ts performance-monitoring.spec.ts
```

### Custom Configuration

```bash
# Run with custom base URL
BASE_URL=http://localhost:3000 ./run-memory-tests.sh

# Run with specific browser
npx playwright test --config=memory-leak.config.ts --project=memory-leak-chrome
```

## Test Reports

### Generated Reports

1. **Memory Leak Report** (`memory-leak-report.json`)
   - Summary of detected memory leaks
   - Test-by-test memory analysis
   - Leak threshold comparisons

2. **Comprehensive Analysis** (`memory-analysis-comprehensive.json`)
   - Detailed memory usage statistics
   - Risk assessment and recommendations
   - Client and server-side memory data

3. **Performance Recommendations** (`memory-performance-recommendations.md`)
   - Specific optimization suggestions
   - Best practices for memory management
   - Implementation guidelines

4. **HTML Reports** (`memory-leak-report/index.html`)
   - Interactive test results
   - Visual memory usage charts
   - Detailed test execution logs

### Report Analysis

#### Memory Usage Metrics

- **JS Heap Used**: Current JavaScript memory usage
- **JS Heap Total**: Total allocated JavaScript memory
- **Memory Growth**: Change in memory usage during test
- **Peak Memory**: Maximum memory usage reached

#### Risk Levels

- **Low Risk**: <20MB growth, <60% utilization
- **Medium Risk**: 20-50MB growth, 60-80% utilization
- **High Risk**: >50MB growth, >80% utilization

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Memory Leak Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  memory-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-node@v3
      with:
        node-version: '18'
    - name: Install dependencies
      run: npm ci
    - name: Install Playwright
      run: npx playwright install chromium
    - name: Start application
      run: npm start &
    - name: Wait for application
      run: npx wait-on http://localhost:8000
    - name: Run memory leak tests
      run: |
        cd tests/ui/memory-leak-testing
        ./run-memory-tests.sh
    - name: Upload test reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: memory-test-reports
        path: tests/test_reports/
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    stages {
        stage('Memory Leak Testing') {
            steps {
                sh 'npm ci'
                sh 'npx playwright install chromium'
                sh 'npm start &'
                sh 'npx wait-on http://localhost:8000'
                dir('tests/ui/memory-leak-testing') {
                    sh './run-memory-tests.sh'
                }
            }
            post {
                always {
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'tests/test_reports',
                        reportFiles: 'memory-leak-report/index.html',
                        reportName: 'Memory Leak Test Report'
                    ])
                }
            }
        }
    }
}
```

## Troubleshooting

### Common Issues

#### 1. Memory API Not Available

**Problem**: `performance.memory` returns undefined
**Solution**: Ensure tests run with `--enable-precise-memory-info` flag

#### 2. Garbage Collection Not Working

**Problem**: `window.gc` is not a function
**Solution**: Verify `--js-flags="--expose-gc"` is included in browser launch options

#### 3. High False Positive Rate

**Problem**: Tests report leaks when none exist
**Solution**: Adjust memory thresholds in test configuration based on application baseline

#### 4. Tests Timeout

**Problem**: Tests exceed timeout limits
**Solution**: Increase timeout in `memory-leak.config.ts` or optimize test operations

### Performance Tips

1. **Run Tests Sequentially**: Use `workers: 1` to avoid interference between tests
2. **Limit Memory**: Set `--max-old-space-size` to detect leaks faster
3. **Clean State**: Ensure proper cleanup between test runs
4. **Monitor System Resources**: Ensure adequate system memory for testing

## Best Practices

### For Developers

1. **Regular Testing**: Run memory tests before major releases
2. **Component Lifecycle**: Properly implement cleanup in component lifecycle methods
3. **Event Listeners**: Always remove event listeners when components unmount
4. **WebSocket Management**: Implement proper connection cleanup
5. **Timer Management**: Clear intervals and timeouts when not needed

### For QA Teams

1. **Baseline Establishment**: Establish memory usage baselines for different features
2. **Regression Testing**: Include memory tests in regression test suites
3. **Performance Monitoring**: Monitor memory usage trends over time
4. **Browser Testing**: Test across different browsers as memory behavior varies

## Customization

### Adding Custom Tests

```typescript
test('should detect memory leaks in custom feature', async ({ page }) => {
  const monitor = memoryLeakDetector.createMonitor(page, 'custom-feature');
  
  await page.goto('/custom-feature');
  await monitor.startMonitoring(500);

  // Your custom test logic here
  
  await monitor.forceGarbageCollection();
  monitor.stopMonitoring();

  const analysis = monitor.getMemoryAnalysis();
  expect(analysis.memoryGrowth).toBeLessThan(analysis.leakThreshold);
});
```

### Custom Thresholds

```typescript
// Adjust memory leak threshold
const leakThreshold = 20 * 1024 * 1024; // 20MB

// Custom monitoring intervals
await monitor.startMonitoring(250); // 250ms intervals
```

### Advanced Monitoring

```typescript
// Monitor specific metrics
const customMetrics = await page.evaluate(() => {
  return {
    customObjects: window.myCustomObjects?.length || 0,
    canvasContexts: document.querySelectorAll('canvas').length,
    videoElements: document.querySelectorAll('video').length
  };
});
```

## Contributing

When adding new memory leak tests:

1. Follow the existing test structure and naming conventions
2. Include proper cleanup in test teardown
3. Document any new thresholds or configuration options
4. Update this README with new test descriptions
5. Ensure tests are deterministic and don't depend on timing

## Support

For issues with memory leak testing:

1. Check the troubleshooting section above
2. Review test reports for specific error details
3. Verify browser configuration and flags
4. Ensure proper test environment setup

## Related Documentation

- [Playwright Memory Testing Guide](https://playwright.dev/docs/test-configuration)
- [Chrome DevTools Memory Tab](https://developer.chrome.com/docs/devtools/memory/)
- [Performance API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Performance_API)
- [Memory Management Best Practices](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Memory_Management)
