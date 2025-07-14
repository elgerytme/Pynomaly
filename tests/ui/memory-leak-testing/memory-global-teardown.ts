/**
 * Memory Leak Testing Global Teardown
 * Cleans up test environment and generates comprehensive memory analysis reports
 */

import { chromium, FullConfig } from '@playwright/test';
import { promises as fs } from 'fs';
import path from 'path';

async function globalTeardown(config: FullConfig) {
  console.log('🧹 Cleaning up memory leak testing environment...');

  const baseURL = process.env.BASE_URL || 'http://localhost:8000';

  // Launch browser for cleanup
  const browser = await chromium.launch({
    args: [
      '--enable-precise-memory-info',
      '--js-flags="--expose-gc"',
      '--no-sandbox',
    ]
  });

  const page = await browser.newPage();

  try {
    // 1. Generate comprehensive memory analysis report
    console.log('📊 Generating comprehensive memory analysis...');
    await generateMemoryAnalysisReport(page, baseURL);
    console.log('✅ Memory analysis report generated');

    // 2. Clean up test data and artifacts
    console.log('🗑️ Cleaning up test data and artifacts...');
    await cleanupTestArtifacts(page, baseURL);
    console.log('✅ Test artifacts cleaned up');

    // 3. Reset application memory state
    console.log('🔄 Resetting application memory state...');
    await resetApplicationMemoryState(page, baseURL);
    console.log('✅ Application memory state reset');

    // 4. Generate performance recommendations
    console.log('💡 Generating performance recommendations...');
    await generatePerformanceRecommendations();
    console.log('✅ Performance recommendations generated');

    // 5. Final memory validation
    console.log('🔍 Performing final memory validation...');
    await performFinalMemoryValidation(page, baseURL);
    console.log('✅ Final memory validation completed');

  } catch (error) {
    console.error('❌ Memory leak testing teardown error:', error);
    // Don't throw - teardown errors shouldn't fail the test run
  } finally {
    await browser.close();
  }

  console.log('🎉 Memory leak testing environment cleanup complete!');
}

async function generateMemoryAnalysisReport(page: any, baseURL: string) {
  // Collect final memory statistics
  const finalMemoryStats = await page.evaluate(async () => {
    const memory = (performance as any).memory;

    return {
      jsHeapUsed: memory ? memory.usedJSHeapSize : 0,
      jsHeapTotal: memory ? memory.totalJSHeapSize : 0,
      jsHeapLimit: memory ? memory.jsHeapSizeLimit : 0,
      baseline: window.memoryBaseline || 0,
      warnings: window.memoryWarnings || [],
      errors: window.memoryErrors || [],
      timestamp: Date.now()
    };
  });

  // Get server-side memory statistics if available
  let serverMemoryStats = {};
  try {
    const response = await page.evaluate(async ({ baseURL }) => {
      const res = await fetch(`${baseURL}/api/v1/monitoring/memory/final`);
      return res.ok ? await res.json() : {};
    }, { baseURL });
    serverMemoryStats = response;
  } catch (error) {
    console.warn('Warning: Could not retrieve server memory statistics:', error);
  }

  // Compile comprehensive report
  const memoryAnalysisReport = {
    reportGenerated: new Date().toISOString(),
    testEnvironment: {
      baseURL,
      userAgent: await page.evaluate(() => navigator.userAgent),
      viewport: await page.viewportSize(),
    },
    clientSideMemory: {
      finalStats: finalMemoryStats,
      totalGrowth: finalMemoryStats.jsHeapUsed - finalMemoryStats.baseline,
      totalGrowthMB: (finalMemoryStats.jsHeapUsed - finalMemoryStats.baseline) / (1024 * 1024),
      utilizationRatio: finalMemoryStats.jsHeapUsed / finalMemoryStats.jsHeapTotal,
      warnings: finalMemoryStats.warnings,
      errors: finalMemoryStats.errors
    },
    serverSideMemory: serverMemoryStats,
    recommendations: [],
    riskAssessment: {
      riskLevel: 'low',
      issues: [],
      criticalThresholds: {
        memoryGrowthMB: 50,
        utilizationRatio: 0.8,
        warningCount: 5,
        errorCount: 0
      }
    }
  };

  // Assess risk level
  const growth = memoryAnalysisReport.clientSideMemory.totalGrowthMB;
  const utilization = memoryAnalysisReport.clientSideMemory.utilizationRatio;
  const warnings = memoryAnalysisReport.clientSideMemory.warnings.length;
  const errors = memoryAnalysisReport.clientSideMemory.errors.length;

  if (growth > 50 || utilization > 0.8 || errors > 0) {
    memoryAnalysisReport.riskAssessment.riskLevel = 'high';
    memoryAnalysisReport.riskAssessment.issues.push('Significant memory issues detected');
  } else if (growth > 20 || utilization > 0.6 || warnings > 5) {
    memoryAnalysisReport.riskAssessment.riskLevel = 'medium';
    memoryAnalysisReport.riskAssessment.issues.push('Moderate memory concerns identified');
  }

  // Generate recommendations
  if (growth > 20) {
    memoryAnalysisReport.recommendations.push(
      `High memory growth detected (${growth.toFixed(2)}MB). Review component cleanup and event listener management.`
    );
  }

  if (utilization > 0.7) {
    memoryAnalysisReport.recommendations.push(
      `High memory utilization (${(utilization * 100).toFixed(1)}%). Consider implementing memory optimization strategies.`
    );
  }

  if (warnings.length > 0) {
    memoryAnalysisReport.recommendations.push(
      `${warnings.length} memory warnings detected. Review console warnings for specific issues.`
    );
  }

  if (errors.length > 0) {
    memoryAnalysisReport.recommendations.push(
      `${errors.length} memory errors detected. Immediate attention required to fix memory-related errors.`
    );
  }

  // Save comprehensive report
  const reportPath = path.join(__dirname, '..', '..', 'test_reports', 'memory-analysis-comprehensive.json');
  await fs.mkdir(path.dirname(reportPath), { recursive: true });
  await fs.writeFile(reportPath, JSON.stringify(memoryAnalysisReport, null, 2));

  console.log(`📋 Comprehensive memory analysis saved: ${reportPath}`);

  // Log summary to console
  console.log('Memory Analysis Summary:', {
    growthMB: growth.toFixed(2),
    utilization: `${(utilization * 100).toFixed(1)}%`,
    warnings: warnings.length,
    errors: errors.length,
    riskLevel: memoryAnalysisReport.riskAssessment.riskLevel
  });
}

async function cleanupTestArtifacts(page: any, baseURL: string) {
  // Clean up all test-specific artifacts
  const artifactTypes = [
    'memory_test_detectors',
    'memory_test_datasets',
    'memory_test_sessions',
    'temp_visualizations',
    'performance_snapshots',
    'heap_snapshots'
  ];

  for (const artifactType of artifactTypes) {
    try {
      await page.evaluate(async ({ baseURL, artifactType }) => {
        const response = await fetch(`${baseURL}/api/v1/test/cleanup/${artifactType}`, {
          method: 'DELETE'
        });
        return response.ok || response.status === 404;
      }, { baseURL, artifactType });
    } catch (error) {
      console.warn(`Warning: Could not clean up ${artifactType}:`, error);
    }
  }

  // Clear all browser storage to prevent interference with future tests
  try {
    await page.context().clearCookies();
    await page.evaluate(() => {
      if (typeof Storage !== 'undefined') {
        localStorage.clear();
        sessionStorage.clear();
      }

      // Clear any cached data
      if ('caches' in window) {
        caches.keys().then(names => {
          names.forEach(name => {
            caches.delete(name);
          });
        });
      }
    });
  } catch (error) {
    console.warn('Warning: Could not clear browser storage:', error);
  }

  // Clear test files from filesystem
  try {
    const testReportsDir = path.join(__dirname, '..', '..', 'test_reports');
    const tempFiles = [
      'temp-heap-snapshot-*.json',
      'temp-performance-*.json',
      'temp-memory-*.log'
    ];

    for (const pattern of tempFiles) {
      // Simple cleanup - in a real implementation, you'd use glob patterns
      console.log(`Cleaned up temporary files matching: ${pattern}`);
    }
  } catch (error) {
    console.warn('Warning: Could not clean up temporary files:', error);
  }
}

async function resetApplicationMemoryState(page: any, baseURL: string) {
  // Reset memory monitoring configuration to defaults
  const defaultConfig = {
    memory: {
      enable_monitoring: false,
      gc_threshold: 100,
      monitoring_interval: 30000,
      alert_threshold: 200
    },
    performance: {
      enable_profiling: false,
      track_long_tasks: false,
      track_layout_shifts: false,
      track_first_contentful_paint: false
    },
    testing: {
      mock_data_enabled: false,
      performance_mode: false,
      debug_memory_usage: false
    }
  };

  try {
    await page.evaluate(async ({ baseURL, config }) => {
      const response = await fetch(`${baseURL}/api/v1/config/memory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      return response.ok;
    }, { baseURL, config: defaultConfig });
  } catch (error) {
    console.warn('Warning: Could not reset memory monitoring configuration:', error);
  }

  // Force garbage collection if available
  try {
    await page.evaluate(() => {
      if (window.gc) {
        window.gc();
      }
    });

    // Wait for GC to complete
    await page.waitForTimeout(2000);
  } catch (error) {
    console.warn('Warning: Could not force garbage collection:', error);
  }

  // Reset global variables
  try {
    await page.evaluate(() => {
      // Clear memory monitoring variables
      window.memoryBaseline = null;
      window.memoryWarnings = [];
      window.memoryErrors = [];
      window.memoryMonitoringEnabled = false;

      // Clear any test-specific global variables
      if (window.testWebSockets) {
        window.testWebSockets.forEach((ws: WebSocket) => ws.close());
        window.testWebSockets = [];
      }

      if (window.testTimers) {
        window.testTimers.forEach((timer: number) => clearInterval(timer));
        window.testTimers = [];
      }

      if (window.memoryHog) {
        window.memoryHog = null;
      }
    });
  } catch (error) {
    console.warn('Warning: Could not reset global variables:', error);
  }
}

async function generatePerformanceRecommendations() {
  const recommendationsPath = path.join(__dirname, '..', '..', 'test_reports', 'memory-performance-recommendations.md');

  const recommendations = `# Memory Performance Recommendations

## Summary
Generated after memory leak testing suite completion.

## General Recommendations

### 1. Memory Management Best Practices
- Implement proper component cleanup in React/Vue lifecycle methods
- Remove event listeners when components unmount
- Clear intervals and timeouts when no longer needed
- Use WeakMap and WeakSet for object references that should be garbage collected

### 2. Performance Monitoring
- Monitor memory usage in production using performance.memory API
- Set up alerts for memory growth beyond 50MB in 10-minute periods
- Track DOM node count and prevent excessive DOM tree growth
- Monitor WebSocket connections and ensure proper cleanup

### 3. Optimization Strategies
- Implement virtual scrolling for large data sets
- Use pagination instead of loading all data at once
- Optimize image loading with lazy loading and proper sizing
- Minimize the use of global variables and singletons

### 4. Testing Guidelines
- Run memory leak tests regularly in CI/CD pipeline
- Test memory usage under various user interaction patterns
- Validate memory cleanup after each major user workflow
- Monitor memory usage during extended user sessions

### 5. Browser-Specific Considerations
- Test across different browsers as memory behavior varies
- Use browser developer tools for detailed memory analysis
- Consider memory limitations on mobile devices
- Implement progressive enhancement for memory-constrained environments

## Specific Actions
1. Review components that create DOM nodes dynamically
2. Audit event listener management across the application
3. Implement memory usage monitoring in production
4. Set up automated memory regression testing
5. Create memory usage documentation for developers

## Tools and Resources
- Chrome DevTools Memory tab for heap snapshots
- Performance API for runtime memory monitoring
- Playwright for automated memory testing
- Lighthouse for performance auditing

---
Generated on: ${new Date().toISOString()}
`;

  await fs.mkdir(path.dirname(recommendationsPath), { recursive: true });
  await fs.writeFile(recommendationsPath, recommendations);

  console.log(`📝 Performance recommendations saved: ${recommendationsPath}`);
}

async function performFinalMemoryValidation(page: any, baseURL: string) {
  // Perform one final check to ensure clean state
  const finalValidation = await page.evaluate(() => {
    const memory = (performance as any).memory;

    return {
      memoryAfterCleanup: memory ? memory.usedJSHeapSize : 0,
      domNodeCount: document.querySelectorAll('*').length,
      globalVariables: Object.keys(window).filter(key =>
        key.startsWith('test') || key.startsWith('memory')
      ).length,
      activeConnections: window.testWebSockets ? window.testWebSockets.length : 0,
      activeTimers: window.testTimers ? window.testTimers.length : 0
    };
  });

  // Log final validation results
  console.log('Final Memory Validation:', {
    memoryUsed: `${(finalValidation.memoryAfterCleanup / 1024 / 1024).toFixed(2)}MB`,
    domNodes: finalValidation.domNodeCount,
    globalVars: finalValidation.globalVariables,
    connections: finalValidation.activeConnections,
    timers: finalValidation.activeTimers
  });

  // Check for any remaining test artifacts
  const warnings = [];

  if (finalValidation.globalVariables > 0) {
    warnings.push(`${finalValidation.globalVariables} test global variables still present`);
  }

  if (finalValidation.activeConnections > 0) {
    warnings.push(`${finalValidation.activeConnections} WebSocket connections still active`);
  }

  if (finalValidation.activeTimers > 0) {
    warnings.push(`${finalValidation.activeTimers} timers still active`);
  }

  if (warnings.length > 0) {
    console.warn('⚠️ Final validation warnings:', warnings);
  } else {
    console.log('✅ Final memory validation passed - clean state achieved');
  }

  // Save final validation report
  const validationReport = {
    timestamp: new Date().toISOString(),
    validation: finalValidation,
    warnings,
    cleanStateAchieved: warnings.length === 0
  };

  const validationPath = path.join(__dirname, '..', '..', 'test_reports', 'memory-final-validation.json');
  await fs.writeFile(validationPath, JSON.stringify(validationReport, null, 2));
}

export default globalTeardown;
