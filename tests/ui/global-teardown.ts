import { FullConfig } from '@playwright/test';

/**
 * Global teardown for cross-browser testing
 * Cleanup and generate final reports
 */
async function globalTeardown(config: FullConfig) {
  console.log('🧹 Running global teardown...');
  
  const fs = require('fs');
  const path = require('path');
  
  try {
    // Generate cross-browser compatibility summary
    await generateCompatibilitySummary();
    
    // Archive test artifacts
    await archiveTestArtifacts();
    
    // Generate final test report
    await generateFinalReport();
    
    console.log('✅ Global teardown completed successfully');
  } catch (error) {
    console.error('❌ Global teardown failed:', error);
  }
}

/**
 * Generate cross-browser compatibility summary
 */
async function generateCompatibilitySummary() {
  console.log('📊 Generating cross-browser compatibility summary...');
  
  const fs = require('fs');
  const path = require('path');
  
  try {
    // Read test results
    const resultsPath = 'test_reports/playwright-results.json';
    if (!fs.existsSync(resultsPath)) {
      console.warn('⚠️  No test results found for compatibility summary');
      return;
    }
    
    const results = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
    
    // Analyze results by browser/project
    const summary = {
      timestamp: new Date().toISOString(),
      overview: {
        totalTests: 0,
        passedTests: 0,
        failedTests: 0,
        skippedTests: 0,
        passRate: 0
      },
      byBrowser: {},
      byCategory: {
        desktop: { passed: 0, failed: 0, total: 0 },
        mobile: { passed: 0, failed: 0, total: 0 },
        tablet: { passed: 0, failed: 0, total: 0 }
      },
      incompatibilities: [],
      recommendations: []
    };
    
    // Process test suites
    if (results.suites) {
      for (const suite of results.suites) {
        for (const spec of suite.specs || []) {
          for (const test of spec.tests || []) {
            summary.overview.totalTests++;
            
            const projectName = test.projectName || 'unknown';
            if (!summary.byBrowser[projectName]) {
              summary.byBrowser[projectName] = {
                passed: 0,
                failed: 0,
                skipped: 0,
                total: 0,
                issues: []
              };
            }
            
            summary.byBrowser[projectName].total++;
            
            // Determine test outcome
            if (test.outcome === 'passed') {
              summary.overview.passedTests++;
              summary.byBrowser[projectName].passed++;
            } else if (test.outcome === 'failed') {
              summary.overview.failedTests++;
              summary.byBrowser[projectName].failed++;
              
              // Record browser-specific failure
              summary.byBrowser[projectName].issues.push({
                test: spec.title,
                error: test.errors?.[0]?.message || 'Unknown error'
              });
            } else {
              summary.overview.skippedTests++;
              summary.byBrowser[projectName].skipped++;
            }
            
            // Categorize by device type
            if (projectName.includes('Mobile')) {
              summary.byCategory.mobile.total++;
              if (test.outcome === 'passed') summary.byCategory.mobile.passed++;
              if (test.outcome === 'failed') summary.byCategory.mobile.failed++;
            } else if (projectName.includes('iPad') || projectName.includes('Tablet')) {
              summary.byCategory.tablet.total++;
              if (test.outcome === 'passed') summary.byCategory.tablet.passed++;
              if (test.outcome === 'failed') summary.byCategory.tablet.failed++;
            } else {
              summary.byCategory.desktop.total++;
              if (test.outcome === 'passed') summary.byCategory.desktop.passed++;
              if (test.outcome === 'failed') summary.byCategory.desktop.failed++;
            }
          }
        }
      }
    }
    
    // Calculate pass rate
    summary.overview.passRate = summary.overview.totalTests > 0 
      ? (summary.overview.passedTests / summary.overview.totalTests) * 100 
      : 0;
    
    // Identify cross-browser incompatibilities
    summary.incompatibilities = identifyIncompatibilities(summary.byBrowser);
    
    // Generate recommendations
    summary.recommendations = generateCompatibilityRecommendations(summary);
    
    // Save summary
    const summaryPath = 'test_reports/cross-browser/compatibility-summary.json';
    fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
    
    // Generate HTML report
    await generateCompatibilityHTMLReport(summary);
    
    console.log(`  ✅ Compatibility summary generated (${summary.overview.passRate.toFixed(1)}% pass rate)`);
  } catch (error) {
    console.error('  ❌ Failed to generate compatibility summary:', error);
  }
}

/**
 * Identify cross-browser incompatibilities
 */
function identifyIncompatibilities(browserResults: any) {
  const incompatibilities = [];
  const browsers = Object.keys(browserResults);
  
  // Find tests that fail in some browsers but pass in others
  const testFailures = {};
  
  for (const [browser, results] of Object.entries(browserResults) as any) {
    for (const issue of results.issues || []) {
      if (!testFailures[issue.test]) {
        testFailures[issue.test] = { failedIn: [], passedIn: [] };
      }
      testFailures[issue.test].failedIn.push(browser);
    }
  }
  
  // Identify which browsers passed tests that failed in others
  for (const [testName, failures] of Object.entries(testFailures) as any) {
    const passedIn = browsers.filter(browser => !failures.failedIn.includes(browser));
    
    if (failures.failedIn.length > 0 && passedIn.length > 0) {
      incompatibilities.push({
        test: testName,
        failedIn: failures.failedIn,
        passedIn: passedIn,
        severity: failures.failedIn.length >= browsers.length / 2 ? 'high' : 'medium'
      });
    }
  }
  
  return incompatibilities;
}

/**
 * Generate compatibility recommendations
 */
function generateCompatibilityRecommendations(summary: any) {
  const recommendations = [];
  
  // Overall pass rate recommendations
  if (summary.overview.passRate < 90) {
    recommendations.push({
      type: 'critical',
      issue: 'Low overall pass rate',
      description: `Test pass rate is ${summary.overview.passRate.toFixed(1)}%. Target should be >95% for production deployment.`,
      action: 'Review and fix failing tests before deployment'
    });
  }
  
  // Browser-specific recommendations
  for (const [browser, results] of Object.entries(summary.byBrowser) as any) {
    const passRate = results.total > 0 ? (results.passed / results.total) * 100 : 0;
    
    if (passRate < 85) {
      recommendations.push({
        type: 'high',
        issue: `Poor compatibility with ${browser}`,
        description: `${browser} has ${passRate.toFixed(1)}% pass rate with ${results.failed} failures.`,
        action: `Investigate ${browser}-specific issues and implement browser-specific fixes`
      });
    }
  }
  
  // Device category recommendations
  for (const [category, results] of Object.entries(summary.byCategory) as any) {
    const passRate = results.total > 0 ? (results.passed / results.total) * 100 : 0;
    
    if (passRate < 90) {
      recommendations.push({
        type: 'medium',
        issue: `${category} compatibility issues`,
        description: `${category} devices have ${passRate.toFixed(1)}% pass rate.`,
        action: `Review responsive design and ${category}-specific functionality`
      });
    }
  }
  
  // Incompatibility recommendations
  if (summary.incompatibilities.length > 0) {
    const highSeverityCount = summary.incompatibilities.filter(i => i.severity === 'high').length;
    
    recommendations.push({
      type: highSeverityCount > 0 ? 'critical' : 'high',
      issue: 'Cross-browser incompatibilities detected',
      description: `${summary.incompatibilities.length} tests show browser-specific behavior (${highSeverityCount} high severity).`,
      action: 'Implement progressive enhancement and browser-specific polyfills'
    });
  }
  
  return recommendations.sort((a, b) => {
    const priority = { critical: 4, high: 3, medium: 2, low: 1 };
    return priority[b.type] - priority[a.type];
  });
}

/**
 * Generate HTML compatibility report
 */
async function generateCompatibilityHTMLReport(summary: any) {
  const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cross-Browser Compatibility Report - Pynomaly</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #f8fafc; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }
        .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .good { color: #10b981; }
        .warning { color: #f59e0b; }
        .critical { color: #ef4444; }
        .table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .table th, .table td { padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }
        .table th { background: #f9fafb; font-weight: 600; }
        .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.875em; font-weight: 500; }
        .badge.good { background: #d1fae5; color: #065f46; }
        .badge.warning { background: #fef3c7; color: #92400e; }
        .badge.critical { background: #fee2e2; color: #991b1b; }
        .recommendation { background: white; border-radius: 8px; padding: 16px; margin-bottom: 16px; border-left: 4px solid #3b82f6; }
        .recommendation.critical { border-left-color: #ef4444; }
        .recommendation.high { border-left-color: #f59e0b; }
        .recommendation.medium { border-left-color: #10b981; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Cross-Browser Compatibility Report</h1>
            <p>Generated on ${new Date().toLocaleString()}</p>
        </div>
        
        <div class="summary">
            <div class="card">
                <h3>Overall Pass Rate</h3>
                <div class="metric ${summary.overview.passRate >= 95 ? 'good' : summary.overview.passRate >= 85 ? 'warning' : 'critical'}">${summary.overview.passRate.toFixed(1)}%</div>
                <p>${summary.overview.passedTests}/${summary.overview.totalTests} tests passed</p>
            </div>
            <div class="card">
                <h3>Browser Coverage</h3>
                <div class="metric">${Object.keys(summary.byBrowser).length}</div>
                <p>Browsers tested</p>
            </div>
            <div class="card">
                <h3>Incompatibilities</h3>
                <div class="metric ${summary.incompatibilities.length === 0 ? 'good' : summary.incompatibilities.length <= 2 ? 'warning' : 'critical'}">${summary.incompatibilities.length}</div>
                <p>Cross-browser issues</p>
            </div>
            <div class="card">
                <h3>Recommendations</h3>
                <div class="metric">${summary.recommendations.length}</div>
                <p>Action items</p>
            </div>
        </div>
        
        ${generateBrowserTableHTML(summary.byBrowser)}
        ${generateIncompatibilityTableHTML(summary.incompatibilities)}
        ${generateRecommendationsHTML(summary.recommendations)}
    </div>
</body>
</html>`;
  
  const fs = require('fs');
  const htmlPath = 'test_reports/cross-browser/compatibility-report.html';
  fs.writeFileSync(htmlPath, html);
}

function generateBrowserTableHTML(browserResults: any) {
  return `
        <div class="card">
            <h2>Browser Compatibility Results</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Browser/Device</th>
                        <th>Tests</th>
                        <th>Passed</th>
                        <th>Failed</th>
                        <th>Pass Rate</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    ${Object.entries(browserResults).map(([browser, results]: any) => {
                      const passRate = results.total > 0 ? (results.passed / results.total) * 100 : 0;
                      const status = passRate >= 95 ? 'good' : passRate >= 85 ? 'warning' : 'critical';
                      return `
                        <tr>
                            <td><strong>${browser}</strong></td>
                            <td>${results.total}</td>
                            <td>${results.passed}</td>
                            <td>${results.failed}</td>
                            <td>${passRate.toFixed(1)}%</td>
                            <td><span class="badge ${status}">${status.toUpperCase()}</span></td>
                        </tr>
                      `;
                    }).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function generateIncompatibilityTableHTML(incompatibilities: any[]) {
  if (incompatibilities.length === 0) {
    return '<div class="card"><h2>Cross-Browser Incompatibilities</h2><p>No cross-browser incompatibilities detected! 🎉</p></div>';
  }

  return `
        <div class="card">
            <h2>Cross-Browser Incompatibilities</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Test</th>
                        <th>Failed In</th>
                        <th>Passed In</th>
                        <th>Severity</th>
                    </tr>
                </thead>
                <tbody>
                    ${incompatibilities.map(issue => `
                        <tr>
                            <td>${issue.test}</td>
                            <td>${issue.failedIn.join(', ')}</td>
                            <td>${issue.passedIn.join(', ')}</td>
                            <td><span class="badge ${issue.severity === 'high' ? 'critical' : 'warning'}">${issue.severity.toUpperCase()}</span></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function generateRecommendationsHTML(recommendations: any[]) {
  if (recommendations.length === 0) {
    return '<div class="card"><h2>Recommendations</h2><p>No action items. Great cross-browser compatibility! ✅</p></div>';
  }

  return `
        <div class="card">
            <h2>Recommendations</h2>
            ${recommendations.map(rec => `
                <div class="recommendation ${rec.type}">
                    <h3>${rec.issue}</h3>
                    <p>${rec.description}</p>
                    <p><strong>Action:</strong> ${rec.action}</p>
                    <span class="badge ${rec.type}">${rec.type.toUpperCase()} PRIORITY</span>
                </div>
            `).join('')}
        </div>
    `;
}

/**
 * Archive test artifacts
 */
async function archiveTestArtifacts() {
  console.log('📁 Archiving test artifacts...');
  
  const fs = require('fs');
  const path = require('path');
  
  const artifactDirs = [
    'test_reports/playwright-report',
    'test_reports/cross-browser',
    'test_reports/visual-regression'
  ];
  
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const archiveDir = `test_reports/archives/${timestamp}`;
  
  try {
    fs.mkdirSync(archiveDir, { recursive: true });
    
    for (const dir of artifactDirs) {
      if (fs.existsSync(dir)) {
        const targetDir = path.join(archiveDir, path.basename(dir));
        fs.cpSync(dir, targetDir, { recursive: true });
      }
    }
    
    console.log(`  ✅ Artifacts archived to ${archiveDir}`);
  } catch (error) {
    console.error('  ❌ Failed to archive artifacts:', error);
  }
}

/**
 * Generate final test report
 */
async function generateFinalReport() {
  console.log('📄 Generating final test report...');
  
  const fs = require('fs');
  
  const summary = {
    timestamp: new Date().toISOString(),
    testRun: {
      duration: process.hrtime.bigint(),
      environment: process.env.NODE_ENV || 'test',
      ci: !!process.env.CI
    },
    artifacts: {
      htmlReport: 'test_reports/playwright-report/index.html',
      jsonResults: 'test_reports/playwright-results.json',
      compatibilityReport: 'test_reports/cross-browser/compatibility-report.html',
      junitResults: 'test_reports/playwright-junit.xml'
    },
    nextSteps: [
      'Review compatibility report for browser-specific issues',
      'Address any critical or high-priority recommendations',
      'Update browser support documentation',
      'Consider implementing progressive enhancement for failing features'
    ]
  };
  
  try {
    const reportPath = 'test_reports/final-report.json';
    fs.writeFileSync(reportPath, JSON.stringify(summary, null, 2));
    console.log(`  ✅ Final report saved to ${reportPath}`);
  } catch (error) {
    console.error('  ❌ Failed to generate final report:', error);
  }
}

export default globalTeardown;