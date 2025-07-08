#!/usr/bin/env node
/**
 * Browser Compatibility Matrix Generator
 * Analyzes cross-browser test results and generates compatibility matrices
 */

const fs = require('fs').promises;
const path = require('path');

class BrowserCompatibilityMatrix {
  constructor(options = {}) {
    this.resultsDir = options.resultsDir || './test_reports';
    this.outputDir = options.outputDir || './test_reports/compatibility-matrix';
    this.browsers = options.browsers || [
      'Desktop Chrome',
      'Desktop Firefox',
      'Desktop Safari',
      'Desktop Edge',
      'Mobile Chrome',
      'Mobile Safari',
      'Mobile Firefox',
      'iPad',
      'Android Tablet'
    ];

    this.matrix = {
      timestamp: new Date().toISOString(),
      browsers: {},
      features: {},
      tests: {},
      compatibility: {},
      summary: {},
      recommendations: []
    };
  }

  async generate() {
    console.log('= Generating browser compatibility matrix...');

    try {
      await fs.mkdir(this.outputDir, { recursive: true });

      // Load test results
      await this.loadTestResults();

      // Analyze browser capabilities
      await this.analyzeBrowserCapabilities();

      // Generate feature compatibility matrix
      this.generateFeatureMatrix();

      // Analyze cross-browser issues
      this.analyzeCrossBrowserIssues();

      // Generate recommendations
      this.generateRecommendations();

      // Create summary
      this.createSummary();

      // Export results
      await this.exportResults();

      console.log(` Compatibility matrix generated successfully`);
      console.log(`=Ê Results saved to: ${this.outputDir}`);

      return this.matrix;

    } catch (error) {
      console.error('L Failed to generate compatibility matrix:', error);
      throw error;
    }
  }

  async loadTestResults() {
    console.log('=Ä Loading test results...');

    try {
      // Load Playwright results
      const playwrightResults = await this.loadPlaywrightResults();

      // Load browser capabilities
      const capabilities = await this.loadBrowserCapabilities();

      // Load cross-browser test results
      const crossBrowserResults = await this.loadCrossBrowserResults();

      this.matrix.browsers = capabilities || {};
      this.matrix.tests = playwrightResults || {};

      console.log(`  Loaded results for ${Object.keys(this.matrix.browsers).length} browsers`);
      console.log(`  Analyzed ${Object.keys(this.matrix.tests).length} test suites`);

    } catch (error) {
      console.warn(`  Warning: Could not load all test results - ${error.message}`);
    }
  }

  async loadPlaywrightResults() {
    try {
      const resultsPath = path.join(this.resultsDir, 'playwright-results.json');
      const data = await fs.readFile(resultsPath, 'utf8');
      return JSON.parse(data);
    } catch (error) {
      return {};
    }
  }

  async loadBrowserCapabilities() {
    try {
      const capabilitiesPath = path.join(this.resultsDir, 'cross-browser/capabilities.json');
      const data = await fs.readFile(capabilitiesPath, 'utf8');
      return JSON.parse(data);
    } catch (error) {
      return {};
    }
  }

  async loadCrossBrowserResults() {
    try {
      const resultsPath = path.join(this.resultsDir, 'cross-browser/compatibility-summary.json');
      const data = await fs.readFile(resultsPath, 'utf8');
      return JSON.parse(data);
    } catch (error) {
      return {};
    }
  }

  async analyzeBrowserCapabilities() {
    console.log('=
 Analyzing browser capabilities...');

    // Define key web features to check
    const keyFeatures = [
      'es6', 'es2017', 'es2018', 'flexbox', 'grid', 'customProperties',
      'serviceWorker', 'webgl', 'webgl2', 'touchEvents', 'pointerEvents',
      'intersectionObserver', 'webComponents', 'pushManager', 'geolocation'
    ];

    // Initialize feature matrix
    this.matrix.features = {};
    keyFeatures.forEach(feature => {
      this.matrix.features[feature] = {
        supported: [],
        unsupported: [],
        supportRate: 0,
        critical: this.isFeatureCritical(feature)
      };
    });

    // Analyze capabilities for each browser
    Object.entries(this.matrix.browsers.browsers || {}).forEach(([browserName, browserData]) => {
      if (browserData.capabilities) {
        keyFeatures.forEach(feature => {
          const isSupported = browserData.capabilities[feature] === true;

          if (isSupported) {
            this.matrix.features[feature].supported.push(browserName);
          } else {
            this.matrix.features[feature].unsupported.push(browserName);
          }
        });
      }
    });

    // Calculate support rates
    const totalBrowsers = Object.keys(this.matrix.browsers.browsers || {}).length;
    Object.keys(this.matrix.features).forEach(feature => {
      const supportedCount = this.matrix.features[feature].supported.length;
      this.matrix.features[feature].supportRate = totalBrowsers > 0 ?
        (supportedCount / totalBrowsers) * 100 : 0;
    });

    console.log(`  Analyzed ${keyFeatures.length} key web features`);
  }

  isFeatureCritical(feature) {
    const criticalFeatures = [
      'es6', 'flexbox', 'customProperties', 'serviceWorker', 'touchEvents'
    ];
    return criticalFeatures.includes(feature);
  }

  generateFeatureMatrix() {
    console.log('=Ê Generating feature compatibility matrix...');

    this.matrix.compatibility = {};

    // Create browser x feature matrix
    this.browsers.forEach(browser => {
      this.matrix.compatibility[browser] = {};

      Object.keys(this.matrix.features).forEach(feature => {
        const isSupported = this.matrix.features[feature].supported.includes(browser);
        const isCritical = this.matrix.features[feature].critical;

        this.matrix.compatibility[browser][feature] = {
          supported: isSupported,
          critical: isCritical,
          status: isSupported ? 'supported' : (isCritical ? 'missing-critical' : 'missing')
        };
      });
    });

    console.log(`  Generated compatibility matrix for ${this.browsers.length} browsers`);
  }

  analyzeCrossBrowserIssues() {
    console.log('=
 Analyzing cross-browser issues...');

    const issues = {
      criticalMissing: [],
      partialSupport: [],
      performanceIssues: [],
      testFailures: []
    };

    // Find features with poor support
    Object.entries(this.matrix.features).forEach(([feature, data]) => {
      if (data.critical && data.supportRate < 100) {
        issues.criticalMissing.push({
          feature,
          supportRate: data.supportRate,
          unsupported: data.unsupported
        });
      } else if (data.supportRate < 80) {
        issues.partialSupport.push({
          feature,
          supportRate: data.supportRate,
          unsupported: data.unsupported
        });
      }
    });

    // Analyze test failures by browser
    if (this.matrix.tests.suites) {
      const browserFailures = {};

      this.matrix.tests.suites.forEach(suite => {
        suite.specs?.forEach(spec => {
          spec.tests?.forEach(test => {
            const browser = test.projectName || 'unknown';

            if (!browserFailures[browser]) {
              browserFailures[browser] = { total: 0, failed: 0 };
            }

            browserFailures[browser].total++;
            if (test.outcome === 'failed') {
              browserFailures[browser].failed++;
            }
          });
        });
      });

      // Identify browsers with high failure rates
      Object.entries(browserFailures).forEach(([browser, stats]) => {
        const failureRate = (stats.failed / stats.total) * 100;
        if (failureRate > 10) {
          issues.testFailures.push({
            browser,
            failureRate,
            failures: stats.failed,
            total: stats.total
          });
        }
      });
    }

    this.matrix.issues = issues;

    console.log(`  Found ${issues.criticalMissing.length} critical compatibility issues`);
    console.log(`  Found ${issues.partialSupport.length} partial support issues`);
    console.log(`  Found ${issues.testFailures.length} browsers with high failure rates`);
  }

  generateRecommendations() {
    console.log('=¡ Generating recommendations...');

    const recommendations = [];

    // Critical feature recommendations
    this.matrix.issues.criticalMissing.forEach(issue => {
      recommendations.push({
        priority: 'critical',
        type: 'feature-support',
        issue: `Missing critical feature: ${issue.feature}`,
        description: `${issue.feature} is not supported in ${issue.unsupported.join(', ')}`,
        action: `Implement polyfill or fallback for ${issue.feature}`,
        affectedBrowsers: issue.unsupported,
        impact: 'high'
      });
    });

    // Partial support recommendations
    this.matrix.issues.partialSupport.forEach(issue => {
      if (issue.supportRate < 50) {
        recommendations.push({
          priority: 'high',
          type: 'feature-support',
          issue: `Poor support for ${issue.feature}`,
          description: `Only ${issue.supportRate.toFixed(1)}% browser support for ${issue.feature}`,
          action: `Consider progressive enhancement or alternative implementation`,
          affectedBrowsers: issue.unsupported,
          impact: 'medium'
        });
      }
    });

    // Test failure recommendations
    this.matrix.issues.testFailures.forEach(issue => {
      recommendations.push({
        priority: issue.failureRate > 25 ? 'critical' : 'high',
        type: 'test-failure',
        issue: `High test failure rate in ${issue.browser}`,
        description: `${issue.failureRate.toFixed(1)}% test failure rate (${issue.failures}/${issue.total})`,
        action: `Investigate and fix ${issue.browser}-specific issues`,
        affectedBrowsers: [issue.browser],
        impact: 'high'
      });
    });

    // Browser coverage recommendations
    const testedBrowsers = Object.keys(this.matrix.compatibility);
    const missingBrowsers = this.browsers.filter(browser => !testedBrowsers.includes(browser));

    if (missingBrowsers.length > 0) {
      recommendations.push({
        priority: 'medium',
        type: 'coverage',
        issue: 'Incomplete browser coverage',
        description: `Missing test coverage for: ${missingBrowsers.join(', ')}`,
        action: 'Add missing browsers to test configuration',
        affectedBrowsers: missingBrowsers,
        impact: 'medium'
      });
    }

    // Sort recommendations by priority
    recommendations.sort((a, b) => {
      const priorityOrder = { critical: 3, high: 2, medium: 1, low: 0 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });

    this.matrix.recommendations = recommendations;

    console.log(`  Generated ${recommendations.length} recommendations`);
  }

  createSummary() {
    const browsers = Object.keys(this.matrix.compatibility);
    const features = Object.keys(this.matrix.features);

    // Calculate overall compatibility score
    let totalChecks = 0;
    let passedChecks = 0;

    browsers.forEach(browser => {
      features.forEach(feature => {
        totalChecks++;
        if (this.matrix.compatibility[browser]?.[feature]?.supported) {
          passedChecks++;
        }
      });
    });

    const overallCompatibility = totalChecks > 0 ? (passedChecks / totalChecks) * 100 : 0;

    // Count critical issues
    const criticalIssues = this.matrix.recommendations.filter(r => r.priority === 'critical').length;
    const highIssues = this.matrix.recommendations.filter(r => r.priority === 'high').length;

    // Determine status
    let status = 'excellent';
    if (criticalIssues > 0) {
      status = 'critical';
    } else if (highIssues > 2) {
      status = 'warning';
    } else if (overallCompatibility < 90) {
      status = 'needs-improvement';
    }

    this.matrix.summary = {
      overallCompatibility: overallCompatibility,
      status: status,
      browsersChecked: browsers.length,
      featuresChecked: features.length,
      totalRecommendations: this.matrix.recommendations.length,
      criticalIssues: criticalIssues,
      highIssues: highIssues,
      bestSupportedFeatures: this.getBestSupportedFeatures(),
      worstSupportedFeatures: this.getWorstSupportedFeatures(),
      mostCompatibleBrowser: this.getMostCompatibleBrowser(),
      leastCompatibleBrowser: this.getLeastCompatibleBrowser()
    };
  }

  getBestSupportedFeatures() {
    return Object.entries(this.matrix.features)
      .filter(([_, data]) => data.supportRate >= 95)
      .map(([feature, _]) => feature)
      .slice(0, 5);
  }

  getWorstSupportedFeatures() {
    return Object.entries(this.matrix.features)
      .filter(([_, data]) => data.supportRate < 80)
      .sort((a, b) => a[1].supportRate - b[1].supportRate)
      .map(([feature, data]) => ({ feature, supportRate: data.supportRate }))
      .slice(0, 5);
  }

  getMostCompatibleBrowser() {
    const browserScores = Object.entries(this.matrix.compatibility).map(([browser, features]) => {
      const supportedCount = Object.values(features).filter(f => f.supported).length;
      const totalCount = Object.keys(features).length;
      const score = totalCount > 0 ? (supportedCount / totalCount) * 100 : 0;
      return { browser, score };
    });

    return browserScores.sort((a, b) => b.score - a.score)[0];
  }

  getLeastCompatibleBrowser() {
    const browserScores = Object.entries(this.matrix.compatibility).map(([browser, features]) => {
      const supportedCount = Object.values(features).filter(f => f.supported).length;
      const totalCount = Object.keys(features).length;
      const score = totalCount > 0 ? (supportedCount / totalCount) * 100 : 0;
      return { browser, score };
    });

    return browserScores.sort((a, b) => a.score - b.score)[0];
  }

  async exportResults() {
    console.log('=Ä Exporting compatibility matrix...');

    // Save JSON results
    const jsonPath = path.join(this.outputDir, 'compatibility-matrix.json');
    await fs.writeFile(jsonPath, JSON.stringify(this.matrix, null, 2));

    // Generate HTML report
    await this.generateHTMLReport();

    // Generate CSV matrix
    await this.generateCSVMatrix();

    // Generate markdown summary
    await this.generateMarkdownSummary();

    console.log(' All reports exported successfully');
  }

  async generateHTMLReport() {
    const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Browser Compatibility Matrix - Pynomaly</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #f8fafc; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }
        .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .excellent { color: #10b981; }
        .warning { color: #f59e0b; }
        .critical { color: #ef4444; }
        .matrix { overflow-x: auto; margin: 20px 0; }
        .matrix-table { width: 100%; border-collapse: collapse; font-size: 0.875em; }
        .matrix-table th, .matrix-table td { padding: 8px; text-align: center; border: 1px solid #e5e7eb; }
        .matrix-table th { background: #f9fafb; font-weight: 600; }
        .supported { background: #d1fae5; color: #065f46; }
        .missing { background: #fee2e2; color: #991b1b; }
        .missing-critical { background: #dc2626; color: white; }
        .feature-critical { font-weight: bold; }
        .recommendations { margin-top: 40px; }
        .recommendation { background: white; border-radius: 8px; padding: 16px; margin-bottom: 16px; border-left: 4px solid #3b82f6; }
        .recommendation.critical { border-left-color: #ef4444; }
        .recommendation.high { border-left-color: #f59e0b; }
        .recommendation.medium { border-left-color: #10b981; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Browser Compatibility Matrix</h1>
            <p>Generated on ${new Date().toLocaleString()}</p>
        </div>

        <div class="summary">
            <div class="card">
                <h3>Overall Compatibility</h3>
                <div class="metric ${this.matrix.summary.status}">${this.matrix.summary.overallCompatibility.toFixed(1)}%</div>
                <p>Status: ${this.matrix.summary.status.toUpperCase()}</p>
            </div>
            <div class="card">
                <h3>Browser Coverage</h3>
                <div class="metric">${this.matrix.summary.browsersChecked}</div>
                <p>Browsers tested</p>
            </div>
            <div class="card">
                <h3>Features Analyzed</h3>
                <div class="metric">${this.matrix.summary.featuresChecked}</div>
                <p>Web features</p>
            </div>
            <div class="card">
                <h3>Critical Issues</h3>
                <div class="metric ${this.matrix.summary.criticalIssues > 0 ? 'critical' : 'excellent'}">${this.matrix.summary.criticalIssues}</div>
                <p>Require immediate attention</p>
            </div>
        </div>

        <div class="matrix">
            <h2>Feature Support Matrix</h2>
            <table class="matrix-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        ${this.browsers.map(browser => `<th>${browser}</th>`).join('')}
                        <th>Support Rate</th>
                    </tr>
                </thead>
                <tbody>
                    ${Object.entries(this.matrix.features).map(([feature, data]) => `
                        <tr>
                            <td class="${data.critical ? 'feature-critical' : ''}">${feature}</td>
                            ${this.browsers.map(browser => {
                              const support = this.matrix.compatibility[browser]?.[feature];
                              const className = support ? (support.supported ? 'supported' : (support.critical ? 'missing-critical' : 'missing')) : 'missing';
                              const symbol = support ? (support.supported ? '' : 'L') : '?';
                              return `<td class="${className}">${symbol}</td>`;
                            }).join('')}
                            <td>${data.supportRate.toFixed(1)}%</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>

        ${this.generateRecommendationsHTML()}
    </div>
</body>
</html>`;

    const htmlPath = path.join(this.outputDir, 'compatibility-matrix.html');
    await fs.writeFile(htmlPath, html);
  }

  generateRecommendationsHTML() {
    if (this.matrix.recommendations.length === 0) {
      return '<div class="card"><h2>Recommendations</h2><p>Excellent compatibility! No issues found. <‰</p></div>';
    }

    return `
        <div class="recommendations">
            <h2>Recommendations (${this.matrix.recommendations.length})</h2>
            ${this.matrix.recommendations.map(rec => `
                <div class="recommendation ${rec.priority}">
                    <h3>${rec.issue}</h3>
                    <p>${rec.description}</p>
                    <p><strong>Action:</strong> ${rec.action}</p>
                    <p><strong>Affected browsers:</strong> ${rec.affectedBrowsers.join(', ')}</p>
                    <span class="badge ${rec.priority}">${rec.priority.toUpperCase()} PRIORITY</span>
                </div>
            `).join('')}
        </div>
    `;
  }

  async generateCSVMatrix() {
    const headers = ['Feature', ...this.browsers, 'Support Rate'];
    const rows = [headers];

    Object.entries(this.matrix.features).forEach(([feature, data]) => {
      const row = [feature];

      this.browsers.forEach(browser => {
        const support = this.matrix.compatibility[browser]?.[feature];
        row.push(support?.supported ? 'Yes' : 'No');
      });

      row.push(`${data.supportRate.toFixed(1)}%`);
      rows.push(row);
    });

    const csv = rows.map(row => row.join(',')).join('\n');
    const csvPath = path.join(this.outputDir, 'compatibility-matrix.csv');
    await fs.writeFile(csvPath, csv);
  }

  async generateMarkdownSummary() {
    const md = `
# Browser Compatibility Matrix Summary

Generated on: ${new Date().toLocaleString()}

## Overview

- **Overall Compatibility**: ${this.matrix.summary.overallCompatibility.toFixed(1)}%
- **Status**: ${this.matrix.summary.status.toUpperCase()}
- **Browsers Tested**: ${this.matrix.summary.browsersChecked}
- **Features Analyzed**: ${this.matrix.summary.featuresChecked}
- **Critical Issues**: ${this.matrix.summary.criticalIssues}

## Best Supported Features

${this.matrix.summary.bestSupportedFeatures.map(feature => `-  ${feature}`).join('\n')}

## Features Needing Attention

${this.matrix.summary.worstSupportedFeatures.map(item => `- L ${item.feature} (${item.supportRate.toFixed(1)}% support)`).join('\n')}

## Browser Compatibility

- **Most Compatible**: ${this.matrix.summary.mostCompatibleBrowser?.browser} (${this.matrix.summary.mostCompatibleBrowser?.score.toFixed(1)}%)
- **Least Compatible**: ${this.matrix.summary.leastCompatibleBrowser?.browser} (${this.matrix.summary.leastCompatibleBrowser?.score.toFixed(1)}%)

## Top Recommendations

${this.matrix.recommendations.slice(0, 5).map((rec, index) =>
  `${index + 1}. **${rec.priority.toUpperCase()}**: ${rec.issue}\n   - ${rec.action}`
).join('\n\n')}

## Next Steps

1. Address critical compatibility issues immediately
2. Implement polyfills for missing features
3. Consider progressive enhancement strategies
4. Update browser support documentation
5. Set up automated compatibility monitoring
`;

    const mdPath = path.join(this.outputDir, 'compatibility-summary.md');
    await fs.writeFile(mdPath, md.trim());
  }
}

// CLI execution
if (require.main === module) {
  const matrix = new BrowserCompatibilityMatrix({
    resultsDir: process.argv[2] || './test_reports',
    outputDir: process.argv[3] || './test_reports/compatibility-matrix'
  });

  matrix.generate()
    .then(results => {
      console.log(`\n=Ê Compatibility matrix generated with ${results.summary.overallCompatibility.toFixed(1)}% overall compatibility`);
      process.exit(results.summary.criticalIssues > 0 ? 1 : 0);
    })
    .catch(error => {
      console.error('Compatibility matrix generation failed:', error);
      process.exit(1);
    });
}

module.exports = BrowserCompatibilityMatrix;
