#!/usr/bin/env node
/**
 * Performance Regression Detector for anomaly_detection Web UI
 * Compares current performance metrics against historical baselines
 * and detects performance regressions
 */

const fs = require('fs').promises;
const path = require('path');

class PerformanceRegressionDetector {
  constructor(options = {}) {
    this.baselineDir = options.baselineDir || './test_reports/performance-baselines';
    this.currentDir = options.currentDir || './test_reports/performance';
    this.outputDir = options.outputDir || './test_reports/regression-analysis';
    this.thresholds = {
      performance: 5,      // 5% degradation threshold
      lcp: 200,           // 200ms threshold for LCP
      fid: 20,            // 20ms threshold for FID
      cls: 0.02,          // 0.02 threshold for CLS
      bundleSize: 0.1,    // 10% bundle size increase
      ...options.thresholds
    };

    this.analysis = {
      timestamp: new Date().toISOString(),
      regressions: [],
      improvements: [],
      stable: [],
      summary: {},
      recommendation: ''
    };
  }

  async detect() {
    console.log('=
 Starting performance regression detection...');

    try {
      await fs.mkdir(this.outputDir, { recursive: true });

      // Load baseline data
      const baseline = await this.loadBaseline();
      if (!baseline) {
        console.log('= No baseline found. Current results will be saved as new baseline.');
        await this.saveAsBaseline();
        return this.analysis;
      }

      // Load current results
      const current = await this.loadCurrentResults();
      if (!current) {
        throw new Error('No current performance results found');
      }

      // Compare metrics
      await this.comparePerformanceScores(baseline, current);
      await this.compareCoreWebVitals(baseline, current);
      await this.compareBundleSizes(baseline, current);

      // Generate analysis summary
      this.generateSummary();

      // Save results
      await this.saveResults();

      // Update baseline if no significant regressions
      if (this.shouldUpdateBaseline()) {
        await this.updateBaseline(current);
      }

      console.log(` Regression detection completed. ${this.analysis.regressions.length} regressions found.`);
      return this.analysis;

    } catch (error) {
      console.error('L Regression detection failed:', error);
      throw error;
    }
  }

  async loadBaseline() {
    try {
      const baselinePath = path.join(this.baselineDir, 'performance-baseline.json');
      const baselineData = await fs.readFile(baselinePath, 'utf8');
      return JSON.parse(baselineData);
    } catch (error) {
      return null;
    }
  }

  async loadCurrentResults() {
    try {
      const currentPath = path.join(this.currentDir, 'performance-test-results.json');
      const currentData = await fs.readFile(currentPath, 'utf8');
      return JSON.parse(currentData);
    } catch (error) {
      return null;
    }
  }

  async comparePerformanceScores(baseline, current) {
    console.log('= Comparing Lighthouse performance scores...');

    // Compare overall performance scores
    for (const currentPage of current.testPages) {
      const baselinePage = baseline.testPages?.find(p => p.name === currentPage.name);
      if (!baselinePage) continue;

      const categories = ['performance', 'accessibility', 'bestPractices', 'seo', 'pwa'];

      for (const category of categories) {
        const baselineScore = baselinePage.scores[category];
        const currentScore = currentPage.scores[category];
        const difference = currentScore - baselineScore;
        const percentChange = (difference / baselineScore) * 100;

        if (Math.abs(percentChange) >= this.thresholds.performance) {
          const change = {
            type: 'lighthouse-score',
            page: currentPage.name,
            category: category,
            baseline: baselineScore,
            current: currentScore,
            difference: difference,
            percentChange: percentChange,
            severity: this.calculateSeverity(Math.abs(percentChange), category)
          };

          if (difference < 0) {
            this.analysis.regressions.push(change);
          } else {
            this.analysis.improvements.push(change);
          }
        } else {
          this.analysis.stable.push({
            type: 'lighthouse-score',
            page: currentPage.name,
            category: category,
            change: 'stable'
          });
        }
      }
    }
  }

  async compareCoreWebVitals(baseline, current) {
    console.log(' Comparing Core Web Vitals...');

    const metrics = ['lcp', 'fid', 'cls'];

    for (const currentPage of current.testPages) {
      const baselinePage = baseline.testPages?.find(p => p.name === currentPage.name);
      if (!baselinePage) continue;

      for (const metric of metrics) {
        const baselineValue = baselinePage.metrics[metric];
        const currentValue = currentPage.metrics[metric];
        const difference = currentValue - baselineValue;
        const threshold = this.thresholds[metric];

        if (Math.abs(difference) >= threshold) {
          const change = {
            type: 'core-web-vitals',
            page: currentPage.name,
            metric: metric,
            baseline: baselineValue,
            current: currentValue,
            difference: difference,
            threshold: threshold,
            severity: this.calculateCWVSeverity(Math.abs(difference), metric)
          };

          if (this.isMetricRegression(metric, difference)) {
            this.analysis.regressions.push(change);
          } else {
            this.analysis.improvements.push(change);
          }
        } else {
          this.analysis.stable.push({
            type: 'core-web-vitals',
            page: currentPage.name,
            metric: metric,
            change: 'stable'
          });
        }
      }
    }
  }

  async compareBundleSizes(baseline, current) {
    console.log('= Comparing bundle sizes...');

    if (!baseline.bundleAnalysis || !current.bundleAnalysis) {
      console.warn('Bundle analysis data not available for comparison');
      return;
    }

    const bundleTypes = ['javascript', 'css', 'images', 'fonts'];

    for (const bundleType of bundleTypes) {
      const baselineBundle = baseline.bundleAnalysis[bundleType];
      const currentBundle = current.bundleAnalysis[bundleType];

      if (!baselineBundle || !currentBundle || baselineBundle.error || currentBundle.error) {
        continue;
      }

      const baselineSize = baselineBundle.totalSize;
      const currentSize = currentBundle.totalSize;
      const difference = currentSize - baselineSize;
      const percentChange = (difference / baselineSize) * 100;

      if (Math.abs(percentChange) >= this.thresholds.bundleSize * 100) {
        const change = {
          type: 'bundle-size',
          bundleType: bundleType,
          baseline: baselineSize,
          current: currentSize,
          difference: difference,
          percentChange: percentChange,
          severity: this.calculateBundleSeverity(Math.abs(percentChange))
        };

        if (difference > 0) {
          this.analysis.regressions.push(change);
        } else {
          this.analysis.improvements.push(change);
        }
      } else {
        this.analysis.stable.push({
          type: 'bundle-size',
          bundleType: bundleType,
          change: 'stable'
        });
      }
    }
  }

  generateSummary() {
    const totalChanges = this.analysis.regressions.length + this.analysis.improvements.length;
    const regressionCount = this.analysis.regressions.length;
    const improvementCount = this.analysis.improvements.length;

    // Calculate severity distribution
    const severityDistribution = {
      critical: 0,
      major: 0,
      minor: 0
    };

    this.analysis.regressions.forEach(regression => {
      severityDistribution[regression.severity]++;
    });

    // Determine overall status
    let overallStatus = 'good';
    if (severityDistribution.critical > 0) {
      overallStatus = 'critical';
    } else if (severityDistribution.major > 2) {
      overallStatus = 'major';
    } else if (regressionCount > improvementCount) {
      overallStatus = 'warning';
    }

    // Generate recommendation
    let recommendation = '';
    if (overallStatus === 'critical') {
      recommendation = 'BLOCK DEPLOYMENT: Critical performance regressions detected. Immediate action required.';
    } else if (overallStatus === 'major') {
      recommendation = 'REVIEW REQUIRED: Multiple major regressions detected. Consider postponing deployment.';
    } else if (overallStatus === 'warning') {
      recommendation = 'MONITOR CLOSELY: Some performance degradation detected. Review and monitor.';
    } else {
      recommendation = 'DEPLOYMENT APPROVED: No significant performance regressions detected.';
    }

    this.analysis.summary = {
      totalChanges,
      regressionCount,
      improvementCount,
      stableCount: this.analysis.stable.length,
      severityDistribution,
      overallStatus,
      recommendation
    };

    this.analysis.recommendation = recommendation;
  }

  async saveResults() {
    // Save detailed analysis
    const analysisPath = path.join(this.outputDir, 'regression-analysis.json');
    await fs.writeFile(analysisPath, JSON.stringify(this.analysis, null, 2));

    // Generate HTML report
    await this.generateHTMLReport();

    // Generate CI-friendly summary
    await this.generateCISummary();

    console.log('= Regression analysis reports generated:');
    console.log(`  - JSON: ${analysisPath}`);
    console.log(`  - HTML: ${path.join(this.outputDir, 'regression-report.html')}`);
    console.log(`  - CI Summary: ${path.join(this.outputDir, 'ci-summary.txt')}`);
  }

  async generateHTMLReport() {
    const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Regression Report - anomaly_detection</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #f8fafc; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .status-banner { padding: 20px; border-radius: 8px; margin-bottom: 30px; text-align: center; font-weight: bold; }
        .status-good { background: #d1fae5; color: #065f46; border: 2px solid #10b981; }
        .status-warning { background: #fef3c7; color: #92400e; border: 2px solid #f59e0b; }
        .status-major { background: #fed7d7; color: #c53030; border: 2px solid #e53e3e; }
        .status-critical { background: #fee2e2; color: #991b1b; border: 2px solid #ef4444; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }
        .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .regression { color: #ef4444; }
        .improvement { color: #10b981; }
        .stable { color: #6b7280; }
        .table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .table th, .table td { padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }
        .table th { background: #f9fafb; font-weight: 600; }
        .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.875em; font-weight: 500; }
        .badge.critical { background: #fee2e2; color: #991b1b; }
        .badge.major { background: #fed7d7; color: #c53030; }
        .badge.minor { background: #fef3c7; color: #92400e; }
        .section { margin-bottom: 40px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Performance Regression Analysis</h1>
            <p>Generated on ${new Date().toLocaleString()}</p>
        </div>

        <div class="status-banner status-${this.analysis.summary.overallStatus}">
            ${this.analysis.recommendation}
        </div>

        <div class="summary">
            <div class="card">
                <h3>Regressions</h3>
                <div class="metric regression">${this.analysis.summary.regressionCount}</div>
                <p>Performance degradations</p>
            </div>
            <div class="card">
                <h3>Improvements</h3>
                <div class="metric improvement">${this.analysis.summary.improvementCount}</div>
                <p>Performance improvements</p>
            </div>
            <div class="card">
                <h3>Stable</h3>
                <div class="metric stable">${this.analysis.summary.stableCount}</div>
                <p>No significant change</p>
            </div>
            <div class="card">
                <h3>Critical Issues</h3>
                <div class="metric ${this.analysis.summary.severityDistribution.critical > 0 ? 'regression' : 'stable'}">${this.analysis.summary.severityDistribution.critical}</div>
                <p>Require immediate attention</p>
            </div>
        </div>

        ${this.generateRegressionsTableHTML()}
        ${this.generateImprovementsTableHTML()}
    </div>
</body>
</html>`;

    const htmlPath = path.join(this.outputDir, 'regression-report.html');
    await fs.writeFile(htmlPath, html);
  }

  generateRegressionsTableHTML() {
    if (this.analysis.regressions.length === 0) {
      return '<div class="section"><div class="card"><h2>Regressions</h2><p>No performance regressions detected! <</p></div></div>';
    }

    return `
        <div class="section">
            <div class="card">
                <h2>Performance Regressions (${this.analysis.regressions.length})</h2>
                <table class="table">
                    <thead>
                        <tr>
                            <th>Type</th>
                            <th>Page/Bundle</th>
                            <th>Metric</th>
                            <th>Baseline</th>
                            <th>Current</th>
                            <th>Change</th>
                            <th>Severity</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${this.analysis.regressions.map(regression => `
                            <tr>
                                <td>${regression.type}</td>
                                <td>${regression.page || regression.bundleType || '-'}</td>
                                <td>${regression.category || regression.metric || regression.bundleType || '-'}</td>
                                <td>${this.formatValue(regression.baseline, regression.type)}</td>
                                <td>${this.formatValue(regression.current, regression.type)}</td>
                                <td>${this.formatChange(regression)}</td>
                                <td><span class="badge ${regression.severity}">${regression.severity.toUpperCase()}</span></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;
  }

  generateImprovementsTableHTML() {
    if (this.analysis.improvements.length === 0) {
      return '';
    }

    return `
        <div class="section">
            <div class="card">
                <h2>Performance Improvements (${this.analysis.improvements.length})</h2>
                <table class="table">
                    <thead>
                        <tr>
                            <th>Type</th>
                            <th>Page/Bundle</th>
                            <th>Metric</th>
                            <th>Baseline</th>
                            <th>Current</th>
                            <th>Improvement</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${this.analysis.improvements.map(improvement => `
                            <tr>
                                <td>${improvement.type}</td>
                                <td>${improvement.page || improvement.bundleType || '-'}</td>
                                <td>${improvement.category || improvement.metric || improvement.bundleType || '-'}</td>
                                <td>${this.formatValue(improvement.baseline, improvement.type)}</td>
                                <td>${this.formatValue(improvement.current, improvement.type)}</td>
                                <td style="color: #10b981;">${this.formatChange(improvement)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;
  }

  async generateCISummary() {
    const summary = `
PERFORMANCE REGRESSION ANALYSIS
==============================

Status: ${this.analysis.summary.overallStatus.toUpperCase()}
Recommendation: ${this.analysis.recommendation}

SUMMARY:
- Regressions: ${this.analysis.summary.regressionCount}
- Improvements: ${this.analysis.summary.improvementCount}
- Stable: ${this.analysis.summary.stableCount}

SEVERITY BREAKDOWN:
- Critical: ${this.analysis.summary.severityDistribution.critical}
- Major: ${this.analysis.summary.severityDistribution.major}
- Minor: ${this.analysis.summary.severityDistribution.minor}

${this.analysis.regressions.length > 0 ?
  `\nCRITICAL REGRESSIONS:\n${this.analysis.regressions
    .filter(r => r.severity === 'critical')
    .map(r => `- ${r.type}: ${r.page || r.bundleType} ${r.category || r.metric} (${this.formatChange(r)})`)
    .join('\n')}` : ''}

Generated: ${new Date().toISOString()}
`;

    const summaryPath = path.join(this.outputDir, 'ci-summary.txt');
    await fs.writeFile(summaryPath, summary.trim());
  }

  async saveAsBaseline() {
    const currentResults = await this.loadCurrentResults();
    if (currentResults) {
      await fs.mkdir(this.baselineDir, { recursive: true });
      const baselinePath = path.join(this.baselineDir, 'performance-baseline.json');
      await fs.writeFile(baselinePath, JSON.stringify(currentResults, null, 2));
      console.log('= Current results saved as new baseline');
    }
  }

  async updateBaseline(currentResults) {
    await fs.mkdir(this.baselineDir, { recursive: true });
    const baselinePath = path.join(this.baselineDir, 'performance-baseline.json');
    await fs.writeFile(baselinePath, JSON.stringify(currentResults, null, 2));
    console.log('= Baseline updated with current results');
  }

  shouldUpdateBaseline() {
    // Don't update baseline if there are critical or major regressions
    return this.analysis.summary.severityDistribution.critical === 0 &&
           this.analysis.summary.severityDistribution.major <= 1;
  }

  // Helper methods
  calculateSeverity(percentChange, category) {
    if (category === 'performance' || category === 'accessibility') {
      if (percentChange >= 15) return 'critical';
      if (percentChange >= 10) return 'major';
      return 'minor';
    }

    if (percentChange >= 20) return 'critical';
    if (percentChange >= 10) return 'major';
    return 'minor';
  }

  calculateCWVSeverity(difference, metric) {
    const severityThresholds = {
      lcp: { critical: 1000, major: 500 },
      fid: { critical: 100, major: 50 },
      cls: { critical: 0.1, major: 0.05 }
    };

    const thresholds = severityThresholds[metric];
    if (difference >= thresholds.critical) return 'critical';
    if (difference >= thresholds.major) return 'major';
    return 'minor';
  }

  calculateBundleSeverity(percentChange) {
    if (percentChange >= 25) return 'critical';
    if (percentChange >= 15) return 'major';
    return 'minor';
  }

  isMetricRegression(metric, difference) {
    // For CLS, lower is better, so positive change is regression
    // For LCP and FID, lower is better, so positive change is regression
    return difference > 0;
  }

  formatValue(value, type) {
    if (type === 'lighthouse-score') return `${value}%`;
    if (type === 'core-web-vitals') return `${Math.round(value)}ms`;
    if (type === 'bundle-size') return this.formatBytes(value);
    return value;
  }

  formatChange(change) {
    if (change.percentChange !== undefined) {
      return `${change.percentChange > 0 ? '+' : ''}${change.percentChange.toFixed(1)}%`;
    }
    if (change.difference !== undefined) {
      return `${change.difference > 0 ? '+' : ''}${change.difference}`;
    }
    return '-';
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }
}

// CLI execution
if (require.main === module) {
  const detector = new PerformanceRegressionDetector({
    baselineDir: process.argv[2] || './test_reports/performance-baselines',
    currentDir: process.argv[3] || './test_reports/performance',
    outputDir: process.argv[4] || './test_reports/regression-analysis'
  });

  detector.detect()
    .then(results => {
      console.log(`\n= Regression analysis completed with ${results.summary.overallStatus} status`);
      const exitCode = results.summary.severityDistribution.critical > 0 ? 2 :
                       results.summary.severityDistribution.major > 2 ? 1 : 0;
      process.exit(exitCode);
    })
    .catch(error => {
      console.error('Regression detection failed:', error);
      process.exit(1);
    });
}

module.exports = PerformanceRegressionDetector;
