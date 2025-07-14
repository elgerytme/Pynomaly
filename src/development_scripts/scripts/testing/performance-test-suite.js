#!/usr/bin/env node
/**
 * Comprehensive Performance Test Suite for Pynomaly Web UI
 * Tests Core Web Vitals, bundle sizes, and optimization opportunities
 */

const lighthouse = require('lighthouse');
const chromeLauncher = require('chrome-launcher');
const fs = require('fs').promises;
const path = require('path');

class PerformanceTestSuite {
  constructor(options = {}) {
    this.baseUrl = options.baseUrl || 'http://localhost:8000';
    this.outputDir = options.outputDir || './test_reports/performance';
    this.thresholds = {
      performance: 85,
      accessibility: 95,
      bestPractices: 90,
      seo: 85,
      pwa: 80,
      // Core Web Vitals
      lcp: 2500,     // Largest Contentful Paint (ms)
      fid: 100,      // First Input Delay (ms)
      cls: 0.1,      // Cumulative Layout Shift
      fcp: 2000,     // First Contentful Paint (ms)
      si: 3000,      // Speed Index (ms)
      tbt: 300       // Total Blocking Time (ms)
    };

    this.testPages = [
      { name: 'homepage', url: '/', critical: true },
      { name: 'dashboard', url: '/dashboard', critical: true },
      { name: 'datasets', url: '/datasets', critical: false },
      { name: 'detectors', url: '/detectors', critical: false },
      { name: 'results', url: '/results', critical: false },
      { name: 'monitor', url: '/monitor', critical: false }
    ];

    this.results = {
      timestamp: new Date().toISOString(),
      testPages: [],
      summary: {},
      recommendations: [],
      bundleAnalysis: {}
    };
  }

  async run() {
    console.log('🚀 Starting Comprehensive Performance Test Suite...\n');

    try {
      // Ensure output directory exists
      await fs.mkdir(this.outputDir, { recursive: true });

      // Run Lighthouse tests for each page
      await this.runLighthouseTests();

      // Analyze bundle sizes
      await this.analyzeBundleSizes();

      // Test Core Web Vitals in real browsers
      await this.testCoreWebVitals();

      // Generate performance budget analysis
      await this.analyzePerformanceBudget();

      // Generate comprehensive report
      await this.generateReports();

      console.log('\n✅ Performance test suite completed successfully!');
      console.log(`📊 Reports saved to: ${this.outputDir}`);

      return this.results;

    } catch (error) {
      console.error('❌ Performance test suite failed:', error);
      throw error;
    }
  }

  async runLighthouseTests() {
    console.log('🔍 Running Lighthouse performance audits...');

    for (const page of this.testPages) {
      console.log(`  Testing ${page.name}...`);

      const chrome = await chromeLauncher.launch({
        chromeFlags: ['--headless', '--no-sandbox', '--disable-dev-shm-usage']
      });

      try {
        const options = {
          logLevel: 'info',
          output: 'json',
          onlyCategories: ['performance', 'accessibility', 'best-practices', 'seo', 'pwa'],
          port: chrome.port,
          settings: {
            emulatedFormFactor: 'desktop',
            throttling: {
              rttMs: 40,
              throughputKbps: 10240,
              cpuSlowdownMultiplier: 1
            }
          }
        };

        const runnerResult = await lighthouse(`${this.baseUrl}${page.url}`, options);
        const lhr = runnerResult.lhr;

        // Extract key metrics
        const pageResult = {
          name: page.name,
          url: page.url,
          critical: page.critical,
          scores: {
            performance: Math.round(lhr.categories.performance.score * 100),
            accessibility: Math.round(lhr.categories.accessibility.score * 100),
            bestPractices: Math.round(lhr.categories['best-practices'].score * 100),
            seo: Math.round(lhr.categories.seo.score * 100),
            pwa: Math.round(lhr.categories.pwa.score * 100)
          },
          metrics: {
            lcp: lhr.audits['largest-contentful-paint'].numericValue,
            fid: lhr.audits['max-potential-fid']?.numericValue || 0,
            cls: lhr.audits['cumulative-layout-shift'].numericValue,
            fcp: lhr.audits['first-contentful-paint'].numericValue,
            si: lhr.audits['speed-index'].numericValue,
            tbt: lhr.audits['total-blocking-time'].numericValue,
            tti: lhr.audits['interactive'].numericValue
          },
          opportunities: this.extractOpportunities(lhr),
          diagnostics: this.extractDiagnostics(lhr),
          passed: this.checkThresholds(lhr, page.critical)
        };

        this.results.testPages.push(pageResult);

        // Save detailed Lighthouse report
        const reportPath = path.join(this.outputDir, `lighthouse-${page.name}.json`);
        await fs.writeFile(reportPath, JSON.stringify(lhr, null, 2));

        console.log(`    Performance: ${pageResult.scores.performance}%`);
        console.log(`    LCP: ${Math.round(pageResult.metrics.lcp)}ms`);
        console.log(`    CLS: ${pageResult.metrics.cls.toFixed(3)}`);

      } finally {
        await chrome.kill();
      }
    }
  }

  async analyzeBundleSizes() {
    console.log('\n📦 Analyzing bundle sizes...');

    try {
      // Check if built files exist
      const staticDir = './src/pynomaly/presentation/web/static';
      const jsDir = path.join(staticDir, 'js/dist');
      const cssDir = path.join(staticDir, 'css');

      const bundleAnalysis = {
        javascript: await this.analyzeBundleDirectory(jsDir, '.js'),
        css: await this.analyzeBundleDirectory(cssDir, '.css'),
        images: await this.analyzeBundleDirectory(path.join(staticDir, 'images'), ['.png', '.jpg', '.svg']),
        fonts: await this.analyzeBundleDirectory(path.join(staticDir, 'fonts'), ['.woff', '.woff2'])
      };

      // Calculate total bundle size
      const totalSize = Object.values(bundleAnalysis).reduce((total, category) => {
        return total + category.totalSize;
      }, 0);

      bundleAnalysis.totalSize = totalSize;
      bundleAnalysis.recommendations = this.generateBundleRecommendations(bundleAnalysis);

      this.results.bundleAnalysis = bundleAnalysis;

      console.log(`  Total bundle size: ${this.formatBytes(totalSize)}`);
      console.log(`  JavaScript: ${this.formatBytes(bundleAnalysis.javascript.totalSize)}`);
      console.log(`  CSS: ${this.formatBytes(bundleAnalysis.css.totalSize)}`);

    } catch (error) {
      console.warn(`  Warning: Bundle analysis failed - ${error.message}`);
      this.results.bundleAnalysis = { error: error.message };
    }
  }

  async analyzeBundleDirectory(dirPath, extensions) {
    const analysis = {
      files: [],
      totalSize: 0,
      largestFile: null,
      fileCount: 0
    };

    try {
      const files = await fs.readdir(dirPath);
      const targetExtensions = Array.isArray(extensions) ? extensions : [extensions];

      for (const file of files) {
        const fileExt = path.extname(file).toLowerCase();
        if (targetExtensions.includes(fileExt)) {
          const filePath = path.join(dirPath, file);
          const stats = await fs.stat(filePath);

          const fileInfo = {
            name: file,
            size: stats.size,
            extension: fileExt
          };

          analysis.files.push(fileInfo);
          analysis.totalSize += stats.size;
          analysis.fileCount++;

          if (!analysis.largestFile || stats.size > analysis.largestFile.size) {
            analysis.largestFile = fileInfo;
          }
        }
      }
    } catch (error) {
      // Directory doesn't exist or is empty
      console.warn(`  Directory not found or empty: ${dirPath}`);
    }

    return analysis;
  }

  async testCoreWebVitals() {
    console.log('\n📊 Testing Core Web Vitals with real user metrics...');

    // This would integrate with tools like web-vitals library
    // For now, we'll use the Lighthouse data we already collected
    const coreWebVitals = {
      lcp: { values: [], threshold: this.thresholds.lcp },
      fid: { values: [], threshold: this.thresholds.fid },
      cls: { values: [], threshold: this.thresholds.cls }
    };

    // Aggregate metrics from all critical pages
    const criticalPages = this.results.testPages.filter(page => page.critical);

    criticalPages.forEach(page => {
      coreWebVitals.lcp.values.push(page.metrics.lcp);
      coreWebVitals.fid.values.push(page.metrics.fid);
      coreWebVitals.cls.values.push(page.metrics.cls);
    });

    // Calculate averages and pass rates
    Object.keys(coreWebVitals).forEach(metric => {
      const values = coreWebVitals[metric].values;
      const threshold = coreWebVitals[metric].threshold;

      coreWebVitals[metric].average = values.reduce((sum, val) => sum + val, 0) / values.length;
      coreWebVitals[metric].worst = Math.max(...values);
      coreWebVitals[metric].best = Math.min(...values);
      coreWebVitals[metric].passRate = (values.filter(val => val <= threshold).length / values.length) * 100;
      coreWebVitals[metric].passed = coreWebVitals[metric].passRate >= 75; // 75% of pages should pass
    });

    this.results.coreWebVitals = coreWebVitals;

    console.log(`  LCP: ${Math.round(coreWebVitals.lcp.average)}ms (${coreWebVitals.lcp.passRate.toFixed(1)}% pass rate)`);
    console.log(`  FID: ${Math.round(coreWebVitals.fid.average)}ms (${coreWebVitals.fid.passRate.toFixed(1)}% pass rate)`);
    console.log(`  CLS: ${coreWebVitals.cls.average.toFixed(3)} (${coreWebVitals.cls.passRate.toFixed(1)}% pass rate)`);
  }

  async analyzePerformanceBudget() {
    console.log('\n💰 Analyzing performance budget...');

    const budget = {
      javascript: { budget: 200 * 1024, actual: this.results.bundleAnalysis.javascript?.totalSize || 0 }, // 200KB
      css: { budget: 50 * 1024, actual: this.results.bundleAnalysis.css?.totalSize || 0 }, // 50KB
      images: { budget: 500 * 1024, actual: this.results.bundleAnalysis.images?.totalSize || 0 }, // 500KB
      fonts: { budget: 100 * 1024, actual: this.results.bundleAnalysis.fonts?.totalSize || 0 }, // 100KB
      totalBundle: { budget: 1000 * 1024, actual: this.results.bundleAnalysis.totalSize || 0 } // 1MB
    };

    const budgetAnalysis = {};

    Object.keys(budget).forEach(category => {
      const item = budget[category];
      const utilizationPercent = (item.actual / item.budget) * 100;
      const overBudget = item.actual > item.budget;

      budgetAnalysis[category] = {
        budget: item.budget,
        actual: item.actual,
        utilizationPercent: utilizationPercent,
        overBudget: overBudget,
        difference: item.actual - item.budget,
        status: overBudget ? 'exceeded' : utilizationPercent > 80 ? 'warning' : 'good'
      };

      const status = overBudget ? '❌' : utilizationPercent > 80 ? '⚠️' : '✅';
      console.log(`  ${status} ${category}: ${this.formatBytes(item.actual)} / ${this.formatBytes(item.budget)} (${utilizationPercent.toFixed(1)}%)`);
    });

    this.results.performanceBudget = budgetAnalysis;
  }

  async generateReports() {
    console.log('\n📄 Generating comprehensive reports...');

    // Generate summary
    this.generateSummary();

    // Generate recommendations
    this.generateRecommendations();

    // Save main results file
    const resultsPath = path.join(this.outputDir, 'performance-test-results.json');
    await fs.writeFile(resultsPath, JSON.stringify(this.results, null, 2));

    // Generate HTML report
    await this.generateHTMLReport();

    // Generate CSV summary for tracking
    await this.generateCSVSummary();

    console.log('  ✅ JSON report saved');
    console.log('  ✅ HTML report saved');
    console.log('  ✅ CSV summary saved');
  }

  generateSummary() {
    const pages = this.results.testPages;
    const criticalPages = pages.filter(p => p.critical);

    const summary = {
      totalPages: pages.length,
      criticalPages: criticalPages.length,
      overallHealth: 'good', // Will be determined below
      averageScores: {},
      coreWebVitalsStatus: {},
      budgetCompliance: true
    };

    // Calculate average scores
    ['performance', 'accessibility', 'bestPractices', 'seo', 'pwa'].forEach(category => {
      const scores = pages.map(page => page.scores[category]);
      summary.averageScores[category] = Math.round(scores.reduce((sum, score) => sum + score, 0) / scores.length);
    });

    // Core Web Vitals status
    if (this.results.coreWebVitals) {
      summary.coreWebVitalsStatus = {
        lcp: this.results.coreWebVitals.lcp.passed,
        fid: this.results.coreWebVitals.fid.passed,
        cls: this.results.coreWebVitals.cls.passed,
        allPassed: this.results.coreWebVitals.lcp.passed &&
                   this.results.coreWebVitals.fid.passed &&
                   this.results.coreWebVitals.cls.passed
      };
    }

    // Budget compliance
    if (this.results.performanceBudget) {
      summary.budgetCompliance = !Object.values(this.results.performanceBudget).some(item => item.overBudget);
    }

    // Determine overall health
    const criticalIssues = [
      summary.averageScores.performance < this.thresholds.performance,
      summary.averageScores.accessibility < this.thresholds.accessibility,
      !summary.coreWebVitalsStatus.allPassed,
      !summary.budgetCompliance
    ].filter(Boolean).length;

    if (criticalIssues === 0) {
      summary.overallHealth = 'excellent';
    } else if (criticalIssues <= 1) {
      summary.overallHealth = 'good';
    } else if (criticalIssues <= 2) {
      summary.overallHealth = 'warning';
    } else {
      summary.overallHealth = 'critical';
    }

    this.results.summary = summary;
  }

  generateRecommendations() {
    const recommendations = [];

    // Performance recommendations
    this.results.testPages.forEach(page => {
      if (page.scores.performance < this.thresholds.performance) {
        page.opportunities.forEach(opp => {
          if (opp.savings > 1000) { // Only include opportunities with >1s savings
            recommendations.push({
              type: 'performance',
              priority: 'high',
              page: page.name,
              issue: opp.title,
              description: opp.description,
              savings: opp.savings,
              category: 'optimization'
            });
          }
        });
      }
    });

    // Bundle size recommendations
    if (this.results.bundleAnalysis.recommendations) {
      recommendations.push(...this.results.bundleAnalysis.recommendations);
    }

    // Core Web Vitals recommendations
    if (this.results.coreWebVitals) {
      if (!this.results.coreWebVitals.lcp.passed) {
        recommendations.push({
          type: 'core-web-vitals',
          priority: 'high',
          issue: 'Largest Contentful Paint optimization needed',
          description: 'LCP is above the recommended threshold. Consider optimizing images, reducing server response times, and minimizing render-blocking resources.',
          category: 'user-experience'
        });
      }

      if (!this.results.coreWebVitals.cls.passed) {
        recommendations.push({
          type: 'core-web-vitals',
          priority: 'high',
          issue: 'Cumulative Layout Shift optimization needed',
          description: 'CLS is above the recommended threshold. Ensure images and ads have explicit dimensions and avoid inserting content above existing content.',
          category: 'user-experience'
        });
      }
    }

    // Sort by priority
    recommendations.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });

    this.results.recommendations = recommendations;
  }

  async generateHTMLReport() {
    const htmlReport = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pynomaly Performance Test Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 20px; background: #f8fafc; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }
        .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .metric.good { color: #10b981; }
        .metric.warning { color: #f59e0b; }
        .metric.critical { color: #ef4444; }
        .table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .table th, .table td { padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }
        .table th { background: #f9fafb; font-weight: 600; }
        .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.875em; font-weight: 500; }
        .badge.good { background: #d1fae5; color: #065f46; }
        .badge.warning { background: #fef3c7; color: #92400e; }
        .badge.critical { background: #fee2e2; color: #991b1b; }
        .recommendations { margin-top: 40px; }
        .recommendation { background: white; border-radius: 8px; padding: 16px; margin-bottom: 16px; border-left: 4px solid #3b82f6; }
        .recommendation.high { border-left-color: #ef4444; }
        .recommendation.medium { border-left-color: #f59e0b; }
        .recommendation.low { border-left-color: #10b981; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Pynomaly Performance Test Report</h1>
            <p>Generated on ${new Date().toLocaleString()}</p>
        </div>

        <div class="summary">
            <div class="card">
                <h3>Overall Health</h3>
                <div class="metric ${this.results.summary.overallHealth}">${this.results.summary.overallHealth.toUpperCase()}</div>
            </div>
            <div class="card">
                <h3>Performance Score</h3>
                <div class="metric ${this.getScoreClass(this.results.summary.averageScores.performance)}">${this.results.summary.averageScores.performance}%</div>
            </div>
            <div class="card">
                <h3>Accessibility Score</h3>
                <div class="metric ${this.getScoreClass(this.results.summary.averageScores.accessibility)}">${this.results.summary.averageScores.accessibility}%</div>
            </div>
            <div class="card">
                <h3>Core Web Vitals</h3>
                <div class="metric ${this.results.summary.coreWebVitalsStatus?.allPassed ? 'good' : 'critical'}">
                    ${this.results.summary.coreWebVitalsStatus?.allPassed ? 'PASS' : 'FAIL'}
                </div>
            </div>
        </div>

        ${this.generatePageResultsHTML()}
        ${this.generateRecommendationsHTML()}
    </div>
</body>
</html>`;

    const htmlPath = path.join(this.outputDir, 'performance-report.html');
    await fs.writeFile(htmlPath, htmlReport);
  }

  generatePageResultsHTML() {
    return `
        <div class="card">
            <h2>Page Performance Results</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Page</th>
                        <th>Performance</th>
                        <th>LCP</th>
                        <th>CLS</th>
                        <th>Overall</th>
                    </tr>
                </thead>
                <tbody>
                    ${this.results.testPages.map(page => `
                        <tr>
                            <td><strong>${page.name}</strong></td>
                            <td><span class="badge ${this.getScoreClass(page.scores.performance)}">${page.scores.performance}%</span></td>
                            <td>${Math.round(page.metrics.lcp)}ms</td>
                            <td>${page.metrics.cls.toFixed(3)}</td>
                            <td><span class="badge ${page.passed ? 'good' : 'critical'}">${page.passed ? 'PASS' : 'FAIL'}</span></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
  }

  generateRecommendationsHTML() {
    if (this.results.recommendations.length === 0) {
      return '<div class="card"><h2>Recommendations</h2><p>No critical issues found. Great job!</p></div>';
    }

    return `
        <div class="recommendations">
            <h2>Recommendations</h2>
            ${this.results.recommendations.map(rec => `
                <div class="recommendation ${rec.priority}">
                    <h3>${rec.issue}</h3>
                    <p>${rec.description}</p>
                    ${rec.savings ? `<p><strong>Potential savings:</strong> ${rec.savings}ms</p>` : ''}
                    <span class="badge ${rec.priority}">${rec.priority.toUpperCase()} PRIORITY</span>
                </div>
            `).join('')}
        </div>
    `;
  }

  async generateCSVSummary() {
    const csv = [
      'Page,Performance Score,Accessibility Score,LCP (ms),CLS,Passed',
      ...this.results.testPages.map(page =>
        `${page.name},${page.scores.performance},${page.scores.accessibility},${Math.round(page.metrics.lcp)},${page.metrics.cls.toFixed(3)},${page.passed}`
      )
    ].join('\n');

    const csvPath = path.join(this.outputDir, 'performance-summary.csv');
    await fs.writeFile(csvPath, csv);
  }

  // Helper methods
  extractOpportunities(lhr) {
    const opportunities = [];
    Object.values(lhr.audits).forEach(audit => {
      if (audit.details && audit.details.type === 'opportunity' && audit.numericValue > 0) {
        opportunities.push({
          title: audit.title,
          description: audit.description,
          savings: audit.numericValue,
          displayValue: audit.displayValue
        });
      }
    });
    return opportunities.sort((a, b) => b.savings - a.savings);
  }

  extractDiagnostics(lhr) {
    const diagnostics = [];
    Object.values(lhr.audits).forEach(audit => {
      if (audit.scoreDisplayMode === 'informative' && audit.score !== null && audit.score < 1) {
        diagnostics.push({
          title: audit.title,
          description: audit.description,
          displayValue: audit.displayValue
        });
      }
    });
    return diagnostics;
  }

  checkThresholds(lhr, isCritical) {
    const scores = {
      performance: Math.round(lhr.categories.performance.score * 100),
      accessibility: Math.round(lhr.categories.accessibility.score * 100)
    };

    const metrics = {
      lcp: lhr.audits['largest-contentful-paint'].numericValue,
      cls: lhr.audits['cumulative-layout-shift'].numericValue
    };

    if (isCritical) {
      return scores.performance >= this.thresholds.performance &&
             scores.accessibility >= this.thresholds.accessibility &&
             metrics.lcp <= this.thresholds.lcp &&
             metrics.cls <= this.thresholds.cls;
    } else {
      return scores.performance >= (this.thresholds.performance - 10) &&
             scores.accessibility >= this.thresholds.accessibility;
    }
  }

  generateBundleRecommendations(bundleAnalysis) {
    const recommendations = [];

    if (bundleAnalysis.javascript.totalSize > 200 * 1024) {
      recommendations.push({
        type: 'bundle-size',
        priority: 'medium',
        issue: 'JavaScript bundle size optimization',
        description: `JavaScript bundle is ${this.formatBytes(bundleAnalysis.javascript.totalSize)}. Consider code splitting, tree shaking, and removing unused dependencies.`,
        category: 'optimization'
      });
    }

    if (bundleAnalysis.css.totalSize > 50 * 1024) {
      recommendations.push({
        type: 'bundle-size',
        priority: 'low',
        issue: 'CSS bundle size optimization',
        description: `CSS bundle is ${this.formatBytes(bundleAnalysis.css.totalSize)}. Consider removing unused CSS and using critical CSS techniques.`,
        category: 'optimization'
      });
    }

    return recommendations;
  }

  getScoreClass(score) {
    if (score >= 90) return 'good';
    if (score >= 70) return 'warning';
    return 'critical';
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
  const suite = new PerformanceTestSuite({
    baseUrl: process.argv[2] || 'http://localhost:8000',
    outputDir: process.argv[3] || './test_reports/performance'
  });

  suite.run()
    .then(results => {
      console.log(`\n📊 Performance test completed with ${results.summary.overallHealth} health status`);
      process.exit(results.summary.overallHealth === 'critical' ? 1 : 0);
    })
    .catch(error => {
      console.error('Performance test failed:', error);
      process.exit(1);
    });
}

module.exports = PerformanceTestSuite;
