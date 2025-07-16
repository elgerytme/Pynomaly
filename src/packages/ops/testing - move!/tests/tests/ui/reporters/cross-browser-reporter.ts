/**
 * Custom Cross-Browser Test Reporter
 * Provides detailed analysis of test results across different browsers and devices
 */

import { Reporter, TestCase, TestResult, FullResult, Suite } from '@playwright/test/reporter';
import * as fs from 'fs';
import * as path from 'path';

interface BrowserMetrics {
  passed: number;
  failed: number;
  skipped: number;
  flaky: number;
  totalDuration: number;
  avgDuration: number;
  errors: string[];
  warnings: string[];
}

interface TestMetrics {
  name: string;
  browsers: Record<string, {
    status: 'passed' | 'failed' | 'skipped' | 'flaky';
    duration: number;
    error?: string;
    retries: number;
  }>;
  overallStatus: 'passed' | 'failed' | 'mixed';
  compatibility: number; // Percentage of browsers that passed
}

interface PerformanceMetric {
  browser: string;
  test: string;
  loadTime?: number;
  lcp?: number;
  fid?: number;
  cls?: number;
  jsExecutionTime?: number;
}

class CrossBrowserReporter implements Reporter {
  private outputDir: string;
  private browserMetrics: Record<string, BrowserMetrics> = {};
  private testMetrics: TestMetrics[] = [];
  private performanceMetrics: PerformanceMetric[] = [];
  private startTime: number = 0;
  private compatibilityIssues: Array<{
    test: string;
    browsers: string[];
    issue: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
  }> = [];

  constructor() {
    this.outputDir = path.join(process.cwd(), 'test_reports', 'cross-browser');
    this.ensureOutputDir();
  }

  private ensureOutputDir(): void {
    if (!fs.existsSync(this.outputDir)) {
      fs.mkdirSync(this.outputDir, { recursive: true });
    }
  }

  onBegin(config: any, suite: Suite): void {
    this.startTime = Date.now();
    console.log('\n🌐 Starting Cross-Browser Compatibility Testing...');
    console.log(`📊 Testing across ${config.projects?.length || 0} browser configurations`);

    // Initialize browser metrics
    if (config.projects) {
      config.projects.forEach((project: any) => {
        this.browserMetrics[project.name] = {
          passed: 0,
          failed: 0,
          skipped: 0,
          flaky: 0,
          totalDuration: 0,
          avgDuration: 0,
          errors: [],
          warnings: []
        };
      });
    }
  }

  onTestEnd(test: TestCase, result: TestResult): void {
    const projectName = test.parent.project()?.name || 'unknown';
    const testName = test.title;

    // Update browser metrics
    if (this.browserMetrics[projectName]) {
      const metrics = this.browserMetrics[projectName];
      metrics.totalDuration += result.duration;

      switch (result.status) {
        case 'passed':
          metrics.passed++;
          break;
        case 'failed':
          metrics.failed++;
          if (result.error) {
            metrics.errors.push(`${testName}: ${result.error.message}`);
          }
          break;
        case 'skipped':
          metrics.skipped++;
          break;
        case 'timedOut':
          metrics.failed++;
          metrics.errors.push(`${testName}: Test timed out`);
          break;
      }

      if (result.retry > 0) {
        metrics.flaky++;
      }
    }

    // Track test across browsers
    let existingTest = this.testMetrics.find(t => t.name === testName);
    if (!existingTest) {
      existingTest = {
        name: testName,
        browsers: {},
        overallStatus: 'passed',
        compatibility: 0
      };
      this.testMetrics.push(existingTest);
    }

    existingTest.browsers[projectName] = {
      status: result.status === 'timedOut' ? 'failed' : result.status,
      duration: result.duration,
      error: result.error?.message,
      retries: result.retry
    };

    // Extract performance metrics if available
    if (result.attachments) {
      result.attachments.forEach(attachment => {
        if (attachment.name.includes('performance') && attachment.body) {
          try {
            const perfData = JSON.parse(attachment.body.toString());
            this.performanceMetrics.push({
              browser: projectName,
              test: testName,
              ...perfData
            });
          } catch (e) {
            // Ignore parsing errors
          }
        }
      });
    }
  }

  onEnd(result: FullResult): void {
    this.calculateCompatibilityScores();
    this.identifyCompatibilityIssues();
    this.generateReports();
    this.printSummary();
  }

  private calculateCompatibilityScores(): void {
    this.testMetrics.forEach(test => {
      const browsers = Object.keys(test.browsers);
      const passedBrowsers = browsers.filter(browser =>
        test.browsers[browser].status === 'passed'
      ).length;

      test.compatibility = browsers.length > 0 ? (passedBrowsers / browsers.length) * 100 : 0;

      // Determine overall status
      const statuses = browsers.map(browser => test.browsers[browser].status);
      if (statuses.every(status => status === 'passed')) {
        test.overallStatus = 'passed';
      } else if (statuses.every(status => status === 'failed' || status === 'skipped')) {
        test.overallStatus = 'failed';
      } else {
        test.overallStatus = 'mixed';
      }
    });

    // Calculate average durations
    Object.keys(this.browserMetrics).forEach(browser => {
      const metrics = this.browserMetrics[browser];
      const totalTests = metrics.passed + metrics.failed + metrics.skipped;
      metrics.avgDuration = totalTests > 0 ? metrics.totalDuration / totalTests : 0;
    });
  }

  private identifyCompatibilityIssues(): void {
    this.testMetrics.forEach(test => {
      const failedBrowsers = Object.keys(test.browsers).filter(browser =>
        test.browsers[browser].status === 'failed'
      );

      if (failedBrowsers.length > 0 && failedBrowsers.length < Object.keys(test.browsers).length) {
        // Determine severity based on which browsers failed
        let severity: 'low' | 'medium' | 'high' | 'critical' = 'low';

        if (failedBrowsers.includes('Desktop Safari') || failedBrowsers.includes('Mobile Safari')) {
          severity = 'high'; // Safari issues are often tricky
        }
        if (failedBrowsers.length > 1) {
          severity = 'medium';
        }
        if (failedBrowsers.length >= Object.keys(test.browsers).length / 2) {
          severity = 'critical';
        }

        // Get common error message
        const errors = failedBrowsers.map(browser => test.browsers[browser].error).filter(Boolean);
        const commonIssue = this.findCommonIssue(errors);

        this.compatibilityIssues.push({
          test: test.name,
          browsers: failedBrowsers,
          issue: commonIssue || 'Browser-specific failure',
          severity
        });
      }
    });
  }

  private findCommonIssue(errors: (string | undefined)[]): string {
    if (errors.length === 0) return 'Unknown error';

    // Common error patterns
    const patterns = [
      { pattern: /timeout/i, description: 'Timeout issues' },
      { pattern: /element.*not.*found/i, description: 'Element not found' },
      { pattern: /network/i, description: 'Network connectivity issues' },
      { pattern: /permission/i, description: 'Permission denied' },
      { pattern: /security/i, description: 'Security policy restrictions' },
      { pattern: /css.*selector/i, description: 'CSS selector issues' },
      { pattern: /javascript.*error/i, description: 'JavaScript execution errors' }
    ];

    for (const pattern of patterns) {
      if (errors.some(error => error && pattern.pattern.test(error))) {
        return pattern.description;
      }
    }

    return errors[0] || 'Unknown error';
  }

  private generateReports(): void {
    this.generateHTMLReport();
    this.generateJSONReport();
    this.generateCSVReport();
    this.generateCompatibilityMatrix();
  }

  private generateHTMLReport(): void {
    const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cross-Browser Compatibility Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #0ea5e9; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .summary-stats { display: flex; justify-content: space-around; background: #f0f9ff; padding: 20px; border-radius: 6px; margin: 20px 0; }
        .summary-stat { text-align: center; }
        .summary-number { font-size: 2em; font-weight: bold; color: #0ea5e9; }
        .summary-label { color: #6b7280; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌐 Cross-Browser Compatibility Report</h1>
        <p>Generated on ${new Date().toLocaleString()}</p>

        <div class="summary-stats">
            <div class="summary-stat">
                <div class="summary-number">${Object.keys(this.browserMetrics).length}</div>
                <div class="summary-label">Browsers Tested</div>
            </div>
            <div class="summary-stat">
                <div class="summary-number">${this.testMetrics.length}</div>
                <div class="summary-label">Tests Executed</div>
            </div>
            <div class="summary-stat">
                <div class="summary-number">${this.compatibilityIssues.length}</div>
                <div class="summary-label">Compatibility Issues</div>
            </div>
            <div class="summary-stat">
                <div class="summary-number">${Math.round(this.calculateOverallCompatibility())}%</div>
                <div class="summary-label">Overall Compatibility</div>
            </div>
        </div>

        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0; color: #6b7280; text-align: center;">
            <p>Report generated by Pynomaly Cross-Browser Testing Suite</p>
        </footer>
    </div>
</body>
</html>`;

    fs.writeFileSync(path.join(this.outputDir, 'cross-browser-report.html'), html);
  }

  private generateJSONReport(): void {
    const report = {
      metadata: {
        generatedAt: new Date().toISOString(),
        totalDuration: Date.now() - this.startTime,
        browsersTestedCount: Object.keys(this.browserMetrics).length,
        testsExecutedCount: this.testMetrics.length,
        compatibilityIssuesCount: this.compatibilityIssues.length,
        overallCompatibility: this.calculateOverallCompatibility()
      },
      browserMetrics: this.browserMetrics,
      testMetrics: this.testMetrics,
      compatibilityIssues: this.compatibilityIssues,
      performanceMetrics: this.performanceMetrics
    };

    fs.writeFileSync(
      path.join(this.outputDir, 'cross-browser-report.json'),
      JSON.stringify(report, null, 2)
    );
  }

  private generateCSVReport(): void {
    const csvData = [
      ['Test Name', 'Browser', 'Status', 'Duration (ms)', 'Retries', 'Error'],
      ...this.testMetrics.flatMap(test =>
        Object.entries(test.browsers).map(([browser, result]) => [
          test.name,
          browser,
          result.status,
          result.duration.toString(),
          result.retries.toString(),
          result.error || ''
        ])
      )
    ];

    const csvContent = csvData.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
    fs.writeFileSync(path.join(this.outputDir, 'cross-browser-results.csv'), csvContent);
  }

  private generateCompatibilityMatrix(): void {
    const browsers = Object.keys(this.browserMetrics);
    const tests = this.testMetrics.map(t => t.name);

    const matrix = tests.map(test => {
      const testMetric = this.testMetrics.find(t => t.name === test);
      return browsers.map(browser => {
        const result = testMetric?.browsers[browser];
        return result ? result.status : 'not-run';
      });
    });

    const matrixData = {
      browsers,
      tests,
      matrix,
      compatibility: this.testMetrics.map(t => t.compatibility)
    };

    fs.writeFileSync(
      path.join(this.outputDir, 'compatibility-matrix.json'),
      JSON.stringify(matrixData, null, 2)
    );
  }

  private calculateOverallCompatibility(): number {
    if (this.testMetrics.length === 0) return 0;

    const totalCompatibility = this.testMetrics.reduce((sum, test) => sum + test.compatibility, 0);
    return totalCompatibility / this.testMetrics.length;
  }

  private printSummary(): void {
    const totalDuration = Date.now() - this.startTime;
    const overallCompatibility = this.calculateOverallCompatibility();

    console.log('\n' + '='.repeat(60));
    console.log('🌐 CROSS-BROWSER COMPATIBILITY TEST SUMMARY');
    console.log('='.repeat(60));

    console.log(`⏱️  Total Duration: ${Math.round(totalDuration / 1000)}s`);
    console.log(`🌏 Browsers Tested: ${Object.keys(this.browserMetrics).length}`);
    console.log(`🧪 Tests Executed: ${this.testMetrics.length}`);
    console.log(`📊 Overall Compatibility: ${Math.round(overallCompatibility)}%`);

    if (this.compatibilityIssues.length > 0) {
      console.log(`🚨 Compatibility Issues: ${this.compatibilityIssues.length}`);

      const criticalIssues = this.compatibilityIssues.filter(i => i.severity === 'critical').length;
      const highIssues = this.compatibilityIssues.filter(i => i.severity === 'high').length;

      if (criticalIssues > 0) {
        console.log(`   🔴 Critical: ${criticalIssues}`);
      }
      if (highIssues > 0) {
        console.log(`   🟠 High: ${highIssues}`);
      }
    } else {
      console.log('✅ No compatibility issues detected!');
    }

    console.log('\n📁 Reports generated:');
    console.log(`   📊 HTML: ${path.join(this.outputDir, 'cross-browser-report.html')}`);
    console.log(`   📋 JSON: ${path.join(this.outputDir, 'cross-browser-report.json')}`);
    console.log(`   📈 CSV: ${path.join(this.outputDir, 'cross-browser-results.csv')}`);
    console.log(`   🎯 Matrix: ${path.join(this.outputDir, 'compatibility-matrix.json')}`);

    console.log('='.repeat(60) + '\n');
  }
}

export default CrossBrowserReporter;
