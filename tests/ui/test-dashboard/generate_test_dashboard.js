/**
 * Centralized Test Reporting Dashboard Generator
 * 
 * This script generates a comprehensive test dashboard that aggregates
 * results from all testing categories: unit, integration, performance,
 * accessibility, visual regression, and load testing.
 */

const fs = require('fs');
const path = require('path');

class TestDashboardGenerator {
  constructor() {
    this.reportsDir = path.join(process.cwd(), 'test_reports');
    this.dashboardDir = path.join(this.reportsDir, 'dashboard');
    this.ensureDirectories();
  }

  ensureDirectories() {
    if (!fs.existsSync(this.reportsDir)) {
      fs.mkdirSync(this.reportsDir, { recursive: true });
    }
    if (!fs.existsSync(this.dashboardDir)) {
      fs.mkdirSync(this.dashboardDir, { recursive: true });
    }
  }

  async generateDashboard() {
    console.log('🔍 Collecting test results...');
    
    const testResults = await this.collectTestResults();
    const performanceMetrics = await this.collectPerformanceMetrics();
    const accessibilityResults = await this.collectAccessibilityResults();
    const visualRegressionResults = await this.collectVisualRegressionResults();
    const loadTestResults = await this.collectLoadTestResults();
    const memoryProfileResults = await this.collectMemoryProfileResults();
    
    console.log('📊 Generating dashboard...');
    
    const dashboardData = {
      timestamp: new Date().toISOString(),
      summary: this.generateSummary({
        testResults,
        performanceMetrics,
        accessibilityResults,
        visualRegressionResults,
        loadTestResults,
        memoryProfileResults
      }),
      testResults,
      performanceMetrics,
      accessibilityResults,
      visualRegressionResults,
      loadTestResults,
      memoryProfileResults,
      trends: await this.generateTrends(),
      recommendations: this.generateRecommendations({
        testResults,
        performanceMetrics,
        accessibilityResults,
        visualRegressionResults,
        loadTestResults,
        memoryProfileResults
      })
    };

    await this.generateHTMLDashboard(dashboardData);
    await this.generateJSONReport(dashboardData);
    
    console.log('✅ Test dashboard generated successfully!');
    console.log(`📁 Dashboard location: ${path.join(this.dashboardDir, 'index.html')}`);
    
    return dashboardData;
  }

  async collectTestResults() {
    const testResultsFiles = [
      'playwright-results.json',
      'vitest-results.json',
      'jest-results.json'
    ];

    let aggregateResults = {
      total: 0,
      passed: 0,
      failed: 0,
      skipped: 0,
      duration: 0,
      details: []
    };

    for (const file of testResultsFiles) {
      const filePath = path.join(this.reportsDir, file);
      if (fs.existsSync(filePath)) {
        try {
          const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
          
          // Parse Playwright results format
          if (file.includes('playwright')) {
            const stats = data.stats || {};
            aggregateResults.total += stats.expected || 0;
            aggregateResults.passed += stats.passed || 0;
            aggregateResults.failed += stats.failed || 0;
            aggregateResults.skipped += stats.skipped || 0;
            aggregateResults.duration += data.duration || 0;
          }
          
          // Parse Vitest results format
          if (file.includes('vitest')) {
            aggregateResults.total += data.numTotalTests || 0;
            aggregateResults.passed += data.numPassedTests || 0;
            aggregateResults.failed += data.numFailedTests || 0;
            aggregateResults.skipped += data.numTodoTests || 0;
            aggregateResults.duration += data.testResults?.reduce((sum, result) => sum + (result.perfStats?.runtime || 0), 0) || 0;
          }

          aggregateResults.details.push({
            file,
            framework: file.split('-')[0],
            data: data
          });
        } catch (error) {
          console.warn(`Failed to parse ${file}:`, error.message);
        }
      }
    }

    return aggregateResults;
  }

  async collectPerformanceMetrics() {
    const performanceFiles = [
      'lighthouse-report.json',
      'performance-baselines/current-baseline.json'
    ];

    let performanceData = {
      coreWebVitals: {},
      lighthouse: {},
      performanceBaseline: {},
      regressions: []
    };

    for (const file of performanceFiles) {
      const filePath = path.join(this.reportsDir, file);
      if (fs.existsSync(filePath)) {
        try {
          const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
          
          if (file.includes('lighthouse')) {
            performanceData.lighthouse = {
              performance: data.categories?.performance?.score * 100 || 0,
              accessibility: data.categories?.accessibility?.score * 100 || 0,
              bestPractices: data.categories?.['best-practices']?.score * 100 || 0,
              seo: data.categories?.seo?.score * 100 || 0,
              pwa: data.categories?.pwa?.score * 100 || 0
            };

            // Extract Core Web Vitals
            const audits = data.audits || {};
            performanceData.coreWebVitals = {
              lcp: audits['largest-contentful-paint']?.displayValue || 'N/A',
              fid: audits['first-input-delay']?.displayValue || 'N/A',
              cls: audits['cumulative-layout-shift']?.displayValue || 'N/A',
              fcp: audits['first-contentful-paint']?.displayValue || 'N/A',
              ttfb: audits['server-response-time']?.displayValue || 'N/A'
            };
          }

          if (file.includes('baseline')) {
            performanceData.performanceBaseline = data;
          }
        } catch (error) {
          console.warn(`Failed to parse ${file}:`, error.message);
        }
      }
    }

    return performanceData;
  }

  async collectAccessibilityResults() {
    const a11yFiles = [
      'accessibility-results.json',
      'axe-results.json'
    ];

    let accessibilityData = {
      violations: [],
      passes: [],
      incomplete: [],
      score: 100,
      summary: {
        critical: 0,
        serious: 0,
        moderate: 0,
        minor: 0
      }
    };

    for (const file of a11yFiles) {
      const filePath = path.join(this.reportsDir, file);
      if (fs.existsSync(filePath)) {
        try {
          const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
          
          if (data.violations) {
            accessibilityData.violations.push(...data.violations);
            
            // Calculate severity summary
            data.violations.forEach(violation => {
              const impact = violation.impact || 'minor';
              if (accessibilityData.summary[impact] !== undefined) {
                accessibilityData.summary[impact]++;
              }
            });
          }

          if (data.passes) {
            accessibilityData.passes.push(...data.passes);
          }

          if (data.incomplete) {
            accessibilityData.incomplete.push(...data.incomplete);
          }
        } catch (error) {
          console.warn(`Failed to parse ${file}:`, error.message);
        }
      }
    }

    // Calculate accessibility score
    const totalViolations = Object.values(accessibilityData.summary).reduce((sum, count) => sum + count, 0);
    const totalPasses = accessibilityData.passes.length;
    
    if (totalViolations + totalPasses > 0) {
      accessibilityData.score = Math.round((totalPasses / (totalPasses + totalViolations)) * 100);
    }

    return accessibilityData;
  }

  async collectVisualRegressionResults() {
    const visualFiles = [
      'visual-regression-results.json',
      'percy-results.json'
    ];

    let visualData = {
      totalSnapshots: 0,
      changedSnapshots: 0,
      newSnapshots: 0,
      removedSnapshots: 0,
      approvedSnapshots: 0,
      changes: []
    };

    for (const file of visualFiles) {
      const filePath = path.join(this.reportsDir, file);
      if (fs.existsSync(filePath)) {
        try {
          const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
          
          // Aggregate visual regression data
          visualData.totalSnapshots += data.totalSnapshots || 0;
          visualData.changedSnapshots += data.changedSnapshots || 0;
          visualData.newSnapshots += data.newSnapshots || 0;
          visualData.removedSnapshots += data.removedSnapshots || 0;
          visualData.approvedSnapshots += data.approvedSnapshots || 0;
          
          if (data.changes) {
            visualData.changes.push(...data.changes);
          }
        } catch (error) {
          console.warn(`Failed to parse ${file}:`, error.message);
        }
      }
    }

    return visualData;
  }

  async collectLoadTestResults() {
    const loadTestFiles = [
      'load-test-results.json',
      'k6-results.json',
      'artillery-results.json'
    ];

    let loadData = {
      scenarios: [],
      summary: {
        totalRequests: 0,
        successfulRequests: 0,
        failedRequests: 0,
        averageResponseTime: 0,
        requestsPerSecond: 0,
        errors: []
      }
    };

    for (const file of loadTestFiles) {
      const filePath = path.join(this.reportsDir, file);
      if (fs.existsSync(filePath)) {
        try {
          const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
          
          if (data.scenarios) {
            loadData.scenarios.push(...data.scenarios);
          } else {
            // Single scenario result
            loadData.scenarios.push({
              name: file.replace('.json', ''),
              ...data
            });
          }

          // Aggregate summary data
          loadData.summary.totalRequests += data.totalRequests || 0;
          loadData.summary.successfulRequests += data.successfulRequests || 0;
          loadData.summary.failedRequests += data.failedRequests || 0;
          
          if (data.averageResponseTime) {
            loadData.summary.averageResponseTime = 
              (loadData.summary.averageResponseTime + data.averageResponseTime) / 2;
          }
          
          if (data.requestsPerSecond) {
            loadData.summary.requestsPerSecond += data.requestsPerSecond;
          }
          
          if (data.errors) {
            loadData.summary.errors.push(...data.errors);
          }
        } catch (error) {
          console.warn(`Failed to parse ${file}:`, error.message);
        }
      }
    }

    return loadData;
  }

  async collectMemoryProfileResults() {
    const memoryFiles = fs.readdirSync(this.reportsDir)
      .filter(file => file.includes('memory-report'))
      .map(file => path.join(this.reportsDir, file));

    let memoryData = {
      profiles: [],
      summary: {
        averageMemoryUsage: 0,
        peakMemoryUsage: 0,
        memoryLeaksDetected: 0,
        stableTests: 0,
        totalTests: 0
      }
    };

    for (const filePath of memoryFiles) {
      if (fs.existsSync(filePath)) {
        try {
          const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
          memoryData.profiles.push(data);
          
          if (data.memoryStats) {
            memoryData.summary.averageMemoryUsage += data.memoryStats.avg || 0;
            memoryData.summary.peakMemoryUsage = Math.max(
              memoryData.summary.peakMemoryUsage,
              data.memoryStats.max || 0
            );
          }
          
          if (data.memoryTrend && data.memoryTrend.increasing) {
            memoryData.summary.memoryLeaksDetected++;
          } else {
            memoryData.summary.stableTests++;
          }
          
          memoryData.summary.totalTests++;
        } catch (error) {
          console.warn(`Failed to parse memory profile:`, error.message);
        }
      }
    }

    if (memoryData.summary.totalTests > 0) {
      memoryData.summary.averageMemoryUsage /= memoryData.summary.totalTests;
    }

    return memoryData;
  }

  generateSummary(allResults) {
    const {
      testResults,
      performanceMetrics,
      accessibilityResults,
      loadTestResults,
      memoryProfileResults
    } = allResults;

    const overallHealth = this.calculateOverallHealth(allResults);
    
    return {
      overallHealth,
      testCoverage: {
        passRate: testResults.total > 0 ? (testResults.passed / testResults.total * 100).toFixed(1) : 0,
        totalTests: testResults.total,
        duration: Math.round(testResults.duration / 1000) // Convert to seconds
      },
      performance: {
        lighthouseScore: performanceMetrics.lighthouse.performance || 0,
        coreWebVitals: performanceMetrics.coreWebVitals,
        regressions: performanceMetrics.regressions.length
      },
      accessibility: {
        score: accessibilityResults.score,
        violations: accessibilityResults.violations.length,
        criticalIssues: accessibilityResults.summary.critical
      },
      reliability: {
        loadTestSuccess: loadTestResults.summary.totalRequests > 0 ? 
          (loadTestResults.summary.successfulRequests / loadTestResults.summary.totalRequests * 100).toFixed(1) : 100,
        memoryStability: memoryProfileResults.summary.totalTests > 0 ?
          (memoryProfileResults.summary.stableTests / memoryProfileResults.summary.totalTests * 100).toFixed(1) : 100
      }
    };
  }

  calculateOverallHealth(allResults) {
    const weights = {
      tests: 0.3,
      performance: 0.25,
      accessibility: 0.2,
      reliability: 0.25
    };

    const scores = {
      tests: allResults.testResults.total > 0 ? 
        (allResults.testResults.passed / allResults.testResults.total * 100) : 100,
      performance: allResults.performanceMetrics.lighthouse.performance || 80,
      accessibility: allResults.accessibilityResults.score,
      reliability: allResults.loadTestResults.summary.totalRequests > 0 ?
        (allResults.loadTestResults.summary.successfulRequests / allResults.loadTestResults.summary.totalRequests * 100) : 100
    };

    const weightedScore = Object.keys(weights).reduce((total, key) => {
      return total + (scores[key] * weights[key]);
    }, 0);

    return Math.round(weightedScore);
  }

  async generateTrends() {
    // Look for historical dashboard data to generate trends
    const historicalFiles = fs.readdirSync(this.dashboardDir)
      .filter(file => file.startsWith('dashboard-') && file.endsWith('.json'))
      .sort()
      .slice(-10); // Last 10 reports

    const trends = {
      testPassRate: [],
      performanceScore: [],
      accessibilityScore: [],
      memoryUsage: [],
      loadTestSuccess: []
    };

    for (const file of historicalFiles) {
      try {
        const data = JSON.parse(fs.readFileSync(path.join(this.dashboardDir, file), 'utf8'));
        const date = data.timestamp;
        
        trends.testPassRate.push({
          date,
          value: parseFloat(data.summary?.testCoverage?.passRate || 0)
        });
        
        trends.performanceScore.push({
          date,
          value: data.summary?.performance?.lighthouseScore || 0
        });
        
        trends.accessibilityScore.push({
          date,
          value: data.summary?.accessibility?.score || 0
        });
        
        trends.loadTestSuccess.push({
          date,
          value: parseFloat(data.summary?.reliability?.loadTestSuccess || 100)
        });
      } catch (error) {
        console.warn(`Failed to parse historical data from ${file}`);
      }
    }

    return trends;
  }

  generateRecommendations(allResults) {
    const recommendations = [];

    // Test coverage recommendations
    if (allResults.testResults.passed / allResults.testResults.total < 0.95) {
      recommendations.push({
        category: 'Testing',
        priority: 'High',
        title: 'Improve Test Pass Rate',
        description: `Current pass rate is ${(allResults.testResults.passed / allResults.testResults.total * 100).toFixed(1)}%. Investigate and fix failing tests.`,
        action: 'Review failed test cases and address underlying issues.'
      });
    }

    // Performance recommendations
    if (allResults.performanceMetrics.lighthouse.performance < 80) {
      recommendations.push({
        category: 'Performance',
        priority: 'High',
        title: 'Optimize Page Performance',
        description: `Lighthouse performance score is ${allResults.performanceMetrics.lighthouse.performance}. Consider optimizing resources and loading strategies.`,
        action: 'Implement code splitting, optimize images, and reduce JavaScript bundle size.'
      });
    }

    // Accessibility recommendations
    if (allResults.accessibilityResults.score < 95) {
      recommendations.push({
        category: 'Accessibility',
        priority: allResults.accessibilityResults.summary.critical > 0 ? 'High' : 'Medium',
        title: 'Improve Accessibility Compliance',
        description: `${allResults.accessibilityResults.violations.length} accessibility violations found, including ${allResults.accessibilityResults.summary.critical} critical issues.`,
        action: 'Address accessibility violations starting with critical and serious issues.'
      });
    }

    // Memory recommendations
    if (allResults.memoryProfileResults.summary.memoryLeaksDetected > 0) {
      recommendations.push({
        category: 'Performance',
        priority: 'Medium',
        title: 'Fix Memory Leaks',
        description: `${allResults.memoryProfileResults.summary.memoryLeaksDetected} potential memory leaks detected.`,
        action: 'Review event listener cleanup and object references in affected components.'
      });
    }

    // Load testing recommendations
    const loadSuccessRate = allResults.loadTestResults.summary.totalRequests > 0 ?
      (allResults.loadTestResults.summary.successfulRequests / allResults.loadTestResults.summary.totalRequests) : 1;
    
    if (loadSuccessRate < 0.95) {
      recommendations.push({
        category: 'Reliability',
        priority: 'High',
        title: 'Improve Load Handling',
        description: `Load test success rate is ${(loadSuccessRate * 100).toFixed(1)}%. Application may not handle concurrent users well.`,
        action: 'Optimize server performance and implement proper rate limiting.'
      });
    }

    return recommendations;
  }

  async generateHTMLDashboard(dashboardData) {
    const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pynomaly Test Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem; }
        .health-score { font-size: 3rem; font-weight: bold; text-align: center; margin-bottom: 1rem; }
        .health-status { text-align: center; font-size: 1.2rem; opacity: 0.9; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
        .card { background: white; border-radius: 10px; padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .card h3 { color: #333; margin-bottom: 1rem; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem; }
        .metric { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
        .metric-value { font-weight: bold; color: #667eea; }
        .status-good { color: #10b981; }
        .status-warning { color: #f59e0b; }
        .status-error { color: #ef4444; }
        .recommendations { background: white; border-radius: 10px; padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .recommendation { border-left: 4px solid #667eea; padding: 1rem; margin-bottom: 1rem; background: #f8fafc; }
        .recommendation.high { border-left-color: #ef4444; }
        .recommendation.medium { border-left-color: #f59e0b; }
        .recommendation.low { border-left-color: #10b981; }
        .chart-container { width: 100%; height: 300px; margin: 1rem 0; }
        .footer { text-align: center; margin-top: 2rem; padding: 1rem; color: #666; }
        .timestamp { font-size: 0.9rem; opacity: 0.7; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="health-score ${this.getHealthStatusClass(dashboardData.summary.overallHealth)}">${dashboardData.summary.overallHealth}%</div>
            <div class="health-status">Overall System Health</div>
            <div class="timestamp">Last Updated: ${new Date(dashboardData.timestamp).toLocaleString()}</div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>🧪 Test Coverage</h3>
                <div class="metric">
                    <span>Pass Rate</span>
                    <span class="metric-value ${this.getStatusClass(parseFloat(dashboardData.summary.testCoverage.passRate))}">${dashboardData.summary.testCoverage.passRate}%</span>
                </div>
                <div class="metric">
                    <span>Total Tests</span>
                    <span class="metric-value">${dashboardData.summary.testCoverage.totalTests}</span>
                </div>
                <div class="metric">
                    <span>Duration</span>
                    <span class="metric-value">${dashboardData.summary.testCoverage.duration}s</span>
                </div>
            </div>

            <div class="card">
                <h3>⚡ Performance</h3>
                <div class="metric">
                    <span>Lighthouse Score</span>
                    <span class="metric-value ${this.getStatusClass(dashboardData.summary.performance.lighthouseScore)}">${dashboardData.summary.performance.lighthouseScore}</span>
                </div>
                <div class="metric">
                    <span>LCP</span>
                    <span class="metric-value">${dashboardData.summary.performance.coreWebVitals.lcp || 'N/A'}</span>
                </div>
                <div class="metric">
                    <span>FCP</span>
                    <span class="metric-value">${dashboardData.summary.performance.coreWebVitals.fcp || 'N/A'}</span>
                </div>
                <div class="metric">
                    <span>CLS</span>
                    <span class="metric-value">${dashboardData.summary.performance.coreWebVitals.cls || 'N/A'}</span>
                </div>
            </div>

            <div class="card">
                <h3>♿ Accessibility</h3>
                <div class="metric">
                    <span>A11y Score</span>
                    <span class="metric-value ${this.getStatusClass(dashboardData.summary.accessibility.score)}">${dashboardData.summary.accessibility.score}%</span>
                </div>
                <div class="metric">
                    <span>Violations</span>
                    <span class="metric-value ${dashboardData.summary.accessibility.violations > 0 ? 'status-error' : 'status-good'}">${dashboardData.summary.accessibility.violations}</span>
                </div>
                <div class="metric">
                    <span>Critical Issues</span>
                    <span class="metric-value ${dashboardData.summary.accessibility.criticalIssues > 0 ? 'status-error' : 'status-good'}">${dashboardData.summary.accessibility.criticalIssues}</span>
                </div>
            </div>

            <div class="card">
                <h3>🔄 Reliability</h3>
                <div class="metric">
                    <span>Load Test Success</span>
                    <span class="metric-value ${this.getStatusClass(parseFloat(dashboardData.summary.reliability.loadTestSuccess))}">${dashboardData.summary.reliability.loadTestSuccess}%</span>
                </div>
                <div class="metric">
                    <span>Memory Stability</span>
                    <span class="metric-value ${this.getStatusClass(parseFloat(dashboardData.summary.reliability.memoryStability))}">${dashboardData.summary.reliability.memoryStability}%</span>
                </div>
            </div>
        </div>

        <div class="recommendations">
            <h3>💡 Recommendations</h3>
            ${dashboardData.recommendations.map(rec => `
                <div class="recommendation ${rec.priority.toLowerCase()}">
                    <h4>${rec.title} (${rec.priority} Priority)</h4>
                    <p><strong>Issue:</strong> ${rec.description}</p>
                    <p><strong>Action:</strong> ${rec.action}</p>
                </div>
            `).join('')}
        </div>

        <div class="footer">
            <p>Generated by Pynomaly Test Dashboard at ${new Date().toLocaleString()}</p>
            <p>🤖 Automated Testing Infrastructure</p>
        </div>
    </div>
</body>
</html>`;

    const htmlPath = path.join(this.dashboardDir, 'index.html');
    fs.writeFileSync(htmlPath, html);
  }

  async generateJSONReport(dashboardData) {
    const jsonPath = path.join(this.dashboardDir, `dashboard-${Date.now()}.json`);
    fs.writeFileSync(jsonPath, JSON.stringify(dashboardData, null, 2));
    
    // Also save as latest
    const latestPath = path.join(this.dashboardDir, 'latest.json');
    fs.writeFileSync(latestPath, JSON.stringify(dashboardData, null, 2));
  }

  getHealthStatusClass(score) {
    if (score >= 90) return 'status-good';
    if (score >= 70) return 'status-warning';
    return 'status-error';
  }

  getStatusClass(score) {
    if (score >= 90) return 'status-good';
    if (score >= 70) return 'status-warning';
    return 'status-error';
  }
}

// CLI execution
if (require.main === module) {
  const generator = new TestDashboardGenerator();
  generator.generateDashboard()
    .then(() => {
      console.log('🎉 Dashboard generation completed successfully!');
      process.exit(0);
    })
    .catch(error => {
      console.error('❌ Dashboard generation failed:', error);
      process.exit(1);
    });
}

module.exports = TestDashboardGenerator;