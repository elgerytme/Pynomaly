#!/usr/bin/env node
/**
 * Bundle Analyzer for Pynomaly Web UI
 * Analyzes JavaScript and CSS bundles for optimization opportunities
 */

const fs = require('fs').promises;
const path = require('path');
const { execSync } = require('child_process');

class BundleAnalyzer {
  constructor(options = {}) {
    this.staticDir = options.staticDir || './src/pynomaly/presentation/web/static';
    this.outputDir = options.outputDir || './test_reports/bundle-analysis';
    this.budgets = {
      javascript: 200 * 1024,  // 200KB
      css: 50 * 1024,          // 50KB
      images: 500 * 1024,      // 500KB
      fonts: 100 * 1024,       // 100KB
      total: 1000 * 1024       // 1MB
    };

    this.analysis = {
      timestamp: new Date().toISOString(),
      bundles: {},
      dependencies: {},
      recommendations: [],
      budgetStatus: {},
      summary: {}
    };
  }

  async analyze() {
    console.log('🔍 Starting bundle analysis...');

    try {
      await fs.mkdir(this.outputDir, { recursive: true });

      // Analyze built bundles
      await this.analyzeBundles();

      // Analyze dependencies
      await this.analyzeDependencies();

      // Check performance budgets
      this.checkBudgets();

      // Generate recommendations
      this.generateRecommendations();

      // Create summary
      this.createSummary();

      // Save analysis results
      await this.saveResults();

      console.log(`✅ Bundle analysis completed. Reports saved to: ${this.outputDir}`);
      return this.analysis;

    } catch (error) {
      console.error('❌ Bundle analysis failed:', error);
      throw error;
    }
  }

  async analyzeBundles() {
    console.log('📦 Analyzing bundle files...');

    const bundleTypes = [
      { type: 'javascript', dir: 'js/dist', extensions: ['.js'] },
      { type: 'css', dir: 'css', extensions: ['.css'] },
      { type: 'images', dir: 'images', extensions: ['.png', '.jpg', '.jpeg', '.svg', '.webp'] },
      { type: 'fonts', dir: 'fonts', extensions: ['.woff', '.woff2', '.ttf'] }
    ];

    for (const bundleType of bundleTypes) {
      const bundleDir = path.join(this.staticDir, bundleType.dir);

      try {
        const analysis = await this.analyzeBundleDirectory(bundleDir, bundleType.extensions);
        this.analysis.bundles[bundleType.type] = {
          ...analysis,
          budget: this.budgets[bundleType.type],
          budgetUtilization: (analysis.totalSize / this.budgets[bundleType.type]) * 100
        };

        console.log(`  ${bundleType.type}: ${this.formatBytes(analysis.totalSize)} (${analysis.fileCount} files)`);

      } catch (error) {
        console.warn(`  Warning: Could not analyze ${bundleType.type} bundle - ${error.message}`);
        this.analysis.bundles[bundleType.type] = { error: error.message };
      }
    }
  }

  async analyzeBundleDirectory(dirPath, extensions) {
    const analysis = {
      files: [],
      totalSize: 0,
      fileCount: 0,
      largestFile: null,
      duplicates: [],
      compressionRatio: 0
    };

    try {
      const files = await fs.readdir(dirPath);

      for (const file of files) {
        const fileExt = path.extname(file).toLowerCase();
        if (extensions.includes(fileExt)) {
          const filePath = path.join(dirPath, file);
          const stats = await fs.stat(filePath);

          const fileInfo = {
            name: file,
            path: filePath,
            size: stats.size,
            sizeFormatted: this.formatBytes(stats.size),
            extension: fileExt,
            lastModified: stats.mtime,
            gzipSize: await this.estimateGzipSize(filePath)
          };

          analysis.files.push(fileInfo);
          analysis.totalSize += stats.size;
          analysis.fileCount++;

          if (!analysis.largestFile || stats.size > analysis.largestFile.size) {
            analysis.largestFile = fileInfo;
          }
        }
      }

      // Sort files by size descending
      analysis.files.sort((a, b) => b.size - a.size);

      // Calculate compression ratio
      const totalGzipSize = analysis.files.reduce((sum, file) => sum + file.gzipSize, 0);
      analysis.compressionRatio = analysis.totalSize > 0 ?
        ((analysis.totalSize - totalGzipSize) / analysis.totalSize) * 100 : 0;

      // Find potential duplicates (by size)
      analysis.duplicates = this.findPotentialDuplicates(analysis.files);

    } catch (error) {
      if (error.code !== 'ENOENT') {
        throw error;
      }
    }

    return analysis;
  }

  async analyzeDependencies() {
    console.log('📦 Analyzing package dependencies...');

    try {
      const packageJson = JSON.parse(await fs.readFile('./package.json', 'utf8'));

      this.analysis.dependencies = {
        production: Object.keys(packageJson.dependencies || {}),
        development: Object.keys(packageJson.devDependencies || {}),
        total: Object.keys({...packageJson.dependencies, ...packageJson.devDependencies}).length,
        bundledSize: await this.estimateDependencySize(packageJson.dependencies || {}),
        recommendations: []
      };

      // Check for heavy dependencies
      const heavyDeps = await this.identifyHeavyDependencies(packageJson.dependencies || {});
      this.analysis.dependencies.heavy = heavyDeps;

      console.log(`  Production dependencies: ${this.analysis.dependencies.production.length}`);
      console.log(`  Development dependencies: ${this.analysis.dependencies.development.length}`);

    } catch (error) {
      console.warn(`  Warning: Could not analyze dependencies - ${error.message}`);
      this.analysis.dependencies = { error: error.message };
    }
  }

  checkBudgets() {
    console.log('📉 Checking performance budgets...');

    const totalSize = Object.values(this.analysis.bundles)
      .reduce((sum, bundle) => sum + (bundle.totalSize || 0), 0);

    // Check individual budgets
    Object.keys(this.budgets).forEach(type => {
      if (type === 'total') {
        this.analysis.budgetStatus[type] = {
          budget: this.budgets[type],
          actual: totalSize,
          utilization: (totalSize / this.budgets[type]) * 100,
          status: totalSize > this.budgets[type] ? 'exceeded' :
                 (totalSize / this.budgets[type] > 0.8 ? 'warning' : 'good')
        };
      } else if (this.analysis.bundles[type] && !this.analysis.bundles[type].error) {
        const bundle = this.analysis.bundles[type];
        this.analysis.budgetStatus[type] = {
          budget: bundle.budget,
          actual: bundle.totalSize,
          utilization: bundle.budgetUtilization,
          status: bundle.totalSize > bundle.budget ? 'exceeded' :
                 bundle.budgetUtilization > 80 ? 'warning' : 'good'
        };
      }
    });

    // Log budget status
    Object.entries(this.analysis.budgetStatus).forEach(([type, status]) => {
      const emoji = status.status === 'exceeded' ? '❌' :
                   status.status === 'warning' ? '⚠️' : '✅';
      console.log(`  ${emoji} ${type}: ${this.formatBytes(status.actual)} / ${this.formatBytes(status.budget)} (${status.utilization.toFixed(1)}%)`);
    });
  }

  generateRecommendations() {
    console.log('💡 Generating optimization recommendations...');

    const recommendations = [];

    // Bundle size recommendations
    Object.entries(this.analysis.bundles).forEach(([type, bundle]) => {
      if (bundle.error) return;

      if (bundle.budgetUtilization > 100) {
        recommendations.push({
          type: 'budget',
          priority: 'high',
          category: type,
          issue: `${type} bundle exceeds budget`,
          description: `${type} bundle is ${this.formatBytes(bundle.totalSize)} (${bundle.budgetUtilization.toFixed(1)}% of budget). Consider code splitting, tree shaking, or removing unused code.`,
          savings: bundle.totalSize - bundle.budget
        });
      } else if (bundle.budgetUtilization > 80) {
        recommendations.push({
          type: 'budget',
          priority: 'medium',
          category: type,
          issue: `${type} bundle approaching budget limit`,
          description: `${type} bundle is ${bundle.budgetUtilization.toFixed(1)}% of budget. Monitor growth and optimize proactively.`,
          savings: 0
        });
      }

      // Large file recommendations
      if (bundle.largestFile && bundle.largestFile.size > 50 * 1024) {
        recommendations.push({
          type: 'optimization',
          priority: 'medium',
          category: type,
          issue: `Large ${type} file detected`,
          description: `${bundle.largestFile.name} is ${bundle.largestFile.sizeFormatted}. Consider splitting or optimizing this file.`,
          file: bundle.largestFile.name,
          savings: bundle.largestFile.size * 0.3 // Estimate 30% reduction
        });
      }

      // Compression recommendations
      if (bundle.compressionRatio < 60 && type === 'javascript') {
        recommendations.push({
          type: 'compression',
          priority: 'low',
          category: type,
          issue: 'Poor compression ratio',
          description: `${type} files have ${bundle.compressionRatio.toFixed(1)}% compression ratio. Enable gzip/brotli compression on the server.`,
          savings: bundle.totalSize * 0.4
        });
      }
    });

    // Dependency recommendations
    if (this.analysis.dependencies.heavy) {
      this.analysis.dependencies.heavy.forEach(dep => {
        recommendations.push({
          type: 'dependency',
          priority: 'medium',
          category: 'javascript',
          issue: `Heavy dependency: ${dep.name}`,
          description: `${dep.name} contributes significantly to bundle size. Consider alternatives or lazy loading.`,
          dependency: dep.name,
          savings: dep.estimatedSize * 0.5
        });
      });
    }

    // Sort by priority and savings
    recommendations.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      const priorityDiff = priorityOrder[b.priority] - priorityOrder[a.priority];
      if (priorityDiff !== 0) return priorityDiff;
      return (b.savings || 0) - (a.savings || 0);
    });

    this.analysis.recommendations = recommendations;
    console.log(`  Generated ${recommendations.length} recommendations`);
  }

  createSummary() {
    const totalSize = Object.values(this.analysis.bundles)
      .reduce((sum, bundle) => sum + (bundle.totalSize || 0), 0);

    const totalBudget = this.budgets.total;
    const budgetUtilization = (totalSize / totalBudget) * 100;

    const exceededBudgets = Object.values(this.analysis.budgetStatus)
      .filter(status => status.status === 'exceeded').length;

    const highPriorityRecommendations = this.analysis.recommendations
      .filter(rec => rec.priority === 'high').length;

    this.analysis.summary = {
      totalSize: totalSize,
      totalSizeFormatted: this.formatBytes(totalSize),
      budgetUtilization: budgetUtilization,
      budgetStatus: budgetUtilization > 100 ? 'exceeded' :
                   budgetUtilization > 80 ? 'warning' : 'good',
      exceededBudgets: exceededBudgets,
      totalRecommendations: this.analysis.recommendations.length,
      highPriorityRecommendations: highPriorityRecommendations,
      potentialSavings: this.analysis.recommendations
        .reduce((sum, rec) => sum + (rec.savings || 0), 0),
      healthScore: this.calculateHealthScore()
    };
  }

  calculateHealthScore() {
    let score = 100;

    // Deduct for budget overruns
    Object.values(this.analysis.budgetStatus).forEach(status => {
      if (status.status === 'exceeded') {
        score -= 20;
      } else if (status.status === 'warning') {
        score -= 10;
      }
    });

    // Deduct for high-priority recommendations
    const highPriorityCount = this.analysis.recommendations
      .filter(rec => rec.priority === 'high').length;
    score -= highPriorityCount * 15;

    // Deduct for medium-priority recommendations
    const mediumPriorityCount = this.analysis.recommendations
      .filter(rec => rec.priority === 'medium').length;
    score -= mediumPriorityCount * 5;

    return Math.max(0, Math.min(100, score));
  }

  async saveResults() {
    // Save main analysis file
    const analysisPath = path.join(this.outputDir, 'bundle-analysis.json');
    await fs.writeFile(analysisPath, JSON.stringify(this.analysis, null, 2));

    // Generate HTML report
    await this.generateHTMLReport();

    // Generate CSV summary
    await this.generateCSVSummary();

    console.log('📄 Reports generated:');
    console.log(`  - JSON: ${analysisPath}`);
    console.log(`  - HTML: ${path.join(this.outputDir, 'bundle-report.html')}`);
    console.log(`  - CSV: ${path.join(this.outputDir, 'bundle-summary.csv')}`);
  }

  async generateHTMLReport() {
    const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bundle Analysis Report - Pynomaly</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #f8fafc; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }
        .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .good { color: #10b981; }
        .warning { color: #f59e0b; }
        .exceeded { color: #ef4444; }
        .table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .table th, .table td { padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }
        .table th { background: #f9fafb; font-weight: 600; }
        .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.875em; font-weight: 500; }
        .badge.good { background: #d1fae5; color: #065f46; }
        .badge.warning { background: #fef3c7; color: #92400e; }
        .badge.exceeded { background: #fee2e2; color: #991b1b; }
        .recommendation { background: white; border-radius: 8px; padding: 16px; margin-bottom: 16px; border-left: 4px solid #3b82f6; }
        .recommendation.high { border-left-color: #ef4444; }
        .recommendation.medium { border-left-color: #f59e0b; }
        .recommendation.low { border-left-color: #10b981; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Bundle Analysis Report</h1>
            <p>Generated on ${new Date().toLocaleString()}</p>
        </div>

        <div class="summary">
            <div class="card">
                <h3>Total Bundle Size</h3>
                <div class="metric ${this.analysis.summary.budgetStatus}">${this.analysis.summary.totalSizeFormatted}</div>
                <p>${this.analysis.summary.budgetUtilization.toFixed(1)}% of budget</p>
            </div>
            <div class="card">
                <h3>Health Score</h3>
                <div class="metric ${this.analysis.summary.healthScore >= 80 ? 'good' : this.analysis.summary.healthScore >= 60 ? 'warning' : 'exceeded'}">
                  ${this.analysis.summary.healthScore}/100</div>
            </div>
            <div class="card">
                <h3>Recommendations</h3>
                <div class="metric">${this.analysis.summary.totalRecommendations}</div>
                <p>${this.analysis.summary.highPriorityRecommendations} high priority</p>
            </div>
            <div class="card">
                <h3>Potential Savings</h3>
                <div class="metric">${this.formatBytes(this.analysis.summary.potentialSavings)}</div>
            </div>
        </div>

        ${this.generateBundleTableHTML()}
        ${this.generateRecommendationsHTML()}
    </div>
</body>
</html>`;

    const htmlPath = path.join(this.outputDir, 'bundle-report.html');
    await fs.writeFile(htmlPath, html);
  }

  generateBundleTableHTML() {
    return `
        <div class="card">
            <h2>Bundle Breakdown</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Type</th>
                        <th>Size</th>
                        <th>Budget</th>
                        <th>Utilization</th>
                        <th>Files</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    ${Object.entries(this.analysis.bundles).map(([type, bundle]) => {
                      if (bundle.error) return '';
                      const status = bundle.budgetUtilization > 100 ? 'exceeded' :
                                   bundle.budgetUtilization > 80 ? 'warning' : 'good';
                      return `
                        <tr>
                            <td><strong>${type}</strong></td>
                            <td>${this.formatBytes(bundle.totalSize)}</td>
                            <td>${this.formatBytes(bundle.budget)}</td>
                            <td>${bundle.budgetUtilization.toFixed(1)}%</td>
                            <td>${bundle.fileCount}</td>
                            <td><span class="badge ${status}">${status.toUpperCase()}</span></td>
                        </tr>
                      `;
                    }).join('')}
                </tbody>
            </table>
        </div>
    `;
  }

  generateRecommendationsHTML() {
    if (this.analysis.recommendations.length === 0) {
      return '<div class="card"><h2>Recommendations</h2><p>No optimization opportunities found. Great job!</p></div>';
    }

    return `
        <div class="card">
            <h2>Optimization Recommendations</h2>
            ${this.analysis.recommendations.map(rec => `
                <div class="recommendation ${rec.priority}">
                    <h3>${rec.issue}</h3>
                    <p>${rec.description}</p>
                    ${rec.savings ? `<p><strong>Potential savings:</strong> ${this.formatBytes(rec.savings)}</p>` : ''}
                    <span class="badge ${rec.priority}">${rec.priority.toUpperCase()} PRIORITY</span>
                </div>
            `).join('')}
        </div>
    `;
  }

  async generateCSVSummary() {
    const csv = [
      'Type,Size (bytes),Size (formatted),Budget (bytes),Utilization (%),Status,File Count',
      ...Object.entries(this.analysis.bundles).map(([type, bundle]) => {
        if (bundle.error) return `${type},ERROR,ERROR,ERROR,ERROR,ERROR,ERROR`;
        const status = bundle.budgetUtilization > 100 ? 'EXCEEDED' :
                     bundle.budgetUtilization > 80 ? 'WARNING' : 'GOOD';
        return `${type},${bundle.totalSize},${this.formatBytes(bundle.totalSize)},${bundle.budget},${bundle.budgetUtilization.toFixed(1)},${status},${bundle.fileCount}`;
      })
    ].join('\n');

    const csvPath = path.join(this.outputDir, 'bundle-summary.csv');
    await fs.writeFile(csvPath, csv);
  }

  // Helper methods
  async estimateGzipSize(filePath) {
    try {
      const content = await fs.readFile(filePath);
      const zlib = require('zlib');
      const compressed = zlib.gzipSync(content);
      return compressed.length;
    } catch (error) {
      return 0;
    }
  }

  findPotentialDuplicates(files) {
    const sizeGroups = {};
    files.forEach(file => {
      if (!sizeGroups[file.size]) {
        sizeGroups[file.size] = [];
      }
      sizeGroups[file.size].push(file);
    });

    return Object.values(sizeGroups)
      .filter(group => group.length > 1)
      .flat();
  }

  async estimateDependencySize(dependencies) {
    // This would require npm package size analysis
    // For now, return a placeholder
    return Object.keys(dependencies).length * 50 * 1024; // Estimate 50KB per package
  }

  async identifyHeavyDependencies(dependencies) {
    // Placeholder for dependency size analysis
    const heavyDeps = [
      { name: 'd3', estimatedSize: 100 * 1024 },
      { name: 'echarts', estimatedSize: 150 * 1024 }
    ];

    return heavyDeps.filter(dep => dependencies[dep.name]);
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
  const analyzer = new BundleAnalyzer({
    staticDir: process.argv[2] || './src/pynomaly/presentation/web/static',
    outputDir: process.argv[3] || './test_reports/bundle-analysis'
  });

  analyzer.analyze()
    .then(results => {
      console.log(`\n✅ Bundle analysis completed with ${results.summary.healthScore}/100 health score`);
      process.exit(results.summary.healthScore < 60 ? 1 : 0);
    })
    .catch(error => {
      console.error('❌ Bundle analysis failed:', error);
      process.exit(1);
    });
}

module.exports = BundleAnalyzer;
