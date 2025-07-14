/**
 * Web Performance Monitoring Script
 * Real-time performance tracking and optimization recommendations
 */

const fs = require('fs');
const path = require('path');
const lighthouse = require('lighthouse');
const chromeLauncher = require('chrome-launcher');

class WebPerformanceMonitor {
  constructor(options = {}) {
    this.options = {
      url: 'http://localhost:8000',
      outputDir: './test_reports/performance',
      budgets: {
        totalJS: 500 * 1024, // 500KB
        totalCSS: 50 * 1024, // 50KB
        mainBundle: 250 * 1024, // 250KB
        chunkSize: 100 * 1024, // 100KB
        imageSize: 500 * 1024, // 500KB
        lcp: 2500, // 2.5s
        fid: 100, // 100ms
        cls: 0.1, // 0.1
        fcp: 1800, // 1.8s
        tti: 3800, // 3.8s
      },
      ...options
    };
    
    this.results = {
      timestamp: new Date().toISOString(),
      metrics: {},
      budgetViolations: [],
      recommendations: [],
      scores: {}
    };
    
    this.ensureOutputDir();
  }
  
  ensureOutputDir() {
    if (!fs.existsSync(this.options.outputDir)) {
      fs.mkdirSync(this.options.outputDir, { recursive: true });
    }
  }
  
  async runFullAudit() {
    console.log('🚀 Starting comprehensive web performance audit...');
    
    try {
      // 1. Bundle size analysis
      console.log('📦 Analyzing bundle sizes...');
      await this.analyzeBundleSizes();
      
      // 2. Lighthouse audit
      console.log('🔍 Running Lighthouse audit...');
      await this.runLighthouseAudit();
      
      // 3. Resource optimization analysis
      console.log('🖼️ Analyzing resource optimization...');
      await this.analyzeResourceOptimization();
      
      // 4. Check performance budgets
      console.log('💰 Checking performance budgets...');
      this.checkPerformanceBudgets();
      
      // 5. Generate recommendations
      console.log('💡 Generating optimization recommendations...');
      this.generateRecommendations();
      
      // 6. Save results
      await this.saveResults();
      
      // 7. Generate report
      await this.generateReport();
      
      console.log('✅ Performance audit completed!');
      console.log(`📊 Report saved to: ${this.options.outputDir}/performance-report.html`);
      
    } catch (error) {
      console.error('❌ Performance audit failed:', error);
      throw error;
    }
  }
  
  async analyzeBundleSizes() {
    const distDir = path.join(process.cwd(), 'src/pynomaly/presentation/web/static/js/dist');
    
    if (!fs.existsSync(distDir)) {
      console.warn('⚠️ Build directory not found. Run npm run build first.');
      return;
    }
    
    const bundleReport = path.join(distDir, 'bundle-report.json');
    
    if (fs.existsSync(bundleReport)) {
      const report = JSON.parse(fs.readFileSync(bundleReport, 'utf8'));
      this.results.bundleAnalysis = report;
      
      // Calculate total sizes
      const totalJS = Object.values(report.bundles).reduce((sum, size) => sum + size, 0);
      this.results.metrics.totalJSSize = totalJS;
      
      console.log(`  📦 Total JS size: ${(totalJS / 1024).toFixed(1)}KB`);
      console.log(`  📦 Main bundle: ${(report.bundles.main / 1024).toFixed(1)}KB`);
      console.log(`  📦 Number of chunks: ${Object.keys(report.bundles).length}`);
      
    } else {
      console.warn('⚠️ Bundle report not found. Run npm run build-js:analyze first.');
    }
  }
  
  async runLighthouseAudit() {
    const chrome = await chromeLauncher.launch({ chromeFlags: ['--headless'] });
    
    try {
      const options = {
        logLevel: 'info',
        output: 'json',
        onlyCategories: ['performance'],
        port: chrome.port,
      };
      
      const runnerResult = await lighthouse(this.options.url, options);
      const lhr = runnerResult.lhr;
      
      // Extract key metrics
      this.results.metrics.lcp = lhr.audits['largest-contentful-paint'].numericValue;
      this.results.metrics.fid = lhr.audits['max-potential-fid'].numericValue;
      this.results.metrics.cls = lhr.audits['cumulative-layout-shift'].numericValue;
      this.results.metrics.fcp = lhr.audits['first-contentful-paint'].numericValue;
      this.results.metrics.tti = lhr.audits['interactive'].numericValue;
      this.results.metrics.speed = lhr.audits['speed-index'].numericValue;
      
      // Performance score
      this.results.scores.performance = Math.round(lhr.categories.performance.score * 100);
      
      // Detailed audit results
      this.results.lighthouseResults = {
        audits: lhr.audits,
        opportunities: lhr.audits,
        diagnostics: lhr.audits
      };
      
      console.log(`  🔍 Performance Score: ${this.results.scores.performance}/100`);
      console.log(`  🔍 LCP: ${this.results.metrics.lcp}ms`);
      console.log(`  🔍 FID: ${this.results.metrics.fid}ms`);
      console.log(`  🔍 CLS: ${this.results.metrics.cls}`);
      
    } finally {
      await chrome.kill();
    }
  }
  
  async analyzeResourceOptimization() {
    const staticDir = path.join(process.cwd(), 'src/pynomaly/presentation/web/static');
    
    // Analyze images
    const imgDir = path.join(staticDir, 'img');
    if (fs.existsSync(imgDir)) {
      const images = this.getFilesRecursively(imgDir, ['.jpg', '.jpeg', '.png', '.gif', '.svg']);
      
      let totalImageSize = 0;
      const largeImages = [];
      
      images.forEach(imgPath => {
        const stats = fs.statSync(imgPath);
        totalImageSize += stats.size;
        
        if (stats.size > this.options.budgets.imageSize) {
          largeImages.push({
            path: path.relative(staticDir, imgPath),
            size: stats.size
          });
        }
      });
      
      this.results.metrics.totalImageSize = totalImageSize;
      this.results.imageAnalysis = {
        totalImages: images.length,
        totalSize: totalImageSize,
        largeImages
      };
      
      console.log(`  🖼️ Total images: ${images.length}`);
      console.log(`  🖼️ Total image size: ${(totalImageSize / 1024).toFixed(1)}KB`);
      console.log(`  🖼️ Large images (>${(this.options.budgets.imageSize / 1024).toFixed(0)}KB): ${largeImages.length}`);
    }
    
    // Analyze CSS
    const cssFiles = this.getFilesRecursively(path.join(staticDir, 'css'), ['.css']);
    let totalCSSSize = 0;
    
    cssFiles.forEach(cssPath => {
      const stats = fs.statSync(cssPath);
      totalCSSSize += stats.size;
    });
    
    this.results.metrics.totalCSSSize = totalCSSSize;
    console.log(`  🎨 Total CSS size: ${(totalCSSSize / 1024).toFixed(1)}KB`);
  }
  
  getFilesRecursively(dir, extensions) {
    const files = [];
    
    if (!fs.existsSync(dir)) return files;
    
    const items = fs.readdirSync(dir);
    
    items.forEach(item => {
      const fullPath = path.join(dir, item);
      const stats = fs.statSync(fullPath);
      
      if (stats.isDirectory()) {
        files.push(...this.getFilesRecursively(fullPath, extensions));
      } else if (extensions.some(ext => item.toLowerCase().endsWith(ext))) {
        files.push(fullPath);
      }
    });
    
    return files;
  }
  
  checkPerformanceBudgets() {
    const violations = [];
    
    // Check bundle size budgets
    if (this.results.metrics.totalJSSize > this.options.budgets.totalJS) {
      violations.push({
        type: 'bundle-size',
        metric: 'Total JS Size',
        actual: this.results.metrics.totalJSSize,
        budget: this.options.budgets.totalJS,
        severity: 'high'
      });
    }
    
    if (this.results.bundleAnalysis?.bundles?.main > this.options.budgets.mainBundle) {
      violations.push({
        type: 'bundle-size',
        metric: 'Main Bundle Size',
        actual: this.results.bundleAnalysis.bundles.main,
        budget: this.options.budgets.mainBundle,
        severity: 'high'
      });
    }
    
    if (this.results.metrics.totalCSSSize > this.options.budgets.totalCSS) {
      violations.push({
        type: 'asset-size',
        metric: 'Total CSS Size',
        actual: this.results.metrics.totalCSSSize,
        budget: this.options.budgets.totalCSS,
        severity: 'medium'
      });
    }
    
    // Check performance metrics budgets
    if (this.results.metrics.lcp > this.options.budgets.lcp) {
      violations.push({
        type: 'performance',
        metric: 'Largest Contentful Paint',
        actual: this.results.metrics.lcp,
        budget: this.options.budgets.lcp,
        severity: 'high'
      });
    }
    
    if (this.results.metrics.fid > this.options.budgets.fid) {
      violations.push({
        type: 'performance',
        metric: 'First Input Delay',
        actual: this.results.metrics.fid,
        budget: this.options.budgets.fid,
        severity: 'high'
      });
    }
    
    if (this.results.metrics.cls > this.options.budgets.cls) {
      violations.push({
        type: 'performance',
        metric: 'Cumulative Layout Shift',
        actual: this.results.metrics.cls,
        budget: this.options.budgets.cls,
        severity: 'medium'
      });
    }
    
    this.results.budgetViolations = violations;
    
    if (violations.length > 0) {
      console.log(`  ⚠️ Budget violations: ${violations.length}`);
      violations.forEach(violation => {
        console.log(`    • ${violation.metric}: ${this.formatValue(violation.actual, violation.type)} (budget: ${this.formatValue(violation.budget, violation.type)})`);
      });
    } else {
      console.log('  ✅ All performance budgets met!');
    }
  }
  
  formatValue(value, type) {
    switch (type) {
      case 'bundle-size':
      case 'asset-size':
        return `${(value / 1024).toFixed(1)}KB`;
      case 'performance':
        return value < 1 ? value.toFixed(3) : `${value}ms`;
      default:
        return value;
    }
  }
  
  generateRecommendations() {
    const recommendations = [];
    
    // Bundle optimization recommendations
    if (this.results.metrics.totalJSSize > this.options.budgets.totalJS) {
      recommendations.push({
        category: 'Bundle Optimization',
        priority: 'high',
        title: 'Reduce JavaScript bundle size',
        description: 'Your JavaScript bundles exceed the recommended size budget.',
        actions: [
          'Implement code splitting for route-based chunks',
          'Enable tree shaking to remove unused code',
          'Use dynamic imports for non-critical components',
          'Consider using a smaller framework or library alternatives'
        ]
      });
    }
    
    // Performance recommendations
    if (this.results.metrics.lcp > this.options.budgets.lcp) {
      recommendations.push({
        category: 'Loading Performance',
        priority: 'high',
        title: 'Improve Largest Contentful Paint',
        description: 'The largest content element takes too long to load.',
        actions: [
          'Optimize and compress images',
          'Use responsive images with appropriate sizes',
          'Implement lazy loading for below-the-fold content',
          'Preload critical resources',
          'Use a CDN for static assets'
        ]
      });
    }
    
    if (this.results.metrics.cls > this.options.budgets.cls) {
      recommendations.push({
        category: 'Layout Stability',
        priority: 'medium',
        title: 'Reduce Cumulative Layout Shift',
        description: 'Page layout shifts are causing poor user experience.',
        actions: [
          'Set explicit width and height for images and videos',
          'Reserve space for ads and dynamic content',
          'Avoid inserting content above existing content',
          'Use CSS transform instead of changing layout properties'
        ]
      });
    }
    
    // Image optimization recommendations
    if (this.results.imageAnalysis?.largeImages?.length > 0) {
      recommendations.push({
        category: 'Asset Optimization',
        priority: 'medium',
        title: 'Optimize large images',
        description: `${this.results.imageAnalysis.largeImages.length} images exceed the recommended size.`,
        actions: [
          'Compress images using modern formats (WebP, AVIF)',
          'Implement responsive images with srcset',
          'Use lazy loading for images',
          'Consider using a image optimization service'
        ]
      });
    }
    
    // Caching recommendations
    recommendations.push({
      category: 'Caching Strategy',
      priority: 'medium',
      title: 'Implement comprehensive caching',
      description: 'Optimize caching strategy for better performance.',
      actions: [
        'Enable service worker caching for offline support',
        'Implement aggressive caching for static assets',
        'Use stale-while-revalidate for API responses',
        'Add cache headers for static resources'
      ]
    });
    
    this.results.recommendations = recommendations;
    
    console.log(`  💡 Generated ${recommendations.length} optimization recommendations`);
  }
  
  async saveResults() {
    const resultsPath = path.join(this.options.outputDir, 'performance-results.json');
    fs.writeFileSync(resultsPath, JSON.stringify(this.results, null, 2));
    console.log(`  💾 Results saved to: ${resultsPath}`);
  }
  
  async generateReport() {
    const reportHtml = this.generateHTMLReport();
    const reportPath = path.join(this.options.outputDir, 'performance-report.html');
    fs.writeFileSync(reportPath, reportHtml);
    
    // Also generate a summary report
    const summaryPath = path.join(this.options.outputDir, 'performance-summary.md');
    const summaryMarkdown = this.generateMarkdownSummary();
    fs.writeFileSync(summaryPath, summaryMarkdown);
  }
  
  generateHTMLReport() {
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pynomaly Web Performance Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 30px; }
        h1 { color: #2563eb; margin-bottom: 30px; }
        h2 { color: #374151; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric { background: #f8fafc; padding: 20px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #1f2937; }
        .metric-label { color: #6b7280; font-size: 0.9em; margin-top: 5px; }
        .violation { background: #fef2f2; border-left: 4px solid #ef4444; padding: 15px; margin: 10px 0; }
        .recommendation { background: #f0f9ff; border-left: 4px solid #3b82f6; padding: 15px; margin: 10px 0; }
        .priority-high { border-left-color: #ef4444; }
        .priority-medium { border-left-color: #f59e0b; }
        .actions { margin-top: 10px; }
        .actions li { margin: 5px 0; }
        .score { font-size: 3em; font-weight: bold; text-align: center; padding: 20px; }
        .score.good { color: #22c55e; }
        .score.needs-improvement { color: #f59e0b; }
        .score.poor { color: #ef4444; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Pynomaly Web Performance Report</h1>
        <p><strong>Generated:</strong> ${this.results.timestamp}</p>
        
        <h2>📊 Performance Score</h2>
        <div class="score ${this.getScoreClass(this.results.scores.performance)}">${this.results.scores.performance}/100</div>
        
        <h2>📈 Core Web Vitals</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">${this.results.metrics.lcp || 'N/A'}ms</div>
                <div class="metric-label">Largest Contentful Paint</div>
            </div>
            <div class="metric">
                <div class="metric-value">${this.results.metrics.fid || 'N/A'}ms</div>
                <div class="metric-label">First Input Delay</div>
            </div>
            <div class="metric">
                <div class="metric-value">${this.results.metrics.cls || 'N/A'}</div>
                <div class="metric-label">Cumulative Layout Shift</div>
            </div>
            <div class="metric">
                <div class="metric-value">${this.results.metrics.fcp || 'N/A'}ms</div>
                <div class="metric-label">First Contentful Paint</div>
            </div>
        </div>
        
        <h2>📦 Bundle Analysis</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">${(this.results.metrics.totalJSSize / 1024).toFixed(1)}KB</div>
                <div class="metric-label">Total JavaScript</div>
            </div>
            <div class="metric">
                <div class="metric-value">${(this.results.metrics.totalCSSSize / 1024).toFixed(1)}KB</div>
                <div class="metric-label">Total CSS</div>
            </div>
            <div class="metric">
                <div class="metric-value">${this.results.imageAnalysis?.totalImages || 0}</div>
                <div class="metric-label">Total Images</div>
            </div>
            <div class="metric">
                <div class="metric-value">${((this.results.metrics.totalImageSize || 0) / 1024).toFixed(1)}KB</div>
                <div class="metric-label">Total Image Size</div>
            </div>
        </div>
        
        ${this.results.budgetViolations.length > 0 ? `
        <h2>⚠️ Budget Violations</h2>
        ${this.results.budgetViolations.map(violation => `
            <div class="violation">
                <strong>${violation.metric}</strong><br>
                Actual: ${this.formatValue(violation.actual, violation.type)} | Budget: ${this.formatValue(violation.budget, violation.type)}
            </div>
        `).join('')}
        ` : '<h2>✅ All Performance Budgets Met</h2>'}
        
        <h2>💡 Optimization Recommendations</h2>
        ${this.results.recommendations.map(rec => `
            <div class="recommendation priority-${rec.priority}">
                <h3>${rec.title}</h3>
                <p>${rec.description}</p>
                <ul class="actions">
                    ${rec.actions.map(action => `<li>${action}</li>`).join('')}
                </ul>
            </div>
        `).join('')}
    </div>
</body>
</html>`;
  }
  
  generateMarkdownSummary() {
    return `# Pynomaly Web Performance Report

**Generated:** ${this.results.timestamp}

## Performance Score: ${this.results.scores.performance}/100

## Core Web Vitals
- **LCP:** ${this.results.metrics.lcp || 'N/A'}ms
- **FID:** ${this.results.metrics.fid || 'N/A'}ms  
- **CLS:** ${this.results.metrics.cls || 'N/A'}
- **FCP:** ${this.results.metrics.fcp || 'N/A'}ms

## Bundle Sizes
- **Total JS:** ${(this.results.metrics.totalJSSize / 1024).toFixed(1)}KB
- **Total CSS:** ${(this.results.metrics.totalCSSSize / 1024).toFixed(1)}KB
- **Total Images:** ${((this.results.metrics.totalImageSize || 0) / 1024).toFixed(1)}KB

## Budget Violations
${this.results.budgetViolations.length === 0 ? '✅ All budgets met!' : 
  this.results.budgetViolations.map(v => `- ⚠️ ${v.metric}: ${this.formatValue(v.actual, v.type)} (budget: ${this.formatValue(v.budget, v.type)})`).join('\n')}

## Top Recommendations
${this.results.recommendations.slice(0, 3).map(rec => `
### ${rec.title}
${rec.description}
${rec.actions.map(action => `- ${action}`).join('\n')}
`).join('\n')}`;
  }
  
  getScoreClass(score) {
    if (score >= 90) return 'good';
    if (score >= 50) return 'needs-improvement';
    return 'poor';
  }
}

// CLI interface
if (require.main === module) {
  const monitor = new WebPerformanceMonitor();
  monitor.runFullAudit().catch(console.error);
}

module.exports = WebPerformanceMonitor;