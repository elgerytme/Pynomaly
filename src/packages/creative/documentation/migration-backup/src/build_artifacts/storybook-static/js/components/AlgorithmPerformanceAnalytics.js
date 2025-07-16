/**
 * Algorithm Performance Analytics Component
 * Advanced analytics for comparing and evaluating anomaly detection algorithms
 */

export class AlgorithmPerformanceAnalytics {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      width: 1200,
      height: 800,
      enableRealTimeUpdates: true,
      enableStatisticalTests: true,
      enableBenchmarking: true,
      refreshInterval: 5000,
      maxHistoryLength: 1000,
      confidenceLevel: 0.95,
      ...options
    };

    this.algorithmData = new Map();
    this.performanceHistory = [];
    this.benchmarkResults = new Map();
    this.statisticalTests = new Map();
    this.currentMetric = 'accuracy';
    this.currentTimeRange = '24h';
    this.comparisonMode = 'relative';

    this.init();
  }

  init() {
    this.createLayout();
    this.setupEventHandlers();
    this.initializeCharts();
    this.startPerformanceTracking();
  }

  createLayout() {
    this.container.innerHTML = `
      <div class="algorithm-analytics-container">
        <div class="analytics-header">
          <div class="header-title">
            <h2>Algorithm Performance Analytics</h2>
            <div class="last-updated">
              Last updated: <span id="last-updated-time">Never</span>
            </div>
          </div>

          <div class="analytics-controls">
            <div class="control-group">
              <label for="metric-selector">Metric:</label>
              <select id="metric-selector">
                <option value="accuracy">Accuracy</option>
                <option value="precision">Precision</option>
                <option value="recall">Recall</option>
                <option value="f1_score">F1 Score</option>
                <option value="auc_roc">AUC-ROC</option>
                <option value="processing_time">Processing Time</option>
                <option value="memory_usage">Memory Usage</option>
                <option value="false_positive_rate">False Positive Rate</option>
                <option value="true_positive_rate">True Positive Rate</option>
              </select>
            </div>

            <div class="control-group">
              <label for="time-range-selector">Time Range:</label>
              <select id="time-range-selector">
                <option value="1h">Last 1 hour</option>
                <option value="6h">Last 6 hours</option>
                <option value="24h" selected>Last 24 hours</option>
                <option value="7d">Last 7 days</option>
                <option value="30d">Last 30 days</option>
                <option value="all">All time</option>
              </select>
            </div>

            <div class="control-group">
              <label for="comparison-mode">Comparison:</label>
              <select id="comparison-mode">
                <option value="absolute">Absolute</option>
                <option value="relative" selected>Relative</option>
                <option value="normalized">Normalized</option>
              </select>
            </div>

            <div class="control-group">
              <button id="run-benchmark" class="btn-primary">Run Benchmark</button>
              <button id="export-report" class="btn-secondary">Export Report</button>
              <button id="statistical-tests" class="btn-info">Statistical Tests</button>
            </div>
          </div>
        </div>

        <div class="analytics-summary">
          <div class="summary-card best-algorithm">
            <div class="summary-icon">🥇</div>
            <div class="summary-content">
              <div class="summary-label">Best Algorithm</div>
              <div class="summary-value" id="best-algorithm">-</div>
              <div class="summary-metric" id="best-algorithm-score">-</div>
            </div>
          </div>

          <div class="summary-card fastest-algorithm">
            <div class="summary-icon">⚡</div>
            <div class="summary-content">
              <div class="summary-label">Fastest Algorithm</div>
              <div class="summary-value" id="fastest-algorithm">-</div>
              <div class="summary-metric" id="fastest-algorithm-time">-</div>
            </div>
          </div>

          <div class="summary-card most-consistent">
            <div class="summary-icon">📊</div>
            <div class="summary-content">
              <div class="summary-label">Most Consistent</div>
              <div class="summary-value" id="most-consistent">-</div>
              <div class="summary-metric" id="consistency-score">-</div>
            </div>
          </div>

          <div class="summary-card efficiency-leader">
            <div class="summary-icon">🎯</div>
            <div class="summary-content">
              <div class="summary-label">Most Efficient</div>
              <div class="summary-value" id="efficiency-leader">-</div>
              <div class="summary-metric" id="efficiency-score">-</div>
            </div>
          </div>
        </div>

        <div class="analytics-content">
          <div class="main-charts">
            <div class="chart-section">
              <div class="chart-header">
                <h3>Performance Comparison</h3>
                <div class="chart-controls">
                  <button id="toggle-error-bars" class="btn-sm btn-secondary">
                    <i class="fas fa-chart-line"></i> Error Bars
                  </button>
                  <button id="toggle-confidence-intervals" class="btn-sm btn-secondary">
                    <i class="fas fa-area-chart"></i> Confidence Intervals
                  </button>
                </div>
              </div>
              <div id="performance-comparison-chart" class="chart-container large"></div>
            </div>

            <div class="chart-section">
              <div class="chart-header">
                <h3>Performance Trends</h3>
                <div class="chart-controls">
                  <button id="toggle-moving-average" class="btn-sm btn-secondary">
                    <i class="fas fa-chart-line"></i> Moving Average
                  </button>
                  <button id="toggle-trend-lines" class="btn-sm btn-secondary">
                    <i class="fas fa-trending-up"></i> Trend Lines
                  </button>
                </div>
              </div>
              <div id="performance-trends-chart" class="chart-container large"></div>
            </div>

            <div class="chart-section">
              <div class="chart-header">
                <h3>Statistical Distribution</h3>
                <div class="chart-controls">
                  <select id="distribution-type">
                    <option value="histogram">Histogram</option>
                    <option value="violin">Violin Plot</option>
                    <option value="box">Box Plot</option>
                    <option value="kde">Kernel Density</option>
                  </select>
                </div>
              </div>
              <div id="distribution-chart" class="chart-container large"></div>
            </div>
          </div>

          <div class="side-panels">
            <div class="panel">
              <div class="panel-header">
                <h4>Algorithm Rankings</h4>
                <button id="refresh-rankings" class="btn-sm btn-secondary">
                  <i class="fas fa-sync"></i>
                </button>
              </div>
              <div class="panel-content">
                <div id="algorithm-rankings" class="rankings-container"></div>
              </div>
            </div>

            <div class="panel">
              <div class="panel-header">
                <h4>Performance Metrics</h4>
                <div class="metrics-controls">
                  <button id="toggle-relative-metrics" class="btn-sm btn-secondary">
                    <i class="fas fa-percent"></i> Relative
                  </button>
                </div>
              </div>
              <div class="panel-content">
                <div id="performance-metrics" class="metrics-container"></div>
              </div>
            </div>

            <div class="panel">
              <div class="panel-header">
                <h4>Statistical Tests</h4>
                <button id="run-all-tests" class="btn-sm btn-info">
                  <i class="fas fa-calculator"></i> Run All
                </button>
              </div>
              <div class="panel-content">
                <div id="statistical-results" class="tests-container"></div>
              </div>
            </div>

            <div class="panel">
              <div class="panel-header">
                <h4>Benchmarking</h4>
                <button id="start-benchmark" class="btn-sm btn-primary">
                  <i class="fas fa-play"></i> Start
                </button>
              </div>
              <div class="panel-content">
                <div id="benchmark-results" class="benchmark-container"></div>
              </div>
            </div>
          </div>
        </div>

        <div class="analytics-footer">
          <div class="footer-stats">
            <span>Algorithms: <span id="total-algorithms">0</span></span>
            <span>•</span>
            <span>Data Points: <span id="total-data-points">0</span></span>
            <span>•</span>
            <span>Confidence Level: <span id="confidence-level">95%</span></span>
            <span>•</span>
            <span>Last Benchmark: <span id="last-benchmark">Never</span></span>
          </div>
          <div class="footer-actions">
            <button id="export-csv" class="btn-link">
              <i class="fas fa-file-csv"></i> Export CSV
            </button>
            <button id="export-pdf" class="btn-link">
              <i class="fas fa-file-pdf"></i> Export PDF
            </button>
            <button id="share-results" class="btn-link">
              <i class="fas fa-share"></i> Share
            </button>
          </div>
        </div>
      </div>

      <!-- Statistical Tests Modal -->
      <div id="statistical-tests-modal" class="modal" style="display: none;">
        <div class="modal-content large">
          <div class="modal-header">
            <h3>Statistical Significance Tests</h3>
            <button id="close-tests-modal" class="btn-close">&times;</button>
          </div>
          <div class="modal-body">
            <div class="tests-tabs">
              <button class="tab-button active" data-tab="anova">ANOVA</button>
              <button class="tab-button" data-tab="ttest">T-Test</button>
              <button class="tab-button" data-tab="wilcoxon">Wilcoxon</button>
              <button class="tab-button" data-tab="friedman">Friedman</button>
              <button class="tab-button" data-tab="multiple">Multiple Comparisons</button>
            </div>
            <div class="tests-content">
              <div id="anova-results" class="test-results active"></div>
              <div id="ttest-results" class="test-results"></div>
              <div id="wilcoxon-results" class="test-results"></div>
              <div id="friedman-results" class="test-results"></div>
              <div id="multiple-results" class="test-results"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Benchmark Configuration Modal -->
      <div id="benchmark-modal" class="modal" style="display: none;">
        <div class="modal-content">
          <div class="modal-header">
            <h3>Benchmark Configuration</h3>
            <button id="close-benchmark-modal" class="btn-close">&times;</button>
          </div>
          <div class="modal-body">
            <div class="form-group">
              <label for="benchmark-datasets">Test Datasets:</label>
              <div class="checkbox-group">
                <label><input type="checkbox" id="synthetic-data" checked> Synthetic Data</label>
                <label><input type="checkbox" id="real-world-data" checked> Real-world Data</label>
                <label><input type="checkbox" id="adversarial-data"> Adversarial Data</label>
              </div>
            </div>

            <div class="form-group">
              <label for="benchmark-metrics">Metrics to Evaluate:</label>
              <div class="checkbox-group">
                <label><input type="checkbox" id="accuracy-metric" checked> Accuracy</label>
                <label><input type="checkbox" id="precision-metric" checked> Precision</label>
                <label><input type="checkbox" id="recall-metric" checked> Recall</label>
                <label><input type="checkbox" id="f1-metric" checked> F1 Score</label>
                <label><input type="checkbox" id="auc-metric" checked> AUC-ROC</label>
                <label><input type="checkbox" id="time-metric" checked> Processing Time</label>
              </div>
            </div>

            <div class="form-group">
              <label for="benchmark-runs">Number of Runs:</label>
              <input type="number" id="benchmark-runs" min="1" max="100" value="10">
            </div>

            <div class="form-group">
              <label for="cross-validation">Cross-Validation Folds:</label>
              <input type="number" id="cross-validation" min="2" max="20" value="5">
            </div>
          </div>
          <div class="modal-footer">
            <button id="start-benchmark-test" class="btn-primary">Start Benchmark</button>
            <button id="cancel-benchmark" class="btn-secondary">Cancel</button>
          </div>
        </div>
      </div>
    `;
  }

  setupEventHandlers() {
    // Metric selector
    document.getElementById('metric-selector').addEventListener('change', (e) => {
      this.currentMetric = e.target.value;
      this.updateAllCharts();
    });

    // Time range selector
    document.getElementById('time-range-selector').addEventListener('change', (e) => {
      this.currentTimeRange = e.target.value;
      this.updateAllCharts();
    });

    // Comparison mode selector
    document.getElementById('comparison-mode').addEventListener('change', (e) => {
      this.comparisonMode = e.target.value;
      this.updateAllCharts();
    });

    // Button event handlers
    document.getElementById('run-benchmark').addEventListener('click', () => {
      this.showBenchmarkModal();
    });

    document.getElementById('export-report').addEventListener('click', () => {
      this.exportPerformanceReport();
    });

    document.getElementById('statistical-tests').addEventListener('click', () => {
      this.showStatisticalTestsModal();
    });

    // Modal event handlers
    this.setupModalHandlers();

    // Chart control handlers
    this.setupChartControlHandlers();
  }

  setupModalHandlers() {
    // Statistical tests modal
    document.getElementById('close-tests-modal').addEventListener('click', () => {
      document.getElementById('statistical-tests-modal').style.display = 'none';
    });

    document.getElementById('run-all-tests').addEventListener('click', () => {
      this.runAllStatisticalTests();
    });

    // Benchmark modal
    document.getElementById('close-benchmark-modal').addEventListener('click', () => {
      document.getElementById('benchmark-modal').style.display = 'none';
    });

    document.getElementById('start-benchmark-test').addEventListener('click', () => {
      this.startBenchmarkTest();
    });

    document.getElementById('cancel-benchmark').addEventListener('click', () => {
      document.getElementById('benchmark-modal').style.display = 'none';
    });

    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
      button.addEventListener('click', (e) => {
        this.switchTab(e.target.dataset.tab);
      });
    });
  }

  setupChartControlHandlers() {
    document.getElementById('toggle-error-bars').addEventListener('click', () => {
      this.toggleErrorBars();
    });

    document.getElementById('toggle-confidence-intervals').addEventListener('click', () => {
      this.toggleConfidenceIntervals();
    });

    document.getElementById('toggle-moving-average').addEventListener('click', () => {
      this.toggleMovingAverage();
    });

    document.getElementById('toggle-trend-lines').addEventListener('click', () => {
      this.toggleTrendLines();
    });

    document.getElementById('distribution-type').addEventListener('change', (e) => {
      this.updateDistributionChart(e.target.value);
    });
  }

  initializeCharts() {
    this.performanceComparisonChart = new PerformanceComparisonChart(
      document.getElementById('performance-comparison-chart')
    );

    this.performanceTrendsChart = new PerformanceTrendsChart(
      document.getElementById('performance-trends-chart')
    );

    this.distributionChart = new DistributionChart(
      document.getElementById('distribution-chart')
    );
  }

  addAlgorithmData(algorithmName, performanceData) {
    if (!this.algorithmData.has(algorithmName)) {
      this.algorithmData.set(algorithmName, {
        name: algorithmName,
        history: [],
        statistics: {
          mean: 0,
          std: 0,
          min: Infinity,
          max: -Infinity,
          count: 0
        }
      });
    }

    const algorithm = this.algorithmData.get(algorithmName);
    algorithm.history.push({
      timestamp: Date.now(),
      ...performanceData
    });

    // Keep history within limits
    if (algorithm.history.length > this.options.maxHistoryLength) {
      algorithm.history.shift();
    }

    // Update statistics
    this.updateAlgorithmStatistics(algorithmName);

    // Update charts and displays
    this.updateAllCharts();
    this.updateSummaryCards();
    this.updateRankings();
    this.updateMetricsDisplay();
  }

  updateAlgorithmStatistics(algorithmName) {
    const algorithm = this.algorithmData.get(algorithmName);
    const values = algorithm.history.map(h => h[this.currentMetric]).filter(v => v !== undefined);

    if (values.length === 0) return;

    algorithm.statistics = {
      mean: values.reduce((a, b) => a + b, 0) / values.length,
      std: Math.sqrt(values.reduce((a, b) => a + Math.pow(b - algorithm.statistics.mean, 2), 0) / values.length),
      min: Math.min(...values),
      max: Math.max(...values),
      count: values.length,
      median: this.calculateMedian(values),
      q1: this.calculatePercentile(values, 0.25),
      q3: this.calculatePercentile(values, 0.75)
    };
  }

  calculateMedian(values) {
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  }

  calculatePercentile(values, percentile) {
    const sorted = values.slice().sort((a, b) => a - b);
    const index = percentile * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index - lower;
    return lower === upper ? sorted[lower] : sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }

  updateAllCharts() {
    this.performanceComparisonChart.update(this.algorithmData, this.currentMetric, this.comparisonMode);
    this.performanceTrendsChart.update(this.algorithmData, this.currentMetric, this.currentTimeRange);
    this.distributionChart.update(this.algorithmData, this.currentMetric);
  }

  updateSummaryCards() {
    const algorithms = Array.from(this.algorithmData.values());

    // Best algorithm (highest mean for current metric)
    const bestAlgorithm = algorithms.reduce((best, current) =>
      current.statistics.mean > best.statistics.mean ? current : best
    );

    document.getElementById('best-algorithm').textContent = bestAlgorithm.name;
    document.getElementById('best-algorithm-score').textContent =
      bestAlgorithm.statistics.mean.toFixed(3);

    // Fastest algorithm (lowest processing time)
    const fastestAlgorithm = algorithms.reduce((fastest, current) => {
      const currentTime = current.history.length > 0 ?
        current.history[current.history.length - 1].processing_time : Infinity;
      const fastestTime = fastest.history.length > 0 ?
        fastest.history[fastest.history.length - 1].processing_time : Infinity;
      return currentTime < fastestTime ? current : fastest;
    });

    document.getElementById('fastest-algorithm').textContent = fastestAlgorithm.name;
    document.getElementById('fastest-algorithm-time').textContent =
      `${fastestAlgorithm.history.length > 0 ? fastestAlgorithm.history[fastestAlgorithm.history.length - 1].processing_time : 0}ms`;

    // Most consistent (lowest standard deviation)
    const mostConsistent = algorithms.reduce((consistent, current) =>
      current.statistics.std < consistent.statistics.std ? current : consistent
    );

    document.getElementById('most-consistent').textContent = mostConsistent.name;
    document.getElementById('consistency-score').textContent =
      (1 - mostConsistent.statistics.std).toFixed(3);

    // Most efficient (best performance per processing time)
    const mostEfficient = algorithms.reduce((efficient, current) => {
      const currentEfficiency = current.statistics.mean /
        (current.history.length > 0 ? current.history[current.history.length - 1].processing_time : 1);
      const efficientEfficiency = efficient.statistics.mean /
        (efficient.history.length > 0 ? efficient.history[efficient.history.length - 1].processing_time : 1);
      return currentEfficiency > efficientEfficiency ? current : efficient;
    });

    document.getElementById('efficiency-leader').textContent = mostEfficient.name;
    document.getElementById('efficiency-score').textContent =
      (mostEfficient.statistics.mean /
        (mostEfficient.history.length > 0 ? mostEfficient.history[mostEfficient.history.length - 1].processing_time : 1)
      ).toFixed(3);
  }

  updateRankings() {
    const rankingsContainer = document.getElementById('algorithm-rankings');
    const algorithms = Array.from(this.algorithmData.values());

    // Sort algorithms by current metric
    algorithms.sort((a, b) => b.statistics.mean - a.statistics.mean);

    rankingsContainer.innerHTML = '';

    algorithms.forEach((algorithm, index) => {
      const rankingItem = document.createElement('div');
      rankingItem.className = 'ranking-item';
      rankingItem.innerHTML = `
        <div class="ranking-position">${index + 1}</div>
        <div class="ranking-algorithm">
          <div class="algorithm-name">${algorithm.name}</div>
          <div class="algorithm-score">${algorithm.statistics.mean.toFixed(3)}</div>
        </div>
        <div class="ranking-change">
          <i class="fas fa-arrow-up trend-up"></i>
        </div>
      `;
      rankingsContainer.appendChild(rankingItem);
    });
  }

  updateMetricsDisplay() {
    const metricsContainer = document.getElementById('performance-metrics');
    const algorithms = Array.from(this.algorithmData.values());

    metricsContainer.innerHTML = '';

    algorithms.forEach(algorithm => {
      const metricsItem = document.createElement('div');
      metricsItem.className = 'metrics-item';
      metricsItem.innerHTML = `
        <div class="metrics-header">
          <div class="algorithm-name">${algorithm.name}</div>
          <div class="metrics-toggle">
            <button class="btn-xs btn-secondary" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">
              <i class="fas fa-chevron-down"></i>
            </button>
          </div>
        </div>
        <div class="metrics-details" style="display: none;">
          <div class="metric-row">
            <span class="metric-label">Mean:</span>
            <span class="metric-value">${algorithm.statistics.mean.toFixed(3)}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Std Dev:</span>
            <span class="metric-value">${algorithm.statistics.std.toFixed(3)}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Min:</span>
            <span class="metric-value">${algorithm.statistics.min.toFixed(3)}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Max:</span>
            <span class="metric-value">${algorithm.statistics.max.toFixed(3)}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Count:</span>
            <span class="metric-value">${algorithm.statistics.count}</span>
          </div>
        </div>
      `;
      metricsContainer.appendChild(metricsItem);
    });
  }

  showBenchmarkModal() {
    document.getElementById('benchmark-modal').style.display = 'block';
  }

  showStatisticalTestsModal() {
    document.getElementById('statistical-tests-modal').style.display = 'block';
    this.runAllStatisticalTests();
  }

  runAllStatisticalTests() {
    const algorithms = Array.from(this.algorithmData.values());
    if (algorithms.length < 2) {
      alert('Need at least 2 algorithms to run statistical tests');
      return;
    }

    // Run ANOVA test
    this.runANOVATest(algorithms);

    // Run pairwise t-tests
    this.runPairwiseTTests(algorithms);

    // Run Wilcoxon rank-sum tests
    this.runWilcoxonTests(algorithms);

    // Run Friedman test
    this.runFriedmanTest(algorithms);

    // Run multiple comparisons correction
    this.runMultipleComparisons(algorithms);
  }

  runANOVATest(algorithms) {
    const groups = algorithms.map(alg =>
      alg.history.map(h => h[this.currentMetric]).filter(v => v !== undefined)
    );

    const anovaResult = this.calculateANOVA(groups);

    document.getElementById('anova-results').innerHTML = `
      <div class="test-result">
        <h4>One-way ANOVA</h4>
        <div class="test-stats">
          <div class="stat-item">
            <span class="stat-label">F-statistic:</span>
            <span class="stat-value">${anovaResult.fStatistic.toFixed(4)}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">p-value:</span>
            <span class="stat-value ${anovaResult.pValue < 0.05 ? 'significant' : ''}">${anovaResult.pValue.toFixed(4)}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">df between:</span>
            <span class="stat-value">${anovaResult.dfBetween}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">df within:</span>
            <span class="stat-value">${anovaResult.dfWithin}</span>
          </div>
        </div>
        <div class="test-interpretation">
          <p>${anovaResult.pValue < 0.05 ?
            'There is a statistically significant difference between algorithms (p < 0.05)' :
            'No statistically significant difference between algorithms (p ≥ 0.05)'
          }</p>
        </div>
      </div>
    `;
  }

  runPairwiseTTests(algorithms) {
    const results = [];

    for (let i = 0; i < algorithms.length; i++) {
      for (let j = i + 1; j < algorithms.length; j++) {
        const group1 = algorithms[i].history.map(h => h[this.currentMetric]).filter(v => v !== undefined);
        const group2 = algorithms[j].history.map(h => h[this.currentMetric]).filter(v => v !== undefined);

        const tTestResult = this.calculateTTest(group1, group2);
        results.push({
          algorithm1: algorithms[i].name,
          algorithm2: algorithms[j].name,
          tStatistic: tTestResult.tStatistic,
          pValue: tTestResult.pValue,
          significant: tTestResult.pValue < 0.05
        });
      }
    }

    document.getElementById('ttest-results').innerHTML = `
      <div class="test-result">
        <h4>Pairwise T-Tests</h4>
        <div class="pairwise-results">
          ${results.map(result => `
            <div class="pairwise-test">
              <div class="comparison">${result.algorithm1} vs ${result.algorithm2}</div>
              <div class="test-stats">
                <span>t = ${result.tStatistic.toFixed(4)}</span>
                <span class="p-value ${result.significant ? 'significant' : ''}">p = ${result.pValue.toFixed(4)}</span>
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  runWilcoxonTests(algorithms) {
    // Placeholder for Wilcoxon rank-sum tests
    document.getElementById('wilcoxon-results').innerHTML = `
      <div class="test-result">
        <h4>Wilcoxon Rank-Sum Tests</h4>
        <p>Non-parametric pairwise comparisons would be implemented here.</p>
      </div>
    `;
  }

  runFriedmanTest(algorithms) {
    // Placeholder for Friedman test
    document.getElementById('friedman-results').innerHTML = `
      <div class="test-result">
        <h4>Friedman Test</h4>
        <p>Non-parametric test for multiple related samples would be implemented here.</p>
      </div>
    `;
  }

  runMultipleComparisons(algorithms) {
    // Placeholder for multiple comparisons correction
    document.getElementById('multiple-results').innerHTML = `
      <div class="test-result">
        <h4>Multiple Comparisons Correction</h4>
        <p>Bonferroni, Benjamini-Hochberg, or other corrections would be applied here.</p>
      </div>
    `;
  }

  calculateANOVA(groups) {
    const n = groups.reduce((sum, group) => sum + group.length, 0);
    const k = groups.length;

    // Calculate overall mean
    const overallMean = groups.reduce((sum, group) =>
      sum + group.reduce((groupSum, value) => groupSum + value, 0), 0
    ) / n;

    // Calculate sum of squares between groups
    const ssBetween = groups.reduce((sum, group) => {
      const groupMean = group.reduce((groupSum, value) => groupSum + value, 0) / group.length;
      return sum + group.length * Math.pow(groupMean - overallMean, 2);
    }, 0);

    // Calculate sum of squares within groups
    const ssWithin = groups.reduce((sum, group) => {
      const groupMean = group.reduce((groupSum, value) => groupSum + value, 0) / group.length;
      return sum + group.reduce((groupSum, value) =>
        groupSum + Math.pow(value - groupMean, 2), 0
      );
    }, 0);

    const dfBetween = k - 1;
    const dfWithin = n - k;

    const msBetween = ssBetween / dfBetween;
    const msWithin = ssWithin / dfWithin;

    const fStatistic = msBetween / msWithin;
    const pValue = this.calculateFPValue(fStatistic, dfBetween, dfWithin);

    return {
      fStatistic,
      pValue,
      dfBetween,
      dfWithin,
      ssBetween,
      ssWithin
    };
  }

  calculateTTest(group1, group2) {
    const n1 = group1.length;
    const n2 = group2.length;

    const mean1 = group1.reduce((sum, val) => sum + val, 0) / n1;
    const mean2 = group2.reduce((sum, val) => sum + val, 0) / n2;

    const var1 = group1.reduce((sum, val) => sum + Math.pow(val - mean1, 2), 0) / (n1 - 1);
    const var2 = group2.reduce((sum, val) => sum + Math.pow(val - mean2, 2), 0) / (n2 - 1);

    const pooledSE = Math.sqrt(var1 / n1 + var2 / n2);
    const tStatistic = (mean1 - mean2) / pooledSE;

    const df = n1 + n2 - 2;
    const pValue = this.calculateTPValue(Math.abs(tStatistic), df);

    return {
      tStatistic,
      pValue,
      df,
      mean1,
      mean2
    };
  }

  calculateFPValue(fStatistic, dfBetween, dfWithin) {
    // Simplified p-value calculation (would use proper F-distribution in real implementation)
    return Math.max(0, Math.min(1, 1 - fStatistic / (fStatistic + dfWithin)));
  }

  calculateTPValue(tStatistic, df) {
    // Simplified p-value calculation (would use proper t-distribution in real implementation)
    return Math.max(0, Math.min(1, 2 * (1 - tStatistic / (tStatistic + df))));
  }

  switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.test-results').forEach(tab => {
      tab.classList.remove('active');
    });

    document.querySelectorAll('.tab-button').forEach(button => {
      button.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(`${tabName}-results`).classList.add('active');
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
  }

  startBenchmarkTest() {
    const config = {
      datasets: {
        synthetic: document.getElementById('synthetic-data').checked,
        realWorld: document.getElementById('real-world-data').checked,
        adversarial: document.getElementById('adversarial-data').checked
      },
      metrics: {
        accuracy: document.getElementById('accuracy-metric').checked,
        precision: document.getElementById('precision-metric').checked,
        recall: document.getElementById('recall-metric').checked,
        f1: document.getElementById('f1-metric').checked,
        auc: document.getElementById('auc-metric').checked,
        time: document.getElementById('time-metric').checked
      },
      runs: parseInt(document.getElementById('benchmark-runs').value),
      crossValidation: parseInt(document.getElementById('cross-validation').value)
    };

    document.getElementById('benchmark-modal').style.display = 'none';
    this.runBenchmark(config);
  }

  runBenchmark(config) {
    // Simulate benchmark execution
    const benchmarkContainer = document.getElementById('benchmark-results');
    benchmarkContainer.innerHTML = '<div class="loading">Running benchmark...</div>';

    setTimeout(() => {
      const results = this.generateBenchmarkResults(config);
      this.displayBenchmarkResults(results);
      document.getElementById('last-benchmark').textContent = new Date().toLocaleString();
    }, 3000);
  }

  generateBenchmarkResults(config) {
    // Generate simulated benchmark results
    const algorithms = Array.from(this.algorithmData.keys());
    const results = {};

    algorithms.forEach(algorithm => {
      results[algorithm] = {
        accuracy: Math.random() * 0.3 + 0.7,
        precision: Math.random() * 0.3 + 0.7,
        recall: Math.random() * 0.3 + 0.7,
        f1: Math.random() * 0.3 + 0.7,
        auc: Math.random() * 0.3 + 0.7,
        processing_time: Math.random() * 100 + 10
      };
    });

    return results;
  }

  displayBenchmarkResults(results) {
    const container = document.getElementById('benchmark-results');
    container.innerHTML = '';

    Object.entries(results).forEach(([algorithm, metrics]) => {
      const resultItem = document.createElement('div');
      resultItem.className = 'benchmark-result';
      resultItem.innerHTML = `
        <div class="result-header">${algorithm}</div>
        <div class="result-metrics">
          <div class="metric">Accuracy: ${metrics.accuracy.toFixed(3)}</div>
          <div class="metric">Precision: ${metrics.precision.toFixed(3)}</div>
          <div class="metric">Recall: ${metrics.recall.toFixed(3)}</div>
          <div class="metric">F1: ${metrics.f1.toFixed(3)}</div>
          <div class="metric">AUC: ${metrics.auc.toFixed(3)}</div>
          <div class="metric">Time: ${metrics.processing_time.toFixed(1)}ms</div>
        </div>
      `;
      container.appendChild(resultItem);
    });
  }

  exportPerformanceReport() {
    const report = {
      timestamp: new Date().toISOString(),
      algorithms: Object.fromEntries(this.algorithmData),
      summary: {
        bestAlgorithm: document.getElementById('best-algorithm').textContent,
        fastestAlgorithm: document.getElementById('fastest-algorithm').textContent,
        mostConsistent: document.getElementById('most-consistent').textContent,
        efficiencyLeader: document.getElementById('efficiency-leader').textContent
      },
      statistics: {},
      benchmarkResults: Object.fromEntries(this.benchmarkResults)
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: 'application/json'
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `algorithm-performance-report-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  startPerformanceTracking() {
    if (this.options.enableRealTimeUpdates) {
      setInterval(() => {
        this.updateLastUpdatedTime();
        this.updateFooterStats();
      }, this.options.refreshInterval);
    }
  }

  updateLastUpdatedTime() {
    document.getElementById('last-updated-time').textContent = new Date().toLocaleTimeString();
  }

  updateFooterStats() {
    document.getElementById('total-algorithms').textContent = this.algorithmData.size;
    document.getElementById('total-data-points').textContent =
      Array.from(this.algorithmData.values()).reduce((sum, alg) => sum + alg.history.length, 0);
    document.getElementById('confidence-level').textContent = `${(this.options.confidenceLevel * 100).toFixed(0)}%`;
  }

  toggleErrorBars() {
    // Toggle error bars in comparison chart
    this.performanceComparisonChart.toggleErrorBars();
  }

  toggleConfidenceIntervals() {
    // Toggle confidence intervals in comparison chart
    this.performanceComparisonChart.toggleConfidenceIntervals();
  }

  toggleMovingAverage() {
    // Toggle moving average in trends chart
    this.performanceTrendsChart.toggleMovingAverage();
  }

  toggleTrendLines() {
    // Toggle trend lines in trends chart
    this.performanceTrendsChart.toggleTrendLines();
  }

  updateDistributionChart(type) {
    this.distributionChart.updateType(type);
  }

  destroy() {
    // Clean up resources
    this.algorithmData.clear();
    this.performanceHistory = [];
    this.benchmarkResults.clear();
    this.statisticalTests.clear();
  }
}

// Chart classes (simplified implementations)
class PerformanceComparisonChart {
  constructor(container) {
    this.container = container;
    this.showErrorBars = false;
    this.showConfidenceIntervals = false;
  }

  update(algorithmData, metric, comparisonMode) {
    // Implementation for performance comparison chart
    this.renderChart(algorithmData, metric, comparisonMode);
  }

  renderChart(algorithmData, metric, comparisonMode) {
    // Clear container
    this.container.innerHTML = '';

    // Create simple bar chart for performance comparison
    const algorithms = Array.from(algorithmData.values());
    const chartData = algorithms.map(alg => ({
      name: alg.name,
      value: alg.statistics.mean || 0,
      error: alg.statistics.std || 0
    }));

    // Create SVG chart
    const svg = d3.select(this.container)
      .append('svg')
      .attr('width', this.container.clientWidth)
      .attr('height', 400);

    // Render bars (simplified)
    const margin = { top: 20, right: 30, bottom: 40, left: 60 };
    const width = this.container.clientWidth - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const xScale = d3.scaleBand()
      .domain(chartData.map(d => d.name))
      .range([0, width])
      .padding(0.1);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(chartData, d => d.value + d.error)])
      .range([height, 0]);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Draw bars
    g.selectAll('.bar')
      .data(chartData)
      .enter().append('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.name))
      .attr('y', d => yScale(d.value))
      .attr('width', xScale.bandwidth())
      .attr('height', d => height - yScale(d.value))
      .attr('fill', '#3b82f6');

    // Add axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append('g')
      .call(d3.axisLeft(yScale));
  }

  toggleErrorBars() {
    this.showErrorBars = !this.showErrorBars;
    // Re-render with/without error bars
  }

  toggleConfidenceIntervals() {
    this.showConfidenceIntervals = !this.showConfidenceIntervals;
    // Re-render with/without confidence intervals
  }
}

class PerformanceTrendsChart {
  constructor(container) {
    this.container = container;
    this.showMovingAverage = false;
    this.showTrendLines = false;
  }

  update(algorithmData, metric, timeRange) {
    // Implementation for performance trends chart
    this.renderChart(algorithmData, metric, timeRange);
  }

  renderChart(algorithmData, metric, timeRange) {
    this.container.innerHTML = '<div class="chart-placeholder">Performance Trends Chart</div>';
  }

  toggleMovingAverage() {
    this.showMovingAverage = !this.showMovingAverage;
  }

  toggleTrendLines() {
    this.showTrendLines = !this.showTrendLines;
  }
}

class DistributionChart {
  constructor(container) {
    this.container = container;
    this.type = 'histogram';
  }

  update(algorithmData, metric) {
    // Implementation for distribution chart
    this.renderChart(algorithmData, metric);
  }

  renderChart(algorithmData, metric) {
    this.container.innerHTML = '<div class="chart-placeholder">Distribution Chart</div>';
  }

  updateType(type) {
    this.type = type;
    // Re-render with new type
  }
}

// Export the class
window.AlgorithmPerformanceAnalytics = AlgorithmPerformanceAnalytics;
