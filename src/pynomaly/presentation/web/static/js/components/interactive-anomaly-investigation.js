/**
 * Interactive Anomaly Investigation Component
 * 
 * Advanced drill-down analysis capabilities for anomaly detection results
 * with feature importance, correlation analysis, and detailed investigation tools
 */

import { TimeSeriesChart, ScatterPlotChart, HeatmapChart } from './d3-chart-library.js';

export class InteractiveAnomalyInvestigation {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' 
            ? document.querySelector(container) 
            : container;
        
        this.options = {
            enableDrillDown: true,
            enableFeatureImportance: true,
            enableCorrelationAnalysis: true,
            enableTimelineZoom: true,
            enableExportFeatures: true,
            maxDataPoints: 10000,
            defaultTimeRange: '24h',
            anomalyThreshold: 0.5,
            ...options
        };

        this.currentAnomaly = null;
        this.investigationData = null;
        this.charts = new Map();
        this.panels = new Map();
        
        this.init();
    }

    init() {
        this.setupContainer();
        this.createInvestigationInterface();
        this.bindEvents();
    }

    setupContainer() {
        this.container.classList.add('anomaly-investigation');
        this.container.innerHTML = '';

        // Create main layout
        this.container.innerHTML = `
            <div class="investigation-header">
                <div class="investigation-title">
                    <h2>Anomaly Investigation</h2>
                    <div class="investigation-breadcrumb" id="breadcrumb">
                        <span class="breadcrumb-item active">Overview</span>
                    </div>
                </div>
                <div class="investigation-controls">
                    <button class="btn btn-secondary" id="export-investigation">Export Analysis</button>
                    <button class="btn btn-secondary" id="share-investigation">Share</button>
                    <button class="btn btn-ghost" id="close-investigation">×</button>
                </div>
            </div>
            <div class="investigation-content">
                <div class="investigation-sidebar">
                    <div class="sidebar-section">
                        <h3>Anomaly Details</h3>
                        <div id="anomaly-summary"></div>
                    </div>
                    <div class="sidebar-section">
                        <h3>Investigation Tools</h3>
                        <div id="investigation-tools"></div>
                    </div>
                    <div class="sidebar-section">
                        <h3>Related Anomalies</h3>
                        <div id="related-anomalies"></div>
                    </div>
                </div>
                <div class="investigation-main">
                    <div class="investigation-tabs">
                        <button class="tab-button active" data-tab="overview">Overview</button>
                        <button class="tab-button" data-tab="features">Feature Analysis</button>
                        <button class="tab-button" data-tab="timeline">Timeline Analysis</button>
                        <button class="tab-button" data-tab="correlation">Correlation Matrix</button>
                        <button class="tab-button" data-tab="similar">Similar Anomalies</button>
                    </div>
                    <div class="investigation-panels">
                        <div class="panel active" id="panel-overview"></div>
                        <div class="panel" id="panel-features"></div>
                        <div class="panel" id="panel-timeline"></div>
                        <div class="panel" id="panel-correlation"></div>
                        <div class="panel" id="panel-similar"></div>
                    </div>
                </div>
            </div>
        `;
    }

    createInvestigationInterface() {
        this.createOverviewPanel();
        this.createFeatureAnalysisPanel();
        this.createTimelineAnalysisPanel();
        this.createCorrelationPanel();
        this.createSimilarAnomaliesPanel();
        this.createInvestigationTools();
    }

    createOverviewPanel() {
        const panel = document.getElementById('panel-overview');
        panel.innerHTML = `
            <div class="overview-grid">
                <div class="overview-card">
                    <h3>Anomaly Score Distribution</h3>
                    <div id="score-distribution-chart" class="chart-container"></div>
                </div>
                <div class="overview-card">
                    <h3>Feature Impact</h3>
                    <div id="feature-impact-chart" class="chart-container"></div>
                </div>
                <div class="overview-card">
                    <h3>Temporal Context</h3>
                    <div id="temporal-context-chart" class="chart-container"></div>
                </div>
                <div class="overview-card">
                    <h3>Key Insights</h3>
                    <div id="key-insights" class="insights-container"></div>
                </div>
            </div>
        `;
    }

    createFeatureAnalysisPanel() {
        const panel = document.getElementById('panel-features');
        panel.innerHTML = `
            <div class="features-layout">
                <div class="features-controls">
                    <div class="control-group">
                        <label>Feature Selection:</label>
                        <select id="feature-selector" multiple>
                            <option value="all">All Features</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Analysis Type:</label>
                        <select id="analysis-type">
                            <option value="importance">Feature Importance</option>
                            <option value="distribution">Distribution Analysis</option>
                            <option value="outliers">Outlier Detection</option>
                            <option value="shap">SHAP Values</option>
                        </select>
                    </div>
                    <button class="btn btn-primary" id="analyze-features">Analyze</button>
                </div>
                <div class="features-content">
                    <div class="features-chart">
                        <div id="feature-analysis-chart" class="chart-container large"></div>
                    </div>
                    <div class="features-details">
                        <h3>Feature Details</h3>
                        <div id="feature-details-table"></div>
                    </div>
                </div>
            </div>
        `;
    }

    createTimelineAnalysisPanel() {
        const panel = document.getElementById('panel-timeline');
        panel.innerHTML = `
            <div class="timeline-layout">
                <div class="timeline-controls">
                    <div class="control-group">
                        <label>Time Range:</label>
                        <select id="time-range">
                            <option value="1h">Last Hour</option>
                            <option value="6h">Last 6 Hours</option>
                            <option value="24h" selected>Last 24 Hours</option>
                            <option value="7d">Last 7 Days</option>
                            <option value="30d">Last 30 Days</option>
                            <option value="custom">Custom Range</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Granularity:</label>
                        <select id="time-granularity">
                            <option value="minute">Minute</option>
                            <option value="hour" selected>Hour</option>
                            <option value="day">Day</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Show:</label>
                        <div class="checkbox-group">
                            <label><input type="checkbox" id="show-anomalies" checked> Anomalies</label>
                            <label><input type="checkbox" id="show-normal" checked> Normal Data</label>
                            <label><input type="checkbox" id="show-predictions"> Predictions</label>
                            <label><input type="checkbox" id="show-confidence"> Confidence Bands</label>
                        </div>
                    </div>
                </div>
                <div class="timeline-content">
                    <div id="timeline-chart" class="chart-container large"></div>
                    <div class="timeline-insights">
                        <h3>Timeline Insights</h3>
                        <div id="timeline-insights-content"></div>
                    </div>
                </div>
            </div>
        `;
    }

    createCorrelationPanel() {
        const panel = document.getElementById('panel-correlation');
        panel.innerHTML = `
            <div class="correlation-layout">
                <div class="correlation-controls">
                    <div class="control-group">
                        <label>Correlation Method:</label>
                        <select id="correlation-method">
                            <option value="pearson">Pearson</option>
                            <option value="spearman">Spearman</option>
                            <option value="kendall">Kendall</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Threshold:</label>
                        <input type="range" id="correlation-threshold" min="0" max="1" step="0.1" value="0.5">
                        <span id="threshold-value">0.5</span>
                    </div>
                    <div class="control-group">
                        <label>Data Subset:</label>
                        <select id="data-subset">
                            <option value="anomalies">Anomalies Only</option>
                            <option value="normal">Normal Data Only</option>
                            <option value="all" selected>All Data</option>
                            <option value="around-anomaly">Around Anomaly</option>
                        </select>
                    </div>
                </div>
                <div class="correlation-content">
                    <div id="correlation-heatmap" class="chart-container large"></div>
                    <div class="correlation-insights">
                        <h3>Strong Correlations</h3>
                        <div id="correlation-insights-content"></div>
                    </div>
                </div>
            </div>
        `;
    }

    createSimilarAnomaliesPanel() {
        const panel = document.getElementById('panel-similar');
        panel.innerHTML = `
            <div class="similar-layout">
                <div class="similar-controls">
                    <div class="control-group">
                        <label>Similarity Method:</label>
                        <select id="similarity-method">
                            <option value="euclidean">Euclidean Distance</option>
                            <option value="cosine">Cosine Similarity</option>
                            <option value="manhattan">Manhattan Distance</option>
                            <option value="feature-based">Feature-based</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Number of Results:</label>
                        <input type="number" id="similarity-count" value="10" min="5" max="50">
                    </div>
                    <div class="control-group">
                        <label>Time Window:</label>
                        <select id="similarity-timeframe">
                            <option value="24h">Last 24 Hours</option>
                            <option value="7d" selected>Last 7 Days</option>
                            <option value="30d">Last 30 Days</option>
                            <option value="all">All Time</option>
                        </select>
                    </div>
                    <button class="btn btn-primary" id="find-similar">Find Similar</button>
                </div>
                <div class="similar-content">
                    <div class="similar-results">
                        <div id="similar-anomalies-list"></div>
                    </div>
                    <div class="similar-visualization">
                        <div id="similarity-scatter" class="chart-container"></div>
                    </div>
                </div>
            </div>
        `;
    }

    createInvestigationTools() {
        const toolsContainer = document.getElementById('investigation-tools');
        toolsContainer.innerHTML = `
            <div class="tool-buttons">
                <button class="tool-btn" id="deep-dive" title="Deep Dive Analysis">
                    🔍 Deep Dive
                </button>
                <button class="tool-btn" id="compare-normal" title="Compare with Normal Data">
                    ⚖️ Compare
                </button>
                <button class="tool-btn" id="prediction-explain" title="Explain Prediction">
                    🧠 Explain
                </button>
                <button class="tool-btn" id="create-rule" title="Create Detection Rule">
                    📝 Create Rule
                </button>
                <button class="tool-btn" id="mark-false-positive" title="Mark as False Positive">
                    ❌ False Positive
                </button>
                <button class="tool-btn" id="escalate" title="Escalate to Expert">
                    🚨 Escalate
                </button>
            </div>
        `;
    }

    bindEvents() {
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Investigation tools
        document.getElementById('deep-dive').addEventListener('click', () => this.performDeepDive());
        document.getElementById('compare-normal').addEventListener('click', () => this.compareWithNormal());
        document.getElementById('prediction-explain').addEventListener('click', () => this.explainPrediction());
        document.getElementById('create-rule').addEventListener('click', () => this.createDetectionRule());
        document.getElementById('mark-false-positive').addEventListener('click', () => this.markFalsePositive());
        document.getElementById('escalate').addEventListener('click', () => this.escalateToExpert());

        // Feature analysis
        document.getElementById('analyze-features').addEventListener('click', () => this.analyzeFeatures());

        // Timeline controls
        document.getElementById('time-range').addEventListener('change', () => this.updateTimeline());
        document.getElementById('time-granularity').addEventListener('change', () => this.updateTimeline());

        // Correlation controls
        document.getElementById('correlation-threshold').addEventListener('input', (e) => {
            document.getElementById('threshold-value').textContent = e.target.value;
            this.updateCorrelationMatrix();
        });

        // Similar anomalies
        document.getElementById('find-similar').addEventListener('click', () => this.findSimilarAnomalies());

        // Export and sharing
        document.getElementById('export-investigation').addEventListener('click', () => this.exportInvestigation());
        document.getElementById('share-investigation').addEventListener('click', () => this.shareInvestigation());
        document.getElementById('close-investigation').addEventListener('click', () => this.close());
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update panels
        document.querySelectorAll('.panel').forEach(panel => {
            panel.classList.remove('active');
        });
        document.getElementById(`panel-${tabName}`).classList.add('active');

        // Load tab content if not already loaded
        this.loadTabContent(tabName);
    }

    loadTabContent(tabName) {
        switch (tabName) {
            case 'overview':
                this.loadOverviewContent();
                break;
            case 'features':
                this.loadFeatureAnalysisContent();
                break;
            case 'timeline':
                this.loadTimelineContent();
                break;
            case 'correlation':
                this.loadCorrelationContent();
                break;
            case 'similar':
                this.loadSimilarAnomaliesContent();
                break;
        }
    }

    // Investigation methods
    async investigateAnomaly(anomaly, context = {}) {
        this.currentAnomaly = anomaly;
        this.investigationData = context;
        
        // Update anomaly summary
        this.updateAnomalySummary();
        
        // Load initial content
        this.loadOverviewContent();
        
        // Fetch additional investigation data
        await this.fetchInvestigationData();
        
        // Show investigation interface
        this.container.style.display = 'block';
        
        // Emit investigation started event
        this.emit('investigation-started', { anomaly, context });
    }

    async fetchInvestigationData() {
        try {
            // Fetch related data for investigation
            const response = await fetch('/api/investigation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    anomalyId: this.currentAnomaly.id,
                    timeRange: this.options.defaultTimeRange,
                    includeFeatures: true,
                    includeContext: true
                })
            });

            this.investigationData = await response.json();
            
            // Update all panels with new data
            this.updateAllPanels();
            
        } catch (error) {
            console.error('Failed to fetch investigation data:', error);
            this.showError('Failed to load investigation data');
        }
    }

    updateAnomalySummary() {
        const summaryContainer = document.getElementById('anomaly-summary');
        const anomaly = this.currentAnomaly;
        
        summaryContainer.innerHTML = `
            <div class="summary-grid">
                <div class="summary-item">
                    <span class="label">Score:</span>
                    <span class="value score-${this.getSeverityClass(anomaly.score)}">${anomaly.score.toFixed(3)}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Timestamp:</span>
                    <span class="value">${new Date(anomaly.timestamp).toLocaleString()}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Severity:</span>
                    <span class="value severity-${anomaly.severity}">${anomaly.severity}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Model:</span>
                    <span class="value">${anomaly.modelId || 'Unknown'}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Dataset:</span>
                    <span class="value">${anomaly.datasetId || 'Unknown'}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Features:</span>
                    <span class="value">${anomaly.features?.length || 0}</span>
                </div>
            </div>
        `;
    }

    loadOverviewContent() {
        this.createScoreDistributionChart();
        this.createFeatureImpactChart();
        this.createTemporalContextChart();
        this.generateKeyInsights();
    }

    createScoreDistributionChart() {
        const container = document.getElementById('score-distribution-chart');
        // Implementation for score distribution visualization
    }

    createFeatureImpactChart() {
        const container = document.getElementById('feature-impact-chart');
        // Implementation for feature impact visualization
    }

    createTemporalContextChart() {
        const container = document.getElementById('temporal-context-chart');
        // Implementation for temporal context visualization
    }

    generateKeyInsights() {
        const container = document.getElementById('key-insights');
        const insights = this.analyzeAnomalyInsights();
        
        container.innerHTML = insights.map(insight => `
            <div class="insight-item ${insight.type}">
                <div class="insight-icon">${insight.icon}</div>
                <div class="insight-content">
                    <div class="insight-title">${insight.title}</div>
                    <div class="insight-description">${insight.description}</div>
                </div>
            </div>
        `).join('');
    }

    analyzeAnomalyInsights() {
        const insights = [];
        const anomaly = this.currentAnomaly;
        
        // Score-based insights
        if (anomaly.score > 0.9) {
            insights.push({
                type: 'critical',
                icon: '🚨',
                title: 'Extremely High Anomaly Score',
                description: 'This anomaly has an exceptionally high confidence score, indicating a significant deviation from normal patterns.'
            });
        }
        
        // Feature-based insights
        if (anomaly.features && anomaly.features.length > 0) {
            const topFeature = anomaly.features.sort((a, b) => b.importance - a.importance)[0];
            insights.push({
                type: 'info',
                icon: '📊',
                title: `Primary Feature: ${topFeature.name}`,
                description: `Feature "${topFeature.name}" shows the strongest deviation with ${(topFeature.importance * 100).toFixed(1)}% contribution to the anomaly score.`
            });
        }
        
        // Temporal insights
        const hour = new Date(anomaly.timestamp).getHours();
        if (hour < 6 || hour > 22) {
            insights.push({
                type: 'warning',
                icon: '🌙',
                title: 'Off-Hours Detection',
                description: 'This anomaly occurred during off-peak hours, which may indicate unusual activity patterns.'
            });
        }
        
        return insights;
    }

    async performDeepDive() {
        // Implementation for deep dive analysis
        console.log('Performing deep dive analysis...');
    }

    async compareWithNormal() {
        // Implementation for comparing with normal data
        console.log('Comparing with normal data...');
    }

    async explainPrediction() {
        // Implementation for prediction explanation
        console.log('Explaining prediction...');
    }

    async createDetectionRule() {
        // Implementation for creating detection rules
        console.log('Creating detection rule...');
    }

    async markFalsePositive() {
        // Implementation for marking false positives
        console.log('Marking as false positive...');
    }

    async escalateToExpert() {
        // Implementation for escalation
        console.log('Escalating to expert...');
    }

    getSeverityClass(score) {
        if (score >= 0.9) return 'critical';
        if (score >= 0.7) return 'high';
        if (score >= 0.5) return 'medium';
        return 'low';
    }

    exportInvestigation() {
        const data = {
            anomaly: this.currentAnomaly,
            investigation: this.investigationData,
            timestamp: new Date().toISOString(),
            insights: this.analyzeAnomalyInsights()
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `anomaly-investigation-${this.currentAnomaly.id}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    shareInvestigation() {
        // Implementation for sharing investigation
        navigator.share({
            title: 'Anomaly Investigation Report',
            text: `Investigation report for anomaly with score ${this.currentAnomaly.score.toFixed(3)}`,
            url: window.location.href
        }).catch(() => {
            // Fallback to clipboard
            navigator.clipboard.writeText(window.location.href);
            alert('Investigation link copied to clipboard');
        });
    }

    close() {
        this.container.style.display = 'none';
        this.emit('investigation-closed');
    }

    // Event emitter methods
    on(event, callback) {
        if (!this.listeners) this.listeners = new Map();
        if (!this.listeners.has(event)) this.listeners.set(event, new Set());
        this.listeners.get(event).add(callback);
    }

    emit(event, data) {
        if (!this.listeners) return;
        const callbacks = this.listeners.get(event);
        if (callbacks) {
            callbacks.forEach(callback => callback(data));
        }
    }

    destroy() {
        // Clean up charts and components
        this.charts.forEach(chart => chart.destroy && chart.destroy());
        this.charts.clear();
        
        // Clear event listeners
        if (this.listeners) this.listeners.clear();
        
        // Clear data
        this.currentAnomaly = null;
        this.investigationData = null;
    }
}

export default InteractiveAnomalyInvestigation;