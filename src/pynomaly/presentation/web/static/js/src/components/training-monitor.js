/**
 * Training Monitor Component
 * 
 * Real-time monitoring interface for automated training pipelines with:
 * - Live progress tracking with WebSocket updates
 * - Training metrics visualization using D3.js
 * - Resource usage monitoring (CPU, memory)
 * - Training history and comparison
 * - Interactive controls for starting/stopping training
 */

import { WebSocketService } from '../services/websocket-service.js';

export class TrainingMonitor {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' ? document.querySelector(container) : container;
        this.options = {
            enableWebSocket: true,
            enableAutoRefresh: true,
            refreshInterval: 5000,
            showResourceMonitoring: true,
            showTrainingHistory: true,
            maxHistoryItems: 50,
            ...options
        };
        
        // State management
        this.activeTrainings = new Map();
        this.trainingHistory = [];
        this.websocketService = null;
        
        // UI components
        this.components = {
            toolbar: null,
            activeTrainings: null,
            trainingDetails: null,
            metricsChart: null,
            resourceChart: null,
            historyTable: null
        };
        
        // Event listeners
        this.listeners = new Map();
        
        this.init();
    }
    
    init() {
        this.createLayout();
        this.setupWebSocket();
        this.loadInitialData();
        this.bindEvents();
        
        if (this.options.enableAutoRefresh) {
            this.startAutoRefresh();
        }
    }
    
    createLayout() {
        this.container.innerHTML = `
            <div class="training-monitor">
                <!-- Header and Controls -->
                <div class="training-monitor__header">
                    <div class="training-monitor__title">
                        <h2>Automated Training Monitor</h2>
                        <div class="training-monitor__status">
                            <span class="status-indicator" data-status="disconnected">
                                <span class="status-dot"></span>
                                <span class="status-text">Connecting...</span>
                            </span>
                        </div>
                    </div>
                    
                    <div class="training-monitor__toolbar">
                        <button class="btn btn--primary" data-action="start-training">
                            <i class="icon-play"></i> Start Training
                        </button>
                        <button class="btn btn--secondary" data-action="refresh">
                            <i class="icon-refresh"></i> Refresh
                        </button>
                        <button class="btn btn--secondary" data-action="settings">
                            <i class="icon-settings"></i> Settings
                        </button>
                    </div>
                </div>
                
                <!-- Active Trainings Grid -->
                <div class="training-monitor__content">
                    <div class="training-monitor__grid">
                        <!-- Active Trainings Panel -->
                        <div class="training-panel">
                            <div class="panel-header">
                                <h3>Active Trainings</h3>
                                <span class="badge badge--info" data-count="active-count">0</span>
                            </div>
                            <div class="panel-content">
                                <div class="training-list" data-component="active-trainings">
                                    <div class="empty-state">
                                        <i class="icon-training"></i>
                                        <p>No active trainings</p>
                                        <button class="btn btn--primary btn--sm" data-action="start-training">
                                            Start New Training
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Training Details Panel -->
                        <div class="training-panel">
                            <div class="panel-header">
                                <h3>Training Details</h3>
                                <div class="panel-actions">
                                    <button class="btn btn--icon" data-action="expand-details" title="Expand">
                                        <i class="icon-expand"></i>
                                    </button>
                                </div>
                            </div>
                            <div class="panel-content">
                                <div class="training-details" data-component="training-details">
                                    <div class="empty-state">
                                        <p>Select a training to view details</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Metrics Visualization Panel -->
                        <div class="training-panel training-panel--wide">
                            <div class="panel-header">
                                <h3>Training Metrics</h3>
                                <div class="panel-controls">
                                    <select class="form-select form-select--sm" data-control="metric-type">
                                        <option value="score">Score Progress</option>
                                        <option value="loss">Loss Curve</option>
                                        <option value="trials">Trial Results</option>
                                        <option value="resource">Resource Usage</option>
                                    </select>
                                </div>
                            </div>
                            <div class="panel-content">
                                <div class="metrics-chart" data-component="metrics-chart">
                                    <svg class="chart-svg"></svg>
                                    <div class="chart-legend"></div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Training History Panel -->
                        <div class="training-panel training-panel--full">
                            <div class="panel-header">
                                <h3>Training History</h3>
                                <div class="panel-controls">
                                    <input type="search" class="form-input form-input--sm" 
                                           placeholder="Search trainings..." data-control="history-search">
                                    <select class="form-select form-select--sm" data-control="history-filter">
                                        <option value="">All Statuses</option>
                                        <option value="completed">Completed</option>
                                        <option value="failed">Failed</option>
                                        <option value="cancelled">Cancelled</option>
                                    </select>
                                </div>
                            </div>
                            <div class="panel-content">
                                <div class="training-history" data-component="training-history">
                                    <div class="table-container">
                                        <table class="data-table">
                                            <thead>
                                                <tr>
                                                    <th>Training ID</th>
                                                    <th>Detector</th>
                                                    <th>Algorithm</th>
                                                    <th>Score</th>
                                                    <th>Duration</th>
                                                    <th>Status</th>
                                                    <th>Started</th>
                                                    <th>Actions</th>
                                                </tr>
                                            </thead>
                                            <tbody data-target="history-rows">
                                                <!-- Dynamic content -->
                                            </tbody>
                                        </table>
                                    </div>
                                    <div class="table-pagination">
                                        <button class="btn btn--sm" data-action="prev-page" disabled>Previous</button>
                                        <span class="pagination-info">Page 1 of 1</span>
                                        <button class="btn btn--sm" data-action="next-page" disabled>Next</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Start Training Modal -->
                <div class="modal" data-modal="start-training">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h3>Start New Training</h3>
                            <button class="btn btn--icon" data-action="close-modal">
                                <i class="icon-close"></i>
                            </button>
                        </div>
                        <div class="modal-body">
                            <form class="training-form" data-form="start-training">
                                <div class="form-grid">
                                    <div class="form-group">
                                        <label>Detector</label>
                                        <select class="form-select" name="detector_id" required>
                                            <option value="">Select detector...</option>
                                        </select>
                                    </div>
                                    
                                    <div class="form-group">
                                        <label>Dataset</label>
                                        <select class="form-select" name="dataset_id" required>
                                            <option value="">Select dataset...</option>
                                        </select>
                                    </div>
                                    
                                    <div class="form-group">
                                        <label>Experiment Name</label>
                                        <input type="text" class="form-input" name="experiment_name" 
                                               placeholder="Optional experiment name">
                                    </div>
                                    
                                    <div class="form-group">
                                        <label>Optimization Objective</label>
                                        <select class="form-select" name="optimization_objective">
                                            <option value="auc">AUC</option>
                                            <option value="precision">Precision</option>
                                            <option value="recall">Recall</option>
                                            <option value="f1_score">F1 Score</option>
                                        </select>
                                    </div>
                                    
                                    <div class="form-group">
                                        <label>Max Algorithms</label>
                                        <input type="number" class="form-input" name="max_algorithms" 
                                               value="3" min="1" max="10">
                                    </div>
                                    
                                    <div class="form-group">
                                        <label>Max Optimization Time (minutes)</label>
                                        <input type="number" class="form-input" name="max_optimization_time" 
                                               value="60" min="1" max="1440">
                                    </div>
                                </div>
                                
                                <div class="form-group">
                                    <div class="form-checkboxes">
                                        <label class="checkbox-label">
                                            <input type="checkbox" name="enable_automl" checked>
                                            <span class="checkbox-custom"></span>
                                            Enable AutoML optimization
                                        </label>
                                        <label class="checkbox-label">
                                            <input type="checkbox" name="enable_ensemble" checked>
                                            <span class="checkbox-custom"></span>
                                            Enable ensemble creation
                                        </label>
                                        <label class="checkbox-label">
                                            <input type="checkbox" name="enable_early_stopping" checked>
                                            <span class="checkbox-custom"></span>
                                            Enable early stopping
                                        </label>
                                    </div>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn--secondary" data-action="close-modal">Cancel</button>
                            <button class="btn btn--primary" data-action="submit-training">Start Training</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Store component references
        this.components.toolbar = this.container.querySelector('.training-monitor__toolbar');
        this.components.activeTrainings = this.container.querySelector('[data-component="active-trainings"]');
        this.components.trainingDetails = this.container.querySelector('[data-component="training-details"]');
        this.components.metricsChart = this.container.querySelector('[data-component="metrics-chart"]');
        this.components.historyTable = this.container.querySelector('[data-component="training-history"]');
        
        this.setupCharts();
    }
    
    setupCharts() {
        // Initialize D3.js charts for metrics visualization
        this.initializeMetricsChart();
        this.initializeResourceChart();
    }
    
    initializeMetricsChart() {
        const chartContainer = this.components.metricsChart.querySelector('.chart-svg');
        const width = chartContainer.clientWidth || 600;
        const height = 300;
        const margin = { top: 20, right: 30, bottom: 40, left: 50 };
        
        const svg = d3.select(chartContainer)
            .attr('width', width)
            .attr('height', height);
        
        // Clear existing content
        svg.selectAll('*').remove();
        
        // Create chart group
        const chartGroup = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Setup scales
        this.chartScales = {
            x: d3.scaleLinear()
                .range([0, width - margin.left - margin.right]),
            y: d3.scaleLinear()
                .range([height - margin.top - margin.bottom, 0])
        };
        
        // Setup axes
        this.chartAxes = {
            x: d3.axisBottom(this.chartScales.x),
            y: d3.axisLeft(this.chartScales.y)
        };
        
        // Add axes to chart
        chartGroup.append('g')
            .attr('class', 'axis axis--x')
            .attr('transform', `translate(0,${height - margin.top - margin.bottom})`);
        
        chartGroup.append('g')
            .attr('class', 'axis axis--y');
        
        // Add chart title
        chartGroup.append('text')
            .attr('class', 'chart-title')
            .attr('x', (width - margin.left - margin.right) / 2)
            .attr('y', -5)
            .attr('text-anchor', 'middle')
            .text('Training Progress');
        
        this.chartGroup = chartGroup;
    }
    
    initializeResourceChart() {
        // Resource usage chart initialization
        // This would be a smaller chart for CPU/memory monitoring
    }
    
    setupWebSocket() {
        if (!this.options.enableWebSocket) return;
        
        this.websocketService = new WebSocketService({
            url: this.getWebSocketUrl(),
            enableLogging: true
        });
        
        // Setup WebSocket event handlers
        this.websocketService.on('connected', () => {
            this.updateConnectionStatus('connected');
            this.subscribeToTrainingUpdates();
        });
        
        this.websocketService.on('disconnected', () => {
            this.updateConnectionStatus('disconnected');
        });
        
        this.websocketService.on('error', (error) => {
            console.error('Training WebSocket error:', error);
            this.updateConnectionStatus('error');
        });
        
        this.websocketService.on('training_update', (data) => {
            this.handleTrainingUpdate(data);
        });
        
        this.websocketService.on('training_progress', (data) => {
            this.handleTrainingProgress(data.data);
        });
    }
    
    getWebSocketUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/ws/training`;
    }
    
    subscribeToTrainingUpdates() {
        if (this.websocketService) {
            this.websocketService.send({
                type: 'subscribe_training_updates'
            });
        }
    }
    
    updateConnectionStatus(status) {
        const statusIndicator = this.container.querySelector('.status-indicator');
        const statusText = statusIndicator.querySelector('.status-text');
        
        statusIndicator.setAttribute('data-status', status);
        
        switch (status) {
            case 'connected':
                statusText.textContent = 'Connected';
                break;
            case 'disconnected':
                statusText.textContent = 'Disconnected';
                break;
            case 'error':
                statusText.textContent = 'Connection Error';
                break;
            default:
                statusText.textContent = 'Connecting...';
        }
    }
    
    handleTrainingUpdate(data) {
        console.log('Training update received:', data);
        
        if (data.training_id) {
            // Update specific training
            this.updateTrainingItem(data.training_id, data);
        } else {
            // General update, refresh all
            this.refreshActiveTrainings();
        }
    }
    
    handleTrainingProgress(progressData) {
        console.log('Training progress:', progressData);
        
        // Update active training item
        this.updateActiveTraining(progressData);
        
        // Update charts if this training is selected
        const selectedTrainingId = this.getSelectedTrainingId();
        if (selectedTrainingId === progressData.training_id) {
            this.updateTrainingDetails(progressData);
            this.updateMetricsChart(progressData);
        }
    }
    
    updateActiveTraining(progressData) {
        this.activeTrainings.set(progressData.training_id, progressData);
        this.renderActiveTrainings();
    }
    
    renderActiveTrainings() {
        const container = this.components.activeTrainings;
        const trainings = Array.from(this.activeTrainings.values());
        
        if (trainings.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="icon-training"></i>
                    <p>No active trainings</p>
                    <button class="btn btn--primary btn--sm" data-action="start-training">
                        Start New Training
                    </button>
                </div>
            `;
            this.updateActiveCount(0);
            return;
        }
        
        const html = trainings.map(training => this.renderTrainingItem(training)).join('');
        container.innerHTML = html;
        this.updateActiveCount(trainings.length);
    }
    
    renderTrainingItem(training) {
        const statusClass = this.getStatusClass(training.status);
        const progressWidth = Math.round(training.progress_percentage);
        
        return `
            <div class="training-item" data-training-id="${training.training_id}">
                <div class="training-item__header">
                    <div class="training-item__title">
                        <strong>${training.training_id.substring(0, 8)}...</strong>
                        <span class="status-badge status-badge--${statusClass}">${training.status}</span>
                    </div>
                    <div class="training-item__actions">
                        <button class="btn btn--icon btn--sm" data-action="view-training" 
                                data-training-id="${training.training_id}" title="View Details">
                            <i class="icon-eye"></i>
                        </button>
                        ${training.status === 'running' ? `
                            <button class="btn btn--icon btn--sm btn--danger" data-action="cancel-training" 
                                    data-training-id="${training.training_id}" title="Cancel">
                                <i class="icon-stop"></i>
                            </button>
                        ` : ''}
                    </div>
                </div>
                
                <div class="training-item__progress">
                    <div class="progress-bar">
                        <div class="progress-bar__fill" style="width: ${progressWidth}%"></div>
                    </div>
                    <div class="progress-text">
                        <span>${training.current_step}</span>
                        <span>${progressWidth}%</span>
                    </div>
                </div>
                
                <div class="training-item__details">
                    ${training.current_algorithm ? `
                        <div class="detail-item">
                            <span class="detail-label">Algorithm:</span>
                            <span class="detail-value">${training.current_algorithm}</span>
                        </div>
                    ` : ''}
                    ${training.best_score ? `
                        <div class="detail-item">
                            <span class="detail-label">Best Score:</span>
                            <span class="detail-value">${training.best_score.toFixed(4)}</span>
                        </div>
                    ` : ''}
                    ${training.current_message ? `
                        <div class="detail-item detail-item--full">
                            <span class="detail-message">${training.current_message}</span>
                        </div>
                    ` : ''}
                </div>
                
                ${training.warnings && training.warnings.length > 0 ? `
                    <div class="training-item__warnings">
                        ${training.warnings.map(warning => `
                            <div class="warning-item">
                                <i class="icon-warning"></i>
                                <span>${warning}</span>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
            </div>
        `;
    }
    
    getStatusClass(status) {
        const statusMap = {
            'idle': 'secondary',
            'scheduled': 'info',
            'running': 'primary',
            'optimizing': 'warning',
            'evaluating': 'info',
            'completed': 'success',
            'failed': 'danger',
            'cancelled': 'secondary'
        };
        return statusMap[status] || 'secondary';
    }
    
    updateActiveCount(count) {
        const countElement = this.container.querySelector('[data-count="active-count"]');
        if (countElement) {
            countElement.textContent = count;
        }
    }
    
    updateTrainingDetails(trainingData) {
        const container = this.components.trainingDetails;
        
        if (!trainingData) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>Select a training to view details</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = `
            <div class="training-details-content">
                <div class="details-header">
                    <h4>Training ${trainingData.training_id.substring(0, 8)}...</h4>
                    <span class="status-badge status-badge--${this.getStatusClass(trainingData.status)}">
                        ${trainingData.status}
                    </span>
                </div>
                
                <div class="details-grid">
                    <div class="detail-group">
                        <h5>Progress</h5>
                        <div class="detail-item">
                            <span class="detail-label">Current Step:</span>
                            <span class="detail-value">${trainingData.current_step}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Progress:</span>
                            <span class="detail-value">${Math.round(trainingData.progress_percentage)}%</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Started:</span>
                            <span class="detail-value">${new Date(trainingData.start_time).toLocaleString()}</span>
                        </div>
                        ${trainingData.estimated_completion ? `
                            <div class="detail-item">
                                <span class="detail-label">ETA:</span>
                                <span class="detail-value">${new Date(trainingData.estimated_completion).toLocaleString()}</span>
                            </div>
                        ` : ''}
                    </div>
                    
                    ${trainingData.current_algorithm ? `
                        <div class="detail-group">
                            <h5>Algorithm</h5>
                            <div class="detail-item">
                                <span class="detail-label">Current:</span>
                                <span class="detail-value">${trainingData.current_algorithm}</span>
                            </div>
                            ${trainingData.current_trial && trainingData.total_trials ? `
                                <div class="detail-item">
                                    <span class="detail-label">Trial:</span>
                                    <span class="detail-value">${trainingData.current_trial} / ${trainingData.total_trials}</span>
                                </div>
                            ` : ''}
                            ${trainingData.best_score ? `
                                <div class="detail-item">
                                    <span class="detail-label">Best Score:</span>
                                    <span class="detail-value">${trainingData.best_score.toFixed(4)}</span>
                                </div>
                            ` : ''}
                            ${trainingData.current_score ? `
                                <div class="detail-item">
                                    <span class="detail-label">Current Score:</span>
                                    <span class="detail-value">${trainingData.current_score.toFixed(4)}</span>
                                </div>
                            ` : ''}
                        </div>
                    ` : ''}
                    
                    ${this.options.showResourceMonitoring ? `
                        <div class="detail-group">
                            <h5>Resources</h5>
                            ${trainingData.memory_usage_mb ? `
                                <div class="detail-item">
                                    <span class="detail-label">Memory:</span>
                                    <span class="detail-value">${Math.round(trainingData.memory_usage_mb)} MB</span>
                                </div>
                            ` : ''}
                            ${trainingData.cpu_usage_percent ? `
                                <div class="detail-item">
                                    <span class="detail-label">CPU:</span>
                                    <span class="detail-value">${Math.round(trainingData.cpu_usage_percent)}%</span>
                                </div>
                            ` : ''}
                        </div>
                    ` : ''}
                </div>
                
                ${trainingData.current_message ? `
                    <div class="detail-message">
                        <h5>Status Message</h5>
                        <p>${trainingData.current_message}</p>
                    </div>
                ` : ''}
            </div>
        `;
    }
    
    updateMetricsChart(trainingData) {
        if (!this.chartGroup || !trainingData.best_score) return;
        
        // For now, just update with current score
        // In a real implementation, we'd track score history over time
        const data = [{
            trial: trainingData.current_trial || 1,
            score: trainingData.current_score || trainingData.best_score
        }];
        
        // Update scales
        this.chartScales.x.domain([0, trainingData.total_trials || 100]);
        this.chartScales.y.domain([0, 1]);
        
        // Update axes
        this.chartGroup.select('.axis--x')
            .call(this.chartAxes.x);
        
        this.chartGroup.select('.axis--y')
            .call(this.chartAxes.y);
        
        // Update data points
        const points = this.chartGroup.selectAll('.data-point')
            .data(data);
        
        points.enter()
            .append('circle')
            .attr('class', 'data-point')
            .attr('r', 4)
            .attr('fill', '#3b82f6')
            .merge(points)
            .attr('cx', d => this.chartScales.x(d.trial))
            .attr('cy', d => this.chartScales.y(d.score));
        
        points.exit().remove();
    }
    
    bindEvents() {
        // Toolbar actions
        this.container.addEventListener('click', (event) => {
            const action = event.target.closest('[data-action]')?.dataset.action;
            
            switch (action) {
                case 'start-training':
                    this.showStartTrainingModal();
                    break;
                case 'refresh':
                    this.refreshAll();
                    break;
                case 'settings':
                    this.showSettings();
                    break;
                case 'view-training':
                    const trainingId = event.target.closest('[data-training-id]')?.dataset.trainingId;
                    this.selectTraining(trainingId);
                    break;
                case 'cancel-training':
                    const cancelId = event.target.closest('[data-training-id]')?.dataset.trainingId;
                    this.cancelTraining(cancelId);
                    break;
                case 'close-modal':
                    this.closeModal();
                    break;
                case 'submit-training':
                    this.submitTraining();
                    break;
            }
        });
        
        // Form submissions
        this.container.addEventListener('submit', (event) => {
            event.preventDefault();
            
            if (event.target.matches('[data-form="start-training"]')) {
                this.submitTraining();
            }
        });
    }
    
    async loadInitialData() {
        try {
            // Load active trainings
            await this.refreshActiveTrainings();
            
            // Load training history
            await this.refreshTrainingHistory();
            
            // Load dropdown options
            await this.loadFormOptions();
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showError('Failed to load training data');
        }
    }
    
    async refreshActiveTrainings() {
        try {
            const response = await fetch('/api/training/active');
            const trainings = await response.json();
            
            this.activeTrainings.clear();
            trainings.forEach(training => {
                this.activeTrainings.set(training.training_id, training);
            });
            
            this.renderActiveTrainings();
            
        } catch (error) {
            console.error('Failed to refresh active trainings:', error);
        }
    }
    
    async refreshTrainingHistory() {
        try {
            const response = await fetch('/api/training/history');
            const data = await response.json();
            
            this.trainingHistory = data.trainings || [];
            this.renderTrainingHistory();
            
        } catch (error) {
            console.error('Failed to refresh training history:', error);
        }
    }
    
    renderTrainingHistory() {
        const tbody = this.container.querySelector('[data-target="history-rows"]');
        
        if (this.trainingHistory.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="8" class="empty-cell">No training history available</td>
                </tr>
            `;
            return;
        }
        
        tbody.innerHTML = this.trainingHistory.map(training => `
            <tr>
                <td><code>${training.training_id.substring(0, 8)}...</code></td>
                <td>${training.detector_id.substring(0, 8)}...</td>
                <td>${training.best_algorithm || 'N/A'}</td>
                <td>${training.best_score ? training.best_score.toFixed(4) : 'N/A'}</td>
                <td>${training.training_time_seconds ? this.formatDuration(training.training_time_seconds) : 'N/A'}</td>
                <td><span class="status-badge status-badge--${this.getStatusClass(training.status)}">${training.status}</span></td>
                <td>${training.start_time ? new Date(training.start_time).toLocaleDateString() : 'N/A'}</td>
                <td>
                    <button class="btn btn--icon btn--sm" data-action="view-result" 
                            data-training-id="${training.training_id}" title="View Details">
                        <i class="icon-eye"></i>
                    </button>
                </td>
            </tr>
        `).join('');
    }
    
    formatDuration(seconds) {
        if (seconds < 60) return `${Math.round(seconds)}s`;
        if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
        return `${Math.round(seconds / 3600)}h`;
    }
    
    async loadFormOptions() {
        try {
            // Load detectors
            const detectorsResponse = await fetch('/api/detectors');
            const detectors = await detectorsResponse.json();
            
            const detectorSelect = this.container.querySelector('select[name="detector_id"]');
            detectorSelect.innerHTML = '<option value="">Select detector...</option>' +
                detectors.map(d => `<option value="${d.id}">${d.name}</option>`).join('');
            
            // Load datasets
            const datasetsResponse = await fetch('/api/datasets');
            const datasets = await datasetsResponse.json();
            
            const datasetSelect = this.container.querySelector('select[name="dataset_id"]');
            datasetSelect.innerHTML = '<option value="">Select dataset...</option>' +
                datasets.map(d => `<option value="${d.id}">${d.name}</option>`).join('');
            
        } catch (error) {
            console.error('Failed to load form options:', error);
        }
    }
    
    showStartTrainingModal() {
        const modal = this.container.querySelector('[data-modal="start-training"]');
        modal.classList.add('modal--active');
    }
    
    closeModal() {
        const modals = this.container.querySelectorAll('.modal');
        modals.forEach(modal => modal.classList.remove('modal--active'));
    }
    
    async submitTraining() {
        const form = this.container.querySelector('[data-form="start-training"]');
        const formData = new FormData(form);
        
        const requestData = {
            detector_id: formData.get('detector_id'),
            dataset_id: formData.get('dataset_id'),
            experiment_name: formData.get('experiment_name') || null,
            optimization_objective: formData.get('optimization_objective'),
            max_algorithms: parseInt(formData.get('max_algorithms')),
            max_optimization_time: parseInt(formData.get('max_optimization_time')) * 60, // Convert to seconds
            enable_automl: formData.has('enable_automl'),
            enable_ensemble: formData.has('enable_ensemble'),
            enable_early_stopping: formData.has('enable_early_stopping')
        };
        
        try {
            const response = await fetch('/api/training/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            this.closeModal();
            this.showSuccess(`Training started: ${result.training_id}`);
            
            // Refresh active trainings
            await this.refreshActiveTrainings();
            
        } catch (error) {
            console.error('Failed to start training:', error);
            this.showError(`Failed to start training: ${error.message}`);
        }
    }
    
    async cancelTraining(trainingId) {
        if (!confirm('Are you sure you want to cancel this training?')) return;
        
        try {
            const response = await fetch(`/api/training/cancel/${trainingId}`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            this.showSuccess('Training cancelled successfully');
            await this.refreshActiveTrainings();
            
        } catch (error) {
            console.error('Failed to cancel training:', error);
            this.showError(`Failed to cancel training: ${error.message}`);
        }
    }
    
    selectTraining(trainingId) {
        const training = this.activeTrainings.get(trainingId);
        if (training) {
            this.updateTrainingDetails(training);
            this.updateMetricsChart(training);
            
            // Update UI selection
            this.container.querySelectorAll('.training-item').forEach(item => {
                item.classList.toggle('training-item--selected', 
                    item.dataset.trainingId === trainingId);
            });
        }
    }
    
    getSelectedTrainingId() {
        const selected = this.container.querySelector('.training-item--selected');
        return selected?.dataset.trainingId || null;
    }
    
    startAutoRefresh() {
        setInterval(() => {
            if (document.visibilityState === 'visible') {
                this.refreshActiveTrainings();
            }
        }, this.options.refreshInterval);
    }
    
    async refreshAll() {
        await Promise.all([
            this.refreshActiveTrainings(),
            this.refreshTrainingHistory()
        ]);
    }
    
    showSuccess(message) {
        // Implementation for success notifications
        console.log('Success:', message);
    }
    
    showError(message) {
        // Implementation for error notifications
        console.error('Error:', message);
    }
    
    showSettings() {
        // Implementation for settings modal
        console.log('Show settings');
    }
    
    destroy() {
        if (this.websocketService) {
            this.websocketService.disconnect();
        }
        
        this.listeners.clear();
        this.activeTrainings.clear();
    }
}

export default TrainingMonitor;