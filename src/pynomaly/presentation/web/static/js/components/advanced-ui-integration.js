/**
 * Advanced UI Integration Component
 * 
 * Integrates all advanced web UI features including:
 * - Interactive anomaly investigation
 * - Enhanced offline capabilities
 * - Real-time WebSocket updates
 * - Advanced D3.js visualizations
 * - PWA functionality
 */

import { InteractiveAnomalyInvestigation } from './interactive-anomaly-investigation.js';
import { EnhancedOfflineManager } from './enhanced-offline-manager.js';
import { RealTimeDashboard } from './real-time-dashboard.js';

export class AdvancedUIIntegration {
    constructor(options = {}) {
        this.options = {
            enableInvestigation: true,
            enableOfflineMode: true,
            enableRealTime: true,
            enablePWA: true,
            enableAdvancedCharts: true,
            enableVoiceCommands: true,
            enableAccessibility: true,
            ...options
        };

        this.components = new Map();
        this.eventBus = new EventTarget();
        this.state = {
            isOnline: navigator.onLine,
            isInvestigating: false,
            realTimeActive: false,
            offlineCapable: false
        };

        this.init();
    }

    async init() {
        console.log('Initializing Advanced UI Integration...');
        
        try {
            // Initialize core components
            await this.initializeOfflineManager();
            await this.initializeInvestigationSystem();
            await this.initializeRealTimeDashboard();
            await this.initializePWAFeatures();
            
            // Setup integrations
            this.setupComponentIntegrations();
            this.setupGlobalEventHandlers();
            this.setupKeyboardShortcuts();
            
            // Initialize UI enhancements
            this.initializeUIEnhancements();
            
            console.log('Advanced UI Integration initialized successfully');
            this.emit('initialized', { components: this.components.size });
            
        } catch (error) {
            console.error('Failed to initialize Advanced UI Integration:', error);
            this.emit('initialization-failed', error);
        }
    }

    async initializeOfflineManager() {
        if (!this.options.enableOfflineMode) return;
        
        const offlineManager = new EnhancedOfflineManager({
            enableOfflineDetection: true,
            enableDataCaching: true,
            enableBackgroundSync: true,
            enableOfflineAnalytics: true
        });
        
        this.components.set('offlineManager', offlineManager);
        
        // Setup offline event handlers
        offlineManager.on('online', () => {
            this.state.isOnline = true;
            this.showNotification('Connection restored', 'success');
            this.emit('connectivity-changed', { online: true });
        });
        
        offlineManager.on('offline', () => {
            this.state.isOnline = false;
            this.showNotification('Working offline', 'info');
            this.emit('connectivity-changed', { online: false });
        });
        
        offlineManager.on('sync-complete', (data) => {
            this.showNotification(`Synced ${data.processed} items`, 'success');
        });
    }

    async initializeInvestigationSystem() {
        if (!this.options.enableInvestigation) return;
        
        const investigationContainer = document.createElement('div');
        investigationContainer.id = 'anomaly-investigation-container';
        investigationContainer.style.display = 'none';
        investigationContainer.className = 'fixed inset-0 z-50 bg-white';
        document.body.appendChild(investigationContainer);
        
        const investigation = new InteractiveAnomalyInvestigation(investigationContainer, {
            enableDrillDown: true,
            enableFeatureImportance: true,
            enableCorrelationAnalysis: true,
            enableTimelineZoom: true
        });
        
        this.components.set('investigation', investigation);
        
        // Setup investigation event handlers
        investigation.on('investigation-started', (data) => {
            this.state.isInvestigating = true;
            investigationContainer.style.display = 'block';
            this.emit('investigation-started', data);
        });
        
        investigation.on('investigation-closed', () => {
            this.state.isInvestigating = false;
            investigationContainer.style.display = 'none';
            this.emit('investigation-closed');
        });
    }

    async initializeRealTimeDashboard() {
        if (!this.options.enableRealTime) return;
        
        const dashboardContainer = document.getElementById('real-time-dashboard');
        if (!dashboardContainer) return;
        
        const realTimeDashboard = new RealTimeDashboard(dashboardContainer, {
            enableRealTime: true,
            enableAlerts: true,
            enableSound: false,
            enableNotifications: true
        });
        
        this.components.set('realTimeDashboard', realTimeDashboard);
        
        // Setup real-time event handlers
        realTimeDashboard.on('anomaly-detected', (anomaly) => {
            this.handleAnomalyDetected(anomaly);
        });
        
        realTimeDashboard.on('connected', () => {
            this.state.realTimeActive = true;
            this.showNotification('Real-time monitoring active', 'success');
        });
        
        realTimeDashboard.on('disconnected', () => {
            this.state.realTimeActive = false;
            this.showNotification('Real-time monitoring disconnected', 'warning');
        });
    }

    async initializePWAFeatures() {
        if (!this.options.enablePWA) return;
        
        try {
            // Initialize enhanced service worker
            if ('serviceWorker' in navigator) {
                const registration = await navigator.serviceWorker.register('/sw-enhanced.js');
                console.log('Enhanced service worker registered');
                
                // Setup service worker message handling
                navigator.serviceWorker.addEventListener('message', (event) => {
                    this.handleServiceWorkerMessage(event);
                });
            }
            
            // Setup install prompt
            this.setupInstallPrompt();
            
            // Setup push notifications
            await this.setupPushNotifications();
            
            this.state.offlineCapable = true;
            
        } catch (error) {
            console.error('Failed to initialize PWA features:', error);
        }
    }

    setupComponentIntegrations() {
        // Integrate investigation with real-time dashboard
        const realTimeDashboard = this.components.get('realTimeDashboard');
        const investigation = this.components.get('investigation');
        
        if (realTimeDashboard && investigation) {
            // Listen for anomaly selection in dashboard
            realTimeDashboard.on('anomaly-selected', (data) => {
                investigation.investigateAnomaly(data.anomaly, {
                    source: 'real-time-dashboard',
                    context: data.context
                });
            });
        }
        
        // Integrate offline manager with all components
        const offlineManager = this.components.get('offlineManager');
        if (offlineManager) {
            // Setup offline detection integration
            this.setupOfflineDetectionIntegration(offlineManager);
        }
    }

    setupOfflineDetectionIntegration(offlineManager) {
        // Override global fetch for anomaly detection
        const originalFetch = window.fetch;
        
        window.fetch = async (...args) => {
            const [url, options] = args;
            
            // Check if this is an anomaly detection request
            if (url.includes('/api/detect') || url.includes('/api/analyze')) {
                if (!navigator.onLine) {
                    // Use offline detection
                    try {
                        const requestData = JSON.parse(options.body);
                        const results = await offlineManager.detectAnomaliesOffline(requestData.data);
                        
                        return new Response(JSON.stringify({
                            results,
                            offline: true,
                            timestamp: Date.now()
                        }), {
                            status: 200,
                            headers: { 'Content-Type': 'application/json' }
                        });
                    } catch (error) {
                        console.error('Offline detection failed:', error);
                        throw error;
                    }
                }
            }
            
            // Use original fetch for other requests
            return originalFetch.apply(window, args);
        };
    }

    setupGlobalEventHandlers() {
        // Handle anomaly detection results
        document.addEventListener('anomaly-detected', (event) => {
            this.handleAnomalyDetected(event.detail);
        });
        
        // Handle chart interactions
        document.addEventListener('chart-point-selected', (event) => {
            this.handleChartPointSelected(event.detail);
        });
        
        // Handle WebSocket events
        document.addEventListener('websocket-message', (event) => {
            this.handleWebSocketMessage(event.detail);
        });
        
        // Handle offline/online events
        window.addEventListener('online', () => {
            this.handleConnectivityChange(true);
        });
        
        window.addEventListener('offline', () => {
            this.handleConnectivityChange(false);
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (event) => {
            // Ctrl/Cmd + I: Open investigation for selected anomaly
            if ((event.ctrlKey || event.metaKey) && event.key === 'i') {
                event.preventDefault();
                this.openInvestigationForSelected();
            }
            
            // Ctrl/Cmd + R: Refresh real-time dashboard
            if ((event.ctrlKey || event.metaKey) && event.key === 'r') {
                event.preventDefault();
                this.refreshRealTimeDashboard();
            }
            
            // Ctrl/Cmd + O: Toggle offline mode
            if ((event.ctrlKey || event.metaKey) && event.key === 'o') {
                event.preventDefault();
                this.toggleOfflineMode();
            }
            
            // Escape: Close investigation or modals
            if (event.key === 'Escape') {
                this.handleEscapeKey();
            }
        });
    }

    initializeUIEnhancements() {
        // Add status indicators
        this.createStatusIndicators();
        
        // Add advanced controls
        this.createAdvancedControls();
        
        // Setup drag and drop for datasets
        this.setupDragAndDrop();
        
        // Initialize tooltips and help system
        this.initializeHelpSystem();
    }

    createStatusIndicators() {
        const statusContainer = document.createElement('div');
        statusContainer.className = 'fixed top-4 right-4 z-40 flex flex-col gap-2';
        statusContainer.id = 'advanced-ui-status';
        
        statusContainer.innerHTML = `
            <div class="status-indicator" id="connectivity-status">
                <span class="indicator online" title="Connection Status"></span>
                <span class="text">Online</span>
            </div>
            <div class="status-indicator" id="offline-capability-status" style="display: none;">
                <span class="indicator offline-ready" title="Offline Capability"></span>
                <span class="text">Offline Ready</span>
            </div>
            <div class="status-indicator" id="real-time-status" style="display: none;">
                <span class="indicator realtime" title="Real-time Status"></span>
                <span class="text">Live</span>
            </div>
        `;
        
        document.body.appendChild(statusContainer);
        
        // Update status indicators
        this.updateStatusIndicators();
    }

    createAdvancedControls() {
        const controlsContainer = document.createElement('div');
        controlsContainer.className = 'fixed bottom-4 right-4 z-40';
        controlsContainer.id = 'advanced-ui-controls';
        
        controlsContainer.innerHTML = `
            <div class="controls-panel bg-white shadow-lg rounded-lg p-4 border">
                <h3 class="text-sm font-semibold mb-3">Advanced Controls</h3>
                <div class="flex flex-col gap-2">
                    <button id="toggle-investigation" class="btn btn-sm btn-outline" disabled>
                        🔍 Investigation
                    </button>
                    <button id="toggle-offline-mode" class="btn btn-sm btn-outline">
                        📱 Offline Mode
                    </button>
                    <button id="export-data" class="btn btn-sm btn-outline">
                        📤 Export Data
                    </button>
                    <button id="performance-stats" class="btn btn-sm btn-outline">
                        📊 Performance
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(controlsContainer);
        
        // Bind control events
        this.bindAdvancedControls();
    }

    bindAdvancedControls() {
        document.getElementById('toggle-investigation').addEventListener('click', () => {
            this.toggleInvestigation();
        });
        
        document.getElementById('toggle-offline-mode').addEventListener('click', () => {
            this.toggleOfflineMode();
        });
        
        document.getElementById('export-data').addEventListener('click', () => {
            this.exportAllData();
        });
        
        document.getElementById('performance-stats').addEventListener('click', () => {
            this.showPerformanceStats();
        });
    }

    setupDragAndDrop() {
        // Setup drag and drop for dataset uploads
        document.addEventListener('dragover', (event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect = 'copy';
        });
        
        document.addEventListener('drop', async (event) => {
            event.preventDefault();
            
            const files = Array.from(event.dataTransfer.files);
            const dataFiles = files.filter(file => 
                file.type === 'application/json' || 
                file.type === 'text/csv' ||
                file.name.endsWith('.json') ||
                file.name.endsWith('.csv')
            );
            
            if (dataFiles.length > 0) {
                await this.handleFileUpload(dataFiles);
            }
        });
    }

    async handleFileUpload(files) {
        const offlineManager = this.components.get('offlineManager');
        
        for (const file of files) {
            try {
                const data = await this.readFile(file);
                const dataset = {
                    id: `uploaded-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                    name: file.name,
                    data: data,
                    uploadedAt: new Date().toISOString(),
                    source: 'drag-drop'
                };
                
                if (offlineManager) {
                    await offlineManager.storeDataset(dataset);
                    this.showNotification(`Dataset "${file.name}" uploaded successfully`, 'success');
                } else {
                    // Upload to server
                    await this.uploadDataset(dataset);
                }
                
            } catch (error) {
                console.error('File upload failed:', error);
                this.showNotification(`Failed to upload "${file.name}"`, 'error');
            }
        }
    }

    readFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    const data = file.type === 'application/json' || file.name.endsWith('.json')
                        ? JSON.parse(event.target.result)
                        : this.parseCSV(event.target.result);
                    resolve(data);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => reject(reader.error);
            reader.readAsText(file);
        });
    }

    // Event handlers
    handleAnomalyDetected(anomaly) {
        console.log('Anomaly detected:', anomaly);
        
        // Show notification
        this.showNotification(
            `Anomaly detected (score: ${anomaly.score.toFixed(3)})`,
            anomaly.score > 0.8 ? 'error' : 'warning'
        );
        
        // Enable investigation button
        const investigationBtn = document.getElementById('toggle-investigation');
        if (investigationBtn) {
            investigationBtn.disabled = false;
            investigationBtn.setAttribute('data-anomaly-id', anomaly.id);
        }
        
        // Store selected anomaly for investigation
        this.selectedAnomaly = anomaly;
        
        // Emit event for other components
        this.emit('anomaly-detected', anomaly);
    }

    handleChartPointSelected(data) {
        console.log('Chart point selected:', data);
        
        if (data.anomaly) {
            this.selectedAnomaly = data.anomaly;
            
            const investigationBtn = document.getElementById('toggle-investigation');
            if (investigationBtn) {
                investigationBtn.disabled = false;
            }
        }
    }

    handleConnectivityChange(isOnline) {
        this.state.isOnline = isOnline;
        this.updateStatusIndicators();
        
        if (isOnline) {
            // Trigger sync when coming back online
            const offlineManager = this.components.get('offlineManager');
            if (offlineManager) {
                offlineManager.syncPendingData();
            }
        }
    }

    handleServiceWorkerMessage(event) {
        const { type, payload } = event.data;
        
        switch (type) {
            case 'CACHE_UPDATED':
                this.showNotification('App updated in background', 'info');
                break;
            case 'OFFLINE_READY':
                this.state.offlineCapable = true;
                this.updateStatusIndicators();
                break;
            case 'SYNC_COMPLETE':
                this.showNotification('Data synchronized', 'success');
                break;
        }
    }

    // Control methods
    toggleInvestigation() {
        if (!this.selectedAnomaly) {
            this.showNotification('No anomaly selected for investigation', 'warning');
            return;
        }
        
        const investigation = this.components.get('investigation');
        if (investigation) {
            if (this.state.isInvestigating) {
                investigation.close();
            } else {
                investigation.investigateAnomaly(this.selectedAnomaly);
            }
        }
    }

    toggleOfflineMode() {
        const offlineManager = this.components.get('offlineManager');
        if (offlineManager) {
            const status = offlineManager.getOfflineStatus();
            if (status.capabilities.offlineDetection) {
                this.showNotification('Offline mode already enabled', 'info');
            } else {
                this.showNotification('Enabling offline capabilities...', 'info');
                // Additional offline setup would go here
            }
        }
    }

    async exportAllData() {
        const offlineManager = this.components.get('offlineManager');
        if (offlineManager) {
            await offlineManager.exportOfflineData();
            this.showNotification('Data exported successfully', 'success');
        }
    }

    showPerformanceStats() {
        const stats = this.gatherPerformanceStats();
        
        const modal = document.createElement('div');
        modal.className = 'modal show';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Performance Statistics</h2>
                    <button class="modal-close">×</button>
                </div>
                <div class="modal-body">
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-label">Components Loaded</div>
                            <div class="stat-value">${this.components.size}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Memory Usage</div>
                            <div class="stat-value">${stats.memoryUsage}MB</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Cache Size</div>
                            <div class="stat-value">${stats.cacheSize}MB</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Offline Capable</div>
                            <div class="stat-value">${this.state.offlineCapable ? 'Yes' : 'No'}</div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-outline modal-close">Close</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        modal.querySelector('.modal-close').addEventListener('click', () => {
            document.body.removeChild(modal);
        });
    }

    gatherPerformanceStats() {
        const stats = {
            memoryUsage: 0,
            cacheSize: 0
        };
        
        if ('memory' in performance) {
            stats.memoryUsage = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
        }
        
        // Additional performance metrics would be gathered here
        
        return stats;
    }

    // Utility methods
    updateStatusIndicators() {
        const connectivityStatus = document.getElementById('connectivity-status');
        const offlineStatus = document.getElementById('offline-capability-status');
        const realTimeStatus = document.getElementById('real-time-status');
        
        if (connectivityStatus) {
            const indicator = connectivityStatus.querySelector('.indicator');
            const text = connectivityStatus.querySelector('.text');
            
            if (this.state.isOnline) {
                indicator.className = 'indicator online';
                text.textContent = 'Online';
            } else {
                indicator.className = 'indicator offline';
                text.textContent = 'Offline';
            }
        }
        
        if (offlineStatus) {
            offlineStatus.style.display = this.state.offlineCapable ? 'flex' : 'none';
        }
        
        if (realTimeStatus) {
            realTimeStatus.style.display = this.state.realTimeActive ? 'flex' : 'none';
        }
    }

    showNotification(message, type = 'info') {
        // Create or update notification system
        let notificationContainer = document.getElementById('notification-container');
        if (!notificationContainer) {
            notificationContainer = document.createElement('div');
            notificationContainer.id = 'notification-container';
            notificationContainer.className = 'fixed top-4 left-4 z-50 flex flex-col gap-2';
            document.body.appendChild(notificationContainer);
        }
        
        const notification = document.createElement('div');
        notification.className = `notification notification-${type} bg-white shadow-lg rounded-lg p-4 border-l-4 max-w-sm`;
        notification.innerHTML = `
            <div class="flex items-center justify-between">
                <span class="text-sm">${message}</span>
                <button class="ml-2 text-gray-400 hover:text-gray-600">×</button>
            </div>
        `;
        
        notificationContainer.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
        
        // Manual close
        notification.querySelector('button').addEventListener('click', () => {
            notification.parentNode.removeChild(notification);
        });
    }

    // Event emitter methods
    emit(event, data) {
        this.eventBus.dispatchEvent(new CustomEvent(event, { detail: data }));
    }

    on(event, callback) {
        this.eventBus.addEventListener(event, callback);
    }

    off(event, callback) {
        this.eventBus.removeEventListener(event, callback);
    }

    // Cleanup
    destroy() {
        // Destroy all components
        this.components.forEach(component => {
            if (component.destroy) {
                component.destroy();
            }
        });
        
        this.components.clear();
        
        // Remove UI elements
        const statusContainer = document.getElementById('advanced-ui-status');
        if (statusContainer) statusContainer.remove();
        
        const controlsContainer = document.getElementById('advanced-ui-controls');
        if (controlsContainer) controlsContainer.remove();
        
        const notificationContainer = document.getElementById('notification-container');
        if (notificationContainer) notificationContainer.remove();
    }
}

// Global initialization
window.addEventListener('DOMContentLoaded', () => {
    if (!window.advancedUIIntegration) {
        window.advancedUIIntegration = new AdvancedUIIntegration();
    }
});

export default AdvancedUIIntegration;