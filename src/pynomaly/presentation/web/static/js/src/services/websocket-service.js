/**
 * WebSocket Service for Real-Time Data
 * Production-ready WebSocket client for live anomaly detection updates
 * with connection management, error handling, and message routing
 */

/**
 * WebSocket Connection States
 */
const CONNECTION_STATES = {
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  DISCONNECTED: 'disconnected',
  RECONNECTING: 'reconnecting',
  ERROR: 'error'
};

/**
 * Message Types for WebSocket Communication
 */
const MESSAGE_TYPES = {
  // System messages
  PING: 'ping',
  PONG: 'pong',
  CONNECT: 'connect',
  DISCONNECT: 'disconnect',
  ERROR: 'error',
  
  // Data messages
  ANOMALY_DETECTED: 'anomaly_detected',
  PERFORMANCE_UPDATE: 'performance_update',
  SYSTEM_ALERT: 'system_alert',
  DATA_UPDATE: 'data_update',
  BATCH_UPDATE: 'batch_update',
  
  // Control messages
  SUBSCRIBE: 'subscribe',
  UNSUBSCRIBE: 'unsubscribe',
  REQUEST_HISTORY: 'request_history',
  
  // Authentication
  AUTHENTICATE: 'authenticate',
  AUTH_SUCCESS: 'auth_success',
  AUTH_FAILED: 'auth_failed'
};

export class WebSocketService {
    constructor(options = {}) {
        this.options = {
            url: options.url || this.getWebSocketUrl(),
            protocols: options.protocols || ['anomaly-detection-v1'],
            maxReconnectAttempts: options.maxReconnectAttempts || 10,
            reconnectInterval: options.reconnectInterval || 3000,
            maxReconnectDelay: options.maxReconnectDelay || 30000,
            heartbeatInterval: options.heartbeatInterval || 30000,
            messageQueueSize: options.messageQueueSize || 1000,
            enableMessageQueue: options.enableMessageQueue !== false,
            enableCompression: options.enableCompression !== false,
            enableLogging: options.enableLogging || false,
            autoConnect: options.autoConnect !== false,
            authentication: options.authentication || null,
            ...options
        };
        
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.listeners = new Map();
        this.subscriptions = new Set();
        this.heartbeatTimer = null;
        this.reconnectTimer = null;
        this.messageQueue = [];
        this.connectionId = null;
        
        this.bindMethods();
        
        if (this.options.autoConnect) {
            this.connect();
        }
    }
    
    bindMethods() {
        this.handleOpen = this.handleOpen.bind(this);
        this.handleMessage = this.handleMessage.bind(this);
        this.handleError = this.handleError.bind(this);
        this.handleClose = this.handleClose.bind(this);
        this.sendHeartbeat = this.sendHeartbeat.bind(this);
    }
    
    getWebSocketUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/ws/anomaly-detection`;
    }
    
    connect() {
        if (this.isConnected || (this.ws && this.ws.readyState === WebSocket.CONNECTING)) {
            return Promise.resolve();
        }
        
        return new Promise((resolve, reject) => {
            try {
                this.log('Connecting to WebSocket...', this.options.url);
                
                this.ws = new WebSocket(this.options.url, this.options.protocols);
                
                // Configure WebSocket properties
                if (this.options.enableCompression && this.ws.extensions) {
                    this.ws.extensions = 'permessage-deflate';
                }
                
                const connectTimeout = setTimeout(() => {
                    if (this.ws.readyState === WebSocket.CONNECTING) {
                        this.ws.close();
                        reject(new Error('Connection timeout'));
                    }
                }, 10000);
                
                this.ws.addEventListener('open', (event) => {
                    clearTimeout(connectTimeout);
                    this.handleOpen(event);
                    resolve();
                });
                
                this.ws.addEventListener('message', this.handleMessage);
                this.ws.addEventListener('error', (error) => {
                    clearTimeout(connectTimeout);
                    this.handleError(error);
                    reject(error);
                });
                this.ws.addEventListener('close', this.handleClose);
                
            } catch (error) {
                this.log('Connection error:', error);
                reject(error);
            }
        });
    }
    
    handleOpen(event) {
        this.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        
        // Start heartbeat
        this.startHeartbeat();
        
        // Send queued messages
        this.processMessageQueue();
        
        // Re-subscribe to previous subscriptions
        this.resubscribe();
        
        // Emit connected event
        this.emit('connected', { event, connectionId: this.connectionId });
    }
    
    handleMessage(event) {
        try {
            const message = JSON.parse(event.data);
            this.log('Received message:', message);
            
            // Handle system messages
            if (message.type === 'system') {
                this.handleSystemMessage(message);
                return;
            }
            
            // Handle heartbeat responses
            if (message.type === 'pong') {
                this.log('Heartbeat acknowledged');
                return;
            }
            
            // Emit message to listeners
            this.emit('message', message);
            
            // Emit specific message type
            if (message.type) {
                this.emit(message.type, message.data || message);
            }
            
            // Handle subscription messages
            if (message.subscription) {
                this.emit(`subscription:${message.subscription}`, message.data || message);
            }
            
        } catch (error) {
            this.log('Error parsing message:', error, event.data);
            this.emit('error', { type: 'parse_error', error, rawData: event.data });
        }
    }
    
    handleSystemMessage(message) {
        switch (message.action) {
            case 'connection_established':
                this.connectionId = message.connectionId;
                this.log('Connection ID received:', this.connectionId);
                break;
                
            case 'subscription_confirmed':
                this.log('Subscription confirmed:', message.subscription);
                this.emit('subscription_confirmed', message);
                break;
                
            case 'subscription_error':
                this.log('Subscription error:', message.error);
                this.emit('subscription_error', message);
                break;
                
            case 'rate_limit_exceeded':
                this.log('Rate limit exceeded');
                this.emit('rate_limit_exceeded', message);
                break;
                
            case 'server_shutdown':
                this.log('Server shutdown notification');
                this.emit('server_shutdown', message);
                break;
                
            default:
                this.log('Unknown system message:', message);
        }
    }
    
    handleError(error) {
        this.log('WebSocket error:', error);
        this.emit('error', { type: 'connection_error', error });
    }
    
    handleClose(event) {
        this.log('WebSocket closed:', event.code, event.reason);
        this.isConnected = false;
        this.stopHeartbeat();
        
        this.emit('disconnected', { 
            code: event.code, 
            reason: event.reason,
            wasClean: event.wasClean 
        });
        
        // Attempt reconnection if not a clean close
        if (!event.wasClean && this.reconnectAttempts < this.options.maxReconnectAttempts) {
            this.scheduleReconnect();
        }
    }
    
    scheduleReconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
        }
        
        this.reconnectAttempts++;
        const delay = Math.min(
            this.options.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1),
            30000
        );
        
        this.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
        
        this.reconnectTimer = setTimeout(() => {
            this.log(`Reconnect attempt ${this.reconnectAttempts}`);
            this.connect().catch(error => {
                this.log('Reconnect failed:', error);
                if (this.reconnectAttempts < this.options.maxReconnectAttempts) {
                    this.scheduleReconnect();
                } else {
                    this.emit('max_reconnect_attempts_reached');
                }
            });
        }, delay);
    }
    
    startHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
        }
        
        this.heartbeatTimer = setInterval(this.sendHeartbeat, this.options.heartbeatInterval);
    }
    
    stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }
    
    sendHeartbeat() {
        if (this.isConnected) {
            this.send({
                type: 'ping',
                timestamp: Date.now()
            });
        }
    }
    
    send(data) {
        if (!this.isConnected) {
            this.log('Queueing message (not connected):', data);
            this.messageQueue.push(data);
            return false;
        }
        
        try {
            const message = typeof data === 'string' ? data : JSON.stringify(data);
            this.ws.send(message);
            this.log('Sent message:', data);
            return true;
        } catch (error) {
            this.log('Error sending message:', error);
            this.emit('error', { type: 'send_error', error, data });
            return false;
        }
    }
    
    processMessageQueue() {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            this.send(message);
        }
    }
    
    // Subscription management
    subscribe(subscription, params = {}) {
        const subscriptionData = {
            type: 'subscribe',
            subscription,
            params,
            timestamp: Date.now()
        };
        
        this.subscriptions.add(subscription);
        this.send(subscriptionData);
        
        this.log('Subscribed to:', subscription, params);
        return () => this.unsubscribe(subscription);
    }
    
    unsubscribe(subscription) {
        this.subscriptions.delete(subscription);
        this.send({
            type: 'unsubscribe',
            subscription,
            timestamp: Date.now()
        });
        
        this.log('Unsubscribed from:', subscription);
    }
    
    resubscribe() {
        this.subscriptions.forEach(subscription => {
            this.send({
                type: 'subscribe',
                subscription,
                timestamp: Date.now()
            });
        });
    }
    
    // Event management
    on(event, listener) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(listener);
        
        return () => this.off(event, listener);
    }
    
    off(event, listener) {
        const eventListeners = this.listeners.get(event);
        if (eventListeners) {
            eventListeners.delete(listener);
            if (eventListeners.size === 0) {
                this.listeners.delete(event);
            }
        }
    }
    
    emit(event, data) {
        const eventListeners = this.listeners.get(event);
        if (eventListeners) {
            eventListeners.forEach(listener => {
                try {
                    listener(data);
                } catch (error) {
                    this.log('Error in event listener:', error);
                }
            });
        }
    }
    
    // Utility methods
    isConnected() {
        return this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN;
    }
    
    getConnectionState() {
        if (!this.ws) return 'disconnected';
        
        switch (this.ws.readyState) {
            case WebSocket.CONNECTING: return 'connecting';
            case WebSocket.OPEN: return 'connected';
            case WebSocket.CLOSING: return 'closing';
            case WebSocket.CLOSED: return 'disconnected';
            default: return 'unknown';
        }
    }
    
    getConnectionInfo() {
        return {
            connectionId: this.connectionId,
            state: this.getConnectionState(),
            reconnectAttempts: this.reconnectAttempts,
            subscriptions: Array.from(this.subscriptions),
            queuedMessages: this.messageQueue.length,
            url: this.options.url
        };
    }
    
    disconnect() {
        this.log('Disconnecting WebSocket');
        
        this.stopHeartbeat();
        
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        if (this.ws) {
            this.ws.close(1000, 'Client disconnect');
        }
        
        this.isConnected = false;
        this.subscriptions.clear();
        this.messageQueue = [];
    }
    
    log(...args) {
        if (this.options.enableLogging) {
            console.log('[WebSocketService]', ...args);
        }
    }
    
    destroy() {
        this.disconnect();
        this.listeners.clear();
        
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }
    }
}

/**
 * Anomaly Detection WebSocket Client
 * 
 * High-level client for anomaly detection specific operations
 */
export class AnomalyWebSocketClient {
    constructor(options = {}) {
        this.options = {
            enableRealTimeDetection: true,
            enableAlerts: true,
            enableMetrics: true,
            enableModelUpdates: true,
            bufferSize: 1000,
            ...options
        };
        
        this.wsService = new WebSocketService(options);
        this.dataBuffer = [];
        this.alertsBuffer = [];
        this.callbacks = new Map();
        
        this.init();
    }
    
    init() {
        // Set up event handlers
        this.wsService.on('connected', () => {
            this.setupSubscriptions();
        });
        
        this.wsService.on('anomaly_detected', (data) => {
            this.handleAnomalyDetected(data);
        });
        
        this.wsService.on('real_time_data', (data) => {
            this.handleRealTimeData(data);
        });
        
        this.wsService.on('alert', (data) => {
            this.handleAlert(data);
        });
        
        this.wsService.on('model_update', (data) => {
            this.handleModelUpdate(data);
        });
        
        this.wsService.on('system_metrics', (data) => {
            this.handleSystemMetrics(data);
        });
    }
    
    setupSubscriptions() {
        if (this.options.enableRealTimeDetection) {
            this.wsService.subscribe('anomaly_detection');
            this.wsService.subscribe('real_time_data');
        }
        
        if (this.options.enableAlerts) {
            this.wsService.subscribe('alerts');
        }
        
        if (this.options.enableMetrics) {
            this.wsService.subscribe('system_metrics');
        }
        
        if (this.options.enableModelUpdates) {
            this.wsService.subscribe('model_updates');
        }
    }
    
    handleAnomalyDetected(data) {
        const anomaly = {
            id: data.id || Date.now(),
            timestamp: data.timestamp || new Date().toISOString(),
            score: data.score,
            severity: data.severity || this.calculateSeverity(data.score),
            features: data.features || [],
            metadata: data.metadata || {},
            datasetId: data.datasetId,
            modelId: data.modelId
        };
        
        this.emit('anomaly_detected', anomaly);
        
        // Auto-alert for high severity anomalies
        if (anomaly.severity === 'critical' || anomaly.severity === 'high') {
            this.handleAlert({
                type: 'anomaly',
                severity: anomaly.severity,
                message: `${anomaly.severity.toUpperCase()} anomaly detected with score ${anomaly.score.toFixed(3)}`,
                anomaly
            });
        }
    }
    
    handleRealTimeData(data) {
        this.dataBuffer.push({
            ...data,
            receivedAt: Date.now()
        });
        
        // Maintain buffer size
        if (this.dataBuffer.length > this.options.bufferSize) {
            this.dataBuffer.shift();
        }
        
        this.emit('real_time_data', data);
    }
    
    handleAlert(alert) {
        const enrichedAlert = {
            id: alert.id || Date.now(),
            timestamp: alert.timestamp || new Date().toISOString(),
            type: alert.type,
            severity: alert.severity || 'info',
            message: alert.message,
            data: alert.data || alert.anomaly,
            acknowledged: false,
            receivedAt: Date.now()
        };
        
        this.alertsBuffer.unshift(enrichedAlert);
        
        // Maintain alerts buffer
        if (this.alertsBuffer.length > this.options.bufferSize) {
            this.alertsBuffer.pop();
        }
        
        this.emit('alert', enrichedAlert);
    }
    
    handleModelUpdate(update) {
        this.emit('model_update', {
            modelId: update.modelId,
            version: update.version,
            performance: update.performance,
            status: update.status,
            timestamp: update.timestamp || new Date().toISOString()
        });
    }
    
    handleSystemMetrics(metrics) {
        this.emit('system_metrics', {
            ...metrics,
            receivedAt: Date.now()
        });
    }
    
    calculateSeverity(score) {
        if (score >= 0.9) return 'critical';
        if (score >= 0.7) return 'high';
        if (score >= 0.5) return 'medium';
        return 'low';
    }
    
    // Public API methods
    startRealTimeDetection(datasetId, config = {}) {
        return this.wsService.send({
            type: 'start_detection',
            datasetId,
            config: {
                algorithm: config.algorithm || 'isolation_forest',
                threshold: config.threshold || 0.5,
                windowSize: config.windowSize || 100,
                ...config
            }
        });
    }
    
    stopRealTimeDetection(datasetId) {
        return this.wsService.send({
            type: 'stop_detection',
            datasetId
        });
    }
    
    sendDataPoint(dataPoint) {
        return this.wsService.send({
            type: 'data_point',
            data: dataPoint,
            timestamp: Date.now()
        });
    }
    
    sendBatchData(dataPoints) {
        return this.wsService.send({
            type: 'batch_data',
            data: dataPoints,
            count: dataPoints.length,
            timestamp: Date.now()
        });
    }
    
    acknowledgeAlert(alertId) {
        const alert = this.alertsBuffer.find(a => a.id === alertId);
        if (alert) {
            alert.acknowledged = true;
            alert.acknowledgedAt = Date.now();
        }
        
        return this.wsService.send({
            type: 'acknowledge_alert',
            alertId,
            timestamp: Date.now()
        });
    }
    
    requestModelRetraining(modelId, config = {}) {
        return this.wsService.send({
            type: 'request_retraining',
            modelId,
            config,
            timestamp: Date.now()
        });
    }
    
    // Data access methods
    getRecentData(count = 100) {
        return this.dataBuffer.slice(-count);
    }
    
    getRecentAlerts(count = 50) {
        return this.alertsBuffer.slice(0, count);
    }
    
    getUnacknowledgedAlerts() {
        return this.alertsBuffer.filter(alert => !alert.acknowledged);
    }
    
    // Event management
    on(event, callback) {
        return this.wsService.on(event, callback);
    }
    
    off(event, callback) {
        return this.wsService.off(event, callback);
    }
    
    emit(event, data) {
        return this.wsService.emit(event, data);
    }
    
    // Connection management
    connect() {
        return this.wsService.connect();
    }
    
    disconnect() {
        return this.wsService.disconnect();
    }
    
    isConnected() {
        return this.wsService.isConnected();
    }
    
    getConnectionInfo() {
        return {
            ...this.wsService.getConnectionInfo(),
            buffers: {
                data: this.dataBuffer.length,
                alerts: this.alertsBuffer.length,
                unacknowledgedAlerts: this.getUnacknowledgedAlerts().length
            }
        };
    }
    
    destroy() {
        this.wsService.destroy();
        this.dataBuffer = [];
        this.alertsBuffer = [];
        this.callbacks.clear();
    }
}

// Export default
export default AnomalyWebSocketClient;