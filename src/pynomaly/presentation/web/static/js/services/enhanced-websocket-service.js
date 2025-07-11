/**
 * Enhanced WebSocket Service for Real-time Dashboard Features
 * Supports real-time data streaming, collaboration, and advanced notifications
 */

class EnhancedWebSocketService {
  constructor(options = {}) {
    this.options = {
      url: this.getWebSocketURL(),
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      enableCompression: true,
      enableBinarySupport: true,
      messageQueueSize: 1000,
      ...options
    };

    this.ws = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.messageQueue = [];
    this.subscriptions = new Map();
    this.heartbeatTimer = null;
    this.reconnectTimer = null;
    this.connectionListeners = new Set();
    this.messageHandlers = new Map();
    this.collaborationState = new Map();

    this.init();
  }

  getWebSocketURL() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/ws/dashboard`;
  }

  init() {
    this.connect();
    this.setupMessageHandlers();
    this.setupHeartbeat();
    this.setupVisibilityHandling();
  }

  connect() {
    try {
      console.log('[WebSocket] Connecting to:', this.options.url);

      this.ws = new WebSocket(this.options.url);

      // Enable compression if supported
      if (this.options.enableCompression) {
        this.ws.binaryType = 'arraybuffer';
      }

      this.ws.onopen = (event) => this.handleOpen(event);
      this.ws.onmessage = (event) => this.handleMessage(event);
      this.ws.onclose = (event) => this.handleClose(event);
      this.ws.onerror = (event) => this.handleError(event);

    } catch (error) {
      console.error('[WebSocket] Connection failed:', error);
      this.scheduleReconnect();
    }
  }

  handleOpen(event) {
    console.log('[WebSocket] Connection established');
    this.isConnected = true;
    this.reconnectAttempts = 0;

    // Send authentication and initialization
    this.authenticate();

    // Process queued messages
    this.processMessageQueue();

    // Start heartbeat
    this.startHeartbeat();

    // Notify listeners
    this.notifyConnectionListeners('connected', event);

    // Subscribe to default channels
    this.subscribeToDefaultChannels();
  }

  handleMessage(event) {
    try {
      let message;

      // Handle binary messages
      if (event.data instanceof ArrayBuffer) {
        message = this.decodeBinaryMessage(event.data);
      } else {
        message = JSON.parse(event.data);
      }

      console.log('[WebSocket] Message received:', message.type);
      this.routeMessage(message);

    } catch (error) {
      console.error('[WebSocket] Message parsing failed:', error);
    }
  }

  handleClose(event) {
    console.log('[WebSocket] Connection closed:', event.code, event.reason);
    this.isConnected = false;
    this.stopHeartbeat();

    this.notifyConnectionListeners('disconnected', event);

    // Schedule reconnection if not intentionally closed
    if (event.code !== 1000) {
      this.scheduleReconnect();
    }
  }

  handleError(event) {
    console.error('[WebSocket] Error:', event);
    this.notifyConnectionListeners('error', event);
  }

  authenticate() {
    const authMessage = {
      type: 'authenticate',
      timestamp: Date.now(),
      clientId: this.generateClientId(),
      userAgent: navigator.userAgent,
      capabilities: {
        compression: this.options.enableCompression,
        binarySupport: this.options.enableBinarySupport,
        collaboration: true,
        realTimeCharts: true
      }
    };

    this.send(authMessage);
  }

  subscribeToDefaultChannels() {
    const defaultChannels = [
      'anomaly_detection',
      'system_metrics',
      'real_time_data',
      'dashboard_updates'
    ];

    defaultChannels.forEach(channel => {
      this.subscribe(channel);
    });
  }

  setupMessageHandlers() {
    // Real-time data handlers
    this.messageHandlers.set('anomaly_detection', (data) => {
      this.handleAnomalyDetection(data);
    });

    this.messageHandlers.set('system_metrics', (data) => {
      this.handleSystemMetrics(data);
    });

    this.messageHandlers.set('real_time_data', (data) => {
      this.handleRealTimeData(data);
    });

    // Collaboration handlers
    this.messageHandlers.set('user_cursor', (data) => {
      this.handleUserCursor(data);
    });

    this.messageHandlers.set('annotation', (data) => {
      this.handleAnnotation(data);
    });

    this.messageHandlers.set('chart_interaction', (data) => {
      this.handleChartInteraction(data);
    });

    // System handlers
    this.messageHandlers.set('heartbeat', (data) => {
      this.handleHeartbeat(data);
    });

    this.messageHandlers.set('error', (data) => {
      this.handleServerError(data);
    });

    this.messageHandlers.set('notification', (data) => {
      this.handleNotification(data);
    });
  }

  routeMessage(message) {
    const handler = this.messageHandlers.get(message.type);
    if (handler) {
      handler(message.payload || message.data);
    } else {
      console.warn('[WebSocket] No handler for message type:', message.type);
    }

    // Notify subscribers
    const subscribers = this.subscriptions.get(message.type);
    if (subscribers) {
      subscribers.forEach(callback => {
        try {
          callback(message.payload || message.data);
        } catch (error) {
          console.error('[WebSocket] Subscriber callback failed:', error);
        }
      });
    }
  }

  handleAnomalyDetection(data) {
    // Dispatch to dashboard components
    window.dispatchEvent(new CustomEvent('anomaly-detected', {
      detail: data
    }));

    // Update real-time charts
    if (window.dashboardInstance) {
      window.dashboardInstance.addAnomalyToStream(data);
    }

    // Show notification for high-severity anomalies
    if (data.severity === 'high') {
      this.showNotification({
        title: 'High Severity Anomaly Detected',
        body: `Anomaly score: ${data.anomaly_score.toFixed(3)}`,
        icon: '/static/icons/anomaly-alert.png',
        requireInteraction: true
      });
    }
  }

  handleSystemMetrics(data) {
    // Update system health indicators
    window.dispatchEvent(new CustomEvent('system-metrics-update', {
      detail: data
    }));

    // Store metrics for offline access
    if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'UPDATE_LOCAL_DATA',
        payload: {
          entityType: 'metrics',
          entityId: 'system_health',
          data: data
        }
      });
    }
  }

  handleRealTimeData(data) {
    // Route to appropriate chart components
    switch (data.chartType) {
      case 'timeline':
        this.updateTimelineChart(data);
        break;
      case 'performance':
        this.updatePerformanceChart(data);
        break;
      case 'stream':
        this.updateStreamChart(data);
        break;
    }
  }

  handleUserCursor(data) {
    if (!this.collaborationState.has('cursors')) {
      this.collaborationState.set('cursors', new Map());
    }

    const cursors = this.collaborationState.get('cursors');
    cursors.set(data.userId, {
      x: data.x,
      y: data.y,
      timestamp: Date.now(),
      userName: data.userName
    });

    // Update cursor displays
    this.updateCollaboratorCursors();
  }

  handleAnnotation(data) {
    // Add collaborative annotation to charts
    window.dispatchEvent(new CustomEvent('collaborative-annotation', {
      detail: data
    }));

    // Store annotation
    this.storeAnnotation(data);
  }

  handleChartInteraction(data) {
    // Synchronize chart interactions across clients
    window.dispatchEvent(new CustomEvent('chart-interaction-sync', {
      detail: data
    }));
  }

  handleHeartbeat(data) {
    // Respond to server heartbeat
    this.send({
      type: 'heartbeat_response',
      timestamp: Date.now(),
      clientTime: Date.now()
    });
  }

  handleServerError(data) {
    console.error('[WebSocket] Server error:', data);

    // Show user-friendly error message
    this.showNotification({
      title: 'Server Error',
      body: data.message || 'An error occurred on the server',
      type: 'error'
    });
  }

  handleNotification(data) {
    this.showNotification(data);
  }

  // Public API methods
  subscribe(channel, callback) {
    if (!this.subscriptions.has(channel)) {
      this.subscriptions.set(channel, new Set());
    }

    if (callback) {
      this.subscriptions.get(channel).add(callback);
    }

    // Send subscription message
    this.send({
      type: 'subscribe',
      channel: channel,
      timestamp: Date.now()
    });

    return () => this.unsubscribe(channel, callback);
  }

  unsubscribe(channel, callback) {
    const subscribers = this.subscriptions.get(channel);
    if (subscribers) {
      if (callback) {
        subscribers.delete(callback);
      } else {
        subscribers.clear();
      }

      if (subscribers.size === 0) {
        this.subscriptions.delete(channel);

        // Send unsubscription message
        this.send({
          type: 'unsubscribe',
          channel: channel,
          timestamp: Date.now()
        });
      }
    }
  }

  send(message) {
    if (this.isConnected) {
      try {
        let data;

        // Use binary encoding for large messages if supported
        if (this.options.enableBinarySupport && this.shouldUseBinaryEncoding(message)) {
          data = this.encodeBinaryMessage(message);
        } else {
          data = JSON.stringify(message);
        }

        this.ws.send(data);
        return true;
      } catch (error) {
        console.error('[WebSocket] Send failed:', error);
        return false;
      }
    } else {
      // Queue message for when connection is restored
      if (this.messageQueue.length < this.options.messageQueueSize) {
        this.messageQueue.push(message);
      }
      return false;
    }
  }

  // Collaboration features
  broadcastCursorPosition(x, y) {
    this.send({
      type: 'user_cursor',
      payload: {
        x: x,
        y: y,
        timestamp: Date.now(),
        userId: this.clientId,
        userName: this.getCurrentUserName()
      }
    });
  }

  createAnnotation(x, y, text, chartId) {
    const annotation = {
      id: this.generateAnnotationId(),
      x: x,
      y: y,
      text: text,
      chartId: chartId,
      userId: this.clientId,
      userName: this.getCurrentUserName(),
      timestamp: Date.now()
    };

    this.send({
      type: 'annotation',
      payload: annotation
    });

    return annotation;
  }

  broadcastChartInteraction(chartId, interaction) {
    this.send({
      type: 'chart_interaction',
      payload: {
        chartId: chartId,
        interaction: interaction,
        userId: this.clientId,
        timestamp: Date.now()
      }
    });
  }

  // Connection management
  addConnectionListener(callback) {
    this.connectionListeners.add(callback);
    return () => this.connectionListeners.delete(callback);
  }

  disconnect() {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
    }
    this.stopHeartbeat();
    this.clearReconnectTimer();
  }

  reconnect() {
    this.disconnect();
    this.reconnectAttempts = 0;
    setTimeout(() => this.connect(), 1000);
  }

  // Private methods
  scheduleReconnect() {
    if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      console.error('[WebSocket] Max reconnection attempts reached');
      this.notifyConnectionListeners('max_retries_exceeded');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(
      this.options.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1),
      30000
    );

    console.log(`[WebSocket] Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  clearReconnectTimer() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  processMessageQueue() {
    while (this.messageQueue.length > 0 && this.isConnected) {
      const message = this.messageQueue.shift();
      this.send(message);
    }
  }

  setupHeartbeat() {
    this.startHeartbeat();

    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.stopHeartbeat();
      } else {
        this.startHeartbeat();
      }
    });
  }

  startHeartbeat() {
    this.stopHeartbeat();

    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected) {
        this.send({
          type: 'heartbeat',
          timestamp: Date.now()
        });
      }
    }, this.options.heartbeatInterval);
  }

  stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  setupVisibilityHandling() {
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && !this.isConnected) {
        // Page became visible and we're disconnected, try to reconnect
        this.reconnect();
      }
    });

    window.addEventListener('online', () => {
      if (!this.isConnected) {
        this.reconnect();
      }
    });

    window.addEventListener('offline', () => {
      this.disconnect();
    });
  }

  notifyConnectionListeners(event, data) {
    this.connectionListeners.forEach(callback => {
      try {
        callback(event, data);
      } catch (error) {
        console.error('[WebSocket] Connection listener failed:', error);
      }
    });
  }

  generateClientId() {
    if (!this.clientId) {
      this.clientId = 'client_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    return this.clientId;
  }

  generateAnnotationId() {
    return 'annotation_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  getCurrentUserName() {
    // Get from user context or session
    return window.currentUser?.name || 'Anonymous User';
  }

  shouldUseBinaryEncoding(message) {
    // Use binary for large data payloads
    const jsonSize = JSON.stringify(message).length;
    return jsonSize > 10000; // 10KB threshold
  }

  encodeBinaryMessage(message) {
    // Simple binary encoding (in production, use more efficient protocols like MessagePack)
    const jsonStr = JSON.stringify(message);
    const encoder = new TextEncoder();
    return encoder.encode(jsonStr);
  }

  decodeBinaryMessage(buffer) {
    const decoder = new TextDecoder();
    const jsonStr = decoder.decode(buffer);
    return JSON.parse(jsonStr);
  }

  showNotification(options) {
    // Use native notifications if available and permitted
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification(options.title, {
        body: options.body,
        icon: options.icon || '/static/icons/pynomaly-icon.png',
        requireInteraction: options.requireInteraction || false
      });
    } else {
      // Fallback to in-app notification
      this.showInAppNotification(options);
    }
  }

  showInAppNotification(options) {
    const notification = document.createElement('div');
    notification.className = `notification ${options.type || 'info'}`;
    notification.innerHTML = `
      <div class="notification-content">
        <div class="notification-title">${options.title}</div>
        <div class="notification-body">${options.body}</div>
      </div>
      <button class="notification-close">&times;</button>
    `;

    document.body.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 5000);

    // Close button
    notification.querySelector('.notification-close').addEventListener('click', () => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    });
  }

  updateCollaboratorCursors() {
    const cursors = this.collaborationState.get('cursors');
    if (!cursors) return;

    // Remove expired cursors
    const now = Date.now();
    cursors.forEach((cursor, userId) => {
      if (now - cursor.timestamp > 5000) { // 5 second timeout
        cursors.delete(userId);
      }
    });

    // Update cursor display
    window.dispatchEvent(new CustomEvent('collaborator-cursors-update', {
      detail: Array.from(cursors.entries())
    }));
  }

  storeAnnotation(annotation) {
    if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'UPDATE_LOCAL_DATA',
        payload: {
          entityType: 'annotation',
          entityId: annotation.id,
          data: annotation
        }
      });
    }
  }

  updateTimelineChart(data) {
    window.dispatchEvent(new CustomEvent('timeline-chart-update', {
      detail: data
    }));
  }

  updatePerformanceChart(data) {
    window.dispatchEvent(new CustomEvent('performance-chart-update', {
      detail: data
    }));
  }

  updateStreamChart(data) {
    window.dispatchEvent(new CustomEvent('stream-chart-update', {
      detail: data
    }));
  }

  // Static methods for singleton pattern
  static getInstance(options) {
    if (!EnhancedWebSocketService.instance) {
      EnhancedWebSocketService.instance = new EnhancedWebSocketService(options);
    }
    return EnhancedWebSocketService.instance;
  }

  static destroyInstance() {
    if (EnhancedWebSocketService.instance) {
      EnhancedWebSocketService.instance.disconnect();
      EnhancedWebSocketService.instance = null;
    }
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = EnhancedWebSocketService;
}

// Global access
if (typeof window !== 'undefined') {
  window.EnhancedWebSocketService = EnhancedWebSocketService;
}

export default EnhancedWebSocketService;
