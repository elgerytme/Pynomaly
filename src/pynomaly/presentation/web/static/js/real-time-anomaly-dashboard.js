/**
 * Real-time Anomaly Detection Dashboard
 * Handles WebSocket connections for live anomaly monitoring
 */

class RealTimeAnomalyDashboard {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.anomalyCount = 0;
        this.lastProcessingRate = 0;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second

        // DOM elements
        this.connectionStatus = document.getElementById('connection-status');
        this.toggleButton = document.getElementById('toggle-stream');
        this.streamStatus = document.getElementById('stream-status');
        this.anomalyCountEl = document.getElementById('anomaly-count');
        this.processingRateEl = document.getElementById('processing-rate');
        this.lastDetectionEl = document.getElementById('last-detection');
        this.anomalyFlash = document.getElementById('anomaly-flash');
        this.anomalyDetails = document.getElementById('anomaly-details');

        this.init();
    }

    init() {
        // Bind event listeners
        if (this.toggleButton) {
            this.toggleButton.addEventListener('click', () => this.toggleConnection());
        }

        // Auto-connect on page load
        setTimeout(() => this.connect(), 1000);

        // Handle page visibility change
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && !this.isConnected) {
                this.connect();
            }
        });

        // Handle page unload
        window.addEventListener('beforeunload', () => {
            if (this.websocket) {
                this.websocket.close();
            }
        });
    }

    connect() {
        if (this.websocket && this.isConnected) {
            console.log('Already connected to anomaly detection stream');
            return;
        }

        try {
            // Determine WebSocket URL
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            const wsUrl = `${protocol}//${host}/api/ws/detections`;

            console.log('Connecting to anomaly detection stream:', wsUrl);

            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = (event) => {
                console.log('Connected to anomaly detection stream');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.reconnectDelay = 1000;

                this.updateConnectionStatus('connected', 'Connected');
                this.updateToggleButton('Disconnect');
                this.updateStreamStatus('Connected');

                // Send initial ping
                this.sendMessage({ type: 'ping' });

                // Start heartbeat
                this.startHeartbeat();
            };

            this.websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleMessage(message);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            this.websocket.onclose = (event) => {
                console.log('Disconnected from anomaly detection stream:', event.code, event.reason);
                this.isConnected = false;

                this.updateConnectionStatus('disconnected', 'Disconnected');
                this.updateToggleButton('Connect');
                this.updateStreamStatus('Disconnected');

                this.stopHeartbeat();

                // Auto-reconnect if not manually closed
                if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.scheduleReconnect();
                }
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('error', 'Error');
                this.updateStreamStatus('Error');
            };

        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.updateConnectionStatus('error', 'Connection Failed');
        }
    }

    disconnect() {
        if (this.websocket) {
            console.log('Manually disconnecting from anomaly detection stream');
            this.websocket.close(1000, 'Manual disconnect');
            this.websocket = null;
        }

        this.stopHeartbeat();
    }

    toggleConnection() {
        if (this.isConnected) {
            this.disconnect();
        } else {
            this.connect();
        }
    }

    sendMessage(message) {
        if (this.websocket && this.isConnected) {
            try {
                this.websocket.send(JSON.stringify(message));
            } catch (error) {
                console.error('Failed to send WebSocket message:', error);
            }
        }
    }

    handleMessage(message) {
        console.log('Received message:', message.type, message);

        switch (message.type) {
            case 'connected':
                console.log('Confirmed connection:', message.message);
                break;

            case 'pong':
                // Heartbeat response
                break;

            case 'anomaly_detection':
                this.handleAnomalyDetection(message.data);
                break;

            case 'stream_started':
                this.updateStreamStatus('Active');
                console.log('Detection stream started');
                break;

            case 'stream_stopped':
                this.updateStreamStatus('Stopped');
                console.log('Detection stream stopped');
                break;

            case 'keepalive':
                // Keep connection alive
                break;

            case 'error':
                console.error('Server error:', message.message);
                this.updateStreamStatus('Error');
                break;

            default:
                console.log('Unknown message type:', message.type);
        }
    }

    handleAnomalyDetection(data) {
        console.log('Anomaly detection result:', data);

        // Update metrics
        if (data.anomalies_detected > 0) {
            this.anomalyCount += data.anomalies_detected;
            this.updateAnomalyCount();
            this.showAnomalyFlash(data);
        }

        // Update processing rate (simplified calculation)
        if (data.processing_time_ms && data.batch_size) {
            const samplesPerSecond = (data.batch_size / data.processing_time_ms) * 1000;
            this.lastProcessingRate = Math.round(samplesPerSecond * 100) / 100;
            this.updateProcessingRate();
        }

        // Update last detection time
        this.updateLastDetection(data.timestamp);

        // Update stream status
        this.updateStreamStatus('Processing');
    }

    showAnomalyFlash(data) {
        if (!this.anomalyFlash || !this.anomalyDetails) return;

        // Update anomaly details
        const anomalyText = data.anomalies_detected === 1
            ? '1 anomaly detected'
            : `${data.anomalies_detected} anomalies detected`;

        const processingTime = Math.round(data.processing_time_ms);
        const detailsText = `${anomalyText} in batch of ${data.batch_size} samples (${processingTime}ms)`;

        this.anomalyDetails.textContent = detailsText;

        // Show flash alert
        this.anomalyFlash.classList.remove('hidden');
        this.anomalyFlash.classList.add('flash-animation');

        // Add pulsing effect to monitor card
        const monitorCard = document.getElementById('anomaly-monitor');
        if (monitorCard) {
            monitorCard.classList.add('anomaly-detected');
        }

        // Hide after 5 seconds
        setTimeout(() => {
            this.anomalyFlash.classList.add('hidden');
            this.anomalyFlash.classList.remove('flash-animation');

            if (monitorCard) {
                monitorCard.classList.remove('anomaly-detected');
            }
        }, 5000);

        // Play notification sound if supported
        this.playNotificationSound();
    }

    playNotificationSound() {
        try {
            // Create a simple beep sound using Web Audio API
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);

            oscillator.frequency.setValueAtTime(800, audioContext.currentTime); // 800 Hz
            gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);

            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.5);
        } catch (error) {
            // Silently fail if audio is not supported
            console.debug('Audio notification not supported:', error);
        }
    }

    updateConnectionStatus(status, text) {
        if (!this.connectionStatus) return;

        // Remove all status classes
        this.connectionStatus.classList.remove('status-connected', 'status-disconnected', 'status-error');

        // Add appropriate status class
        this.connectionStatus.classList.add(`status-${status}`);
        this.connectionStatus.textContent = text;
    }

    updateToggleButton(text) {
        if (this.toggleButton) {
            this.toggleButton.textContent = text;
        }
    }

    updateStreamStatus(status) {
        if (this.streamStatus) {
            this.streamStatus.textContent = status;

            // Update status class
            this.streamStatus.classList.remove('status-info', 'status-normal', 'status-warning', 'status-error');

            switch (status.toLowerCase()) {
                case 'active':
                case 'processing':
                case 'connected':
                    this.streamStatus.classList.add('status-normal');
                    break;
                case 'idle':
                case 'stopped':
                    this.streamStatus.classList.add('status-info');
                    break;
                case 'error':
                case 'disconnected':
                    this.streamStatus.classList.add('status-error');
                    break;
                default:
                    this.streamStatus.classList.add('status-info');
            }
        }
    }

    updateAnomalyCount() {
        if (this.anomalyCountEl) {
            this.anomalyCountEl.textContent = this.anomalyCount.toLocaleString();
        }
    }

    updateProcessingRate() {
        if (this.processingRateEl) {
            this.processingRateEl.textContent = `${this.lastProcessingRate} samples/sec`;
        }
    }

    updateLastDetection(timestamp) {
        if (this.lastDetectionEl && timestamp) {
            try {
                const date = new Date(timestamp);
                const now = new Date();
                const diffMs = now.getTime() - date.getTime();

                let timeText;
                if (diffMs < 60000) { // Less than 1 minute
                    timeText = 'Just now';
                } else if (diffMs < 3600000) { // Less than 1 hour
                    const minutes = Math.floor(diffMs / 60000);
                    timeText = `${minutes}m ago`;
                } else if (diffMs < 86400000) { // Less than 1 day
                    const hours = Math.floor(diffMs / 3600000);
                    timeText = `${hours}h ago`;
                } else {
                    timeText = date.toLocaleDateString();
                }

                this.lastDetectionEl.textContent = timeText;
            } catch (error) {
                console.error('Error updating last detection time:', error);
            }
        }
    }

    startHeartbeat() {
        // Send ping every 30 seconds to keep connection alive
        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected) {
                this.sendMessage({ type: 'ping' });
            }
        }, 30000);
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    scheduleReconnect() {
        this.reconnectAttempts++;
        console.log(`Scheduling reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${this.reconnectDelay}ms`);

        setTimeout(() => {
            console.log(`Reconnection attempt ${this.reconnectAttempts}`);
            this.connect();
        }, this.reconnectDelay);

        // Exponential backoff
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000); // Cap at 30 seconds
    }

    // Public methods for external control
    startDetectionStream() {
        this.sendMessage({ type: 'start_stream' });
    }

    stopDetectionStream() {
        this.sendMessage({ type: 'stop_stream' });
    }

    resetCounters() {
        this.anomalyCount = 0;
        this.updateAnomalyCount();

        if (this.lastDetectionEl) {
            this.lastDetectionEl.textContent = 'Never';
        }

        if (this.processingRateEl) {
            this.processingRateEl.textContent = '0 samples/sec';
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Only initialize if we're on a page with the anomaly monitor
    if (document.getElementById('anomaly-monitor')) {
        window.anomalyDashboard = new RealTimeAnomalyDashboard();

        // Expose some methods globally for debugging
        window.startAnomalyStream = () => window.anomalyDashboard.startDetectionStream();
        window.stopAnomalyStream = () => window.anomalyDashboard.stopDetectionStream();
        window.resetAnomalyCounters = () => window.anomalyDashboard.resetCounters();

        console.log('Real-time anomaly dashboard initialized');
    }
});
