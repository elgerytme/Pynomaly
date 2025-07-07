/**
 * WebSocket client for real-time communication
 */

import { EventEmitter } from './EventEmitter';
import { WebSocketMessage, AnomalyAlert } from '../types';
import { PynomalyError, NetworkError } from '../errors';

export interface WebSocketConfig {
  url: string;
  protocols?: string[];
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  apiKey?: string;
  tenantId?: string;
}

export class WebSocketClient extends EventEmitter {
  private ws: WebSocket | null = null;
  private config: Required<WebSocketConfig>;
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private isConnecting = false;
  private isClosing = false;

  constructor(config: WebSocketConfig) {
    super();
    
    this.config = {
      url: config.url,
      protocols: config.protocols || [],
      reconnectInterval: config.reconnectInterval || 5000,
      maxReconnectAttempts: config.maxReconnectAttempts || 5,
      heartbeatInterval: config.heartbeatInterval || 30000,
      apiKey: config.apiKey || '',
      tenantId: config.tenantId || ''
    };
  }

  /**
   * Connect to WebSocket server
   */
  async connect(): Promise<void> {
    if (this.isConnecting || this.isConnected()) {
      return;
    }

    this.isConnecting = true;
    this.isClosing = false;

    try {
      const url = new URL(this.config.url);
      
      // Add authentication parameters
      if (this.config.apiKey) {
        url.searchParams.set('api_key', this.config.apiKey);
      }
      if (this.config.tenantId) {
        url.searchParams.set('tenant_id', this.config.tenantId);
      }

      this.ws = new WebSocket(url.toString(), this.config.protocols);
      
      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      this.ws.onerror = this.handleError.bind(this);

      // Wait for connection to open
      await new Promise<void>((resolve, reject) => {
        if (!this.ws) {
          reject(new NetworkError('WebSocket connection failed'));
          return;
        }

        const onOpen = () => {
          this.ws?.removeEventListener('open', onOpen);
          this.ws?.removeEventListener('error', onError);
          resolve();
        };

        const onError = (error: Event) => {
          this.ws?.removeEventListener('open', onOpen);
          this.ws?.removeEventListener('error', onError);
          reject(new NetworkError('WebSocket connection failed'));
        };

        this.ws.addEventListener('open', onOpen);
        this.ws.addEventListener('error', onError);
      });
    } catch (error) {
      this.isConnecting = false;
      throw error instanceof PynomalyError ? error : new NetworkError('Failed to connect to WebSocket');
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.isClosing = true;
    this.stopHeartbeat();
    this.stopReconnect();
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Send message to server
   */
  send(message: WebSocketMessage): void {
    if (!this.isConnected()) {
      throw new NetworkError('WebSocket is not connected');
    }

    try {
      this.ws!.send(JSON.stringify(message));
    } catch (error) {
      throw new NetworkError('Failed to send WebSocket message');
    }
  }

  /**
   * Subscribe to anomaly alerts
   */
  subscribeToAnomalyAlerts(processorId?: string): void {
    this.send({
      type: 'subscribe',
      data: {
        event: 'anomaly_alert',
        processor_id: processorId
      },
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Unsubscribe from anomaly alerts
   */
  unsubscribeFromAnomalyAlerts(processorId?: string): void {
    this.send({
      type: 'unsubscribe',
      data: {
        event: 'anomaly_alert',
        processor_id: processorId
      },
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Subscribe to processing status updates
   */
  subscribeToProcessingStatus(processorId?: string): void {
    this.send({
      type: 'subscribe',
      data: {
        event: 'processing_status',
        processor_id: processorId
      },
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Unsubscribe from processing status updates
   */
  unsubscribeFromProcessingStatus(processorId?: string): void {
    this.send({
      type: 'unsubscribe',
      data: {
        event: 'processing_status',
        processor_id: processorId
      },
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Subscribe to training progress updates
   */
  subscribeToTrainingProgress(detectorId?: string): void {
    this.send({
      type: 'subscribe',
      data: {
        event: 'training_progress',
        detector_id: detectorId
      },
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Check if WebSocket is connected
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get connection state
   */
  getReadyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED;
  }

  /**
   * Handle WebSocket open event
   */
  private handleOpen(): void {
    this.isConnecting = false;
    this.reconnectAttempts = 0;
    this.startHeartbeat();
    this.emit('connected');
  }

  /**
   * Handle WebSocket message event
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      this.emit('message', message);

      // Handle specific message types
      switch (message.type) {
        case 'anomaly_alert':
          this.emit('anomaly_alert', message.data as AnomalyAlert);
          break;
        case 'processing_status':
          this.emit('processing_status', message.data);
          break;
        case 'training_progress':
          this.emit('training_progress', message.data);
          break;
        case 'error':
          this.emit('error', new PynomalyError(message.data.message || 'Server error', message.data.code));
          break;
        case 'pong':
          // Heartbeat response
          break;
        default:
          this.emit('unknown_message', message);
      }
    } catch (error) {
      this.emit('error', new PynomalyError('Failed to parse WebSocket message', 'PARSE_ERROR'));
    }
  }

  /**
   * Handle WebSocket close event
   */
  private handleClose(event: CloseEvent): void {
    this.stopHeartbeat();
    this.emit('disconnected', { code: event.code, reason: event.reason });

    if (!this.isClosing && this.reconnectAttempts < this.config.maxReconnectAttempts) {
      this.scheduleReconnect();
    } else if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      this.emit('max_reconnect_attempts_reached');
    }
  }

  /**
   * Handle WebSocket error event
   */
  private handleError(event: Event): void {
    const error = new NetworkError('WebSocket error occurred');
    this.emit('error', error);
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }

    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++;
      this.emit('reconnecting', this.reconnectAttempts);
      
      this.connect().catch(error => {
        this.emit('reconnect_failed', error);
      });
    }, this.config.reconnectInterval);
  }

  /**
   * Stop reconnection attempts
   */
  private stopReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  /**
   * Start heartbeat
   */
  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected()) {
        this.send({
          type: 'ping',
          data: {},
          timestamp: new Date().toISOString()
        });
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Stop heartbeat
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Update authentication
   */
  updateAuth(apiKey: string, tenantId?: string): void {
    this.config.apiKey = apiKey;
    if (tenantId) {
      this.config.tenantId = tenantId;
    }

    // Reconnect with new auth if currently connected
    if (this.isConnected()) {
      this.disconnect();
      this.connect();
    }
  }

  /**
   * Get connection statistics
   */
  getStats(): {
    connected: boolean;
    reconnectAttempts: number;
    maxReconnectAttempts: number;
    readyState: number;
  } {
    return {
      connected: this.isConnected(),
      reconnectAttempts: this.reconnectAttempts,
      maxReconnectAttempts: this.config.maxReconnectAttempts,
      readyState: this.getReadyState()
    };
  }
}