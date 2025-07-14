/**
 * Pynomaly JavaScript SDK WebSocket Client
 * 
 * WebSocket client for real-time updates from Pynomaly services.
 * Handles connection management, message routing, and automatic reconnection.
 */

import { EventEmitter } from 'eventemitter3';
import {
  WebSocketConfig,
  WebSocketMessage,
  DetectionEvent,
  TrainingEvent,
  ExperimentEvent,
  PynomalyError
} from '../types';
import { WebSocketError, NetworkError } from './errors';

/**
 * WebSocket client for real-time communication with Pynomaly services.
 * 
 * Provides event-based communication for live updates on detection results,
 * training progress, and experiment status.
 * 
 * @example
 * ```typescript
 * const wsClient = new PynomalyWebSocketClient({
 *   url: 'wss://api.pynomaly.com/ws',
 *   reconnect: true
 * });
 * 
 * // Listen for detection events
 * wsClient.on('detection:progress', (event) => {
 *   console.log('Detection progress:', event.data.progress);
 * });
 * 
 * // Connect to WebSocket
 * await wsClient.connect('your-api-key');
 * 
 * // Subscribe to specific detector events
 * wsClient.subscribe('detector', 'detector-123');
 * ```
 */
export class PynomalyWebSocketClient extends EventEmitter {
  private readonly config: Required<WebSocketConfig>;
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private isConnecting = false;
  private isIntentionallyClosed = false;
  private subscriptions = new Set<string>();
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private lastPingTime = 0;

  constructor(config: WebSocketConfig) {
    super();
    
    this.config = {
      url: config.url,
      protocols: config.protocols || [],
      reconnect: config.reconnect !== false,
      reconnectInterval: config.reconnectInterval || 5000,
      maxReconnectAttempts: config.maxReconnectAttempts || 10
    };
  }

  /**
   * Check if WebSocket is connected.
   */
  get isConnected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  /**
   * Get current WebSocket state.
   */
  get readyState(): number {
    return this.socket?.readyState ?? WebSocket.CLOSED;
  }

  /**
   * Get WebSocket configuration.
   */
  get configuration(): WebSocketConfig {
    return { ...this.config };
  }

  /**
   * Connect to WebSocket server.
   * 
   * @param apiKey API key for authentication
   * @returns Promise resolving when connected
   */
  async connect(apiKey?: string): Promise<void> {
    if (this.isConnecting || this.isConnected) {
      return;
    }

    this.isConnecting = true;
    this.isIntentionallyClosed = false;

    return new Promise((resolve, reject) => {
      try {
        // Build WebSocket URL with authentication
        const url = new URL(this.config.url);
        if (apiKey) {
          url.searchParams.set('token', apiKey);
        }

        // Create WebSocket connection
        this.socket = new WebSocket(url.toString(), this.config.protocols);

        // Set up event handlers
        this.socket.onopen = () => {
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.emit('connected');
          this.startHeartbeat();
          resolve();
        };

        this.socket.onmessage = (event) => {
          this.handleMessage(event);
        };

        this.socket.onclose = (event) => {
          this.handleClose(event);
        };

        this.socket.onerror = (event) => {
          this.isConnecting = false;
          const error = new WebSocketError('WebSocket connection failed', {
            originalError: event
          });
          this.emit('error', error);
          reject(error);
        };

        // Connection timeout
        const timeout = setTimeout(() => {
          if (this.isConnecting) {
            this.isConnecting = false;
            this.socket?.close();
            const error = new WebSocketError('Connection timeout');
            reject(error);
          }
        }, 10000);

        // Clear timeout on successful connection
        this.once('connected', () => clearTimeout(timeout));

      } catch (error) {
        this.isConnecting = false;
        const wsError = new WebSocketError('Failed to create WebSocket connection', {
          originalError: error
        });
        reject(wsError);
      }
    });
  }

  /**
   * Disconnect from WebSocket server.
   */
  disconnect(): void {
    this.isIntentionallyClosed = true;
    this.stopHeartbeat();
    this.clearReconnectTimer();
    
    if (this.socket) {
      this.socket.close(1000, 'Client disconnect');
      this.socket = null;
    }
    
    this.emit('disconnected');
  }

  /**
   * Subscribe to real-time updates for a specific resource.
   * 
   * @param type Resource type (detector, training, experiment)
   * @param id Resource ID
   */
  subscribe(type: 'detector' | 'training' | 'experiment', id: string): void {
    const subscription = `${type}:${id}`;
    
    if (this.subscriptions.has(subscription)) {
      return;
    }

    this.subscriptions.add(subscription);

    if (this.isConnected) {
      this.sendMessage({
        type: 'subscribe',
        data: { resource: type, id },
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Unsubscribe from real-time updates for a specific resource.
   * 
   * @param type Resource type (detector, training, experiment)
   * @param id Resource ID
   */
  unsubscribe(type: 'detector' | 'training' | 'experiment', id: string): void {
    const subscription = `${type}:${id}`;
    
    if (!this.subscriptions.has(subscription)) {
      return;
    }

    this.subscriptions.delete(subscription);

    if (this.isConnected) {
      this.sendMessage({
        type: 'unsubscribe',
        data: { resource: type, id },
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Send a message to the WebSocket server.
   * 
   * @param message Message to send
   */
  private sendMessage(message: WebSocketMessage): void {
    if (!this.isConnected) {
      throw new WebSocketError('WebSocket is not connected');
    }

    try {
      this.socket!.send(JSON.stringify(message));
    } catch (error) {
      throw new WebSocketError('Failed to send WebSocket message', {
        originalError: error
      });
    }
  }

  /**
   * Handle incoming WebSocket messages.
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      
      switch (message.type) {
        case 'pong':
          this.handlePong(message);
          break;
        case 'detection:started':
        case 'detection:progress':
        case 'detection:completed':
        case 'detection:error':
          this.emit(message.type, message as DetectionEvent);
          break;
        case 'training:started':
        case 'training:progress':
        case 'training:completed':
        case 'training:error':
          this.emit(message.type, message as TrainingEvent);
          break;
        case 'experiment:started':
        case 'experiment:progress':
        case 'experiment:completed':
        case 'experiment:error':
          this.emit(message.type, message as ExperimentEvent);
          break;
        case 'error':
          this.emit('error', new WebSocketError(message.data.message, {
            details: message.data
          }));
          break;
        default:
          // Emit generic message event for unknown types
          this.emit('message', message);
      }
    } catch (error) {
      this.emit('error', new WebSocketError('Failed to parse WebSocket message', {
        originalError: error,
        details: { rawMessage: event.data }
      }));
    }
  }

  /**
   * Handle WebSocket close event.
   */
  private handleClose(event: CloseEvent): void {
    this.isConnecting = false;
    this.stopHeartbeat();
    
    this.emit('disconnected', {
      code: event.code,
      reason: event.reason,
      wasClean: event.wasClean
    });

    // Attempt reconnection if not intentionally closed
    if (!this.isIntentionallyClosed && this.config.reconnect) {
      this.scheduleReconnect();
    }
  }

  /**
   * Schedule a reconnection attempt.
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      this.emit('error', new WebSocketError(
        `Max reconnection attempts (${this.config.maxReconnectAttempts}) exceeded`
      ));
      return;
    }

    this.reconnectAttempts++;
    
    const delay = Math.min(
      this.config.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1),
      30000 // Max 30 seconds
    );

    this.emit('reconnecting', {
      attempt: this.reconnectAttempts,
      maxAttempts: this.config.maxReconnectAttempts,
      delay
    });

    this.reconnectTimer = setTimeout(async () => {
      try {
        await this.connect();
        // Resubscribe to all previous subscriptions
        this.resubscribeAll();
      } catch (error) {
        this.emit('error', new WebSocketError('Reconnection failed', {
          originalError: error
        }));
        this.scheduleReconnect();
      }
    }, delay);
  }

  /**
   * Clear reconnection timer.
   */
  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  /**
   * Resubscribe to all active subscriptions.
   */
  private resubscribeAll(): void {
    for (const subscription of this.subscriptions) {
      const [type, id] = subscription.split(':');
      this.sendMessage({
        type: 'subscribe',
        data: { resource: type, id },
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Start heartbeat to keep connection alive.
   */
  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected) {
        this.lastPingTime = Date.now();
        this.sendMessage({
          type: 'ping',
          data: { timestamp: this.lastPingTime },
          timestamp: new Date().toISOString()
        });
      }
    }, 30000); // Ping every 30 seconds
  }

  /**
   * Stop heartbeat timer.
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Handle pong message from server.
   */
  private handlePong(message: WebSocketMessage): void {
    const latency = Date.now() - this.lastPingTime;
    this.emit('heartbeat', { latency });
  }

  /**
   * Clean up resources.
   */
  dispose(): void {
    this.disconnect();
    this.removeAllListeners();
    this.subscriptions.clear();
  }
}

/**
 * Create a WebSocket client with default configuration.
 * 
 * @param config WebSocket configuration
 * @returns Configured PynomalyWebSocketClient instance
 */
export function createWebSocketClient(config: WebSocketConfig): PynomalyWebSocketClient {
  return new PynomalyWebSocketClient(config);
}