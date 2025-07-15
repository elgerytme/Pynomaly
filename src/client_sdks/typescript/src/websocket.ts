/**
 * WebSocket client implementation for real-time updates
 * Provides reliable WebSocket connections with auto-reconnection and event handling
 */

import {
  WebSocketConfig,
  StreamMessage,
  StreamMessageType,
  StreamEventHandlers,
  StreamDetectionResult,
  StreamAlert,
  EventCallback,
} from './types';
import { WebSocketError, NetworkError } from './errors';

export interface WebSocketClientConfig extends WebSocketConfig {
  /** Authentication token */
  authToken?: string;
  /** API key for authentication */
  apiKey?: string;
  /** Custom headers for WebSocket connection */
  headers?: Record<string, string>;
  /** Connection timeout in milliseconds */
  connectionTimeout?: number;
  /** Message timeout in milliseconds */
  messageTimeout?: number;
}

/**
 * WebSocket client with automatic reconnection and comprehensive event handling
 */
export class WebSocketClient {
  private ws?: WebSocket;
  private readonly config: Required<WebSocketClientConfig>;
  private readonly eventHandlers: StreamEventHandlers = {};
  private connectionState: 'disconnected' | 'connecting' | 'connected' | 'reconnecting' = 'disconnected';
  private reconnectAttempts = 0;
  private heartbeatInterval?: NodeJS.Timeout;
  private reconnectTimeout?: NodeJS.Timeout;
  private connectionTimeout?: NodeJS.Timeout;
  private lastHeartbeat?: number;
  private messageQueue: StreamMessage[] = [];
  private subscriptions = new Set<string>();

  constructor(config: WebSocketClientConfig) {
    this.config = {
      enabled: true,
      autoReconnect: true,
      heartbeatInterval: 30000,
      maxReconnectAttempts: 10,
      reconnectDelay: 1000,
      connectionTimeout: 10000,
      messageTimeout: 5000,
      ...config,
    };
  }

  /**
   * Connect to WebSocket server
   */
  async connect(): Promise<void> {
    if (this.connectionState === 'connected' || this.connectionState === 'connecting') {
      return;
    }

    this.connectionState = 'connecting';

    try {
      await this.createConnection();
      this.connectionState = 'connected';
      this.reconnectAttempts = 0;
      this.startHeartbeat();
      this.processMessageQueue();
      this.eventHandlers.onConnect?.();
    } catch (error) {
      this.connectionState = 'disconnected';
      throw new WebSocketError(
        `Failed to connect to WebSocket: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.connectionState = 'disconnected';
    this.stopHeartbeat();
    this.clearReconnectTimeout();
    this.clearConnectionTimeout();

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = undefined;
    }

    this.eventHandlers.onDisconnect?.();
  }

  /**
   * Send message to server
   */
  async sendMessage(message: StreamMessage): Promise<void> {
    if (this.connectionState !== 'connected') {
      if (this.config.autoReconnect) {
        this.messageQueue.push(message);
        await this.reconnect();
        return;
      }
      throw new WebSocketError('WebSocket not connected');
    }

    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new WebSocketError('WebSocket connection not ready');
    }

    try {
      const messageString = JSON.stringify({
        ...message,
        timestamp: Date.now(),
      });
      this.ws.send(messageString);
    } catch (error) {
      throw new WebSocketError(
        `Failed to send message: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Subscribe to real-time updates for a stream
   */
  async subscribeToStream(streamId: string): Promise<void> {
    await this.sendMessage({
      type: 'subscribe',
      data: { streamId },
      timestamp: Date.now(),
      streamId,
    });
    this.subscriptions.add(streamId);
  }

  /**
   * Unsubscribe from stream updates
   */
  async unsubscribeFromStream(streamId: string): Promise<void> {
    await this.sendMessage({
      type: 'unsubscribe',
      data: { streamId },
      timestamp: Date.now(),
      streamId,
    });
    this.subscriptions.delete(streamId);
  }

  /**
   * Set event handlers
   */
  setEventHandlers(handlers: Partial<StreamEventHandlers>): void {
    Object.assign(this.eventHandlers, handlers);
  }

  /**
   * Add event handler for specific event type
   */
  on<T = any>(event: 'data', handler: EventCallback<StreamDetectionResult>): void;
  on<T = any>(event: 'alert', handler: EventCallback<StreamAlert>): void;
  on<T = any>(event: 'error', handler: EventCallback<Error>): void;
  on<T = any>(event: 'connect' | 'disconnect' | 'reconnect', handler: EventCallback<void>): void;
  on<T = any>(event: string, handler: EventCallback<T>): void {
    switch (event) {
      case 'data':
        this.eventHandlers.onData = handler as EventCallback<StreamDetectionResult>;
        break;
      case 'alert':
        this.eventHandlers.onAlert = handler as EventCallback<StreamAlert>;
        break;
      case 'error':
        this.eventHandlers.onError = handler as EventCallback<Error>;
        break;
      case 'connect':
        this.eventHandlers.onConnect = handler as EventCallback<void>;
        break;
      case 'disconnect':
        this.eventHandlers.onDisconnect = handler as EventCallback<void>;
        break;
      case 'reconnect':
        this.eventHandlers.onReconnect = handler as EventCallback<void>;
        break;
    }
  }

  /**
   * Get current connection state
   */
  getConnectionState(): string {
    return this.connectionState;
  }

  /**
   * Get connection statistics
   */
  getConnectionStats(): {
    state: string;
    reconnectAttempts: number;
    subscriptions: number;
    queuedMessages: number;
    lastHeartbeat?: number;
    uptime?: number;
  } {
    return {
      state: this.connectionState,
      reconnectAttempts: this.reconnectAttempts,
      subscriptions: this.subscriptions.size,
      queuedMessages: this.messageQueue.length,
      lastHeartbeat: this.lastHeartbeat,
      uptime: this.lastHeartbeat ? Date.now() - this.lastHeartbeat : undefined,
    };
  }

  /**
   * Create WebSocket connection
   */
  private async createConnection(): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl = this.buildWebSocketUrl();
      this.ws = new WebSocket(wsUrl);

      // Set connection timeout
      this.connectionTimeout = setTimeout(() => {
        this.ws?.close();
        reject(new WebSocketError('Connection timeout'));
      }, this.config.connectionTimeout);

      this.ws.onopen = () => {
        this.clearConnectionTimeout();
        this.setupEventHandlers();
        resolve();
      };

      this.ws.onerror = (event) => {
        this.clearConnectionTimeout();
        reject(new WebSocketError('WebSocket connection error'));
      };

      this.ws.onclose = () => {
        this.clearConnectionTimeout();
        reject(new WebSocketError('WebSocket connection closed during setup'));
      };
    });
  }

  /**
   * Setup WebSocket event handlers
   */
  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onmessage = (event) => {
      try {
        const message: StreamMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
        this.eventHandlers.onError?.(
          new WebSocketError('Failed to parse incoming message')
        );
      }
    };

    this.ws.onclose = (event) => {
      this.handleDisconnection(event);
    };

    this.ws.onerror = (event) => {
      console.error('WebSocket error:', event);
      this.eventHandlers.onError?.(
        new WebSocketError('WebSocket connection error')
      );
    };
  }

  /**
   * Handle incoming messages
   */
  private handleMessage(message: StreamMessage): void {
    switch (message.type) {
      case 'data':
        this.eventHandlers.onData?.(message.data as StreamDetectionResult);
        break;
      case 'alert':
        this.eventHandlers.onAlert?.(message.data as StreamAlert);
        break;
      case 'error':
        this.eventHandlers.onError?.(
          new WebSocketError(message.data?.message || 'Server error')
        );
        break;
      case 'heartbeat':
        this.handleHeartbeat();
        break;
      case 'status_update':
        this.handleStatusUpdate(message.data);
        break;
      default:
        console.warn('Unknown message type:', message.type);
    }
  }

  /**
   * Handle disconnection
   */
  private handleDisconnection(event: CloseEvent): void {
    this.connectionState = 'disconnected';
    this.stopHeartbeat();

    if (this.config.autoReconnect && this.reconnectAttempts < this.config.maxReconnectAttempts) {
      this.scheduleReconnect();
    } else {
      this.eventHandlers.onDisconnect?.();
    }
  }

  /**
   * Handle heartbeat message
   */
  private handleHeartbeat(): void {
    this.lastHeartbeat = Date.now();
  }

  /**
   * Handle status update message
   */
  private handleStatusUpdate(data: any): void {
    // Handle server status updates
    console.log('Status update:', data);
  }

  /**
   * Start heartbeat mechanism
   */
  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.connectionState === 'connected') {
        this.sendHeartbeat();
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Stop heartbeat mechanism
   */
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = undefined;
    }
  }

  /**
   * Send heartbeat message
   */
  private async sendHeartbeat(): Promise<void> {
    try {
      await this.sendMessage({
        type: 'heartbeat',
        data: { timestamp: Date.now() },
        timestamp: Date.now(),
      });
    } catch (error) {
      console.error('Failed to send heartbeat:', error);
    }
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    this.connectionState = 'reconnecting';
    this.reconnectAttempts++;

    const delay = this.calculateReconnectDelay();
    this.reconnectTimeout = setTimeout(() => {
      this.reconnect();
    }, delay);
  }

  /**
   * Attempt to reconnect
   */
  private async reconnect(): Promise<void> {
    try {
      await this.connect();
      this.eventHandlers.onReconnect?.();
      await this.resubscribeToStreams();
    } catch (error) {
      if (this.reconnectAttempts < this.config.maxReconnectAttempts) {
        this.scheduleReconnect();
      } else {
        this.connectionState = 'disconnected';
        this.eventHandlers.onError?.(
          new WebSocketError('Max reconnection attempts reached')
        );
      }
    }
  }

  /**
   * Resubscribe to all streams after reconnection
   */
  private async resubscribeToStreams(): Promise<void> {
    for (const streamId of this.subscriptions) {
      try {
        await this.subscribeToStream(streamId);
      } catch (error) {
        console.error(`Failed to resubscribe to stream ${streamId}:`, error);
      }
    }
  }

  /**
   * Process queued messages
   */
  private async processMessageQueue(): Promise<void> {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message) {
        try {
          await this.sendMessage(message);
        } catch (error) {
          console.error('Failed to send queued message:', error);
          // Re-queue the message
          this.messageQueue.unshift(message);
          break;
        }
      }
    }
  }

  /**
   * Calculate reconnection delay with exponential backoff
   */
  private calculateReconnectDelay(): number {
    const baseDelay = this.config.reconnectDelay;
    const exponentialDelay = Math.min(baseDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);
    const jitter = Math.random() * 0.1 * exponentialDelay;
    return exponentialDelay + jitter;
  }

  /**
   * Build WebSocket URL with authentication
   */
  private buildWebSocketUrl(): string {
    if (!this.config.url) {
      throw new WebSocketError('WebSocket URL not configured');
    }

    const url = new URL(this.config.url);
    
    // Add authentication parameters
    if (this.config.authToken) {
      url.searchParams.append('token', this.config.authToken);
    } else if (this.config.apiKey) {
      url.searchParams.append('api_key', this.config.apiKey);
    }

    return url.toString();
  }

  /**
   * Clear connection timeout
   */
  private clearConnectionTimeout(): void {
    if (this.connectionTimeout) {
      clearTimeout(this.connectionTimeout);
      this.connectionTimeout = undefined;
    }
  }

  /**
   * Clear reconnect timeout
   */
  private clearReconnectTimeout(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = undefined;
    }
  }
}

/**
 * Streaming data manager for handling real-time anomaly detection
 */
export class StreamingManager {
  private readonly wsClient: WebSocketClient;
  private readonly activeStreams = new Map<string, StreamConfig>();
  private readonly streamBuffers = new Map<string, any[]>();

  constructor(wsClient: WebSocketClient) {
    this.wsClient = wsClient;
    this.setupEventHandlers();
  }

  /**
   * Create a new data stream
   */
  async createStream(streamId: string, config: StreamConfig): Promise<void> {
    this.activeStreams.set(streamId, config);
    this.streamBuffers.set(streamId, []);
    
    await this.wsClient.subscribeToStream(streamId);
  }

  /**
   * Send data to a stream
   */
  async sendStreamData(streamId: string, data: number[]): Promise<void> {
    if (!this.activeStreams.has(streamId)) {
      throw new WebSocketError(`Stream ${streamId} not found`);
    }

    await this.wsClient.sendMessage({
      type: 'data',
      data: { streamId, data },
      timestamp: Date.now(),
      streamId,
    });
  }

  /**
   * Close a stream
   */
  async closeStream(streamId: string): Promise<void> {
    await this.wsClient.unsubscribeFromStream(streamId);
    this.activeStreams.delete(streamId);
    this.streamBuffers.delete(streamId);
  }

  /**
   * Get stream statistics
   */
  getStreamStats(streamId: string): {
    bufferSize: number;
    config?: StreamConfig;
    isActive: boolean;
  } {
    return {
      bufferSize: this.streamBuffers.get(streamId)?.length || 0,
      config: this.activeStreams.get(streamId),
      isActive: this.activeStreams.has(streamId),
    };
  }

  /**
   * Setup event handlers for WebSocket client
   */
  private setupEventHandlers(): void {
    this.wsClient.on('data', (result: StreamDetectionResult) => {
      this.handleStreamData(result);
    });

    this.wsClient.on('alert', (alert: StreamAlert) => {
      this.handleStreamAlert(alert);
    });
  }

  /**
   * Handle incoming stream data
   */
  private handleStreamData(result: StreamDetectionResult): void {
    const buffer = this.streamBuffers.get(result.streamId);
    if (buffer) {
      buffer.push(result);
      
      // Maintain buffer size limit
      const config = this.activeStreams.get(result.streamId);
      if (config?.bufferSize && buffer.length > config.bufferSize) {
        buffer.shift();
      }
    }
  }

  /**
   * Handle stream alerts
   */
  private handleStreamAlert(alert: StreamAlert): void {
    console.warn('Stream alert:', alert);
  }
}