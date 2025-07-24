/**
 * Streaming client for real-time anomaly detection
 */

import { EventEmitter } from 'eventemitter3';
import {
  StreamingClientConfig,
  AnomalyData,
  StreamingConfig,
  AlgorithmType,
  StreamingError,
  ValidationError,
  ConnectionError,
} from './types';

// WebSocket types for different environments
interface WebSocketLike {
  send(data: string): void;
  close(code?: number, reason?: string): void;
  addEventListener(type: string, listener: (event: any) => void): void;
  removeEventListener(type: string, listener: (event: any) => void): void;
  readyState: number;
}

export interface StreamingEvents {
  connect: () => void;
  disconnect: () => void;
  anomaly: (anomaly: AnomalyData) => void;
  error: (error: Error) => void;
  message: (data: any) => void;
}

export class StreamingClient extends EventEmitter<StreamingEvents> {
  private readonly config: Required<StreamingClientConfig>;
  private websocket: WebSocketLike | null = null;
  private connected = false;
  private running = false;
  private buffer: number[][] = [];
  private reconnectTimeout: NodeJS.Timeout | null = null;

  constructor(config: StreamingClientConfig) {
    super();

    this.config = {
      wsUrl: config.wsUrl,
      bufferSize: config.bufferSize ?? 100,
      detectionThreshold: config.detectionThreshold ?? 0.5,
      batchSize: config.batchSize ?? 10,
      algorithm: config.algorithm ?? AlgorithmType.ISOLATION_FOREST,
      autoRetrain: config.autoRetrain ?? false,
      apiKey: config.apiKey,
      autoReconnect: config.autoReconnect ?? true,
      reconnectDelay: config.reconnectDelay ?? 5000,
    };
  }

  /**
   * Start the streaming client
   */
  async start(): Promise<void> {
    if (this.running) {
      return;
    }

    this.running = true;
    await this.connect();
  }

  /**
   * Stop the streaming client
   */
  stop(): void {
    this.running = false;
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }

    this.connected = false;
    this.buffer = [];
  }

  /**
   * Send a data point for anomaly detection
   */
  sendData(dataPoint: number[]): void {
    if (!Array.isArray(dataPoint)) {
      throw new ValidationError('Data point must be an array', 'dataPoint', dataPoint);
    }

    if (!this.running) {
      throw new StreamingError('Client is not running');
    }

    this.buffer.push(dataPoint);

    // Process batch when buffer is full
    if (this.buffer.length >= this.config.batchSize) {
      const batch = this.buffer.splice(0, this.config.batchSize);
      this.sendBatch(batch);
    }
  }

  /**
   * Send multiple data points at once
   */
  sendBatch(batch: number[][]): void {
    if (!this.connected || !this.websocket) {
      // Buffer the data for later sending
      this.buffer.unshift(...batch);
      return;
    }

    try {
      const message = {
        type: 'batch',
        data: batch,
        timestamp: new Date().toISOString(),
      };

      this.websocket.send(JSON.stringify(message));
    } catch (error) {
      this.handleError(new StreamingError(`Failed to send batch: ${error}`));
    }
  }

  /**
   * Check if client is connected
   */
  get isConnected(): boolean {
    return this.connected;
  }

  /**
   * Get current buffer size
   */
  get bufferSize(): number {
    return this.buffer.length;
  }

  private async connect(): Promise<void> {
    try {
      const WebSocketClass = await this.getWebSocketClass();
      const wsUrl = this.buildWebSocketUrl();

      this.websocket = new WebSocketClass(wsUrl) as WebSocketLike;

      this.websocket.addEventListener('open', this.handleOpen.bind(this));
      this.websocket.addEventListener('message', this.handleMessage.bind(this));
      this.websocket.addEventListener('close', this.handleClose.bind(this));
      this.websocket.addEventListener('error', this.handleWebSocketError.bind(this));

    } catch (error) {
      throw new ConnectionError(`Failed to create WebSocket connection: ${error}`);
    }
  }

  private async getWebSocketClass(): Promise<any> {
    // Browser environment
    if (typeof window !== 'undefined' && window.WebSocket) {
      return window.WebSocket;
    }

    // Node.js environment
    if (typeof global !== 'undefined') {
      try {
        const { WebSocket } = await import('ws');
        return WebSocket;
      } catch (error) {
        throw new Error('WebSocket library not available. Install "ws" package for Node.js');
      }
    }

    throw new Error('WebSocket not supported in this environment');
  }

  private buildWebSocketUrl(): string {
    const url = new URL(this.config.wsUrl);
    
    if (this.config.apiKey) {
      url.searchParams.set('token', this.config.apiKey);
    }

    return url.toString();
  }

  private handleOpen(): void {
    this.connected = true;

    // Send initial configuration
    if (this.websocket) {
      const configMessage = {
        type: 'config',
        config: {
          bufferSize: this.config.bufferSize,
          detectionThreshold: this.config.detectionThreshold,
          batchSize: this.config.batchSize,
          algorithm: this.config.algorithm,
          autoRetrain: this.config.autoRetrain,
        },
      };

      this.websocket.send(JSON.stringify(configMessage));
    }

    // Send any buffered data
    if (this.buffer.length > 0) {
      const batch = this.buffer.splice(0);
      this.sendBatch(batch);
    }

    this.emit('connect');
  }

  private handleMessage(event: { data: string }): void {
    try {
      const data = JSON.parse(event.data);
      const messageType = data.type;

      switch (messageType) {
        case 'anomaly':
          this.handleAnomalyMessage(data.data);
          break;
        case 'error':
          this.handleError(new StreamingError(data.message || 'Unknown error'));
          break;
        case 'ping':
          // Respond to ping with pong
          if (this.websocket) {
            this.websocket.send(JSON.stringify({ type: 'pong' }));
          }
          break;
        default:
          this.emit('message', data);
      }
    } catch (error) {
      this.handleError(new StreamingError(`Invalid JSON message: ${event.data}`));
    }
  }

  private handleAnomalyMessage(anomalyData: any): void {
    try {
      // Convert from snake_case to camelCase
      const anomaly: AnomalyData = {
        index: anomalyData.index,
        score: anomalyData.score,
        dataPoint: anomalyData.data_point,
        confidence: anomalyData.confidence,
        timestamp: anomalyData.timestamp,
      };

      this.emit('anomaly', anomaly);
    } catch (error) {
      this.handleError(new StreamingError(`Error processing anomaly data: ${error}`));
    }
  }

  private handleClose(): void {
    this.connected = false;
    this.websocket = null;

    this.emit('disconnect');

    // Auto-reconnect if enabled and still running
    if (this.config.autoReconnect && this.running) {
      this.reconnectTimeout = setTimeout(() => {
        if (this.running) {
          this.connect().catch(error => this.handleError(error));
        }
      }, this.config.reconnectDelay);
    }
  }

  private handleWebSocketError(event: any): void {
    const error = new StreamingError(`WebSocket error: ${event.message || event.type}`);
    this.handleError(error);
  }

  private handleError(error: Error): void {
    this.emit('error', error);
  }
}

// Factory function for creating streaming clients
export function createStreamingClient(config: StreamingClientConfig): StreamingClient {
  return new StreamingClient(config);
}