/**
 * WebSocket client for real-time updates
 */

import { EventEmitter } from 'eventemitter3';
import {
  WebSocketMessage,
  JobStatus,
  EventMap,
  EventCallback,
  PynomalyConfig,
  AuthToken
} from '../types';

// Universal WebSocket implementation
interface WebSocketLike {
  send(data: string): void;
  close(code?: number, reason?: string): void;
  addEventListener(type: string, listener: EventListenerOrEventListenerObject): void;
  removeEventListener(type: string, listener: EventListenerOrEventListenerObject): void;
  readyState: number;
  CONNECTING: number;
  OPEN: number;
  CLOSING: number;
  CLOSED: number;
}

import { Environment } from '../utils/environment';

// WebSocket factory for universal compatibility
function createWebSocket(url: string, protocols?: string[]): WebSocketLike {
  const WebSocketClass = Environment.getWebSocketClass();
  
  if (WebSocketClass) {
    return new WebSocketClass(url, protocols);
  } else {
    throw new Error('WebSocket not available in this environment. For Node.js, install the ws package: npm install ws');
  }
}

export interface WebSocketConfig {
  url: string;
  protocols?: string[];
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  messageQueueSize?: number;
  debug?: boolean;
}

export class PynomalyWebSocket extends EventEmitter<EventMap> {
  private ws: WebSocketLike | null = null;
  private config: WebSocketConfig;
  private authToken: AuthToken | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private reconnectAttempts: number = 0;
  private messageQueue: string[] = [];
  private subscriptions: Map<string, Set<string>> = new Map();
  private isConnecting: boolean = false;
  private isManuallyDisconnected: boolean = false;

  constructor(config: WebSocketConfig) {
    super();
    this.config = {
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      messageQueueSize: 100,
      debug: false,
      ...config
    };
  }

  // Connection management
  async connect(authToken?: AuthToken): Promise<void> {
    if (this.isConnecting || this.isConnected()) {
      return;
    }

    this.isConnecting = true;
    this.isManuallyDisconnected = false;
    this.authToken = authToken || this.authToken;

    return new Promise((resolve, reject) => {
      try {
        const url = this.buildWebSocketUrl();
        this.ws = createWebSocket(url, this.config.protocols);

        this.ws.addEventListener('open', () => {
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          this.processMessageQueue();
          
          if (this.config.debug) {
            console.log('WebSocket connected');
          }
          
          this.emit('connection:open');
          resolve();
        });

        this.ws.addEventListener('close', (event: any) => {
          this.isConnecting = false;
          this.cleanup();
          
          if (this.config.debug) {
            console.log('WebSocket disconnected:', event.code, event.reason);
          }
          
          this.emit('connection:close');
          
          if (!this.isManuallyDisconnected) {
            this.scheduleReconnect();
          }
        });

        this.ws.addEventListener('error', (error: any) => {
          this.isConnecting = false;
          
          if (this.config.debug) {
            console.error('WebSocket error:', error);
          }
          
          this.emit('connection:error', error);
          reject(error);
        });

        this.ws.addEventListener('message', (event: any) => {
          this.handleMessage(event.data);
        });

        // Connection timeout
        setTimeout(() => {
          if (this.isConnecting) {
            this.isConnecting = false;
            reject(new Error('WebSocket connection timeout'));
          }
        }, 10000);

      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  disconnect(): void {
    this.isManuallyDisconnected = true;
    this.cleanup();
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === this.ws?.OPEN;
  }

  // Message handling
  private handleMessage(data: string): void {
    try {
      const message: WebSocketMessage = JSON.parse(data);
      
      if (this.config.debug) {
        console.log('WebSocket message received:', message);
      }

      // Handle different message types
      switch (message.type) {
        case 'job_status':
          this.emit('job:status', message.data as JobStatus);
          break;
          
        case 'result':
          this.emit('job:completed', message.data);
          break;
          
        case 'error':
          this.emit('job:failed', { error: message.data });
          break;
          
        case 'notification':
          this.emit('message', message);
          break;
          
        default:
          this.emit('message', message);
      }
      
    } catch (error) {
      if (this.config.debug) {
        console.error('Error parsing WebSocket message:', error);
      }
      this.emit('connection:error', error as Error);
    }
  }

  // Subscription management
  subscribeToJob(jobId: string): void {
    this.send({
      type: 'subscribe',
      target: 'job',
      id: jobId
    });
    
    this.addSubscription('job', jobId);
  }

  unsubscribeFromJob(jobId: string): void {
    this.send({
      type: 'unsubscribe',
      target: 'job',
      id: jobId
    });
    
    this.removeSubscription('job', jobId);
  }

  subscribeToNotifications(userId: string): void {
    this.send({
      type: 'subscribe',
      target: 'notifications',
      id: userId
    });
    
    this.addSubscription('notifications', userId);
  }

  unsubscribeFromNotifications(userId: string): void {
    this.send({
      type: 'unsubscribe',
      target: 'notifications',
      id: userId
    });
    
    this.removeSubscription('notifications', userId);
  }

  private addSubscription(type: string, id: string): void {
    if (!this.subscriptions.has(type)) {
      this.subscriptions.set(type, new Set());
    }
    this.subscriptions.get(type)!.add(id);
  }

  private removeSubscription(type: string, id: string): void {
    const subs = this.subscriptions.get(type);
    if (subs) {
      subs.delete(id);
      if (subs.size === 0) {
        this.subscriptions.delete(type);
      }
    }
  }

  // Message sending
  send(data: any): void {
    const message = JSON.stringify(data);
    
    if (this.isConnected()) {
      this.ws!.send(message);
      
      if (this.config.debug) {
        console.log('WebSocket message sent:', data);
      }
    } else {
      this.queueMessage(message);
    }
  }

  private queueMessage(message: string): void {
    if (this.messageQueue.length >= this.config.messageQueueSize!) {
      this.messageQueue.shift(); // Remove oldest message
    }
    
    this.messageQueue.push(message);
    
    if (this.config.debug) {
      console.log('Message queued:', message);
    }
  }

  private processMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.isConnected()) {
      const message = this.messageQueue.shift()!;
      this.ws!.send(message);
      
      if (this.config.debug) {
        console.log('Queued message sent:', message);
      }
    }
  }

  // Heartbeat mechanism
  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected()) {
        this.send({ type: 'ping', timestamp: Date.now() });
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  // Reconnection logic
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts! || this.isManuallyDisconnected) {
      return;
    }

    this.reconnectAttempts++;
    
    if (this.config.debug) {
      console.log(`Scheduling reconnect attempt ${this.reconnectAttempts}/${this.config.maxReconnectAttempts}`);
    }

    this.reconnectTimer = setTimeout(() => {
      this.connect(this.authToken).catch(error => {
        if (this.config.debug) {
          console.error('Reconnection failed:', error);
        }
      });
    }, this.config.reconnectInterval! * this.reconnectAttempts);
  }

  // URL building
  private buildWebSocketUrl(): string {
    const baseUrl = this.config.url.replace(/^http/, 'ws');
    const url = new URL(baseUrl);
    
    if (this.authToken) {
      url.searchParams.set('token', this.authToken.token);
    }
    
    return url.toString();
  }

  // Cleanup
  private cleanup(): void {
    this.stopHeartbeat();
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  // Utility methods
  getConnectionState(): string {
    if (!this.ws) return 'DISCONNECTED';
    
    switch (this.ws.readyState) {
      case this.ws.CONNECTING: return 'CONNECTING';
      case this.ws.OPEN: return 'OPEN';
      case this.ws.CLOSING: return 'CLOSING';
      case this.ws.CLOSED: return 'CLOSED';
      default: return 'UNKNOWN';
    }
  }

  getSubscriptions(): Record<string, string[]> {
    const result: Record<string, string[]> = {};
    
    for (const [type, ids] of this.subscriptions) {
      result[type] = Array.from(ids);
    }
    
    return result;
  }

  getQueuedMessageCount(): number {
    return this.messageQueue.length;
  }

  // Event emitter overrides for type safety
  on<K extends keyof EventMap>(event: K, callback: EventCallback<EventMap[K]>): this {
    return super.on(event, callback);
  }

  off<K extends keyof EventMap>(event: K, callback: EventCallback<EventMap[K]>): this {
    return super.off(event, callback);
  }

  once<K extends keyof EventMap>(event: K, callback: EventCallback<EventMap[K]>): this {
    return super.once(event, callback);
  }

  emit<K extends keyof EventMap>(event: K, ...args: Parameters<EventCallback<EventMap[K]>>): boolean {
    return super.emit(event, ...args);
  }

  // Destroy method
  destroy(): void {
    this.disconnect();
    this.removeAllListeners();
    this.subscriptions.clear();
    this.messageQueue.length = 0;
  }
}

// Factory function for creating WebSocket instances
export function createPynomalyWebSocket(config: WebSocketConfig): PynomalyWebSocket {
  return new PynomalyWebSocket(config);
}

// WebSocket manager for handling multiple connections
export class WebSocketManager {
  private connections: Map<string, PynomalyWebSocket> = new Map();
  private defaultConfig: Partial<WebSocketConfig> = {};

  constructor(defaultConfig: Partial<WebSocketConfig> = {}) {
    this.defaultConfig = defaultConfig;
  }

  createConnection(name: string, config: WebSocketConfig): PynomalyWebSocket {
    const fullConfig = { ...this.defaultConfig, ...config };
    const ws = new PynomalyWebSocket(fullConfig);
    
    this.connections.set(name, ws);
    
    return ws;
  }

  getConnection(name: string): PynomalyWebSocket | undefined {
    return this.connections.get(name);
  }

  closeConnection(name: string): void {
    const ws = this.connections.get(name);
    if (ws) {
      ws.destroy();
      this.connections.delete(name);
    }
  }

  closeAllConnections(): void {
    for (const [name, ws] of this.connections) {
      ws.destroy();
    }
    this.connections.clear();
  }

  getConnectionNames(): string[] {
    return Array.from(this.connections.keys());
  }

  getActiveConnections(): string[] {
    return Array.from(this.connections.entries())
      .filter(([, ws]) => ws.isConnected())
      .map(([name]) => name);
  }
}