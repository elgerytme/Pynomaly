/**
 * Unit tests for PynomalyWebSocket and WebSocketManager
 */

import { PynomalyWebSocket, WebSocketManager, createPynomalyWebSocket } from '../../src/core/websocket';
import { createMockWebSocket, flushPromises, waitFor } from '../utils/test-helpers';

// Mock WebSocket
global.WebSocket = jest.fn();

describe('PynomalyWebSocket', () => {
  let mockWS: any;
  let pynomalyWS: PynomalyWebSocket;

  beforeEach(() => {
    mockWS = createMockWebSocket();
    (global.WebSocket as jest.Mock).mockImplementation(() => mockWS);

    pynomalyWS = new PynomalyWebSocket('wss://test.com', {
      maxReconnectAttempts: 3,
      reconnectInterval: 1000,
      heartbeatInterval: 5000,
      messageQueueSize: 100
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('constructor', () => {
    it('should create PynomalyWebSocket with config', () => {
      expect(pynomalyWS).toBeInstanceOf(PynomalyWebSocket);
      expect(pynomalyWS.url).toBe('wss://test.com');
    });

    it('should use default config values', () => {
      const wsWithDefaults = new PynomalyWebSocket('wss://test.com');
      expect(wsWithDefaults.config.maxReconnectAttempts).toBe(5);
      expect(wsWithDefaults.config.reconnectInterval).toBe(2000);
    });
  });

  describe('connection management', () => {
    it('should connect successfully', async () => {
      const connectPromise = pynomalyWS.connect();
      
      // Simulate connection
      setTimeout(() => {
        if (mockWS.onopen) mockWS.onopen({} as Event);
      }, 0);

      await connectPromise;

      expect(pynomalyWS.isConnected()).toBe(true);
      expect(pynomalyWS.readyState).toBe(WebSocket.OPEN);
    });

    it('should handle connection failure', async () => {
      const connectPromise = pynomalyWS.connect();
      
      // Simulate connection error
      setTimeout(() => {
        if (mockWS.onerror) mockWS.onerror({} as Event);
      }, 0);

      await expect(connectPromise).rejects.toThrow('WebSocket connection failed');
    });

    it('should disconnect successfully', async () => {
      // First connect
      const connectPromise = pynomalyWS.connect();
      setTimeout(() => {
        if (mockWS.onopen) mockWS.onopen({} as Event);
      }, 0);
      await connectPromise;

      // Then disconnect
      pynomalyWS.disconnect();
      setTimeout(() => {
        if (mockWS.onclose) mockWS.onclose({} as CloseEvent);
      }, 0);

      await flushPromises();

      expect(pynomalyWS.isConnected()).toBe(false);
      expect(mockWS.close).toHaveBeenCalled();
    });

    it('should handle connection close event', async () => {
      const closeListener = jest.fn();
      pynomalyWS.on('close', closeListener);

      // Connect first
      const connectPromise = pynomalyWS.connect();
      setTimeout(() => {
        if (mockWS.onopen) mockWS.onopen({} as Event);
      }, 0);
      await connectPromise;

      // Simulate close
      setTimeout(() => {
        if (mockWS.onclose) mockWS.onclose({ code: 1000, reason: 'Normal closure' } as CloseEvent);
      }, 0);

      await flushPromises();

      expect(closeListener).toHaveBeenCalledWith({ code: 1000, reason: 'Normal closure' });
    });
  });

  describe('message handling', () => {
    beforeEach(async () => {
      // Connect before each test
      const connectPromise = pynomalyWS.connect();
      setTimeout(() => {
        if (mockWS.onopen) mockWS.onopen({} as Event);
      }, 0);
      await connectPromise;
    });

    it('should send message when connected', () => {
      const message = { type: 'test', data: 'hello' };
      pynomalyWS.send(message);

      expect(mockWS.send).toHaveBeenCalledWith(JSON.stringify(message));
    });

    it('should queue messages when not connected', () => {
      // Simulate disconnected state
      pynomalyWS.disconnect();

      const message = { type: 'test', data: 'hello' };
      pynomalyWS.send(message);

      // Message should be queued, not sent immediately
      expect(mockWS.send).not.toHaveBeenCalled();
    });

    it('should send queued messages on reconnection', async () => {
      // Disconnect
      pynomalyWS.disconnect();
      
      // Queue messages
      const message1 = { type: 'test1', data: 'hello1' };
      const message2 = { type: 'test2', data: 'hello2' };
      pynomalyWS.send(message1);
      pynomalyWS.send(message2);

      // Reconnect
      const connectPromise = pynomalyWS.connect();
      setTimeout(() => {
        if (mockWS.onopen) mockWS.onopen({} as Event);
      }, 0);
      await connectPromise;

      expect(mockWS.send).toHaveBeenCalledWith(JSON.stringify(message1));
      expect(mockWS.send).toHaveBeenCalledWith(JSON.stringify(message2));
    });

    it('should receive and parse messages', async () => {
      const messageListener = jest.fn();
      pynomalyWS.on('message', messageListener);

      const testMessage = { type: 'response', data: { result: 'success' } };

      // Simulate receiving message
      setTimeout(() => {
        if (mockWS.onmessage) {
          mockWS.onmessage({ data: JSON.stringify(testMessage) } as MessageEvent);
        }
      }, 0);

      await flushPromises();

      expect(messageListener).toHaveBeenCalledWith(testMessage);
    });

    it('should handle malformed messages gracefully', async () => {
      const errorListener = jest.fn();
      pynomalyWS.on('error', errorListener);

      // Simulate receiving malformed message
      setTimeout(() => {
        if (mockWS.onmessage) {
          mockWS.onmessage({ data: 'invalid json' } as MessageEvent);
        }
      }, 0);

      await flushPromises();

      expect(errorListener).toHaveBeenCalled();
    });
  });

  describe('reconnection logic', () => {
    it('should attempt reconnection on unexpected close', async () => {
      // Connect first
      const connectPromise = pynomalyWS.connect();
      setTimeout(() => {
        if (mockWS.onopen) mockWS.onopen({} as Event);
      }, 0);
      await connectPromise;

      // Simulate unexpected close
      setTimeout(() => {
        if (mockWS.onclose) mockWS.onclose({ code: 1006, reason: 'Abnormal closure' } as CloseEvent);
      }, 0);

      await flushPromises();

      // Should attempt to reconnect
      expect(global.WebSocket).toHaveBeenCalledTimes(2);
    });

    it('should not reconnect on normal close', async () => {
      // Connect first
      const connectPromise = pynomalyWS.connect();
      setTimeout(() => {
        if (mockWS.onopen) mockWS.onopen({} as Event);
      }, 0);
      await connectPromise;

      // Simulate normal close
      setTimeout(() => {
        if (mockWS.onclose) mockWS.onclose({ code: 1000, reason: 'Normal closure' } as CloseEvent);
      }, 0);

      await flushPromises();

      // Should not attempt to reconnect
      expect(global.WebSocket).toHaveBeenCalledTimes(1);
    });

    it('should limit reconnection attempts', async () => {
      const wsWithLimitedRetries = new PynomalyWebSocket('wss://test.com', {
        maxReconnectAttempts: 2,
        reconnectInterval: 100
      });

      // Mock failed connections
      (global.WebSocket as jest.Mock).mockImplementation(() => {
        const failedMockWS = createMockWebSocket();
        setTimeout(() => {
          if (failedMockWS.onerror) failedMockWS.onerror({} as Event);
        }, 0);
        return failedMockWS;
      });

      try {
        await wsWithLimitedRetries.connect();
      } catch (error) {
        // Expected to fail after max attempts
      }

      // Should have tried initial + 2 retries = 3 total
      expect(global.WebSocket).toHaveBeenCalledTimes(3);
    });
  });

  describe('heartbeat mechanism', () => {
    beforeEach(async () => {
      // Connect before each test
      const connectPromise = pynomalyWS.connect();
      setTimeout(() => {
        if (mockWS.onopen) mockWS.onopen({} as Event);
      }, 0);
      await connectPromise;
    });

    it('should send ping messages periodically', async () => {
      const wsWithFastHeartbeat = new PynomalyWebSocket('wss://test.com', {
        heartbeatInterval: 100
      });

      const connectPromise = wsWithFastHeartbeat.connect();
      setTimeout(() => {
        const fastMockWS = createMockWebSocket();
        (global.WebSocket as jest.Mock).mockReturnValue(fastMockWS);
        if (fastMockWS.onopen) fastMockWS.onopen({} as Event);
      }, 0);
      await connectPromise;

      // Wait for heartbeat
      await new Promise(resolve => setTimeout(resolve, 150));

      expect(mockWS.send).toHaveBeenCalledWith(JSON.stringify({ type: 'ping' }));
    });

    it('should handle pong responses', async () => {
      const messageListener = jest.fn();
      pynomalyWS.on('message', messageListener);

      // Simulate pong response
      setTimeout(() => {
        if (mockWS.onmessage) {
          mockWS.onmessage({ data: JSON.stringify({ type: 'pong' }) } as MessageEvent);
        }
      }, 0);

      await flushPromises();

      // Pong messages should not be forwarded to user listeners
      expect(messageListener).not.toHaveBeenCalled();
    });
  });

  describe('subscription management', () => {
    beforeEach(async () => {
      // Connect before each test
      const connectPromise = pynomalyWS.connect();
      setTimeout(() => {
        if (mockWS.onopen) mockWS.onopen({} as Event);
      }, 0);
      await connectPromise;
    });

    it('should subscribe to channels', () => {
      pynomalyWS.subscribe('anomaly-detection');

      expect(mockWS.send).toHaveBeenCalledWith(JSON.stringify({
        type: 'subscribe',
        channel: 'anomaly-detection'
      }));
    });

    it('should unsubscribe from channels', () => {
      pynomalyWS.unsubscribe('anomaly-detection');

      expect(mockWS.send).toHaveBeenCalledWith(JSON.stringify({
        type: 'unsubscribe',
        channel: 'anomaly-detection'
      }));
    });

    it('should track subscriptions', () => {
      pynomalyWS.subscribe('channel1');
      pynomalyWS.subscribe('channel2');

      expect(pynomalyWS.getSubscriptions()).toContain('channel1');
      expect(pynomalyWS.getSubscriptions()).toContain('channel2');

      pynomalyWS.unsubscribe('channel1');
      
      expect(pynomalyWS.getSubscriptions()).not.toContain('channel1');
      expect(pynomalyWS.getSubscriptions()).toContain('channel2');
    });

    it('should resubscribe on reconnection', async () => {
      // Subscribe to channels
      pynomalyWS.subscribe('channel1');
      pynomalyWS.subscribe('channel2');

      // Clear send calls from initial subscriptions
      mockWS.send.mockClear();

      // Simulate disconnect and reconnect
      pynomalyWS.disconnect();
      const connectPromise = pynomalyWS.connect();
      setTimeout(() => {
        if (mockWS.onopen) mockWS.onopen({} as Event);
      }, 0);
      await connectPromise;

      // Should resubscribe to both channels
      expect(mockWS.send).toHaveBeenCalledWith(JSON.stringify({
        type: 'subscribe',
        channel: 'channel1'
      }));
      expect(mockWS.send).toHaveBeenCalledWith(JSON.stringify({
        type: 'subscribe',
        channel: 'channel2'
      }));
    });
  });
});

describe('WebSocketManager', () => {
  let manager: WebSocketManager;

  beforeEach(() => {
    manager = new WebSocketManager();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('connection management', () => {
    it('should create and manage multiple connections', () => {
      const mockWS1 = createMockWebSocket();
      const mockWS2 = createMockWebSocket();
      
      (global.WebSocket as jest.Mock)
        .mockReturnValueOnce(mockWS1)
        .mockReturnValueOnce(mockWS2);

      const conn1 = manager.createConnection('conn1', 'wss://test1.com');
      const conn2 = manager.createConnection('conn2', 'wss://test2.com');

      expect(manager.getConnection('conn1')).toBe(conn1);
      expect(manager.getConnection('conn2')).toBe(conn2);
      expect(manager.getConnections()).toHaveProperty('conn1');
      expect(manager.getConnections()).toHaveProperty('conn2');
    });

    it('should remove connections', () => {
      const mockWS = createMockWebSocket();
      (global.WebSocket as jest.Mock).mockReturnValue(mockWS);

      const conn = manager.createConnection('test', 'wss://test.com');
      expect(manager.getConnection('test')).toBe(conn);

      manager.removeConnection('test');
      expect(manager.getConnection('test')).toBeUndefined();
    });

    it('should disconnect all connections', () => {
      const mockWS1 = createMockWebSocket();
      const mockWS2 = createMockWebSocket();
      
      (global.WebSocket as jest.Mock)
        .mockReturnValueOnce(mockWS1)
        .mockReturnValueOnce(mockWS2);

      const conn1 = manager.createConnection('conn1', 'wss://test1.com');
      const conn2 = manager.createConnection('conn2', 'wss://test2.com');

      const disconnectSpy1 = jest.spyOn(conn1, 'disconnect');
      const disconnectSpy2 = jest.spyOn(conn2, 'disconnect');

      manager.disconnectAll();

      expect(disconnectSpy1).toHaveBeenCalled();
      expect(disconnectSpy2).toHaveBeenCalled();
    });
  });

  describe('broadcasting', () => {
    it('should broadcast to all connections', async () => {
      const mockWS1 = createMockWebSocket();
      const mockWS2 = createMockWebSocket();
      
      (global.WebSocket as jest.Mock)
        .mockReturnValueOnce(mockWS1)
        .mockReturnValueOnce(mockWS2);

      const conn1 = manager.createConnection('conn1', 'wss://test1.com');
      const conn2 = manager.createConnection('conn2', 'wss://test2.com');

      // Mock connections as connected
      (conn1 as any).connected = true;
      (conn2 as any).connected = true;

      const message = { type: 'broadcast', data: 'hello all' };
      manager.broadcast(message);

      expect(mockWS1.send).toHaveBeenCalledWith(JSON.stringify(message));
      expect(mockWS2.send).toHaveBeenCalledWith(JSON.stringify(message));
    });

    it('should broadcast to specific connections', async () => {
      const mockWS1 = createMockWebSocket();
      const mockWS2 = createMockWebSocket();
      const mockWS3 = createMockWebSocket();
      
      (global.WebSocket as jest.Mock)
        .mockReturnValueOnce(mockWS1)
        .mockReturnValueOnce(mockWS2)
        .mockReturnValueOnce(mockWS3);

      const conn1 = manager.createConnection('conn1', 'wss://test1.com');
      const conn2 = manager.createConnection('conn2', 'wss://test2.com');
      const conn3 = manager.createConnection('conn3', 'wss://test3.com');

      // Mock connections as connected
      (conn1 as any).connected = true;
      (conn2 as any).connected = true;
      (conn3 as any).connected = true;

      const message = { type: 'targeted', data: 'hello some' };
      manager.broadcast(message, ['conn1', 'conn3']);

      expect(mockWS1.send).toHaveBeenCalledWith(JSON.stringify(message));
      expect(mockWS2.send).not.toHaveBeenCalled();
      expect(mockWS3.send).toHaveBeenCalledWith(JSON.stringify(message));
    });
  });
});

describe('createPynomalyWebSocket', () => {
  it('should create PynomalyWebSocket with factory function', () => {
    const mockWS = createMockWebSocket();
    (global.WebSocket as jest.Mock).mockReturnValue(mockWS);

    const ws = createPynomalyWebSocket('wss://test.com', {
      maxReconnectAttempts: 3
    });

    expect(ws).toBeInstanceOf(PynomalyWebSocket);
    expect(ws.url).toBe('wss://test.com');
    expect(ws.config.maxReconnectAttempts).toBe(3);
  });
});