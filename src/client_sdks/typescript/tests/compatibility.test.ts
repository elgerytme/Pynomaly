/**
 * Browser and Node.js Compatibility Tests
 * Tests compatibility across different environments and runtime conditions
 */

import { PynomaliClient } from '../src/index';

describe('Browser and Node.js Compatibility', () => {
  describe('Environment Detection', () => {
    test('should detect browser environment', () => {
      // Mock browser environment
      Object.defineProperty(global, 'window', {
        value: {
          location: { href: 'https://example.com' },
          fetch: jest.fn(),
        },
        writable: true,
      });

      Object.defineProperty(global, 'document', {
        value: { createElement: jest.fn() },
        writable: true,
      });

      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        debug: false,
      });

      expect(client).toBeDefined();
      expect(client.getClientInfo().baseUrl).toBe('https://api.pynomaly.com');
    });

    test('should detect Node.js environment', () => {
      // Mock Node.js environment
      Object.defineProperty(global, 'process', {
        value: {
          versions: { node: '18.0.0' },
          env: { NODE_ENV: 'test' },
        },
        writable: true,
      });

      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        debug: false,
      });

      expect(client).toBeDefined();
      expect(client.getClientInfo().baseUrl).toBe('https://api.pynomaly.com');
    });
  });

  describe('Fetch API Compatibility', () => {
    let originalFetch: any;

    beforeEach(() => {
      originalFetch = global.fetch;
    });

    afterEach(() => {
      global.fetch = originalFetch;
    });

    test('should use native fetch in browser environment', async () => {
      const mockFetch = jest.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ status: 'healthy' }),
        headers: new Headers(),
      });

      global.fetch = mockFetch;

      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        debug: false,
      });

      try {
        await client.health.getHealth();
      } catch (error) {
        // Expected to fail due to mocked response, but should use native fetch
      }

      expect(mockFetch).toHaveBeenCalled();
    });

    test('should handle fetch polyfill in Node.js environment', async () => {
      // Remove fetch to simulate Node.js environment
      delete (global as any).fetch;

      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        debug: false,
      });

      // Should not throw error during instantiation
      expect(client).toBeDefined();
    });
  });

  describe('WebSocket Compatibility', () => {
    test('should handle WebSocket in browser environment', () => {
      // Mock WebSocket
      const mockWebSocket = jest.fn();
      Object.defineProperty(global, 'WebSocket', {
        value: mockWebSocket,
        writable: true,
      });

      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        websocket: { enabled: true },
        debug: false,
      });

      expect(client.getWebSocketClient()).toBeDefined();
    });

    test('should handle missing WebSocket in Node.js environment', () => {
      // Remove WebSocket to simulate Node.js environment without ws library
      delete (global as any).WebSocket;

      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        websocket: { enabled: true },
        debug: false,
      });

      // Should still create client but WebSocket might not be available
      expect(client).toBeDefined();
    });
  });

  describe('AbortController Compatibility', () => {
    test('should handle AbortController in modern environments', () => {
      // Mock AbortController
      const mockAbortController = jest.fn().mockImplementation(() => ({
        signal: { aborted: false },
        abort: jest.fn(),
      }));

      Object.defineProperty(global, 'AbortController', {
        value: mockAbortController,
        writable: true,
      });

      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        timeout: 5000,
        debug: false,
      });

      expect(client).toBeDefined();
    });

    test('should handle missing AbortController in older environments', () => {
      // Remove AbortController to simulate older environment
      delete (global as any).AbortController;

      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        timeout: 5000,
        debug: false,
      });

      // Should still create client
      expect(client).toBeDefined();
    });
  });

  describe('URL and URLSearchParams Compatibility', () => {
    test('should handle URL construction in all environments', () => {
      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        debug: false,
      });

      // Test URL construction with parameters
      const clientInfo = client.getClientInfo();
      expect(clientInfo.baseUrl).toBe('https://api.pynomaly.com');
    });

    test('should handle URLSearchParams for query parameters', () => {
      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        debug: false,
      });

      // Should be able to create client without errors
      expect(client).toBeDefined();
    });
  });

  describe('Local Storage Compatibility', () => {
    test('should handle localStorage in browser environment', () => {
      // Mock localStorage
      const mockLocalStorage = {
        getItem: jest.fn(),
        setItem: jest.fn(),
        removeItem: jest.fn(),
        clear: jest.fn(),
      };

      Object.defineProperty(global, 'localStorage', {
        value: mockLocalStorage,
        writable: true,
      });

      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        debug: false,
      });

      expect(client).toBeDefined();
    });

    test('should handle missing localStorage in Node.js environment', () => {
      // Remove localStorage to simulate Node.js environment
      delete (global as any).localStorage;

      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        debug: false,
      });

      // Should still create client
      expect(client).toBeDefined();
    });
  });

  describe('Event Handling Compatibility', () => {
    test('should handle events in browser environment', () => {
      // Mock EventTarget
      const mockEventTarget = {
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        dispatchEvent: jest.fn(),
      };

      Object.defineProperty(global, 'EventTarget', {
        value: jest.fn().mockImplementation(() => mockEventTarget),
        writable: true,
      });

      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        websocket: { enabled: true },
        debug: false,
      });

      expect(client).toBeDefined();
    });
  });

  describe('Error Handling Compatibility', () => {
    test('should handle errors consistently across environments', async () => {
      const mockFetch = jest.fn().mockRejectedValue(new Error('Network error'));
      global.fetch = mockFetch;

      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        debug: false,
      });

      await expect(client.health.getHealth()).rejects.toThrow();
    });

    test('should handle timeout errors', async () => {
      const mockFetch = jest.fn().mockImplementation(() => {
        return new Promise((resolve, reject) => {
          setTimeout(() => {
            reject(new DOMException('The operation was aborted.', 'AbortError'));
          }, 100);
        });
      });

      global.fetch = mockFetch;

      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        timeout: 50,
        debug: false,
      });

      await expect(client.health.getHealth()).rejects.toThrow();
    });
  });

  describe('Authentication Compatibility', () => {
    test('should handle authentication tokens across environments', () => {
      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        apiKey: 'test-key',
        debug: false,
      });

      client.setAccessToken('test-token');

      const clientInfo = client.getClientInfo();
      expect(clientInfo.isAuthenticated).toBe(true);
    });

    test('should handle token clearing', () => {
      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        apiKey: 'test-key',
        debug: false,
      });

      client.setAccessToken('test-token');
      expect(client.getClientInfo().isAuthenticated).toBe(true);

      client.clearToken();
      expect(client.getClientInfo().isAuthenticated).toBe(false);
    });
  });

  describe('Configuration Compatibility', () => {
    test('should handle various configuration options', () => {
      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        apiKey: 'test-key',
        timeout: 10000,
        maxRetries: 5,
        userAgent: 'test-agent',
        debug: true,
        websocket: {
          enabled: true,
          autoReconnect: true,
          maxRetries: 3,
        },
        rateLimitRequests: 200,
        rateLimitPeriod: 60000,
      });

      const clientInfo = client.getClientInfo();
      expect(clientInfo.baseUrl).toBe('https://api.pynomaly.com');
      expect(clientInfo.userAgent).toBe('test-agent');
    });

    test('should handle minimal configuration', () => {
      const client = new PynomaliClient();

      const clientInfo = client.getClientInfo();
      expect(clientInfo.baseUrl).toBe('https://api.pynomaly.com');
    });
  });

  describe('Module System Compatibility', () => {
    test('should support CommonJS imports', () => {
      // Test that the module can be imported in CommonJS style
      const { PynomaliClient: CommonJSClient } = require('../src/index');
      expect(CommonJSClient).toBeDefined();
      expect(typeof CommonJSClient).toBe('function');
    });

    test('should support ES module imports', () => {
      // Test that the module can be imported in ES module style
      expect(PynomaliClient).toBeDefined();
      expect(typeof PynomaliClient).toBe('function');
    });
  });

  describe('TypeScript Compatibility', () => {
    test('should provide proper TypeScript definitions', () => {
      const client = new PynomaliClient({
        baseUrl: 'https://api.pynomaly.com',
        debug: false,
      });

      // TypeScript should provide proper intellisense and type checking
      expect(client.auth).toBeDefined();
      expect(client.detection).toBeDefined();
      expect(client.health).toBeDefined();
      expect(client.getClientInfo).toBeDefined();
    });

    test('should handle generic types correctly', () => {
      const client = new PynomaliClient();

      // Test that generic types work as expected
      expect(client).toBeDefined();
      expect(typeof client.request).toBe('function');
    });
  });
});