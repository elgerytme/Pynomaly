/**
 * Unit tests for Environment utilities
 */

import { Environment } from '../../src/utils/environment';
import { mockBrowserEnvironment, mockNodeEnvironment } from '../utils/test-helpers';

describe('Environment', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('platform detection', () => {
    it('should detect browser environment', () => {
      mockBrowserEnvironment();

      expect(Environment.isBrowser()).toBe(true);
      expect(Environment.isNode()).toBe(false);
      expect(Environment.isWebWorker()).toBe(false);
      expect(Environment.isReactNative()).toBe(false);
    });

    it('should detect Node.js environment', () => {
      mockNodeEnvironment();

      expect(Environment.isBrowser()).toBe(false);
      expect(Environment.isNode()).toBe(true);
      expect(Environment.isWebWorker()).toBe(false);
      expect(Environment.isReactNative()).toBe(false);
    });

    it('should detect Web Worker environment', () => {
      // Mock Web Worker environment
      Object.defineProperty(global, 'importScripts', {
        value: jest.fn(),
        writable: true
      });
      Object.defineProperty(global, 'WorkerGlobalScope', {
        value: function() {},
        writable: true
      });

      expect(Environment.isWebWorker()).toBe(true);
      expect(Environment.isBrowser()).toBe(false);
      expect(Environment.isNode()).toBe(false);
    });

    it('should detect React Native environment', () => {
      // Mock React Native environment
      Object.defineProperty(global, 'navigator', {
        value: {
          product: 'ReactNative'
        },
        writable: true
      });

      expect(Environment.isReactNative()).toBe(true);
      expect(Environment.isBrowser()).toBe(false);
      expect(Environment.isNode()).toBe(false);
    });
  });

  describe('feature detection', () => {
    beforeEach(() => {
      mockBrowserEnvironment();
    });

    it('should detect WebSocket support', () => {
      Object.defineProperty(global, 'WebSocket', {
        value: function() {},
        writable: true
      });

      expect(Environment.hasWebSocket()).toBe(true);

      // Test without WebSocket
      delete (global as any).WebSocket;
      expect(Environment.hasWebSocket()).toBe(false);
    });

    it('should detect localStorage support', () => {
      Object.defineProperty(global, 'localStorage', {
        value: {
          getItem: jest.fn(),
          setItem: jest.fn(),
          removeItem: jest.fn(),
          clear: jest.fn()
        },
        writable: true
      });

      expect(Environment.hasLocalStorage()).toBe(true);

      // Test without localStorage
      delete (global as any).localStorage;
      expect(Environment.hasLocalStorage()).toBe(false);
    });

    it('should detect sessionStorage support', () => {
      Object.defineProperty(global, 'sessionStorage', {
        value: {
          getItem: jest.fn(),
          setItem: jest.fn(),
          removeItem: jest.fn(),
          clear: jest.fn()
        },
        writable: true
      });

      expect(Environment.hasSessionStorage()).toBe(true);

      // Test without sessionStorage
      delete (global as any).sessionStorage;
      expect(Environment.hasSessionStorage()).toBe(false);
    });

    it('should detect IndexedDB support', () => {
      Object.defineProperty(global, 'indexedDB', {
        value: {
          open: jest.fn()
        },
        writable: true
      });

      expect(Environment.hasIndexedDB()).toBe(true);

      // Test without IndexedDB
      delete (global as any).indexedDB;
      expect(Environment.hasIndexedDB()).toBe(false);
    });

    it('should detect crypto support', () => {
      Object.defineProperty(global, 'crypto', {
        value: {
          getRandomValues: jest.fn(),
          subtle: {}
        },
        writable: true
      });

      expect(Environment.hasCrypto()).toBe(true);

      // Test without crypto
      delete (global as any).crypto;
      expect(Environment.hasCrypto()).toBe(false);
    });

    it('should detect file system support', () => {
      mockNodeEnvironment();

      // Mock fs module for Node.js
      jest.doMock('fs', () => ({
        readFileSync: jest.fn(),
        writeFileSync: jest.fn()
      }));

      expect(Environment.hasFileSystem()).toBe(true);

      // Test browser environment (no file system)
      mockBrowserEnvironment();
      expect(Environment.hasFileSystem()).toBe(false);
    });
  });

  describe('universal implementations', () => {
    it('should get WebSocket implementation', () => {
      // Browser WebSocket
      mockBrowserEnvironment();
      const BrowserWebSocket = jest.fn();
      Object.defineProperty(global, 'WebSocket', {
        value: BrowserWebSocket,
        writable: true
      });

      expect(Environment.getWebSocket()).toBe(BrowserWebSocket);

      // Node.js WebSocket (would require 'ws' package)
      mockNodeEnvironment();
      delete (global as any).WebSocket;

      expect(() => Environment.getWebSocket()).toThrow();
    });

    it('should get storage implementation', () => {
      mockBrowserEnvironment();

      // localStorage
      const mockLocalStorage = {
        getItem: jest.fn(),
        setItem: jest.fn(),
        removeItem: jest.fn(),
        clear: jest.fn()
      };
      Object.defineProperty(global, 'localStorage', {
        value: mockLocalStorage,
        writable: true
      });

      expect(Environment.getStorage('local')).toBe(mockLocalStorage);

      // sessionStorage
      const mockSessionStorage = {
        getItem: jest.fn(),
        setItem: jest.fn(),
        removeItem: jest.fn(),
        clear: jest.fn()
      };
      Object.defineProperty(global, 'sessionStorage', {
        value: mockSessionStorage,
        writable: true
      });

      expect(Environment.getStorage('session')).toBe(mockSessionStorage);

      // Default to memory storage when not available
      delete (global as any).localStorage;
      delete (global as any).sessionStorage;

      const memoryStorage = Environment.getStorage('local');
      expect(memoryStorage).toBeDefined();
      expect(memoryStorage.getItem).toBeDefined();
      expect(memoryStorage.setItem).toBeDefined();
    });

    it('should get crypto implementation', () => {
      const mockCrypto = {
        getRandomValues: jest.fn(),
        subtle: {}
      };
      Object.defineProperty(global, 'crypto', {
        value: mockCrypto,
        writable: true
      });

      expect(Environment.getCrypto()).toBe(mockCrypto);

      // Test fallback when crypto not available
      delete (global as any).crypto;
      expect(() => Environment.getCrypto()).toThrow();
    });

    it('should get timer implementation', () => {
      const timers = Environment.getTimers();

      expect(timers.setTimeout).toBe(setTimeout);
      expect(timers.clearTimeout).toBe(clearTimeout);
      expect(timers.setInterval).toBe(setInterval);
      expect(timers.clearInterval).toBe(clearInterval);
    });

    it('should get URL implementation', () => {
      // Browser URL
      const BrowserURL = global.URL;
      expect(Environment.getURL()).toBe(BrowserURL);

      // Test when URL is not available
      delete (global as any).URL;
      expect(() => Environment.getURL()).toThrow();
    });
  });

  describe('environment info', () => {
    it('should get browser info', () => {
      mockBrowserEnvironment();
      
      Object.defineProperty(global, 'navigator', {
        value: {
          userAgent: 'Mozilla/5.0 (Test Browser)',
          language: 'en-US',
          languages: ['en-US', 'en']
        },
        writable: true
      });

      const info = Environment.getBrowserInfo();

      expect(info.userAgent).toBe('Mozilla/5.0 (Test Browser)');
      expect(info.language).toBe('en-US');
      expect(info.languages).toEqual(['en-US', 'en']);
    });

    it('should get Node.js info', () => {
      mockNodeEnvironment();

      const info = Environment.getNodeInfo();

      expect(info.version).toBe('v16.0.0');
      expect(info.platform).toBeDefined();
      expect(info.arch).toBeDefined();
    });

    it('should get capabilities', () => {
      mockBrowserEnvironment();

      // Mock all features as available
      Object.defineProperty(global, 'WebSocket', { value: function() {}, writable: true });
      Object.defineProperty(global, 'localStorage', { value: {}, writable: true });
      Object.defineProperty(global, 'sessionStorage', { value: {}, writable: true });
      Object.defineProperty(global, 'indexedDB', { value: {}, writable: true });
      Object.defineProperty(global, 'crypto', { value: { getRandomValues: jest.fn() }, writable: true });

      const capabilities = Environment.getCapabilities();

      expect(capabilities.webSocket).toBe(true);
      expect(capabilities.localStorage).toBe(true);
      expect(capabilities.sessionStorage).toBe(true);
      expect(capabilities.indexedDB).toBe(true);
      expect(capabilities.crypto).toBe(true);
      expect(capabilities.fileSystem).toBe(false); // Browser doesn't have file system
    });
  });

  describe('environment validation', () => {
    it('should validate required features', () => {
      mockBrowserEnvironment();

      // All features available
      Object.defineProperty(global, 'WebSocket', { value: function() {}, writable: true });
      Object.defineProperty(global, 'localStorage', { value: {}, writable: true });
      Object.defineProperty(global, 'crypto', { value: { getRandomValues: jest.fn() }, writable: true });

      expect(Environment.isSupported(['webSocket', 'localStorage', 'crypto'])).toBe(true);

      // Missing feature
      delete (global as any).WebSocket;
      expect(Environment.isSupported(['webSocket', 'localStorage', 'crypto'])).toBe(false);
    });

    it('should get missing features', () => {
      mockBrowserEnvironment();

      // Some features missing
      delete (global as any).WebSocket;
      delete (global as any).crypto;

      const missing = Environment.getMissingFeatures(['webSocket', 'localStorage', 'crypto']);
      expect(missing).toContain('webSocket');
      expect(missing).toContain('crypto');
      expect(missing).not.toContain('localStorage');
    });
  });

  describe('polyfill checking', () => {
    it('should check if polyfills are needed', () => {
      mockBrowserEnvironment();

      // No polyfills needed
      Object.defineProperty(global, 'WebSocket', { value: function() {}, writable: true });
      Object.defineProperty(global, 'crypto', { value: { getRandomValues: jest.fn() }, writable: true });

      expect(Environment.needsPolyfills()).toBe(false);

      // Polyfills needed
      delete (global as any).WebSocket;
      expect(Environment.needsPolyfills()).toBe(true);
    });

    it('should suggest polyfills', () => {
      mockBrowserEnvironment();

      delete (global as any).WebSocket;
      delete (global as any).crypto;

      const suggestions = Environment.getPolyfillSuggestions();
      expect(suggestions.length).toBeGreaterThan(0);
      expect(suggestions.some(s => s.includes('WebSocket'))).toBe(true);
    });
  });
});