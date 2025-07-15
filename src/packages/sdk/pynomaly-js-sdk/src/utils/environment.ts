/**
 * Environment detection and compatibility utilities
 */

export interface EnvironmentInfo {
  isBrowser: boolean;
  isNode: boolean;
  isWebWorker: boolean;
  isReactNative: boolean;
  hasWebSocket: boolean;
  hasLocalStorage: boolean;
  hasSessionStorage: boolean;
  hasIndexedDB: boolean;
  hasCrypto: boolean;
  hasFileSystem: boolean;
  platform: string;
  userAgent?: string;
}

export class Environment {
  private static _info: EnvironmentInfo | null = null;

  // Get comprehensive environment information
  static getInfo(): EnvironmentInfo {
    if (Environment._info) {
      return Environment._info;
    }

    const info: EnvironmentInfo = {
      isBrowser: false,
      isNode: false,
      isWebWorker: false,
      isReactNative: false,
      hasWebSocket: false,
      hasLocalStorage: false,
      hasSessionStorage: false,
      hasIndexedDB: false,
      hasCrypto: false,
      hasFileSystem: false,
      platform: 'unknown',
      userAgent: undefined
    };

    // Detect browser environment
    if (typeof window !== 'undefined' && typeof window.document !== 'undefined') {
      info.isBrowser = true;
      info.platform = 'browser';
      info.userAgent = window.navigator?.userAgent;
      
      // Check for browser-specific features
      info.hasWebSocket = 'WebSocket' in window;
      info.hasLocalStorage = Environment.checkLocalStorage();
      info.hasSessionStorage = Environment.checkSessionStorage();
      info.hasIndexedDB = 'indexedDB' in window;
      info.hasCrypto = 'crypto' in window && 'getRandomValues' in window.crypto;
    }
    // Detect Node.js environment
    else if (typeof global !== 'undefined' && typeof process !== 'undefined' && process.versions && process.versions.node) {
      info.isNode = true;
      info.platform = 'node';
      info.hasFileSystem = true;
      
      // Check for Node.js specific features
      try {
        info.hasCrypto = !!require('crypto');
      } catch (e) {
        info.hasCrypto = false;
      }
      
      // Check for WebSocket support (ws package)
      try {
        require('ws');
        info.hasWebSocket = true;
      } catch (e) {
        info.hasWebSocket = false;
      }
    }
    // Detect Web Worker environment
    else if (typeof self !== 'undefined' && typeof importScripts === 'function') {
      info.isWebWorker = true;
      info.platform = 'webworker';
      info.hasCrypto = 'crypto' in self && 'getRandomValues' in self.crypto;
    }
    // Detect React Native environment
    else if (typeof navigator !== 'undefined' && navigator.product === 'ReactNative') {
      info.isReactNative = true;
      info.platform = 'reactnative';
      info.hasCrypto = 'crypto' in global;
    }

    Environment._info = info;
    return info;
  }

  // Check if localStorage is available and working
  private static checkLocalStorage(): boolean {
    try {
      if (typeof window === 'undefined' || !window.localStorage) {
        return false;
      }
      
      const test = '__pynomaly_test__';
      window.localStorage.setItem(test, test);
      window.localStorage.removeItem(test);
      return true;
    } catch (e) {
      return false;
    }
  }

  // Check if sessionStorage is available and working
  private static checkSessionStorage(): boolean {
    try {
      if (typeof window === 'undefined' || !window.sessionStorage) {
        return false;
      }
      
      const test = '__pynomaly_test__';
      window.sessionStorage.setItem(test, test);
      window.sessionStorage.removeItem(test);
      return true;
    } catch (e) {
      return false;
    }
  }

  // Convenience methods
  static isBrowser(): boolean {
    return Environment.getInfo().isBrowser;
  }

  static isNode(): boolean {
    return Environment.getInfo().isNode;
  }

  static isWebWorker(): boolean {
    return Environment.getInfo().isWebWorker;
  }

  static isReactNative(): boolean {
    return Environment.getInfo().isReactNative;
  }

  static hasWebSocket(): boolean {
    return Environment.getInfo().hasWebSocket;
  }

  static hasLocalStorage(): boolean {
    return Environment.getInfo().hasLocalStorage;
  }

  static hasSessionStorage(): boolean {
    return Environment.getInfo().hasSessionStorage;
  }

  static hasCrypto(): boolean {
    return Environment.getInfo().hasCrypto;
  }

  static getPlatform(): string {
    return Environment.getInfo().platform;
  }

  static getUserAgent(): string | undefined {
    return Environment.getInfo().userAgent;
  }

  // Get appropriate storage mechanism
  static getStorage(): Storage | null {
    if (Environment.hasLocalStorage()) {
      return window.localStorage;
    }
    return null;
  }

  static getSessionStorage(): Storage | null {
    if (Environment.hasSessionStorage()) {
      return window.sessionStorage;
    }
    return null;
  }

  // Get appropriate WebSocket implementation
  static getWebSocketClass(): typeof WebSocket | null {
    const info = Environment.getInfo();
    
    if (info.isBrowser && info.hasWebSocket) {
      return window.WebSocket;
    }
    
    if (info.isNode) {
      try {
        return require('ws');
      } catch (e) {
        return null;
      }
    }
    
    if (info.isWebWorker && 'WebSocket' in self) {
      return self.WebSocket;
    }
    
    return null;
  }

  // Get appropriate crypto implementation
  static getCrypto(): Crypto | any | null {
    const info = Environment.getInfo();
    
    if (info.isBrowser && info.hasCrypto) {
      return window.crypto;
    }
    
    if (info.isNode) {
      try {
        return require('crypto');
      } catch (e) {
        return null;
      }
    }
    
    if (info.isWebWorker && 'crypto' in self) {
      return self.crypto;
    }
    
    return null;
  }

  // Generate secure random values
  static getRandomValues(array: Uint8Array): Uint8Array {
    const crypto = Environment.getCrypto();
    
    if (crypto && crypto.getRandomValues) {
      return crypto.getRandomValues(array);
    }
    
    // Fallback for environments without crypto
    for (let i = 0; i < array.length; i++) {
      array[i] = Math.floor(Math.random() * 256);
    }
    
    return array;
  }

  // Check if feature is supported
  static supportsFeature(feature: string): boolean {
    const info = Environment.getInfo();
    
    switch (feature) {
      case 'websocket':
        return info.hasWebSocket;
      case 'localStorage':
        return info.hasLocalStorage;
      case 'sessionStorage':
        return info.hasSessionStorage;
      case 'indexedDB':
        return info.hasIndexedDB;
      case 'crypto':
        return info.hasCrypto;
      case 'fileSystem':
        return info.hasFileSystem;
      default:
        return false;
    }
  }

  // Log environment information
  static logEnvironmentInfo(): void {
    const info = Environment.getInfo();
    console.log('Pynomaly SDK Environment Info:', {
      platform: info.platform,
      features: {
        webSocket: info.hasWebSocket,
        localStorage: info.hasLocalStorage,
        sessionStorage: info.hasSessionStorage,
        indexedDB: info.hasIndexedDB,
        crypto: info.hasCrypto,
        fileSystem: info.hasFileSystem
      },
      userAgent: info.userAgent
    });
  }

  // Reset cached environment info (for testing)
  static resetCache(): void {
    Environment._info = null;
  }
}