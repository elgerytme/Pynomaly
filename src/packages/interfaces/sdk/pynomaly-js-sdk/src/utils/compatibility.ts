/**
 * Cross-platform compatibility utilities
 */

import { Environment } from './environment';

// Universal storage interface
export interface UniversalStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
  clear(): void;
  length: number;
  key(index: number): string | null;
}

// Memory storage fallback
class MemoryStorage implements UniversalStorage {
  private storage: Map<string, string> = new Map();

  get length(): number {
    return this.storage.size;
  }

  getItem(key: string): string | null {
    return this.storage.get(key) || null;
  }

  setItem(key: string, value: string): void {
    this.storage.set(key, value);
  }

  removeItem(key: string): void {
    this.storage.delete(key);
  }

  clear(): void {
    this.storage.clear();
  }

  key(index: number): string | null {
    const keys = Array.from(this.storage.keys());
    return keys[index] || null;
  }
}

// Node.js file system storage
class FileSystemStorage implements UniversalStorage {
  private fs: any;
  private path: any;
  private storageDir: string;
  private cache: Map<string, string> = new Map();

  constructor(storageDir: string = '.pynomaly-storage') {
    if (Environment.isNode()) {
      try {
        this.fs = require('fs');
        this.path = require('path');
        this.storageDir = this.path.resolve(storageDir);
        this.ensureStorageDir();
        this.loadFromDisk();
      } catch (e) {
        console.warn('FileSystemStorage: Failed to initialize file system storage, falling back to memory storage');
        throw e;
      }
    } else {
      throw new Error('FileSystemStorage is only available in Node.js environment');
    }
  }

  private ensureStorageDir(): void {
    if (!this.fs.existsSync(this.storageDir)) {
      this.fs.mkdirSync(this.storageDir, { recursive: true });
    }
  }

  private getFilePath(key: string): string {
    const safeName = key.replace(/[^a-zA-Z0-9_-]/g, '_');
    return this.path.join(this.storageDir, `${safeName}.json`);
  }

  private loadFromDisk(): void {
    try {
      const files = this.fs.readdirSync(this.storageDir);
      for (const file of files) {
        if (file.endsWith('.json')) {
          const key = file.replace('.json', '').replace(/_/g, '');
          const filePath = this.path.join(this.storageDir, file);
          try {
            const data = this.fs.readFileSync(filePath, 'utf8');
            const parsed = JSON.parse(data);
            this.cache.set(parsed.key, parsed.value);
          } catch (e) {
            console.warn(`FileSystemStorage: Failed to load file ${file}`);
          }
        }
      }
    } catch (e) {
      console.warn('FileSystemStorage: Failed to load from disk');
    }
  }

  get length(): number {
    return this.cache.size;
  }

  getItem(key: string): string | null {
    return this.cache.get(key) || null;
  }

  setItem(key: string, value: string): void {
    this.cache.set(key, value);
    
    // Persist to disk
    try {
      const filePath = this.getFilePath(key);
      const data = JSON.stringify({ key, value, timestamp: Date.now() });
      this.fs.writeFileSync(filePath, data, 'utf8');
    } catch (e) {
      console.warn(`FileSystemStorage: Failed to persist key ${key}`);
    }
  }

  removeItem(key: string): void {
    this.cache.delete(key);
    
    // Remove from disk
    try {
      const filePath = this.getFilePath(key);
      if (this.fs.existsSync(filePath)) {
        this.fs.unlinkSync(filePath);
      }
    } catch (e) {
      console.warn(`FileSystemStorage: Failed to remove key ${key}`);
    }
  }

  clear(): void {
    this.cache.clear();
    
    // Clear disk storage
    try {
      const files = this.fs.readdirSync(this.storageDir);
      for (const file of files) {
        if (file.endsWith('.json')) {
          this.fs.unlinkSync(this.path.join(this.storageDir, file));
        }
      }
    } catch (e) {
      console.warn('FileSystemStorage: Failed to clear disk storage');
    }
  }

  key(index: number): string | null {
    const keys = Array.from(this.cache.keys());
    return keys[index] || null;
  }
}

// Cross-platform storage factory
export class StorageFactory {
  private static memoryStorage: MemoryStorage | null = null;
  private static fileSystemStorage: FileSystemStorage | null = null;

  static getStorage(type: 'localStorage' | 'sessionStorage' | 'memory' | 'filesystem' = 'localStorage'): UniversalStorage {
    switch (type) {
      case 'localStorage':
        if (Environment.hasLocalStorage()) {
          return window.localStorage;
        }
        return StorageFactory.getMemoryStorage();

      case 'sessionStorage':
        if (Environment.hasSessionStorage()) {
          return window.sessionStorage;
        }
        return StorageFactory.getMemoryStorage();

      case 'memory':
        return StorageFactory.getMemoryStorage();

      case 'filesystem':
        if (Environment.isNode()) {
          return StorageFactory.getFileSystemStorage();
        }
        return StorageFactory.getMemoryStorage();

      default:
        return StorageFactory.getMemoryStorage();
    }
  }

  private static getMemoryStorage(): MemoryStorage {
    if (!StorageFactory.memoryStorage) {
      StorageFactory.memoryStorage = new MemoryStorage();
    }
    return StorageFactory.memoryStorage;
  }

  private static getFileSystemStorage(): FileSystemStorage {
    if (!StorageFactory.fileSystemStorage) {
      try {
        StorageFactory.fileSystemStorage = new FileSystemStorage();
      } catch (e) {
        console.warn('Failed to create FileSystemStorage, falling back to memory storage');
        return StorageFactory.getMemoryStorage();
      }
    }
    return StorageFactory.fileSystemStorage;
  }
}

// Cross-platform timer utilities
export class TimerUtils {
  static setTimeout(callback: () => void, delay: number): NodeJS.Timeout | number {
    return setTimeout(callback, delay);
  }

  static setInterval(callback: () => void, delay: number): NodeJS.Timeout | number {
    return setInterval(callback, delay);
  }

  static clearTimeout(id: NodeJS.Timeout | number): void {
    clearTimeout(id as any);
  }

  static clearInterval(id: NodeJS.Timeout | number): void {
    clearInterval(id as any);
  }

  static setImmediate(callback: () => void): NodeJS.Immediate | number {
    if (typeof setImmediate !== 'undefined') {
      return setImmediate(callback);
    }
    return TimerUtils.setTimeout(callback, 0) as number;
  }

  static clearImmediate(id: NodeJS.Immediate | number): void {
    if (typeof clearImmediate !== 'undefined') {
      clearImmediate(id as NodeJS.Immediate);
    } else {
      TimerUtils.clearTimeout(id as number);
    }
  }

  static requestAnimationFrame(callback: () => void): number {
    if (typeof requestAnimationFrame !== 'undefined') {
      return requestAnimationFrame(callback);
    }
    return TimerUtils.setTimeout(callback, 16) as number; // ~60fps
  }

  static cancelAnimationFrame(id: number): void {
    if (typeof cancelAnimationFrame !== 'undefined') {
      cancelAnimationFrame(id);
    } else {
      TimerUtils.clearTimeout(id);
    }
  }
}

// Cross-platform crypto utilities
export class CryptoUtils {
  static getRandomValues(array: Uint8Array): Uint8Array {
    return Environment.getRandomValues(array);
  }

  static generateSecureRandom(length: number): Uint8Array {
    const array = new Uint8Array(length);
    return CryptoUtils.getRandomValues(array);
  }

  static generateSecureRandomString(length: number): string {
    const array = CryptoUtils.generateSecureRandom(length);
    return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
  }

  static generateUUID(): string {
    const bytes = CryptoUtils.generateSecureRandom(16);
    
    // Set version (4) and variant bits
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    
    const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
    return `${hex.substr(0, 8)}-${hex.substr(8, 4)}-${hex.substr(12, 4)}-${hex.substr(16, 4)}-${hex.substr(20, 12)}`;
  }

  static hash(data: string, algorithm: 'sha256' | 'sha1' | 'md5' = 'sha256'): Promise<string> {
    if (Environment.isBrowser() && window.crypto && window.crypto.subtle) {
      return CryptoUtils.browserHash(data, algorithm);
    } else if (Environment.isNode()) {
      return CryptoUtils.nodeHash(data, algorithm);
    } else {
      return CryptoUtils.fallbackHash(data);
    }
  }

  private static async browserHash(data: string, algorithm: string): Promise<string> {
    const encoder = new TextEncoder();
    const dataBuffer = encoder.encode(data);
    const hashBuffer = await window.crypto.subtle.digest(algorithm.toUpperCase(), dataBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }

  private static async nodeHash(data: string, algorithm: string): Promise<string> {
    const crypto = require('crypto');
    const hash = crypto.createHash(algorithm);
    hash.update(data);
    return hash.digest('hex');
  }

  private static async fallbackHash(data: string): Promise<string> {
    // Simple hash function for fallback (not cryptographically secure)
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      const char = data.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(16);
  }
}

// Cross-platform HTTP utilities
export class HTTPUtils {
  static async request(url: string, options: RequestInit = {}): Promise<Response> {
    if (typeof fetch !== 'undefined') {
      return fetch(url, options);
    } else if (Environment.isNode()) {
      // Use axios as fallback in Node.js
      const axios = require('axios');
      try {
        const response = await axios({
          method: options.method || 'GET',
          url,
          headers: options.headers,
          data: options.body,
          timeout: 30000,
          validateStatus: () => true // Don't throw for HTTP error codes
        });
        
        // Convert axios response to fetch-like response
        return {
          ok: response.status >= 200 && response.status < 300,
          status: response.status,
          statusText: response.statusText,
          headers: new Map(Object.entries(response.headers)),
          json: async () => response.data,
          text: async () => typeof response.data === 'string' ? response.data : JSON.stringify(response.data),
          blob: async () => response.data,
          arrayBuffer: async () => response.data
        } as Response;
      } catch (error) {
        throw new Error(`HTTP request failed: ${error.message}`);
      }
    } else {
      throw new Error('HTTP requests are not supported in this environment');
    }
  }

  static createAbortController(): AbortController {
    if (typeof AbortController !== 'undefined') {
      return new AbortController();
    } else {
      // Fallback implementation
      return {
        signal: { aborted: false },
        abort: () => {}
      } as AbortController;
    }
  }
}

// Cross-platform event utilities
export class EventUtils {
  static addEventListener(target: EventTarget | any, event: string, handler: EventListener): void {
    if (target && typeof target.addEventListener === 'function') {
      target.addEventListener(event, handler);
    } else if (target && typeof target.on === 'function') {
      target.on(event, handler);
    }
  }

  static removeEventListener(target: EventTarget | any, event: string, handler: EventListener): void {
    if (target && typeof target.removeEventListener === 'function') {
      target.removeEventListener(event, handler);
    } else if (target && typeof target.off === 'function') {
      target.off(event, handler);
    }
  }

  static createCustomEvent(type: string, detail: any = null): CustomEvent {
    if (typeof CustomEvent !== 'undefined') {
      return new CustomEvent(type, { detail });
    } else {
      // Fallback for environments without CustomEvent
      const event = document.createEvent('CustomEvent');
      event.initCustomEvent(type, false, false, detail);
      return event;
    }
  }
}

// Compatibility checker
export class CompatibilityChecker {
  static checkBrowserCompatibility(): { compatible: boolean; issues: string[] } {
    const issues: string[] = [];
    
    if (!Environment.isBrowser()) {
      return { compatible: true, issues: [] };
    }

    // Check for required features
    if (!window.Promise) {
      issues.push('Promise support is required');
    }
    
    if (!window.fetch && !window.XMLHttpRequest) {
      issues.push('HTTP request support is required');
    }
    
    if (!window.JSON) {
      issues.push('JSON support is required');
    }
    
    if (!Array.prototype.forEach) {
      issues.push('Array.forEach support is required');
    }
    
    if (!Object.keys) {
      issues.push('Object.keys support is required');
    }

    return {
      compatible: issues.length === 0,
      issues
    };
  }

  static checkNodeCompatibility(): { compatible: boolean; issues: string[] } {
    const issues: string[] = [];
    
    if (!Environment.isNode()) {
      return { compatible: true, issues: [] };
    }

    // Check Node.js version
    const nodeVersion = process.version;
    const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0]);
    
    if (majorVersion < 14) {
      issues.push(`Node.js version ${nodeVersion} is not supported. Please use Node.js 14 or higher.`);
    }

    // Check for required modules
    const requiredModules = ['fs', 'path', 'crypto', 'url'];
    for (const module of requiredModules) {
      try {
        require(module);
      } catch (e) {
        issues.push(`Required module '${module}' is not available`);
      }
    }

    return {
      compatible: issues.length === 0,
      issues
    };
  }

  static getRecommendations(): string[] {
    const recommendations: string[] = [];
    
    if (Environment.isBrowser()) {
      const browserCheck = CompatibilityChecker.checkBrowserCompatibility();
      if (!browserCheck.compatible) {
        recommendations.push('Update your browser to a more recent version');
        recommendations.push('Consider using a polyfill library like core-js');
      }
    }
    
    if (Environment.isNode()) {
      const nodeCheck = CompatibilityChecker.checkNodeCompatibility();
      if (!nodeCheck.compatible) {
        recommendations.push('Update Node.js to version 14 or higher');
        recommendations.push('Install missing dependencies: npm install ws axios');
      }
    }
    
    return recommendations;
  }
}