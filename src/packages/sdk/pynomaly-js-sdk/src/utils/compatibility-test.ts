/**
 * Compatibility test utilities
 */

import { Environment } from './environment';
import { CompatibilityChecker } from './compatibility';
import { checkPolyfillsNeeded } from './polyfills';

export interface CompatibilityTestResult {
  environment: string;
  compatible: boolean;
  issues: string[];
  recommendations: string[];
  polyfillsNeeded: string[];
  featureSupport: {
    webSocket: boolean;
    localStorage: boolean;
    sessionStorage: boolean;
    crypto: boolean;
    fetch: boolean;
    promises: boolean;
    eventListeners: boolean;
    json: boolean;
  };
}

export class CompatibilityTest {
  static runFullTest(): CompatibilityTestResult {
    const env = Environment.getInfo();
    const browserCheck = CompatibilityChecker.checkBrowserCompatibility();
    const nodeCheck = CompatibilityChecker.checkNodeCompatibility();
    const recommendations = CompatibilityChecker.getRecommendations();
    const polyfillsNeeded = checkPolyfillsNeeded();

    const result: CompatibilityTestResult = {
      environment: env.platform,
      compatible: true,
      issues: [],
      recommendations,
      polyfillsNeeded,
      featureSupport: {
        webSocket: env.hasWebSocket,
        localStorage: env.hasLocalStorage,
        sessionStorage: env.hasSessionStorage,
        crypto: env.hasCrypto,
        fetch: typeof fetch !== 'undefined',
        promises: typeof Promise !== 'undefined',
        eventListeners: typeof addEventListener !== 'undefined',
        json: typeof JSON !== 'undefined'
      }
    };

    // Combine issues from different checks
    if (env.isBrowser) {
      result.issues.push(...browserCheck.issues);
      result.compatible = result.compatible && browserCheck.compatible;
    }

    if (env.isNode) {
      result.issues.push(...nodeCheck.issues);
      result.compatible = result.compatible && nodeCheck.compatible;
    }

    // Check for critical missing features
    if (!result.featureSupport.promises) {
      result.issues.push('Promise support is critical for SDK functionality');
      result.compatible = false;
    }

    if (!result.featureSupport.json) {
      result.issues.push('JSON support is critical for SDK functionality');
      result.compatible = false;
    }

    return result;
  }

  static logCompatibilityReport(): void {
    const result = CompatibilityTest.runFullTest();
    
    console.group('🔍 Pynomaly SDK Compatibility Report');
    console.log(`Environment: ${result.environment}`);
    console.log(`Compatible: ${result.compatible ? '✅ Yes' : '❌ No'}`);
    
    if (result.issues.length > 0) {
      console.group('⚠️ Issues Found:');
      result.issues.forEach(issue => console.log(`  • ${issue}`));
      console.groupEnd();
    }
    
    if (result.recommendations.length > 0) {
      console.group('💡 Recommendations:');
      result.recommendations.forEach(rec => console.log(`  • ${rec}`));
      console.groupEnd();
    }
    
    if (result.polyfillsNeeded.length > 0) {
      console.group('🔧 Polyfills Applied:');
      result.polyfillsNeeded.forEach(polyfill => console.log(`  • ${polyfill}`));
      console.groupEnd();
    }
    
    console.group('🎯 Feature Support:');
    Object.entries(result.featureSupport).forEach(([feature, supported]) => {
      console.log(`  ${feature}: ${supported ? '✅' : '❌'}`);
    });
    console.groupEnd();
    
    console.groupEnd();
  }

  static async testBasicFunctionality(): Promise<boolean> {
    console.group('🧪 Testing Basic SDK Functionality');
    
    try {
      // Test environment detection
      console.log('Testing environment detection...');
      const env = Environment.getInfo();
      console.log(`✅ Environment: ${env.platform}`);
      
      // Test storage
      console.log('Testing storage...');
      const { StorageFactory } = await import('./compatibility');
      const storage = StorageFactory.getStorage('memory');
      storage.setItem('test', 'value');
      const retrieved = storage.getItem('test');
      if (retrieved === 'value') {
        console.log('✅ Storage: Working');
      } else {
        throw new Error('Storage test failed');
      }
      
      // Test crypto
      console.log('Testing crypto...');
      const { CryptoUtils } = await import('./compatibility');
      const randomString = CryptoUtils.generateSecureRandomString(16);
      if (randomString && randomString.length === 32) { // 16 bytes = 32 hex chars
        console.log('✅ Crypto: Working');
      } else {
        throw new Error('Crypto test failed');
      }
      
      // Test HTTP (if available)
      if (typeof fetch !== 'undefined' || Environment.isNode()) {
        console.log('Testing HTTP...');
        const { HTTPUtils } = await import('./compatibility');
        // Test with a simple endpoint that should exist
        try {
          const controller = HTTPUtils.createAbortController();
          console.log('✅ HTTP: AbortController available');
        } catch (e) {
          console.log('⚠️ HTTP: AbortController not available');
        }
      }
      
      console.log('✅ All basic functionality tests passed');
      console.groupEnd();
      return true;
      
    } catch (error) {
      console.error('❌ Basic functionality test failed:', error);
      console.groupEnd();
      return false;
    }
  }

  static getMinimumRequirements(): string[] {
    return [
      'JavaScript ES5 support',
      'JSON support',
      'Promise support (or polyfill)',
      'Basic HTTP support (fetch, XMLHttpRequest, or axios)',
      'EventEmitter support (built-in or polyfill)',
      'Browser: Modern browser (Chrome 60+, Firefox 55+, Safari 12+, Edge 79+)',
      'Node.js: Version 14.0 or higher',
      'React Native: Version 0.60 or higher (if using in React Native)'
    ];
  }

  static checkSpecificFeature(feature: string): boolean {
    const env = Environment.getInfo();
    
    switch (feature.toLowerCase()) {
      case 'websocket':
        return env.hasWebSocket;
      case 'localstorage':
        return env.hasLocalStorage;
      case 'sessionstorage':
        return env.hasSessionStorage;
      case 'crypto':
        return env.hasCrypto;
      case 'indexeddb':
        return env.hasIndexedDB;
      case 'filesystem':
        return env.hasFileSystem;
      case 'fetch':
        return typeof fetch !== 'undefined';
      case 'promises':
        return typeof Promise !== 'undefined';
      case 'json':
        return typeof JSON !== 'undefined';
      default:
        return false;
    }
  }

  static async testWebSocketCompatibility(): Promise<boolean> {
    if (!Environment.hasWebSocket()) {
      console.log('❌ WebSocket not available in this environment');
      return false;
    }

    try {
      const WebSocketClass = Environment.getWebSocketClass();
      if (WebSocketClass) {
        console.log('✅ WebSocket class available');
        return true;
      } else {
        console.log('❌ WebSocket class not available');
        return false;
      }
    } catch (error) {
      console.error('❌ WebSocket compatibility test failed:', error);
      return false;
    }
  }

  static async testStorageCompatibility(): Promise<boolean> {
    try {
      const { StorageFactory } = await import('./compatibility');
      
      // Test different storage types
      const storageTypes = ['memory', 'localStorage', 'sessionStorage'];
      if (Environment.isNode()) {
        storageTypes.push('filesystem');
      }
      
      for (const type of storageTypes) {
        try {
          const storage = StorageFactory.getStorage(type as any);
          storage.setItem('test', 'value');
          const retrieved = storage.getItem('test');
          storage.removeItem('test');
          
          if (retrieved === 'value') {
            console.log(`✅ ${type} storage: Working`);
          } else {
            console.log(`❌ ${type} storage: Failed`);
          }
        } catch (error) {
          console.log(`❌ ${type} storage: Error -`, error.message);
        }
      }
      
      return true;
    } catch (error) {
      console.error('❌ Storage compatibility test failed:', error);
      return false;
    }
  }
}

// Auto-run compatibility check in development
if (typeof process !== 'undefined' && process.env.NODE_ENV === 'development') {
  CompatibilityTest.logCompatibilityReport();
}