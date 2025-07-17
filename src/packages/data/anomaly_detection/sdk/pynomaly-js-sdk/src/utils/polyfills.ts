/**
 * Polyfills for cross-platform compatibility
 */

import { Environment } from './environment';

// TextEncoder/TextDecoder polyfill for Node.js environments
export function setupTextEncoding(): void {
  if (typeof TextEncoder === 'undefined' || typeof TextDecoder === 'undefined') {
    if (Environment.isNode()) {
      try {
        const util = require('util');
        if (!global.TextEncoder) {
          global.TextEncoder = util.TextEncoder;
        }
        if (!global.TextDecoder) {
          global.TextDecoder = util.TextDecoder;
        }
      } catch (e) {
        // Fallback implementation
        global.TextEncoder = class TextEncoder {
          encode(input: string): Uint8Array {
            const bytes = [];
            for (let i = 0; i < input.length; i++) {
              const code = input.charCodeAt(i);
              if (code < 0x80) {
                bytes.push(code);
              } else if (code < 0x800) {
                bytes.push(0xc0 | (code >> 6));
                bytes.push(0x80 | (code & 0x3f));
              } else if (code < 0xd800 || code >= 0xe000) {
                bytes.push(0xe0 | (code >> 12));
                bytes.push(0x80 | ((code >> 6) & 0x3f));
                bytes.push(0x80 | (code & 0x3f));
              } else {
                // Surrogate pair
                i++;
                const hi = code;
                const lo = input.charCodeAt(i);
                const codePoint = 0x10000 + (((hi & 0x3ff) << 10) | (lo & 0x3ff));
                bytes.push(0xf0 | (codePoint >> 18));
                bytes.push(0x80 | ((codePoint >> 12) & 0x3f));
                bytes.push(0x80 | ((codePoint >> 6) & 0x3f));
                bytes.push(0x80 | (codePoint & 0x3f));
              }
            }
            return new Uint8Array(bytes);
          }
        };

        global.TextDecoder = class TextDecoder {
          decode(input: Uint8Array): string {
            let result = '';
            let i = 0;
            while (i < input.length) {
              const byte1 = input[i++];
              if (byte1 < 0x80) {
                result += String.fromCharCode(byte1);
              } else if ((byte1 & 0xe0) === 0xc0) {
                const byte2 = input[i++];
                result += String.fromCharCode(((byte1 & 0x1f) << 6) | (byte2 & 0x3f));
              } else if ((byte1 & 0xf0) === 0xe0) {
                const byte2 = input[i++];
                const byte3 = input[i++];
                result += String.fromCharCode(((byte1 & 0x0f) << 12) | ((byte2 & 0x3f) << 6) | (byte3 & 0x3f));
              } else if ((byte1 & 0xf8) === 0xf0) {
                const byte2 = input[i++];
                const byte3 = input[i++];
                const byte4 = input[i++];
                const codePoint = ((byte1 & 0x07) << 18) | ((byte2 & 0x3f) << 12) | ((byte3 & 0x3f) << 6) | (byte4 & 0x3f);
                if (codePoint > 0xffff) {
                  codePoint -= 0x10000;
                  result += String.fromCharCode(0xd800 + (codePoint >> 10));
                  result += String.fromCharCode(0xdc00 + (codePoint & 0x3ff));
                } else {
                  result += String.fromCharCode(codePoint);
                }
              }
            }
            return result;
          }
        };
      }
    }
  }
}

// URL polyfill for environments that don't have it
export function setupURL(): void {
  if (typeof URL === 'undefined') {
    if (Environment.isNode()) {
      try {
        global.URL = require('url').URL;
      } catch (e) {
        // Fallback URL implementation
        global.URL = class URL {
          protocol: string;
          hostname: string;
          port: string;
          pathname: string;
          search: string;
          hash: string;
          searchParams: URLSearchParams;

          constructor(url: string, base?: string) {
            const match = url.match(/^(https?:)\/\/([^\/]+)(\/[^?#]*)?(\?[^#]*)?(#.*)?$/);
            if (!match) {
              throw new Error('Invalid URL');
            }

            this.protocol = match[1];
            const hostPort = match[2].split(':');
            this.hostname = hostPort[0];
            this.port = hostPort[1] || (this.protocol === 'https:' ? '443' : '80');
            this.pathname = match[3] || '/';
            this.search = match[4] || '';
            this.hash = match[5] || '';
            this.searchParams = new URLSearchParams(this.search);
          }

          toString(): string {
            return `${this.protocol}//${this.hostname}${this.port !== '80' && this.port !== '443' ? ':' + this.port : ''}${this.pathname}${this.search}${this.hash}`;
          }
        };
      }
    }
  }
}

// URLSearchParams polyfill
export function setupURLSearchParams(): void {
  if (typeof URLSearchParams === 'undefined') {
    global.URLSearchParams = class URLSearchParams {
      private params: Map<string, string> = new Map();

      constructor(init?: string | URLSearchParams | Record<string, string>) {
        if (typeof init === 'string') {
          this.parseString(init);
        } else if (init instanceof URLSearchParams) {
          this.params = new Map(init.params);
        } else if (init && typeof init === 'object') {
          for (const [key, value] of Object.entries(init)) {
            this.params.set(key, String(value));
          }
        }
      }

      private parseString(str: string): void {
        if (str.startsWith('?')) {
          str = str.slice(1);
        }
        const pairs = str.split('&');
        for (const pair of pairs) {
          const [key, value] = pair.split('=');
          if (key) {
            this.params.set(decodeURIComponent(key), decodeURIComponent(value || ''));
          }
        }
      }

      append(name: string, value: string): void {
        const existing = this.params.get(name);
        if (existing) {
          this.params.set(name, existing + ',' + value);
        } else {
          this.params.set(name, value);
        }
      }

      delete(name: string): void {
        this.params.delete(name);
      }

      get(name: string): string | null {
        return this.params.get(name) || null;
      }

      getAll(name: string): string[] {
        const value = this.params.get(name);
        return value ? value.split(',') : [];
      }

      has(name: string): boolean {
        return this.params.has(name);
      }

      set(name: string, value: string): void {
        this.params.set(name, value);
      }

      toString(): string {
        const pairs: string[] = [];
        for (const [key, value] of this.params) {
          pairs.push(`${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
        }
        return pairs.join('&');
      }
    };
  }
}

// Base64 polyfill for environments that don't have btoa/atob
export function setupBase64(): void {
  if (typeof btoa === 'undefined' || typeof atob === 'undefined') {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=';

    if (typeof btoa === 'undefined') {
      global.btoa = function(input: string): string {
        let str = input;
        let output = '';
        for (let block = 0, charCode, i = 0, map = chars; str.charAt(i | 0) || (map = '=', i % 1); output += map.charAt(63 & block >> 8 - i % 1 * 8)) {
          charCode = str.charCodeAt(i += 3/4);
          if (charCode > 0xFF) {
            throw new Error("'btoa' failed: The string to be encoded contains characters outside of the Latin1 range.");
          }
          block = block << 8 | charCode;
        }
        return output;
      };
    }

    if (typeof atob === 'undefined') {
      global.atob = function(input: string): string {
        let str = input.replace(/=+$/, '');
        let output = '';
        if (str.length % 4 === 1) {
          throw new Error("'atob' failed: The string to be decoded is not correctly encoded.");
        }
        for (let bc = 0, bs = 0, buffer, i = 0; buffer = str.charAt(i++); ~buffer && (bs = bc % 4 ? bs * 64 + buffer : buffer, bc++ % 4) ? output += String.fromCharCode(255 & bs >> (-2 * bc & 6)) : 0) {
          buffer = chars.indexOf(buffer);
        }
        return output;
      };
    }
  }
}

// Fetch polyfill for Node.js
export function setupFetch(): void {
  if (typeof fetch === 'undefined') {
    if (Environment.isNode()) {
      try {
        const fetch = require('node-fetch');
        global.fetch = fetch.default || fetch;
        global.Headers = fetch.Headers;
        global.Request = fetch.Request;
        global.Response = fetch.Response;
      } catch (e) {
        // node-fetch not available, will use axios instead
        console.warn('node-fetch not available, using axios for HTTP requests');
      }
    }
  }
}

// Performance polyfill
export function setupPerformance(): void {
  if (typeof performance === 'undefined') {
    if (Environment.isNode()) {
      try {
        const { performance } = require('perf_hooks');
        global.performance = performance;
      } catch (e) {
        // Fallback performance implementation
        global.performance = {
          now: () => Date.now(),
          timeOrigin: Date.now()
        };
      }
    } else {
      // Fallback for other environments
      global.performance = {
        now: () => Date.now(),
        timeOrigin: Date.now()
      };
    }
  }
}

// Console polyfill for environments without console
export function setupConsole(): void {
  if (typeof console === 'undefined') {
    global.console = {
      log: () => {},
      error: () => {},
      warn: () => {},
      info: () => {},
      debug: () => {},
      trace: () => {},
      group: () => {},
      groupEnd: () => {},
      time: () => {},
      timeEnd: () => {},
      assert: () => {},
      clear: () => {},
      count: () => {},
      countReset: () => {},
      dir: () => {},
      dirxml: () => {},
      table: () => {}
    };
  }
}

// Setup all polyfills
export function setupPolyfills(): void {
  setupTextEncoding();
  setupURL();
  setupURLSearchParams();
  setupBase64();
  setupFetch();
  setupPerformance();
  setupConsole();
}

// Check if polyfills are needed
export function checkPolyfillsNeeded(): string[] {
  const needed: string[] = [];
  
  if (typeof TextEncoder === 'undefined' || typeof TextDecoder === 'undefined') {
    needed.push('TextEncoder/TextDecoder');
  }
  
  if (typeof URL === 'undefined') {
    needed.push('URL');
  }
  
  if (typeof URLSearchParams === 'undefined') {
    needed.push('URLSearchParams');
  }
  
  if (typeof btoa === 'undefined' || typeof atob === 'undefined') {
    needed.push('Base64');
  }
  
  if (typeof fetch === 'undefined') {
    needed.push('Fetch');
  }
  
  if (typeof performance === 'undefined') {
    needed.push('Performance');
  }
  
  if (typeof console === 'undefined') {
    needed.push('Console');
  }
  
  return needed;
}