/**
 * Jest test setup configuration
 */

// Import jest-extended if available
try {
  require('jest-extended');
} catch (e) {
  // jest-extended not available, continue without it
}

// Mock WebSocket for Node.js environment
if (typeof global !== 'undefined' && !global.WebSocket) {
  global.WebSocket = jest.fn().mockImplementation(() => ({
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    send: jest.fn(),
    close: jest.fn(),
    readyState: 1,
    CONNECTING: 0,
    OPEN: 1,
    CLOSING: 2,
    CLOSED: 3
  }));
}

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
  length: 0,
  key: jest.fn()
};

Object.defineProperty(global, 'localStorage', {
  value: localStorageMock,
  writable: true
});

// Mock sessionStorage
const sessionStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
  length: 0,
  key: jest.fn()
};

Object.defineProperty(global, 'sessionStorage', {
  value: sessionStorageMock,
  writable: true
});

// Mock crypto.getRandomValues
Object.defineProperty(global, 'crypto', {
  value: {
    getRandomValues: jest.fn().mockImplementation((arr: Uint8Array) => {
      for (let i = 0; i < arr.length; i++) {
        arr[i] = Math.floor(Math.random() * 256);
      }
      return arr;
    })
  }
});

// Mock fetch
global.fetch = jest.fn();

// Mock performance
Object.defineProperty(global, 'performance', {
  value: {
    now: jest.fn().mockReturnValue(Date.now()),
    timeOrigin: Date.now()
  }
});

// Mock console methods to reduce noise in tests
global.console = {
  ...global.console,
  warn: jest.fn(),
  error: jest.fn(),
  log: jest.fn(),
  debug: jest.fn()
};

// Setup test environment variables
process.env.NODE_ENV = 'test';

// Global test utilities
declare global {
  namespace jest {
    interface Matchers<R> {
      toBeWithinRange(floor: number, ceiling: number): R;
    }
  }
}

// Custom matchers
expect.extend({
  toBeWithinRange(received: number, floor: number, ceiling: number) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () => `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true
      };
    } else {
      return {
        message: () => `expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false
      };
    }
  }
});

// Cleanup after each test
afterEach(() => {
  jest.clearAllMocks();
  localStorageMock.clear();
  sessionStorageMock.clear();
});