/**
 * Utility Functions Index
 * Centralized exports for better tree shaking
 */

// Core utilities
export { default as debounce } from './debounce.js';
export { default as throttle } from './throttle.js';
export { default as formatters } from './formatters.js';
export { default as validators } from './validators.js';

// Performance utilities
export { default as LazyLoader } from './lazy-loader.js';
export { default as PerformanceMonitor } from './performance-monitor.js';
export { default as CacheManager } from './cache-manager.js';

// DOM utilities
export { default as DOMUtils } from './dom-utils.js';
export { default as EventBus } from './event-bus.js';

// API utilities
export { default as ApiClient } from './api-client.js';
export { default as RequestQueue } from './request-queue.js';

// PWA utilities
export { default as PWAManager } from './pwa-manager.js';
export { default as ServiceWorkerManager } from './service-worker-manager.js';

// Accessibility utilities
export { default as AccessibilityManager } from './accessibility.js';
export { default as KeyboardManager } from './keyboard-manager.js';

// Data utilities
export { default as DataProcessor } from './data-processor.js';
export { default as DataValidator } from './data-validator.js';

// Error handling
export { default as ErrorHandler } from './error-handler.js';
export { default as Logger } from './logger.js';
