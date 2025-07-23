/**
 * Official TypeScript/JavaScript client for platform services
 */

// Main client exports
export { PlatformClient } from './client';
export { AnomalyDetectionClient } from './services/anomaly-detection';
export { MLOpsClient } from './services/mlops';

// Configuration exports
export { ClientConfig, Environment } from './config';

// Type exports
export type * from './types';

// Error exports
export {
  PlatformError,
  AuthenticationError,
  AuthorizationError,
  ValidationError,
  NotFoundError,
  RateLimitError,
  ServerError,
  TimeoutError,
  ConnectionError
} from './errors';

// Utility exports
export { createClient } from './utils/factory';

// Version
export const VERSION = '0.1.0';