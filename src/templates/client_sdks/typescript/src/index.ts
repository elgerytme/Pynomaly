/**
 * anomaly_detection TypeScript SDK
 *
 * Official TypeScript/JavaScript client library for the anomaly_detection anomaly detection API.
 * This SDK provides convenient access to the anomaly_detection API with full TypeScript support,
 * authentication handling, error management, and comprehensive documentation.
 *
 * Features:
 * - Complete API coverage with type-safe client methods
 * - JWT and API Key authentication support
 * - Automatic retry logic with exponential backoff
 * - Rate limiting and request throttling
 * - Comprehensive error handling
 * - Promise-based async/await support
 * - Built-in logging and debugging
 * - Node.js and browser compatibility
 *
 * @example
 * ```typescript
 * import { AnomalyDetectionClient } from '@anomaly_detection/client';
 *
 * const client = new AnomalyDetectionClient({
 *   baseUrl: 'https://api.anomaly_detection.com',
 *   apiKey: 'your-api-key'
 * });
 *
 * // Authenticate (if using JWT)
 * // await client.auth.login('username', 'password');
 *
 * // Detect anomalies
 * const result = await client.detection.detect({
 *   data: [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
 *   algorithm: 'isolation_forest',
 *   parameters: { contamination: 0.1 }
 * });
 *
 * console.log('Anomalies detected:', result.anomalies);
 * ```
 */

export { AnomalyDetectionClient } from './client';
export { anomaly-detectionError, AuthenticationError, AuthorizationError, ValidationError, ServerError, NetworkError, RateLimitError } from './errors';
export * from './types';
export * from './auth';
export * from './rate-limiter';
