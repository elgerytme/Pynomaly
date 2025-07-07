/**
 * Pynomaly JavaScript/TypeScript SDK
 * 
 * Official client library for the Pynomaly anomaly detection platform.
 * Provides comprehensive access to all platform features including:
 * - Anomaly detection with multiple algorithms
 * - Real-time streaming data processing
 * - A/B testing and experimentation
 * - User and tenant management
 * - Compliance and audit logging
 */

export * from './client/PynomalyClient';
export * from './client/DetectionClient';
export * from './client/StreamingClient';
export * from './client/ABTestingClient';
export * from './client/UserManagementClient';
export * from './client/ComplianceClient';

export * from './types';
export * from './errors';
export * from './utils/WebSocketClient';
export * from './utils/EventEmitter';

// Re-export main client as default
export { PynomalyClient as default } from './client/PynomalyClient';