/**
 * Comprehensive error handling for the Pynomaly TypeScript SDK
 * Provides structured error types with detailed information
 */

import { ErrorResponse } from './types';

/**
 * Base error class for all Pynomaly SDK errors
 */
export class PynomaliError extends Error {
  public readonly code: string;
  public readonly statusCode?: number;
  public readonly details?: any;
  public readonly requestId?: string;
  public readonly timestamp: string;

  constructor(
    message: string,
    code: string = 'PYNOMALY_ERROR',
    statusCode?: number,
    details?: any,
    requestId?: string
  ) {
    super(message);
    this.name = this.constructor.name;
    this.code = code;
    this.statusCode = statusCode;
    this.details = details;
    this.requestId = requestId;
    this.timestamp = new Date().toISOString();

    // Maintain proper stack trace
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /**
   * Convert error to JSON representation
   */
  toJSON(): ErrorResponse {
    return {
      code: this.code,
      message: this.message,
      details: this.details,
      requestId: this.requestId,
      timestamp: this.timestamp,
    };
  }

  /**
   * Create error from API response
   */
  static fromResponse(response: ErrorResponse): PynomaliError {
    return new PynomaliError(
      response.message,
      response.code,
      undefined,
      response.details,
      response.requestId
    );
  }
}

/**
 * Authentication related errors
 */
export class AuthenticationError extends PynomaliError {
  constructor(
    message: string = 'Authentication failed',
    details?: any,
    requestId?: string
  ) {
    super(message, 'AUTHENTICATION_ERROR', 401, details, requestId);
  }
}

/**
 * Authorization/permission related errors
 */
export class AuthorizationError extends PynomaliError {
  public readonly requiredPermissions?: string[];
  public readonly userPermissions?: string[];

  constructor(
    message: string = 'Access forbidden',
    requiredPermissions?: string[],
    userPermissions?: string[],
    requestId?: string
  ) {
    super(message, 'AUTHORIZATION_ERROR', 403, {
      requiredPermissions,
      userPermissions,
    }, requestId);
    this.requiredPermissions = requiredPermissions;
    this.userPermissions = userPermissions;
  }
}

/**
 * Input validation errors
 */
export class ValidationError extends PynomaliError {
  public readonly fieldErrors?: Record<string, string[]>;

  constructor(
    message: string = 'Validation failed',
    fieldErrors?: Record<string, string[]>,
    requestId?: string
  ) {
    super(message, 'VALIDATION_ERROR', 400, { fieldErrors }, requestId);
    this.fieldErrors = fieldErrors;
  }

  /**
   * Add field error
   */
  addFieldError(field: string, error: string): void {
    if (!this.fieldErrors) {
      (this as any).fieldErrors = {};
    }
    if (!this.fieldErrors![field]) {
      this.fieldErrors![field] = [];
    }
    this.fieldErrors![field].push(error);
  }

  /**
   * Get errors for specific field
   */
  getFieldErrors(field: string): string[] {
    return this.fieldErrors?.[field] || [];
  }

  /**
   * Check if field has errors
   */
  hasFieldErrors(field: string): boolean {
    return (this.fieldErrors?.[field]?.length || 0) > 0;
  }
}

/**
 * Server-side errors (5xx)
 */
export class ServerError extends PynomaliError {
  constructor(
    message: string = 'Internal server error',
    statusCode: number = 500,
    details?: any,
    requestId?: string
  ) {
    super(message, 'SERVER_ERROR', statusCode, details, requestId);
  }
}

/**
 * Network connectivity errors
 */
export class NetworkError extends PynomaliError {
  public readonly isTimeout: boolean;
  public readonly isOffline: boolean;

  constructor(
    message: string = 'Network error',
    isTimeout: boolean = false,
    isOffline: boolean = false,
    requestId?: string
  ) {
    super(message, 'NETWORK_ERROR', undefined, {
      isTimeout,
      isOffline,
    }, requestId);
    this.isTimeout = isTimeout;
    this.isOffline = isOffline;
  }
}

/**
 * Rate limiting errors
 */
export class RateLimitError extends PynomaliError {
  public readonly retryAfter: number;
  public readonly limit: number;
  public readonly remaining: number;
  public readonly resetTime: number;

  constructor(
    message: string = 'Rate limit exceeded',
    retryAfter: number = 60,
    limit: number = 100,
    remaining: number = 0,
    resetTime?: number,
    requestId?: string
  ) {
    super(message, 'RATE_LIMIT_ERROR', 429, {
      retryAfter,
      limit,
      remaining,
      resetTime,
    }, requestId);
    this.retryAfter = retryAfter;
    this.limit = limit;
    this.remaining = remaining;
    this.resetTime = resetTime || Date.now() + (retryAfter * 1000);
  }
}

/**
 * Resource not found errors
 */
export class NotFoundError extends PynomaliError {
  public readonly resourceType?: string;
  public readonly resourceId?: string;

  constructor(
    message: string = 'Resource not found',
    resourceType?: string,
    resourceId?: string,
    requestId?: string
  ) {
    super(message, 'NOT_FOUND_ERROR', 404, {
      resourceType,
      resourceId,
    }, requestId);
    this.resourceType = resourceType;
    this.resourceId = resourceId;
  }
}

/**
 * Configuration errors
 */
export class ConfigurationError extends PynomaliError {
  public readonly configKey?: string;
  public readonly expectedType?: string;
  public readonly actualValue?: any;

  constructor(
    message: string = 'Configuration error',
    configKey?: string,
    expectedType?: string,
    actualValue?: any
  ) {
    super(message, 'CONFIGURATION_ERROR', undefined, {
      configKey,
      expectedType,
      actualValue,
    });
    this.configKey = configKey;
    this.expectedType = expectedType;
    this.actualValue = actualValue;
  }
}

/**
 * Algorithm or model related errors
 */
export class AlgorithmError extends PynomaliError {
  public readonly algorithmName?: string;
  public readonly parameters?: Record<string, any>;

  constructor(
    message: string = 'Algorithm error',
    algorithmName?: string,
    parameters?: Record<string, any>,
    requestId?: string
  ) {
    super(message, 'ALGORITHM_ERROR', undefined, {
      algorithmName,
      parameters,
    }, requestId);
    this.algorithmName = algorithmName;
    this.parameters = parameters;
  }
}

/**
 * Data processing errors
 */
export class DataError extends PynomaliError {
  public readonly dataSize?: number;
  public readonly expectedFormat?: string;
  public readonly actualFormat?: string;

  constructor(
    message: string = 'Data processing error',
    dataSize?: number,
    expectedFormat?: string,
    actualFormat?: string,
    requestId?: string
  ) {
    super(message, 'DATA_ERROR', undefined, {
      dataSize,
      expectedFormat,
      actualFormat,
    }, requestId);
    this.dataSize = dataSize;
    this.expectedFormat = expectedFormat;
    this.actualFormat = actualFormat;
  }
}

/**
 * Streaming related errors
 */
export class StreamingError extends PynomaliError {
  public readonly streamId?: string;
  public readonly connectionState?: string;

  constructor(
    message: string = 'Streaming error',
    streamId?: string,
    connectionState?: string,
    requestId?: string
  ) {
    super(message, 'STREAMING_ERROR', undefined, {
      streamId,
      connectionState,
    }, requestId);
    this.streamId = streamId;
    this.connectionState = connectionState;
  }
}

/**
 * WebSocket connection errors
 */
export class WebSocketError extends PynomaliError {
  public readonly wsCode?: number;
  public readonly wsReason?: string;
  public readonly wasClean?: boolean;

  constructor(
    message: string = 'WebSocket error',
    wsCode?: number,
    wsReason?: string,
    wasClean?: boolean
  ) {
    super(message, 'WEBSOCKET_ERROR', undefined, {
      wsCode,
      wsReason,
      wasClean,
    });
    this.wsCode = wsCode;
    this.wsReason = wsReason;
    this.wasClean = wasClean;
  }
}

/**
 * Model training errors
 */
export class TrainingError extends PynomaliError {
  public readonly jobId?: string;
  public readonly phase?: string;
  public readonly progress?: number;

  constructor(
    message: string = 'Training error',
    jobId?: string,
    phase?: string,
    progress?: number,
    requestId?: string
  ) {
    super(message, 'TRAINING_ERROR', undefined, {
      jobId,
      phase,
      progress,
    }, requestId);
    this.jobId = jobId;
    this.phase = phase;
    this.progress = progress;
  }
}

/**
 * SDK version compatibility errors
 */
export class CompatibilityError extends PynomaliError {
  public readonly sdkVersion?: string;
  public readonly apiVersion?: string;
  public readonly minimumSdkVersion?: string;

  constructor(
    message: string = 'Version compatibility error',
    sdkVersion?: string,
    apiVersion?: string,
    minimumSdkVersion?: string
  ) {
    super(message, 'COMPATIBILITY_ERROR', undefined, {
      sdkVersion,
      apiVersion,
      minimumSdkVersion,
    });
    this.sdkVersion = sdkVersion;
    this.apiVersion = apiVersion;
    this.minimumSdkVersion = minimumSdkVersion;
  }
}

/**
 * Error handler utility class
 */
export class ErrorHandler {
  /**
   * Create appropriate error from HTTP response
   */
  static fromHttpResponse(
    status: number,
    data: any,
    requestId?: string
  ): PynomaliError {
    const message = data?.message || data?.error || 'Unknown error';
    const details = data?.details;

    switch (status) {
      case 400:
        return new ValidationError(message, data?.fieldErrors, requestId);
      case 401:
        return new AuthenticationError(message, details, requestId);
      case 403:
        return new AuthorizationError(
          message,
          data?.requiredPermissions,
          data?.userPermissions,
          requestId
        );
      case 404:
        return new NotFoundError(
          message,
          data?.resourceType,
          data?.resourceId,
          requestId
        );
      case 429:
        return new RateLimitError(
          message,
          data?.retryAfter,
          data?.limit,
          data?.remaining,
          data?.resetTime,
          requestId
        );
      case 500:
      case 502:
      case 503:
      case 504:
        return new ServerError(message, status, details, requestId);
      default:
        return new PynomaliError(message, 'HTTP_ERROR', status, details, requestId);
    }
  }

  /**
   * Check if error is retryable
   */
  static isRetryable(error: Error): boolean {
    if (error instanceof NetworkError) {
      return true;
    }
    if (error instanceof ServerError) {
      return error.statusCode !== 501; // Not Implemented is not retryable
    }
    if (error instanceof RateLimitError) {
      return true;
    }
    return false;
  }

  /**
   * Get retry delay for error
   */
  static getRetryDelay(error: Error, attempt: number): number {
    if (error instanceof RateLimitError) {
      return error.retryAfter * 1000;
    }
    
    // Exponential backoff with jitter
    const baseDelay = Math.pow(2, attempt) * 1000;
    const jitter = Math.random() * 0.1 * baseDelay;
    return baseDelay + jitter;
  }

  /**
   * Format error for logging
   */
  static formatForLogging(error: Error): Record<string, any> {
    if (error instanceof PynomaliError) {
      return {
        name: error.name,
        code: error.code,
        message: error.message,
        statusCode: error.statusCode,
        details: error.details,
        requestId: error.requestId,
        timestamp: error.timestamp,
        stack: error.stack,
      };
    }

    return {
      name: error.name,
      message: error.message,
      stack: error.stack,
    };
  }

  /**
   * Check if error indicates authentication is required
   */
  static requiresAuthentication(error: Error): boolean {
    return error instanceof AuthenticationError;
  }

  /**
   * Check if error indicates insufficient permissions
   */
  static requiresAuthorization(error: Error): boolean {
    return error instanceof AuthorizationError;
  }

  /**
   * Extract user-friendly message from error
   */
  static getUserFriendlyMessage(error: Error): string {
    if (error instanceof ValidationError) {
      return 'Please check your input and try again.';
    }
    if (error instanceof AuthenticationError) {
      return 'Please log in to continue.';
    }
    if (error instanceof AuthorizationError) {
      return 'You do not have permission to perform this action.';
    }
    if (error instanceof NetworkError) {
      if (error.isOffline) {
        return 'Please check your internet connection and try again.';
      }
      if (error.isTimeout) {
        return 'The request timed out. Please try again.';
      }
      return 'A network error occurred. Please try again.';
    }
    if (error instanceof RateLimitError) {
      return `Too many requests. Please wait ${error.retryAfter} seconds and try again.`;
    }
    if (error instanceof ServerError) {
      return 'A server error occurred. Please try again later.';
    }
    if (error instanceof NotFoundError) {
      return 'The requested resource was not found.';
    }
    
    return error.message || 'An unexpected error occurred.';
  }
}