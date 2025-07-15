/**
 * Pynomaly JavaScript SDK Error Classes
 * 
 * Custom error classes for the Pynomaly SDK with specific error handling
 * for different types of failures.
 */

import { PynomalyError } from '../types';

/**
 * Base error class for all Pynomaly SDK errors.
 */
export class PynomalySDKError extends Error implements PynomalyError {
  public readonly details: Record<string, any>;
  public readonly statusCode?: number;
  public readonly responseData?: Record<string, any>;

  constructor(
    message: string,
    details: Record<string, any> = {}
  ) {
    super(message);
    this.name = 'PynomalySDKError';
    this.details = details;
    this.statusCode = details.statusCode;
    this.responseData = details.responseData;

    // Maintain proper stack trace for where our error was thrown (only available on V8)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, PynomalySDKError);
    }
  }

  /**
   * Convert error to JSON representation.
   */
  toJSON(): Record<string, any> {
    return {
      name: this.name,
      message: this.message,
      details: this.details,
      statusCode: this.statusCode,
      responseData: this.responseData,
      stack: this.stack
    };
  }

  /**
   * Get a user-friendly error message.
   */
  getUserMessage(): string {
    return this.message;
  }
}

/**
 * Error thrown when authentication fails.
 */
export class AuthenticationError extends PynomalySDKError {
  constructor(
    message: string = 'Authentication failed',
    details: Record<string, any> = {}
  ) {
    super(message, details);
    this.name = 'AuthenticationError';
  }

  getUserMessage(): string {
    return 'Authentication failed. Please check your API key and try again.';
  }
}

/**
 * Error thrown when request validation fails.
 */
export class ValidationError extends PynomalySDKError {
  public readonly field?: string;

  constructor(
    message: string,
    field?: string,
    details: Record<string, any> = {}
  ) {
    super(message, details);
    this.name = 'ValidationError';
    this.field = field;
  }

  getUserMessage(): string {
    if (this.field) {
      return `Validation error in field '${this.field}': ${this.message}`;
    }
    return `Validation error: ${this.message}`;
  }
}

/**
 * Error thrown when API request fails.
 */
export class APIError extends PynomalySDKError {
  constructor(
    message: string,
    details: Record<string, any> = {}
  ) {
    super(message, details);
    this.name = 'APIError';
  }

  getUserMessage(): string {
    if (this.statusCode) {
      if (this.statusCode >= 500) {
        return 'Server error occurred. Please try again later.';
      }
      if (this.statusCode === 404) {
        return 'The requested resource was not found.';
      }
      if (this.statusCode === 429) {
        return 'Too many requests. Please wait before trying again.';
      }
    }
    return this.message;
  }
}

/**
 * Error thrown when a requested resource is not found.
 */
export class ResourceNotFoundError extends APIError {
  public readonly resourceType: string;
  public readonly resourceId: string;

  constructor(
    resourceType: string,
    resourceId: string,
    details: Record<string, any> = {}
  ) {
    const message = `${resourceType} with ID '${resourceId}' not found`;
    super(message, { ...details, statusCode: 404 });
    this.name = 'ResourceNotFoundError';
    this.resourceType = resourceType;
    this.resourceId = resourceId;
  }

  getUserMessage(): string {
    return `The requested ${this.resourceType.toLowerCase()} could not be found.`;
  }
}

/**
 * Error thrown when API rate limit is exceeded.
 */
export class RateLimitError extends APIError {
  public readonly retryAfter?: number;

  constructor(
    retryAfter?: number,
    details: Record<string, any> = {}
  ) {
    let message = 'API rate limit exceeded';
    if (retryAfter) {
      message += `, retry after ${retryAfter} seconds`;
    }
    super(message, { ...details, statusCode: 429 });
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }

  getUserMessage(): string {
    if (this.retryAfter) {
      return `Rate limit exceeded. Please wait ${this.retryAfter} seconds before trying again.`;
    }
    return 'Rate limit exceeded. Please wait before trying again.';
  }
}

/**
 * Error thrown when there are issues with data processing.
 */
export class DataError extends PynomalySDKError {
  public readonly dataInfo: Record<string, any>;

  constructor(
    message: string,
    dataInfo: Record<string, any> = {}
  ) {
    super(message, dataInfo);
    this.name = 'DataError';
    this.dataInfo = dataInfo;
  }

  getUserMessage(): string {
    return `Data processing error: ${this.message}`;
  }
}

/**
 * Error thrown when there are issues with model operations.
 */
export class ModelError extends PynomalySDKError {
  public readonly modelInfo: Record<string, any>;

  constructor(
    message: string,
    modelInfo: Record<string, any> = {}
  ) {
    super(message, modelInfo);
    this.name = 'ModelError';
    this.modelInfo = modelInfo;
  }

  getUserMessage(): string {
    return `Model error: ${this.message}`;
  }
}

/**
 * Error thrown when there are configuration issues.
 */
export class ConfigurationError extends PynomalySDKError {
  public readonly configKey?: string;

  constructor(
    message: string,
    configKey?: string,
    details: Record<string, any> = {}
  ) {
    super(message, details);
    this.name = 'ConfigurationError';
    this.configKey = configKey;
  }

  getUserMessage(): string {
    if (this.configKey) {
      return `Configuration error in '${this.configKey}': ${this.message}`;
    }
    return `Configuration error: ${this.message}`;
  }
}

/**
 * Error thrown when network operations fail.
 */
export class NetworkError extends PynomalySDKError {
  constructor(
    message: string,
    details: Record<string, any> = {}
  ) {
    super(message, details);
    this.name = 'NetworkError';
  }

  getUserMessage(): string {
    return 'Network error occurred. Please check your connection and try again.';
  }
}

/**
 * Error thrown when WebSocket operations fail.
 */
export class WebSocketError extends PynomalySDKError {
  constructor(
    message: string,
    details: Record<string, any> = {}
  ) {
    super(message, details);
    this.name = 'WebSocketError';
  }

  getUserMessage(): string {
    return `WebSocket error: ${this.message}`;
  }
}

/**
 * Utility function to check if an error is a Pynomaly SDK error.
 */
export function isPynomalyError(error: any): error is PynomalySDKError {
  return error instanceof PynomalySDKError;
}

/**
 * Utility function to get a user-friendly error message from any error.
 */
export function getErrorMessage(error: any): string {
  if (isPynomalyError(error)) {
    return error.getUserMessage();
  }
  
  if (error instanceof Error) {
    return error.message;
  }
  
  if (typeof error === 'string') {
    return error;
  }
  
  return 'An unexpected error occurred';
}

/**
 * Utility function to extract error details for logging.
 */
export function getErrorDetails(error: any): Record<string, any> {
  if (isPynomalyError(error)) {
    return error.toJSON();
  }
  
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack
    };
  }
  
  return { error: String(error) };
}