/**
 * Error classes for Pynomaly SDK
 */

export class PynomalyError extends Error {
  public readonly code: string;
  public readonly statusCode?: number;
  public readonly details?: Record<string, any>;

  constructor(
    message: string,
    code: string = 'UNKNOWN_ERROR',
    statusCode?: number,
    details?: Record<string, any>
  ) {
    super(message);
    this.name = 'PynomalyError';
    this.code = code;
    this.statusCode = statusCode;
    this.details = details;

    // Maintains proper stack trace for where our error was thrown (only available on V8)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, PynomalyError);
    }
  }

  toJSON() {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      statusCode: this.statusCode,
      details: this.details,
      stack: this.stack
    };
  }
}

export class AuthenticationError extends PynomalyError {
  constructor(message: string = 'Authentication failed', details?: Record<string, any>) {
    super(message, 'AUTHENTICATION_ERROR', 401, details);
    this.name = 'AuthenticationError';
  }
}

export class AuthorizationError extends PynomalyError {
  constructor(message: string = 'Access denied', details?: Record<string, any>) {
    super(message, 'AUTHORIZATION_ERROR', 403, details);
    this.name = 'AuthorizationError';
  }
}

export class ValidationError extends PynomalyError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, 'VALIDATION_ERROR', 400, details);
    this.name = 'ValidationError';
  }
}

export class NotFoundError extends PynomalyError {
  constructor(resource: string, id?: string) {
    const message = id ? `${resource} with id '${id}' not found` : `${resource} not found`;
    super(message, 'NOT_FOUND_ERROR', 404);
    this.name = 'NotFoundError';
  }
}

export class ConflictError extends PynomalyError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, 'CONFLICT_ERROR', 409, details);
    this.name = 'ConflictError';
  }
}

export class RateLimitError extends PynomalyError {
  constructor(message: string = 'Rate limit exceeded', retryAfter?: number) {
    super(message, 'RATE_LIMIT_ERROR', 429, { retryAfter });
    this.name = 'RateLimitError';
  }
}

export class ServerError extends PynomalyError {
  constructor(message: string = 'Internal server error', details?: Record<string, any>) {
    super(message, 'SERVER_ERROR', 500, details);
    this.name = 'ServerError';
  }
}

export class NetworkError extends PynomalyError {
  constructor(message: string = 'Network error', details?: Record<string, any>) {
    super(message, 'NETWORK_ERROR', undefined, details);
    this.name = 'NetworkError';
  }
}

export class TimeoutError extends PynomalyError {
  constructor(message: string = 'Request timeout', timeout?: number) {
    super(message, 'TIMEOUT_ERROR', 408, { timeout });
    this.name = 'TimeoutError';
  }
}

export class StreamingError extends PynomalyError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, 'STREAMING_ERROR', undefined, details);
    this.name = 'StreamingError';
  }
}

export class TestingError extends PynomalyError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, 'TESTING_ERROR', undefined, details);
    this.name = 'TestingError';
  }
}

export class ComplianceError extends PynomalyError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, 'COMPLIANCE_ERROR', undefined, details);
    this.name = 'ComplianceError';
  }
}

/**
 * Create appropriate error from HTTP response
 */
export function createErrorFromResponse(
  status: number,
  data: any,
  message?: string
): PynomalyError {
  const errorMessage = message || data?.message || data?.error || 'Request failed';
  const errorCode = data?.code;
  const details = data?.details;

  switch (status) {
    case 400:
      return new ValidationError(errorMessage, details);
    case 401:
      return new AuthenticationError(errorMessage, details);
    case 403:
      return new AuthorizationError(errorMessage, details);
    case 404:
      return new NotFoundError(errorMessage);
    case 409:
      return new ConflictError(errorMessage, details);
    case 429:
      return new RateLimitError(errorMessage, data?.retryAfter);
    case 500:
    case 502:
    case 503:
    case 504:
      return new ServerError(errorMessage, details);
    case 408:
      return new TimeoutError(errorMessage, details);
    default:
      return new PynomalyError(errorMessage, errorCode || 'HTTP_ERROR', status, details);
  }
}

/**
 * Check if error is retryable
 */
export function isRetryableError(error: PynomalyError | Error): boolean {
  if (error instanceof PynomalyError) {
    return [
      'NETWORK_ERROR',
      'TIMEOUT_ERROR',
      'SERVER_ERROR',
      'RATE_LIMIT_ERROR'
    ].includes(error.code) || (error.statusCode && error.statusCode >= 500);
  }
  
  // Handle network errors from axios or fetch
  return error.message.includes('network') || 
         error.message.includes('timeout') ||
         error.message.includes('ECONNRESET') ||
         error.message.includes('ENOTFOUND');
}