/**
 * Pynomaly JavaScript SDK Core Client
 * 
 * Main client class for interacting with Pynomaly API services.
 * Handles authentication, request management, and high-level API access.
 */

import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import { EventEmitter } from 'eventemitter3';
import {
  PynomalyConfig,
  ApiResponse,
  PynomalyError,
  RequestConfig,
  RetryConfig,
  HealthStatus
} from '../types';
import { DataScienceAPI } from './dataScience';
import { PynomalySDKError, AuthenticationError, APIError } from './errors';

/**
 * Main Pynomaly SDK client for web applications.
 * 
 * Provides authenticated access to Pynomaly services with high-level APIs
 * for common anomaly detection workflows in web environments.
 * 
 * @example
 * ```typescript
 * const client = new PynomalyClient({
 *   baseUrl: 'https://api.pynomaly.com',
 *   apiKey: 'your-api-key'
 * });
 * 
 * // Initialize the client
 * await client.initialize();
 * 
 * // Use data science API
 * const detectors = await client.dataScience.listDetectors();
 * const result = await client.dataScience.detectAnomalies('detector-123', data);
 * ```
 */
export class PynomalyClient extends EventEmitter {
  private readonly config: Required<PynomalyConfig>;
  private readonly httpClient: AxiosInstance;
  private readonly retryConfig: RetryConfig;
  private _dataScience: DataScienceAPI | null = null;
  private _isInitialized = false;

  constructor(config: PynomalyConfig) {
    super();
    
    // Set default configuration
    this.config = {
      baseUrl: config.baseUrl.replace(/\/$/, ''),
      apiKey: config.apiKey || '',
      timeout: config.timeout || 30000,
      maxRetries: config.maxRetries || 3,
      debug: config.debug || false
    };

    this.retryConfig = {
      maxRetries: this.config.maxRetries,
      baseDelay: 1000,
      maxDelay: 10000,
      backoffFactor: 2
    };

    // Create HTTP client
    this.httpClient = axios.create({
      baseURL: this.config.baseUrl,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': '@pynomaly/js-sdk/0.1.0'
      }
    });

    // Set authentication header if API key provided
    if (this.config.apiKey) {
      this.httpClient.defaults.headers.common['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    // Add request interceptor for debugging
    if (this.config.debug) {
      this.httpClient.interceptors.request.use(request => {
        console.log('[Pynomaly SDK] Request:', request);
        return request;
      });
    }

    // Add response interceptor for debugging and error handling
    this.httpClient.interceptors.response.use(
      response => {
        if (this.config.debug) {
          console.log('[Pynomaly SDK] Response:', response);
        }
        return response;
      },
      error => {
        if (this.config.debug) {
          console.error('[Pynomaly SDK] Error:', error);
        }
        return Promise.reject(this.transformError(error));
      }
    );
  }

  /**
   * Access to data science API operations.
   */
  get dataScience(): DataScienceAPI {
    if (!this._dataScience) {
      this._dataScience = new DataScienceAPI(this);
    }
    return this._dataScience;
  }

  /**
   * Check if client is initialized.
   */
  get isInitialized(): boolean {
    return this._isInitialized;
  }

  /**
   * Get client configuration.
   */
  get configuration(): PynomalyConfig {
    return { ...this.config };
  }

  /**
   * Initialize the client and verify connectivity.
   * 
   * @returns Promise resolving to health status
   */
  async initialize(): Promise<HealthStatus> {
    try {
      const health = await this.healthCheck();
      this._isInitialized = true;
      this.emit('initialized', { health });
      return health;
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Check API health status.
   * 
   * @returns Promise resolving to health status information
   */
  async healthCheck(): Promise<HealthStatus> {
    try {
      const response = await this.request({
        method: 'GET',
        url: '/health'
      });
      return response.data;
    } catch (error) {
      throw new APIError('Health check failed', { originalError: error });
    }
  }

  /**
   * Update the API key for authentication.
   * 
   * @param apiKey New API key
   */
  setApiKey(apiKey: string): void {
    this.config.apiKey = apiKey;
    this.httpClient.defaults.headers.common['Authorization'] = `Bearer ${apiKey}`;
  }

  /**
   * Make an authenticated API request with retry logic.
   * 
   * @param config Request configuration
   * @returns Promise resolving to API response
   */
  async request<T = any>(config: RequestConfig): Promise<ApiResponse<T>> {
    const url = config.url.startsWith('/api/v1') ? config.url : `/api/v1${config.url}`;
    
    for (let attempt = 0; attempt <= this.retryConfig.maxRetries; attempt++) {
      try {
        const response: AxiosResponse = await this.httpClient.request({
          method: config.method,
          url,
          data: config.data,
          params: config.params,
          headers: config.headers,
          timeout: config.timeout || this.config.timeout
        });

        return {
          statusCode: response.status,
          data: response.data,
          headers: response.headers as Record<string, string>,
          success: response.status >= 200 && response.status < 300
        };

      } catch (error) {
        const isLastAttempt = attempt === this.retryConfig.maxRetries;
        const shouldRetry = this.shouldRetry(error as AxiosError, attempt);

        if (isLastAttempt || !shouldRetry) {
          throw this.transformError(error as AxiosError);
        }

        // Wait before retrying with exponential backoff
        const delay = Math.min(
          this.retryConfig.baseDelay * Math.pow(this.retryConfig.backoffFactor, attempt),
          this.retryConfig.maxDelay
        );
        
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    // Should never reach here, but just in case
    throw new APIError(`Max retries (${this.retryConfig.maxRetries}) exceeded`);
  }

  /**
   * Upload files to the API.
   * 
   * @param url Upload endpoint
   * @param files Files to upload
   * @param data Additional form data
   * @returns Promise resolving to API response
   */
  async uploadFiles<T = any>(
    url: string,
    files: File | File[],
    data?: Record<string, any>
  ): Promise<ApiResponse<T>> {
    const formData = new FormData();
    
    // Add files
    const fileArray = Array.isArray(files) ? files : [files];
    fileArray.forEach((file, index) => {
      formData.append(`file${fileArray.length > 1 ? index : ''}`, file);
    });

    // Add additional data
    if (data) {
      Object.entries(data).forEach(([key, value]) => {
        formData.append(key, typeof value === 'string' ? value : JSON.stringify(value));
      });
    }

    return this.request({
      method: 'POST',
      url,
      data: formData,
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
  }

  /**
   * Determine if a request should be retried.
   */
  private shouldRetry(error: AxiosError, attempt: number): boolean {
    if (attempt >= this.retryConfig.maxRetries) {
      return false;
    }

    // Don't retry client errors (4xx) except 408 and 429
    if (error.response) {
      const status = error.response.status;
      if (status >= 400 && status < 500 && status !== 408 && status !== 429) {
        return false;
      }
    }

    // Retry on network errors, timeouts, and server errors
    return (
      !error.response || // Network error
      error.code === 'ECONNABORTED' || // Timeout
      (error.response.status >= 500) || // Server error
      (error.response.status === 408) || // Request timeout
      (error.response.status === 429) // Rate limit
    );
  }

  /**
   * Transform axios errors to SDK errors.
   */
  private transformError(error: AxiosError): PynomalyError {
    if (error.response) {
      const status = error.response.status;
      const data = error.response.data as any;
      
      if (status === 401) {
        return new AuthenticationError('Invalid API key or authentication failed', {
          statusCode: status,
          responseData: data
        });
      }

      if (status >= 400 && status < 500) {
        const message = data?.detail || data?.message || `Client error (${status})`;
        return new APIError(message, {
          statusCode: status,
          responseData: data
        });
      }

      if (status >= 500) {
        const message = data?.detail || data?.message || `Server error (${status})`;
        return new APIError(message, {
          statusCode: status,
          responseData: data
        });
      }
    }

    if (error.code === 'ECONNABORTED') {
      return new APIError(`Request timeout after ${this.config.timeout}ms`, {
        originalError: error
      });
    }

    if (error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED') {
      return new APIError('Connection error: Unable to reach Pynomaly API', {
        originalError: error
      });
    }

    return new PynomalySDKError(`Unexpected error: ${error.message}`, {
      originalError: error
    });
  }

  /**
   * Clean up resources.
   */
  dispose(): void {
    this.removeAllListeners();
    this._dataScience = null;
    this._isInitialized = false;
  }
}

/**
 * Create a Pynomaly client with default configuration.
 * 
 * @param config Client configuration
 * @returns Configured PynomalyClient instance
 */
export function createClient(config: PynomalyConfig): PynomalyClient {
  return new PynomalyClient(config);
}