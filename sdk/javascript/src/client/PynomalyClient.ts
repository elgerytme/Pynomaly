/**
 * Main Pynomaly SDK client
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { PynomalyConfig, AuthConfig, ApiResponse, HealthStatus } from '../types';
import { PynomalyError, createErrorFromResponse, isRetryableError } from '../errors';
import { DetectionClient } from './DetectionClient';
import { StreamingClient } from './StreamingClient';
import { ABTestingClient } from './ABTestingClient';
import { UserManagementClient } from './UserManagementClient';
import { ComplianceClient } from './ComplianceClient';

export class PynomalyClient {
  private readonly httpClient: AxiosInstance;
  private readonly config: Required<PynomalyConfig>;

  // Sub-clients for different API areas
  public readonly detection: DetectionClient;
  public readonly streaming: StreamingClient;
  public readonly abTesting: ABTestingClient;
  public readonly users: UserManagementClient;
  public readonly compliance: ComplianceClient;

  constructor(config: PynomalyConfig) {
    // Set default configuration
    this.config = {
      baseUrl: config.baseUrl,
      apiKey: config.apiKey || '',
      tenantId: config.tenantId || '',
      timeout: config.timeout || 30000,
      retryAttempts: config.retryAttempts || 3,
      retryDelay: config.retryDelay || 1000
    };

    // Validate required configuration
    if (!this.config.baseUrl) {
      throw new PynomalyError('baseUrl is required in configuration', 'INVALID_CONFIG');
    }

    // Create HTTP client
    this.httpClient = this.createHttpClient();

    // Initialize sub-clients
    this.detection = new DetectionClient(this.httpClient);
    this.streaming = new StreamingClient(this.httpClient);
    this.abTesting = new ABTestingClient(this.httpClient);
    this.users = new UserManagementClient(this.httpClient);
    this.compliance = new ComplianceClient(this.httpClient);
  }

  /**
   * Create configured HTTP client
   */
  private createHttpClient(): AxiosInstance {
    const client = axios.create({
      baseURL: this.config.baseUrl,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': `pynomaly-js-sdk/1.0.0`,
        ...(this.config.apiKey && { 'Authorization': `Bearer ${this.config.apiKey}` }),
        ...(this.config.tenantId && { 'X-Tenant-ID': this.config.tenantId })
      }
    });

    // Request interceptor for common headers
    client.interceptors.request.use(
      (config) => {
        // Add timestamp for debugging
        config.headers['X-Request-ID'] = this.generateRequestId();
        config.headers['X-Request-Timestamp'] = new Date().toISOString();
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling and retries
    client.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;

        // Don't retry if we've already retried max times
        if (originalRequest._retryCount >= this.config.retryAttempts) {
          return Promise.reject(this.handleError(error));
        }

        // Check if error is retryable
        if (isRetryableError(error)) {
          originalRequest._retryCount = (originalRequest._retryCount || 0) + 1;

          // Calculate delay with exponential backoff
          const delay = this.config.retryDelay * Math.pow(2, originalRequest._retryCount - 1);
          
          await this.sleep(delay);
          return client(originalRequest);
        }

        return Promise.reject(this.handleError(error));
      }
    );

    return client;
  }

  /**
   * Handle HTTP errors and convert to Pynomaly errors
   */
  private handleError(error: any): PynomalyError {
    if (error.response) {
      // HTTP error response
      return createErrorFromResponse(
        error.response.status,
        error.response.data,
        error.message
      );
    } else if (error.request) {
      // Network error
      return new PynomalyError(
        'Network error: No response received',
        'NETWORK_ERROR',
        undefined,
        { originalError: error.message }
      );
    } else {
      // Other error
      return new PynomalyError(
        error.message || 'Unknown error',
        'UNKNOWN_ERROR',
        undefined,
        { originalError: error }
      );
    }
  }

  /**
   * Generate unique request ID
   */
  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Sleep utility for retries
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Set authentication credentials
   */
  public setAuth(auth: AuthConfig): void {
    this.config.apiKey = auth.apiKey;
    if (auth.tenantId) {
      this.config.tenantId = auth.tenantId;
    }

    // Update HTTP client headers
    this.httpClient.defaults.headers['Authorization'] = `Bearer ${auth.apiKey}`;
    if (auth.tenantId) {
      this.httpClient.defaults.headers['X-Tenant-ID'] = auth.tenantId;
    }
  }

  /**
   * Clear authentication
   */
  public clearAuth(): void {
    this.config.apiKey = '';
    this.config.tenantId = '';
    delete this.httpClient.defaults.headers['Authorization'];
    delete this.httpClient.defaults.headers['X-Tenant-ID'];
  }

  /**
   * Get current configuration
   */
  public getConfig(): Readonly<PynomalyConfig> {
    return { ...this.config };
  }

  /**
   * Update configuration
   */
  public updateConfig(updates: Partial<PynomalyConfig>): void {
    Object.assign(this.config, updates);

    // Update HTTP client if needed
    if (updates.baseUrl) {
      this.httpClient.defaults.baseURL = updates.baseUrl;
    }
    if (updates.timeout) {
      this.httpClient.defaults.timeout = updates.timeout;
    }
    if (updates.apiKey) {
      this.httpClient.defaults.headers['Authorization'] = `Bearer ${updates.apiKey}`;
    }
    if (updates.tenantId) {
      this.httpClient.defaults.headers['X-Tenant-ID'] = updates.tenantId;
    }
  }

  /**
   * Make a raw HTTP request
   */
  public async request<T = any>(config: AxiosRequestConfig): Promise<ApiResponse<T>> {
    try {
      const response: AxiosResponse<T> = await this.httpClient(config);
      return {
        success: true,
        data: response.data,
        metadata: {
          status: response.status,
          headers: response.headers
        }
      };
    } catch (error) {
      throw error instanceof PynomalyError ? error : this.handleError(error);
    }
  }

  /**
   * Health check
   */
  public async healthCheck(): Promise<HealthStatus> {
    try {
      const response = await this.httpClient.get<HealthStatus>('/health');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get API version information
   */
  public async getVersion(): Promise<{ version: string; build?: string; environment?: string }> {
    try {
      const response = await this.httpClient.get('/version');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Test authentication
   */
  public async testAuth(): Promise<{ authenticated: boolean; user?: any; tenant?: any }> {
    try {
      const response = await this.httpClient.get('/auth/test');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get available algorithms
   */
  public async getAvailableAlgorithms(): Promise<string[]> {
    try {
      const response = await this.httpClient.get('/algorithms');
      return response.data.algorithms || [];
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get algorithm information
   */
  public async getAlgorithmInfo(algorithm: string): Promise<any> {
    try {
      const response = await this.httpClient.get(`/algorithms/${algorithm}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Upload file (for datasets)
   */
  public async uploadFile(
    file: File | Buffer,
    filename: string,
    contentType?: string
  ): Promise<{ file_id: string; url: string }> {
    try {
      const formData = new FormData();
      
      if (file instanceof File) {
        formData.append('file', file, filename);
      } else {
        // Handle Buffer for Node.js
        const blob = new Blob([file], { type: contentType || 'application/octet-stream' });
        formData.append('file', blob, filename);
      }

      const response = await this.httpClient.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Download file
   */
  public async downloadFile(fileId: string): Promise<ArrayBuffer> {
    try {
      const response = await this.httpClient.get(`/download/${fileId}`, {
        responseType: 'arraybuffer'
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Close the client and clean up resources
   */
  public close(): void {
    // Clear any pending requests
    // Note: Axios doesn't have a built-in close method, but we can clear auth
    this.clearAuth();
  }

  /**
   * Create a new client instance with different configuration
   */
  public static create(config: PynomalyConfig): PynomalyClient {
    return new PynomalyClient(config);
  }

  /**
   * Get the underlying HTTP client for advanced usage
   */
  public getHttpClient(): AxiosInstance {
    return this.httpClient;
  }
}