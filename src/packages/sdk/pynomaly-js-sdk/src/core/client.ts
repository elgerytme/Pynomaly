/**
 * Core Pynomaly client implementation
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { EventEmitter } from 'eventemitter3';
import {
  PynomalyConfig,
  ApiResponse,
  AuthToken,
  User,
  AnomalyDetectionRequest,
  AnomalyDetectionResult,
  DataQualityRequest,
  DataQualityResult,
  DataProfilingRequest,
  DataProfilingResult,
  BatchRequest,
  BatchResult,
  JobStatus,
  PynomalyError,
  RequestOptions,
  PaginationOptions,
  PaginatedResponse,
  EventMap,
  EventCallback
} from '../types';

export class PynomalyClient extends EventEmitter<EventMap> {
  private http: AxiosInstance;
  private config: PynomalyConfig;
  private authToken: AuthToken | null = null;
  private refreshTimer: NodeJS.Timeout | null = null;

  constructor(config: PynomalyConfig) {
    super();
    this.config = {
      timeout: 30000,
      retryAttempts: 3,
      enableWebSocket: true,
      debug: false,
      version: 'v1',
      ...config
    };

    this.http = axios.create({
      baseURL: `${this.config.baseUrl}/api/${this.config.version}`,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': `pynomaly-js-sdk/1.0.0`
      }
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.http.interceptors.request.use(
      (config) => {
        if (this.authToken) {
          config.headers.Authorization = `${this.authToken.tokenType} ${this.authToken.token}`;
        } else if (this.config.apiKey) {
          config.headers['X-API-Key'] = this.config.apiKey;
        }

        if (this.config.debug) {
          console.log('Request:', config);
        }

        return config;
      },
      (error) => {
        if (this.config.debug) {
          console.error('Request error:', error);
        }
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.http.interceptors.response.use(
      (response) => {
        if (this.config.debug) {
          console.log('Response:', response);
        }
        return response;
      },
      async (error) => {
        if (this.config.debug) {
          console.error('Response error:', error);
        }

        const originalRequest = error.config;

        // Handle token refresh
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            await this.refreshToken();
            return this.http(originalRequest);
          } catch (refreshError) {
            this.emit('connection:error', refreshError as Error);
            throw refreshError;
          }
        }

        // Handle rate limiting
        if (error.response?.status === 429) {
          const retryAfter = error.response.headers['retry-after'] || 1;
          await this.delay(retryAfter * 1000);
          return this.http(originalRequest);
        }

        throw this.createError(error);
      }
    );
  }

  private createError(error: any): PynomalyError {
    const pynomalyError: PynomalyError = {
      code: error.response?.data?.code || 'UNKNOWN_ERROR',
      message: error.response?.data?.message || error.message || 'Unknown error',
      details: error.response?.data?.details,
      requestId: error.response?.headers['x-request-id'],
      timestamp: new Date()
    };

    return pynomalyError;
  }

  private async delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Authentication methods
  async authenticate(credentials: { email: string; password: string }): Promise<User> {
    const response = await this.http.post<ApiResponse<{ user: User; token: AuthToken }>>('/auth/login', credentials);
    
    if (!response.data.success || !response.data.data) {
      throw new Error('Authentication failed');
    }

    this.authToken = response.data.data.token;
    this.setupTokenRefresh();
    
    return response.data.data.user;
  }

  async authenticateWithApiKey(apiKey: string): Promise<User> {
    const tempConfig = { ...this.config, apiKey };
    const tempClient = axios.create({
      baseURL: `${this.config.baseUrl}/api/${this.config.version}`,
      headers: { 'X-API-Key': apiKey }
    });

    const response = await tempClient.get<ApiResponse<User>>('/auth/me');
    
    if (!response.data.success || !response.data.data) {
      throw new Error('API key authentication failed');
    }

    this.config.apiKey = apiKey;
    return response.data.data;
  }

  async refreshToken(): Promise<void> {
    if (!this.authToken?.refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await this.http.post<ApiResponse<AuthToken>>('/auth/refresh', {
      refreshToken: this.authToken.refreshToken
    });

    if (!response.data.success || !response.data.data) {
      throw new Error('Token refresh failed');
    }

    this.authToken = response.data.data;
    this.setupTokenRefresh();
  }

  private setupTokenRefresh(): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
    }

    if (this.authToken) {
      const expiresIn = this.authToken.expiresAt.getTime() - Date.now();
      const refreshTime = Math.max(expiresIn - 60000, 0); // Refresh 1 minute before expiry

      this.refreshTimer = setTimeout(async () => {
        try {
          await this.refreshToken();
        } catch (error) {
          this.emit('connection:error', error as Error);
        }
      }, refreshTime);
    }
  }

  async logout(): Promise<void> {
    if (this.authToken) {
      await this.http.post('/auth/logout');
      this.authToken = null;
      
      if (this.refreshTimer) {
        clearTimeout(this.refreshTimer);
        this.refreshTimer = null;
      }
    }
  }

  // Anomaly Detection
  async detectAnomalies(request: AnomalyDetectionRequest): Promise<AnomalyDetectionResult> {
    const response = await this.http.post<ApiResponse<AnomalyDetectionResult>>('/anomaly/detect', request);
    
    if (!response.data.success || !response.data.data) {
      throw new Error('Anomaly detection failed');
    }

    return response.data.data;
  }

  async detectAnomaliesAsync(request: AnomalyDetectionRequest): Promise<string> {
    const response = await this.http.post<ApiResponse<{ jobId: string }>>('/anomaly/detect/async', request);
    
    if (!response.data.success || !response.data.data) {
      throw new Error('Async anomaly detection failed');
    }

    return response.data.data.jobId;
  }

  // Data Quality
  async analyzeDataQuality(request: DataQualityRequest): Promise<DataQualityResult> {
    const response = await this.http.post<ApiResponse<DataQualityResult>>('/quality/analyze', request);
    
    if (!response.data.success || !response.data.data) {
      throw new Error('Data quality analysis failed');
    }

    return response.data.data;
  }

  async analyzeDataQualityAsync(request: DataQualityRequest): Promise<string> {
    const response = await this.http.post<ApiResponse<{ jobId: string }>>('/quality/analyze/async', request);
    
    if (!response.data.success || !response.data.data) {
      throw new Error('Async data quality analysis failed');
    }

    return response.data.data.jobId;
  }

  // Data Profiling
  async profileData(request: DataProfilingRequest): Promise<DataProfilingResult> {
    const response = await this.http.post<ApiResponse<DataProfilingResult>>('/profiling/profile', request);
    
    if (!response.data.success || !response.data.data) {
      throw new Error('Data profiling failed');
    }

    return response.data.data;
  }

  async profileDataAsync(request: DataProfilingRequest): Promise<string> {
    const response = await this.http.post<ApiResponse<{ jobId: string }>>('/profiling/profile/async', request);
    
    if (!response.data.success || !response.data.data) {
      throw new Error('Async data profiling failed');
    }

    return response.data.data.jobId;
  }

  // Batch Operations
  async processBatch(request: BatchRequest): Promise<BatchResult> {
    const response = await this.http.post<ApiResponse<BatchResult>>('/batch/process', request);
    
    if (!response.data.success || !response.data.data) {
      throw new Error('Batch processing failed');
    }

    return response.data.data;
  }

  async processBatchAsync(request: BatchRequest): Promise<string> {
    const response = await this.http.post<ApiResponse<{ jobId: string }>>('/batch/process/async', request);
    
    if (!response.data.success || !response.data.data) {
      throw new Error('Async batch processing failed');
    }

    return response.data.data.jobId;
  }

  // Job Management
  async getJobStatus(jobId: string): Promise<JobStatus> {
    const response = await this.http.get<ApiResponse<JobStatus>>(`/jobs/${jobId}`);
    
    if (!response.data.success || !response.data.data) {
      throw new Error('Failed to get job status');
    }

    return response.data.data;
  }

  async cancelJob(jobId: string): Promise<void> {
    await this.http.delete(`/jobs/${jobId}`);
  }

  async getJobResult(jobId: string): Promise<any> {
    const response = await this.http.get<ApiResponse<any>>(`/jobs/${jobId}/result`);
    
    if (!response.data.success || !response.data.data) {
      throw new Error('Failed to get job result');
    }

    return response.data.data;
  }

  // Pagination helpers
  async getWithPagination<T>(
    endpoint: string,
    options: PaginationOptions = {}
  ): Promise<PaginatedResponse<T>> {
    const params = new URLSearchParams();
    
    if (options.page) params.append('page', options.page.toString());
    if (options.pageSize) params.append('pageSize', options.pageSize.toString());
    if (options.sortBy) params.append('sortBy', options.sortBy);
    if (options.sortOrder) params.append('sortOrder', options.sortOrder);

    const response = await this.http.get<ApiResponse<PaginatedResponse<T>>>(
      `${endpoint}?${params.toString()}`
    );
    
    if (!response.data.success || !response.data.data) {
      throw new Error('Failed to fetch paginated data');
    }

    return response.data.data;
  }

  // Generic request method
  async request<T>(
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<T> {
    const config: AxiosRequestConfig = {
      method: options.method || 'GET',
      url: endpoint,
      headers: options.headers,
      timeout: options.timeout,
      data: options.data
    };

    let attempt = 0;
    const maxAttempts = options.retries || this.config.retryAttempts || 3;

    while (attempt < maxAttempts) {
      try {
        const response = await this.http(config);
        
        if (response.data.success) {
          return response.data.data;
        } else {
          throw new Error(response.data.error || 'Request failed');
        }
      } catch (error) {
        attempt++;
        
        if (attempt >= maxAttempts) {
          throw error;
        }

        // Exponential backoff
        await this.delay(Math.pow(2, attempt) * 1000);
      }
    }

    throw new Error('Max retry attempts exceeded');
  }

  // Health check
  async healthCheck(): Promise<{ status: string; timestamp: Date }> {
    const response = await this.http.get<ApiResponse<{ status: string; timestamp: string }>>('/health');
    
    if (!response.data.success || !response.data.data) {
      throw new Error('Health check failed');
    }

    return {
      status: response.data.data.status,
      timestamp: new Date(response.data.data.timestamp)
    };
  }

  // Event methods
  on<K extends keyof EventMap>(event: K, callback: EventCallback<EventMap[K]>): this {
    return super.on(event, callback);
  }

  off<K extends keyof EventMap>(event: K, callback: EventCallback<EventMap[K]>): this {
    return super.off(event, callback);
  }

  once<K extends keyof EventMap>(event: K, callback: EventCallback<EventMap[K]>): this {
    return super.once(event, callback);
  }

  emit<K extends keyof EventMap>(event: K, ...args: Parameters<EventCallback<EventMap[K]>>): boolean {
    return super.emit(event, ...args);
  }

  // Auth token getter for AuthManager
  getAuthToken(): AuthToken | null {
    return this.authToken;
  }

  // Cleanup
  destroy(): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
      this.refreshTimer = null;
    }

    this.removeAllListeners();
    this.authToken = null;
  }
}