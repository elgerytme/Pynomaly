/**
 * Pynomaly TypeScript Client Implementation
 *
 * This module provides the main client class for interacting with the Pynomaly API.
 * Includes comprehensive functionality with TypeScript type safety.
 */

import { AuthManager } from './auth';
import { RateLimiter } from './rate-limiter';
import {
    PynomaliError,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    ServerError,
    NetworkError,
    RateLimitError
} from './errors';
import {
    ClientConfig,
    DetectionRequest,
    DetectionResponse,
    TrainingRequest,
    TrainingResponse,
    AuthToken,
    ModelInfo,
    DatasetInfo,
    HealthStatus,
    HttpMethod,
    RequestOptions
} from './types';

/**
 * Main Pynomali client for TypeScript/JavaScript applications.
 */
export class PynomaliClient {
    private readonly baseUrl: string;
    private readonly timeout: number;
    private readonly maxRetries: number;
    private readonly userAgent: string;
    private readonly authManager: AuthManager;
    private readonly rateLimiter: RateLimiter;

    // API modules
    public readonly auth: AuthAPI;
    public readonly detection: DetectionAPI;
    public readonly training: TrainingAPI;
    public readonly datasets: DatasetsAPI;
    public readonly models: ModelsAPI;
    public readonly streaming: StreamingAPI;
    public readonly explainability: ExplainabilityAPI;
    public readonly health: HealthAPI;

    constructor(config: ClientConfig = {}) {
        this.baseUrl = (config.baseUrl || 'https://api.pynomaly.com').replace(/\/$/, '');
        this.timeout = config.timeout || 30000;
        this.maxRetries = config.maxRetries || 3;
        this.userAgent = config.userAgent || `pynomaly-typescript-sdk/1.0.0`;

        // Initialize auth manager
        this.authManager = new AuthManager(config.apiKey);

        // Initialize rate limiter
        this.rateLimiter = new RateLimiter(
            config.rateLimitRequests || 100,
            config.rateLimitPeriod || 60000
        );

        // Initialize API modules
        this.auth = new AuthAPI(this);
        this.detection = new DetectionAPI(this);
        this.training = new TrainingAPI(this);
        this.datasets = new DatasetsAPI(this);
        this.models = new ModelsAPI(this);
        this.streaming = new StreamingAPI(this);
        this.explainability = new ExplainabilityAPI(this);
        this.health = new HealthAPI(this);
    }

    /**
     * Make HTTP request with error handling and rate limiting.
     */
    public async request<T = any>(
        method: HttpMethod,
        endpoint: string,
        options: RequestOptions = {}
    ): Promise<T> {
        // Rate limiting
        await this.rateLimiter.waitIfNeeded();

        const url = this.buildUrl(endpoint);
        const headers = this.getHeaders(options.headers);
        const requestTimeout = options.timeout || this.timeout;

        console.debug(`Making ${method} request to ${url}`);

        const requestOptions: RequestInit = {
            method,
            headers,
            signal: AbortSignal.timeout(requestTimeout)
        };

        if (options.data && method !== 'GET') {
            requestOptions.body = JSON.stringify(options.data);
        }

        try {
            const response = await this.fetchWithRetry(url, requestOptions);
            return await this.handleResponse<T>(response);
        } catch (error) {
            if (error instanceof DOMException && error.name === 'TimeoutError') {
                throw new NetworkError(`Request timeout after ${requestTimeout}ms`);
            }
            if (error instanceof TypeError) {
                throw new NetworkError('Network error: ' + error.message);
            }
            throw error;
        }
    }

    /**
     * Fetch with automatic retry logic.
     */
    private async fetchWithRetry(url: string, options: RequestInit): Promise<Response> {
        let lastError: Error;

        for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
            try {
                const response = await fetch(url, options);

                // Don't retry on client errors (4xx), except 429 (rate limit)
                if (response.status >= 400 && response.status < 500 && response.status !== 429) {
                    return response;
                }

                // Retry on server errors (5xx) or rate limit (429)
                if (response.status >= 500 || response.status === 429) {
                    if (attempt === this.maxRetries) {
                        return response;
                    }

                    // Exponential backoff
                    const delay = Math.pow(2, attempt) * 1000;
                    await this.sleep(delay);
                    continue;
                }

                return response;
            } catch (error) {
                lastError = error as Error;

                if (attempt === this.maxRetries) {
                    throw lastError;
                }

                // Exponential backoff
                const delay = Math.pow(2, attempt) * 1000;
                await this.sleep(delay);
            }
        }

        throw lastError!;
    }

    /**
     * Handle HTTP response and raise appropriate exceptions.
     */
    private async handleResponse<T>(response: Response): Promise<T> {
        const status = response.status;

        if (status === 401) {
            throw new AuthenticationError('Authentication failed');
        } else if (status === 403) {
            throw new AuthorizationError('Access forbidden');
        } else if (status === 400) {
            const errorData = await this.safeJsonParse(response);
            throw new ValidationError(errorData?.message || 'Validation error');
        } else if (status === 429) {
            const retryAfter = response.headers.get('Retry-After') || '60';
            throw new RateLimitError(`Rate limit exceeded. Retry after ${retryAfter} seconds`);
        } else if (status >= 500) {
            throw new ServerError(`Server error: ${status}`);
        } else if (status < 200 || status >= 300) {
            throw new PynomaliError(`Unexpected status code: ${status}`);
        }

        // Parse JSON response
        if (response.headers.get('content-length') !== '0') {
            try {
                return await response.json();
            } catch (error) {
                throw new PynomaliError('Invalid JSON response');
            }
        }

        return undefined as any;
    }

    /**
     * Safely parse JSON response.
     */
    private async safeJsonParse(response: Response): Promise<any> {
        try {
            return await response.json();
        } catch {
            return null;
        }
    }

    /**
     * Build full URL from endpoint.
     */
    private buildUrl(endpoint: string): string {
        return `${this.baseUrl}/${endpoint.replace(/^\//, '')}`;
    }

    /**
     * Get request headers with authentication.
     */
    private getHeaders(additionalHeaders: Record<string, string> = {}): Record<string, string> {
        const headers: Record<string, string> = {
            'User-Agent': this.userAgent,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            ...additionalHeaders
        };

        // Add authentication headers
        const authHeaders = this.authManager.getAuthHeaders();
        Object.assign(headers, authHeaders);

        return headers;
    }

    /**
     * Sleep utility for retry delays.
     */
    private sleep(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Set JWT token for authentication.
     */
    public setAccessToken(token: string): void {
        this.authManager.setJwtToken(token);
    }

    /**
     * Clear authentication token.
     */
    public clearToken(): void {
        this.authManager.clearToken();
    }
}

/**
 * Authentication API methods.
 */
export class AuthAPI {
    constructor(private client: PynomaliClient) {}

    async login(username: string, password: string): Promise<AuthToken> {
        const response = await this.client.request<AuthToken>('POST', '/auth/login', {
            data: { username, password }
        });

        // Store token in client
        this.client.setAccessToken(response.accessToken);

        return response;
    }

    async refreshToken(refreshToken: string): Promise<AuthToken> {
        const response = await this.client.request<AuthToken>('POST', '/auth/refresh', {
            data: { refreshToken }
        });

        // Store token in client
        this.client.setAccessToken(response.accessToken);

        return response;
    }

    async logout(): Promise<void> {
        await this.client.request('POST', '/auth/logout');
        this.client.clearToken();
    }
}

/**
 * Anomaly detection API methods.
 */
export class DetectionAPI {
    constructor(private client: PynomaliClient) {}

    async detect(request: DetectionRequest): Promise<DetectionResponse> {
        return await this.client.request<DetectionResponse>('POST', '/detection/detect', {
            data: request
        });
    }

    async batchDetect(datasets: number[][], algorithm = 'isolation_forest', parameters: Record<string, any> = {}): Promise<DetectionResponse[]> {
        const response = await this.client.request<{ results: DetectionResponse[] }>('POST', '/detection/batch', {
            data: { datasets, algorithm, parameters }
        });

        return response.results;
    }
}

/**
 * Model training API methods.
 */
export class TrainingAPI {
    constructor(private client: PynomaliClient) {}

    async trainModel(request: TrainingRequest): Promise<TrainingResponse> {
        return await this.client.request<TrainingResponse>('POST', '/training/train', {
            data: request
        });
    }

    async getTrainingStatus(jobId: string): Promise<{ status: string; progress: number }> {
        return await this.client.request('GET', `/training/status/${jobId}`);
    }
}

/**
 * Datasets API methods.
 */
export class DatasetsAPI {
    constructor(private client: PynomaliClient) {}

    async listDatasets(): Promise<DatasetInfo[]> {
        return await this.client.request<DatasetInfo[]>('GET', '/datasets');
    }

    async getDataset(id: string): Promise<DatasetInfo> {
        return await this.client.request<DatasetInfo>('GET', `/datasets/${id}`);
    }

    async createDataset(name: string, data: number[]): Promise<DatasetInfo> {
        return await this.client.request<DatasetInfo>('POST', '/datasets', {
            data: { name, data }
        });
    }

    async deleteDataset(id: string): Promise<void> {
        await this.client.request('DELETE', `/datasets/${id}`);
    }
}

/**
 * Models API methods.
 */
export class ModelsAPI {
    constructor(private client: PynomaliClient) {}

    async listModels(): Promise<ModelInfo[]> {
        return await this.client.request<ModelInfo[]>('GET', '/models');
    }

    async getModel(id: string): Promise<ModelInfo> {
        return await this.client.request<ModelInfo>('GET', `/models/${id}`);
    }

    async deleteModel(id: string): Promise<void> {
        await this.client.request('DELETE', `/models/${id}`);
    }
}

/**
 * Streaming API methods.
 */
export class StreamingAPI {
    constructor(private client: PynomaliClient) {}

    async createStream(config: { bufferSize?: number; windowSize?: number }): Promise<{ streamId: string }> {
        return await this.client.request('POST', '/streaming/create', { data: config });
    }

    async sendData(streamId: string, data: number[]): Promise<DetectionResponse> {
        return await this.client.request<DetectionResponse>('POST', `/streaming/${streamId}/data`, {
            data: { data }
        });
    }

    async closeStream(streamId: string): Promise<void> {
        await this.client.request('DELETE', `/streaming/${streamId}`);
    }
}

/**
 * Explainability API methods.
 */
export class ExplainabilityAPI {
    constructor(private client: PynomaliClient) {}

    async explainDetection(data: number[], modelId?: string): Promise<{ explanations: any[] }> {
        return await this.client.request('POST', '/explainability/explain', {
            data: { data, modelId }
        });
    }

    async getFeatureImportance(modelId: string): Promise<{ features: Array<{ name: string; importance: number }> }> {
        return await this.client.request('GET', `/explainability/importance/${modelId}`);
    }
}

/**
 * Health API methods.
 */
export class HealthAPI {
    constructor(private client: PynomaliClient) {}

    async getHealth(): Promise<HealthStatus> {
        return await this.client.request<HealthStatus>('GET', '/health');
    }

    async getMetrics(): Promise<Record<string, any>> {
        return await this.client.request('GET', '/metrics');
    }
}
