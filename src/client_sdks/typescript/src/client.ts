/**
 * Pynomaly TypeScript Client Implementation
 *
 * This module provides the main client class for interacting with the Pynomaly API.
 * Includes comprehensive functionality with TypeScript type safety, WebSocket support,
 * and modern Promise-based async/await API.
 */

import { AuthManager, SessionManager } from './auth';
import { RateLimiter } from './rate-limiter';
import { WebSocketClient, StreamingManager } from './websocket';
import {
    PynomaliError,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    ServerError,
    NetworkError,
    RateLimitError,
    ErrorHandler
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
    RequestOptions,
    StreamConfig,
    StreamEventHandlers,
    PaginationOptions,
    PaginatedResponse,
    FilterOptions,
    ExplanationRequest,
    ExplanationResult,
    SystemMetrics,
    LoginCredentials,
    UserProfile
} from './types';

/**
 * Main Pynomali client for TypeScript/JavaScript applications.
 * Provides comprehensive API access with modern async/await support,
 * WebSocket streaming, and enhanced error handling.
 */
export class PynomaliClient {
    private readonly baseUrl: string;
    private readonly timeout: number;
    private readonly maxRetries: number;
    private readonly userAgent: string;
    private readonly authManager: AuthManager;
    private readonly sessionManager: SessionManager;
    private readonly rateLimiter: RateLimiter;
    private readonly wsClient?: WebSocketClient;
    private readonly streamingManager?: StreamingManager;
    private readonly debug: boolean;

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
        this.debug = config.debug || false;

        // Initialize auth and session managers
        this.authManager = new AuthManager({
            apiKey: config.apiKey,
            autoRefresh: true,
        });
        this.sessionManager = new SessionManager();

        // Initialize rate limiter
        this.rateLimiter = new RateLimiter({
            maxRequests: config.rateLimitRequests || 100,
            windowMs: config.rateLimitPeriod || 60000,
            adaptive: true,
        });

        // Initialize WebSocket client if enabled
        if (config.websocket?.enabled) {
            const wsUrl = config.websocket.url || this.buildWebSocketUrl();
            this.wsClient = new WebSocketClient({
                ...config.websocket,
                url: wsUrl,
                authToken: this.authManager.getJwtToken(),
                apiKey: config.apiKey,
            });

            // Initialize streaming manager
            this.streamingManager = new StreamingManager(this.wsClient);
        }

        // Set up token refresh callback
        this.authManager.setTokenRefreshCallback(async () => {
            await this.refreshAuthToken();
        });

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
     * Make HTTP request with enhanced error handling, rate limiting, and debugging.
     */
    public async request<T = any>(
        method: HttpMethod,
        endpoint: string,
        options: RequestOptions = {}
    ): Promise<T> {
        // Check authentication if required
        if (!this.authManager.isAuthenticated() && this.requiresAuth(endpoint)) {
            throw new AuthenticationError('Authentication required for this endpoint');
        }

        // Refresh token if needed
        if (this.authManager.needsRefresh()) {
            await this.refreshAuthToken();
        }

        // Rate limiting (skip if requested)
        if (!options.skipRateLimit) {
            await this.rateLimiter.waitIfNeeded();
        }

        const url = this.buildUrl(endpoint, options.params);
        const headers = this.getHeaders(options.headers);
        const requestTimeout = options.timeout || this.timeout;

        if (this.debug) {
            console.debug(`[Pynomaly SDK] ${method} ${url}`, {
                headers,
                data: options.data,
            });
        }

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
            const result = await this.handleResponse<T>(response);
            
            if (this.debug) {
                console.debug(`[Pynomaly SDK] Response:`, result);
            }
            
            return result;
        } catch (error) {
            if (this.debug) {
                console.error(`[Pynomaly SDK] Error:`, error);
            }
            
            if (error instanceof DOMException && error.name === 'TimeoutError') {
                throw new NetworkError(`Request timeout after ${requestTimeout}ms`, true);
            }
            if (error instanceof TypeError) {
                throw new NetworkError('Network error: ' + error.message, false, !navigator.onLine);
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
     * Handle HTTP response with comprehensive error mapping.
     */
    private async handleResponse<T>(response: Response): Promise<T> {
        const status = response.status;
        const requestId = response.headers.get('X-Request-ID') || undefined;

        // Try to parse response data for error details
        let responseData: any = null;
        try {
            if (response.headers.get('content-type')?.includes('application/json')) {
                responseData = await response.json();
            }
        } catch {
            // Ignore JSON parsing errors for non-JSON responses
        }

        if (status >= 400) {
            const error = ErrorHandler.fromHttpResponse(status, responseData, requestId);
            throw error;
        }

        if (status < 200 || status >= 300) {
            throw new PynomaliError(`Unexpected status code: ${status}`, 'HTTP_ERROR', status, responseData, requestId);
        }

        // Return parsed JSON or empty response
        if (responseData !== null) {
            return responseData;
        }

        // Handle empty responses
        if (response.headers.get('content-length') === '0' || status === 204) {
            return undefined as any;
        }

        // Try to parse response body
        try {
            return await response.json();
        } catch (error) {
            // For non-JSON responses, return as text
            try {
                const text = await response.text();
                return text as any;
            } catch {
                return undefined as any;
            }
        }
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
     * Build full URL from endpoint with query parameters.
     */
    private buildUrl(endpoint: string, params?: Record<string, any>): string {
        const baseUrl = `${this.baseUrl}/${endpoint.replace(/^\//, '')}`;
        
        if (params && Object.keys(params).length > 0) {
            const searchParams = new URLSearchParams();
            for (const [key, value] of Object.entries(params)) {
                if (value !== undefined && value !== null) {
                    searchParams.append(key, String(value));
                }
            }
            return `${baseUrl}?${searchParams.toString()}`;
        }
        
        return baseUrl;
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
        this.sessionManager.endSession();
    }

    /**
     * Get WebSocket client for real-time updates.
     */
    public getWebSocketClient(): WebSocketClient | undefined {
        return this.wsClient;
    }

    /**
     * Get streaming manager for real-time data processing.
     */
    public getStreamingManager(): StreamingManager | undefined {
        return this.streamingManager;
    }

    /**
     * Connect to WebSocket for real-time updates.
     */
    public async connectWebSocket(eventHandlers?: Partial<StreamEventHandlers>): Promise<void> {
        if (!this.wsClient) {
            throw new PynomaliError('WebSocket not enabled. Enable in client config.');
        }

        if (eventHandlers) {
            this.wsClient.setEventHandlers(eventHandlers);
        }

        await this.wsClient.connect();
    }

    /**
     * Disconnect from WebSocket.
     */
    public disconnectWebSocket(): void {
        this.wsClient?.disconnect();
    }

    /**
     * Get client configuration and status.
     */
    public getClientInfo(): {
        baseUrl: string;
        userAgent: string;
        isAuthenticated: boolean;
        isConnected: boolean;
        rateLimitStatus: any;
        sessionInfo?: UserProfile;
    } {
        return {
            baseUrl: this.baseUrl,
            userAgent: this.userAgent,
            isAuthenticated: this.authManager.isAuthenticated(),
            isConnected: this.wsClient?.getConnectionState() === 'connected',
            rateLimitStatus: this.rateLimiter.getStatus(),
            sessionInfo: this.sessionManager.getUserProfile(),
        };
    }

    /**
     * Check if endpoint requires authentication.
     */
    private requiresAuth(endpoint: string): boolean {
        // Public endpoints that don't require authentication
        const publicEndpoints = ['/health', '/metrics', '/auth/login', '/auth/register'];
        return !publicEndpoints.some(path => endpoint.startsWith(path));
    }

    /**
     * Build WebSocket URL from base URL.
     */
    private buildWebSocketUrl(): string {
        const wsProtocol = this.baseUrl.startsWith('https://') ? 'wss://' : 'ws://';
        const baseWithoutProtocol = this.baseUrl.replace(/^https?:\/\//, '');
        return `${wsProtocol}${baseWithoutProtocol}/ws`;
    }

    /**
     * Refresh authentication token.
     */
    private async refreshAuthToken(): Promise<void> {
        const refreshToken = this.authManager.getRefreshToken();
        if (!refreshToken) {
            throw new AuthenticationError('No refresh token available');
        }

        try {
            const newToken = await this.auth.refreshToken(refreshToken);
            this.authManager.setAuthToken(newToken);
        } catch (error) {
            this.authManager.clearToken();
            throw new AuthenticationError('Failed to refresh token');
        }
    }
}

/**
 * Authentication API methods with enhanced session management.
 */
export class AuthAPI {
    constructor(private client: PynomaliClient) {}

    /**
     * Login with username and password.
     */
    async login(credentials: LoginCredentials): Promise<{ token: AuthToken; user: UserProfile }>;
    async login(username: string, password: string, mfaCode?: string): Promise<{ token: AuthToken; user: UserProfile }>;
    async login(
        credentialsOrUsername: LoginCredentials | string,
        password?: string,
        mfaCode?: string
    ): Promise<{ token: AuthToken; user: UserProfile }> {
        let loginData: LoginCredentials;

        if (typeof credentialsOrUsername === 'string') {
            loginData = {
                username: credentialsOrUsername,
                password: password!,
                mfaCode,
            };
        } else {
            loginData = credentialsOrUsername;
        }

        const response = await this.client.request<{ token: AuthToken; user: UserProfile }>('POST', '/auth/login', {
            data: loginData,
            skipRateLimit: false,
        });

        // Store token and user session
        this.client.setAccessToken(response.token.accessToken);
        (this.client as any).authManager.setAuthToken(response.token);
        (this.client as any).sessionManager.startSession(response.user);

        return response;
    }

    /**
     * Refresh authentication token.
     */
    async refreshToken(refreshToken: string): Promise<AuthToken> {
        const response = await this.client.request<AuthToken>('POST', '/auth/refresh', {
            data: { refreshToken },
            skipRateLimit: true,
        });

        // Update stored token
        this.client.setAccessToken(response.accessToken);
        (this.client as any).authManager.setAuthToken(response);

        return response;
    }

    /**
     * Logout and clear session.
     */
    async logout(): Promise<void> {
        try {
            await this.client.request('POST', '/auth/logout', {
                skipRateLimit: true,
            });
        } catch (error) {
            // Continue with logout even if server request fails
            console.warn('Logout request failed, clearing local session:', error);
        }

        this.client.clearToken();
        this.client.disconnectWebSocket();
    }

    /**
     * Get current user profile.
     */
    async getCurrentUser(): Promise<UserProfile> {
        return await this.client.request<UserProfile>('GET', '/auth/me');
    }

    /**
     * Update user profile.
     */
    async updateProfile(updates: Partial<UserProfile>): Promise<UserProfile> {
        return await this.client.request<UserProfile>('PATCH', '/auth/me', {
            data: updates,
        });
    }

    /**
     * Change password.
     */
    async changePassword(currentPassword: string, newPassword: string): Promise<void> {
        await this.client.request('POST', '/auth/change-password', {
            data: {
                currentPassword,
                newPassword,
            },
        });
    }

    /**
     * Request password reset.
     */
    async requestPasswordReset(email: string): Promise<void> {
        await this.client.request('POST', '/auth/password-reset', {
            data: { email },
            skipRateLimit: false,
        });
    }

    /**
     * Verify email address.
     */
    async verifyEmail(token: string): Promise<void> {
        await this.client.request('POST', '/auth/verify-email', {
            data: { token },
        });
    }

    /**
     * Enable two-factor authentication.
     */
    async enableMFA(): Promise<{ qrCode: string; backupCodes: string[] }> {
        return await this.client.request('POST', '/auth/mfa/enable');
    }

    /**
     * Disable two-factor authentication.
     */
    async disableMFA(mfaCode: string): Promise<void> {
        await this.client.request('POST', '/auth/mfa/disable', {
            data: { mfaCode },
        });
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
