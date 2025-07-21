/**
 * anomaly_detection Java SDK
 *
 * Official Java client library for the anomaly_detection anomaly detection API.
 * This SDK provides convenient access to the anomaly_detection API with full type safety,
 * authentication handling, error management, and comprehensive documentation.
 *
 * Features:
 * - Complete API coverage with type-safe client methods
 * - JWT and API Key authentication support
 * - Automatic retry logic with exponential backoff
 * - Rate limiting and request throttling
 * - Comprehensive error handling
 * - Async and sync support with CompletableFuture
 * - Built-in logging and debugging
 * - Jackson JSON serialization
 *
 * Example Usage:
 * <pre>
 * {@code
 * AnomalyDetectionClient client = AnomalyDetectionClient.builder()
 *     .baseUrl("https://api.anomaly_detection.com")
 *     .apiKey("your-api-key")
 *     .build();
 *
 * // Authenticate (if using JWT)
 * // AuthToken token = client.auth().login("username", "password");
 *
 * // Detect anomalies
 * DetectionRequest request = DetectionRequest.builder()
 *     .data(Arrays.asList(1.0, 2.0, 3.0, 100.0, 4.0, 5.0))
 *     .algorithm("isolation_forest")
 *     .parameters(Map.of("contamination", 0.1))
 *     .build();
 *
 * DetectionResponse result = client.detection().detect(request);
 * System.out.println("Anomalies detected: " + result.getAnomalies());
 *
 * client.close();
 * }
 * </pre>
 */
package com.anomaly_detection.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.anomaly_detection.client.api.*;
import com.anomaly_detection.client.auth.AuthManager;
import com.anomaly_detection.client.exception.*;
import com.anomaly_detection.client.http.HttpClient;
import com.anomaly_detection.client.http.RateLimiter;
import com.anomaly_detection.client.model.*;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * Main anomaly-detection client for Java applications.
 */
public class AnomalyDetectionClient implements AutoCloseable {

    private static final Logger LOGGER = LoggerFactory.getLogger(AnomalyDetectionClient.class);

    private static final String DEFAULT_BASE_URL = "https://api.anomaly_detection.com";
    private static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(30);
    private static final int DEFAULT_MAX_RETRIES = 3;
    private static final String USER_AGENT = "anomaly_detection-java-sdk/1.0.0";

    private final String baseUrl;
    private final Duration timeout;
    private final int maxRetries;
    private final ObjectMapper objectMapper;
    private final OkHttpClient httpClient;
    private final AuthManager authManager;
    private final RateLimiter rateLimiter;

    // API modules
    private final AuthAPI authAPI;
    private final DetectionAPI detectionAPI;
    private final TrainingAPI trainingAPI;
    private final DatasetsAPI datasetsAPI;
    private final ModelsAPI modelsAPI;
    private final StreamingAPI streamingAPI;
    private final ExplainabilityAPI explainabilityAPI;
    private final HealthAPI healthAPI;

    private AnomalyDetectionClient(Builder builder) {
        this.baseUrl = builder.baseUrl.replaceAll("/$", "");
        this.timeout = builder.timeout;
        this.maxRetries = builder.maxRetries;

        // Setup JSON serialization
        this.objectMapper = new ObjectMapper()
            .registerModule(new JavaTimeModule());

        // Setup HTTP client with retry logic
        this.httpClient = new OkHttpClient.Builder()
            .connectTimeout(timeout.toMillis(), TimeUnit.MILLISECONDS)
            .readTimeout(timeout.toMillis(), TimeUnit.MILLISECONDS)
            .writeTimeout(timeout.toMillis(), TimeUnit.MILLISECONDS)
            .addInterceptor(new RetryInterceptor(maxRetries))
            .addInterceptor(new LoggingInterceptor())
            .build();

        // Setup authentication and rate limiting
        this.authManager = new AuthManager(builder.apiKey);
        this.rateLimiter = new RateLimiter(builder.rateLimitRequests, builder.rateLimitPeriod);

        // Initialize API modules
        this.authAPI = new AuthAPI(this);
        this.detectionAPI = new DetectionAPI(this);
        this.trainingAPI = new TrainingAPI(this);
        this.datasetsAPI = new DatasetsAPI(this);
        this.modelsAPI = new ModelsAPI(this);
        this.streamingAPI = new StreamingAPI(this);
        this.explainabilityAPI = new ExplainabilityAPI(this);
        this.healthAPI = new HealthAPI(this);
    }

    /**
     * Create a new client builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Make HTTP request with error handling and rate limiting.
     */
    public <T> T request(String method, String endpoint, Object data, Class<T> responseType) throws anomaly-detectionException {
        return requestAsync(method, endpoint, data, responseType).join();
    }

    /**
     * Make async HTTP request with error handling and rate limiting.
     */
    public <T> CompletableFuture<T> requestAsync(String method, String endpoint, Object data, Class<T> responseType) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                // Rate limiting
                rateLimiter.waitIfNeeded();

                String url = buildUrl(endpoint);
                Request.Builder requestBuilder = new Request.Builder()
                    .url(url)
                    .headers(getHeaders());

                // Add request body for non-GET requests
                RequestBody body = null;
                if (data != null && !"GET".equals(method)) {
                    String json = objectMapper.writeValueAsString(data);
                    body = RequestBody.create(json, MediaType.get("application/json"));
                }

                switch (method.toUpperCase()) {
                    case "GET":
                        requestBuilder.get();
                        break;
                    case "POST":
                        requestBuilder.post(body != null ? body : RequestBody.create("", null));
                        break;
                    case "PUT":
                        requestBuilder.put(body != null ? body : RequestBody.create("", null));
                        break;
                    case "DELETE":
                        requestBuilder.delete();
                        break;
                    default:
                        throw new IllegalArgumentException("Unsupported HTTP method: " + method);
                }

                Request request = requestBuilder.build();
                LOGGER.debug("Making {} request to {}", method, url);

                try (Response response = httpClient.newCall(request).execute()) {
                    return handleResponse(response, responseType);
                }

            } catch (Exception e) {
                if (e instanceof anomaly-detectionException) {
                    throw new RuntimeException(e);
                }
                throw new RuntimeException(new NetworkException("Request failed: " + e.getMessage(), e));
            }
        });
    }

    /**
     * Handle HTTP response and raise appropriate exceptions.
     */
    private <T> T handleResponse(Response response, Class<T> responseType) throws IOException, anomaly-detectionException {
        int status = response.code();

        if (status == 401) {
            throw new AuthenticationException("Authentication failed");
        } else if (status == 403) {
            throw new AuthorizationException("Access forbidden");
        } else if (status == 400) {
            String errorBody = response.body() != null ? response.body().string() : "";
            ErrorResponse errorResponse = null;
            try {
                errorResponse = objectMapper.readValue(errorBody, ErrorResponse.class);
            } catch (Exception ignored) {}

            String message = errorResponse != null ? errorResponse.getMessage() : "Validation error";
            throw new ValidationException(message);
        } else if (status == 429) {
            String retryAfter = response.header("Retry-After", "60");
            throw new RateLimitException("Rate limit exceeded. Retry after " + retryAfter + " seconds");
        } else if (status >= 500) {
            throw new ServerException("Server error: " + status);
        } else if (status < 200 || status >= 300) {
            throw new anomaly-detectionException("Unexpected status code: " + status);
        }

        // Parse JSON response
        if (response.body() != null && response.body().contentLength() != 0) {
            String responseBody = response.body().string();

            if (responseType == Void.class) {
                return null;
            }

            if (responseType == String.class) {
                return responseType.cast(responseBody);
            }

            return objectMapper.readValue(responseBody, responseType);
        }

        return null;
    }

    /**
     * Build full URL from endpoint.
     */
    private String buildUrl(String endpoint) {
        return baseUrl + "/" + endpoint.replaceFirst("^/", "");
    }

    /**
     * Get request headers with authentication.
     */
    private Headers getHeaders() {
        Headers.Builder builder = new Headers.Builder()
            .add("User-Agent", USER_AGENT)
            .add("Content-Type", "application/json")
            .add("Accept", "application/json");

        // Add authentication headers
        authManager.getAuthHeaders().forEach(builder::add);

        return builder.build();
    }

    /**
     * Set JWT token for authentication.
     */
    public void setAccessToken(String token) {
        authManager.setJwtToken(token);
    }

    /**
     * Clear authentication token.
     */
    public void clearToken() {
        authManager.clearToken();
    }

    // API module getters
    public AuthAPI auth() { return authAPI; }
    public DetectionAPI detection() { return detectionAPI; }
    public TrainingAPI training() { return trainingAPI; }
    public DatasetsAPI datasets() { return datasetsAPI; }
    public ModelsAPI models() { return modelsAPI; }
    public StreamingAPI streaming() { return streamingAPI; }
    public ExplainabilityAPI explainability() { return explainabilityAPI; }
    public HealthAPI health() { return healthAPI; }

    @Override
    public void close() {
        httpClient.dispatcher().executorService().shutdown();
        httpClient.connectionPool().evictAll();
    }

    /**
     * Builder for AnomalyDetectionClient.
     */
    public static class Builder {
        private String baseUrl = DEFAULT_BASE_URL;
        private Duration timeout = DEFAULT_TIMEOUT;
        private int maxRetries = DEFAULT_MAX_RETRIES;
        private String apiKey;
        private int rateLimitRequests = 100;
        private Duration rateLimitPeriod = Duration.ofMinutes(1);

        public Builder baseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
            return this;
        }

        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public Builder maxRetries(int maxRetries) {
            this.maxRetries = maxRetries;
            return this;
        }

        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        public Builder rateLimit(int requests, Duration period) {
            this.rateLimitRequests = requests;
            this.rateLimitPeriod = period;
            return this;
        }

        public AnomalyDetectionClient build() {
            return new AnomalyDetectionClient(this);
        }
    }

    /**
     * Retry interceptor for HTTP requests.
     */
    private static class RetryInterceptor implements Interceptor {
        private final int maxRetries;

        public RetryInterceptor(int maxRetries) {
            this.maxRetries = maxRetries;
        }

        @Override
        public Response intercept(Chain chain) throws IOException {
            Request request = chain.request();
            Response response = null;
            IOException lastException = null;

            for (int attempt = 0; attempt <= maxRetries; attempt++) {
                try {
                    if (response != null) {
                        response.close();
                    }

                    response = chain.proceed(request);

                    // Don't retry on client errors (4xx), except 429 (rate limit)
                    if (response.code() >= 400 && response.code() < 500 && response.code() != 429) {
                        return response;
                    }

                    // Retry on server errors (5xx) or rate limit (429)
                    if (response.code() >= 500 || response.code() == 429) {
                        if (attempt == maxRetries) {
                            return response;
                        }

                        // Exponential backoff
                        long delay = (long) Math.pow(2, attempt) * 1000;
                        try {
                            Thread.sleep(delay);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new IOException("Request interrupted", e);
                        }
                        continue;
                    }

                    return response;

                } catch (IOException e) {
                    lastException = e;

                    if (attempt == maxRetries) {
                        throw lastException;
                    }

                    // Exponential backoff
                    long delay = (long) Math.pow(2, attempt) * 1000;
                    try {
                        Thread.sleep(delay);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new IOException("Request interrupted", ie);
                    }
                }
            }

            throw lastException;
        }
    }

    /**
     * Logging interceptor for debugging.
     */
    private static class LoggingInterceptor implements Interceptor {
        @Override
        public Response intercept(Chain chain) throws IOException {
            Request request = chain.request();

            LOGGER.debug("→ {} {}", request.method(), request.url());

            long startTime = System.nanoTime();
            Response response = chain.proceed(request);
            long endTime = System.nanoTime();

            long duration = TimeUnit.NANOSECONDS.toMillis(endTime - startTime);
            LOGGER.debug("← {} {} ({}ms)", response.code(), request.url(), duration);

            return response;
        }
    }
}
