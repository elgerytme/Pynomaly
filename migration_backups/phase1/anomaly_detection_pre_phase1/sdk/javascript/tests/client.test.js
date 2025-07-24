/**
 * Tests for the JavaScript/TypeScript anomaly detection client
 */

const axios = require('axios');
const { 
    AnomalyDetectionClient, 
    AlgorithmType,
    ValidationError,
    APIError,
    ConnectionError,
    TimeoutError
} = require('../src');

// Mock axios
jest.mock('axios');
const mockedAxios = axios;

describe('AnomalyDetectionClient', () => {
    let client;
    const mockBaseUrl = 'http://localhost:8000';

    beforeEach(() => {
        client = new AnomalyDetectionClient({
            baseUrl: mockBaseUrl,
            timeout: 30000,
            maxRetries: 3
        });
        
        // Reset all mocks
        jest.clearAllMocks();
    });

    describe('Constructor', () => {
        test('should initialize with basic configuration', () => {
            const basicClient = new AnomalyDetectionClient({
                baseUrl: mockBaseUrl
            });

            expect(basicClient.config.baseUrl).toBe(mockBaseUrl);
            expect(basicClient.config.timeout).toBe(30000); // default
            expect(basicClient.config.maxRetries).toBe(3); // default
        });

        test('should initialize with full configuration', () => {
            const fullClient = new AnomalyDetectionClient({
                baseUrl: 'http://example.com',
                apiKey: 'test-key',
                timeout: 60000,
                maxRetries: 5,
                headers: { 'Custom-Header': 'custom-value' }
            });

            expect(fullClient.config.baseUrl).toBe('http://example.com');
            expect(fullClient.config.apiKey).toBe('test-key');
            expect(fullClient.config.timeout).toBe(60000);
            expect(fullClient.config.maxRetries).toBe(5);
            expect(fullClient.config.headers['Custom-Header']).toBe('custom-value');
        });

        test('should set up authorization header when API key is provided', () => {
            const clientWithKey = new AnomalyDetectionClient({
                baseUrl: mockBaseUrl,
                apiKey: 'test-api-key'
            });

            expect(clientWithKey.axios.defaults.headers.Authorization).toBe('Bearer test-api-key');
        });
    });

    describe('detectAnomalies', () => {
        const sampleData = [
            [1.0, 2.0],
            [1.1, 2.1],
            [10.0, 20.0] // anomalous point
        ];

        const mockResponse = {
            data: {
                anomalies: [
                    {
                        index: 2,
                        score: 0.85,
                        dataPoint: [10.0, 20.0],
                        confidence: 0.9
                    }
                ],
                totalPoints: 3,
                anomalyCount: 1,
                algorithmUsed: 'isolation_forest',
                executionTime: 0.15,
                metadata: {}
            }
        };

        test('should detect anomalies successfully', async () => {
            mockedAxios.request.mockResolvedValue(mockResponse);

            const result = await client.detectAnomalies(
                sampleData,
                AlgorithmType.ISOLATION_FOREST,
                { contamination: 0.3 }
            );

            expect(mockedAxios.request).toHaveBeenCalledWith({
                method: 'POST',
                url: '/api/v1/detect',
                json: {
                    data: sampleData,
                    algorithm: 'isolation_forest',
                    parameters: { contamination: 0.3 },
                    return_explanations: false
                },
                params: undefined
            });

            expect(result.anomalyCount).toBe(1);
            expect(result.totalPoints).toBe(3);
            expect(result.anomalies).toHaveLength(1);
            expect(result.anomalies[0].index).toBe(2);
            expect(result.anomalies[0].score).toBe(0.85);
        });

        test('should handle empty data validation', async () => {
            await expect(client.detectAnomalies([])).rejects.toThrow(ValidationError);
        });

        test('should handle invalid data format validation', async () => {
            const invalidData = ['not', 'arrays'];
            await expect(client.detectAnomalies(invalidData)).rejects.toThrow(ValidationError);
        });

        test('should retry on connection errors', async () => {
            const connectionError = new Error('ECONNREFUSED');
            connectionError.code = 'ECONNREFUSED';
            
            mockedAxios.request
                .mockRejectedValueOnce(connectionError)
                .mockResolvedValue(mockResponse);

            const result = await client.detectAnomalies(sampleData);

            expect(mockedAxios.request).toHaveBeenCalledTimes(2);
            expect(result.anomalyCount).toBe(1);
        });

        test('should handle API errors', async () => {
            const apiError = {
                response: {
                    status: 400,
                    data: { detail: 'Invalid algorithm' }
                }
            };
            mockedAxios.request.mockRejectedValue(apiError);

            await expect(client.detectAnomalies(sampleData)).rejects.toThrow(APIError);
        });

        test('should handle timeout errors', async () => {
            const timeoutError = new Error('timeout');
            timeoutError.code = 'ECONNABORTED';
            mockedAxios.request.mockRejectedValue(timeoutError);

            await expect(client.detectAnomalies(sampleData)).rejects.toThrow(TimeoutError);
        });
    });

    describe('batchDetect', () => {
        test('should process batch requests', async () => {
            const batchRequest = {
                data: [[1, 2], [3, 4]],
                algorithm: AlgorithmType.ISOLATION_FOREST,
                parameters: { contamination: 0.1 },
                returnExplanations: true
            };

            const mockResponse = {
                data: {
                    anomalies: [],
                    totalPoints: 2,
                    anomalyCount: 0,
                    algorithmUsed: 'isolation_forest',
                    executionTime: 0.1,
                    metadata: {}
                }
            };

            mockedAxios.request.mockResolvedValue(mockResponse);

            const result = await client.batchDetect(batchRequest);

            expect(mockedAxios.request).toHaveBeenCalledWith({
                method: 'POST',
                url: '/api/v1/batch/detect',
                json: batchRequest,
                params: undefined
            });

            expect(result.totalPoints).toBe(2);
            expect(result.anomalyCount).toBe(0);
        });
    });

    describe('trainModel', () => {
        test('should train model successfully', async () => {
            const trainingRequest = {
                data: [[1, 2], [3, 4]],
                algorithm: AlgorithmType.ISOLATION_FOREST,
                modelName: 'test-model'
            };

            const mockResponse = {
                data: {
                    modelId: 'model-123',
                    trainingTime: 5.2,
                    performanceMetrics: { accuracy: 0.95 },
                    validationMetrics: { f1Score: 0.92 },
                    modelInfo: {
                        modelId: 'model-123',
                        algorithm: 'isolation_forest',
                        createdAt: '2023-01-01T00:00:00Z',
                        trainingDataSize: 100,
                        performanceMetrics: { accuracy: 0.95 },
                        hyperparameters: { nEstimators: 100 },
                        version: '1.0',
                        status: 'trained'
                    }
                }
            };

            mockedAxios.request.mockResolvedValue(mockResponse);

            const result = await client.trainModel(trainingRequest);

            expect(mockedAxios.request).toHaveBeenCalledWith({
                method: 'POST',
                url: '/api/v1/models/train',
                json: trainingRequest,
                params: undefined
            });

            expect(result.modelId).toBe('model-123');
            expect(result.trainingTime).toBe(5.2);
        });
    });

    describe('getModel', () => {
        test('should get model information', async () => {
            const modelId = 'model-123';
            const mockResponse = {
                data: {
                    modelId: 'model-123',
                    algorithm: 'isolation_forest',
                    createdAt: '2023-01-01T00:00:00Z',
                    trainingDataSize: 100,
                    performanceMetrics: { accuracy: 0.95 },
                    hyperparameters: { nEstimators: 100 },
                    version: '1.0',
                    status: 'trained'
                }
            };

            mockedAxios.request.mockResolvedValue(mockResponse);

            const result = await client.getModel(modelId);

            expect(mockedAxios.request).toHaveBeenCalledWith({
                method: 'GET',
                url: `/api/v1/models/${modelId}`,
                json: undefined,
                params: undefined
            });

            expect(result.modelId).toBe('model-123');
            expect(result.algorithm).toBe('isolation_forest');
        });

        test('should validate model ID', async () => {
            await expect(client.getModel('')).rejects.toThrow(ValidationError);
        });
    });

    describe('listModels', () => {
        test('should list all models', async () => {
            const mockResponse = {
                data: {
                    models: [
                        {
                            modelId: 'model-1',
                            algorithm: 'isolation_forest',
                            createdAt: '2023-01-01T00:00:00Z',
                            trainingDataSize: 100,
                            performanceMetrics: { accuracy: 0.95 },
                            hyperparameters: { nEstimators: 100 },
                            version: '1.0',
                            status: 'trained'
                        },
                        {
                            modelId: 'model-2',
                            algorithm: 'local_outlier_factor',
                            createdAt: '2023-01-01T01:00:00Z',
                            trainingDataSize: 50,
                            performanceMetrics: { accuracy: 0.88 },
                            hyperparameters: { nNeighbors: 20 },
                            version: '1.0',
                            status: 'trained'
                        }
                    ]
                }
            };

            mockedAxios.request.mockResolvedValue(mockResponse);

            const result = await client.listModels();

            expect(mockedAxios.request).toHaveBeenCalledWith({
                method: 'GET',
                url: '/api/v1/models',
                json: undefined,
                params: undefined
            });

            expect(result).toHaveLength(2);
            expect(result[0].modelId).toBe('model-1');
            expect(result[1].modelId).toBe('model-2');
        });
    });

    describe('deleteModel', () => {
        test('should delete model', async () => {
            const modelId = 'model-123';
            const mockResponse = {
                data: { message: 'Model deleted successfully' }
            };

            mockedAxios.request.mockResolvedValue(mockResponse);

            const result = await client.deleteModel(modelId);

            expect(mockedAxios.request).toHaveBeenCalledWith({
                method: 'DELETE',
                url: `/api/v1/models/${modelId}`,
                json: undefined,
                params: undefined
            });

            expect(result.message).toBe('Model deleted successfully');
        });

        test('should validate model ID', async () => {
            await expect(client.deleteModel('')).rejects.toThrow(ValidationError);
        });
    });

    describe('explainAnomaly', () => {
        test('should explain anomaly', async () => {
            const dataPoint = [10.0, 20.0];
            const mockResponse = {
                data: {
                    anomalyIndex: 0,
                    featureImportance: { feature0: 0.8, feature1: 0.2 },
                    shapValues: [0.3, 0.1],
                    explanationText: 'High values in feature 0',
                    confidence: 0.9
                }
            };

            mockedAxios.request.mockResolvedValue(mockResponse);

            const result = await client.explainAnomaly(dataPoint, {
                algorithm: AlgorithmType.ISOLATION_FOREST,
                method: 'shap'
            });

            expect(mockedAxios.request).toHaveBeenCalledWith({
                method: 'POST',
                url: '/api/v1/explain',
                json: {
                    data_point: dataPoint,
                    method: 'shap',
                    algorithm: 'isolation_forest'
                },
                params: undefined
            });

            expect(result.anomalyIndex).toBe(0);
            expect(result.featureImportance.feature0).toBe(0.8);
            expect(result.explanationText).toBe('High values in feature 0');
        });

        test('should validate data point', async () => {
            await expect(client.explainAnomaly([])).rejects.toThrow(ValidationError);
        });
    });

    describe('getHealth', () => {
        test('should get health status', async () => {
            const mockResponse = {
                data: {
                    status: 'healthy',
                    timestamp: '2023-01-01T00:00:00Z',
                    version: '1.0.0',
                    uptime: 3600.5,
                    components: { database: 'healthy', cache: 'healthy' },
                    metrics: { requestsPerSecond: 100 }
                }
            };

            mockedAxios.request.mockResolvedValue(mockResponse);

            const result = await client.getHealth();

            expect(mockedAxios.request).toHaveBeenCalledWith({
                method: 'GET',
                url: '/api/v1/health',
                json: undefined,
                params: undefined
            });

            expect(result.status).toBe('healthy');
            expect(result.version).toBe('1.0.0');
            expect(result.uptime).toBe(3600.5);
        });
    });

    describe('getMetrics', () => {
        test('should get service metrics', async () => {
            const mockResponse = {
                data: {
                    requestsPerSecond: 100,
                    averageResponseTime: 0.05,
                    activeConnections: 25
                }
            };

            mockedAxios.request.mockResolvedValue(mockResponse);

            const result = await client.getMetrics();

            expect(mockedAxios.request).toHaveBeenCalledWith({
                method: 'GET',
                url: '/api/v1/metrics',
                json: undefined,
                params: undefined
            });

            expect(result.requestsPerSecond).toBe(100);
            expect(result.averageResponseTime).toBe(0.05);
        });
    });

    describe('uploadData', () => {
        test('should upload data successfully', async () => {
            const data = [[1, 2], [3, 4]];
            const datasetName = 'test-dataset';
            const description = 'Test dataset';
            
            const mockResponse = {
                data: {
                    datasetId: 'dataset-123',
                    message: 'Data uploaded successfully'
                }
            };

            mockedAxios.request.mockResolvedValue(mockResponse);

            const result = await client.uploadData(data, datasetName, description);

            expect(mockedAxios.request).toHaveBeenCalledWith({
                method: 'POST',
                url: '/api/v1/data/upload',
                json: {
                    data: data,
                    name: datasetName,
                    description: description
                },
                params: undefined
            });

            expect(result.datasetId).toBe('dataset-123');
            expect(result.message).toBe('Data uploaded successfully');
        });

        test('should validate data', async () => {
            await expect(client.uploadData([], 'test')).rejects.toThrow(ValidationError);
        });

        test('should validate dataset name', async () => {
            await expect(client.uploadData([[1, 2]], '')).rejects.toThrow(ValidationError);
        });
    });

    describe('Error Handling', () => {
        test('should handle different HTTP status codes', async () => {
            const testCases = [
                { status: 400, expectedError: APIError },
                { status: 401, expectedError: APIError },
                { status: 404, expectedError: APIError },
                { status: 429, expectedError: APIError },
                { status: 500, expectedError: APIError }
            ];

            for (const testCase of testCases) {
                const error = {
                    response: {
                        status: testCase.status,
                        data: { detail: `Error ${testCase.status}` }
                    }
                };

                mockedAxios.request.mockRejectedValue(error);

                await expect(client.detectAnomalies([[1, 2]])).rejects.toThrow(testCase.expectedError);
            }
        });

        test('should handle network errors', async () => {
            const networkError = new Error('Network Error');
            networkError.code = 'ENOTFOUND';
            mockedAxios.request.mockRejectedValue(networkError);

            await expect(client.detectAnomalies([[1, 2]])).rejects.toThrow(ConnectionError);
        });

        test('should handle malformed JSON responses', async () => {
            const response = {
                data: 'invalid json'
            };
            mockedAxios.request.mockResolvedValue(response);

            // This should not throw but handle gracefully
            const result = await client.getMetrics();
            expect(result).toBe('invalid json');
        });
    });

    describe('Retry Logic', () => {
        test('should retry on transient failures and succeed', async () => {
            const connectionError = new Error('ECONNRESET');
            connectionError.code = 'ECONNRESET';
            
            const successResponse = {
                data: {
                    anomalies: [],
                    totalPoints: 1,
                    anomalyCount: 0,
                    algorithmUsed: 'isolation_forest',
                    executionTime: 0.1,
                    metadata: {}
                }
            };

            mockedAxios.request
                .mockRejectedValueOnce(connectionError)
                .mockRejectedValueOnce(connectionError)
                .mockResolvedValue(successResponse);

            const result = await client.detectAnomalies([[1, 2]]);

            expect(mockedAxios.request).toHaveBeenCalledTimes(3);
            expect(result.totalPoints).toBe(1);
        });

        test('should not retry on client errors (4xx)', async () => {
            const clientError = {
                response: {
                    status: 400,
                    data: { detail: 'Bad Request' }
                }
            };

            mockedAxios.request.mockRejectedValue(clientError);

            await expect(client.detectAnomalies([[1, 2]])).rejects.toThrow(APIError);

            // Should not retry on 4xx errors
            expect(mockedAxios.request).toHaveBeenCalledTimes(1);
        });

        test('should fail after max retries', async () => {
            const connectionError = new Error('ECONNRESET');
            connectionError.code = 'ECONNRESET';
            
            mockedAxios.request.mockRejectedValue(connectionError);

            await expect(client.detectAnomalies([[1, 2]])).rejects.toThrow(ConnectionError);

            // Should have tried maxRetries + 1 times
            expect(mockedAxios.request).toHaveBeenCalledTimes(4);
        });
    });

    describe('Request Configuration', () => {
        test('should include custom headers in requests', () => {
            const clientWithHeaders = new AnomalyDetectionClient({
                baseUrl: mockBaseUrl,
                headers: { 'X-Custom': 'test-value' }
            });

            expect(clientWithHeaders.axios.defaults.headers['X-Custom']).toBe('test-value');
        });

        test('should set correct timeout', () => {
            const customTimeout = 60000;
            const clientWithTimeout = new AnomalyDetectionClient({
                baseUrl: mockBaseUrl,
                timeout: customTimeout
            });

            expect(clientWithTimeout.axios.defaults.timeout).toBe(customTimeout);
        });

        test('should handle base URL with trailing slash', () => {
            const clientWithSlash = new AnomalyDetectionClient({
                baseUrl: 'http://localhost:8000/'
            });

            expect(clientWithSlash.config.baseUrl).toBe('http://localhost:8000');
        });
    });
});