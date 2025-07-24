/**
 * Basic Anomaly Detection Example using JavaScript SDK
 * 
 * This example demonstrates the basic usage of the JavaScript SDK for anomaly detection.
 */

const { 
    AnomalyDetectionClient, 
    AlgorithmType,
    ValidationError,
    APIError,
    ConnectionError 
} = require('@anomaly-detection/sdk');

/**
 * Generate sample data with some anomalies
 */
function generateSampleData() {
    const normalData = [];
    const anomalousData = [];
    
    // Generate normal data points around [0, 0]
    for (let i = 0; i < 100; i++) {
        normalData.push([
            Math.random() * 2 - 1, // Random between -1 and 1
            Math.random() * 2 - 1
        ]);
    }
    
    // Generate anomalous data points around [5, 5]
    for (let i = 0; i < 10; i++) {
        anomalousData.push([
            Math.random() * 2 + 4, // Random between 4 and 6
            Math.random() * 2 + 4
        ]);
    }
    
    // Combine and shuffle
    const allData = [...normalData, ...anomalousData];
    return shuffleArray(allData);
}

/**
 * Shuffle array using Fisher-Yates algorithm
 */
function shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

/**
 * Basic detection example
 */
async function basicDetectionExample() {
    console.log('=== Basic Detection Example ===');
    
    const client = new AnomalyDetectionClient({
        baseUrl: 'http://localhost:8000',
        timeout: 30000
    });
    
    try {
        // Generate sample data
        const data = generateSampleData();
        console.log(`Generated ${data.length} data points`);
        
        // Detect anomalies using Isolation Forest
        console.log('\nDetecting anomalies with Isolation Forest...');
        const result = await client.detectAnomalies(
            data,
            AlgorithmType.ISOLATION_FOREST,
            { contamination: 0.1 },
            false // return explanations
        );
        
        console.log(`Detection completed in ${result.executionTime.toFixed(3)} seconds`);
        console.log(`Found ${result.anomalyCount} anomalies out of ${result.totalPoints} points`);
        
        // Print details of detected anomalies
        result.anomalies.slice(0, 5).forEach((anomaly, i) => {
            console.log(`  Anomaly ${i+1}: Index=${anomaly.index}, Score=${anomaly.score.toFixed(4)}`);
        });
        
        if (result.anomalies.length > 5) {
            console.log(`  ... and ${result.anomalies.length - 5} more`);
        }
        
        return result;
        
    } catch (error) {
        console.error('Detection failed:', error.message);
        throw error;
    }
}

/**
 * Algorithm comparison example
 */
async function algorithmComparisonExample() {
    console.log('\n=== Algorithm Comparison Example ===');
    
    const client = new AnomalyDetectionClient({
        baseUrl: 'http://localhost:8000',
        timeout: 30000
    });
    
    const data = generateSampleData().slice(0, 50); // Use smaller dataset
    console.log(`Using ${data.length} data points for comparison`);
    
    const algorithms = [
        AlgorithmType.ISOLATION_FOREST,
        AlgorithmType.LOCAL_OUTLIER_FACTOR,
        AlgorithmType.ONE_CLASS_SVM,
        AlgorithmType.ENSEMBLE
    ];
    
    const results = [];
    
    for (const algorithm of algorithms) {
        try {
            console.log(`\nTesting ${algorithm}...`);
            const startTime = Date.now();
            
            const result = await client.detectAnomalies(data, algorithm);
            const duration = (Date.now() - startTime) / 1000;
            
            results.push({
                algorithm,
                anomalyCount: result.anomalyCount,
                executionTime: result.executionTime,
                clientTime: duration
            });
            
            console.log(`  ${algorithm}: ${result.anomalyCount} anomalies (${result.executionTime.toFixed(3)}s server, ${duration.toFixed(3)}s total)`);
            
        } catch (error) {
            console.log(`  ${algorithm}: Error - ${error.message}`);
            results.push({
                algorithm,
                error: error.message
            });
        }
    }
    
    // Summary
    console.log('\n--- Algorithm Comparison Summary ---');
    results.forEach(result => {
        if (result.error) {
            console.log(`${result.algorithm}: Failed - ${result.error}`);
        } else {
            console.log(`${result.algorithm}: ${result.anomalyCount} anomalies, ${result.executionTime.toFixed(3)}s`);
        }
    });
    
    return results;
}

/**
 * Batch processing example
 */
async function batchProcessingExample() {
    console.log('\n=== Batch Processing Example ===');
    
    const client = new AnomalyDetectionClient({
        baseUrl: 'http://localhost:8000',
        timeout: 60000 // Longer timeout for batch processing
    });
    
    try {
        // Generate larger dataset
        const normalData = [];
        const anomalousData = [];
        
        for (let i = 0; i < 500; i++) {
            normalData.push([
                Math.random() * 2 - 1,
                Math.random() * 2 - 1,
                Math.random() * 2 - 1
            ]);
        }
        
        for (let i = 0; i < 50; i++) {
            anomalousData.push([
                Math.random() * 2 + 3,
                Math.random() * 2 + 3,
                Math.random() * 2 + 3
            ]);
        }
        
        const data = shuffleArray([...normalData, ...anomalousData]);
        
        // Create batch request
        const batchRequest = {
            data: data,
            algorithm: AlgorithmType.ISOLATION_FOREST,
            parameters: { contamination: 0.1 },
            returnExplanations: true
        };
        
        console.log(`Processing batch of ${data.length} points...`);
        const result = await client.batchDetect(batchRequest);
        
        console.log(`Batch processing completed in ${result.executionTime.toFixed(3)} seconds`);
        console.log(`Found ${result.anomalyCount} anomalies`);
        
        // Analyze results
        if (result.anomalies.length > 0) {
            const scores = result.anomalies.map(a => a.score);
            const minScore = Math.min(...scores);
            const maxScore = Math.max(...scores);
            const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
            
            console.log(`Anomaly scores: min=${minScore.toFixed(4)}, max=${maxScore.toFixed(4)}, avg=${avgScore.toFixed(4)}`);
        }
        
        return result;
        
    } catch (error) {
        console.error('Batch processing failed:', error.message);
        throw error;
    }
}

/**
 * Model management example
 */
async function modelManagementExample() {
    console.log('\n=== Model Management Example ===');
    
    const client = new AnomalyDetectionClient({
        baseUrl: 'http://localhost:8000',
        timeout: 60000
    });
    
    try {
        // Generate training data
        const trainingData = generateSampleData();
        
        // Train a model
        console.log('Training a new model...');
        const trainingRequest = {
            data: trainingData,
            algorithm: AlgorithmType.ISOLATION_FOREST,
            hyperparameters: { contamination: 0.1, n_estimators: 100 },
            validationSplit: 0.2,
            modelName: 'example-model-js'
        };
        
        const trainingResult = await client.trainModel(trainingRequest);
        console.log(`Model trained: ${trainingResult.modelId}`);
        console.log(`Training time: ${trainingResult.trainingTime.toFixed(2)}s`);
        
        // List all models
        console.log('\nListing all models...');
        const models = await client.listModels();
        console.log(`Found ${models.length} models:`);
        
        models.forEach(model => {
            console.log(`  - ${model.modelId}: ${model.algorithm} (${model.status})`);
            console.log(`    Created: ${new Date(model.createdAt).toLocaleString()}`);
        });
        
        // Get specific model info
        if (models.length > 0) {
            const modelId = models[0].modelId;
            console.log(`\nGetting info for model ${modelId}...`);
            const modelInfo = await client.getModel(modelId);
            console.log(`  Algorithm: ${modelInfo.algorithm}`);
            console.log(`  Version: ${modelInfo.version}`);
            console.log(`  Training data size: ${modelInfo.trainingDataSize}`);
            console.log(`  Performance metrics:`, modelInfo.performanceMetrics);
        }
        
        return trainingResult;
        
    } catch (error) {
        console.error('Model management failed:', error.message);
        throw error;
    }
}

/**
 * Health check example
 */
async function healthCheckExample() {
    console.log('\n=== Health Check Example ===');
    
    const client = new AnomalyDetectionClient({
        baseUrl: 'http://localhost:8000',
        timeout: 10000
    });
    
    try {
        // Check service health
        const health = await client.getHealth();
        
        console.log(`Service Status: ${health.status}`);
        console.log(`Version: ${health.version}`);
        console.log(`Uptime: ${health.uptime.toFixed(1)} seconds`);
        
        if (health.components && Object.keys(health.components).length > 0) {
            console.log('Components:');
            Object.entries(health.components).forEach(([component, status]) => {
                console.log(`  ${component}: ${status}`);
            });
        }
        
        // Get metrics
        console.log('\nGetting service metrics...');
        const metrics = await client.getMetrics();
        console.log('Service Metrics:');
        Object.entries(metrics).forEach(([key, value]) => {
            if (typeof value === 'number') {
                console.log(`  ${key}: ${value}`);
            }
        });
        
        return health;
        
    } catch (error) {
        console.error('Health check failed:', error.message);
        throw error;
    }
}

/**
 * Error handling example
 */
async function errorHandlingExample() {
    console.log('\n=== Error Handling Example ===');
    
    const client = new AnomalyDetectionClient({
        baseUrl: 'http://localhost:8000',
        timeout: 5000
    });
    
    // Test different error scenarios
    const errorTests = [
        {
            name: 'Empty data validation',
            test: () => client.detectAnomalies([], AlgorithmType.ISOLATION_FOREST)
        },
        {
            name: 'Invalid data format',
            test: () => client.detectAnomalies(['not', 'numbers'], AlgorithmType.ISOLATION_FOREST)
        },
        {
            name: 'Non-existent model',
            test: () => client.getModel('non-existent-model-id')
        }
    ];
    
    for (const { name, test } of errorTests) {
        try {
            console.log(`\nTesting: ${name}`);
            await test();
            console.log('  ❌ Expected error but got success');
        } catch (error) {
            if (error instanceof ValidationError) {
                console.log(`  ✅ Validation Error: ${error.message}`);
            } else if (error instanceof APIError) {
                console.log(`  ✅ API Error (${error.statusCode}): ${error.message}`);
            } else if (error instanceof ConnectionError) {
                console.log(`  ✅ Connection Error: ${error.message}`);
            } else {
                console.log(`  ⚠️  Unknown Error: ${error.message}`);
            }
        }
    }
}

/**
 * Main function to run all examples
 */
async function main() {
    console.log('Anomaly Detection JavaScript SDK Examples');
    console.log('='.repeat(50));
    
    try {
        // Run basic detection
        await basicDetectionExample();
        
        // Run algorithm comparison
        await algorithmComparisonExample();
        
        // Run batch processing
        await batchProcessingExample();
        
        // Run model management
        await modelManagementExample();
        
        // Run health check
        await healthCheckExample();
        
        // Run error handling
        await errorHandlingExample();
        
    } catch (error) {
        console.error('\n❌ Example execution failed:', error.message);
        
        if (error instanceof ConnectionError) {
            console.log('\n💡 Make sure the anomaly detection service is running at http://localhost:8000');
        }
    }
    
    console.log('\n' + '='.repeat(50));
    console.log('Examples completed!');
}

// Run examples if this file is executed directly
if (require.main === module) {
    main().catch(console.error);
}

module.exports = {
    generateSampleData,
    basicDetectionExample,
    algorithmComparisonExample,
    batchProcessingExample,
    modelManagementExample,
    healthCheckExample,
    errorHandlingExample
};