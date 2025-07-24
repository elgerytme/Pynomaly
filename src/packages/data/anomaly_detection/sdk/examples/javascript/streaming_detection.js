/**
 * Streaming Anomaly Detection Example using JavaScript SDK
 * 
 * This example demonstrates real-time anomaly detection using WebSocket streaming.
 */

const { 
    StreamingClient, 
    AlgorithmType,
    StreamingError 
} = require('@anomaly-detection/sdk');

const readline = require('readline');

/**
 * Example streaming anomaly detection class
 */
class AnomalyDetectionStream {
    constructor(wsUrl = 'ws://localhost:8000/ws/stream') {
        this.wsUrl = wsUrl;
        this.anomalyCount = 0;
        this.totalPoints = 0;
        this.running = false;
        
        // Configure streaming
        const config = {
            wsUrl: wsUrl,
            bufferSize: 50,
            detectionThreshold: 0.6,
            batchSize: 5,
            algorithm: AlgorithmType.ISOLATION_FOREST,
            autoRetrain: false,
            autoReconnect: true,
            reconnectDelay: 3000
        };
        
        // Initialize streaming client
        this.client = new StreamingClient(config);
        this.setupHandlers();
    }
    
    setupHandlers() {
        // Connection handlers
        this.client.on('connect', () => {
            console.log('✅ Connected to streaming service');
            this.running = true;
        });
        
        this.client.on('disconnect', () => {
            console.log('❌ Disconnected from streaming service');
            this.running = false;
        });
        
        // Anomaly detection handler
        this.client.on('anomaly', (anomalyData) => {
            this.anomalyCount++;
            console.log('🚨 ANOMALY DETECTED:');
            console.log(`   Index: ${anomalyData.index}`);
            console.log(`   Score: ${anomalyData.score.toFixed(4)}`);
            console.log(`   Data Point: [${anomalyData.dataPoint.map(x => x.toFixed(3)).join(', ')}]`);
            if (anomalyData.confidence) {
                console.log(`   Confidence: ${anomalyData.confidence.toFixed(4)}`);
            }
            console.log(`   Total anomalies so far: ${this.anomalyCount}`);
            console.log('-'.repeat(50));
        });
        
        // Error handler
        this.client.on('error', (error) => {
            console.error('❌ Streaming error:', error.message);
        });
        
        // Raw message handler (optional)
        this.client.on('message', (data) => {
            // Handle any other message types
            if (data.type !== 'anomaly' && data.type !== 'ping') {
                console.log('📨 Received message:', data);
            }
        });
    }
    
    async startStreaming() {
        console.log('Starting streaming anomaly detection...');
        await this.client.start();
    }
    
    stopStreaming() {
        console.log('Stopping streaming...');
        this.client.stop();
        this.running = false;
    }
    
    sendDataPoint(dataPoint) {
        try {
            this.client.sendData(dataPoint);
            this.totalPoints++;
        } catch (error) {
            console.error('Error sending data:', error.message);
        }
    }
    
    async generateAndSendData(duration = 30000, interval = 1000) {
        console.log(`Generating sample data for ${duration/1000} seconds...`);
        console.log('Normal data will be around [0, 0], anomalies around [5, 5]');
        console.log('-'.repeat(50));
        
        const startTime = Date.now();
        let intervalId;
        
        return new Promise((resolve) => {
            intervalId = setInterval(() => {
                if (Date.now() - startTime >= duration || !this.running) {
                    clearInterval(intervalId);
                    console.log(`\n✅ Sent ${this.totalPoints} data points`);
                    console.log(`✅ Detected ${this.anomalyCount} anomalies`);
                    resolve();
                    return;
                }
                
                // Generate mostly normal data with occasional anomalies
                let dataPoint;
                if (Math.random() < 0.1) { // 10% chance of anomaly
                    // Anomalous data point
                    dataPoint = [
                        Math.random() * 1 + 4.5, // Between 4.5 and 5.5
                        Math.random() * 1 + 4.5
                    ];
                    console.log(`📤 Sending anomalous point: [${dataPoint.map(x => x.toFixed(3)).join(', ')}]`);
                } else {
                    // Normal data point
                    dataPoint = [
                        (Math.random() - 0.5) * 2, // Between -1 and 1
                        (Math.random() - 0.5) * 2
                    ];
                    console.log(`📤 Sending normal point: [${dataPoint.map(x => x.toFixed(3)).join(', ')}]`);
                }
                
                this.sendDataPoint(dataPoint);
            }, interval);
        });
    }
}

/**
 * Interactive streaming example
 */
async function interactiveStreamingExample() {
    console.log('=== Interactive Streaming Example ===');
    
    const stream = new AnomalyDetectionStream();
    await stream.startStreaming();
    
    // Wait for connection
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    if (!stream.running) {
        console.log('Failed to connect to streaming service');
        return;
    }
    
    console.log('\nInteractive mode - Enter data points as comma-separated values');
    console.log('Example: 1.5,2.3 or 5.0,5.0 (for anomaly)');
    console.log('Type "quit" to exit');
    
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
    
    const askForInput = () => {
        rl.question('\nEnter data point (x,y): ', (input) => {
            const trimmed = input.trim();
            
            if (trimmed.toLowerCase() === 'quit' || trimmed.toLowerCase() === 'exit' || trimmed === 'q') {
                rl.close();
                stream.stopStreaming();
                console.log(`\nSession summary:`);
                console.log(`Total points sent: ${stream.totalPoints}`);
                console.log(`Anomalies detected: ${stream.anomalyCount}`);
                return;
            }
            
            try {
                const values = trimmed.split(',').map(x => parseFloat(x.trim()));
                if (values.length !== 2 || values.some(isNaN)) {
                    console.log('Please enter exactly 2 numeric values separated by comma');
                } else {
                    stream.sendDataPoint(values);
                    console.log(`Sent: [${values.join(', ')}]`);
                }
            } catch (error) {
                console.log('Invalid input. Please enter numeric values separated by comma');
            }
            
            // Continue asking for input
            if (stream.running) {
                askForInput();
            }
        });
    };
    
    askForInput();
}

/**
 * Automated streaming example
 */
async function automatedStreamingExample() {
    console.log('=== Automated Streaming Example ===');
    
    const stream = new AnomalyDetectionStream();
    await stream.startStreaming();
    
    // Wait for connection
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    if (!stream.running) {
        console.log('Failed to connect to streaming service');
        return;
    }
    
    try {
        // Run automated data generation
        await stream.generateAndSendData(20000, 500); // 20 seconds, 500ms intervals
    } catch (error) {
        console.error('Automated streaming failed:', error);
    } finally {
        stream.stopStreaming();
    }
}

/**
 * Batch streaming example
 */
async function batchStreamingExample() {
    console.log('=== Batch Streaming Example ===');
    
    // Generate batch data
    const normalBatch = [];
    const anomalyBatch = [];
    
    for (let i = 0; i < 20; i++) {
        normalBatch.push([
            (Math.random() - 0.5) * 2,
            (Math.random() - 0.5) * 2
        ]);
    }
    
    for (let i = 0; i < 5; i++) {
        anomalyBatch.push([
            Math.random() * 1 + 4.5,
            Math.random() * 1 + 4.5
        ]);
    }
    
    const stream = new AnomalyDetectionStream();
    await stream.startStreaming();
    
    // Wait for connection
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    if (!stream.running) {
        console.log('Failed to connect to streaming service');
        return;
    }
    
    try {
        console.log('Sending normal data batch...');
        for (let i = 0; i < normalBatch.length; i++) {
            const point = normalBatch[i];
            stream.sendDataPoint(point);
            console.log(`Sent normal point ${i+1}/20: [${point.map(x => x.toFixed(3)).join(', ')}]`);
            await new Promise(resolve => setTimeout(resolve, 200));
        }
        
        console.log('\nSending anomalous data batch...');
        for (let i = 0; i < anomalyBatch.length; i++) {
            const point = anomalyBatch[i];
            stream.sendDataPoint(point);
            console.log(`Sent anomaly point ${i+1}/5: [${point.map(x => x.toFixed(3)).join(', ')}]`);
            await new Promise(resolve => setTimeout(resolve, 200));
        }
        
        // Wait for processing
        console.log('\nWaiting for processing to complete...');
        await new Promise(resolve => setTimeout(resolve, 5000));
        
    } catch (error) {
        console.error('Batch streaming failed:', error);
    } finally {
        stream.stopStreaming();
    }
}

/**
 * Performance streaming example
 */
async function performanceStreamingExample() {
    console.log('=== Performance Streaming Example ===');
    
    // Configure for high throughput
    const config = {
        wsUrl: 'ws://localhost:8000/ws/stream',
        bufferSize: 100,
        detectionThreshold: 0.5,
        batchSize: 20, // Larger batches for efficiency
        algorithm: AlgorithmType.ISOLATION_FOREST,
        autoReconnect: true
    };
    
    const client = new StreamingClient(config);
    
    // Performance tracking
    let anomalyCount = 0;
    let startTime = null;
    let pointsSent = 0;
    
    client.on('connect', () => {
        console.log('✅ Connected - Starting performance test');
        startTime = Date.now();
    });
    
    client.on('anomaly', (anomalyData) => {
        anomalyCount++;
        if (anomalyCount % 10 === 0) {
            const elapsed = (Date.now() - startTime) / 1000;
            const rate = elapsed > 0 ? pointsSent / elapsed : 0;
            console.log(`📊 Performance: ${anomalyCount} anomalies detected, ${rate.toFixed(1)} points/sec`);
        }
    });
    
    client.on('error', (error) => {
        console.error('❌ Error:', error.message);
    });
    
    try {
        await client.start();
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait for connection
        
        // Generate high volume of data
        console.log('Sending high volume data stream (1000 points)...');
        
        for (let i = 0; i < 1000; i++) {
            if (i % 100 === 0) {
                console.log(`Sent ${i}/1000 points...`);
            }
            
            // Mix of normal and anomalous data
            let dataPoint;
            if (Math.random() < 0.05) { // 5% anomalies
                dataPoint = [
                    Math.random() * 1 + 3.5,
                    Math.random() * 1 + 3.5
                ];
            } else {
                dataPoint = [
                    (Math.random() - 0.5) * 2,
                    (Math.random() - 0.5) * 2
                ];
            }
            
            client.sendData(dataPoint);
            pointsSent++;
            
            // Small delay to avoid overwhelming
            await new Promise(resolve => setTimeout(resolve, 10));
        }
        
        // Wait for final processing
        console.log('Waiting for final processing...');
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        // Calculate final statistics
        const totalTime = (Date.now() - startTime) / 1000;
        const throughput = pointsSent / totalTime;
        
        console.log('\n📊 Performance Results:');
        console.log(`   Total points: ${pointsSent}`);
        console.log(`   Total time: ${totalTime.toFixed(2)} seconds`);
        console.log(`   Throughput: ${throughput.toFixed(1)} points/second`);
        console.log(`   Anomalies detected: ${anomalyCount}`);
        console.log(`   Anomaly rate: ${((anomalyCount/pointsSent)*100).toFixed(2)}%`);
        
    } catch (error) {
        console.error('Performance test failed:', error);
    } finally {
        client.stop();
    }
}

/**
 * Error handling example for streaming
 */
async function streamingErrorHandlingExample() {
    console.log('=== Streaming Error Handling Example ===');
    
    // Test with invalid URL
    const invalidClient = new StreamingClient({
        wsUrl: 'ws://invalid-url:9999/ws/stream',
        autoReconnect: false
    });
    
    let errorReceived = false;
    
    invalidClient.on('error', (error) => {
        console.log('✅ Expected connection error:', error.message);
        errorReceived = true;
    });
    
    try {
        await invalidClient.start();
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        if (!errorReceived) {
            console.log('❌ Expected error but none received');
        }
        
    } catch (error) {
        console.log('✅ Connection failed as expected:', error.message);
    } finally {
        invalidClient.stop();
    }
    
    // Test sending invalid data
    console.log('\nTesting invalid data handling...');
    const validClient = new StreamingClient({
        wsUrl: 'ws://localhost:8000/ws/stream'
    });
    
    try {
        await validClient.start();
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Try to send invalid data
        try {
            validClient.sendData('invalid data');
            console.log('❌ Expected validation error but got success');
        } catch (error) {
            console.log('✅ Validation error caught:', error.message);
        }
        
        try {
            validClient.sendData([]);
            console.log('❌ Expected validation error but got success');
        } catch (error) {
            console.log('✅ Empty array validation error caught:', error.message);
        }
        
    } catch (error) {
        console.log('Connection error (expected if service is down):', error.message);
    } finally {
        validClient.stop();
    }
}

/**
 * Main function to run streaming examples
 */
async function main() {
    console.log('Anomaly Detection JavaScript Streaming Examples');
    console.log('='.repeat(50));
    
    const examples = {
        '1': { name: 'Automated Streaming', func: automatedStreamingExample },
        '2': { name: 'Interactive Streaming', func: interactiveStreamingExample },
        '3': { name: 'Batch Streaming', func: batchStreamingExample },
        '4': { name: 'Performance Test', func: performanceStreamingExample },
        '5': { name: 'Error Handling', func: streamingErrorHandlingExample }
    };
    
    // Check if running in interactive mode
    if (process.argv.includes('--interactive')) {
        console.log('\nAvailable examples:');
        Object.entries(examples).forEach(([key, { name }]) => {
            console.log(`  ${key}. ${name}`);
        });
        
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
        
        rl.question('\nSelect example (1-5) or "all" to run all: ', async (choice) => {
            rl.close();
            
            if (choice.toLowerCase() === 'all') {
                for (const [key, { name, func }] of Object.entries(examples)) {
                    console.log(`\n${'='.repeat(20)} ${name} ${'='.repeat(20)}`);
                    try {
                        await func();
                    } catch (error) {
                        console.error('Example failed:', error.message);
                    }
                    
                    if (key !== '5') { // Don't wait after last example
                        console.log('\nPress Enter to continue to next example...');
                        await new Promise(resolve => {
                            const rl2 = readline.createInterface({
                                input: process.stdin,
                                output: process.stdout
                            });
                            rl2.question('', () => {
                                rl2.close();
                                resolve();
                            });
                        });
                    }
                }
            } else if (examples[choice]) {
                const { name, func } = examples[choice];
                console.log(`\n${'='.repeat(20)} ${name} ${'='.repeat(20)}`);
                await func();
            } else {
                console.log('Invalid choice. Running automated example...');
                await automatedStreamingExample();
            }
            
            console.log('\n' + '='.repeat(50));
            console.log('Streaming examples completed!');
        });
    } else {
        // Run automated example by default
        try {
            await automatedStreamingExample();
        } catch (error) {
            console.error('Example execution failed:', error.message);
            
            if (error instanceof StreamingError || error.message.includes('connect')) {
                console.log('\n💡 Make sure the anomaly detection service is running at ws://localhost:8000/ws/stream');
            }
        }
        
        console.log('\n' + '='.repeat(50));
        console.log('Streaming examples completed!');
    }
}

// Run examples if this file is executed directly
if (require.main === module) {
    main().catch(console.error);
}

module.exports = {
    AnomalyDetectionStream,
    interactiveStreamingExample,
    automatedStreamingExample,
    batchStreamingExample,
    performanceStreamingExample,
    streamingErrorHandlingExample
};