/**
 * Getting Started Example
 * 
 * This example demonstrates the basic usage of the Pynomaly JavaScript SDK,
 * including client initialization, authentication, and simple anomaly detection.
 */

const { PynomalyClient } = require('@pynomaly/js-sdk');

async function gettingStartedExample() {
  console.log('🚀 Pynomaly JavaScript SDK - Getting Started Example\n');

  // Step 1: Initialize the client
  console.log('1. Initializing Pynomaly client...');
  const client = new PynomalyClient({
    apiKey: process.env.PYNOMALY_API_KEY || 'your-api-key',
    baseUrl: process.env.PYNOMALY_BASE_URL || 'https://api.pynomaly.com',
    timeout: 30000,
    debug: true
  });

  try {
    // Step 2: Authenticate with API key
    console.log('2. Authenticating with API key...');
    const authResult = await client.authenticateWithApiKey(
      process.env.PYNOMALY_API_KEY || 'your-api-key'
    );
    console.log('✅ Authentication successful!');
    console.log('   Token expires at:', authResult.token.expiresAt);

    // Step 3: Prepare sample data
    console.log('\n3. Preparing sample data...');
    const sampleData = [
      [1.2, 2.1, 0.5],    // Normal data point
      [1.5, 2.3, 0.7],    // Normal data point
      [1.1, 1.9, 0.6],    // Normal data point
      [1.4, 2.2, 0.5],    // Normal data point
      [10.0, 20.0, 5.0],  // Anomaly - significantly different
      [1.3, 2.0, 0.6],    // Normal data point
      [1.6, 2.4, 0.8]     // Normal data point
    ];
    console.log('   Sample data prepared with', sampleData.length, 'data points');

    // Step 4: Detect anomalies
    console.log('\n4. Detecting anomalies...');
    const anomalyResult = await client.detectAnomalies({
      data: sampleData,
      algorithm: 'isolation_forest',
      parameters: {
        contamination: 0.2,  // Expect ~20% anomalies
        n_estimators: 100    // Number of trees in the forest
      }
    });

    // Step 5: Display results
    console.log('\n📊 Anomaly Detection Results:');
    console.log('   Algorithm used:', anomalyResult.algorithm);
    console.log('   Total data points:', anomalyResult.metrics.totalPoints);
    console.log('   Anomalies detected:', anomalyResult.metrics.anomalyCount);
    console.log('   Anomaly rate:', (anomalyResult.metrics.anomalyRate * 100).toFixed(1) + '%');
    console.log('   Processing time:', anomalyResult.processingTime + 'ms');

    // Step 6: Display individual anomalies
    if (anomalyResult.anomalies.length > 0) {
      console.log('\n🔍 Detected Anomalies:');
      anomalyResult.anomalies.forEach((anomaly, index) => {
        console.log(`   Anomaly ${index + 1}:`);
        console.log(`     Index: ${anomaly.index}`);
        console.log(`     Data: [${anomaly.data.join(', ')}]`);
        console.log(`     Score: ${anomaly.score.toFixed(3)}`);
        console.log(`     Confidence: ${(anomaly.confidence * 100).toFixed(1)}%`);
        console.log(`     Explanation: ${anomaly.explanation}`);
      });
    }

    // Step 7: Optional - Check API health
    console.log('\n5. Checking API health...');
    const healthStatus = await client.healthCheck();
    console.log('   API Status:', healthStatus.status);
    console.log('   Response time:', healthStatus.responseTime + 'ms');

    console.log('\n✅ Getting started example completed successfully!');

  } catch (error) {
    console.error('\n❌ Error occurred during example execution:');
    console.error('   Error type:', error.constructor.name);
    console.error('   Message:', error.message);
    
    if (error.response) {
      console.error('   HTTP Status:', error.response.status);
      console.error('   Response data:', error.response.data);
    }
    
    // Handle specific error types
    if (error.code === 'AUTH_TOKEN_EXPIRED') {
      console.error('\n💡 Tip: Your authentication token has expired. Please refresh it.');
    } else if (error.code === 'NETWORK_ERROR') {
      console.error('\n💡 Tip: Check your network connection and API endpoint.');
    } else if (error.code === 'INVALID_API_KEY') {
      console.error('\n💡 Tip: Verify your API key is correct and has proper permissions.');
    }

  } finally {
    // Clean up resources
    console.log('\n6. Cleaning up resources...');
    client.disconnect();
    console.log('   Client disconnected.');
  }
}

// Additional helper function to demonstrate event handling
function setupEventHandlers(client) {
  console.log('Setting up event handlers...');

  // Listen for authentication events
  client.on('authenticated', (authResult) => {
    console.log('🔐 Authentication event: User authenticated');
  });

  // Listen for request events
  client.on('requestStart', (config) => {
    console.log('📤 Request started:', config.method?.toUpperCase(), config.url);
  });

  client.on('requestComplete', (response) => {
    console.log('📥 Request completed:', response.status, response.statusText);
  });

  // Listen for error events
  client.on('error', (error) => {
    console.error('⚠️  SDK Error:', error.message);
  });
}

// Run the example if this file is executed directly
if (require.main === module) {
  // Check for required environment variables
  if (!process.env.PYNOMALY_API_KEY) {
    console.log('⚠️  Warning: PYNOMALY_API_KEY environment variable not set.');
    console.log('   Using placeholder API key for demonstration.');
    console.log('   Set your API key with: export PYNOMALY_API_KEY="your-actual-api-key"');
  }

  gettingStartedExample()
    .then(() => {
      console.log('\n🎉 Example finished successfully!');
      process.exit(0);
    })
    .catch((error) => {
      console.error('\n💥 Example failed:', error.message);
      process.exit(1);
    });
}

module.exports = { gettingStartedExample, setupEventHandlers };