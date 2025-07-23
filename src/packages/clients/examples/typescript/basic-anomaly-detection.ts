#!/usr/bin/env npx ts-node
/**
 * Basic anomaly detection example using TypeScript SDK
 * 
 * This example demonstrates:
 * - Simple anomaly detection
 * - Error handling
 * - Configuration options
 * - Type safety with TypeScript
 */

import { PlatformClient, Environment, ValidationError, RateLimitError, ServerError } from '../../platform_client_ts/src';

// Sample data generator
function createSampleData(): number[][] {
  const data: number[][] = [];
  
  // Generate normal data points (around 0-10 range)
  for (let i = 0; i < 100; i++) {
    const x = Math.random() * 10;
    const y = Math.random() * 10;
    data.push([x, y]);
  }
  
  // Add some clear anomalies
  const anomalies: number[][] = [
    [100, 100],  // Far away anomaly
    [-50, -50],  // Another far away anomaly
    [200, 5],    // High X value
    [5, 200],    // High Y value
  ];
  
  // Combine and shuffle
  const allData = [...data, ...anomalies];
  
  // Shuffle array
  for (let i = allData.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [allData[i], allData[j]] = [allData[j], allData[i]];
  }
  
  return allData;
}

async function basicDetectionExample() {
  console.log('🔍 Basic Anomaly Detection Example');
  console.log('=' .repeat(40));
  
  // Initialize client
  const client = new PlatformClient({
    apiKey: process.env.ANOMALY_DETECTION_API_KEY || 'demo-api-key',
    environment: Environment.Local,
    timeout: 30000,
    maxRetries: 3
  });
  
  try {
    // Check service health
    console.log('📊 Checking service health...');
    const health = await client.healthCheck();
    console.log(`   Status: ${health.anomalyDetection?.status || 'unknown'}`);
    console.log();
    
    // Get available algorithms
    console.log('🤖 Available algorithms:');
    try {
      const algorithms = await client.anomalyDetection.getAlgorithms();
      algorithms.slice(0, 3).forEach(algo => {
        console.log(`   - ${algo.displayName}: ${algo.description}`);
      });
    } catch (error) {
      console.log(`   Could not fetch algorithms: ${error}`);
    }
    console.log();
    
    // Create sample data
    const data = createSampleData();
    console.log(`📈 Generated ${data.length} data points`);
    
    // Perform basic anomaly detection
    console.log('🔍 Detecting anomalies...');
    const result = await client.anomalyDetection.detect({
      data,
      algorithm: 'isolation_forest',
      contamination: 0.1,
      parameters: { n_estimators: 100 }
    });
    
    // Display results with type safety
    console.log(`✅ Detection completed in ${result.processingTimeMs.toFixed(2)}ms`);
    console.log(`   Total samples: ${result.totalSamples}`);
    console.log(`   Anomalies found: ${result.anomalyCount}`);
    console.log(`   Anomaly indices: ${result.anomalies.slice(0, 10).join(', ')}...`);
    
    if (result.scores) {
      const avgScore = result.scores.reduce((a, b) => a + b, 0) / result.scores.length;
      const maxScore = Math.max(...result.scores);
      console.log(`   Average anomaly score: ${avgScore.toFixed(4)}`);
      console.log(`   Maximum anomaly score: ${maxScore.toFixed(4)}`);
    }
    console.log();
    
    // Try ensemble detection
    console.log('🎯 Ensemble detection with multiple algorithms...');
    try {
      const ensembleResult = await client.anomalyDetection.detectEnsemble({
        data: data.slice(0, 50), // Use smaller dataset
        algorithms: ['isolation_forest', 'one_class_svm'],
        votingStrategy: 'majority',
        contamination: 0.1
      });
      
      console.log(`✅ Ensemble detection completed in ${ensembleResult.processingTimeMs.toFixed(2)}ms`);
      console.log(`   Algorithms used: ${ensembleResult.algorithmsUsed.join(', ')}`);
      console.log(`   Anomalies found: ${ensembleResult.anomalyCount}`);
      console.log(`   Voting strategy: ${ensembleResult.votingStrategy}`);
      
    } catch (error) {
      console.log(`   Ensemble detection failed: ${error}`);
    }
    console.log();
    
    // Try model training
    console.log('🎓 Training a custom model...');
    try {
      const trainingResult = await client.anomalyDetection.trainModel({
        data: data.slice(0, 80), // Use 80% for training
        algorithm: 'isolation_forest',
        name: `demo_model_${Date.now()}`,
        description: 'Demo model trained from TypeScript example',
        contamination: 0.1,
        parameters: { n_estimators: 50 }
      });
      
      const model = trainingResult.model;
      console.log('✅ Model trained successfully');
      console.log(`   Model ID: ${model.id}`);
      console.log(`   Model name: ${model.name}`);
      console.log(`   Algorithm: ${model.algorithm}`);
      console.log(`   Training time: ${trainingResult.trainingTimeMs.toFixed(2)}ms`);
      
      // Use the trained model for prediction
      console.log('🔮 Making predictions with trained model...');
      const predictionResult = await client.anomalyDetection.predict({
        data: data.slice(80), // Use remaining 20% for prediction
        modelId: model.id
      });
      
      console.log('✅ Prediction completed');
      console.log(`   Anomalies found: ${predictionResult.anomalyCount}`);
      console.log(`   Processing time: ${predictionResult.processingTimeMs.toFixed(2)}ms`);
      
    } catch (error) {
      console.log(`   Model training/prediction failed: ${error}`);
    }
    console.log();
    
  } catch (error) {
    if (error instanceof ValidationError) {
      console.log(`❌ Validation error: ${error.message}`);
      if (error.details) {
        error.details.forEach(detail => {
          console.log(`   - ${detail.field}: ${detail.message}`);
        });
      }
    } else if (error instanceof RateLimitError) {
      console.log(`⏳ Rate limited: ${error.message}`);
      console.log(`   Retry after: ${error.retryAfter} seconds`);
    } else if (error instanceof ServerError) {
      console.log(`🚨 Server error: ${error.message}`);
    } else {
      console.log(`💥 Unexpected error: ${error}`);
    }
  } finally {
    await client.close();
  }
}

async function batchProcessingExample() {
  console.log('📦 Batch Processing Example');
  console.log('=' .repeat(30));
  
  const client = new PlatformClient({
    apiKey: process.env.ANOMALY_DETECTION_API_KEY || 'demo-api-key',
    environment: Environment.Local,
  });
  
  try {
    // Create multiple datasets
    const datasets = [];
    for (let i = 0; i < 3; i++) {
      datasets.push({
        id: `dataset_${i + 1}`,
        data: createSampleData().slice(0, 30) // Smaller datasets for batch
      });
    }
    
    console.log(`📊 Processing ${datasets.length} datasets in batch...`);
    
    const batchResult = await client.anomalyDetection.batchDetect({
      datasets,
      algorithm: 'isolation_forest',
      contamination: 0.1,
      parallelProcessing: true
    });
    
    console.log('✅ Batch processing completed');
    console.log(`   Total datasets: ${batchResult.totalDatasets}`);
    console.log(`   Successful: ${batchResult.successfulCount}`);
    console.log(`   Failed: ${batchResult.failedCount}`);
    console.log(`   Total time: ${batchResult.totalProcessingTimeMs.toFixed(2)}ms`);
    
    // Show results for each dataset
    batchResult.results.forEach((result, index) => {
      if (result.error) {
        console.log(`   Dataset ${index + 1}: Error - ${result.error}`);
      } else {
        console.log(`   Dataset ${index + 1}: ${result.result.anomalyCount} anomalies found`);
      }
    });
    
  } catch (error) {
    console.log(`💥 Batch processing failed: ${error}`);
  } finally {
    await client.close();
  }
}

async function modelManagementExample() {
  console.log('\n🎯 Model Management Example');
  console.log('=' .repeat(30));
  
  const client = new PlatformClient({
    apiKey: process.env.ANOMALY_DETECTION_API_KEY || 'demo-api-key',
    environment: Environment.Local,
  });
  
  try {
    // List existing models
    console.log('📋 Listing existing models...');
    const modelsList = await client.anomalyDetection.listModels({
      page: 1,
      pageSize: 10
    });
    
    console.log(`   Found ${modelsList.pagination.totalItems} total models`);
    console.log(`   Showing page ${modelsList.pagination.page} of ${modelsList.pagination.totalPages}`);
    
    if (modelsList.data.length > 0) {
      console.log('   Recent models:');
      modelsList.data.slice(0, 3).forEach(model => {
        console.log(`   - ${model.name} (${model.algorithm}) - Status: ${model.status}`);
      });
      
      // Get detailed info for first model
      const firstModel = modelsList.data[0];
      console.log(`\n🔍 Detailed info for model: ${firstModel.name}`);
      const modelDetails = await client.anomalyDetection.getModel(firstModel.id);
      console.log(`   ID: ${modelDetails.id}`);
      console.log(`   Algorithm: ${modelDetails.algorithm}`);
      console.log(`   Created: ${modelDetails.createdAt}`);
      console.log(`   Training samples: ${modelDetails.trainingSamples}`);
      console.log(`   Contamination: ${modelDetails.contamination}`);
    } else {
      console.log('   No models found');
    }
    
  } catch (error) {
    console.log(`💥 Model management failed: ${error}`);
  } finally {
    await client.close();
  }
}

async function main() {
  console.log('🚀 Platform TypeScript SDK Examples');
  console.log('=' .repeat(50));
  
  // Check API key
  const apiKey = process.env.ANOMALY_DETECTION_API_KEY;
  if (!apiKey) {
    console.log('⚠️  No API key found in ANOMALY_DETECTION_API_KEY environment variable');
    console.log('   Using demo API key - some features may not work');
    console.log();
  }
  
  try {
    await basicDetectionExample();
    await batchProcessingExample();
    await modelManagementExample();
    
    console.log('\n✨ All examples completed successfully!');
    
  } catch (error) {
    console.log(`\n💥 Examples failed: ${error}`);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

export { main as runExamples };