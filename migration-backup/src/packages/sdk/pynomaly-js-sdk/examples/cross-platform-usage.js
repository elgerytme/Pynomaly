/**
 * Cross-platform usage example
 * This example demonstrates how to use the SDK in both browser and Node.js environments
 */

// Import the SDK
const { 
  PynomalyClient, 
  AuthManager, 
  Environment, 
  CompatibilityTest,
  setupPolyfills 
} = require('../dist/index.js'); // Adjust path as needed

// Ensure polyfills are set up
setupPolyfills();

// Check environment and compatibility
console.log('🌍 Environment Detection:');
console.log('Platform:', Environment.getPlatform());
console.log('Is Browser:', Environment.isBrowser());
console.log('Is Node.js:', Environment.isNode());
console.log('Has WebSocket:', Environment.hasWebSocket());
console.log('Has Local Storage:', Environment.hasLocalStorage());
console.log('Has Crypto:', Environment.hasCrypto());

// Run compatibility test
console.log('\n🔍 Running Compatibility Test:');
CompatibilityTest.logCompatibilityReport();

// Test basic functionality
console.log('\n🧪 Testing Basic Functionality:');
CompatibilityTest.testBasicFunctionality().then(success => {
  if (success) {
    console.log('✅ Basic functionality test passed');
    runSDKExample();
  } else {
    console.log('❌ Basic functionality test failed');
  }
});

async function runSDKExample() {
  console.log('\n🚀 Running SDK Example:');
  
  try {
    // Initialize client
    const client = new PynomalyClient({
      apiKey: 'demo-api-key',
      baseUrl: 'https://api.pynomaly.com',
      debug: true
    });

    // Initialize auth manager
    const authManager = new AuthManager({
      enablePersistence: true,
      autoRefresh: true
    });

    // Listen for auth events
    authManager.on('auth:login', ({ user }) => {
      console.log('✅ User logged in:', user.email);
    });

    authManager.on('auth:error', ({ error, type }) => {
      console.log(`❌ Auth error (${type}):`, error);
    });

    // Example: Simulate login (you would use real credentials)
    console.log('🔐 Simulating authentication...');
    try {
      // This would normally authenticate with real credentials
      // await authManager.login({ email: 'user@example.com', password: 'password' }, client);
      console.log('ℹ️ Authentication skipped (demo mode)');
    } catch (error) {
      console.log('ℹ️ Authentication failed (expected in demo mode)');
    }

    // Example: Check auth state
    console.log('🔍 Current auth state:');
    console.log('- Authenticated:', authManager.isAuthenticated());
    console.log('- User:', authManager.getUser());
    console.log('- Token:', authManager.getToken() ? 'Present' : 'Not present');

    // Example: Test anomaly detection (this would normally require authentication)
    console.log('\n🔍 Testing anomaly detection...');
    const testData = [
      [1, 2, 3],
      [2, 3, 4],
      [3, 4, 5],
      [100, 200, 300] // This should be detected as an anomaly
    ];

    try {
      // This would normally work with a real API
      // const result = await client.detectAnomalies({
      //   data: testData,
      //   algorithm: 'isolation_forest'
      // });
      // console.log('✅ Anomaly detection result:', result);
      console.log('ℹ️ Anomaly detection skipped (demo mode)');
    } catch (error) {
      console.log('ℹ️ Anomaly detection failed (expected in demo mode)');
    }

    // Example: Test WebSocket connectivity
    if (Environment.hasWebSocket()) {
      console.log('\n🔗 Testing WebSocket connectivity...');
      try {
        const { PynomalyWebSocket } = require('../dist/index.js');
        const ws = new PynomalyWebSocket({
          url: 'wss://api.pynomaly.com/ws',
          debug: true
        });

        ws.on('connection:open', () => {
          console.log('✅ WebSocket connected');
          ws.disconnect();
        });

        ws.on('connection:error', (error) => {
          console.log('ℹ️ WebSocket connection failed (expected in demo mode)');
        });

        // This would normally connect to a real WebSocket
        // await ws.connect();
        console.log('ℹ️ WebSocket connection skipped (demo mode)');
      } catch (error) {
        console.log('ℹ️ WebSocket test failed (expected in demo mode)');
      }
    } else {
      console.log('⚠️ WebSocket not available in this environment');
    }

    console.log('\n✅ SDK example completed successfully!');

  } catch (error) {
    console.error('❌ SDK example failed:', error);
  }
}

// Platform-specific examples
if (Environment.isBrowser()) {
  console.log('\n🌐 Browser-specific features:');
  
  // Example: Using localStorage
  try {
    localStorage.setItem('pynomaly-test', 'browser-value');
    console.log('✅ localStorage available');
  } catch (e) {
    console.log('❌ localStorage not available');
  }

  // Example: Using fetch
  if (typeof fetch !== 'undefined') {
    console.log('✅ Fetch API available');
  } else {
    console.log('❌ Fetch API not available');
  }

  // Example: Using WebSocket
  if (typeof WebSocket !== 'undefined') {
    console.log('✅ WebSocket API available');
  } else {
    console.log('❌ WebSocket API not available');
  }
}

if (Environment.isNode()) {
  console.log('\n🖥️ Node.js-specific features:');
  
  // Example: Using file system
  try {
    const fs = require('fs');
    console.log('✅ File system access available');
  } catch (e) {
    console.log('❌ File system access not available');
  }

  // Example: Using crypto
  try {
    const crypto = require('crypto');
    console.log('✅ Crypto module available');
  } catch (e) {
    console.log('❌ Crypto module not available');
  }

  // Example: Using WebSocket (ws package)
  try {
    const WS = require('ws');
    console.log('✅ ws package available');
  } catch (e) {
    console.log('❌ ws package not available (install with: npm install ws)');
  }
}

console.log('\n🎉 Cross-platform example completed!');
console.log('💡 This example demonstrates how the SDK adapts to different environments');
console.log('💡 In production, you would use real API credentials and endpoints');