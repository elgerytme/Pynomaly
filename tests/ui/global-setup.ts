/**
 * Playwright Global Setup
 * Prepares the testing environment before all tests run
 */

import { FullConfig } from '@playwright/test';
import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

async function globalSetup(config: FullConfig) {
  console.log('🚀 Starting Playwright Global Setup...');

  // Create test directories if they don't exist
  const testDirs = [
    'test_reports',
    'test_reports/screenshots',
    'test_reports/videos',
    'test_reports/traces',
    'test_reports/percy',
    'tests/ui/screenshots/actual',
    'tests/ui/screenshots/expected',
    'tests/ui/screenshots/diff'
  ];

  testDirs.forEach(dir => {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
      console.log(`📁 Created directory: ${dir}`);
    }
  });

  // Set environment variables for testing
  process.env.PYNOMALY_ENVIRONMENT = 'testing';
  process.env.PYNOMALY_LOG_LEVEL = 'WARNING';
  process.env.PYNOMALY_AUTH_ENABLED = 'false';
  process.env.PYNOMALY_DOCS_ENABLED = 'true';
  process.env.PYNOMALY_CORS_ENABLED = 'true';

  // Initialize Percy if PERCY_TOKEN is available
  if (process.env.PERCY_TOKEN) {
    console.log('🎨 Percy visual testing enabled');
  } else {
    console.log('ℹ️  Percy visual testing disabled (PERCY_TOKEN not set)');
  }

  // Build web assets if needed
  if (!fs.existsSync('src/pynomaly/presentation/web/static/css/styles.css')) {
    console.log('🔨 Building web assets...');
    try {
      execSync('npm run build-css', { stdio: 'inherit' });
    } catch (error) {
      console.warn('⚠️  Failed to build CSS assets, continuing with tests');
    }
  }

  // Verify test data directory
  const testDataDir = 'tests/ui/test_data';
  if (!fs.existsSync(testDataDir)) {
    fs.mkdirSync(testDataDir, { recursive: true });
    
    // Create sample test data
    const sampleData = {
      datasets: [
        { id: 1, name: 'Sample Dataset', type: 'tabular', size: 1000 },
        { id: 2, name: 'Time Series Data', type: 'time_series', size: 5000 }
      ],
      detectors: [
        { id: 1, name: 'Isolation Forest', type: 'pyod', status: 'trained' },
        { id: 2, name: 'LSTM Autoencoder', type: 'deep_learning', status: 'ready' }
      ]
    };
    
    fs.writeFileSync(
      path.join(testDataDir, 'sample_data.json'),
      JSON.stringify(sampleData, null, 2)
    );
  }

  console.log('✅ Playwright Global Setup completed successfully');
}

export default globalSetup;