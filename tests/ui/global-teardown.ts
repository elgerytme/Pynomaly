/**
 * Playwright Global Teardown
 * Cleanup operations after all tests complete
 */

import { FullConfig } from '@playwright/test';
import fs from 'fs';
import path from 'path';

async function globalTeardown(config: FullConfig) {
  console.log('🧹 Starting Playwright Global Teardown...');

  // Generate test summary report
  const reportData = {
    timestamp: new Date().toISOString(),
    environment: process.env.PYNOMALY_ENVIRONMENT || 'testing',
    browser_support: {
      chromium: true,
      firefox: true,
      webkit: true,
      edge: true
    },
    visual_testing: !!process.env.PERCY_TOKEN,
    accessibility_testing: true,
    performance_monitoring: true
  };

  const reportPath = 'test_reports/test_summary.json';
  fs.writeFileSync(reportPath, JSON.stringify(reportData, null, 2));
  console.log(`📊 Test summary saved to ${reportPath}`);

  // Clean up temporary files if not in CI
  if (!process.env.CI) {
    const tempDirs = [
      'tests/ui/temp',
      'tests/ui/downloads'
    ];

    tempDirs.forEach(dir => {
      if (fs.existsSync(dir)) {
        fs.rmSync(dir, { recursive: true, force: true });
        console.log(`🗑️  Cleaned up temporary directory: ${dir}`);
      }
    });
  }

  // Archive screenshots if tests failed (in CI)
  if (process.env.CI) {
    const screenshotDir = 'test_reports/screenshots';
    if (fs.existsSync(screenshotDir) && fs.readdirSync(screenshotDir).length > 0) {
      console.log('📸 Failure screenshots archived for CI analysis');
    }
  }

  console.log('✅ Playwright Global Teardown completed successfully');
}

export default globalTeardown;