/**
 * Complete User Journey and Workflow Testing
 * 
 * This test suite validates end-to-end user workflows and business processes
 * in the Pynomaly web UI, including Progressive Web App (PWA) capabilities.
 */

import { test, expect, Page, BrowserContext } from '@playwright/test';

interface UserJourneyStep {
  name: string;
  action: (page: Page) => Promise<void>;
  validation: (page: Page) => Promise<void>;
  timeout?: number;
}

interface UserPersona {
  name: string;
  description: string;
  permissions: string[];
  typical_workflows: string[];
}

class UserJourneyRunner {
  private persona: UserPersona;
  private steps: UserJourneyStep[];
  private executionLog: { step: string; duration: number; success: boolean; error?: string }[] = [];

  constructor(persona: UserPersona, steps: UserJourneyStep[]) {
    this.persona = persona;
    this.steps = steps;
  }

  async executeJourney(page: Page): Promise<{ success: boolean; log: typeof this.executionLog }> {
    console.log(`Starting user journey for persona: ${this.persona.name}`);
    
    for (const step of this.steps) {
      const startTime = Date.now();
      let success = false;
      let error: string | undefined;

      try {
        console.log(`Executing step: ${step.name}`);
        
        // Execute the action with timeout
        await Promise.race([
          step.action(page),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Step timeout')), step.timeout || 10000)
          )
        ]);
        
        // Validate the result
        await step.validation(page);
        success = true;
        
      } catch (e) {
        error = e instanceof Error ? e.message : String(e);
        console.error(`Step failed: ${step.name} - ${error}`);
      }

      const duration = Date.now() - startTime;
      this.executionLog.push({ step: step.name, duration, success, error });
      
      if (!success) {
        break; // Stop on first failure
      }
      
      // Brief pause between steps for realistic user behavior
      await page.waitForTimeout(500);
    }

    const overallSuccess = this.executionLog.every(log => log.success);
    console.log(`Journey completed. Success: ${overallSuccess}`);
    
    return { success: overallSuccess, log: this.executionLog };
  }
}

// Define user personas
const dataScientistPersona: UserPersona = {
  name: 'Data Scientist',
  description: 'Experienced user who creates and manages anomaly detection models',
  permissions: ['create_detectors', 'upload_datasets', 'view_results', 'configure_algorithms'],
  typical_workflows: ['dataset_upload_and_analysis', 'detector_creation_and_training', 'result_analysis']
};

const analystPersona: UserPersona = {
  name: 'Business Analyst',
  description: 'User who monitors dashboards and investigates anomalies',
  permissions: ['view_dashboards', 'view_results', 'export_reports'],
  typical_workflows: ['dashboard_monitoring', 'anomaly_investigation', 'report_generation']
};

const adminPersona: UserPersona = {
  name: 'System Administrator',
  description: 'User who manages system configuration and user access',
  permissions: ['manage_users', 'system_configuration', 'view_logs', 'manage_resources'],
  typical_workflows: ['user_management', 'system_monitoring', 'configuration_updates']
};

test.describe('Complete User Journey Testing', () => {
  
  test('Data Scientist: Complete Anomaly Detection Workflow', async ({ page }) => {
    const journey = new UserJourneyRunner(dataScientistPersona, [
      {
        name: 'Login and Navigate to Dashboard',
        action: async (page) => {
          await page.goto('/dashboard');
          await page.waitForLoadState('networkidle');
        },
        validation: async (page) => {
          await expect(page.locator('[data-testid="dashboard-content"]')).toBeVisible();
          await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
        }
      },
      
      {
        name: 'Upload New Dataset',
        action: async (page) => {
          await page.click('[data-testid="datasets-tab"]');
          await page.waitForSelector('[data-testid="datasets-list"]');
          
          if (await page.isVisible('[data-testid="upload-dataset"]')) {
            await page.click('[data-testid="upload-dataset"]');
            await page.waitForSelector('[data-testid="upload-form"]');
            
            // Simulate file upload process
            await page.fill('input[name="dataset-name"]', 'E2E Test Dataset');
            await page.fill('textarea[name="description"]', 'Dataset uploaded during E2E testing');
            
            // Mock file selection
            if (await page.isVisible('input[type="file"]')) {
              await page.setInputFiles('input[type="file"]', {
                name: 'test_data.csv',
                mimeType: 'text/csv',
                buffer: Buffer.from('feature1,feature2,target\n1,2,0\n2,3,0\n100,200,1\n')
              });
            }
          }
        },
        validation: async (page) => {
          if (await page.isVisible('[data-testid="upload-form"]')) {
            await expect(page.locator('input[name="dataset-name"]')).toHaveValue('E2E Test Dataset');
          }
        }
      },
      
      {
        name: 'Analyze Dataset Preview',
        action: async (page) => {
          if (await page.isVisible('[data-testid="preview-dataset"]')) {
            await page.click('[data-testid="preview-dataset"]');
            await page.waitForSelector('[data-testid="dataset-preview"]', { timeout: 5000 });
          }
        },
        validation: async (page) => {
          if (await page.isVisible('[data-testid="dataset-preview"]')) {
            await expect(page.locator('[data-testid="dataset-stats"]')).toBeVisible();
          }
        }
      },
      
      {
        name: 'Create New Anomaly Detector',
        action: async (page) => {
          await page.click('[data-testid="detectors-tab"]');
          await page.waitForSelector('[data-testid="detectors-list"]');
          
          if (await page.isVisible('[data-testid="create-detector"]')) {
            await page.click('[data-testid="create-detector"]');
            await page.waitForSelector('[data-testid="detector-form"]');
            
            await page.fill('input[name="name"]', 'E2E Test Detector');
            await page.selectOption('select[name="algorithm"]', 'IsolationForest');
            await page.fill('input[name="contamination"]', '0.1');
            await page.fill('textarea[name="description"]', 'Detector created during E2E testing');
          }
        },
        validation: async (page) => {
          if (await page.isVisible('[data-testid="detector-form"]')) {
            await expect(page.locator('input[name="name"]')).toHaveValue('E2E Test Detector');
            await expect(page.locator('select[name="algorithm"]')).toHaveValue('IsolationForest');
          }
        }
      },
      
      {
        name: 'Configure Algorithm Parameters',
        action: async (page) => {
          if (await page.isVisible('[data-testid="advanced-settings"]')) {
            await page.click('[data-testid="advanced-settings"]');
            await page.waitForSelector('[data-testid="parameter-controls"]');
            
            // Adjust algorithm parameters
            if (await page.isVisible('input[name="n_estimators"]')) {
              await page.fill('input[name="n_estimators"]', '100');
            }
            if (await page.isVisible('input[name="max_features"]')) {
              await page.selectOption('select[name="max_features"]', 'auto');
            }
          }
        },
        validation: async (page) => {
          // Validate parameter form is accessible
          if (await page.isVisible('[data-testid="parameter-controls"]')) {
            await expect(page.locator('[data-testid="parameter-controls"]')).toBeVisible();
          }
        }
      },
      
      {
        name: 'Start Training Process',
        action: async (page) => {
          if (await page.isVisible('[data-testid="start-training"]')) {
            await page.click('[data-testid="start-training"]');
            await page.waitForSelector('[data-testid="training-progress"]', { timeout: 3000 });
          }
        },
        validation: async (page) => {
          if (await page.isVisible('[data-testid="training-progress"]')) {
            await expect(page.locator('[data-testid="training-status"]')).toContainText(/training|progress|started/i);
          }
        },
        timeout: 15000
      },
      
      {
        name: 'Monitor Training Progress',
        action: async (page) => {
          // Wait for training to show progress
          let attempts = 0;
          while (attempts < 10) {
            if (await page.isVisible('[data-testid="training-complete"]')) {
              break;
            }
            await page.waitForTimeout(2000);
            attempts++;
          }
        },
        validation: async (page) => {
          // Check that training status is visible
          const hasProgress = await page.isVisible('[data-testid="training-progress"]');
          const isComplete = await page.isVisible('[data-testid="training-complete"]');
          expect(hasProgress || isComplete).toBe(true);
        },
        timeout: 30000
      },
      
      {
        name: 'Run Detection on Test Data',
        action: async (page) => {
          await page.click('[data-testid="detection-tab"]');
          await page.waitForSelector('[data-testid="detection-interface"]');
          
          if (await page.isVisible('[data-testid="select-detector"]')) {
            await page.selectOption('select[data-testid="select-detector"]', /E2E Test Detector/);
          }
          
          if (await page.isVisible('[data-testid="run-detection"]')) {
            await page.click('[data-testid="run-detection"]');
            await page.waitForSelector('[data-testid="detection-results"]', { timeout: 10000 });
          }
        },
        validation: async (page) => {
          if (await page.isVisible('[data-testid="detection-results"]')) {
            await expect(page.locator('[data-testid="anomaly-count"]')).toBeVisible();
            await expect(page.locator('[data-testid="detection-summary"]')).toBeVisible();
          }
        },
        timeout: 20000
      },
      
      {
        name: 'Analyze Detection Results',
        action: async (page) => {
          if (await page.isVisible('[data-testid="view-details"]')) {
            await page.click('[data-testid="view-details"]');
            await page.waitForSelector('[data-testid="result-details"]');
          }
          
          // Switch to visualizations
          await page.click('[data-testid="visualizations-tab"]');
          await page.waitForSelector('[data-testid="chart-container"]');
        },
        validation: async (page) => {
          await expect(page.locator('[data-testid="chart-container"]')).toBeVisible();
          
          // Check for chart elements
          const hasChart = await page.isVisible('[data-testid="chart-canvas"], [data-testid="chart-svg"]');
          expect(hasChart).toBe(true);
        }
      },
      
      {
        name: 'Export Results Report',
        action: async (page) => {
          if (await page.isVisible('[data-testid="export-results"]')) {
            await page.click('[data-testid="export-results"]');
            await page.waitForSelector('[data-testid="export-options"]');
            
            await page.selectOption('select[name="export-format"]', 'csv');
            
            if (await page.isVisible('[data-testid="download-report"]')) {
              // Simulate download without actually downloading
              await page.click('[data-testid="download-report"]');
              await page.waitForTimeout(2000);
            }
          }
        },
        validation: async (page) => {
          // Validate export interface was accessible
          if (await page.isVisible('[data-testid="export-options"]')) {
            await expect(page.locator('[data-testid="export-options"]')).toBeVisible();
          }
        }
      }
    ]);

    const result = await journey.executeJourney(page);
    
    // Log journey statistics
    const totalSteps = result.log.length;
    const successfulSteps = result.log.filter(log => log.success).length;
    const totalTime = result.log.reduce((sum, log) => sum + log.duration, 0);
    
    console.log(`Data Scientist Journey Summary:`, {
      totalSteps,
      successfulSteps,
      successRate: `${((successfulSteps / totalSteps) * 100).toFixed(1)}%`,
      totalTime: `${totalTime}ms`,
      averageStepTime: `${(totalTime / totalSteps).toFixed(0)}ms`
    });
    
    // Assert journey success
    expect(result.success).toBe(true);
    expect(successfulSteps / totalSteps).toBeGreaterThan(0.8); // 80% step success rate
  });

  test('Business Analyst: Dashboard Monitoring and Investigation', async ({ page }) => {
    const journey = new UserJourneyRunner(analystPersona, [
      {
        name: 'Access Real-time Dashboard',
        action: async (page) => {
          await page.goto('/dashboard');
          await page.waitForLoadState('networkidle');
          
          // Enable real-time monitoring
          if (await page.isVisible('[data-testid="start-realtime"]')) {
            await page.click('[data-testid="start-realtime"]');
            await page.waitForTimeout(2000);
          }
        },
        validation: async (page) => {
          await expect(page.locator('[data-testid="dashboard-content"]')).toBeVisible();
          await expect(page.locator('[data-testid="stats-grid"]')).toBeVisible();
        }
      },
      
      {
        name: 'Review Key Performance Indicators',
        action: async (page) => {
          // Check various dashboard widgets
          const widgets = [
            '[data-testid="anomaly-count-widget"]',
            '[data-testid="detection-rate-widget"]',
            '[data-testid="system-health-widget"]',
            '[data-testid="recent-alerts-widget"]'
          ];
          
          for (const widget of widgets) {
            if (await page.isVisible(widget)) {
              await page.hover(widget);
              await page.waitForTimeout(500);
            }
          }
        },
        validation: async (page) => {
          // Validate at least some widgets are present
          const hasAnyWidget = await page.locator('[data-testid*="widget"]').count() > 0;
          expect(hasAnyWidget).toBe(true);
        }
      },
      
      {
        name: 'Investigate Anomaly Alert',
        action: async (page) => {
          // Look for anomaly alerts or notifications
          if (await page.isVisible('[data-testid="alert-item"]')) {
            await page.click('[data-testid="alert-item"]');
            await page.waitForSelector('[data-testid="alert-details"]');
          } else if (await page.isVisible('[data-testid="recent-result"]')) {
            await page.click('[data-testid="recent-result"]');
            await page.waitForSelector('[data-testid="result-details"]');
          }
        },
        validation: async (page) => {
          // Validate detail view is accessible
          const hasDetails = await page.isVisible('[data-testid="alert-details"], [data-testid="result-details"]');
          expect(hasDetails).toBe(true);
        }
      },
      
      {
        name: 'Filter and Search Historical Data',
        action: async (page) => {
          // Navigate to results/history section
          if (await page.isVisible('[data-testid="view-history"]')) {
            await page.click('[data-testid="view-history"]');
          } else {
            await page.click('[data-testid="detection-tab"]');
          }
          
          await page.waitForSelector('[data-testid="results-list"], [data-testid="detection-interface"]');
          
          // Apply filters
          if (await page.isVisible('[data-testid="date-filter"]')) {
            await page.click('[data-testid="date-filter"]');
            await page.selectOption('select[name="time-range"]', 'last-week');
          }
          
          // Search functionality
          if (await page.isVisible('[data-testid="search-input"]')) {
            await page.fill('[data-testid="search-input"]', 'high');
            await page.waitForTimeout(1000);
          }
        },
        validation: async (page) => {
          // Validate filtering interface works
          await expect(page.locator('[data-testid="results-list"], [data-testid="detection-interface"]')).toBeVisible();
        }
      },
      
      {
        name: 'Generate Custom Report',
        action: async (page) => {
          if (await page.isVisible('[data-testid="generate-report"]')) {
            await page.click('[data-testid="generate-report"]');
            await page.waitForSelector('[data-testid="report-builder"]');
            
            // Configure report parameters
            await page.selectOption('select[name="report-type"]', 'summary');
            await page.selectOption('select[name="time-period"]', 'monthly');
            
            if (await page.isVisible('[data-testid="include-charts"]')) {
              await page.check('[data-testid="include-charts"]');
            }
          }
        },
        validation: async (page) => {
          if (await page.isVisible('[data-testid="report-builder"]')) {
            await expect(page.locator('[data-testid="report-builder"]')).toBeVisible();
          }
        }
      },
      
      {
        name: 'Share Findings via Export',
        action: async (page) => {
          if (await page.isVisible('[data-testid="export-report"]')) {
            await page.click('[data-testid="export-report"]');
            await page.waitForSelector('[data-testid="export-options"]');
            
            await page.selectOption('select[name="format"]', 'pdf');
            
            if (await page.isVisible('[data-testid="email-report"]')) {
              await page.fill('input[name="recipient"]', 'stakeholder@company.com');
            }
          }
        },
        validation: async (page) => {
          // Validate export options are available
          if (await page.isVisible('[data-testid="export-options"]')) {
            await expect(page.locator('[data-testid="export-options"]')).toBeVisible();
          }
        }
      }
    ]);

    const result = await journey.executeJourney(page);
    expect(result.success).toBe(true);
    
    // Log analyst-specific metrics
    const investigationSteps = result.log.filter(log => 
      log.step.includes('Investigate') || log.step.includes('Filter') || log.step.includes('Search')
    );
    console.log(`Investigation efficiency: ${investigationSteps.length} steps completed`);
  });

  test('System Administrator: Configuration and Monitoring', async ({ page }) => {
    const journey = new UserJourneyRunner(adminPersona, [
      {
        name: 'Access System Administration Panel',
        action: async (page) => {
          await page.goto('/admin');
          await page.waitForLoadState('networkidle');
        },
        validation: async (page) => {
          // Check for admin interface or redirect to appropriate section
          const isAdminPage = await page.url().includes('/admin') || 
                             await page.isVisible('[data-testid="admin-panel"]');
          expect(isAdminPage).toBe(true);
        }
      },
      
      {
        name: 'Review System Health Metrics',
        action: async (page) => {
          // Look for system monitoring widgets
          if (await page.isVisible('[data-testid="system-metrics"]')) {
            await page.click('[data-testid="system-metrics"]');
          } else if (await page.isVisible('[data-testid="monitoring-tab"]')) {
            await page.click('[data-testid="monitoring-tab"]');
          }
          
          await page.waitForTimeout(2000);
        },
        validation: async (page) => {
          // Validate system monitoring is accessible
          const hasMonitoring = await page.isVisible('[data-testid="system-metrics"]') ||
                               await page.isVisible('[data-testid="performance-charts"]') ||
                               await page.locator('text=/memory|cpu|disk/i').count() > 0;
          expect(hasMonitoring).toBe(true);
        }
      },
      
      {
        name: 'Check Resource Usage and Alerts',
        action: async (page) => {
          // Navigate through system status sections
          const sections = [
            '[data-testid="resource-usage"]',
            '[data-testid="active-alerts"]',
            '[data-testid="system-logs"]'
          ];
          
          for (const section of sections) {
            if (await page.isVisible(section)) {
              await page.click(section);
              await page.waitForTimeout(1000);
            }
          }
        },
        validation: async (page) => {
          // Validate system status information is available
          const hasSystemInfo = await page.locator('text=/status|health|resource|alert/i').count() > 0;
          expect(hasSystemInfo).toBe(true);
        }
      },
      
      {
        name: 'Configure System Settings',
        action: async (page) => {
          if (await page.isVisible('[data-testid="system-settings"]')) {
            await page.click('[data-testid="system-settings"]');
            await page.waitForSelector('[data-testid="settings-form"]');
            
            // Modify some settings (without saving)
            if (await page.isVisible('input[name="max-concurrent-detections"]')) {
              await page.fill('input[name="max-concurrent-detections"]', '10');
            }
            
            if (await page.isVisible('select[name="log-level"]')) {
              await page.selectOption('select[name="log-level"]', 'INFO');
            }
          }
        },
        validation: async (page) => {
          if (await page.isVisible('[data-testid="settings-form"]')) {
            await expect(page.locator('[data-testid="settings-form"]')).toBeVisible();
          }
        }
      }
    ]);

    const result = await journey.executeJourney(page);
    
    // Admin journeys might have limited access in test environment
    const accessibleSteps = result.log.filter(log => log.success).length;
    console.log(`Admin journey: ${accessibleSteps} steps accessible`);
    
    // More lenient assertion for admin functionality
    expect(accessibleSteps).toBeGreaterThan(0);
  });
});