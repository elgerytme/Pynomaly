/**
 * Advanced Component Patterns for Pynomaly Design System
 * Demonstrates complex UI patterns and interactive components
 */

export default {
  title: 'Patterns/Advanced Components',
  parameters: {
    docs: {
      description: {
        component: 'Advanced component patterns and interactive examples for the Pynomaly design system.'
      }
    }
  }
};

// Advanced Dashboard Card Pattern
export const DashboardCards = () => {
  const container = document.createElement('div');
  container.className = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 p-6';
  container.innerHTML = `
    <!-- Metric Card -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div class="flex items-center justify-between">
        <div>
          <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Total Anomalies</p>
          <p class="text-3xl font-bold text-gray-900 dark:text-white">1,247</p>
          <p class="text-sm text-green-600 dark:text-green-400 mt-1">
            <span class="inline-flex items-center">
              ↗ +12.5% from last week
            </span>
          </p>
        </div>
        <div class="p-3 bg-blue-100 dark:bg-blue-900 rounded-full">
          <svg class="w-6 h-6 text-blue-600 dark:text-blue-400" fill="currentColor" viewBox="0 0 20 20">
            <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
        </div>
      </div>
      <div class="mt-4">
        <div class="flex items-center text-sm text-gray-500 dark:text-gray-400">
          <span class="mr-2">Confidence:</span>
          <div class="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div class="bg-blue-600 h-2 rounded-full" style="width: 87%"></div>
          </div>
          <span class="ml-2">87%</span>
        </div>
      </div>
    </div>

    <!-- Status Card -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white">System Status</h3>
        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
          Operational
        </span>
      </div>
      <div class="space-y-3">
        <div class="flex items-center justify-between">
          <span class="text-sm text-gray-600 dark:text-gray-400">API Response</span>
          <div class="flex items-center">
            <div class="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
            <span class="text-sm font-medium text-gray-900 dark:text-white">98ms</span>
          </div>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-sm text-gray-600 dark:text-gray-400">Data Processing</span>
          <div class="flex items-center">
            <div class="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
            <span class="text-sm font-medium text-gray-900 dark:text-white">Normal</span>
          </div>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-sm text-gray-600 dark:text-gray-400">ML Models</span>
          <div class="flex items-center">
            <div class="w-2 h-2 bg-yellow-400 rounded-full mr-2"></div>
            <span class="text-sm font-medium text-gray-900 dark:text-white">Training</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Activity Feed Card -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Recent Activity</h3>
      <div class="space-y-4">
        <div class="flex items-start space-x-3">
          <div class="flex-shrink-0">
            <div class="w-8 h-8 bg-red-100 dark:bg-red-900 rounded-full flex items-center justify-center">
              <svg class="w-4 h-4 text-red-600 dark:text-red-400" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
              </svg>
            </div>
          </div>
          <div class="flex-1 min-w-0">
            <p class="text-sm font-medium text-gray-900 dark:text-white">High anomaly detected</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">CPU usage spike in server-01</p>
            <p class="text-xs text-gray-400 dark:text-gray-500 mt-1">2 minutes ago</p>
          </div>
        </div>
        <div class="flex items-start space-x-3">
          <div class="flex-shrink-0">
            <div class="w-8 h-8 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
              <svg class="w-4 h-4 text-blue-600 dark:text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
              </svg>
            </div>
          </div>
          <div class="flex-1 min-w-0">
            <p class="text-sm font-medium text-gray-900 dark:text-white">Model training completed</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">IsolationForest accuracy: 94.2%</p>
            <p class="text-xs text-gray-400 dark:text-gray-500 mt-1">15 minutes ago</p>
          </div>
        </div>
      </div>
    </div>
  `;
  return container;
};

// Interactive Data Table Pattern
export const DataTable = () => {
  const container = document.createElement('div');
  container.className = 'bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700';
  container.innerHTML = `
    <!-- Table Header -->
    <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white">Anomaly Detection Results</h3>
        <div class="flex items-center space-x-3">
          <div class="relative">
            <input type="text" placeholder="Search anomalies..."
                   class="w-64 pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent">
            <svg class="absolute left-3 top-2.5 w-4 h-4 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clip-rule="evenodd"/>
            </svg>
          </div>
          <button class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-blue-600 hover:bg-blue-700 focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
            <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd"/>
            </svg>
            Export
          </button>
        </div>
      </div>
    </div>

    <!-- Table Content -->
    <div class="overflow-x-auto">
      <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
        <thead class="bg-gray-50 dark:bg-gray-900">
          <tr>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              <input type="checkbox" class="rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500">
            </th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:text-gray-900 dark:hover:text-gray-200">
              Timestamp ↓
            </th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Feature
            </th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Value
            </th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Anomaly Score
            </th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Status
            </th>
            <th class="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Actions
            </th>
          </tr>
        </thead>
        <tbody class="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
          <tr class="hover:bg-gray-50 dark:hover:bg-gray-700">
            <td class="px-6 py-4 whitespace-nowrap">
              <input type="checkbox" class="rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500">
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
              2025-06-26 14:32:15
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
              CPU Usage
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
              87.4%
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
              <div class="flex items-center">
                <div class="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                  <div class="bg-red-600 h-2 rounded-full" style="width: 92%"></div>
                </div>
                <span class="text-sm font-medium text-gray-900 dark:text-white">0.92</span>
              </div>
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
              <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
                High Anomaly
              </span>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
              <div class="flex items-center justify-end space-x-2">
                <button class="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-200">
                  View
                </button>
                <button class="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200">
                  Export
                </button>
              </div>
            </td>
          </tr>
          <tr class="hover:bg-gray-50 dark:hover:bg-gray-700">
            <td class="px-6 py-4 whitespace-nowrap">
              <input type="checkbox" class="rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500">
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
              2025-06-26 14:31:45
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
              Memory Usage
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
              45.2%
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
              <div class="flex items-center">
                <div class="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                  <div class="bg-green-600 h-2 rounded-full" style="width: 15%"></div>
                </div>
                <span class="text-sm font-medium text-gray-900 dark:text-white">0.15</span>
              </div>
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
              <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                Normal
              </span>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
              <div class="flex items-center justify-end space-x-2">
                <button class="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-200">
                  View
                </button>
                <button class="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200">
                  Export
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Table Footer -->
    <div class="px-6 py-3 bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
      <div class="flex items-center justify-between">
        <div class="text-sm text-gray-700 dark:text-gray-300">
          Showing <span class="font-medium">1</span> to <span class="font-medium">10</span> of <span class="font-medium">247</span> results
        </div>
        <nav class="flex items-center space-x-1">
          <button class="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-l-lg hover:bg-gray-50 hover:text-gray-700 dark:bg-gray-800 dark:border-gray-600 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white">
            Previous
          </button>
          <button class="px-3 py-2 text-sm font-medium text-blue-600 bg-blue-50 border border-blue-300 hover:bg-blue-100 dark:bg-blue-900 dark:border-blue-600 dark:text-blue-400">
            1
          </button>
          <button class="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 hover:bg-gray-50 hover:text-gray-700 dark:bg-gray-800 dark:border-gray-600 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white">
            2
          </button>
          <button class="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 hover:bg-gray-50 hover:text-gray-700 dark:bg-gray-800 dark:border-gray-600 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white">
            3
          </button>
          <button class="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-r-lg hover:bg-gray-50 hover:text-gray-700 dark:bg-gray-800 dark:border-gray-600 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white">
            Next
          </button>
        </nav>
      </div>
    </div>
  `;
  return container;
};

// Advanced Form Pattern
export const AdvancedForm = () => {
  const container = document.createElement('div');
  container.className = 'max-w-4xl mx-auto p-6';
  container.innerHTML = `
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
      <!-- Form Header -->
      <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <h2 class="text-xl font-semibold text-gray-900 dark:text-white">Configure Anomaly Detection</h2>
        <p class="mt-1 text-sm text-gray-600 dark:text-gray-400">Set up your anomaly detection parameters and upload your dataset.</p>
      </div>

      <!-- Form Progress -->
      <div class="px-6 py-4 bg-gray-50 dark:bg-gray-900">
        <div class="flex items-center">
          <div class="flex items-center text-blue-600 dark:text-blue-400">
            <div class="flex items-center justify-center w-8 h-8 bg-blue-600 text-white rounded-full text-sm font-medium">
              1
            </div>
            <span class="ml-2 text-sm font-medium">Dataset</span>
          </div>
          <div class="flex-1 h-1 mx-4 bg-blue-600 rounded"></div>
          <div class="flex items-center text-blue-600 dark:text-blue-400">
            <div class="flex items-center justify-center w-8 h-8 bg-blue-600 text-white rounded-full text-sm font-medium">
              2
            </div>
            <span class="ml-2 text-sm font-medium">Algorithm</span>
          </div>
          <div class="flex-1 h-1 mx-4 bg-gray-300 dark:bg-gray-600 rounded"></div>
          <div class="flex items-center text-gray-500 dark:text-gray-400">
            <div class="flex items-center justify-center w-8 h-8 bg-gray-300 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-full text-sm font-medium">
              3
            </div>
            <span class="ml-2 text-sm font-medium">Review</span>
          </div>
        </div>
      </div>

      <!-- Form Content -->
      <div class="px-6 py-6">
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <!-- Left Column -->
          <div class="space-y-6">
            <!-- File Upload -->
            <div>
              <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Dataset Upload
              </label>
              <div class="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-6 text-center hover:border-gray-400 dark:hover:border-gray-500 transition-colors cursor-pointer">
                <svg class="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                  <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <p class="mt-2 text-sm text-gray-600 dark:text-gray-400">
                  <span class="font-medium text-blue-600 dark:text-blue-400 hover:text-blue-500">Click to upload</span>
                  or drag and drop
                </p>
                <p class="text-xs text-gray-500 dark:text-gray-400">CSV, JSON, or Parquet files up to 100MB</p>
              </div>
            </div>

            <!-- Algorithm Selection -->
            <div>
              <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Detection Algorithm
              </label>
              <select class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                <option>Isolation Forest</option>
                <option>Local Outlier Factor</option>
                <option>One-Class SVM</option>
                <option>Autoencoder</option>
                <option>LSTM Autoencoder</option>
              </select>
            </div>

            <!-- Parameters -->
            <div class="grid grid-cols-2 gap-4">
              <div>
                <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Contamination Rate
                </label>
                <div class="relative">
                  <input type="range" min="0.01" max="0.5" step="0.01" value="0.1"
                         class="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer">
                  <span class="absolute -top-8 left-1/2 transform -translate-x-1/2 text-sm text-gray-600 dark:text-gray-400">10%</span>
                </div>
              </div>
              <div>
                <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Random State
                </label>
                <input type="number" value="42"
                       class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent">
              </div>
            </div>
          </div>

          <!-- Right Column -->
          <div class="space-y-6">
            <!-- Preview -->
            <div>
              <h3 class="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Data Preview</h3>
              <div class="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 h-48 flex items-center justify-center border border-gray-200 dark:border-gray-700">
                <div class="text-center">
                  <svg class="mx-auto h-8 w-8 text-gray-400 mb-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clip-rule="evenodd"/>
                  </svg>
                  <p class="text-sm text-gray-500 dark:text-gray-400">Upload a dataset to see preview</p>
                </div>
              </div>
            </div>

            <!-- Advanced Options -->
            <div>
              <h3 class="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Advanced Options</h3>
              <div class="space-y-3">
                <label class="flex items-center">
                  <input type="checkbox" class="rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500">
                  <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">Enable feature scaling</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" class="rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500">
                  <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">Generate explanations</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" checked class="rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500">
                  <span class="ml-2 text-sm text-gray-700 dark:text-gray-300">Save model for reuse</span>
                </label>
              </div>
            </div>

            <!-- Validation -->
            <div class="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
              <div class="flex items-start">
                <svg class="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 mr-3" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
                </svg>
                <div>
                  <h4 class="text-sm font-medium text-blue-800 dark:text-blue-300">Configuration Valid</h4>
                  <p class="text-sm text-blue-700 dark:text-blue-400 mt-1">
                    Your current settings are optimized for the selected algorithm.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Form Actions -->
      <div class="px-6 py-4 bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between">
        <button class="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
          Save as Draft
        </button>
        <div class="flex items-center space-x-3">
          <button class="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
            Previous
          </button>
          <button class="px-6 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
            Next Step
          </button>
        </div>
      </div>
    </div>
  `;
  return container;
};

// Modal Dialog Pattern
export const ModalDialog = () => {
  const container = document.createElement('div');
  container.className = 'relative';
  container.innerHTML = `
    <!-- Modal Overlay -->
    <div class="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity z-50">
      <div class="fixed inset-0 z-10 overflow-y-auto">
        <div class="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
          <!-- Modal Panel -->
          <div class="relative transform overflow-hidden rounded-lg bg-white dark:bg-gray-800 px-4 pb-4 pt-5 text-left shadow-xl transition-all sm:my-8 sm:w-full sm:max-w-lg sm:p-6">
            <div class="absolute right-0 top-0 hidden pr-4 pt-4 sm:block">
              <button type="button" class="rounded-md bg-white dark:bg-gray-800 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
                <span class="sr-only">Close</span>
                <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div class="sm:flex sm:items-start">
              <div class="mx-auto flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-red-100 dark:bg-red-900 sm:mx-0 sm:h-10 sm:w-10">
                <svg class="h-6 w-6 text-red-600 dark:text-red-400" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                </svg>
              </div>

              <div class="mt-3 text-center sm:ml-4 sm:mt-0 sm:text-left">
                <h3 class="text-base font-semibold leading-6 text-gray-900 dark:text-white">
                  Delete Anomaly Detection Model
                </h3>
                <div class="mt-2">
                  <p class="text-sm text-gray-500 dark:text-gray-400">
                    Are you sure you want to delete this model? This action cannot be undone and all associated data will be permanently removed.
                  </p>
                </div>

                <div class="mt-4 bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
                  <div class="text-sm">
                    <div class="font-medium text-gray-900 dark:text-white">Model Details:</div>
                    <div class="mt-1 text-gray-600 dark:text-gray-400">
                      <div>Name: IsolationForest_v2.1</div>
                      <div>Created: June 25, 2025</div>
                      <div>Accuracy: 94.2%</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div class="mt-5 sm:mt-4 sm:flex sm:flex-row-reverse">
              <button type="button" class="inline-flex w-full justify-center rounded-md bg-red-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-red-500 sm:ml-3 sm:w-auto">
                Delete Model
              </button>
              <button type="button" class="mt-3 inline-flex w-full justify-center rounded-md bg-white dark:bg-gray-800 px-3 py-2 text-sm font-semibold text-gray-900 dark:text-white shadow-sm ring-1 ring-inset ring-gray-300 dark:ring-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 sm:mt-0 sm:w-auto">
                Cancel
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
  return container;
};

// Toast Notification Pattern
export const ToastNotifications = () => {
  const container = document.createElement('div');
  container.className = 'fixed top-4 right-4 z-50 space-y-3';
  container.innerHTML = `
    <!-- Success Toast -->
    <div class="max-w-sm w-full bg-white dark:bg-gray-800 shadow-lg rounded-lg pointer-events-auto ring-1 ring-black ring-opacity-5 overflow-hidden">
      <div class="p-4">
        <div class="flex items-start">
          <div class="flex-shrink-0">
            <svg class="h-6 w-6 text-green-400" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div class="ml-3 w-0 flex-1 pt-0.5">
            <p class="text-sm font-medium text-gray-900 dark:text-white">Model training completed!</p>
            <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">Your IsolationForest model achieved 94.2% accuracy.</p>
          </div>
          <div class="ml-4 flex flex-shrink-0">
            <button class="inline-flex rounded-md bg-white dark:bg-gray-800 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
              <span class="sr-only">Close</span>
              <svg class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
      <div class="bg-green-500 h-1 w-full">
        <div class="bg-green-600 h-1 animate-pulse" style="width: 80%"></div>
      </div>
    </div>

    <!-- Error Toast -->
    <div class="max-w-sm w-full bg-white dark:bg-gray-800 shadow-lg rounded-lg pointer-events-auto ring-1 ring-black ring-opacity-5 overflow-hidden">
      <div class="p-4">
        <div class="flex items-start">
          <div class="flex-shrink-0">
            <svg class="h-6 w-6 text-red-400" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
            </svg>
          </div>
          <div class="ml-3 w-0 flex-1 pt-0.5">
            <p class="text-sm font-medium text-gray-900 dark:text-white">Upload failed</p>
            <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">The dataset format is not supported. Please use CSV or JSON.</p>
          </div>
          <div class="ml-4 flex flex-shrink-0">
            <button class="inline-flex rounded-md bg-white dark:bg-gray-800 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
              <span class="sr-only">Close</span>
              <svg class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
      <div class="bg-red-500 h-1 w-full">
        <div class="bg-red-600 h-1 animate-pulse" style="width: 60%"></div>
      </div>
    </div>

    <!-- Info Toast -->
    <div class="max-w-sm w-full bg-white dark:bg-gray-800 shadow-lg rounded-lg pointer-events-auto ring-1 ring-black ring-opacity-5 overflow-hidden">
      <div class="p-4">
        <div class="flex items-start">
          <div class="flex-shrink-0">
            <svg class="h-6 w-6 text-blue-400" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853L15 14.25M12 2.25c5.385 0 9.75 4.365 9.75 9.75s-4.365 9.75-9.75 9.75S2.25 17.635 2.25 12 6.615 2.25 12 2.25z" />
            </svg>
          </div>
          <div class="ml-3 w-0 flex-1 pt-0.5">
            <p class="text-sm font-medium text-gray-900 dark:text-white">System maintenance</p>
            <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">Scheduled maintenance will begin in 30 minutes.</p>
          </div>
          <div class="ml-4 flex flex-shrink-0">
            <button class="inline-flex rounded-md bg-white dark:bg-gray-800 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
              <span class="sr-only">Close</span>
              <svg class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
      <div class="bg-blue-500 h-1 w-full">
        <div class="bg-blue-600 h-1 animate-pulse" style="width: 40%"></div>
      </div>
    </div>
  `;
  return container;
};

// Loading States Pattern
export const LoadingStates = () => {
  const container = document.createElement('div');
  container.className = 'space-y-8 p-6';
  container.innerHTML = `
    <!-- Skeleton Loading -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Skeleton Loading</h3>
      <div class="animate-pulse">
        <div class="flex items-center space-x-4 mb-4">
          <div class="w-12 h-12 bg-gray-300 dark:bg-gray-600 rounded-full"></div>
          <div class="flex-1">
            <div class="h-4 bg-gray-300 dark:bg-gray-600 rounded w-3/4 mb-2"></div>
            <div class="h-3 bg-gray-300 dark:bg-gray-600 rounded w-1/2"></div>
          </div>
        </div>
        <div class="space-y-3">
          <div class="h-4 bg-gray-300 dark:bg-gray-600 rounded"></div>
          <div class="h-4 bg-gray-300 dark:bg-gray-600 rounded w-5/6"></div>
          <div class="h-4 bg-gray-300 dark:bg-gray-600 rounded w-4/6"></div>
        </div>
      </div>
    </div>

    <!-- Spinner Loading -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Spinner Loading</h3>
      <div class="flex items-center justify-center py-12">
        <div class="relative">
          <div class="w-12 h-12 border-4 border-gray-200 dark:border-gray-700 rounded-full animate-spin">
            <div class="absolute top-0 left-0 w-full h-full border-4 border-transparent border-t-blue-600 rounded-full animate-spin"></div>
          </div>
          <p class="mt-4 text-sm text-gray-600 dark:text-gray-400 text-center">Training model...</p>
        </div>
      </div>
    </div>

    <!-- Progress Bar Loading -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Progress Bar Loading</h3>
      <div class="space-y-4">
        <div>
          <div class="flex items-center justify-between mb-2">
            <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Processing dataset</span>
            <span class="text-sm text-gray-500 dark:text-gray-400">67%</span>
          </div>
          <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div class="bg-blue-600 h-2 rounded-full transition-all duration-300" style="width: 67%"></div>
          </div>
        </div>
        <div class="text-sm text-gray-600 dark:text-gray-400">
          Step 2 of 3: Feature extraction and normalization
        </div>
      </div>
    </div>

    <!-- Dots Loading -->
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Dots Loading</h3>
      <div class="flex items-center justify-center py-12">
        <div class="flex space-x-2">
          <div class="w-3 h-3 bg-blue-600 rounded-full animate-bounce"></div>
          <div class="w-3 h-3 bg-blue-600 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
          <div class="w-3 h-3 bg-blue-600 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
        </div>
      </div>
    </div>
  `;
  return container;
};

DashboardCards.parameters = {
  docs: {
    description: {
      story: 'Comprehensive dashboard card components with metrics, status indicators, and activity feeds.'
    }
  }
};

DataTable.parameters = {
  docs: {
    description: {
      story: 'Feature-rich data table with search, filtering, pagination, and bulk actions.'
    }
  }
};

AdvancedForm.parameters = {
  docs: {
    description: {
      story: 'Multi-step form with file upload, validation, and progress tracking.'
    }
  }
};

ModalDialog.parameters = {
  docs: {
    description: {
      story: 'Accessible modal dialog with confirmation pattern and proper focus management.'
    }
  }
};

ToastNotifications.parameters = {
  docs: {
    description: {
      story: 'Toast notification system with different states and auto-dismiss functionality.'
    }
  }
};

LoadingStates.parameters = {
  docs: {
    description: {
      story: 'Various loading state patterns including skeletons, spinners, and progress indicators.'
    }
  }
};
