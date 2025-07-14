export default {
  title: 'Components/Foundation',
  parameters: {
    docs: {
      description: {
        component: 'Foundation components provide the basic building blocks for the Pynomaly design system.'
      }
    }
  }
};

// Button Component Stories
export const Buttons = () => {
  const container = document.createElement('div');
  container.className = 'space-y-6';

  container.innerHTML = `
    <div>
      <h3 class="text-lg font-semibold mb-4">Button Variants</h3>
      <div class="flex flex-wrap gap-4">
        <button class="btn-primary">Primary Button</button>
        <button class="btn-secondary">Secondary Button</button>
        <button class="btn-success">Success Button</button>
        <button class="btn-warning">Warning Button</button>
        <button class="btn-danger">Danger Button</button>
        <button class="btn-ghost">Ghost Button</button>
      </div>
    </div>

    <div>
      <h3 class="text-lg font-semibold mb-4">Button Sizes</h3>
      <div class="flex flex-wrap items-center gap-4">
        <button class="btn-primary btn-xs">Extra Small</button>
        <button class="btn-primary btn-sm">Small</button>
        <button class="btn-primary">Default</button>
        <button class="btn-primary btn-lg">Large</button>
        <button class="btn-primary btn-xl">Extra Large</button>
      </div>
    </div>

    <div>
      <h3 class="text-lg font-semibold mb-4">Button States</h3>
      <div class="flex flex-wrap gap-4">
        <button class="btn-primary">Normal</button>
        <button class="btn-primary" disabled>Disabled</button>
        <button class="btn-primary loading">
          <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          Loading
        </button>
      </div>
    </div>
  `;

  return container;
};

Buttons.parameters = {
  docs: {
    storyDescription: 'Button components with various styles, sizes, and states for different use cases.'
  }
};

// Status Indicators
export const StatusIndicators = () => {
  const container = document.createElement('div');
  container.className = 'space-y-6';

  container.innerHTML = `
    <div>
      <h3 class="text-lg font-semibold mb-4">Anomaly Status Indicators</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div class="status-indicator status-normal">
          <span class="status-icon"></span>
          <span class="status-text">Normal</span>
          <span class="status-value">98.5%</span>
        </div>

        <div class="status-indicator status-anomaly">
          <span class="status-icon"> </span>
          <span class="status-text">Anomaly Detected</span>
          <span class="status-value">2.1%</span>
        </div>

        <div class="status-indicator status-warning">
          <span class="status-icon">=á</span>
          <span class="status-text">Warning</span>
          <span class="status-value">5.3%</span>
        </div>

        <div class="status-indicator status-unknown">
          <span class="status-icon">S</span>
          <span class="status-text">Unknown</span>
          <span class="status-value">--</span>
        </div>

        <div class="status-indicator status-processing">
          <span class="status-icon">
            <svg class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          </span>
          <span class="status-text">Processing</span>
          <span class="status-value">...</span>
        </div>

        <div class="status-indicator status-error">
          <span class="status-icon">L</span>
          <span class="status-text">Error</span>
          <span class="status-value">--</span>
        </div>
      </div>
    </div>

    <div>
      <h3 class="text-lg font-semibold mb-4">Confidence Levels</h3>
      <div class="space-y-3">
        <div class="confidence-indicator confidence-high">
          <div class="confidence-bar">
            <div class="confidence-fill" style="width: 95%"></div>
          </div>
          <span class="confidence-label">High Confidence (95%)</span>
        </div>

        <div class="confidence-indicator confidence-medium">
          <div class="confidence-bar">
            <div class="confidence-fill" style="width: 75%"></div>
          </div>
          <span class="confidence-label">Medium Confidence (75%)</span>
        </div>

        <div class="confidence-indicator confidence-low">
          <div class="confidence-bar">
            <div class="confidence-fill" style="width: 45%"></div>
          </div>
          <span class="confidence-label">Low Confidence (45%)</span>
        </div>
      </div>
    </div>
  `;

  return container;
};

// Form Components
export const FormElements = () => {
  const container = document.createElement('div');
  container.className = 'space-y-6';

  container.innerHTML = `
    <div>
      <h3 class="text-lg font-semibold mb-4">Input Fields</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="form-group">
          <label class="form-label">Dataset Name</label>
          <input type="text" class="form-input" placeholder="Enter dataset name">
          <span class="form-help">Choose a descriptive name for your dataset</span>
        </div>

        <div class="form-group">
          <label class="form-label">Algorithm Selection</label>
          <select class="form-select">
            <option>Isolation Forest</option>
            <option>One-Class SVM</option>
            <option>Local Outlier Factor</option>
            <option>DBSCAN</option>
          </select>
        </div>

        <div class="form-group">
          <label class="form-label">Contamination Rate</label>
          <input type="range" class="form-range" min="0" max="1" step="0.01" value="0.1">
          <div class="flex justify-between text-sm text-gray-500">
            <span>0%</span>
            <span>10%</span>
            <span>100%</span>
          </div>
        </div>

        <div class="form-group">
          <label class="form-label">Configuration</label>
          <textarea class="form-textarea" rows="4" placeholder="Enter JSON configuration"></textarea>
        </div>
      </div>
    </div>

    <div>
      <h3 class="text-lg font-semibold mb-4">Form States</h3>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="form-group">
          <label class="form-label">Valid Input</label>
          <input type="text" class="form-input form-input-valid" value="dataset_v1.csv">
          <span class="form-success">File format is valid</span>
        </div>

        <div class="form-group">
          <label class="form-label">Invalid Input</label>
          <input type="text" class="form-input form-input-invalid" value="invalid_file.txt">
          <span class="form-error">Unsupported file format</span>
        </div>

        <div class="form-group">
          <label class="form-label">Disabled Input</label>
          <input type="text" class="form-input" disabled value="Read-only value">
        </div>
      </div>
    </div>
  `;

  return container;
};

// Cards and Containers
export const CardsAndContainers = () => {
  const container = document.createElement('div');
  container.className = 'space-y-6';

  container.innerHTML = `
    <div>
      <h3 class="text-lg font-semibold mb-4">Card Variants</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div class="card">
          <div class="card-header">
            <h4 class="card-title">Basic Card</h4>
          </div>
          <div class="card-body">
            <p class="card-text">This is a basic card with header and body content.</p>
          </div>
        </div>

        <div class="card card-interactive">
          <div class="card-header">
            <h4 class="card-title">Interactive Card</h4>
            <span class="badge badge-success">Active</span>
          </div>
          <div class="card-body">
            <p class="card-text">This card has hover and focus states.</p>
            <button class="btn-primary btn-sm">Action</button>
          </div>
        </div>

        <div class="card card-highlighted">
          <div class="card-header">
            <h4 class="card-title">Highlighted Card</h4>
          </div>
          <div class="card-body">
            <p class="card-text">This card draws attention with highlighting.</p>
          </div>
        </div>
      </div>
    </div>

    <div>
      <h3 class="text-lg font-semibold mb-4">Dashboard Widgets</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class="widget widget-metric">
          <div class="widget-header">
            <h5 class="widget-title">Total Datasets</h5>
          </div>
          <div class="widget-content">
            <div class="metric-value">1,234</div>
            <div class="metric-change metric-increase">+12.5%</div>
          </div>
        </div>

        <div class="widget widget-metric">
          <div class="widget-header">
            <h5 class="widget-title">Anomalies Detected</h5>
          </div>
          <div class="widget-content">
            <div class="metric-value">56</div>
            <div class="metric-change metric-decrease">-8.3%</div>
          </div>
        </div>

        <div class="widget widget-metric">
          <div class="widget-header">
            <h5 class="widget-title">Detection Accuracy</h5>
          </div>
          <div class="widget-content">
            <div class="metric-value">94.7%</div>
            <div class="metric-change metric-stable">+0.2%</div>
          </div>
        </div>

        <div class="widget widget-metric">
          <div class="widget-header">
            <h5 class="widget-title">Processing Time</h5>
          </div>
          <div class="widget-content">
            <div class="metric-value">2.3s</div>
            <div class="metric-change metric-increase">+15%</div>
          </div>
        </div>
      </div>
    </div>
  `;

  return container;
};

// Navigation Components
export const Navigation = () => {
  const container = document.createElement('div');
  container.className = 'space-y-6';

  container.innerHTML = `
    <div>
      <h3 class="text-lg font-semibold mb-4">Breadcrumbs</h3>
      <nav class="breadcrumb">
        <a href="#" class="breadcrumb-item">Dashboard</a>
        <span class="breadcrumb-separator">/</span>
        <a href="#" class="breadcrumb-item">Datasets</a>
        <span class="breadcrumb-separator">/</span>
        <span class="breadcrumb-item breadcrumb-current">anomaly_data_v2.csv</span>
      </nav>
    </div>

    <div>
      <h3 class="text-lg font-semibold mb-4">Tabs</h3>
      <div class="tabs">
        <button class="tab tab-active" data-tab="overview">Overview</button>
        <button class="tab" data-tab="configuration">Configuration</button>
        <button class="tab" data-tab="results">Results</button>
        <button class="tab" data-tab="history">History</button>
      </div>

      <div class="tab-content">
        <div class="tab-pane tab-pane-active" id="overview">
          <p class="text-gray-600">Overview content goes here...</p>
        </div>
      </div>
    </div>

    <div>
      <h3 class="text-lg font-semibold mb-4">Pagination</h3>
      <nav class="pagination">
        <button class="pagination-btn pagination-btn-disabled">Previous</button>
        <button class="pagination-btn pagination-btn-active">1</button>
        <button class="pagination-btn">2</button>
        <button class="pagination-btn">3</button>
        <span class="pagination-ellipsis">...</span>
        <button class="pagination-btn">10</button>
        <button class="pagination-btn">Next</button>
      </nav>
    </div>
  `;

  return container;
};

// Feedback Components
export const FeedbackComponents = () => {
  const container = document.createElement('div');
  container.className = 'space-y-6';

  container.innerHTML = `
    <div>
      <h3 class="text-lg font-semibold mb-4">Alerts</h3>
      <div class="space-y-4">
        <div class="alert alert-success">
          <div class="alert-icon"></div>
          <div class="alert-content">
            <div class="alert-title">Success!</div>
            <div class="alert-message">Your anomaly detection model has been trained successfully.</div>
          </div>
          <button class="alert-close">×</button>
        </div>

        <div class="alert alert-warning">
          <div class="alert-icon"> </div>
          <div class="alert-content">
            <div class="alert-title">Warning</div>
            <div class="alert-message">The dataset contains missing values that may affect detection accuracy.</div>
          </div>
          <button class="alert-close">×</button>
        </div>

        <div class="alert alert-danger">
          <div class="alert-icon">L</div>
          <div class="alert-content">
            <div class="alert-title">Error</div>
            <div class="alert-message">Failed to load dataset. Please check the file format and try again.</div>
          </div>
          <button class="alert-close">×</button>
        </div>

        <div class="alert alert-info">
          <div class="alert-icon">9</div>
          <div class="alert-content">
            <div class="alert-title">Information</div>
            <div class="alert-message">New algorithm updates are available. <a href="#" class="alert-link">Learn more</a></div>
          </div>
        </div>
      </div>
    </div>

    <div>
      <h3 class="text-lg font-semibold mb-4">Badges</h3>
      <div class="flex flex-wrap gap-2">
        <span class="badge badge-primary">Primary</span>
        <span class="badge badge-secondary">Secondary</span>
        <span class="badge badge-success">Success</span>
        <span class="badge badge-warning">Warning</span>
        <span class="badge badge-danger">Danger</span>
        <span class="badge badge-info">Info</span>
        <span class="badge badge-outline">Outline</span>
      </div>
    </div>

    <div>
      <h3 class="text-lg font-semibold mb-4">Progress Indicators</h3>
      <div class="space-y-4">
        <div class="progress-bar">
          <div class="progress-fill" style="width: 75%"></div>
          <span class="progress-label">Training Progress: 75%</span>
        </div>

        <div class="progress-bar progress-bar-success">
          <div class="progress-fill" style="width: 100%"></div>
          <span class="progress-label">Validation Complete: 100%</span>
        </div>

        <div class="progress-bar progress-bar-warning">
          <div class="progress-fill" style="width: 45%"></div>
          <span class="progress-label">Data Quality: 45%</span>
        </div>
      </div>
    </div>
  `;

  return container;
};

// Data Visualization Components
export const DataVisualization = () => {
  const container = document.createElement('div');
  container.className = 'space-y-6';

  container.innerHTML = `
    <div>
      <h3 class="text-lg font-semibold mb-4">Chart Placeholders</h3>
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="chart-container">
          <div class="chart-header">
            <h4 class="chart-title">Anomaly Detection Timeline</h4>
            <div class="chart-actions">
              <button class="btn-ghost btn-sm">Export</button>
            </div>
          </div>
          <div class="chart-content">
            <div class="chart-placeholder">
              <div class="chart-icon">=Ê</div>
              <p>Time series chart showing anomaly detection over time</p>
            </div>
          </div>
        </div>

        <div class="chart-container">
          <div class="chart-header">
            <h4 class="chart-title">Algorithm Performance</h4>
          </div>
          <div class="chart-content">
            <div class="chart-placeholder">
              <div class="chart-icon">=È</div>
              <p>Bar chart comparing algorithm accuracy</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div>
      <h3 class="text-lg font-semibold mb-4">Data Tables</h3>
      <div class="table-container">
        <table class="data-table">
          <thead>
            <tr>
              <th class="sortable">Dataset Name</th>
              <th class="sortable">Records</th>
              <th class="sortable">Anomalies</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>customer_transactions.csv</td>
              <td>10,524</td>
              <td>127</td>
              <td><span class="badge badge-success">Complete</span></td>
              <td>
                <button class="btn-ghost btn-xs">View</button>
                <button class="btn-ghost btn-xs">Download</button>
              </td>
            </tr>
            <tr>
              <td>network_traffic.json</td>
              <td>5,892</td>
              <td>23</td>
              <td><span class="badge badge-warning">Processing</span></td>
              <td>
                <button class="btn-ghost btn-xs" disabled>View</button>
                <button class="btn-ghost btn-xs">Cancel</button>
              </td>
            </tr>
            <tr>
              <td>sensor_data.parquet</td>
              <td>25,183</td>
              <td>445</td>
              <td><span class="badge badge-danger">Error</span></td>
              <td>
                <button class="btn-ghost btn-xs">Retry</button>
                <button class="btn-ghost btn-xs">Delete</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  `;

  return container;
};

// Add interactivity to stories
const addInteractivity = () => {
  // Tab functionality
  document.addEventListener('click', (e) => {
    if (e.target.classList.contains('tab')) {
      const tabsContainer = e.target.closest('.tabs');
      const allTabs = tabsContainer.querySelectorAll('.tab');
      const allPanes = tabsContainer.parentElement.querySelectorAll('.tab-pane');

      allTabs.forEach(tab => tab.classList.remove('tab-active'));
      allPanes.forEach(pane => pane.classList.remove('tab-pane-active'));

      e.target.classList.add('tab-active');
    }

    // Alert close functionality
    if (e.target.classList.contains('alert-close')) {
      e.target.closest('.alert').style.display = 'none';
    }
  });
};

// Initialize interactivity when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', addInteractivity);
} else {
  addInteractivity();
}
