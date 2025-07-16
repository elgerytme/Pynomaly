/**
 * Export Functionality Component
 * Comprehensive export capabilities for results, reports, and data
 */

export class ExportFunctionality {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      enableMultipleFormats: true,
      enableCustomTemplates: true,
      enableScheduledExports: true,
      enableBatchExports: true,
      enableCompression: true,
      maxFileSize: 100 * 1024 * 1024, // 100MB
      defaultFormat: 'json',
      supportedFormats: ['json', 'csv', 'xlsx', 'pdf', 'xml', 'yaml', 'parquet'],
      ...options
    };

    this.exportHistory = [];
    this.exportTemplates = new Map();
    this.scheduledExports = [];
    this.currentData = null;
    this.exportQueue = [];
    this.isExporting = false;

    this.init();
  }

  init() {
    this.createInterface();
    this.setupEventHandlers();
    this.loadExportTemplates();
    this.loadExportHistory();
    this.initializeDefaultTemplates();
  }

  createInterface() {
    this.container.innerHTML = `
      <div class="export-functionality-container">
        <div class="export-header">
          <div class="export-title">
            <h3>Export Data & Reports</h3>
            <div class="export-stats">
              <span id="export-count">0</span> exports completed
            </div>
          </div>

          <div class="export-actions">
            <button id="quick-export" class="btn-primary">
              <i class="fas fa-download"></i> Quick Export
            </button>
            <button id="advanced-export" class="btn-secondary">
              <i class="fas fa-cog"></i> Advanced Export
            </button>
            <button id="schedule-export" class="btn-info">
              <i class="fas fa-clock"></i> Schedule Export
            </button>
          </div>
        </div>

        <div class="export-content">
          <div class="export-main">
            <div class="export-section">
              <h4>Export Format</h4>
              <div class="format-selector">
                <div class="format-grid">
                  <div class="format-option" data-format="json">
                    <div class="format-icon">📄</div>
                    <div class="format-name">JSON</div>
                    <div class="format-description">JavaScript Object Notation</div>
                  </div>
                  <div class="format-option" data-format="csv">
                    <div class="format-icon">📊</div>
                    <div class="format-name">CSV</div>
                    <div class="format-description">Comma-Separated Values</div>
                  </div>
                  <div class="format-option" data-format="xlsx">
                    <div class="format-icon">📈</div>
                    <div class="format-name">Excel</div>
                    <div class="format-description">Microsoft Excel Spreadsheet</div>
                  </div>
                  <div class="format-option" data-format="pdf">
                    <div class="format-icon">📋</div>
                    <div class="format-name">PDF</div>
                    <div class="format-description">Portable Document Format</div>
                  </div>
                  <div class="format-option" data-format="xml">
                    <div class="format-icon">🔖</div>
                    <div class="format-name">XML</div>
                    <div class="format-description">Extensible Markup Language</div>
                  </div>
                  <div class="format-option" data-format="yaml">
                    <div class="format-icon">⚙️</div>
                    <div class="format-name">YAML</div>
                    <div class="format-description">YAML Ain't Markup Language</div>
                  </div>
                </div>
              </div>
            </div>

            <div class="export-section">
              <h4>Export Content</h4>
              <div class="content-selector">
                <div class="content-categories">
                  <div class="category-group">
                    <h5>Data</h5>
                    <label class="checkbox-label">
                      <input type="checkbox" id="export-raw-data" checked>
                      Raw Detection Data
                    </label>
                    <label class="checkbox-label">
                      <input type="checkbox" id="export-filtered-data" checked>
                      Filtered Data
                    </label>
                    <label class="checkbox-label">
                      <input type="checkbox" id="export-anomalies-only">
                      Anomalies Only
                    </label>
                    <label class="checkbox-label">
                      <input type="checkbox" id="export-metadata">
                      Metadata
                    </label>
                  </div>

                  <div class="category-group">
                    <h5>Results</h5>
                    <label class="checkbox-label">
                      <input type="checkbox" id="export-detection-results" checked>
                      Detection Results
                    </label>
                    <label class="checkbox-label">
                      <input type="checkbox" id="export-algorithm-performance">
                      Algorithm Performance
                    </label>
                    <label class="checkbox-label">
                      <input type="checkbox" id="export-statistical-summary">
                      Statistical Summary
                    </label>
                    <label class="checkbox-label">
                      <input type="checkbox" id="export-confidence-scores">
                      Confidence Scores
                    </label>
                  </div>

                  <div class="category-group">
                    <h5>Reports</h5>
                    <label class="checkbox-label">
                      <input type="checkbox" id="export-executive-summary">
                      Executive Summary
                    </label>
                    <label class="checkbox-label">
                      <input type="checkbox" id="export-detailed-report">
                      Detailed Report
                    </label>
                    <label class="checkbox-label">
                      <input type="checkbox" id="export-visualizations">
                      Visualizations
                    </label>
                    <label class="checkbox-label">
                      <input type="checkbox" id="export-recommendations">
                      Recommendations
                    </label>
                  </div>
                </div>
              </div>
            </div>

            <div class="export-section">
              <h4>Export Options</h4>
              <div class="export-options">
                <div class="option-group">
                  <label for="export-filename">Filename:</label>
                  <input type="text" id="export-filename" value="anomaly-detection-export" class="form-control">
                </div>

                <div class="option-group">
                  <label for="date-range">Date Range:</label>
                  <div class="date-range-controls">
                    <input type="date" id="export-start-date" class="form-control">
                    <span class="date-separator">to</span>
                    <input type="date" id="export-end-date" class="form-control">
                  </div>
                </div>

                <div class="option-group">
                  <label>
                    <input type="checkbox" id="include-timestamp"> Include Timestamp in Filename
                  </label>
                </div>

                <div class="option-group">
                  <label>
                    <input type="checkbox" id="compress-export"> Compress Export (ZIP)
                  </label>
                </div>

                <div class="option-group">
                  <label>
                    <input type="checkbox" id="include-charts"> Include Charts & Visualizations
                  </label>
                </div>
              </div>
            </div>
          </div>

          <div class="export-sidebar">
            <div class="sidebar-panel">
              <div class="panel-header">
                <h4>Export Templates</h4>
                <button id="create-template" class="btn-sm btn-primary">
                  <i class="fas fa-plus"></i> Create
                </button>
              </div>
              <div class="panel-content">
                <div id="export-templates" class="templates-list"></div>
              </div>
            </div>

            <div class="sidebar-panel">
              <div class="panel-header">
                <h4>Export Queue</h4>
                <button id="clear-queue" class="btn-sm btn-secondary">
                  <i class="fas fa-trash"></i> Clear
                </button>
              </div>
              <div class="panel-content">
                <div id="export-queue" class="queue-list">
                  <div class="queue-empty">No exports in queue</div>
                </div>
              </div>
            </div>

            <div class="sidebar-panel">
              <div class="panel-header">
                <h4>Export History</h4>
                <button id="clear-history" class="btn-sm btn-secondary">
                  <i class="fas fa-history"></i> Clear
                </button>
              </div>
              <div class="panel-content">
                <div id="export-history" class="history-list"></div>
              </div>
            </div>

            <div class="sidebar-panel">
              <div class="panel-header">
                <h4>Scheduled Exports</h4>
                <button id="manage-scheduled" class="btn-sm btn-info">
                  <i class="fas fa-cog"></i> Manage
                </button>
              </div>
              <div class="panel-content">
                <div id="scheduled-exports" class="scheduled-list"></div>
              </div>
            </div>
          </div>
        </div>

        <div class="export-footer">
          <div class="export-progress" id="export-progress" style="display: none;">
            <div class="progress-bar">
              <div class="progress-fill" id="progress-fill"></div>
            </div>
            <div class="progress-text">
              <span id="progress-message">Preparing export...</span>
              <span id="progress-percentage">0%</span>
            </div>
          </div>

          <div class="export-actions">
            <button id="preview-export" class="btn-secondary">
              <i class="fas fa-eye"></i> Preview
            </button>
            <button id="start-export" class="btn-primary">
              <i class="fas fa-download"></i> Start Export
            </button>
            <button id="cancel-export" class="btn-danger" style="display: none;">
              <i class="fas fa-times"></i> Cancel
            </button>
          </div>
        </div>
      </div>

      <!-- Export Preview Modal -->
      <div id="export-preview-modal" class="modal" style="display: none;">
        <div class="modal-content large">
          <div class="modal-header">
            <h3>Export Preview</h3>
            <button id="close-preview-modal" class="btn-close">&times;</button>
          </div>
          <div class="modal-body">
            <div class="preview-tabs">
              <button class="tab-button active" data-tab="data">Data</button>
              <button class="tab-button" data-tab="structure">Structure</button>
              <button class="tab-button" data-tab="summary">Summary</button>
            </div>
            <div class="preview-content">
              <div id="data-preview" class="preview-tab active"></div>
              <div id="structure-preview" class="preview-tab"></div>
              <div id="summary-preview" class="preview-tab"></div>
            </div>
          </div>
          <div class="modal-footer">
            <button id="export-from-preview" class="btn-primary">Export Now</button>
            <button id="close-preview" class="btn-secondary">Close</button>
          </div>
        </div>
      </div>

      <!-- Create Template Modal -->
      <div id="create-template-modal" class="modal" style="display: none;">
        <div class="modal-content">
          <div class="modal-header">
            <h3>Create Export Template</h3>
            <button id="close-template-modal" class="btn-close">&times;</button>
          </div>
          <div class="modal-body">
            <div class="form-group">
              <label for="template-name">Template Name:</label>
              <input type="text" id="template-name" class="form-control" placeholder="Enter template name">
            </div>
            <div class="form-group">
              <label for="template-description">Description:</label>
              <textarea id="template-description" class="form-control" rows="3" placeholder="Optional description"></textarea>
            </div>
            <div class="form-group">
              <label>
                <input type="checkbox" id="template-default"> Set as default template
              </label>
            </div>
          </div>
          <div class="modal-footer">
            <button id="save-template" class="btn-primary">Save Template</button>
            <button id="cancel-template" class="btn-secondary">Cancel</button>
          </div>
        </div>
      </div>

      <!-- Schedule Export Modal -->
      <div id="schedule-export-modal" class="modal" style="display: none;">
        <div class="modal-content">
          <div class="modal-header">
            <h3>Schedule Export</h3>
            <button id="close-schedule-modal" class="btn-close">&times;</button>
          </div>
          <div class="modal-body">
            <div class="form-group">
              <label for="schedule-name">Schedule Name:</label>
              <input type="text" id="schedule-name" class="form-control" placeholder="Enter schedule name">
            </div>
            <div class="form-group">
              <label for="schedule-frequency">Frequency:</label>
              <select id="schedule-frequency" class="form-control">
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
                <option value="custom">Custom</option>
              </select>
            </div>
            <div class="form-group">
              <label for="schedule-time">Time:</label>
              <input type="time" id="schedule-time" class="form-control" value="09:00">
            </div>
            <div class="form-group">
              <label for="schedule-email">Email Results To:</label>
              <input type="email" id="schedule-email" class="form-control" placeholder="email@example.com">
            </div>
            <div class="form-group">
              <label>
                <input type="checkbox" id="schedule-active" checked> Active
              </label>
            </div>
          </div>
          <div class="modal-footer">
            <button id="save-schedule" class="btn-primary">Save Schedule</button>
            <button id="cancel-schedule" class="btn-secondary">Cancel</button>
          </div>
        </div>
      </div>
    `;
  }

  setupEventHandlers() {
    // Format selection
    document.querySelectorAll('.format-option').forEach(option => {
      option.addEventListener('click', (e) => {
        this.selectFormat(e.currentTarget.dataset.format);
      });
    });

    // Main action buttons
    document.getElementById('quick-export').addEventListener('click', () => {
      this.performQuickExport();
    });

    document.getElementById('advanced-export').addEventListener('click', () => {
      this.showAdvancedExportModal();
    });

    document.getElementById('schedule-export').addEventListener('click', () => {
      this.showScheduleExportModal();
    });

    // Export actions
    document.getElementById('start-export').addEventListener('click', () => {
      this.startExport();
    });

    document.getElementById('cancel-export').addEventListener('click', () => {
      this.cancelExport();
    });

    document.getElementById('preview-export').addEventListener('click', () => {
      this.showExportPreview();
    });

    // Template actions
    document.getElementById('create-template').addEventListener('click', () => {
      this.showCreateTemplateModal();
    });

    // Queue actions
    document.getElementById('clear-queue').addEventListener('click', () => {
      this.clearExportQueue();
    });

    // History actions
    document.getElementById('clear-history').addEventListener('click', () => {
      this.clearExportHistory();
    });

    // Modal handlers
    this.setupModalHandlers();

    // Filename generation
    document.getElementById('include-timestamp').addEventListener('change', () => {
      this.updateFilename();
    });

    // Date range validation
    document.getElementById('export-start-date').addEventListener('change', () => {
      this.validateDateRange();
    });

    document.getElementById('export-end-date').addEventListener('change', () => {
      this.validateDateRange();
    });
  }

  setupModalHandlers() {
    // Preview modal
    document.getElementById('close-preview-modal').addEventListener('click', () => {
      document.getElementById('export-preview-modal').style.display = 'none';
    });

    document.getElementById('export-from-preview').addEventListener('click', () => {
      document.getElementById('export-preview-modal').style.display = 'none';
      this.startExport();
    });

    document.getElementById('close-preview').addEventListener('click', () => {
      document.getElementById('export-preview-modal').style.display = 'none';
    });

    // Template modal
    document.getElementById('close-template-modal').addEventListener('click', () => {
      document.getElementById('create-template-modal').style.display = 'none';
    });

    document.getElementById('save-template').addEventListener('click', () => {
      this.saveExportTemplate();
    });

    document.getElementById('cancel-template').addEventListener('click', () => {
      document.getElementById('create-template-modal').style.display = 'none';
    });

    // Schedule modal
    document.getElementById('close-schedule-modal').addEventListener('click', () => {
      document.getElementById('schedule-export-modal').style.display = 'none';
    });

    document.getElementById('save-schedule').addEventListener('click', () => {
      this.saveExportSchedule();
    });

    document.getElementById('cancel-schedule').addEventListener('click', () => {
      document.getElementById('schedule-export-modal').style.display = 'none';
    });

    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
      button.addEventListener('click', (e) => {
        this.switchPreviewTab(e.target.dataset.tab);
      });
    });
  }

  setData(data) {
    this.currentData = data;
    this.updateExportOptions();
  }

  updateExportOptions() {
    if (!this.currentData) return;

    // Update filename with data info
    const count = this.currentData.length;
    const defaultName = `anomaly-detection-${count}-records`;
    document.getElementById('export-filename').value = defaultName;

    // Set default date range based on data
    if (this.currentData.length > 0) {
      const dates = this.currentData.map(item => new Date(item.timestamp)).sort();
      const startDate = dates[0];
      const endDate = dates[dates.length - 1];

      document.getElementById('export-start-date').value = startDate.toISOString().split('T')[0];
      document.getElementById('export-end-date').value = endDate.toISOString().split('T')[0];
    }
  }

  selectFormat(format) {
    // Update UI to show selected format
    document.querySelectorAll('.format-option').forEach(option => {
      option.classList.remove('selected');
    });

    document.querySelector(`[data-format="${format}"]`).classList.add('selected');
    this.selectedFormat = format;

    // Update available options based on format
    this.updateFormatSpecificOptions(format);
  }

  updateFormatSpecificOptions(format) {
    const visualizationsCheckbox = document.getElementById('include-charts');
    const compressionCheckbox = document.getElementById('compress-export');

    // Some formats don't support visualizations
    if (format === 'csv' || format === 'json' || format === 'xml' || format === 'yaml') {
      visualizationsCheckbox.disabled = true;
      visualizationsCheckbox.checked = false;
    } else {
      visualizationsCheckbox.disabled = false;
    }

    // Enable compression for all formats except PDF
    compressionCheckbox.disabled = format === 'pdf';
  }

  performQuickExport() {
    if (!this.currentData) {
      alert('No data available to export');
      return;
    }

    const exportConfig = {
      format: 'json',
      filename: 'quick-export',
      includeTimestamp: true,
      compress: false,
      content: {
        rawData: true,
        detectionResults: true,
        metadata: true
      }
    };

    this.executeExport(exportConfig);
  }

  startExport() {
    if (!this.currentData) {
      alert('No data available to export');
      return;
    }

    const exportConfig = this.buildExportConfig();
    this.executeExport(exportConfig);
  }

  buildExportConfig() {
    const format = this.selectedFormat || 'json';
    const filename = document.getElementById('export-filename').value || 'export';
    const includeTimestamp = document.getElementById('include-timestamp').checked;
    const compress = document.getElementById('compress-export').checked;
    const includeCharts = document.getElementById('include-charts').checked;

    const content = {
      rawData: document.getElementById('export-raw-data').checked,
      filteredData: document.getElementById('export-filtered-data').checked,
      anomaliesOnly: document.getElementById('export-anomalies-only').checked,
      metadata: document.getElementById('export-metadata').checked,
      detectionResults: document.getElementById('export-detection-results').checked,
      algorithmPerformance: document.getElementById('export-algorithm-performance').checked,
      statisticalSummary: document.getElementById('export-statistical-summary').checked,
      confidenceScores: document.getElementById('export-confidence-scores').checked,
      executiveSummary: document.getElementById('export-executive-summary').checked,
      detailedReport: document.getElementById('export-detailed-report').checked,
      visualizations: document.getElementById('export-visualizations').checked,
      recommendations: document.getElementById('export-recommendations').checked
    };

    const dateRange = {
      startDate: document.getElementById('export-start-date').value,
      endDate: document.getElementById('export-end-date').value
    };

    return {
      format,
      filename,
      includeTimestamp,
      compress,
      includeCharts,
      content,
      dateRange
    };
  }

  executeExport(config) {
    this.showProgress();
    this.isExporting = true;

    // Add to queue
    const exportJob = {
      id: Date.now(),
      config: config,
      status: 'processing',
      progress: 0,
      startTime: Date.now()
    };

    this.exportQueue.push(exportJob);
    this.updateExportQueue();

    // Process export
    this.processExport(exportJob);
  }

  async processExport(exportJob) {
    try {
      const { config } = exportJob;

      // Update progress
      this.updateProgress(10, 'Filtering data...');

      // Filter data based on config
      let exportData = this.filterDataForExport(config);

      this.updateProgress(30, 'Preparing export data...');

      // Build export structure
      const exportStructure = this.buildExportStructure(exportData, config);

      this.updateProgress(50, 'Generating export file...');

      // Generate file based on format
      const fileContent = await this.generateFileContent(exportStructure, config.format);

      this.updateProgress(70, 'Processing visualizations...');

      // Include charts if requested
      if (config.includeCharts && config.format !== 'csv') {
        await this.includeVisualizations(exportStructure, config);
      }

      this.updateProgress(90, 'Finalizing export...');

      // Create final filename
      const filename = this.generateFilename(config);

      // Download file
      this.downloadFile(fileContent, filename, config.format);

      this.updateProgress(100, 'Export completed!');

      // Update export job status
      exportJob.status = 'completed';
      exportJob.endTime = Date.now();
      exportJob.filename = filename;

      // Add to history
      this.addToExportHistory(exportJob);

      // Clean up
      setTimeout(() => {
        this.hideProgress();
        this.isExporting = false;
        this.updateExportQueue();
      }, 1000);

    } catch (error) {
      console.error('Export failed:', error);

      exportJob.status = 'failed';
      exportJob.error = error.message;

      this.updateProgress(0, 'Export failed');
      this.hideProgress();
      this.isExporting = false;

      alert('Export failed: ' + error.message);
    }
  }

  filterDataForExport(config) {
    let data = [...this.currentData];

    // Apply date range filter
    if (config.dateRange.startDate || config.dateRange.endDate) {
      data = data.filter(item => {
        const itemDate = new Date(item.timestamp);
        if (config.dateRange.startDate && itemDate < new Date(config.dateRange.startDate)) {
          return false;
        }
        if (config.dateRange.endDate && itemDate > new Date(config.dateRange.endDate)) {
          return false;
        }
        return true;
      });
    }

    // Apply anomalies only filter
    if (config.content.anomaliesOnly) {
      data = data.filter(item => item.is_anomaly);
    }

    return data;
  }

  buildExportStructure(data, config) {
    const structure = {};

    if (config.content.rawData) {
      structure.rawData = data;
    }

    if (config.content.filteredData) {
      structure.filteredData = data;
    }

    if (config.content.metadata) {
      structure.metadata = {
        exportDate: new Date().toISOString(),
        recordCount: data.length,
        anomalyCount: data.filter(item => item.is_anomaly).length,
        dateRange: config.dateRange,
        exportConfig: config
      };
    }

    if (config.content.detectionResults) {
      structure.detectionResults = this.aggregateDetectionResults(data);
    }

    if (config.content.algorithmPerformance) {
      structure.algorithmPerformance = this.calculateAlgorithmPerformance(data);
    }

    if (config.content.statisticalSummary) {
      structure.statisticalSummary = this.generateStatisticalSummary(data);
    }

    if (config.content.confidenceScores) {
      structure.confidenceScores = this.extractConfidenceScores(data);
    }

    if (config.content.executiveSummary) {
      structure.executiveSummary = this.generateExecutiveSummary(data);
    }

    if (config.content.detailedReport) {
      structure.detailedReport = this.generateDetailedReport(data);
    }

    if (config.content.recommendations) {
      structure.recommendations = this.generateRecommendations(data);
    }

    return structure;
  }

  async generateFileContent(structure, format) {
    switch (format) {
      case 'json':
        return JSON.stringify(structure, null, 2);

      case 'csv':
        return this.generateCSV(structure);

      case 'xlsx':
        return await this.generateExcel(structure);

      case 'pdf':
        return await this.generatePDF(structure);

      case 'xml':
        return this.generateXML(structure);

      case 'yaml':
        return this.generateYAML(structure);

      default:
        return JSON.stringify(structure, null, 2);
    }
  }

  generateCSV(structure) {
    // Generate CSV from the main data
    const data = structure.rawData || structure.filteredData || [];

    if (data.length === 0) {
      return '';
    }

    // Get headers
    const headers = Object.keys(data[0]);

    // Generate CSV content
    const csvContent = [
      headers.join(','),
      ...data.map(row =>
        headers.map(header => {
          const value = row[header];
          if (typeof value === 'object') {
            return JSON.stringify(value);
          }
          return `"${value}"`;
        }).join(',')
      )
    ].join('\n');

    return csvContent;
  }

  async generateExcel(structure) {
    // This would typically use a library like SheetJS
    // For now, return CSV format
    return this.generateCSV(structure);
  }

  async generatePDF(structure) {
    // This would typically use a library like jsPDF
    // For now, return a text representation
    return this.generateTextReport(structure);
  }

  generateXML(structure) {
    const xmlContent = this.objectToXML(structure, 'export');
    return `<?xml version="1.0" encoding="UTF-8"?>\n${xmlContent}`;
  }

  generateYAML(structure) {
    // Simple YAML generation
    return this.objectToYAML(structure);
  }

  generateTextReport(structure) {
    let report = 'Anomaly Detection Export Report\n';
    report += '='.repeat(50) + '\n\n';

    if (structure.metadata) {
      report += `Export Date: ${structure.metadata.exportDate}\n`;
      report += `Record Count: ${structure.metadata.recordCount}\n`;
      report += `Anomaly Count: ${structure.metadata.anomalyCount}\n\n`;
    }

    if (structure.executiveSummary) {
      report += 'Executive Summary:\n';
      report += structure.executiveSummary + '\n\n';
    }

    if (structure.statisticalSummary) {
      report += 'Statistical Summary:\n';
      report += JSON.stringify(structure.statisticalSummary, null, 2) + '\n\n';
    }

    if (structure.recommendations) {
      report += 'Recommendations:\n';
      structure.recommendations.forEach((rec, index) => {
        report += `${index + 1}. ${rec}\n`;
      });
    }

    return report;
  }

  objectToXML(obj, rootName = 'root') {
    let xml = `<${rootName}>`;

    for (const [key, value] of Object.entries(obj)) {
      if (Array.isArray(value)) {
        xml += `<${key}>`;
        value.forEach(item => {
          xml += this.objectToXML(item, 'item');
        });
        xml += `</${key}>`;
      } else if (typeof value === 'object' && value !== null) {
        xml += this.objectToXML(value, key);
      } else {
        xml += `<${key}>${value}</${key}>`;
      }
    }

    xml += `</${rootName}>`;
    return xml;
  }

  objectToYAML(obj, indent = 0) {
    const spaces = '  '.repeat(indent);
    let yaml = '';

    for (const [key, value] of Object.entries(obj)) {
      if (Array.isArray(value)) {
        yaml += `${spaces}${key}:\n`;
        value.forEach(item => {
          yaml += `${spaces}  - ${typeof item === 'object' ? '\n' + this.objectToYAML(item, indent + 2) : item}\n`;
        });
      } else if (typeof value === 'object' && value !== null) {
        yaml += `${spaces}${key}:\n${this.objectToYAML(value, indent + 1)}`;
      } else {
        yaml += `${spaces}${key}: ${value}\n`;
      }
    }

    return yaml;
  }

  generateFilename(config) {
    let filename = config.filename;

    if (config.includeTimestamp) {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      filename += `-${timestamp}`;
    }

    filename += `.${config.format}`;

    if (config.compress) {
      filename += '.zip';
    }

    return filename;
  }

  downloadFile(content, filename, format) {
    let blob;

    if (format === 'xlsx') {
      blob = new Blob([content], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
    } else if (format === 'pdf') {
      blob = new Blob([content], { type: 'application/pdf' });
    } else if (format === 'csv') {
      blob = new Blob([content], { type: 'text/csv' });
    } else {
      blob = new Blob([content], { type: 'application/json' });
    }

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // Helper methods for data processing
  aggregateDetectionResults(data) {
    const results = {};

    data.forEach(item => {
      const algorithm = item.algorithm || 'unknown';
      if (!results[algorithm]) {
        results[algorithm] = {
          totalDetections: 0,
          anomalies: 0,
          avgConfidence: 0,
          avgScore: 0
        };
      }

      results[algorithm].totalDetections++;
      if (item.is_anomaly) {
        results[algorithm].anomalies++;
      }
      results[algorithm].avgConfidence += item.confidence || 0;
      results[algorithm].avgScore += item.anomaly_score || 0;
    });

    // Calculate averages
    Object.keys(results).forEach(algorithm => {
      const result = results[algorithm];
      result.avgConfidence /= result.totalDetections;
      result.avgScore /= result.totalDetections;
      result.anomalyRate = result.anomalies / result.totalDetections;
    });

    return results;
  }

  calculateAlgorithmPerformance(data) {
    // Calculate performance metrics for each algorithm
    const performance = {};

    data.forEach(item => {
      const algorithm = item.algorithm || 'unknown';
      if (!performance[algorithm]) {
        performance[algorithm] = {
          name: algorithm,
          totalSamples: 0,
          truePositives: 0,
          falsePositives: 0,
          trueNegatives: 0,
          falseNegatives: 0,
          processingTimes: []
        };
      }

      performance[algorithm].totalSamples++;
      if (item.processing_time) {
        performance[algorithm].processingTimes.push(item.processing_time);
      }

      // These would be calculated based on ground truth if available
      // For now, using mock values
      if (item.is_anomaly) {
        performance[algorithm].truePositives++;
      } else {
        performance[algorithm].trueNegatives++;
      }
    });

    // Calculate derived metrics
    Object.keys(performance).forEach(algorithm => {
      const perf = performance[algorithm];
      const total = perf.truePositives + perf.falsePositives + perf.trueNegatives + perf.falseNegatives;

      if (total > 0) {
        perf.accuracy = (perf.truePositives + perf.trueNegatives) / total;
        perf.precision = perf.truePositives / (perf.truePositives + perf.falsePositives) || 0;
        perf.recall = perf.truePositives / (perf.truePositives + perf.falseNegatives) || 0;
        perf.f1Score = 2 * (perf.precision * perf.recall) / (perf.precision + perf.recall) || 0;
      }

      if (perf.processingTimes.length > 0) {
        perf.avgProcessingTime = perf.processingTimes.reduce((a, b) => a + b, 0) / perf.processingTimes.length;
      }
    });

    return performance;
  }

  generateStatisticalSummary(data) {
    const summary = {
      totalRecords: data.length,
      anomalies: data.filter(item => item.is_anomaly).length,
      normalRecords: data.filter(item => !item.is_anomaly).length,
      timeRange: {
        start: Math.min(...data.map(item => new Date(item.timestamp).getTime())),
        end: Math.max(...data.map(item => new Date(item.timestamp).getTime()))
      },
      algorithms: [...new Set(data.map(item => item.algorithm))],
      avgAnomalyScore: data.reduce((sum, item) => sum + (item.anomaly_score || 0), 0) / data.length,
      avgConfidence: data.reduce((sum, item) => sum + (item.confidence || 0), 0) / data.length
    };

    summary.anomalyRate = summary.anomalies / summary.totalRecords;

    return summary;
  }

  extractConfidenceScores(data) {
    return data.map(item => ({
      timestamp: item.timestamp,
      algorithm: item.algorithm,
      confidence: item.confidence || 0,
      anomaly_score: item.anomaly_score || 0,
      is_anomaly: item.is_anomaly
    }));
  }

  generateExecutiveSummary(data) {
    const summary = this.generateStatisticalSummary(data);

    return `
Executive Summary:

This report covers ${summary.totalRecords} data points analyzed using ${summary.algorithms.length} anomaly detection algorithms.

Key Findings:
- ${summary.anomalies} anomalies were detected (${(summary.anomalyRate * 100).toFixed(1)}% anomaly rate)
- Average anomaly score: ${summary.avgAnomalyScore.toFixed(3)}
- Average confidence: ${(summary.avgConfidence * 100).toFixed(1)}%

Time Period: ${new Date(summary.timeRange.start).toLocaleDateString()} to ${new Date(summary.timeRange.end).toLocaleDateString()}

Algorithms Used: ${summary.algorithms.join(', ')}

Recommendations:
- ${summary.anomalyRate > 0.1 ? 'High anomaly rate detected - investigate potential causes' : 'Anomaly rate within normal range'}
- ${summary.avgConfidence < 0.7 ? 'Low confidence scores - consider algorithm tuning' : 'Confidence scores acceptable'}
- Regular monitoring recommended for continuous improvement
    `.trim();
  }

  generateDetailedReport(data) {
    const summary = this.generateStatisticalSummary(data);
    const performance = this.calculateAlgorithmPerformance(data);

    return {
      summary: summary,
      performance: performance,
      timeSeriesAnalysis: this.generateTimeSeriesAnalysis(data),
      anomalyDetails: this.generateAnomalyDetails(data),
      algorithmComparison: this.generateAlgorithmComparison(data)
    };
  }

  generateRecommendations(data) {
    const recommendations = [];
    const summary = this.generateStatisticalSummary(data);

    if (summary.anomalyRate > 0.15) {
      recommendations.push('High anomaly rate detected - investigate data quality and potential system issues');
    }

    if (summary.avgConfidence < 0.6) {
      recommendations.push('Low confidence scores - consider algorithm parameter tuning or ensemble methods');
    }

    if (summary.algorithms.length === 1) {
      recommendations.push('Consider using multiple algorithms for better detection accuracy');
    }

    recommendations.push('Implement regular monitoring and alerting for continuous anomaly detection');
    recommendations.push('Review and update detection thresholds based on business requirements');

    return recommendations;
  }

  generateTimeSeriesAnalysis(data) {
    // Group data by time intervals
    const hourlyData = {};

    data.forEach(item => {
      const hour = new Date(item.timestamp).getHours();
      if (!hourlyData[hour]) {
        hourlyData[hour] = { total: 0, anomalies: 0 };
      }
      hourlyData[hour].total++;
      if (item.is_anomaly) {
        hourlyData[hour].anomalies++;
      }
    });

    return hourlyData;
  }

  generateAnomalyDetails(data) {
    return data.filter(item => item.is_anomaly).map(item => ({
      timestamp: item.timestamp,
      algorithm: item.algorithm,
      anomaly_score: item.anomaly_score,
      confidence: item.confidence,
      features: item.features || {}
    }));
  }

  generateAlgorithmComparison(data) {
    const comparison = {};

    data.forEach(item => {
      const algorithm = item.algorithm || 'unknown';
      if (!comparison[algorithm]) {
        comparison[algorithm] = {
          name: algorithm,
          detections: 0,
          avgScore: 0,
          avgConfidence: 0,
          scores: []
        };
      }

      comparison[algorithm].detections++;
      comparison[algorithm].avgScore += item.anomaly_score || 0;
      comparison[algorithm].avgConfidence += item.confidence || 0;
      comparison[algorithm].scores.push(item.anomaly_score || 0);
    });

    // Calculate final averages
    Object.keys(comparison).forEach(algorithm => {
      const comp = comparison[algorithm];
      comp.avgScore /= comp.detections;
      comp.avgConfidence /= comp.detections;
      comp.scoreVariance = this.calculateVariance(comp.scores);
    });

    return comparison;
  }

  calculateVariance(values) {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    return values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  }

  // UI Methods
  showProgress() {
    document.getElementById('export-progress').style.display = 'block';
    document.getElementById('start-export').style.display = 'none';
    document.getElementById('cancel-export').style.display = 'inline-block';
  }

  hideProgress() {
    document.getElementById('export-progress').style.display = 'none';
    document.getElementById('start-export').style.display = 'inline-block';
    document.getElementById('cancel-export').style.display = 'none';
  }

  updateProgress(percentage, message) {
    document.getElementById('progress-fill').style.width = percentage + '%';
    document.getElementById('progress-percentage').textContent = percentage + '%';
    document.getElementById('progress-message').textContent = message;
  }

  cancelExport() {
    this.isExporting = false;
    this.hideProgress();

    // Cancel current export job
    if (this.exportQueue.length > 0) {
      this.exportQueue[this.exportQueue.length - 1].status = 'cancelled';
      this.updateExportQueue();
    }
  }

  showExportPreview() {
    if (!this.currentData) {
      alert('No data available to preview');
      return;
    }

    const config = this.buildExportConfig();
    const filteredData = this.filterDataForExport(config);
    const structure = this.buildExportStructure(filteredData, config);

    this.displayPreview(structure, config);
    document.getElementById('export-preview-modal').style.display = 'block';
  }

  displayPreview(structure, config) {
    // Data preview
    const dataPreview = document.getElementById('data-preview');
    dataPreview.innerHTML = `
      <div class="preview-section">
        <h4>Data Sample (First 10 records)</h4>
        <pre>${JSON.stringify(structure.rawData?.slice(0, 10) || [], null, 2)}</pre>
      </div>
    `;

    // Structure preview
    const structurePreview = document.getElementById('structure-preview');
    structurePreview.innerHTML = `
      <div class="preview-section">
        <h4>Export Structure</h4>
        <pre>${JSON.stringify(Object.keys(structure), null, 2)}</pre>
      </div>
    `;

    // Summary preview
    const summaryPreview = document.getElementById('summary-preview');
    summaryPreview.innerHTML = `
      <div class="preview-section">
        <h4>Export Summary</h4>
        <div class="summary-stats">
          <div class="stat-item">
            <span class="stat-label">Format:</span>
            <span class="stat-value">${config.format.toUpperCase()}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Records:</span>
            <span class="stat-value">${structure.rawData?.length || 0}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Sections:</span>
            <span class="stat-value">${Object.keys(structure).length}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Estimated Size:</span>
            <span class="stat-value">${this.estimateFileSize(structure)} KB</span>
          </div>
        </div>
      </div>
    `;
  }

  estimateFileSize(structure) {
    const jsonString = JSON.stringify(structure);
    return Math.round(jsonString.length / 1024);
  }

  switchPreviewTab(tabName) {
    document.querySelectorAll('.preview-tab').forEach(tab => {
      tab.classList.remove('active');
    });

    document.querySelectorAll('.tab-button').forEach(button => {
      button.classList.remove('active');
    });

    document.getElementById(`${tabName}-preview`).classList.add('active');
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
  }

  showCreateTemplateModal() {
    document.getElementById('create-template-modal').style.display = 'block';
  }

  saveExportTemplate() {
    const name = document.getElementById('template-name').value;
    const description = document.getElementById('template-description').value;
    const isDefault = document.getElementById('template-default').checked;

    if (!name.trim()) {
      alert('Please enter a template name');
      return;
    }

    const template = {
      id: Date.now(),
      name: name,
      description: description,
      isDefault: isDefault,
      config: this.buildExportConfig(),
      timestamp: Date.now()
    };

    this.exportTemplates.set(template.id, template);
    this.saveExportTemplates();
    this.updateExportTemplatesDisplay();

    document.getElementById('create-template-modal').style.display = 'none';
    document.getElementById('template-name').value = '';
    document.getElementById('template-description').value = '';
    document.getElementById('template-default').checked = false;
  }

  showScheduleExportModal() {
    document.getElementById('schedule-export-modal').style.display = 'block';
  }

  saveExportSchedule() {
    const name = document.getElementById('schedule-name').value;
    const frequency = document.getElementById('schedule-frequency').value;
    const time = document.getElementById('schedule-time').value;
    const email = document.getElementById('schedule-email').value;
    const active = document.getElementById('schedule-active').checked;

    if (!name.trim()) {
      alert('Please enter a schedule name');
      return;
    }

    const schedule = {
      id: Date.now(),
      name: name,
      frequency: frequency,
      time: time,
      email: email,
      active: active,
      config: this.buildExportConfig(),
      timestamp: Date.now(),
      nextRun: this.calculateNextRun(frequency, time)
    };

    this.scheduledExports.push(schedule);
    this.saveScheduledExports();
    this.updateScheduledExportsDisplay();

    document.getElementById('schedule-export-modal').style.display = 'none';
    document.getElementById('schedule-name').value = '';
    document.getElementById('schedule-email').value = '';
  }

  calculateNextRun(frequency, time) {
    const now = new Date();
    const nextRun = new Date();

    const [hours, minutes] = time.split(':').map(Number);
    nextRun.setHours(hours, minutes, 0, 0);

    switch (frequency) {
      case 'daily':
        if (nextRun <= now) {
          nextRun.setDate(nextRun.getDate() + 1);
        }
        break;
      case 'weekly':
        nextRun.setDate(nextRun.getDate() + 7);
        break;
      case 'monthly':
        nextRun.setMonth(nextRun.getMonth() + 1);
        break;
    }

    return nextRun.getTime();
  }

  updateExportTemplatesDisplay() {
    const templatesList = document.getElementById('export-templates');
    templatesList.innerHTML = '';

    this.exportTemplates.forEach(template => {
      const templateItem = document.createElement('div');
      templateItem.className = 'template-item';
      templateItem.innerHTML = `
        <div class="template-header">
          <div class="template-name">${template.name}</div>
          <div class="template-actions">
            <button class="apply-template btn-sm btn-primary" data-template-id="${template.id}">
              Apply
            </button>
            <button class="delete-template btn-sm btn-danger" data-template-id="${template.id}">
              Delete
            </button>
          </div>
        </div>
        <div class="template-description">${template.description}</div>
        ${template.isDefault ? '<div class="template-default">Default</div>' : ''}
      `;

      templateItem.querySelector('.apply-template').addEventListener('click', () => {
        this.applyExportTemplate(template.id);
      });

      templateItem.querySelector('.delete-template').addEventListener('click', () => {
        this.deleteExportTemplate(template.id);
      });

      templatesList.appendChild(templateItem);
    });
  }

  applyExportTemplate(templateId) {
    const template = this.exportTemplates.get(templateId);
    if (!template) return;

    const config = template.config;

    // Apply template configuration to UI
    this.selectFormat(config.format);
    document.getElementById('export-filename').value = config.filename;
    document.getElementById('include-timestamp').checked = config.includeTimestamp;
    document.getElementById('compress-export').checked = config.compress;
    document.getElementById('include-charts').checked = config.includeCharts;

    // Apply content selections
    Object.keys(config.content).forEach(key => {
      const element = document.getElementById(`export-${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`);
      if (element) {
        element.checked = config.content[key];
      }
    });

    // Apply date range
    if (config.dateRange.startDate) {
      document.getElementById('export-start-date').value = config.dateRange.startDate;
    }
    if (config.dateRange.endDate) {
      document.getElementById('export-end-date').value = config.dateRange.endDate;
    }
  }

  deleteExportTemplate(templateId) {
    if (confirm('Are you sure you want to delete this template?')) {
      this.exportTemplates.delete(templateId);
      this.saveExportTemplates();
      this.updateExportTemplatesDisplay();
    }
  }

  updateExportQueue() {
    const queueList = document.getElementById('export-queue');
    queueList.innerHTML = '';

    if (this.exportQueue.length === 0) {
      queueList.innerHTML = '<div class="queue-empty">No exports in queue</div>';
      return;
    }

    this.exportQueue.forEach(job => {
      const queueItem = document.createElement('div');
      queueItem.className = `queue-item ${job.status}`;
      queueItem.innerHTML = `
        <div class="queue-header">
          <div class="queue-filename">${job.config.filename}</div>
          <div class="queue-status">${job.status}</div>
        </div>
        <div class="queue-details">
          <div class="queue-format">${job.config.format.toUpperCase()}</div>
          <div class="queue-time">${new Date(job.startTime).toLocaleTimeString()}</div>
        </div>
      `;
      queueList.appendChild(queueItem);
    });
  }

  clearExportQueue() {
    this.exportQueue = [];
    this.updateExportQueue();
  }

  addToExportHistory(exportJob) {
    this.exportHistory.unshift(exportJob);

    // Keep only last 50 exports
    if (this.exportHistory.length > 50) {
      this.exportHistory.pop();
    }

    this.saveExportHistory();
    this.updateExportHistoryDisplay();
    this.updateExportStats();
  }

  updateExportHistoryDisplay() {
    const historyList = document.getElementById('export-history');
    historyList.innerHTML = '';

    this.exportHistory.forEach(job => {
      const historyItem = document.createElement('div');
      historyItem.className = `history-item ${job.status}`;
      historyItem.innerHTML = `
        <div class="history-header">
          <div class="history-filename">${job.filename || job.config.filename}</div>
          <div class="history-status">${job.status}</div>
        </div>
        <div class="history-details">
          <div class="history-format">${job.config.format.toUpperCase()}</div>
          <div class="history-time">${new Date(job.startTime).toLocaleString()}</div>
        </div>
        ${job.status === 'completed' ? `
          <div class="history-actions">
            <button class="download-again btn-sm btn-secondary" data-job-id="${job.id}">
              Download Again
            </button>
          </div>
        ` : ''}
      `;

      const downloadBtn = historyItem.querySelector('.download-again');
      if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
          this.downloadAgain(job.id);
        });
      }

      historyList.appendChild(historyItem);
    });
  }

  downloadAgain(jobId) {
    const job = this.exportHistory.find(j => j.id === jobId);
    if (job) {
      this.executeExport(job.config);
    }
  }

  clearExportHistory() {
    if (confirm('Are you sure you want to clear export history?')) {
      this.exportHistory = [];
      this.updateExportHistoryDisplay();
      this.updateExportStats();
    }
  }

  updateScheduledExportsDisplay() {
    const scheduledList = document.getElementById('scheduled-exports');
    scheduledList.innerHTML = '';

    this.scheduledExports.forEach(schedule => {
      const scheduleItem = document.createElement('div');
      scheduleItem.className = `scheduled-item ${schedule.active ? 'active' : 'inactive'}`;
      scheduleItem.innerHTML = `
        <div class="scheduled-header">
          <div class="scheduled-name">${schedule.name}</div>
          <div class="scheduled-status">${schedule.active ? 'Active' : 'Inactive'}</div>
        </div>
        <div class="scheduled-details">
          <div class="scheduled-frequency">${schedule.frequency}</div>
          <div class="scheduled-time">${schedule.time}</div>
        </div>
        <div class="scheduled-next">
          Next run: ${new Date(schedule.nextRun).toLocaleString()}
        </div>
        <div class="scheduled-actions">
          <button class="toggle-schedule btn-sm btn-secondary" data-schedule-id="${schedule.id}">
            ${schedule.active ? 'Disable' : 'Enable'}
          </button>
          <button class="delete-schedule btn-sm btn-danger" data-schedule-id="${schedule.id}">
            Delete
          </button>
        </div>
      `;

      scheduleItem.querySelector('.toggle-schedule').addEventListener('click', () => {
        this.toggleSchedule(schedule.id);
      });

      scheduleItem.querySelector('.delete-schedule').addEventListener('click', () => {
        this.deleteSchedule(schedule.id);
      });

      scheduledList.appendChild(scheduleItem);
    });
  }

  toggleSchedule(scheduleId) {
    const schedule = this.scheduledExports.find(s => s.id === scheduleId);
    if (schedule) {
      schedule.active = !schedule.active;
      this.saveScheduledExports();
      this.updateScheduledExportsDisplay();
    }
  }

  deleteSchedule(scheduleId) {
    if (confirm('Are you sure you want to delete this scheduled export?')) {
      this.scheduledExports = this.scheduledExports.filter(s => s.id !== scheduleId);
      this.saveScheduledExports();
      this.updateScheduledExportsDisplay();
    }
  }

  updateExportStats() {
    const completedExports = this.exportHistory.filter(job => job.status === 'completed').length;
    document.getElementById('export-count').textContent = completedExports;
  }

  updateFilename() {
    const includeTimestamp = document.getElementById('include-timestamp').checked;
    const baseFilename = document.getElementById('export-filename').value.replace(/-\d{4}-\d{2}-\d{2}T.*/, '');

    if (includeTimestamp) {
      const timestamp = new Date().toISOString().split('T')[0];
      document.getElementById('export-filename').value = `${baseFilename}-${timestamp}`;
    } else {
      document.getElementById('export-filename').value = baseFilename;
    }
  }

  validateDateRange() {
    const startDate = document.getElementById('export-start-date').value;
    const endDate = document.getElementById('export-end-date').value;

    if (startDate && endDate && new Date(startDate) > new Date(endDate)) {
      alert('Start date must be before end date');
      document.getElementById('export-end-date').value = '';
    }
  }

  // Persistence methods
  loadExportTemplates() {
    const saved = localStorage.getItem('pynomaly_export_templates');
    if (saved) {
      const templates = JSON.parse(saved);
      this.exportTemplates = new Map(templates);
      this.updateExportTemplatesDisplay();
    }
  }

  saveExportTemplates() {
    localStorage.setItem('pynomaly_export_templates', JSON.stringify([...this.exportTemplates]));
  }

  loadExportHistory() {
    const saved = localStorage.getItem('pynomaly_export_history');
    if (saved) {
      this.exportHistory = JSON.parse(saved);
      this.updateExportHistoryDisplay();
      this.updateExportStats();
    }
  }

  saveExportHistory() {
    localStorage.setItem('pynomaly_export_history', JSON.stringify(this.exportHistory));
  }

  loadScheduledExports() {
    const saved = localStorage.getItem('pynomaly_scheduled_exports');
    if (saved) {
      this.scheduledExports = JSON.parse(saved);
      this.updateScheduledExportsDisplay();
    }
  }

  saveScheduledExports() {
    localStorage.setItem('pynomaly_scheduled_exports', JSON.stringify(this.scheduledExports));
  }

  initializeDefaultTemplates() {
    // Create some default templates if none exist
    if (this.exportTemplates.size === 0) {
      const defaultTemplates = [
        {
          id: 1,
          name: 'Basic Data Export',
          description: 'Export raw data and basic results',
          isDefault: true,
          config: {
            format: 'json',
            filename: 'basic-export',
            includeTimestamp: true,
            compress: false,
            includeCharts: false,
            content: {
              rawData: true,
              detectionResults: true,
              metadata: true
            }
          }
        },
        {
          id: 2,
          name: 'Complete Report',
          description: 'Full report with all data and visualizations',
          isDefault: false,
          config: {
            format: 'pdf',
            filename: 'complete-report',
            includeTimestamp: true,
            compress: true,
            includeCharts: true,
            content: {
              rawData: true,
              detectionResults: true,
              algorithmPerformance: true,
              statisticalSummary: true,
              executiveSummary: true,
              detailedReport: true,
              visualizations: true,
              recommendations: true
            }
          }
        }
      ];

      defaultTemplates.forEach(template => {
        this.exportTemplates.set(template.id, template);
      });

      this.saveExportTemplates();
      this.updateExportTemplatesDisplay();
    }
  }

  destroy() {
    // Clean up resources
    this.exportHistory = [];
    this.exportTemplates.clear();
    this.scheduledExports = [];
    this.exportQueue = [];
    this.currentData = null;
  }
}

// Export the class
window.ExportFunctionality = ExportFunctionality;
