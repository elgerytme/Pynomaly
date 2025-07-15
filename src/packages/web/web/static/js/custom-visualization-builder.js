/**
 * Custom Visualization Builder
 * Allows users to create custom charts with drag-and-drop interface
 */

class CustomVisualizationBuilder {
  constructor() {
    this.init();
    this.chartTypes = [
      { id: 'line', name: 'Line Chart', icon: '📈' },
      { id: 'bar', name: 'Bar Chart', icon: '📊' },
      { id: 'scatter', name: 'Scatter Plot', icon: '⚫' },
      { id: 'heatmap', name: 'Heatmap', icon: '🔥' },
      { id: 'histogram', name: 'Histogram', icon: '📊' },
      { id: 'box', name: 'Box Plot', icon: '📦' },
      { id: 'area', name: 'Area Chart', icon: '🏔️' },
      { id: 'pie', name: 'Pie Chart', icon: '🥧' }
    ];
    this.dataFields = [
      { id: 'timestamp', name: 'Timestamp', type: 'datetime' },
      { id: 'anomaly_score', name: 'Anomaly Score', type: 'numeric' },
      { id: 'detector_id', name: 'Detector ID', type: 'categorical' },
      { id: 'feature_values', name: 'Feature Values', type: 'numeric' },
      { id: 'sample_id', name: 'Sample ID', type: 'categorical' },
      { id: 'confidence', name: 'Confidence', type: 'numeric' },
      { id: 'severity', name: 'Severity', type: 'categorical' }
    ];
    this.customCharts = [];
  }

  init() {
    this.createBuilderInterface();
    this.setupEventListeners();
  }

  createBuilderInterface() {
    const builderContainer = document.createElement('div');
    builderContainer.id = 'custom-viz-builder';
    builderContainer.className = 'fixed inset-0 bg-gray-600 bg-opacity-50 hidden items-center justify-center z-50';

    builderContainer.innerHTML = `
      <div class="bg-white rounded-lg shadow-xl max-w-6xl w-full mx-4 max-h-screen overflow-y-auto">
        <div class="px-6 py-4 border-b border-gray-200">
          <div class="flex justify-between items-center">
            <h2 class="text-xl font-semibold text-gray-900">Custom Visualization Builder</h2>
            <button id="close-builder" class="text-gray-400 hover:text-gray-600">
              <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
              </svg>
            </button>
          </div>
        </div>

        <div class="flex">
          <!-- Left Panel: Chart Types & Settings -->
          <div class="w-1/3 border-r border-gray-200 p-6">
            <div class="mb-6">
              <h3 class="text-lg font-medium text-gray-900 mb-3">Chart Types</h3>
              <div class="grid grid-cols-2 gap-2" id="chart-types-grid">
                ${this.chartTypes.map(type => `
                  <button class="chart-type-btn p-3 border border-gray-200 rounded-lg hover:bg-blue-50 hover:border-blue-300 transition-colors" data-type="${type.id}">
                    <div class="text-2xl mb-1">${type.icon}</div>
                    <div class="text-sm font-medium">${type.name}</div>
                  </button>
                `).join('')}
              </div>
            </div>

            <div class="mb-6">
              <h3 class="text-lg font-medium text-gray-900 mb-3">Data Fields</h3>
              <div class="space-y-2" id="data-fields-list">
                ${this.dataFields.map(field => `
                  <div class="data-field p-2 border border-gray-200 rounded cursor-move hover:bg-gray-50"
                       draggable="true" data-field="${field.id}" data-type="${field.type}">
                    <div class="font-medium text-sm">${field.name}</div>
                    <div class="text-xs text-gray-500">${field.type}</div>
                  </div>
                `).join('')}
              </div>
            </div>

            <div class="mb-6">
              <h3 class="text-lg font-medium text-gray-900 mb-3">Chart Settings</h3>
              <div class="space-y-4" id="chart-settings">
                <div>
                  <label class="block text-sm font-medium text-gray-700 mb-1">Chart Title</label>
                  <input type="text" id="chart-title" class="w-full px-3 py-2 border border-gray-300 rounded-md" placeholder="Enter chart title">
                </div>
                <div>
                  <label class="block text-sm font-medium text-gray-700 mb-1">Width</label>
                  <input type="range" id="chart-width" min="300" max="800" value="600" class="w-full">
                  <span class="text-sm text-gray-500" id="width-value">600px</span>
                </div>
                <div>
                  <label class="block text-sm font-medium text-gray-700 mb-1">Height</label>
                  <input type="range" id="chart-height" min="200" max="600" value="400" class="w-full">
                  <span class="text-sm text-gray-500" id="height-value">400px</span>
                </div>
                <div>
                  <label class="block text-sm font-medium text-gray-700 mb-1">Color Scheme</label>
                  <select id="color-scheme" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                    <option value="default">Default</option>
                    <option value="blue">Blue</option>
                    <option value="green">Green</option>
                    <option value="red">Red</option>
                    <option value="rainbow">Rainbow</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          <!-- Middle Panel: Chart Configuration -->
          <div class="w-1/3 border-r border-gray-200 p-6">
            <div class="mb-6">
              <h3 class="text-lg font-medium text-gray-900 mb-3">Chart Configuration</h3>
              <div class="text-center text-gray-500 mb-4" id="selected-chart-type">
                Select a chart type to begin
              </div>

              <!-- Chart Mapping Areas -->
              <div class="space-y-4" id="chart-mapping">
                <div class="drop-zone border-2 border-dashed border-gray-300 rounded-lg p-4 min-h-16"
                     data-axis="x" id="x-axis-zone">
                  <div class="text-sm font-medium text-gray-700 mb-2">X-Axis</div>
                  <div class="text-xs text-gray-500">Drop a data field here</div>
                </div>

                <div class="drop-zone border-2 border-dashed border-gray-300 rounded-lg p-4 min-h-16"
                     data-axis="y" id="y-axis-zone">
                  <div class="text-sm font-medium text-gray-700 mb-2">Y-Axis</div>
                  <div class="text-xs text-gray-500">Drop a data field here</div>
                </div>

                <div class="drop-zone border-2 border-dashed border-gray-300 rounded-lg p-4 min-h-16"
                     data-axis="color" id="color-zone" style="display: none;">
                  <div class="text-sm font-medium text-gray-700 mb-2">Color By</div>
                  <div class="text-xs text-gray-500">Drop a categorical field here</div>
                </div>

                <div class="drop-zone border-2 border-dashed border-gray-300 rounded-lg p-4 min-h-16"
                     data-axis="size" id="size-zone" style="display: none;">
                  <div class="text-sm font-medium text-gray-700 mb-2">Size By</div>
                  <div class="text-xs text-gray-500">Drop a numeric field here</div>
                </div>
              </div>

              <div class="mt-6">
                <button id="generate-chart" class="w-full bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 disabled:bg-gray-300" disabled>
                  Generate Chart
                </button>
              </div>
            </div>
          </div>

          <!-- Right Panel: Preview -->
          <div class="w-1/3 p-6">
            <div class="mb-6">
              <h3 class="text-lg font-medium text-gray-900 mb-3">Preview</h3>
              <div id="chart-preview" class="border border-gray-200 rounded-lg p-4 min-h-64 bg-gray-50">
                <div class="text-center text-gray-500 mt-16">
                  Chart preview will appear here
                </div>
              </div>
            </div>

            <div class="space-y-3">
              <button id="save-custom-chart" class="w-full bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-600" disabled>
                Save Chart
              </button>
              <button id="export-custom-chart" class="w-full bg-yellow-500 text-white px-4 py-2 rounded-lg hover:bg-yellow-600" disabled>
                Export Chart
              </button>
              <button id="share-custom-chart" class="w-full bg-purple-500 text-white px-4 py-2 rounded-lg hover:bg-purple-600" disabled>
                Share Chart
              </button>
            </div>
          </div>
        </div>
      </div>
    `;

    document.body.appendChild(builderContainer);
  }

  setupEventListeners() {
    // Close builder
    document.getElementById('close-builder').addEventListener('click', () => {
      this.hideBuilder();
    });

    // Chart type selection
    document.querySelectorAll('.chart-type-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        this.selectChartType(e.target.closest('.chart-type-btn').dataset.type);
      });
    });

    // Drag and drop for data fields
    document.querySelectorAll('.data-field').forEach(field => {
      field.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('text/plain', JSON.stringify({
          field: e.target.dataset.field,
          type: e.target.dataset.type,
          name: e.target.querySelector('.font-medium').textContent
        }));
      });
    });

    // Drop zones
    document.querySelectorAll('.drop-zone').forEach(zone => {
      zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('border-blue-500', 'bg-blue-50');
      });

      zone.addEventListener('dragleave', (e) => {
        zone.classList.remove('border-blue-500', 'bg-blue-50');
      });

      zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('border-blue-500', 'bg-blue-50');

        const data = JSON.parse(e.dataTransfer.getData('text/plain'));
        this.assignFieldToAxis(zone.dataset.axis, data);
      });
    });

    // Settings inputs
    document.getElementById('chart-width').addEventListener('input', (e) => {
      document.getElementById('width-value').textContent = e.target.value + 'px';
      this.updatePreview();
    });

    document.getElementById('chart-height').addEventListener('input', (e) => {
      document.getElementById('height-value').textContent = e.target.value + 'px';
      this.updatePreview();
    });

    document.getElementById('chart-title').addEventListener('input', () => {
      this.updatePreview();
    });

    document.getElementById('color-scheme').addEventListener('change', () => {
      this.updatePreview();
    });

    // Action buttons
    document.getElementById('generate-chart').addEventListener('click', () => {
      this.generateChart();
    });

    document.getElementById('save-custom-chart').addEventListener('click', () => {
      this.saveCustomChart();
    });

    document.getElementById('export-custom-chart').addEventListener('click', () => {
      this.exportCustomChart();
    });

    document.getElementById('share-custom-chart').addEventListener('click', () => {
      this.shareCustomChart();
    });
  }

  showBuilder() {
    document.getElementById('custom-viz-builder').classList.remove('hidden');
    document.getElementById('custom-viz-builder').classList.add('flex');
  }

  hideBuilder() {
    document.getElementById('custom-viz-builder').classList.add('hidden');
    document.getElementById('custom-viz-builder').classList.remove('flex');
  }

  selectChartType(type) {
    this.selectedChartType = type;

    // Update UI
    document.querySelectorAll('.chart-type-btn').forEach(btn => {
      btn.classList.remove('bg-blue-100', 'border-blue-500');
    });

    document.querySelector(`[data-type="${type}"]`).classList.add('bg-blue-100', 'border-blue-500');

    const chartType = this.chartTypes.find(ct => ct.id === type);
    document.getElementById('selected-chart-type').textContent = `Selected: ${chartType.name}`;

    // Show/hide relevant drop zones
    this.updateDropZones(type);
    this.checkCanGenerate();
  }

  updateDropZones(chartType) {
    // Reset zones
    document.querySelectorAll('.drop-zone').forEach(zone => {
      zone.style.display = 'block';
    });

    // Chart-specific zone visibility
    switch (chartType) {
      case 'pie':
        document.getElementById('y-axis-zone').style.display = 'none';
        document.getElementById('color-zone').style.display = 'block';
        break;
      case 'scatter':
        document.getElementById('color-zone').style.display = 'block';
        document.getElementById('size-zone').style.display = 'block';
        break;
      case 'heatmap':
        document.getElementById('color-zone').style.display = 'none';
        document.getElementById('size-zone').style.display = 'none';
        break;
      default:
        document.getElementById('color-zone').style.display = 'block';
        document.getElementById('size-zone').style.display = 'none';
    }
  }

  assignFieldToAxis(axis, fieldData) {
    if (!this.chartConfig) {
      this.chartConfig = {};
    }

    this.chartConfig[axis] = fieldData;

    // Update UI
    const zone = document.querySelector(`[data-axis="${axis}"]`);
    zone.innerHTML = `
      <div class="text-sm font-medium text-gray-700 mb-2">${axis.toUpperCase()}-Axis</div>
      <div class="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm">
        ${fieldData.name}
        <button class="ml-2 text-blue-600 hover:text-blue-800" onclick="customVizBuilder.clearAxis('${axis}')">&times;</button>
      </div>
    `;

    this.checkCanGenerate();
    this.updatePreview();
  }

  clearAxis(axis) {
    if (this.chartConfig) {
      delete this.chartConfig[axis];
    }

    // Reset zone UI
    const zone = document.querySelector(`[data-axis="${axis}"]`);
    const axisLabel = axis === 'color' ? 'Color By' : axis === 'size' ? 'Size By' : `${axis.toUpperCase()}-Axis`;
    zone.innerHTML = `
      <div class="text-sm font-medium text-gray-700 mb-2">${axisLabel}</div>
      <div class="text-xs text-gray-500">Drop a data field here</div>
    `;

    this.checkCanGenerate();
    this.updatePreview();
  }

  checkCanGenerate() {
    const hasRequiredFields = this.chartConfig &&
                            this.chartConfig.x &&
                            (this.selectedChartType === 'pie' || this.chartConfig.y) &&
                            this.selectedChartType;

    document.getElementById('generate-chart').disabled = !hasRequiredFields;
  }

  updatePreview() {
    if (!this.selectedChartType || !this.chartConfig) return;

    const preview = document.getElementById('chart-preview');
    preview.innerHTML = `
      <div class="text-center">
        <div class="text-lg font-medium mb-2">${document.getElementById('chart-title').value || 'Custom Chart'}</div>
        <div class="text-sm text-gray-600 mb-4">${this.chartTypes.find(ct => ct.id === this.selectedChartType).name}</div>
        <div class="bg-gray-200 rounded p-4">
          <div class="text-xs text-gray-500 mb-2">Configuration:</div>
          <div class="text-xs space-y-1">
            ${Object.entries(this.chartConfig).map(([axis, field]) =>
              `<div><strong>${axis.toUpperCase()}:</strong> ${field.name}</div>`
            ).join('')}
          </div>
        </div>
        <div class="mt-4 text-xs text-gray-500">
          ${document.getElementById('chart-width').value}px × ${document.getElementById('chart-height').value}px
        </div>
      </div>
    `;
  }

  generateChart() {
    if (!this.selectedChartType || !this.chartConfig) return;

    const chartConfig = {
      type: this.selectedChartType,
      title: document.getElementById('chart-title').value || 'Custom Chart',
      width: parseInt(document.getElementById('chart-width').value),
      height: parseInt(document.getElementById('chart-height').value),
      colorScheme: document.getElementById('color-scheme').value,
      fields: this.chartConfig,
      data: this.generateSampleData()
    };

    this.renderChart(chartConfig);

    // Enable action buttons
    document.getElementById('save-custom-chart').disabled = false;
    document.getElementById('export-custom-chart').disabled = false;
    document.getElementById('share-custom-chart').disabled = false;
  }

  generateSampleData() {
    // Generate sample data based on configured fields
    const data = [];
    for (let i = 0; i < 50; i++) {
      const point = {};

      Object.entries(this.chartConfig).forEach(([axis, field]) => {
        switch (field.type) {
          case 'numeric':
            point[field.field] = Math.random() * 100;
            break;
          case 'categorical':
            point[field.field] = ['Category A', 'Category B', 'Category C'][Math.floor(Math.random() * 3)];
            break;
          case 'datetime':
            point[field.field] = new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000);
            break;
        }
      });

      data.push(point);
    }

    return data;
  }

  renderChart(config) {
    const preview = document.getElementById('chart-preview');
    preview.innerHTML = '';

    const chartContainer = document.createElement('div');
    chartContainer.style.width = config.width + 'px';
    chartContainer.style.height = config.height + 'px';
    preview.appendChild(chartContainer);

    // Use ECharts for rendering
    const chart = echarts.init(chartContainer);
    const option = this.createEChartsOption(config);
    chart.setOption(option);

    this.currentChart = { chart, config };
  }

  createEChartsOption(config) {
    const { type, title, fields, data, colorScheme } = config;

    const colorSchemes = {
      default: ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'],
      blue: ['#1E40AF', '#3B82F6', '#60A5FA', '#93C5FD', '#DBEAFE'],
      green: ['#065F46', '#10B981', '#34D399', '#6EE7B7', '#A7F3D0'],
      red: ['#991B1B', '#EF4444', '#F87171', '#FCA5A5', '#FEE2E2'],
      rainbow: ['#EF4444', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6']
    };

    const colors = colorSchemes[colorScheme] || colorSchemes.default;

    let option = {
      title: { text: title, left: 'center' },
      tooltip: { trigger: 'item' },
      color: colors
    };

    // Configure based on chart type
    switch (type) {
      case 'line':
      case 'bar':
      case 'area':
        option.xAxis = {
          type: fields.x.type === 'datetime' ? 'time' : 'category',
          data: fields.x.type !== 'datetime' ? data.map(d => d[fields.x.field]) : undefined
        };
        option.yAxis = { type: 'value' };
        option.series = [{
          type: type === 'area' ? 'line' : type,
          data: data.map(d => fields.x.type === 'datetime' ?
            [d[fields.x.field], d[fields.y.field]] : d[fields.y.field]),
          areaStyle: type === 'area' ? {} : undefined
        }];
        break;

      case 'scatter':
        option.xAxis = { type: 'value' };
        option.yAxis = { type: 'value' };
        option.series = [{
          type: 'scatter',
          data: data.map(d => [d[fields.x.field], d[fields.y.field]]),
          symbolSize: d => fields.size ? d[fields.size.field] / 5 : 8
        }];
        break;

      case 'pie':
        const pieData = {};
        data.forEach(d => {
          const key = d[fields.x.field];
          pieData[key] = (pieData[key] || 0) + 1;
        });
        option.series = [{
          type: 'pie',
          data: Object.entries(pieData).map(([name, value]) => ({ name, value })),
          radius: '70%'
        }];
        break;

      case 'heatmap':
        option.xAxis = { type: 'category', data: [...new Set(data.map(d => d[fields.x.field]))] };
        option.yAxis = { type: 'category', data: [...new Set(data.map(d => d[fields.y.field]))] };
        option.visualMap = { min: 0, max: 100, calculable: true };
        option.series = [{
          type: 'heatmap',
          data: data.map(d => [d[fields.x.field], d[fields.y.field], Math.random() * 100])
        }];
        break;
    }

    return option;
  }

  saveCustomChart() {
    if (!this.currentChart) return;

    const chartData = {
      id: Date.now().toString(),
      name: this.currentChart.config.title,
      config: this.currentChart.config,
      created: new Date().toISOString()
    };

    this.customCharts.push(chartData);

    // Save to localStorage
    localStorage.setItem('pynomaly_custom_charts', JSON.stringify(this.customCharts));

    alert('Chart saved successfully!');
  }

  exportCustomChart() {
    if (!this.currentChart) return;

    const url = this.currentChart.chart.getDataURL({
      pixelRatio: 2,
      backgroundColor: '#fff'
    });

    const link = document.createElement('a');
    link.download = `${this.currentChart.config.title.replace(/\s+/g, '_')}_chart.png`;
    link.href = url;
    link.click();
  }

  shareCustomChart() {
    if (!this.currentChart) return;

    const shareData = {
      title: this.currentChart.config.title,
      config: this.currentChart.config,
      url: window.location.href
    };

    const shareUrl = `${window.location.origin}/share-chart?data=${encodeURIComponent(JSON.stringify(shareData))}`;

    navigator.clipboard.writeText(shareUrl).then(() => {
      alert('Share URL copied to clipboard!');
    });
  }

  loadSavedCharts() {
    const saved = localStorage.getItem('pynomaly_custom_charts');
    if (saved) {
      this.customCharts = JSON.parse(saved);
    }
    return this.customCharts;
  }
}

// Initialize the custom visualization builder
const customVizBuilder = new CustomVisualizationBuilder();

// Add button to open builder
document.addEventListener('DOMContentLoaded', function() {
  const button = document.createElement('button');
  button.innerHTML = 'Custom Chart Builder';
  button.className = 'px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 ml-2';
  button.onclick = () => customVizBuilder.showBuilder();

  const controls = document.querySelector('.viz-controls');
  if (controls) {
    controls.appendChild(button);
  }
});
