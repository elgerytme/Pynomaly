/**
 * Chart Components Stories
 * Interactive documentation for Pynomaly data visualization components
 */

export default {
  title: 'Components/Charts',
  tags: ['autodocs'],
  parameters: {
    docs: {
      description: {
        component: 'Accessible data visualization components for anomaly detection in the Pynomaly platform. All charts include alternative text, keyboard navigation, and screen reader support.',
      },
    },
    a11y: {
      config: {
        rules: [
          { id: 'color-contrast', enabled: true },
          { id: 'focus-visible', enabled: true },
          { id: 'img-alt', enabled: true },
        ],
      },
    },
  },
};

// Helper function to generate sample data
const generateSampleData = (length = 50, anomalies = 5) => {
  const data = [];
  const anomalyIndices = new Set();
  
  // Generate random anomaly positions
  while (anomalyIndices.size < anomalies) {
    anomalyIndices.add(Math.floor(Math.random() * length));
  }
  
  for (let i = 0; i < length; i++) {
    const timestamp = new Date(Date.now() - (length - i) * 60000).toISOString();
    const isAnomaly = anomalyIndices.has(i);
    const baseValue = 50 + Math.sin(i * 0.2) * 20 + Math.random() * 10;
    const value = isAnomaly ? baseValue + (Math.random() > 0.5 ? 30 : -30) : baseValue;
    
    data.push({
      timestamp,
      value: parseFloat(value.toFixed(2)),
      anomalyScore: isAnomaly ? Math.random() * 0.4 + 0.6 : Math.random() * 0.3,
      isAnomaly,
    });
  }
  
  return data;
};

// Time Series Chart Component
const createTimeSeriesChart = ({
  data = generateSampleData(),
  width = 800,
  height = 400,
  title = 'Anomaly Detection Results',
  showLegend = true,
  interactive = true,
  threshold = 0.5,
}) => {
  const container = document.createElement('div');
  container.className = 'chart-container bg-white rounded-lg shadow-lg p-6';
  container.setAttribute('role', 'img');
  container.setAttribute('aria-label', title);
  
  // Chart title
  const titleElement = document.createElement('h3');
  titleElement.id = 'chart-title-' + Math.random().toString(36).substr(2, 9);
  titleElement.className = 'text-lg font-semibold text-gray-900 mb-4';
  titleElement.textContent = title;
  container.appendChild(titleElement);
  
  // Chart description (screen reader)
  const description = document.createElement('div');
  description.className = 'sr-only';
  description.id = 'chart-desc-' + Math.random().toString(36).substr(2, 9);
  const anomaliesCount = data.filter(d => d.isAnomaly).length;
  description.textContent = `Time series chart showing ${data.length} data points over time, with ${anomaliesCount} anomalies detected. Normal values range from ${Math.min(...data.filter(d => !d.isAnomaly).map(d => d.value)).toFixed(1)} to ${Math.max(...data.filter(d => !d.isAnomaly).map(d => d.value)).toFixed(1)}. Anomalies are highlighted in red.`;
  container.appendChild(description);
  
  // SVG Chart
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.className = 'w-full h-auto border rounded';
  svg.setAttribute('aria-labelledby', titleElement.id);
  svg.setAttribute('aria-describedby', description.id);
  
  // Chart margins
  const margin = { top: 20, right: 30, bottom: 40, left: 50 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  
  // Scales
  const minValue = Math.min(...data.map(d => d.value));
  const maxValue = Math.max(...data.map(d => d.value));
  const valueRange = maxValue - minValue;
  const yScale = (value) => chartHeight - ((value - minValue) / valueRange) * chartHeight;
  const xScale = (index) => (index / (data.length - 1)) * chartWidth;
  
  // Chart group
  const chartGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  chartGroup.setAttribute('transform', `translate(${margin.left}, ${margin.top})`);
  svg.appendChild(chartGroup);
  
  // Grid lines
  const gridGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  gridGroup.setAttribute('class', 'grid');
  chartGroup.appendChild(gridGroup);
  
  // Horizontal grid lines
  for (let i = 0; i <= 5; i++) {
    const y = (i / 5) * chartHeight;
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', 0);
    line.setAttribute('y1', y);
    line.setAttribute('x2', chartWidth);
    line.setAttribute('y2', y);
    line.setAttribute('stroke', '#e5e7eb');
    line.setAttribute('stroke-width', '1');
    gridGroup.appendChild(line);
  }
  
  // Data line path
  const linePath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  const pathData = data.map((d, i) => {
    const x = xScale(i);
    const y = yScale(d.value);
    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
  }).join(' ');
  
  linePath.setAttribute('d', pathData);
  linePath.setAttribute('fill', 'none');
  linePath.setAttribute('stroke', '#3b82f6');
  linePath.setAttribute('stroke-width', '2');
  chartGroup.appendChild(linePath);
  
  // Data points
  data.forEach((d, i) => {
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', xScale(i));
    circle.setAttribute('cy', yScale(d.value));
    circle.setAttribute('r', d.isAnomaly ? 6 : 3);
    circle.setAttribute('fill', d.isAnomaly ? '#ef4444' : '#3b82f6');
    circle.setAttribute('stroke', '#ffffff');
    circle.setAttribute('stroke-width', '2');
    
    if (interactive) {
      circle.setAttribute('tabindex', '0');
      circle.setAttribute('role', 'button');
      circle.setAttribute('aria-label', `Data point ${i + 1}: Value ${d.value}, ${d.isAnomaly ? 'Anomaly' : 'Normal'} at ${new Date(d.timestamp).toLocaleTimeString()}`);
      circle.style.cursor = 'pointer';
      
      // Hover effects
      circle.addEventListener('mouseenter', () => {
        circle.setAttribute('r', (d.isAnomaly ? 6 : 3) + 2);
        showTooltip(d, i);
      });
      
      circle.addEventListener('mouseleave', () => {
        circle.setAttribute('r', d.isAnomaly ? 6 : 3);
        hideTooltip();
      });
      
      // Keyboard interaction
      circle.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          showTooltip(d, i, true);
        }
      });
    }
    
    chartGroup.appendChild(circle);
  });
  
  // Axes
  // X-axis
  const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  xAxis.setAttribute('x1', 0);
  xAxis.setAttribute('y1', chartHeight);
  xAxis.setAttribute('x2', chartWidth);
  xAxis.setAttribute('y2', chartHeight);
  xAxis.setAttribute('stroke', '#374151');
  xAxis.setAttribute('stroke-width', '2');
  chartGroup.appendChild(xAxis);
  
  // Y-axis
  const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  yAxis.setAttribute('x1', 0);
  yAxis.setAttribute('y1', 0);
  yAxis.setAttribute('x2', 0);
  yAxis.setAttribute('y2', chartHeight);
  yAxis.setAttribute('stroke', '#374151');
  yAxis.setAttribute('stroke-width', '2');
  chartGroup.appendChild(yAxis);
  
  // Axis labels
  const yAxisLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  yAxisLabel.setAttribute('x', -margin.left + 15);
  yAxisLabel.setAttribute('y', chartHeight / 2);
  yAxisLabel.setAttribute('text-anchor', 'middle');
  yAxisLabel.setAttribute('transform', `rotate(-90, ${-margin.left + 15}, ${chartHeight / 2})`);
  yAxisLabel.setAttribute('fill', '#374151');
  yAxisLabel.setAttribute('font-size', '12');
  yAxisLabel.textContent = 'Value';
  chartGroup.appendChild(yAxisLabel);
  
  const xAxisLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  xAxisLabel.setAttribute('x', chartWidth / 2);
  xAxisLabel.setAttribute('y', chartHeight + margin.bottom - 5);
  xAxisLabel.setAttribute('text-anchor', 'middle');
  xAxisLabel.setAttribute('fill', '#374151');
  xAxisLabel.setAttribute('font-size', '12');
  xAxisLabel.textContent = 'Time';
  chartGroup.appendChild(xAxisLabel);
  
  container.appendChild(svg);
  
  // Legend
  if (showLegend) {
    const legend = document.createElement('div');
    legend.className = 'flex items-center justify-center gap-6 mt-4';
    legend.setAttribute('role', 'group');
    legend.setAttribute('aria-label', 'Chart legend');
    
    const normalLegend = document.createElement('div');
    normalLegend.className = 'flex items-center gap-2';
    normalLegend.innerHTML = `
      <div class="w-3 h-3 bg-blue-500 rounded-full"></div>
      <span class="text-sm text-gray-600">Normal Data</span>
    `;
    
    const anomalyLegend = document.createElement('div');
    anomalyLegend.className = 'flex items-center gap-2';
    anomalyLegend.innerHTML = `
      <div class="w-3 h-3 bg-red-500 rounded-full"></div>
      <span class="text-sm text-gray-600">Anomalies</span>
    `;
    
    legend.appendChild(normalLegend);
    legend.appendChild(anomalyLegend);
    container.appendChild(legend);
  }
  
  // Tooltip
  const tooltip = document.createElement('div');
  tooltip.className = 'absolute bg-gray-800 text-white p-2 rounded shadow-lg text-sm pointer-events-none opacity-0 transition-opacity';
  tooltip.style.cssText = 'z-index: 1000; transform: translate(-50%, -100%);';
  tooltip.setAttribute('role', 'tooltip');
  document.body.appendChild(tooltip);
  
  const showTooltip = (d, i, persistent = false) => {
    tooltip.innerHTML = `
      <div class="font-semibold">${d.isAnomaly ? 'Anomaly' : 'Normal Data'}</div>
      <div>Value: ${d.value}</div>
      <div>Score: ${d.anomalyScore.toFixed(3)}</div>
      <div>Time: ${new Date(d.timestamp).toLocaleString()}</div>
    `;
    tooltip.style.opacity = '1';
    
    // Position tooltip (simplified for demo)
    tooltip.style.left = '50%';
    tooltip.style.top = '50%';
  };
  
  const hideTooltip = () => {
    tooltip.style.opacity = '0';
  };
  
  // Alternative data table
  const tableContainer = document.createElement('details');
  tableContainer.className = 'mt-6';
  tableContainer.innerHTML = `
    <summary class="cursor-pointer text-blue-600 hover:text-blue-800 font-medium">
      View data table (${data.length} data points)
    </summary>
  `;
  
  const table = document.createElement('table');
  table.className = 'mt-4 w-full text-sm border-collapse border border-gray-300';
  table.innerHTML = `
    <caption class="sr-only">Anomaly detection data table with timestamps, values, and anomaly scores</caption>
    <thead>
      <tr class="bg-gray-50">
        <th scope="col" class="border border-gray-300 px-3 py-2 text-left">Index</th>
        <th scope="col" class="border border-gray-300 px-3 py-2 text-left">Timestamp</th>
        <th scope="col" class="border border-gray-300 px-3 py-2 text-right">Value</th>
        <th scope="col" class="border border-gray-300 px-3 py-2 text-right">Anomaly Score</th>
        <th scope="col" class="border border-gray-300 px-3 py-2 text-center">Status</th>
      </tr>
    </thead>
    <tbody>
      ${data.map((d, i) => `
        <tr class="${d.isAnomaly ? 'bg-red-50' : 'bg-white'}">
          <td class="border border-gray-300 px-3 py-2">${i + 1}</td>
          <td class="border border-gray-300 px-3 py-2">${new Date(d.timestamp).toLocaleString()}</td>
          <td class="border border-gray-300 px-3 py-2 text-right">${d.value}</td>
          <td class="border border-gray-300 px-3 py-2 text-right">${d.anomalyScore.toFixed(3)}</td>
          <td class="border border-gray-300 px-3 py-2 text-center">
            <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
              d.isAnomaly 
                ? 'bg-red-100 text-red-800' 
                : 'bg-green-100 text-green-800'
            }">
              ${d.isAnomaly ? 'Anomaly' : 'Normal'}
            </span>
          </td>
        </tr>
      `).join('')}
    </tbody>
  `;
  
  tableContainer.appendChild(table);
  container.appendChild(tableContainer);
  
  return container;
};

// Anomaly Score Distribution Chart
const createScoreDistribution = ({ data = generateSampleData() }) => {
  const container = document.createElement('div');
  container.className = 'chart-container bg-white rounded-lg shadow-lg p-6';
  
  const title = document.createElement('h3');
  title.className = 'text-lg font-semibold text-gray-900 mb-4';
  title.textContent = 'Anomaly Score Distribution';
  container.appendChild(title);
  
  // Create histogram data
  const bins = 10;
  const minScore = 0;
  const maxScore = 1;
  const binWidth = (maxScore - minScore) / bins;
  const histogram = new Array(bins).fill(0);
  
  data.forEach(d => {
    const binIndex = Math.min(Math.floor(d.anomalyScore / binWidth), bins - 1);
    histogram[binIndex]++;
  });
  
  const maxCount = Math.max(...histogram);
  
  // Create bar chart
  const chartContainer = document.createElement('div');
  chartContainer.className = 'flex items-end gap-1 h-48 p-4 border rounded';
  chartContainer.setAttribute('role', 'img');
  chartContainer.setAttribute('aria-label', 'Histogram showing distribution of anomaly scores');
  
  histogram.forEach((count, i) => {
    const bar = document.createElement('div');
    const height = (count / maxCount) * 100;
    bar.className = 'bg-blue-500 transition-all duration-200 hover:bg-blue-600';
    bar.style.cssText = `
      height: ${height}%;
      flex: 1;
      min-height: 2px;
      position: relative;
    `;
    bar.setAttribute('title', `Score range ${(i * binWidth).toFixed(2)}-${((i + 1) * binWidth).toFixed(2)}: ${count} data points`);
    
    // Label
    const label = document.createElement('div');
    label.className = 'absolute -bottom-6 left-1/2 transform -translate-x-1/2 text-xs text-gray-600';
    label.textContent = (i * binWidth).toFixed(1);
    bar.appendChild(label);
    
    chartContainer.appendChild(bar);
  });
  
  container.appendChild(chartContainer);
  
  return container;
};

// Stories
export const TimeSeriesChart = {
  render: () => createTimeSeriesChart({
    title: 'Real-time Anomaly Detection',
    data: generateSampleData(100, 8),
    interactive: true,
  }),
  parameters: {
    docs: {
      description: {
        story: 'Interactive time series chart showing anomaly detection results. Red points indicate detected anomalies. Hover over points for details, or use keyboard navigation.',
      },
    },
  },
};

export const SimpleChart = {
  render: () => createTimeSeriesChart({
    title: 'Basic Anomaly Chart',
    data: generateSampleData(30, 3),
    interactive: false,
    showLegend: false,
  }),
  parameters: {
    docs: {
      description: {
        story: 'Simplified chart without interactive features, suitable for static displays.',
      },
    },
  },
};

export const HighAnomalyData = {
  render: () => createTimeSeriesChart({
    title: 'High Anomaly Rate Dataset',
    data: generateSampleData(50, 15),
    interactive: true,
  }),
  parameters: {
    docs: {
      description: {
        story: 'Chart displaying a dataset with a high number of anomalies to test visualization clarity.',
      },
    },
  },
};

export const ScoreDistribution = {
  render: () => createScoreDistribution({
    data: generateSampleData(200, 20),
  }),
  parameters: {
    docs: {
      description: {
        story: 'Histogram showing the distribution of anomaly scores across the dataset.',
      },
    },
  },
};

export const DashboardCharts = {
  render: () => {
    const dashboard = document.createElement('div');
    dashboard.className = 'grid grid-cols-1 lg:grid-cols-2 gap-6';
    
    const timeSeriesData = generateSampleData(60, 6);
    
    const mainChart = createTimeSeriesChart({
      title: 'Primary Detection Results',
      data: timeSeriesData,
      interactive: true,
    });
    
    const distributionChart = createScoreDistribution({
      data: timeSeriesData,
    });
    
    dashboard.appendChild(mainChart);
    dashboard.appendChild(distributionChart);
    
    return dashboard;
  },
  parameters: {
    docs: {
      description: {
        story: 'Dashboard layout showing multiple related charts for comprehensive anomaly analysis.',
      },
    },
  },
};

export const AccessibilityFeatures = {
  render: () => {
    const container = document.createElement('div');
    container.className = 'space-y-6';
    
    const title = document.createElement('h2');
    title.className = 'text-xl font-bold text-gray-900';
    title.textContent = 'Accessibility Features Demo';
    container.appendChild(title);
    
    const features = document.createElement('div');
    features.className = 'bg-blue-50 border border-blue-200 rounded-lg p-4 space-y-2';
    features.innerHTML = `
      <h3 class="font-semibold text-blue-900">Accessibility Features:</h3>
      <ul class="text-sm text-blue-800 space-y-1">
        <li>• <strong>Screen reader support:</strong> Charts have descriptive aria-labels and alternative text</li>
        <li>• <strong>Keyboard navigation:</strong> All interactive elements are keyboard accessible</li>
        <li>• <strong>Alternative data table:</strong> Expandable table view for all chart data</li>
        <li>• <strong>High contrast colors:</strong> Sufficient color contrast for all elements</li>
        <li>• <strong>Tooltip information:</strong> Detailed information on hover and focus</li>
        <li>• <strong>Color independence:</strong> Information not conveyed by color alone</li>
      </ul>
    `;
    container.appendChild(features);
    
    const accessibleChart = createTimeSeriesChart({
      title: 'Fully Accessible Anomaly Chart',
      data: generateSampleData(40, 5),
      interactive: true,
    });
    container.appendChild(accessibleChart);
    
    return container;
  },
  parameters: {
    docs: {
      description: {
        story: 'Demonstration of all accessibility features implemented in Pynomaly charts.',
      },
    },
  },
};
