/**
 * Pynomaly Component Library
 * Interactive UI components for anomaly detection platform
 */

class ComponentLibrary {
  constructor() {
    this.components = new Map();
    this.eventBus = new EventTarget();
    this.init();
  }
  
  init() {
    this.registerComponents();
    this.setupEventListeners();
    this.initializeComponents();
  }
  
  /**
   * Register all available components
   */
  registerComponents() {
    this.components.set('anomaly-chart', AnomalyChart);
    this.components.set('data-table', DataTable);
    this.components.set('model-card', ModelCard);
    this.components.set('detection-status', DetectionStatus);
    this.components.set('notification-center', NotificationCenter);
    this.components.set('file-uploader', FileUploader);
    this.components.set('progress-indicator', ProgressIndicator);
    this.components.set('interactive-dashboard', InteractiveDashboard);
    this.components.set('algorithm-selector', AlgorithmSelector);
    this.components.set('threshold-slider', ThresholdSlider);
  }
  
  /**
   * Setup global event listeners
   */
  setupEventListeners() {
    document.addEventListener('DOMContentLoaded', () => {
      this.initializeComponents();
    });
    
    // Handle dynamic component loading
    this.eventBus.addEventListener('component:load', (event) => {
      this.loadComponent(event.detail.type, event.detail.target, event.detail.options);
    });
  }
  
  /**
   * Initialize all components on the page
   */
  initializeComponents() {
    // Find all component elements and initialize them
    document.querySelectorAll('[data-component]').forEach(element => {
      const componentType = element.getAttribute('data-component');
      const options = this.parseOptions(element.getAttribute('data-options'));
      this.loadComponent(componentType, element, options);
    });
  }
  
  /**
   * Load a specific component
   */
  loadComponent(type, element, options = {}) {
    const ComponentClass = this.components.get(type);
    if (ComponentClass) {
      new ComponentClass(element, options);
    } else {
      console.warn(`Component type "${type}" not found`);
    }
  }
  
  /**
   * Parse options from data attribute
   */
  parseOptions(optionsString) {
    if (!optionsString) return {};
    try {
      return JSON.parse(optionsString);
    } catch (error) {
      console.warn('Failed to parse component options:', error);
      return {};
    }
  }
}

/**
 * Base Component Class
 */
class BaseComponent {
  constructor(element, options = {}) {
    this.element = element;
    this.options = { ...this.defaultOptions, ...options };
    this.id = this.generateId();
    this.state = {};
    this.events = new EventTarget();
    
    this.init();
  }
  
  get defaultOptions() {
    return {};
  }
  
  init() {
    this.setupElement();
    this.render();
    this.bindEvents();
  }
  
  setupElement() {
    this.element.setAttribute('data-component-id', this.id);
    this.element.classList.add('component', `component--${this.constructor.name.toLowerCase()}`);
  }
  
  render() {
    // Override in subclasses
  }
  
  bindEvents() {
    // Override in subclasses
  }
  
  generateId() {
    return `component-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  setState(newState) {
    this.state = { ...this.state, ...newState };
    this.onStateChange();
  }
  
  onStateChange() {
    this.render();
  }
  
  emit(event, data) {
    this.events.dispatchEvent(new CustomEvent(event, { detail: data }));
  }
  
  on(event, handler) {
    this.events.addEventListener(event, handler);
  }
  
  destroy() {
    this.element.removeAttribute('data-component-id');
    this.element.classList.remove('component', `component--${this.constructor.name.toLowerCase()}`);
    this.element.innerHTML = '';
  }
}

/**
 * Anomaly Chart Component
 * Interactive time series visualization with anomaly highlighting
 */
class AnomalyChart extends BaseComponent {
  get defaultOptions() {
    return {
      width: 800,
      height: 400,
      margin: { top: 20, right: 30, bottom: 40, left: 50 },
      showBrush: true,
      showTooltip: true,
      animationDuration: 750,
      colors: {
        normal: '#22c55e',
        anomaly: '#ef4444',
        threshold: '#f59e0b',
        prediction: '#8b5cf6'
      }
    };
  }
  
  init() {
    super.init();
    this.loadD3();
  }
  
  async loadD3() {
    if (typeof d3 === 'undefined') {
      await this.loadScript('/static/js/d3.min.js');
    }
    this.setupChart();
  }
  
  loadScript(src) {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = src;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }
  
  setupChart() {
    const { width, height, margin } = this.options;
    
    this.svg = d3.select(this.element)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('class', 'anomaly-chart');
    
    this.g = this.svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    this.width = width - margin.left - margin.right;
    this.height = height - margin.top - margin.bottom;
    
    this.setupScales();
    this.setupAxes();
    this.setupInteractivity();
  }
  
  setupScales() {
    this.xScale = d3.scaleTime().range([0, this.width]);
    this.yScale = d3.scaleLinear().range([this.height, 0]);
  }
  
  setupAxes() {
    this.xAxis = d3.axisBottom(this.xScale);
    this.yAxis = d3.axisLeft(this.yScale);
    
    this.g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${this.height})`);
    
    this.g.append('g')
      .attr('class', 'y-axis');
  }
  
  setupInteractivity() {
    if (this.options.showBrush) {
      this.brush = d3.brushX()
        .extent([[0, 0], [this.width, this.height]])
        .on('brush end', (event) => this.onBrush(event));
      
      this.g.append('g')
        .attr('class', 'brush')
        .call(this.brush);
    }
    
    if (this.options.showTooltip) {
      this.tooltip = this.createTooltip();
    }
  }
  
  createTooltip() {
    return d3.select('body').append('div')
      .attr('class', 'chart-tooltip')
      .style('opacity', 0)
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.8)')
      .style('color', 'white')
      .style('padding', '8px')
      .style('border-radius', '4px')
      .style('font-size', '12px')
      .style('pointer-events', 'none');
  }
  
  render() {
    if (!this.state.data) return;
    
    this.updateScales();
    this.updateAxes();
    this.renderDataPoints();
    this.renderAnomalies();
    this.renderThreshold();
  }
  
  updateScales() {
    const { data } = this.state;
    this.xScale.domain(d3.extent(data, d => d.timestamp));
    this.yScale.domain(d3.extent(data, d => d.value));
  }
  
  updateAxes() {
    this.g.select('.x-axis')
      .transition()
      .duration(this.options.animationDuration)
      .call(this.xAxis);
    
    this.g.select('.y-axis')
      .transition()
      .duration(this.options.animationDuration)
      .call(this.yAxis);
  }
  
  renderDataPoints() {
    const line = d3.line()
      .x(d => this.xScale(d.timestamp))
      .y(d => this.yScale(d.value))
      .curve(d3.curveMonotoneX);
    
    const path = this.g.selectAll('.data-line')
      .data([this.state.data]);
    
    path.enter()
      .append('path')
      .attr('class', 'data-line')
      .attr('fill', 'none')
      .attr('stroke', this.options.colors.normal)
      .attr('stroke-width', 2)
      .merge(path)
      .transition()
      .duration(this.options.animationDuration)
      .attr('d', line);
  }
  
  renderAnomalies() {
    const anomalies = this.state.data.filter(d => d.isAnomaly);
    
    const circles = this.g.selectAll('.anomaly-point')
      .data(anomalies);
    
    circles.enter()
      .append('circle')
      .attr('class', 'anomaly-point')
      .attr('r', 0)
      .attr('fill', this.options.colors.anomaly)
      .merge(circles)
      .on('mouseover', (event, d) => this.showTooltip(event, d))
      .on('mouseout', () => this.hideTooltip())
      .transition()
      .duration(this.options.animationDuration)
      .attr('cx', d => this.xScale(d.timestamp))
      .attr('cy', d => this.yScale(d.value))
      .attr('r', 4);
    
    circles.exit()
      .transition()
      .duration(this.options.animationDuration)
      .attr('r', 0)
      .remove();
  }
  
  renderThreshold() {
    if (!this.state.threshold) return;
    
    const thresholdLine = this.g.selectAll('.threshold-line')
      .data([this.state.threshold]);
    
    thresholdLine.enter()
      .append('line')
      .attr('class', 'threshold-line')
      .attr('stroke', this.options.colors.threshold)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5')
      .merge(thresholdLine)
      .transition()
      .duration(this.options.animationDuration)
      .attr('x1', 0)
      .attr('x2', this.width)
      .attr('y1', d => this.yScale(d))
      .attr('y2', d => this.yScale(d));
  }
  
  showTooltip(event, d) {
    if (!this.tooltip) return;
    
    this.tooltip.transition()
      .duration(200)
      .style('opacity', .9);
    
    this.tooltip.html(`
      <div><strong>Time:</strong> ${d.timestamp.toLocaleString()}</div>
      <div><strong>Value:</strong> ${d.value.toFixed(2)}</div>
      <div><strong>Status:</strong> ${d.isAnomaly ? 'Anomaly' : 'Normal'}</div>
      ${d.score ? `<div><strong>Score:</strong> ${d.score.toFixed(3)}</div>` : ''}
    `)
      .style('left', (event.pageX + 10) + 'px')
      .style('top', (event.pageY - 28) + 'px');
  }
  
  hideTooltip() {
    if (!this.tooltip) return;
    
    this.tooltip.transition()
      .duration(500)
      .style('opacity', 0);
  }
  
  onBrush(event) {
    if (!event.selection) return;
    
    const [x0, x1] = event.selection;
    const range = [this.xScale.invert(x0), this.xScale.invert(x1)];
    
    this.emit('brush', { range });
  }
  
  setData(data) {
    this.setState({ data });
  }
  
  setThreshold(threshold) {
    this.setState({ threshold });
  }
}

/**
 * Data Table Component
 * Sortable, filterable table with pagination
 */
class DataTable extends BaseComponent {
  get defaultOptions() {
    return {
      pageSize: 10,
      sortable: true,
      filterable: true,
      searchable: true,
      exportable: true,
      selectable: false,
      columns: []
    };
  }
  
  render() {
    this.element.innerHTML = `
      <div class="data-table">
        ${this.renderToolbar()}
        ${this.renderTable()}
        ${this.renderPagination()}
      </div>
    `;
    
    this.bindTableEvents();
  }
  
  renderToolbar() {
    return `
      <div class="data-table__toolbar flex justify-between items-center mb-4">
        <div class="flex items-center gap-4">
          ${this.options.searchable ? `
            <div class="relative">
              <input type="text" placeholder="Search..." 
                     class="form-input search-input pl-10" 
                     data-action="search">
              <svg class="w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" 
                   fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clip-rule="evenodd"></path>
              </svg>
            </div>
          ` : ''}
          
          <select class="form-select" data-action="page-size">
            <option value="10">10 per page</option>
            <option value="25">25 per page</option>
            <option value="50">50 per page</option>
            <option value="100">100 per page</option>
          </select>
        </div>
        
        <div class="flex items-center gap-2">
          ${this.options.exportable ? `
            <button class="btn-base btn-secondary btn-sm" data-action="export">
              <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd"></path>
              </svg>
              Export
            </button>
          ` : ''}
          
          <button class="btn-base btn-secondary btn-sm" data-action="refresh">
            <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd"></path>
            </svg>
            Refresh
          </button>
        </div>
      </div>
    `;
  }
  
  renderTable() {
    const { data, columns } = this.state;
    if (!data || !columns) return '<div class="text-center py-8 text-gray-500">No data available</div>';
    
    return `
      <div class="overflow-x-auto">
        <table class="w-full">
          <thead class="bg-gray-50">
            <tr>
              ${this.options.selectable ? '<th class="px-6 py-3"><input type="checkbox" data-action="select-all"></th>' : ''}
              ${columns.map(col => `
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider ${this.options.sortable ? 'cursor-pointer hover:bg-gray-100' : ''}"
                    data-action="sort" data-column="${col.key}">
                  <div class="flex items-center gap-2">
                    ${col.label}
                    ${this.options.sortable ? `
                      <svg class="w-4 h-4 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M5 12a1 1 0 102 0V6.414l1.293 1.293a1 1 0 001.414-1.414l-3-3a1 1 0 00-1.414 0l-3 3a1 1 0 001.414 1.414L5 6.414V12zM15 8a1 1 0 10-2 0v5.586l-1.293-1.293a1 1 0 00-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L15 13.586V8z"></path>
                      </svg>
                    ` : ''}
                  </div>
                </th>
              `).join('')}
            </tr>
          </thead>
          <tbody class="bg-white divide-y divide-gray-200">
            ${this.renderRows()}
          </tbody>
        </table>
      </div>
    `;
  }
  
  renderRows() {
    const { data, columns, currentPage = 0, pageSize = this.options.pageSize } = this.state;
    const start = currentPage * pageSize;
    const end = start + pageSize;
    const pageData = data.slice(start, end);
    
    return pageData.map(row => `
      <tr class="hover:bg-gray-50">
        ${this.options.selectable ? `<td class="px-6 py-4"><input type="checkbox" data-action="select-row" value="${row.id}"></td>` : ''}
        ${columns.map(col => `
          <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
            ${this.formatCellValue(row[col.key], col)}
          </td>
        `).join('')}
      </tr>
    `).join('');
  }
  
  formatCellValue(value, column) {
    if (column.formatter) {
      return column.formatter(value);
    }
    
    if (column.type === 'number') {
      return typeof value === 'number' ? value.toLocaleString() : value;
    }
    
    if (column.type === 'date') {
      return value instanceof Date ? value.toLocaleDateString() : value;
    }
    
    if (column.type === 'boolean') {
      return value ? '<span class="badge badge-secondary">Yes</span>' : '<span class="badge badge-neutral">No</span>';
    }
    
    return value;
  }
  
  renderPagination() {
    const { data, currentPage = 0, pageSize = this.options.pageSize } = this.state;
    if (!data) return '';
    
    const totalPages = Math.ceil(data.length / pageSize);
    if (totalPages <= 1) return '';
    
    return `
      <div class="flex items-center justify-between px-4 py-3 bg-white border-t border-gray-200 sm:px-6">
        <div class="flex items-center">
          <p class="text-sm text-gray-700">
            Showing ${currentPage * pageSize + 1} to ${Math.min((currentPage + 1) * pageSize, data.length)} of ${data.length} results
          </p>
        </div>
        <div class="flex items-center gap-2">
          <button class="btn-base btn-secondary btn-sm" 
                  data-action="prev-page" 
                  ${currentPage === 0 ? 'disabled' : ''}>
            Previous
          </button>
          
          <div class="flex gap-1">
            ${this.renderPageNumbers(currentPage, totalPages)}
          </div>
          
          <button class="btn-base btn-secondary btn-sm" 
                  data-action="next-page" 
                  ${currentPage >= totalPages - 1 ? 'disabled' : ''}>
            Next
          </button>
        </div>
      </div>
    `;
  }
  
  renderPageNumbers(currentPage, totalPages) {
    const pages = [];
    const maxVisible = 5;
    
    let start = Math.max(0, currentPage - Math.floor(maxVisible / 2));
    let end = Math.min(totalPages, start + maxVisible);
    
    if (end - start < maxVisible) {
      start = Math.max(0, end - maxVisible);
    }
    
    for (let i = start; i < end; i++) {
      pages.push(`
        <button class="btn-base btn-sm ${i === currentPage ? 'btn-primary' : 'btn-secondary'}" 
                data-action="goto-page" 
                data-page="${i}">
          ${i + 1}
        </button>
      `);
    }
    
    return pages.join('');
  }
  
  bindTableEvents() {
    this.element.addEventListener('click', (event) => {
      const action = event.target.closest('[data-action]')?.getAttribute('data-action');
      if (!action) return;
      
      switch (action) {
        case 'sort':
          this.handleSort(event.target.closest('[data-column]').getAttribute('data-column'));
          break;
        case 'prev-page':
          this.handlePrevPage();
          break;
        case 'next-page':
          this.handleNextPage();
          break;
        case 'goto-page':
          this.handleGotoPage(parseInt(event.target.getAttribute('data-page')));
          break;
        case 'export':
          this.handleExport();
          break;
        case 'refresh':
          this.handleRefresh();
          break;
      }
    });
    
    this.element.addEventListener('change', (event) => {
      const action = event.target.getAttribute('data-action');
      if (!action) return;
      
      switch (action) {
        case 'page-size':
          this.handlePageSizeChange(parseInt(event.target.value));
          break;
      }
    });
    
    this.element.addEventListener('input', (event) => {
      const action = event.target.getAttribute('data-action');
      if (action === 'search') {
        this.handleSearch(event.target.value);
      }
    });
  }
  
  handleSort(column) {
    const { sortColumn, sortDirection } = this.state;
    const newDirection = sortColumn === column && sortDirection === 'asc' ? 'desc' : 'asc';
    
    this.setState({
      sortColumn: column,
      sortDirection: newDirection,
      currentPage: 0
    });
    
    this.sortData(column, newDirection);
  }
  
  sortData(column, direction) {
    const { data } = this.state;
    const sortedData = [...data].sort((a, b) => {
      const aVal = a[column];
      const bVal = b[column];
      
      if (aVal < bVal) return direction === 'asc' ? -1 : 1;
      if (aVal > bVal) return direction === 'asc' ? 1 : -1;
      return 0;
    });
    
    this.setState({ data: sortedData });
  }
  
  handlePrevPage() {
    const { currentPage } = this.state;
    if (currentPage > 0) {
      this.setState({ currentPage: currentPage - 1 });
    }
  }
  
  handleNextPage() {
    const { currentPage, data, pageSize = this.options.pageSize } = this.state;
    const totalPages = Math.ceil(data.length / pageSize);
    if (currentPage < totalPages - 1) {
      this.setState({ currentPage: currentPage + 1 });
    }
  }
  
  handleGotoPage(page) {
    this.setState({ currentPage: page });
  }
  
  handlePageSizeChange(newPageSize) {
    this.setState({ 
      pageSize: newPageSize,
      currentPage: 0
    });
  }
  
  handleSearch(query) {
    const { originalData, columns } = this.state;
    if (!originalData) return;
    
    if (!query.trim()) {
      this.setState({ data: originalData, currentPage: 0 });
      return;
    }
    
    const filteredData = originalData.filter(row => {
      return columns.some(col => {
        const value = String(row[col.key]).toLowerCase();
        return value.includes(query.toLowerCase());
      });
    });
    
    this.setState({ 
      data: filteredData,
      currentPage: 0
    });
  }
  
  handleExport() {
    const { data, columns } = this.state;
    if (!data || !columns) return;
    
    const csv = this.generateCSV(data, columns);
    this.downloadCSV(csv, 'data-export.csv');
  }
  
  generateCSV(data, columns) {
    const headers = columns.map(col => col.label).join(',');
    const rows = data.map(row => {
      return columns.map(col => {
        const value = row[col.key];
        return typeof value === 'string' && value.includes(',') ? `"${value}"` : value;
      }).join(',');
    });
    
    return [headers, ...rows].join('\n');
  }
  
  downloadCSV(csv, filename) {
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
  
  handleRefresh() {
    this.emit('refresh');
  }
  
  setData(data, columns) {
    this.setState({ 
      data: [...data],
      originalData: [...data],
      columns,
      currentPage: 0
    });
  }
}

// Initialize the component library
window.componentLibrary = new ComponentLibrary();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ComponentLibrary, BaseComponent, AnomalyChart, DataTable };
}