/**
 * Advanced Dashboard Layout System
 * Drag-and-drop, resizable dashboard with widget management
 */

import { dashboardState } from '../state/dashboard-state.js';

export class DashboardLayoutEngine {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      gridSize: 20,
      minWidgetWidth: 200,
      minWidgetHeight: 150,
      maxColumns: 12,
      autoResize: true,
      persistLayout: true,
      animationDuration: 300,
      ...options
    };
    
    this.widgets = new Map();
    this.layout = [];
    this.dragState = null;
    this.resizeState = null;
    this.gridWidth = 0;
    this.columnWidth = 0;
    
    this.init();
  }
  
  init() {
    this.createLayoutContainer();
    this.calculateGrid();
    this.setupEventHandlers();
    this.loadSavedLayout();
    this.setupResizeObserver();
  }
  
  createLayoutContainer() {
    this.container.className = `${this.container.className} dashboard-layout`.trim();
    this.container.innerHTML = `
      <div class="layout-header">
        <div class="layout-title">
          <h2>Dashboard</h2>
          <span class="layout-info">${this.widgets.size} widgets</span>
        </div>
        <div class="layout-controls">
          <button class="btn-secondary btn-sm" data-action="add-widget">
            <svg class="w-4 h-4 mr-1">
              <path d="M12 6v6m0 0v6m0-6h6m-6 0H6"/>
            </svg>
            Add Widget
          </button>
          <button class="btn-secondary btn-sm" data-action="layout-menu">
            <svg class="w-4 h-4">
              <path d="M12 6v.01M12 12v.01M12 18v.01"/>
            </svg>
          </button>
        </div>
      </div>
      <div class="layout-grid" id="layout-grid">
        <!-- Widgets will be dynamically added here -->
      </div>
      <div class="layout-overlay" style="display: none;">
        <div class="drop-indicator"></div>
      </div>
    `;
    
    this.grid = this.container.querySelector('#layout-grid');
    this.overlay = this.container.querySelector('.layout-overlay');
    this.dropIndicator = this.container.querySelector('.drop-indicator');
  }
  
  calculateGrid() {
    const containerRect = this.container.getBoundingClientRect();
    this.gridWidth = containerRect.width - 40; // Account for padding
    this.columnWidth = this.gridWidth / this.options.maxColumns;
    
    // Update CSS custom properties
    this.container.style.setProperty('--grid-columns', this.options.maxColumns);
    this.container.style.setProperty('--column-width', `${this.columnWidth}px`);
    this.container.style.setProperty('--grid-size', `${this.options.gridSize}px`);
  }
  
  setupEventHandlers() {
    // Add widget button
    this.container.querySelector('[data-action="add-widget"]').addEventListener('click', () => {
      this.showAddWidgetDialog();
    });
    
    // Layout menu button
    this.container.querySelector('[data-action="layout-menu"]').addEventListener('click', () => {
      this.showLayoutMenu();
    });
    
    // Global mouse events for dragging
    document.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    document.addEventListener('mouseup', (e) => this.handleMouseUp(e));
    
    // Touch events for mobile
    document.addEventListener('touchmove', (e) => this.handleTouchMove(e), { passive: false });
    document.addEventListener('touchend', (e) => this.handleTouchEnd(e));
    
    // Prevent default drag behavior
    this.grid.addEventListener('dragstart', (e) => e.preventDefault());
  }
  
  setupResizeObserver() {
    if (!window.ResizeObserver) return;
    
    const resizeObserver = new ResizeObserver(() => {
      this.calculateGrid();
      this.repositionWidgets();
    });
    
    resizeObserver.observe(this.container);
  }
  
  addWidget(widgetConfig) {
    const widget = this.createWidget(widgetConfig);
    const position = this.findOptimalPosition(widgetConfig.width, widgetConfig.height);
    
    widget.layout = {
      x: position.x,
      y: position.y,
      width: widgetConfig.width || 4,
      height: widgetConfig.height || 3,
      id: widget.id
    };
    
    this.widgets.set(widget.id, widget);
    this.layout.push(widget.layout);
    
    this.renderWidget(widget);
    this.updateLayoutInfo();
    this.saveLayout();
    
    // Emit widget added event
    this.container.dispatchEvent(new CustomEvent('widgetAdded', {
      detail: { widget, layout: widget.layout }
    }));
    
    return widget;
  }
  
  createWidget(config) {
    const widget = {
      id: config.id || `widget-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type: config.type,
      title: config.title || 'Unnamed Widget',
      component: config.component,
      options: config.options || {},
      data: config.data || null,
      created: new Date(),
      element: null
    };
    
    return widget;
  }
  
  findOptimalPosition(width, height) {
    // Find the best position to place a widget
    const occupied = this.getOccupiedCells();
    
    for (let y = 0; y < 100; y++) { // Max 100 rows
      for (let x = 0; x <= this.options.maxColumns - width; x++) {
        if (this.isPositionAvailable(x, y, width, height, occupied)) {
          return { x, y };
        }
      }
    }
    
    // If no position found, place at bottom
    const maxY = Math.max(0, ...this.layout.map(item => item.y + item.height));
    return { x: 0, y: maxY };
  }
  
  getOccupiedCells() {
    const occupied = new Set();
    
    this.layout.forEach(item => {
      for (let x = item.x; x < item.x + item.width; x++) {
        for (let y = item.y; y < item.y + item.height; y++) {
          occupied.add(`${x},${y}`);
        }
      }
    });
    
    return occupied;
  }
  
  isPositionAvailable(x, y, width, height, occupied) {
    for (let dx = 0; dx < width; dx++) {
      for (let dy = 0; dy < height; dy++) {
        if (occupied.has(`${x + dx},${y + dy}`)) {
          return false;
        }
      }
    }
    return true;
  }
  
  renderWidget(widget) {
    const element = document.createElement('div');
    element.className = 'dashboard-widget';
    element.dataset.widgetId = widget.id;
    element.dataset.widgetType = widget.type;
    
    element.innerHTML = `
      <div class="widget-header">
        <div class="widget-title">${widget.title}</div>
        <div class="widget-controls">
          <button class="widget-btn" data-action="settings" title="Settings">
            <svg class="w-4 h-4"><path d="M12 15a3 3 0 100-6 3 3 0 000 6z"/></svg>
          </button>
          <button class="widget-btn" data-action="fullscreen" title="Fullscreen">
            <svg class="w-4 h-4"><path d="M8 3H5a2 2 0 00-2 2v3m18 0V5a2 2 0 00-2-2h-3m0 18h3a2 2 0 002-2v-3M3 16v3a2 2 0 002 2h3"/></svg>
          </button>
          <button class="widget-btn" data-action="remove" title="Remove">
            <svg class="w-4 h-4"><path d="M6 18L18 6M6 6l12 12"/></svg>
          </button>
        </div>
        <div class="widget-drag-handle" title="Drag to move">
          <svg class="w-4 h-4"><path d="M4 6h16M4 12h16M4 18h16"/></svg>
        </div>
      </div>
      <div class="widget-content" id="widget-content-${widget.id}">
        <div class="widget-loading">
          <div class="loading-spinner"></div>
          <span>Loading widget...</span>
        </div>
      </div>
      <div class="widget-resize-handle"></div>
    `;
    
    widget.element = element;
    this.grid.appendChild(element);
    
    // Position the widget
    this.positionWidget(widget);
    
    // Setup widget-specific event handlers
    this.setupWidgetEventHandlers(widget);
    
    // Load widget content
    this.loadWidgetContent(widget);
    
    return element;
  }
  
  positionWidget(widget) {
    const { x, y, width, height } = widget.layout;
    const element = widget.element;
    
    const pixelX = x * this.columnWidth;
    const pixelY = y * this.options.gridSize;
    const pixelWidth = width * this.columnWidth - 10; // Account for gaps
    const pixelHeight = height * this.options.gridSize - 10;
    
    element.style.transform = `translate(${pixelX}px, ${pixelY}px)`;
    element.style.width = `${pixelWidth}px`;
    element.style.height = `${pixelHeight}px`;
    element.style.zIndex = widget.layout.z || 1;
  }
  
  setupWidgetEventHandlers(widget) {
    const element = widget.element;
    
    // Drag handle
    const dragHandle = element.querySelector('.widget-drag-handle');
    dragHandle.addEventListener('mousedown', (e) => this.startDrag(e, widget));
    dragHandle.addEventListener('touchstart', (e) => this.startDrag(e, widget), { passive: false });
    
    // Resize handle
    const resizeHandle = element.querySelector('.widget-resize-handle');
    resizeHandle.addEventListener('mousedown', (e) => this.startResize(e, widget));
    resizeHandle.addEventListener('touchstart', (e) => this.startResize(e, widget), { passive: false });
    
    // Widget controls
    element.querySelector('[data-action="settings"]').addEventListener('click', () => {
      this.showWidgetSettings(widget);
    });
    
    element.querySelector('[data-action="fullscreen"]').addEventListener('click', () => {
      this.toggleWidgetFullscreen(widget);
    });
    
    element.querySelector('[data-action="remove"]').addEventListener('click', () => {
      this.removeWidget(widget.id);
    });
  }
  
  async loadWidgetContent(widget) {
    const contentContainer = widget.element.querySelector(`#widget-content-${widget.id}`);
    
    try {
      let content;
      
      switch (widget.type) {
        case 'anomaly-timeline':
          content = await this.createTimelineWidget(widget);
          break;
        case 'anomaly-heatmap':
          content = await this.createHeatmapWidget(widget);
          break;
        case 'metrics-summary':
          content = await this.createMetricsWidget(widget);
          break;
        case 'alert-list':
          content = await this.createAlertListWidget(widget);
          break;
        case 'dataset-info':
          content = await this.createDatasetInfoWidget(widget);
          break;
        case 'custom-chart':
          content = await this.createCustomChartWidget(widget);
          break;
        default:
          content = this.createDefaultWidget(widget);
      }
      
      contentContainer.innerHTML = '';
      contentContainer.appendChild(content);
      
      // Initialize widget component if needed
      if (widget.component && typeof widget.component.init === 'function') {
        widget.component.init(contentContainer, widget.options);
      }
      
    } catch (error) {
      console.error(`Failed to load widget ${widget.id}:`, error);
      contentContainer.innerHTML = `
        <div class="widget-error">
          <div class="error-icon">⚠️</div>
          <div class="error-message">Failed to load widget</div>
          <button class="btn-sm btn-secondary" onclick="this.closest('.dashboard-widget').dispatchEvent(new CustomEvent('reload'))">
            Retry
          </button>
        </div>
      `;
    }
  }
  
  async createTimelineWidget(widget) {
    const { createAnomalyTimeline } = await import('../charts/anomaly-timeline.js');
    
    const container = document.createElement('div');
    container.className = 'chart-widget-container';
    container.style.height = '100%';
    
    // Create timeline chart
    const chart = createAnomalyTimeline(container, {
      width: widget.element.clientWidth - 20,
      height: widget.element.clientHeight - 60,
      interactive: true,
      showLegend: false // Compact mode for widgets
    });
    
    // Load data
    const anomalies = dashboardState.getters.getFilteredAnomalies();
    chart.setData(anomalies);
    
    widget.component = chart;
    return container;
  }
  
  async createHeatmapWidget(widget) {
    const { createAnomalyHeatmap } = await import('../charts/anomaly-heatmap.js');
    
    const container = document.createElement('div');
    container.className = 'chart-widget-container';
    container.style.height = '100%';
    
    // Create heatmap chart
    const chart = createAnomalyHeatmap(container, {
      width: widget.element.clientWidth - 20,
      height: widget.element.clientHeight - 60,
      interactive: true,
      showLegend: false
    });
    
    // Generate sample heatmap data
    const heatmapData = this.generateSampleHeatmapData();
    chart.setData(heatmapData);
    
    widget.component = chart;
    return container;
  }
  
  createMetricsWidget(widget) {
    const container = document.createElement('div');
    container.className = 'metrics-widget';
    
    const metrics = dashboardState.getStateSlice('data.metrics');
    
    container.innerHTML = `
      <div class="metrics-grid">
        <div class="metric-item">
          <div class="metric-value">${metrics.totalDataPoints.toLocaleString()}</div>
          <div class="metric-label">Total Points</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${metrics.anomalyCount.toLocaleString()}</div>
          <div class="metric-label">Anomalies</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${(metrics.anomalyRate * 100).toFixed(1)}%</div>
          <div class="metric-label">Anomaly Rate</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${metrics.lastUpdate ? new Date(metrics.lastUpdate).toLocaleTimeString() : 'Never'}</div>
          <div class="metric-label">Last Update</div>
        </div>
      </div>
    `;
    
    return container;
  }
  
  createAlertListWidget(widget) {
    const container = document.createElement('div');
    container.className = 'alert-list-widget';
    
    const alerts = dashboardState.getStateSlice('data.alerts').slice(0, 5);
    
    if (alerts.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">✅</div>
          <div class="empty-message">No active alerts</div>
        </div>
      `;
    } else {
      container.innerHTML = `
        <div class="alert-list">
          ${alerts.map(alert => `
            <div class="alert-item">
              <div class="alert-icon">🚨</div>
              <div class="alert-content">
                <div class="alert-title">${alert.title}</div>
                <div class="alert-time">${new Date(alert.timestamp).toLocaleTimeString()}</div>
              </div>
            </div>
          `).join('')}
        </div>
      `;
    }
    
    return container;
  }
  
  createDatasetInfoWidget(widget) {
    const container = document.createElement('div');
    container.className = 'dataset-info-widget';
    
    const datasets = dashboardState.getStateSlice('data.datasets');
    const currentDataset = dashboardState.getStateSlice('data.currentDataset');
    
    container.innerHTML = `
      <div class="dataset-summary">
        <div class="summary-item">
          <span class="summary-label">Total Datasets:</span>
          <span class="summary-value">${datasets.length}</span>
        </div>
        <div class="summary-item">
          <span class="summary-label">Current Dataset:</span>
          <span class="summary-value">${currentDataset?.name || 'None selected'}</span>
        </div>
        <div class="summary-item">
          <span class="summary-label">Records:</span>
          <span class="summary-value">${currentDataset?.recordCount?.toLocaleString() || 'N/A'}</span>
        </div>
        <div class="summary-item">
          <span class="summary-label">Features:</span>
          <span class="summary-value">${currentDataset?.features?.length || 'N/A'}</span>
        </div>
      </div>
    `;
    
    return container;
  }
  
  createCustomChartWidget(widget) {
    const container = document.createElement('div');
    container.className = 'custom-chart-widget';
    
    // Placeholder for custom chart
    container.innerHTML = `
      <div class="chart-placeholder">
        <div class="placeholder-icon">📊</div>
        <div class="placeholder-text">Custom Chart</div>
        <div class="placeholder-subtitle">Configure in widget settings</div>
      </div>
    `;
    
    return container;
  }
  
  createDefaultWidget(widget) {
    const container = document.createElement('div');
    container.className = 'default-widget';
    
    container.innerHTML = `
      <div class="widget-placeholder">
        <div class="placeholder-icon">🧩</div>
        <div class="placeholder-text">Widget: ${widget.type}</div>
        <div class="placeholder-subtitle">Not implemented yet</div>
      </div>
    `;
    
    return container;
  }
  
  startDrag(event, widget) {
    event.preventDefault();
    event.stopPropagation();
    
    const clientX = event.clientX || event.touches[0].clientX;
    const clientY = event.clientY || event.touches[0].clientY;
    
    this.dragState = {
      widget,
      startX: clientX,
      startY: clientY,
      startLayout: { ...widget.layout },
      isDragging: false
    };
    
    widget.element.classList.add('dragging');
    this.overlay.style.display = 'block';
  }
  
  startResize(event, widget) {
    event.preventDefault();
    event.stopPropagation();
    
    const clientX = event.clientX || event.touches[0].clientX;
    const clientY = event.clientY || event.touches[0].clientY;
    
    this.resizeState = {
      widget,
      startX: clientX,
      startY: clientY,
      startLayout: { ...widget.layout },
      isResizing: false
    };
    
    widget.element.classList.add('resizing');
  }
  
  handleMouseMove(event) {
    if (this.dragState) {
      this.handleDragMove(event.clientX, event.clientY);
    } else if (this.resizeState) {
      this.handleResizeMove(event.clientX, event.clientY);
    }
  }
  
  handleTouchMove(event) {
    if (this.dragState || this.resizeState) {
      event.preventDefault();
      const touch = event.touches[0];
      if (this.dragState) {
        this.handleDragMove(touch.clientX, touch.clientY);
      } else if (this.resizeState) {
        this.handleResizeMove(touch.clientX, touch.clientY);
      }
    }
  }
  
  handleDragMove(clientX, clientY) {
    const { widget, startX, startY, startLayout } = this.dragState;
    
    if (!this.dragState.isDragging) {
      const distance = Math.sqrt(
        Math.pow(clientX - startX, 2) + Math.pow(clientY - startY, 2)
      );
      if (distance > 5) {
        this.dragState.isDragging = true;
      } else {
        return;
      }
    }
    
    const deltaX = clientX - startX;
    const deltaY = clientY - startY;
    
    const newX = Math.max(0, Math.min(
      this.options.maxColumns - widget.layout.width,
      Math.round((startLayout.x * this.columnWidth + deltaX) / this.columnWidth)
    ));
    
    const newY = Math.max(0, Math.round(
      (startLayout.y * this.options.gridSize + deltaY) / this.options.gridSize
    ));
    
    // Check for collisions
    if (this.isPositionAvailable(newX, newY, widget.layout.width, widget.layout.height, 
        this.getOccupiedCells().filter(cell => !this.isInWidget(cell, widget.layout)))) {
      
      widget.layout.x = newX;
      widget.layout.y = newY;
      this.positionWidget(widget);
      this.showDropIndicator(newX, newY, widget.layout.width, widget.layout.height);
    }
  }
  
  handleResizeMove(clientX, clientY) {
    const { widget, startX, startY, startLayout } = this.resizeState;
    
    if (!this.resizeState.isResizing) {
      const distance = Math.sqrt(
        Math.pow(clientX - startX, 2) + Math.pow(clientY - startY, 2)
      );
      if (distance > 5) {
        this.resizeState.isResizing = true;
      } else {
        return;
      }
    }
    
    const deltaX = clientX - startX;
    const deltaY = clientY - startY;
    
    const newWidth = Math.max(2, Math.min(
      this.options.maxColumns - widget.layout.x,
      Math.round((startLayout.width * this.columnWidth + deltaX) / this.columnWidth)
    ));
    
    const newHeight = Math.max(2, Math.round(
      (startLayout.height * this.options.gridSize + deltaY) / this.options.gridSize
    ));
    
    widget.layout.width = newWidth;
    widget.layout.height = newHeight;
    this.positionWidget(widget);
    
    // Resize widget content if needed
    if (widget.component && typeof widget.component.resize === 'function') {
      setTimeout(() => widget.component.resize(), 50);
    }
  }
  
  handleMouseUp(event) {
    this.handleDragEnd();
    this.handleResizeEnd();
  }
  
  handleTouchEnd(event) {
    this.handleDragEnd();
    this.handleResizeEnd();
  }
  
  handleDragEnd() {
    if (!this.dragState) return;
    
    const { widget } = this.dragState;
    
    widget.element.classList.remove('dragging');
    this.overlay.style.display = 'none';
    this.hideDropIndicator();
    
    // Update layout in state
    const layoutIndex = this.layout.findIndex(item => item.id === widget.id);
    if (layoutIndex !== -1) {
      this.layout[layoutIndex] = { ...widget.layout };
    }
    
    this.saveLayout();
    this.compactLayout();
    
    // Emit widget moved event
    this.container.dispatchEvent(new CustomEvent('widgetMoved', {
      detail: { widget, layout: widget.layout }
    }));
    
    this.dragState = null;
  }
  
  handleResizeEnd() {
    if (!this.resizeState) return;
    
    const { widget } = this.resizeState;
    
    widget.element.classList.remove('resizing');
    
    // Update layout in state
    const layoutIndex = this.layout.findIndex(item => item.id === widget.id);
    if (layoutIndex !== -1) {
      this.layout[layoutIndex] = { ...widget.layout };
    }
    
    this.saveLayout();
    
    // Emit widget resized event
    this.container.dispatchEvent(new CustomEvent('widgetResized', {
      detail: { widget, layout: widget.layout }
    }));
    
    this.resizeState = null;
  }
  
  showDropIndicator(x, y, width, height) {
    const pixelX = x * this.columnWidth;
    const pixelY = y * this.options.gridSize;
    const pixelWidth = width * this.columnWidth - 10;
    const pixelHeight = height * this.options.gridSize - 10;
    
    this.dropIndicator.style.transform = `translate(${pixelX}px, ${pixelY}px)`;
    this.dropIndicator.style.width = `${pixelWidth}px`;
    this.dropIndicator.style.height = `${pixelHeight}px`;
    this.dropIndicator.style.display = 'block';
  }
  
  hideDropIndicator() {
    this.dropIndicator.style.display = 'none';
  }
  
  isInWidget(cell, layout) {
    const [x, y] = cell.split(',').map(Number);
    return x >= layout.x && x < layout.x + layout.width &&
           y >= layout.y && y < layout.y + layout.height;
  }
  
  compactLayout() {
    // Move widgets up to fill gaps
    let changed = true;
    while (changed) {
      changed = false;
      
      for (const widget of this.widgets.values()) {
        const layout = widget.layout;
        const newY = layout.y - 1;
        
        if (newY >= 0) {
          const occupied = this.getOccupiedCells();
          if (this.isPositionAvailable(layout.x, newY, layout.width, layout.height,
              occupied.filter(cell => !this.isInWidget(cell, layout)))) {
            
            layout.y = newY;
            this.positionWidget(widget);
            changed = true;
          }
        }
      }
    }
    
    if (changed) {
      this.saveLayout();
    }
  }
  
  removeWidget(widgetId) {
    const widget = this.widgets.get(widgetId);
    if (!widget) return;
    
    // Remove from DOM
    widget.element.remove();
    
    // Cleanup widget component
    if (widget.component && typeof widget.component.destroy === 'function') {
      widget.component.destroy();
    }
    
    // Remove from state
    this.widgets.delete(widgetId);
    this.layout = this.layout.filter(item => item.id !== widgetId);
    
    this.updateLayoutInfo();
    this.saveLayout();
    this.compactLayout();
    
    // Emit widget removed event
    this.container.dispatchEvent(new CustomEvent('widgetRemoved', {
      detail: { widgetId }
    }));
  }
  
  showAddWidgetDialog() {
    const dialog = document.createElement('div');
    dialog.className = 'modal-overlay';
    dialog.innerHTML = `
      <div class="modal-dialog">
        <div class="modal-header">
          <h3>Add Widget</h3>
          <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">×</button>
        </div>
        <div class="modal-body">
          <div class="widget-gallery">
            <div class="widget-option" data-type="anomaly-timeline">
              <div class="widget-preview">📈</div>
              <div class="widget-info">
                <div class="widget-name">Timeline Chart</div>
                <div class="widget-description">Anomaly detection over time</div>
              </div>
            </div>
            <div class="widget-option" data-type="anomaly-heatmap">
              <div class="widget-preview">🔥</div>
              <div class="widget-info">
                <div class="widget-name">Heatmap</div>
                <div class="widget-description">Feature anomaly heatmap</div>
              </div>
            </div>
            <div class="widget-option" data-type="metrics-summary">
              <div class="widget-preview">📊</div>
              <div class="widget-info">
                <div class="widget-name">Metrics Summary</div>
                <div class="widget-description">Key performance metrics</div>
              </div>
            </div>
            <div class="widget-option" data-type="alert-list">
              <div class="widget-preview">🚨</div>
              <div class="widget-info">
                <div class="widget-name">Alert List</div>
                <div class="widget-description">Recent alerts and notifications</div>
              </div>
            </div>
            <div class="widget-option" data-type="dataset-info">
              <div class="widget-preview">📋</div>
              <div class="widget-info">
                <div class="widget-name">Dataset Info</div>
                <div class="widget-description">Current dataset information</div>
              </div>
            </div>
            <div class="widget-option" data-type="custom-chart">
              <div class="widget-preview">🎨</div>
              <div class="widget-info">
                <div class="widget-name">Custom Chart</div>
                <div class="widget-description">Configurable visualization</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
    
    document.body.appendChild(dialog);
    
    // Handle widget selection
    dialog.querySelectorAll('.widget-option').forEach(option => {
      option.addEventListener('click', () => {
        const widgetType = option.dataset.type;
        this.addWidget({
          type: widgetType,
          title: option.querySelector('.widget-name').textContent,
          width: this.getDefaultWidgetSize(widgetType).width,
          height: this.getDefaultWidgetSize(widgetType).height
        });
        dialog.remove();
      });
    });
  }
  
  getDefaultWidgetSize(widgetType) {
    const sizes = {
      'anomaly-timeline': { width: 6, height: 4 },
      'anomaly-heatmap': { width: 6, height: 5 },
      'metrics-summary': { width: 4, height: 3 },
      'alert-list': { width: 4, height: 4 },
      'dataset-info': { width: 3, height: 3 },
      'custom-chart': { width: 4, height: 4 }
    };
    
    return sizes[widgetType] || { width: 4, height: 3 };
  }
  
  showLayoutMenu() {
    // Implementation for layout menu (save/load layouts, etc.)
    console.log('Show layout menu');
  }
  
  showWidgetSettings(widget) {
    // Implementation for widget settings dialog
    console.log('Show widget settings for:', widget.id);
  }
  
  toggleWidgetFullscreen(widget) {
    widget.element.classList.toggle('widget-fullscreen');
    
    if (widget.component && typeof widget.component.resize === 'function') {
      setTimeout(() => widget.component.resize(), 300);
    }
  }
  
  updateLayoutInfo() {
    const layoutInfo = this.container.querySelector('.layout-info');
    if (layoutInfo) {
      layoutInfo.textContent = `${this.widgets.size} widgets`;
    }
  }
  
  repositionWidgets() {
    this.widgets.forEach(widget => {
      this.positionWidget(widget);
      if (widget.component && typeof widget.component.resize === 'function') {
        widget.component.resize();
      }
    });
  }
  
  saveLayout() {
    if (!this.options.persistLayout) return;
    
    const layoutData = {
      layout: this.layout,
      widgets: Array.from(this.widgets.values()).map(widget => ({
        id: widget.id,
        type: widget.type,
        title: widget.title,
        options: widget.options
      })),
      timestamp: new Date().toISOString()
    };
    
    localStorage.setItem('dashboard-layout', JSON.stringify(layoutData));
  }
  
  loadSavedLayout() {
    if (!this.options.persistLayout) return;
    
    try {
      const savedData = localStorage.getItem('dashboard-layout');
      if (savedData) {
        const layoutData = JSON.parse(savedData);
        
        // Restore widgets
        layoutData.widgets.forEach(widgetData => {
          const widget = this.createWidget(widgetData);
          const layout = layoutData.layout.find(l => l.id === widget.id);
          
          if (layout) {
            widget.layout = layout;
            this.widgets.set(widget.id, widget);
            this.renderWidget(widget);
          }
        });
        
        this.layout = layoutData.layout;
        this.updateLayoutInfo();
      }
    } catch (error) {
      console.warn('Failed to load saved layout:', error);
    }
  }
  
  generateSampleHeatmapData() {
    const features = ['CPU Usage', 'Memory', 'Disk I/O', 'Network', 'Temperature'];
    const timeSlots = Array.from({ length: 24 }, (_, i) => `${i.toString().padStart(2, '0')}:00`);
    const data = [];
    
    features.forEach(feature => {
      timeSlots.forEach(time => {
        data.push({
          x: time,
          y: feature,
          value: Math.random() * 100,
          anomalyScore: Math.random()
        });
      });
    });
    
    return data;
  }
  
  clearLayout() {
    this.widgets.forEach(widget => this.removeWidget(widget.id));
  }
  
  exportLayout() {
    return {
      layout: this.layout,
      widgets: Array.from(this.widgets.values())
    };
  }
  
  importLayout(layoutData) {
    this.clearLayout();
    
    layoutData.widgets.forEach(widgetData => {
      const widget = this.createWidget(widgetData);
      const layout = layoutData.layout.find(l => l.id === widget.id);
      
      if (layout) {
        widget.layout = layout;
        this.widgets.set(widget.id, widget);
        this.renderWidget(widget);
      }
    });
    
    this.layout = layoutData.layout;
    this.updateLayoutInfo();
  }
  
  destroy() {
    this.widgets.forEach(widget => {
      if (widget.component && typeof widget.component.destroy === 'function') {
        widget.component.destroy();
      }
    });
    
    this.widgets.clear();
    this.layout = [];
  }
}

// Factory function
export function createDashboardLayout(container, options = {}) {
  return new DashboardLayoutEngine(container, options);
}

// Auto-initialize dashboard layouts
export function initializeDashboardLayouts() {
  document.querySelectorAll('[data-component="dashboard-layout"]').forEach(container => {
    new DashboardLayoutEngine(container);
  });
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeDashboardLayouts);
} else {
  initializeDashboardLayouts();
}
