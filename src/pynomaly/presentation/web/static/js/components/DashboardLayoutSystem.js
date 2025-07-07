/**
 * Dashboard Layout System for Pynomaly
 *
 * Features:
 * - Drag-and-drop grid layout with resizable widgets
 * - Responsive breakpoint handling
 * - Widget library with anomaly detection components
 * - Layout persistence and state management
 * - Accessibility-first interactions
 * - Real-time layout synchronization
 */

class DashboardLayoutSystem {
  constructor(container, options = {}) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;

    this.options = {
      columns: 12,
      rowHeight: 100,
      margin: [10, 10],
      containerPadding: [10, 10],
      breakpoints: {
        lg: 1200,
        md: 996,
        sm: 768,
        xs: 480,
        xxs: 0,
      },
      cols: {
        lg: 12,
        md: 10,
        sm: 6,
        xs: 4,
        xxs: 2,
      },
      compactType: "vertical",
      preventCollision: false,
      isDraggable: true,
      isResizable: true,
      useCSSTransforms: true,
      transformScale: 1,
      autoSize: true,
      verticalCompact: true,
      maxRows: Infinity,
      ...options,
    };

    this.state = {
      layouts: {},
      currentBreakpoint: "lg",
      widgets: new Map(),
      isDragging: false,
      isResizing: false,
      draggedElement: null,
      mousePosition: { x: 0, y: 0 },
      gridSize: { width: 0, height: 0 },
    };

    this.eventHandlers = new Map();
    this.widgetRegistry = new Map();
    this.animationFrameId = null;

    this.init();
  }

  init() {
    this.setupContainer();
    this.registerBuiltinWidgets();
    this.bindEvents();
    this.handleResize();
    this.initializeLayout();
  }

  setupContainer() {
    this.container.className = `dashboard-layout ${this.container.className || ""}`;
    this.container.setAttribute("role", "main");
    this.container.setAttribute(
      "aria-label",
      "Dashboard layout with draggable widgets",
    );

    this.container.innerHTML = `
      <div class="dashboard-header">
        <div class="dashboard-controls">
          <button class="btn btn--secondary" id="add-widget-btn" aria-label="Add new widget">
            <span aria-hidden="true">+</span> Add Widget
          </button>
          <button class="btn btn--secondary" id="layout-settings-btn" aria-label="Layout settings">
            <span aria-hidden="true">⚙️</span> Settings
          </button>
          <button class="btn btn--secondary" id="save-layout-btn" aria-label="Save current layout">
            <span aria-hidden="true">💾</span> Save
          </button>
          <button class="btn btn--secondary" id="reset-layout-btn" aria-label="Reset to default layout">
            <span aria-hidden="true">🔄</span> Reset
          </button>
        </div>
        <div class="dashboard-status">
          <span class="status-indicator" id="layout-status">Ready</span>
        </div>
      </div>
      <div class="dashboard-grid" id="dashboard-grid" role="grid"></div>
      <div class="widget-palette" id="widget-palette" style="display: none;">
        <div class="palette-header">
          <h3>Available Widgets</h3>
          <button class="btn btn--sm" id="close-palette-btn" aria-label="Close widget palette">×</button>
        </div>
        <div class="palette-content" id="palette-content"></div>
      </div>
    `;

    this.gridContainer = this.container.querySelector("#dashboard-grid");
    this.widgetPalette = this.container.querySelector("#widget-palette");
    this.statusIndicator = this.container.querySelector("#layout-status");
  }

  registerBuiltinWidgets() {
    // Anomaly Detection Widgets
    this.registerWidget("anomaly-chart", {
      name: "Anomaly Chart",
      description: "Real-time anomaly detection visualization",
      defaultSize: { w: 6, h: 4 },
      minSize: { w: 4, h: 3 },
      maxSize: { w: 12, h: 8 },
      category: "analytics",
      icon: "📊",
      render: this.renderAnomalyChart.bind(this),
    });

    this.registerWidget("time-series", {
      name: "Time Series Plot",
      description: "Interactive time series data visualization",
      defaultSize: { w: 8, h: 4 },
      minSize: { w: 6, h: 3 },
      maxSize: { w: 12, h: 6 },
      category: "analytics",
      icon: "📈",
      render: this.renderTimeSeriesPlot.bind(this),
    });

    this.registerWidget("metrics-summary", {
      name: "Metrics Summary",
      description: "Key performance indicators and statistics",
      defaultSize: { w: 4, h: 3 },
      minSize: { w: 3, h: 2 },
      maxSize: { w: 6, h: 4 },
      category: "metrics",
      icon: "📋",
      render: this.renderMetricsSummary.bind(this),
    });

    this.registerWidget("alert-feed", {
      name: "Alert Feed",
      description: "Real-time alerts and notifications",
      defaultSize: { w: 4, h: 5 },
      minSize: { w: 3, h: 4 },
      maxSize: { w: 6, h: 8 },
      category: "monitoring",
      icon: "🚨",
      render: this.renderAlertFeed.bind(this),
    });

    this.registerWidget("data-quality", {
      name: "Data Quality",
      description: "Data quality metrics and validation results",
      defaultSize: { w: 6, h: 3 },
      minSize: { w: 4, h: 2 },
      maxSize: { w: 8, h: 4 },
      category: "quality",
      icon: "✅",
      render: this.renderDataQuality.bind(this),
    });

    this.registerWidget("model-performance", {
      name: "Model Performance",
      description: "ML model performance metrics and trends",
      defaultSize: { w: 6, h: 4 },
      minSize: { w: 4, h: 3 },
      maxSize: { w: 8, h: 5 },
      category: "ml",
      icon: "🧠",
      render: this.renderModelPerformance.bind(this),
    });
  }

  registerWidget(type, config) {
    this.widgetRegistry.set(type, {
      type,
      ...config,
      id: Math.random().toString(36).substr(2, 9),
    });
  }

  bindEvents() {
    // Control buttons
    this.container
      .querySelector("#add-widget-btn")
      .addEventListener("click", () => {
        this.showWidgetPalette();
      });

    this.container
      .querySelector("#layout-settings-btn")
      .addEventListener("click", () => {
        this.showLayoutSettings();
      });

    this.container
      .querySelector("#save-layout-btn")
      .addEventListener("click", () => {
        this.saveLayout();
      });

    this.container
      .querySelector("#reset-layout-btn")
      .addEventListener("click", () => {
        this.resetLayout();
      });

    this.container
      .querySelector("#close-palette-btn")
      .addEventListener("click", () => {
        this.hideWidgetPalette();
      });

    // Window resize
    window.addEventListener(
      "resize",
      this.debounce(() => {
        this.handleResize();
      }, 250),
    );

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case "s":
            e.preventDefault();
            this.saveLayout();
            break;
          case "z":
            e.preventDefault();
            if (e.shiftKey) {
              this.redo();
            } else {
              this.undo();
            }
            break;
        }
      }
    });

    // Grid events
    this.gridContainer.addEventListener(
      "mousedown",
      this.handleMouseDown.bind(this),
    );
    this.gridContainer.addEventListener(
      "mousemove",
      this.handleMouseMove.bind(this),
    );
    this.gridContainer.addEventListener(
      "mouseup",
      this.handleMouseUp.bind(this),
    );
    this.gridContainer.addEventListener(
      "touchstart",
      this.handleTouchStart.bind(this),
    );
    this.gridContainer.addEventListener(
      "touchmove",
      this.handleTouchMove.bind(this),
    );
    this.gridContainer.addEventListener(
      "touchend",
      this.handleTouchEnd.bind(this),
    );
  }

  handleResize() {
    const containerWidth = this.container.offsetWidth;
    const newBreakpoint = this.getBreakpointFromWidth(containerWidth);

    if (newBreakpoint !== this.state.currentBreakpoint) {
      this.state.currentBreakpoint = newBreakpoint;
      this.onBreakpointChange(newBreakpoint);
    }

    this.calculateGridDimensions();
    this.updateLayout();
  }

  getBreakpointFromWidth(width) {
    const breakpoints = Object.entries(this.options.breakpoints).sort(
      ([, a], [, b]) => b - a,
    );

    for (const [breakpoint, minWidth] of breakpoints) {
      if (width >= minWidth) {
        return breakpoint;
      }
    }

    return "xxs";
  }

  calculateGridDimensions() {
    const containerWidth = this.gridContainer.offsetWidth;
    const cols = this.options.cols[this.state.currentBreakpoint];
    const margin = this.options.margin[0];
    const containerPadding = this.options.containerPadding[0];

    this.state.gridSize.width =
      (containerWidth - containerPadding * 2 - (cols - 1) * margin) / cols;
    this.state.gridSize.height = this.options.rowHeight;
  }

  initializeLayout() {
    // Load saved layout or create default
    const savedLayouts = this.loadSavedLayouts();

    if (savedLayouts && Object.keys(savedLayouts).length > 0) {
      this.state.layouts = savedLayouts;
    } else {
      this.createDefaultLayout();
    }

    this.renderLayout();
  }

  createDefaultLayout() {
    this.state.layouts = {
      lg: [
        { i: "anomaly-chart-1", x: 0, y: 0, w: 6, h: 4, type: "anomaly-chart" },
        { i: "time-series-1", x: 6, y: 0, w: 6, h: 4, type: "time-series" },
        { i: "metrics-1", x: 0, y: 4, w: 4, h: 3, type: "metrics-summary" },
        { i: "alerts-1", x: 4, y: 4, w: 4, h: 3, type: "alert-feed" },
        { i: "quality-1", x: 8, y: 4, w: 4, h: 3, type: "data-quality" },
      ],
    };

    // Generate responsive layouts
    this.generateResponsiveLayouts();
  }

  generateResponsiveLayouts() {
    const baseLayout = this.state.layouts.lg;

    // Medium devices
    this.state.layouts.md = baseLayout.map((item) => ({
      ...item,
      w: Math.min(item.w, this.options.cols.md),
      x: Math.min(item.x, this.options.cols.md - item.w),
    }));

    // Small devices
    this.state.layouts.sm = baseLayout.map((item) => ({
      ...item,
      w: Math.min(item.w * 2, this.options.cols.sm),
      x: 0,
    }));

    // Extra small devices
    this.state.layouts.xs = baseLayout.map((item, index) => ({
      ...item,
      w: this.options.cols.xs,
      x: 0,
      y: index * item.h,
    }));

    this.state.layouts.xxs = baseLayout.map((item, index) => ({
      ...item,
      w: this.options.cols.xxs,
      x: 0,
      y: index * item.h,
    }));
  }

  renderLayout() {
    const currentLayout =
      this.state.layouts[this.state.currentBreakpoint] || [];
    this.gridContainer.innerHTML = "";

    currentLayout.forEach((item) => {
      this.renderWidget(item);
    });

    this.compactLayout();
    this.updateStatusIndicator();
  }

  renderWidget(layoutItem) {
    const widgetConfig = this.widgetRegistry.get(layoutItem.type);
    if (!widgetConfig) {
      console.warn(`Widget type "${layoutItem.type}" not found`);
      return;
    }

    const widget = document.createElement("div");
    widget.className = "dashboard-widget";
    widget.setAttribute("data-widget-id", layoutItem.i);
    widget.setAttribute("data-widget-type", layoutItem.type);
    widget.setAttribute("role", "region");
    widget.setAttribute("aria-label", `${widgetConfig.name} widget`);
    widget.tabIndex = 0;

    const position = this.calculatePosition(layoutItem);
    widget.style.cssText = `
      position: absolute;
      left: ${position.left}px;
      top: ${position.top}px;
      width: ${position.width}px;
      height: ${position.height}px;
      z-index: 1;
      transition: all 0.2s ease;
    `;

    widget.innerHTML = `
      <div class="widget-header">
        <div class="widget-title">
          <span class="widget-icon" aria-hidden="true">${widgetConfig.icon}</span>
          <span class="widget-name">${widgetConfig.name}</span>
        </div>
        <div class="widget-controls">
          <button class="widget-btn" aria-label="Configure ${widgetConfig.name}" data-action="configure">
            ⚙️
          </button>
          <button class="widget-btn" aria-label="Remove ${widgetConfig.name}" data-action="remove">
            ×
          </button>
        </div>
      </div>
      <div class="widget-content" id="widget-content-${layoutItem.i}">
        <div class="widget-loading">Loading...</div>
      </div>
      <div class="widget-resize-handle" aria-label="Resize ${widgetConfig.name}" role="button" tabindex="0"></div>
    `;

    // Bind widget events
    this.bindWidgetEvents(widget, layoutItem);

    this.gridContainer.appendChild(widget);
    this.state.widgets.set(layoutItem.i, {
      element: widget,
      config: widgetConfig,
      layout: layoutItem,
    });

    // Render widget content
    this.renderWidgetContent(layoutItem);
  }

  renderWidgetContent(layoutItem) {
    const widgetConfig = this.widgetRegistry.get(layoutItem.type);
    const contentContainer = document.getElementById(
      `widget-content-${layoutItem.i}`,
    );

    if (widgetConfig && widgetConfig.render && contentContainer) {
      // Simulate async loading
      setTimeout(() => {
        contentContainer.innerHTML = "";
        widgetConfig.render(contentContainer, layoutItem);
      }, 100);
    }
  }

  calculatePosition(layoutItem) {
    const { w, h, x, y } = layoutItem;
    const margin = this.options.margin;
    const containerPadding = this.options.containerPadding;

    return {
      left: containerPadding[0] + x * (this.state.gridSize.width + margin[0]),
      top: containerPadding[1] + y * (this.state.gridSize.height + margin[1]),
      width: w * this.state.gridSize.width + (w - 1) * margin[0],
      height: h * this.state.gridSize.height + (h - 1) * margin[1],
    };
  }

  bindWidgetEvents(widget, layoutItem) {
    const header = widget.querySelector(".widget-header");
    const resizeHandle = widget.querySelector(".widget-resize-handle");
    const controls = widget.querySelectorAll(".widget-btn");

    // Drag functionality
    if (this.options.isDraggable) {
      header.style.cursor = "move";
      header.addEventListener("mousedown", (e) =>
        this.startDrag(e, widget, layoutItem),
      );
      header.addEventListener("touchstart", (e) =>
        this.startDrag(e, widget, layoutItem),
      );
    }

    // Resize functionality
    if (this.options.isResizable) {
      resizeHandle.addEventListener("mousedown", (e) =>
        this.startResize(e, widget, layoutItem),
      );
      resizeHandle.addEventListener("touchstart", (e) =>
        this.startResize(e, widget, layoutItem),
      );
    }

    // Control buttons
    controls.forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const action = btn.dataset.action;

        switch (action) {
          case "configure":
            this.configureWidget(layoutItem.i);
            break;
          case "remove":
            this.removeWidget(layoutItem.i);
            break;
        }
      });
    });

    // Keyboard navigation
    widget.addEventListener("keydown", (e) => {
      this.handleWidgetKeyboard(e, widget, layoutItem);
    });
  }

  startDrag(e, widget, layoutItem) {
    if (!this.options.isDraggable) return;

    e.preventDefault();
    this.state.isDragging = true;
    this.state.draggedElement = { widget, layoutItem };

    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    this.state.mousePosition = {
      x: clientX - widget.offsetLeft,
      y: clientY - widget.offsetTop,
    };

    widget.style.zIndex = "1000";
    widget.classList.add("dragging");

    document.addEventListener("mousemove", this.onDrag.bind(this));
    document.addEventListener("mouseup", this.onDragEnd.bind(this));
    document.addEventListener("touchmove", this.onDrag.bind(this));
    document.addEventListener("touchend", this.onDragEnd.bind(this));

    this.announceToUser(
      `Started dragging ${this.widgetRegistry.get(layoutItem.type).name} widget`,
    );
  }

  onDrag(e) {
    if (!this.state.isDragging || !this.state.draggedElement) return;

    e.preventDefault();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    const { widget } = this.state.draggedElement;

    const newLeft = clientX - this.state.mousePosition.x;
    const newTop = clientY - this.state.mousePosition.y;

    widget.style.left = `${newLeft}px`;
    widget.style.top = `${newTop}px`;

    // Update layout position
    this.updateDraggedItemPosition(newLeft, newTop);
  }

  onDragEnd(e) {
    if (!this.state.isDragging) return;

    const { widget, layoutItem } = this.state.draggedElement;

    widget.style.zIndex = "1";
    widget.classList.remove("dragging");

    // Snap to grid
    this.snapToGrid(layoutItem);
    this.compactLayout();
    this.updateLayout();

    this.state.isDragging = false;
    this.state.draggedElement = null;

    document.removeEventListener("mousemove", this.onDrag);
    document.removeEventListener("mouseup", this.onDragEnd);
    document.removeEventListener("touchmove", this.onDrag);
    document.removeEventListener("touchend", this.onDragEnd);

    this.announceToUser(
      `Moved ${this.widgetRegistry.get(layoutItem.type).name} widget`,
    );
    this.onLayoutChange();
  }

  updateDraggedItemPosition(left, top) {
    const { layoutItem } = this.state.draggedElement;
    const containerPadding = this.options.containerPadding;
    const margin = this.options.margin;

    // Convert pixel position to grid position
    const x = Math.round(
      (left - containerPadding[0]) / (this.state.gridSize.width + margin[0]),
    );
    const y = Math.round(
      (top - containerPadding[1]) / (this.state.gridSize.height + margin[1]),
    );

    layoutItem.x = Math.max(
      0,
      Math.min(
        x,
        this.options.cols[this.state.currentBreakpoint] - layoutItem.w,
      ),
    );
    layoutItem.y = Math.max(0, y);
  }

  snapToGrid(layoutItem) {
    const position = this.calculatePosition(layoutItem);
    const widget = this.state.widgets.get(layoutItem.i).element;

    widget.style.left = `${position.left}px`;
    widget.style.top = `${position.top}px`;
  }

  startResize(e, widget, layoutItem) {
    if (!this.options.isResizable) return;

    e.preventDefault();
    e.stopPropagation();

    this.state.isResizing = true;
    this.state.draggedElement = { widget, layoutItem };

    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    this.state.mousePosition = {
      x: clientX,
      y: clientY,
      startWidth: layoutItem.w,
      startHeight: layoutItem.h,
    };

    widget.classList.add("resizing");

    document.addEventListener("mousemove", this.onResize.bind(this));
    document.addEventListener("mouseup", this.onResizeEnd.bind(this));
    document.addEventListener("touchmove", this.onResize.bind(this));
    document.addEventListener("touchend", this.onResizeEnd.bind(this));

    this.announceToUser(
      `Started resizing ${this.widgetRegistry.get(layoutItem.type).name} widget`,
    );
  }

  onResize(e) {
    if (!this.state.isResizing || !this.state.draggedElement) return;

    e.preventDefault();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    const deltaX = clientX - this.state.mousePosition.x;
    const deltaY = clientY - this.state.mousePosition.y;

    const { layoutItem } = this.state.draggedElement;
    const widgetConfig = this.widgetRegistry.get(layoutItem.type);

    // Calculate new size in grid units
    const gridDeltaX = Math.round(
      deltaX / (this.state.gridSize.width + this.options.margin[0]),
    );
    const gridDeltaY = Math.round(
      deltaY / (this.state.gridSize.height + this.options.margin[1]),
    );

    const newWidth = Math.max(
      widgetConfig.minSize.w,
      Math.min(
        widgetConfig.maxSize.w,
        this.state.mousePosition.startWidth + gridDeltaX,
      ),
    );

    const newHeight = Math.max(
      widgetConfig.minSize.h,
      Math.min(
        widgetConfig.maxSize.h,
        this.state.mousePosition.startHeight + gridDeltaY,
      ),
    );

    layoutItem.w = newWidth;
    layoutItem.h = newHeight;

    this.updateWidgetSize(layoutItem);
  }

  onResizeEnd(e) {
    if (!this.state.isResizing) return;

    const { widget, layoutItem } = this.state.draggedElement;

    widget.classList.remove("resizing");

    this.compactLayout();
    this.updateLayout();

    this.state.isResizing = false;
    this.state.draggedElement = null;

    document.removeEventListener("mousemove", this.onResize);
    document.removeEventListener("mouseup", this.onResizeEnd);
    document.removeEventListener("touchmove", this.onResize);
    document.removeEventListener("touchend", this.onResizeEnd);

    this.announceToUser(
      `Resized ${this.widgetRegistry.get(layoutItem.type).name} widget`,
    );
    this.onLayoutChange();
  }

  updateWidgetSize(layoutItem) {
    const position = this.calculatePosition(layoutItem);
    const widget = this.state.widgets.get(layoutItem.i).element;

    widget.style.width = `${position.width}px`;
    widget.style.height = `${position.height}px`;
  }

  compactLayout() {
    if (!this.options.verticalCompact) return;

    const currentLayout = this.state.layouts[this.state.currentBreakpoint];
    const sorted = [...currentLayout].sort((a, b) => {
      if (a.y < b.y) return -1;
      if (a.y > b.y) return 1;
      return a.x - b.x;
    });

    for (const item of sorted) {
      item.y = this.getMinY(item, sorted);
    }
  }

  getMinY(item, layout) {
    let minY = 0;

    for (const otherItem of layout) {
      if (otherItem === item) continue;

      if (this.collides(item, otherItem)) {
        minY = Math.max(minY, otherItem.y + otherItem.h);
      }
    }

    return minY;
  }

  collides(item1, item2) {
    return !(
      item1.x >= item2.x + item2.w ||
      item1.x + item1.w <= item2.x ||
      item1.y >= item2.y + item2.h ||
      item1.y + item1.h <= item2.y
    );
  }

  updateLayout() {
    const currentLayout = this.state.layouts[this.state.currentBreakpoint];

    currentLayout.forEach((item) => {
      const widget = this.state.widgets.get(item.i);
      if (widget) {
        const position = this.calculatePosition(item);
        const element = widget.element;

        element.style.left = `${position.left}px`;
        element.style.top = `${position.top}px`;
        element.style.width = `${position.width}px`;
        element.style.height = `${position.height}px`;
      }
    });
  }

  addWidget(type, customLayout = {}) {
    const widgetConfig = this.widgetRegistry.get(type);
    if (!widgetConfig) {
      console.error(`Widget type "${type}" not found`);
      return;
    }

    const widgetId = `${type}-${Date.now()}`;
    const layout = this.state.layouts[this.state.currentBreakpoint];

    // Find available position
    const position = this.findAvailablePosition(widgetConfig.defaultSize);

    const newItem = {
      i: widgetId,
      type,
      ...widgetConfig.defaultSize,
      ...position,
      ...customLayout,
    };

    // Add to all breakpoints
    Object.keys(this.state.layouts).forEach((breakpoint) => {
      if (!this.state.layouts[breakpoint]) {
        this.state.layouts[breakpoint] = [];
      }
      this.state.layouts[breakpoint].push({ ...newItem });
    });

    this.renderWidget(newItem);
    this.compactLayout();
    this.updateLayout();
    this.onLayoutChange();

    this.announceToUser(`Added ${widgetConfig.name} widget to dashboard`);
    return widgetId;
  }

  findAvailablePosition(size) {
    const layout = this.state.layouts[this.state.currentBreakpoint];
    const cols = this.options.cols[this.state.currentBreakpoint];

    // Try to place at the bottom
    let maxY = 0;
    layout.forEach((item) => {
      maxY = Math.max(maxY, item.y + item.h);
    });

    // Check if we can fit at the bottom
    for (let x = 0; x <= cols - size.w; x++) {
      const testItem = { x, y: maxY, w: size.w, h: size.h };
      let collision = false;

      for (const item of layout) {
        if (this.collides(testItem, item)) {
          collision = true;
          break;
        }
      }

      if (!collision) {
        return { x, y: maxY };
      }
    }

    // If we can't fit at the bottom, place at the very bottom
    return { x: 0, y: maxY };
  }

  removeWidget(widgetId) {
    const widget = this.state.widgets.get(widgetId);
    if (!widget) return;

    const widgetName = widget.config.name;

    // Remove from all layouts
    Object.keys(this.state.layouts).forEach((breakpoint) => {
      this.state.layouts[breakpoint] = this.state.layouts[breakpoint].filter(
        (item) => item.i !== widgetId,
      );
    });

    // Remove from DOM
    widget.element.remove();
    this.state.widgets.delete(widgetId);

    this.compactLayout();
    this.updateLayout();
    this.onLayoutChange();

    this.announceToUser(`Removed ${widgetName} widget from dashboard`);
  }

  // Widget Rendering Methods
  renderAnomalyChart(container, layoutItem) {
    container.innerHTML = `
      <div class="chart-container">
        <div class="chart-header">
          <h4>Anomaly Detection</h4>
          <span class="chart-status">Live</span>
        </div>
        <div class="chart-body">
          <canvas id="anomaly-chart-${layoutItem.i}" width="100%" height="200"></canvas>
        </div>
        <div class="chart-footer">
          <span class="metric">Anomalies: <strong>23</strong></span>
          <span class="metric">Accuracy: <strong>94.2%</strong></span>
        </div>
      </div>
    `;
  }

  renderTimeSeriesPlot(container, layoutItem) {
    container.innerHTML = `
      <div class="chart-container">
        <div class="chart-header">
          <h4>Time Series Analysis</h4>
          <select class="chart-control">
            <option>Last 24h</option>
            <option>Last 7d</option>
            <option>Last 30d</option>
          </select>
        </div>
        <div class="chart-body">
          <div class="time-series-chart" style="height: 200px; background: linear-gradient(45deg, #f0f0f0 25%, transparent 25%), linear-gradient(-45deg, #f0f0f0 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #f0f0f0 75%), linear-gradient(-45deg, transparent 75%, #f0f0f0 75%); background-size: 20px 20px; background-position: 0 0, 0 10px, 10px -10px, -10px 0px; display: flex; align-items: center; justify-content: center; color: #666;">
            📈 Time Series Visualization
          </div>
        </div>
      </div>
    `;
  }

  renderMetricsSummary(container, layoutItem) {
    container.innerHTML = `
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-value">1,245</div>
          <div class="metric-label">Total Records</div>
          <div class="metric-change positive">+12%</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">23</div>
          <div class="metric-label">Anomalies</div>
          <div class="metric-change negative">-5%</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">94.2%</div>
          <div class="metric-label">Accuracy</div>
          <div class="metric-change positive">+1.2%</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">0.15ms</div>
          <div class="metric-label">Avg Response</div>
          <div class="metric-change neutral">±0%</div>
        </div>
      </div>
    `;
  }

  renderAlertFeed(container, layoutItem) {
    container.innerHTML = `
      <div class="alert-feed">
        <div class="alert-item severity-high">
          <div class="alert-icon">🚨</div>
          <div class="alert-content">
            <div class="alert-title">High CPU Anomaly Detected</div>
            <div class="alert-time">2 minutes ago</div>
          </div>
        </div>
        <div class="alert-item severity-medium">
          <div class="alert-icon">⚠️</div>
          <div class="alert-content">
            <div class="alert-title">Memory Usage Spike</div>
            <div class="alert-time">15 minutes ago</div>
          </div>
        </div>
        <div class="alert-item severity-low">
          <div class="alert-icon">ℹ️</div>
          <div class="alert-content">
            <div class="alert-title">Model Retrained Successfully</div>
            <div class="alert-time">1 hour ago</div>
          </div>
        </div>
        <div class="alert-item severity-medium">
          <div class="alert-icon">🔄</div>
          <div class="alert-content">
            <div class="alert-title">Data Source Reconnected</div>
            <div class="alert-time">2 hours ago</div>
          </div>
        </div>
      </div>
    `;
  }

  renderDataQuality(container, layoutItem) {
    container.innerHTML = `
      <div class="quality-dashboard">
        <div class="quality-score">
          <div class="score-circle">
            <div class="score-value">87%</div>
            <div class="score-label">Overall Quality</div>
          </div>
        </div>
        <div class="quality-metrics">
          <div class="quality-item">
            <span class="quality-name">Completeness</span>
            <span class="quality-value">92%</span>
          </div>
          <div class="quality-item">
            <span class="quality-name">Accuracy</span>
            <span class="quality-value">89%</span>
          </div>
          <div class="quality-item">
            <span class="quality-name">Consistency</span>
            <span class="quality-value">85%</span>
          </div>
          <div class="quality-item">
            <span class="quality-name">Validity</span>
            <span class="quality-value">94%</span>
          </div>
        </div>
      </div>
    `;
  }

  renderModelPerformance(container, layoutItem) {
    container.innerHTML = `
      <div class="model-performance">
        <div class="performance-header">
          <h4>Model Metrics</h4>
          <select class="model-selector">
            <option>Isolation Forest</option>
            <option>LSTM Autoencoder</option>
            <option>Ensemble Model</option>
          </select>
        </div>
        <div class="performance-metrics">
          <div class="perf-metric">
            <span class="perf-label">Precision</span>
            <span class="perf-value">0.94</span>
            <div class="perf-bar">
              <div class="perf-fill" style="width: 94%"></div>
            </div>
          </div>
          <div class="perf-metric">
            <span class="perf-label">Recall</span>
            <span class="perf-value">0.87</span>
            <div class="perf-bar">
              <div class="perf-fill" style="width: 87%"></div>
            </div>
          </div>
          <div class="perf-metric">
            <span class="perf-label">F1-Score</span>
            <span class="perf-value">0.91</span>
            <div class="perf-bar">
              <div class="perf-fill" style="width: 91%"></div>
            </div>
          </div>
          <div class="perf-metric">
            <span class="perf-label">AUC-ROC</span>
            <span class="perf-value">0.96</span>
            <div class="perf-bar">
              <div class="perf-fill" style="width: 96%"></div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // Utility Methods
  showWidgetPalette() {
    this.widgetPalette.style.display = "block";
    this.renderWidgetPalette();

    // Focus management
    this.widgetPalette.querySelector("#close-palette-btn").focus();
  }

  hideWidgetPalette() {
    this.widgetPalette.style.display = "none";
  }

  renderWidgetPalette() {
    const paletteContent = this.container.querySelector("#palette-content");
    const categories = {};

    // Group widgets by category
    this.widgetRegistry.forEach((widget) => {
      if (!categories[widget.category]) {
        categories[widget.category] = [];
      }
      categories[widget.category].push(widget);
    });

    paletteContent.innerHTML = Object.entries(categories)
      .map(
        ([category, widgets]) => `
      <div class="widget-category">
        <h4 class="category-title">${category.charAt(0).toUpperCase() + category.slice(1)}</h4>
        <div class="widget-grid">
          ${widgets
            .map(
              (widget) => `
            <button class="widget-card" data-widget-type="${widget.type}" tabindex="0">
              <div class="widget-card-icon">${widget.icon}</div>
              <div class="widget-card-name">${widget.name}</div>
              <div class="widget-card-desc">${widget.description}</div>
            </button>
          `,
            )
            .join("")}
        </div>
      </div>
    `,
      )
      .join("");

    // Bind widget card events
    paletteContent.querySelectorAll(".widget-card").forEach((card) => {
      card.addEventListener("click", () => {
        const widgetType = card.dataset.widgetType;
        this.addWidget(widgetType);
        this.hideWidgetPalette();
      });
    });
  }

  configureWidget(widgetId) {
    const widget = this.state.widgets.get(widgetId);
    if (!widget) return;

    // Placeholder for widget configuration modal
    this.announceToUser(
      `Configuration for ${widget.config.name} widget (feature coming soon)`,
    );
  }

  saveLayout() {
    try {
      localStorage.setItem(
        "pynomaly-dashboard-layout",
        JSON.stringify(this.state.layouts),
      );
      this.updateStatusIndicator("Layout saved successfully", "success");
      this.announceToUser("Dashboard layout saved successfully");
    } catch (error) {
      console.error("Failed to save layout:", error);
      this.updateStatusIndicator("Failed to save layout", "error");
    }
  }

  loadSavedLayouts() {
    try {
      const saved = localStorage.getItem("pynomaly-dashboard-layout");
      return saved ? JSON.parse(saved) : null;
    } catch (error) {
      console.error("Failed to load saved layout:", error);
      return null;
    }
  }

  resetLayout() {
    if (
      confirm(
        "Are you sure you want to reset the dashboard to the default layout? This will remove all customizations.",
      )
    ) {
      this.createDefaultLayout();
      this.renderLayout();
      this.updateStatusIndicator("Layout reset to default", "success");
      this.announceToUser("Dashboard layout reset to default");
      this.onLayoutChange();
    }
  }

  updateStatusIndicator(message = "Ready", type = "normal") {
    this.statusIndicator.textContent = message;
    this.statusIndicator.className = `status-indicator status-${type}`;

    if (type !== "normal") {
      setTimeout(() => {
        this.updateStatusIndicator();
      }, 3000);
    }
  }

  onLayoutChange() {
    if (this.options.onLayoutChange) {
      this.options.onLayoutChange(
        this.state.layouts,
        this.state.currentBreakpoint,
      );
    }
  }

  onBreakpointChange(newBreakpoint) {
    if (this.options.onBreakpointChange) {
      this.options.onBreakpointChange(newBreakpoint, this.state.layouts);
    }

    this.renderLayout();
  }

  handleWidgetKeyboard(e, widget, layoutItem) {
    const step = e.shiftKey ? 5 : 1;

    switch (e.key) {
      case "ArrowLeft":
        e.preventDefault();
        layoutItem.x = Math.max(0, layoutItem.x - step);
        this.snapToGrid(layoutItem);
        this.compactLayout();
        this.updateLayout();
        break;
      case "ArrowRight":
        e.preventDefault();
        layoutItem.x = Math.min(
          this.options.cols[this.state.currentBreakpoint] - layoutItem.w,
          layoutItem.x + step,
        );
        this.snapToGrid(layoutItem);
        this.compactLayout();
        this.updateLayout();
        break;
      case "ArrowUp":
        e.preventDefault();
        layoutItem.y = Math.max(0, layoutItem.y - step);
        this.snapToGrid(layoutItem);
        this.compactLayout();
        this.updateLayout();
        break;
      case "ArrowDown":
        e.preventDefault();
        layoutItem.y += step;
        this.snapToGrid(layoutItem);
        this.compactLayout();
        this.updateLayout();
        break;
      case "Delete":
      case "Backspace":
        e.preventDefault();
        this.removeWidget(layoutItem.i);
        break;
    }
  }

  handleMouseDown(e) {
    this.handlePointerDown(e, "mouse");
  }

  handleMouseMove(e) {
    this.handlePointerMove(e, "mouse");
  }

  handleMouseUp(e) {
    this.handlePointerUp(e, "mouse");
  }

  handleTouchStart(e) {
    this.handlePointerDown(e, "touch");
  }

  handleTouchMove(e) {
    this.handlePointerMove(e, "touch");
  }

  handleTouchEnd(e) {
    this.handlePointerUp(e, "touch");
  }

  handlePointerDown(e, type) {
    // Delegate to specific drag/resize handlers
  }

  handlePointerMove(e, type) {
    // Handle global pointer movement
  }

  handlePointerUp(e, type) {
    // Handle global pointer release
  }

  announceToUser(message) {
    if (
      this.options.accessibility &&
      this.options.accessibility.announceActions
    ) {
      // Create or update live region for screen readers
      let liveRegion = document.getElementById("dashboard-live-region");
      if (!liveRegion) {
        liveRegion = document.createElement("div");
        liveRegion.id = "dashboard-live-region";
        liveRegion.setAttribute("aria-live", "polite");
        liveRegion.setAttribute("aria-atomic", "true");
        liveRegion.className = "sr-only";
        document.body.appendChild(liveRegion);
      }

      liveRegion.textContent = message;
    }
  }

  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  destroy() {
    // Clean up event listeners
    window.removeEventListener("resize", this.handleResize);
    document.removeEventListener("keydown", this.handleKeydown);

    // Clear widgets
    this.state.widgets.clear();

    // Clear container
    this.container.innerHTML = "";

    // Cancel any pending animation frames
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
  }
}

// Export for module systems
if (typeof module !== "undefined" && module.exports) {
  module.exports = { DashboardLayoutSystem };
}

// Global access
window.DashboardLayoutSystem = DashboardLayoutSystem;
