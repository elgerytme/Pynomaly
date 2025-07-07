/**
 * Drag-and-Drop Dashboard Layout System
 *
 * Features:
 * - Responsive grid layout with drag-and-drop reordering
 * - Dynamic widget resizing and positioning
 * - Customizable dashboard configurations
 * - Real-time layout persistence
 * - Accessibility support with keyboard navigation
 * - Touch-friendly mobile interactions
 * - Undo/redo functionality
 * - Layout templates and presets
 */

class DragDropDashboard {
  constructor(container, options = {}) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;

    this.options = {
      columns: 12,
      rowHeight: 120,
      margin: 16,
      minItemWidth: 2,
      minItemHeight: 2,
      maxItemWidth: 12,
      maxItemHeight: 8,
      enableResize: true,
      enableDrag: true,
      enableKeyboard: true,
      saveLayout: true,
      layoutKey: "dashboard-layout",
      widgets: [],
      breakpoints: {
        lg: 1200,
        md: 996,
        sm: 768,
        xs: 480,
      },
      ...options,
    };

    this.state = {
      layout: [],
      widgets: new Map(),
      draggedItem: null,
      resizedItem: null,
      isEditing: false,
      currentBreakpoint: "lg",
      history: [],
      historyIndex: -1,
      focusedWidget: null,
    };

    this.eventHandlers = new Map();
    this.observers = new Set();

    this.init();
  }

  init() {
    this.setupContainer();
    this.calculateDimensions();
    this.createLayout();
    this.bindEvents();
    this.loadLayout();
    this.updateBreakpoint();
    this.setupAccessibility();
  }

  setupContainer() {
    this.container.className = "drag-drop-dashboard";
    this.container.setAttribute("role", "application");
    this.container.setAttribute(
      "aria-label",
      "Customizable dashboard with drag and drop widgets",
    );

    // Create dashboard wrapper
    this.container.innerHTML = `
      <div class="dashboard-header">
        <div class="dashboard-controls">
          <button class="btn btn--secondary btn--sm toggle-edit-btn" aria-pressed="false">
            <span class="edit-icon">✏️</span>
            <span class="edit-text">Edit Layout</span>
          </button>
          <div class="layout-controls" style="display: none;">
            <button class="btn btn--outline btn--sm undo-btn" disabled title="Undo last change">
              ↶ Undo
            </button>
            <button class="btn btn--outline btn--sm redo-btn" disabled title="Redo last change">
              ↷ Redo
            </button>
            <button class="btn btn--outline btn--sm reset-btn" title="Reset to default layout">
              🔄 Reset
            </button>
            <button class="btn btn--outline btn--sm save-template-btn" title="Save as template">
              💾 Save Template
            </button>
          </div>
        </div>
        <div class="dashboard-info">
          <span class="current-breakpoint">Desktop Layout</span>
        </div>
      </div>
      <div class="dashboard-grid" role="grid">
        <div class="grid-background"></div>
        <div class="drop-zones"></div>
        <div class="widgets-container"></div>
      </div>
      <div class="widget-palette" style="display: none;">
        <h3>Available Widgets</h3>
        <div class="palette-widgets"></div>
      </div>
    `;

    this.dashboardGrid = this.container.querySelector(".dashboard-grid");
    this.widgetsContainer = this.container.querySelector(".widgets-container");
    this.gridBackground = this.container.querySelector(".grid-background");
    this.dropZones = this.container.querySelector(".drop-zones");
    this.widgetPalette = this.container.querySelector(".widget-palette");
    this.toggleEditBtn = this.container.querySelector(".toggle-edit-btn");
    this.layoutControls = this.container.querySelector(".layout-controls");
  }

  calculateDimensions() {
    const containerRect = this.container.getBoundingClientRect();
    this.containerWidth = containerRect.width;
    this.containerHeight = containerRect.height;

    this.columnWidth =
      (this.containerWidth - this.options.margin * (this.options.columns + 1)) /
      this.options.columns;

    // Update CSS custom properties
    this.container.style.setProperty(
      "--dashboard-columns",
      this.options.columns,
    );
    this.container.style.setProperty(
      "--dashboard-column-width",
      `${this.columnWidth}px`,
    );
    this.container.style.setProperty(
      "--dashboard-row-height",
      `${this.options.rowHeight}px`,
    );
    this.container.style.setProperty(
      "--dashboard-margin",
      `${this.options.margin}px`,
    );
  }

  createLayout() {
    this.renderGridBackground();
    this.renderDropZones();
    this.renderWidgets();
    this.updateLayoutInfo();
  }

  renderGridBackground() {
    // Create visual grid lines for better positioning feedback
    this.gridBackground.innerHTML = "";

    // Vertical lines
    for (let i = 0; i <= this.options.columns; i++) {
      const line = document.createElement("div");
      line.className = "grid-line grid-line--vertical";
      line.style.left = `${i * (this.columnWidth + this.options.margin)}px`;
      this.gridBackground.appendChild(line);
    }

    // Horizontal lines (dynamic based on content)
    const maxRows = Math.max(6, this.getMaxRow() + 2);
    for (let i = 0; i <= maxRows; i++) {
      const line = document.createElement("div");
      line.className = "grid-line grid-line--horizontal";
      line.style.top = `${i * (this.options.rowHeight + this.options.margin)}px`;
      this.gridBackground.appendChild(line);
    }
  }

  renderDropZones() {
    if (!this.state.isEditing) {
      this.dropZones.style.display = "none";
      return;
    }

    this.dropZones.style.display = "block";
    this.dropZones.innerHTML = "";

    const maxRows = Math.max(6, this.getMaxRow() + 2);

    for (let row = 0; row < maxRows; row++) {
      for (let col = 0; col < this.options.columns; col++) {
        if (!this.isPositionOccupied(col, row)) {
          const dropZone = document.createElement("div");
          dropZone.className = "drop-zone";
          dropZone.dataset.col = col;
          dropZone.dataset.row = row;
          dropZone.style.cssText = `
            left: ${col * (this.columnWidth + this.options.margin) + this.options.margin}px;
            top: ${row * (this.options.rowHeight + this.options.margin) + this.options.margin}px;
            width: ${this.columnWidth}px;
            height: ${this.options.rowHeight}px;
          `;
          this.dropZones.appendChild(dropZone);
        }
      }
    }
  }

  renderWidgets() {
    this.widgetsContainer.innerHTML = "";

    for (const widget of this.state.layout) {
      this.renderWidget(widget);
    }
  }

  renderWidget(widget) {
    const widgetElement = document.createElement("div");
    widgetElement.className = `dashboard-widget ${widget.type}`;
    widgetElement.dataset.id = widget.id;
    widgetElement.setAttribute("role", "gridcell");
    widgetElement.setAttribute("aria-label", `${widget.title} widget`);
    widgetElement.setAttribute("tabindex", "0");

    if (this.state.isEditing) {
      widgetElement.classList.add("editable");
      widgetElement.setAttribute("draggable", "true");
    }

    // Position and size the widget
    this.positionWidget(widgetElement, widget);

    // Create widget content
    widgetElement.innerHTML = `
      ${
        this.state.isEditing
          ? `
        <div class="widget-controls">
          <button class="widget-control widget-control--move" 
                  aria-label="Move ${widget.title}"
                  title="Drag to move">
            ⋮⋮
          </button>
          ${
            this.options.enableResize
              ? `
            <button class="widget-control widget-control--resize" 
                    aria-label="Resize ${widget.title}"
                    title="Drag to resize">
              ⤡
            </button>
          `
              : ""
          }
          <button class="widget-control widget-control--remove" 
                  aria-label="Remove ${widget.title}"
                  title="Remove widget">
            ×
          </button>
        </div>
      `
          : ""
      }
      <div class="widget-header">
        <h3 class="widget-title">${widget.title}</h3>
        ${widget.subtitle ? `<p class="widget-subtitle">${widget.subtitle}</p>` : ""}
      </div>
      <div class="widget-content">
        ${this.renderWidgetContent(widget)}
      </div>
      ${
        this.state.isEditing
          ? `
        <div class="resize-handle" aria-hidden="true"></div>
      `
          : ""
      }
    `;

    this.widgetsContainer.appendChild(widgetElement);
    this.bindWidgetEvents(widgetElement, widget);

    return widgetElement;
  }

  renderWidgetContent(widget) {
    switch (widget.type) {
      case "chart":
        return `
          <div class="chart-placeholder" data-chart-type="${widget.chartType || "line"}">
            <div class="chart-icon">📊</div>
            <div class="chart-label">${widget.chartType || "Line"} Chart</div>
          </div>
        `;

      case "metric":
        return `
          <div class="metric-display">
            <div class="metric-value">${widget.value || "---"}</div>
            <div class="metric-label">${widget.metric || "Metric"}</div>
            <div class="metric-change ${widget.change >= 0 ? "positive" : "negative"}">
              ${widget.change >= 0 ? "↗" : "↘"} ${Math.abs(widget.change || 0)}%
            </div>
          </div>
        `;

      case "table":
        return `
          <div class="table-placeholder">
            <div class="table-icon">📋</div>
            <div class="table-label">Data Table</div>
            <div class="table-info">${widget.rows || 0} rows</div>
          </div>
        `;

      case "text":
        return `
          <div class="text-content">
            ${widget.content || "Text widget content"}
          </div>
        `;

      case "iframe":
        return `
          <iframe src="${widget.src || "about:blank"}" 
                  frameborder="0" 
                  style="width: 100%; height: 100%;"
                  title="${widget.title}">
          </iframe>
        `;

      default:
        return `
          <div class="widget-placeholder">
            <div class="placeholder-icon">🔧</div>
            <div class="placeholder-label">Custom Widget</div>
          </div>
        `;
    }
  }

  positionWidget(element, widget) {
    const left =
      widget.x * (this.columnWidth + this.options.margin) + this.options.margin;
    const top =
      widget.y * (this.options.rowHeight + this.options.margin) +
      this.options.margin;
    const width =
      widget.w * this.columnWidth + (widget.w - 1) * this.options.margin;
    const height =
      widget.h * this.options.rowHeight + (widget.h - 1) * this.options.margin;

    element.style.cssText = `
      position: absolute;
      left: ${left}px;
      top: ${top}px;
      width: ${width}px;
      height: ${height}px;
      z-index: ${widget.z || 1};
    `;
  }

  bindEvents() {
    // Edit mode toggle
    this.toggleEditBtn.addEventListener("click", () => {
      this.toggleEditMode();
    });

    // Layout controls
    this.container.querySelector(".undo-btn").addEventListener("click", () => {
      this.undo();
    });

    this.container.querySelector(".redo-btn").addEventListener("click", () => {
      this.redo();
    });

    this.container.querySelector(".reset-btn").addEventListener("click", () => {
      this.resetLayout();
    });

    this.container
      .querySelector(".save-template-btn")
      .addEventListener("click", () => {
        this.saveTemplate();
      });

    // Global drag and drop events
    this.container.addEventListener("dragover", this.handleDragOver.bind(this));
    this.container.addEventListener("drop", this.handleDrop.bind(this));

    // Keyboard navigation
    this.container.addEventListener("keydown", this.handleKeyDown.bind(this));

    // Resize observer
    if (window.ResizeObserver) {
      this.resizeObserver = new ResizeObserver(() => {
        this.handleResize();
      });
      this.resizeObserver.observe(this.container);
    }

    // Window resize fallback
    window.addEventListener(
      "resize",
      this.debounce(() => {
        this.handleResize();
      }, 250),
    );
  }

  bindWidgetEvents(element, widget) {
    // Drag start
    element.addEventListener("dragstart", (e) => {
      if (!this.state.isEditing) return;
      this.handleDragStart(e, widget);
    });

    // Drag end
    element.addEventListener("dragend", (e) => {
      this.handleDragEnd(e, widget);
    });

    // Widget controls
    const removeBtn = element.querySelector(".widget-control--remove");
    if (removeBtn) {
      removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.removeWidget(widget.id);
      });
    }

    // Resize handle
    const resizeHandle = element.querySelector(".resize-handle");
    if (resizeHandle) {
      this.bindResizeEvents(resizeHandle, element, widget);
    }

    // Focus and keyboard events
    element.addEventListener("focus", () => {
      this.state.focusedWidget = widget.id;
      this.updateWidgetFocus();
    });

    element.addEventListener("blur", () => {
      if (this.state.focusedWidget === widget.id) {
        this.state.focusedWidget = null;
        this.updateWidgetFocus();
      }
    });

    // Touch events for mobile
    this.bindTouchEvents(element, widget);
  }

  bindResizeEvents(handle, element, widget) {
    let isResizing = false;
    let startX, startY, startWidth, startHeight;

    const startResize = (e) => {
      if (!this.state.isEditing) return;

      isResizing = true;
      this.state.resizedItem = widget;
      element.classList.add("resizing");

      const rect = element.getBoundingClientRect();
      startX = e.clientX || (e.touches && e.touches[0].clientX);
      startY = e.clientY || (e.touches && e.touches[0].clientY);
      startWidth = rect.width;
      startHeight = rect.height;

      e.preventDefault();
    };

    const doResize = (e) => {
      if (!isResizing) return;

      const currentX = e.clientX || (e.touches && e.touches[0].clientX);
      const currentY = e.clientY || (e.touches && e.touches[0].clientY);

      const deltaX = currentX - startX;
      const deltaY = currentY - startY;

      const newWidth = Math.max(
        this.options.minItemWidth * this.columnWidth,
        Math.min(
          this.options.maxItemWidth * this.columnWidth,
          startWidth + deltaX,
        ),
      );

      const newHeight = Math.max(
        this.options.minItemHeight * this.options.rowHeight,
        Math.min(
          this.options.maxItemHeight * this.options.rowHeight,
          startHeight + deltaY,
        ),
      );

      // Convert to grid units
      const newW = Math.round(newWidth / this.columnWidth);
      const newH = Math.round(newHeight / this.options.rowHeight);

      // Update widget dimensions
      if (this.canResizeWidget(widget, newW, newH)) {
        widget.w = newW;
        widget.h = newH;
        this.positionWidget(element, widget);
        this.renderDropZones();
      }
    };

    const endResize = () => {
      if (!isResizing) return;

      isResizing = false;
      this.state.resizedItem = null;
      element.classList.remove("resizing");

      this.saveToHistory();
      this.saveLayout();
      this.announceChange(`Resized ${widget.title} to ${widget.w}x${widget.h}`);
    };

    // Mouse events
    handle.addEventListener("mousedown", startResize);
    document.addEventListener("mousemove", doResize);
    document.addEventListener("mouseup", endResize);

    // Touch events
    handle.addEventListener("touchstart", startResize, { passive: false });
    document.addEventListener("touchmove", doResize, { passive: false });
    document.addEventListener("touchend", endResize);
  }

  bindTouchEvents(element, widget) {
    let touchStartTime = 0;
    let longPressTimer = null;
    let isDragging = false;

    element.addEventListener("touchstart", (e) => {
      if (!this.state.isEditing) return;

      touchStartTime = Date.now();

      // Long press to start drag
      longPressTimer = setTimeout(() => {
        isDragging = true;
        element.classList.add("dragging");
        this.state.draggedItem = widget;
      }, 500);
    });

    element.addEventListener("touchmove", (e) => {
      if (!isDragging) {
        clearTimeout(longPressTimer);
        return;
      }

      e.preventDefault();
      const touch = e.touches[0];
      const elementUnderTouch = document.elementFromPoint(
        touch.clientX,
        touch.clientY,
      );

      // Highlight drop zones
      this.highlightDropZone(elementUnderTouch);
    });

    element.addEventListener("touchend", (e) => {
      clearTimeout(longPressTimer);

      if (isDragging) {
        const touch = e.changedTouches[0];
        const elementUnderTouch = document.elementFromPoint(
          touch.clientX,
          touch.clientY,
        );

        this.handleTouchDrop(elementUnderTouch, widget);

        isDragging = false;
        element.classList.remove("dragging");
        this.state.draggedItem = null;
      }
    });
  }

  handleDragStart(e, widget) {
    this.state.draggedItem = widget;
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", widget.id);

    // Create custom drag image
    const dragImage = e.target.cloneNode(true);
    dragImage.style.transform = "rotate(5deg)";
    dragImage.style.opacity = "0.8";
    document.body.appendChild(dragImage);
    e.dataTransfer.setDragImage(dragImage, e.offsetX, e.offsetY);

    setTimeout(() => {
      document.body.removeChild(dragImage);
    }, 0);

    this.container.classList.add("dragging");
    e.target.classList.add("dragging");

    this.announceChange(`Started moving ${widget.title}`);
  }

  handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";

    const dropZone = e.target.closest(".drop-zone");
    if (dropZone) {
      this.highlightDropZone(dropZone);
    }
  }

  handleDrop(e) {
    e.preventDefault();

    const dropZone = e.target.closest(".drop-zone");
    if (dropZone && this.state.draggedItem) {
      const col = parseInt(dropZone.dataset.col);
      const row = parseInt(dropZone.dataset.row);

      this.moveWidget(this.state.draggedItem.id, col, row);
    }

    this.clearHighlights();
  }

  handleDragEnd(e, widget) {
    this.container.classList.remove("dragging");
    e.target.classList.remove("dragging");
    this.state.draggedItem = null;
    this.clearHighlights();
  }

  handleTouchDrop(element, widget) {
    const dropZone = element?.closest(".drop-zone");
    if (dropZone) {
      const col = parseInt(dropZone.dataset.col);
      const row = parseInt(dropZone.dataset.row);
      this.moveWidget(widget.id, col, row);
    }
    this.clearHighlights();
  }

  highlightDropZone(element) {
    // Clear previous highlights
    this.container
      .querySelectorAll(".drop-zone.highlighted")
      .forEach((zone) => {
        zone.classList.remove("highlighted");
      });

    // Highlight current drop zone
    const dropZone = element?.closest(".drop-zone");
    if (dropZone) {
      dropZone.classList.add("highlighted");
    }
  }

  clearHighlights() {
    this.container
      .querySelectorAll(".drop-zone.highlighted")
      .forEach((zone) => {
        zone.classList.remove("highlighted");
      });
  }

  handleKeyDown(e) {
    if (!this.options.enableKeyboard || !this.state.focusedWidget) return;

    const widget = this.state.layout.find(
      (w) => w.id === this.state.focusedWidget,
    );
    if (!widget) return;

    let moved = false;
    const step = e.shiftKey ? 5 : 1; // Shift for larger steps

    switch (e.key) {
      case "ArrowLeft":
        if (widget.x > 0) {
          widget.x = Math.max(0, widget.x - step);
          moved = true;
        }
        break;
      case "ArrowRight":
        if (widget.x + widget.w < this.options.columns) {
          widget.x = Math.min(this.options.columns - widget.w, widget.x + step);
          moved = true;
        }
        break;
      case "ArrowUp":
        if (widget.y > 0) {
          widget.y = Math.max(0, widget.y - step);
          moved = true;
        }
        break;
      case "ArrowDown":
        widget.y += step;
        moved = true;
        break;
      case "Delete":
      case "Backspace":
        if (this.state.isEditing) {
          e.preventDefault();
          this.removeWidget(widget.id);
        }
        break;
      case "+":
      case "=":
        if (this.state.isEditing && e.ctrlKey) {
          e.preventDefault();
          this.resizeWidget(widget.id, widget.w + 1, widget.h + 1);
        }
        break;
      case "-":
        if (this.state.isEditing && e.ctrlKey) {
          e.preventDefault();
          this.resizeWidget(widget.id, widget.w - 1, widget.h - 1);
        }
        break;
    }

    if (moved) {
      e.preventDefault();
      this.saveToHistory();
      this.renderWidgets();
      this.renderDropZones();
      this.saveLayout();
      this.announceChange(
        `Moved ${widget.title} to position ${widget.x}, ${widget.y}`,
      );
    }
  }

  handleResize() {
    this.calculateDimensions();
    this.updateBreakpoint();
    this.renderWidgets();
    this.renderGridBackground();
    this.renderDropZones();
  }

  // Widget management methods
  addWidget(widgetConfig) {
    const widget = {
      id: widgetConfig.id || `widget-${Date.now()}`,
      type: widgetConfig.type || "text",
      title: widgetConfig.title || "New Widget",
      x: widgetConfig.x || 0,
      y: widgetConfig.y || 0,
      w: widgetConfig.w || 2,
      h: widgetConfig.h || 2,
      z: widgetConfig.z || 1,
      ...widgetConfig,
    };

    // Find available position if not specified
    if (!widgetConfig.x && !widgetConfig.y) {
      const position = this.findAvailablePosition(widget.w, widget.h);
      widget.x = position.x;
      widget.y = position.y;
    }

    this.state.layout.push(widget);
    this.saveToHistory();
    this.renderWidgets();
    this.renderDropZones();
    this.saveLayout();

    this.announceChange(`Added ${widget.title} widget`);
    return widget;
  }

  removeWidget(widgetId) {
    const widgetIndex = this.state.layout.findIndex((w) => w.id === widgetId);
    if (widgetIndex === -1) return;

    const widget = this.state.layout[widgetIndex];
    this.state.layout.splice(widgetIndex, 1);

    this.saveToHistory();
    this.renderWidgets();
    this.renderDropZones();
    this.saveLayout();

    this.announceChange(`Removed ${widget.title} widget`);
  }

  moveWidget(widgetId, newX, newY) {
    const widget = this.state.layout.find((w) => w.id === widgetId);
    if (!widget) return;

    // Check if new position is valid
    if (this.canMoveWidget(widget, newX, newY)) {
      widget.x = newX;
      widget.y = newY;

      this.saveToHistory();
      this.renderWidgets();
      this.renderDropZones();
      this.saveLayout();

      this.announceChange(`Moved ${widget.title} to position ${newX}, ${newY}`);
    }
  }

  resizeWidget(widgetId, newW, newH) {
    const widget = this.state.layout.find((w) => w.id === widgetId);
    if (!widget) return;

    // Validate new dimensions
    newW = Math.max(
      this.options.minItemWidth,
      Math.min(this.options.maxItemWidth, newW),
    );
    newH = Math.max(
      this.options.minItemHeight,
      Math.min(this.options.maxItemHeight, newH),
    );

    if (this.canResizeWidget(widget, newW, newH)) {
      widget.w = newW;
      widget.h = newH;

      this.saveToHistory();
      this.renderWidgets();
      this.renderDropZones();
      this.saveLayout();

      this.announceChange(`Resized ${widget.title} to ${newW}x${newH}`);
    }
  }

  // Validation methods
  canMoveWidget(widget, newX, newY) {
    // Check bounds
    if (newX < 0 || newY < 0 || newX + widget.w > this.options.columns) {
      return false;
    }

    // Check for collisions with other widgets
    return !this.state.layout.some((other) => {
      if (other.id === widget.id) return false;

      return !(
        newX >= other.x + other.w ||
        newX + widget.w <= other.x ||
        newY >= other.y + other.h ||
        newY + widget.h <= other.y
      );
    });
  }

  canResizeWidget(widget, newW, newH) {
    // Check bounds
    if (widget.x + newW > this.options.columns) return false;

    // Check for collisions
    return !this.state.layout.some((other) => {
      if (other.id === widget.id) return false;

      return !(
        widget.x >= other.x + other.w ||
        widget.x + newW <= other.x ||
        widget.y >= other.y + other.h ||
        widget.y + newH <= other.y
      );
    });
  }

  isPositionOccupied(x, y) {
    return this.state.layout.some((widget) => {
      return (
        x >= widget.x &&
        x < widget.x + widget.w &&
        y >= widget.y &&
        y < widget.y + widget.h
      );
    });
  }

  findAvailablePosition(width, height) {
    for (let y = 0; y < 20; y++) {
      // Max 20 rows
      for (let x = 0; x <= this.options.columns - width; x++) {
        if (this.canPlaceWidget(x, y, width, height)) {
          return { x, y };
        }
      }
    }

    // If no position found, place at the end
    return { x: 0, y: this.getMaxRow() + 1 };
  }

  canPlaceWidget(x, y, width, height) {
    for (let dy = 0; dy < height; dy++) {
      for (let dx = 0; dx < width; dx++) {
        if (this.isPositionOccupied(x + dx, y + dy)) {
          return false;
        }
      }
    }
    return true;
  }

  getMaxRow() {
    return Math.max(0, ...this.state.layout.map((w) => w.y + w.h - 1));
  }

  // Layout management
  toggleEditMode() {
    this.state.isEditing = !this.state.isEditing;

    this.toggleEditBtn.setAttribute("aria-pressed", this.state.isEditing);
    this.toggleEditBtn.querySelector(".edit-text").textContent = this.state
      .isEditing
      ? "Exit Edit"
      : "Edit Layout";

    this.layoutControls.style.display = this.state.isEditing ? "flex" : "none";
    this.widgetPalette.style.display = this.state.isEditing ? "block" : "none";

    this.container.classList.toggle("editing", this.state.isEditing);

    this.renderWidgets();
    this.renderDropZones();

    this.announceChange(
      this.state.isEditing ? "Edit mode enabled" : "Edit mode disabled",
    );
  }

  saveToHistory() {
    // Remove any future history if we're not at the end
    if (this.state.historyIndex < this.state.history.length - 1) {
      this.state.history = this.state.history.slice(
        0,
        this.state.historyIndex + 1,
      );
    }

    // Add current state to history
    this.state.history.push(JSON.parse(JSON.stringify(this.state.layout)));
    this.state.historyIndex++;

    // Limit history size
    if (this.state.history.length > 50) {
      this.state.history.shift();
      this.state.historyIndex--;
    }

    this.updateHistoryButtons();
  }

  undo() {
    if (this.state.historyIndex > 0) {
      this.state.historyIndex--;
      this.state.layout = JSON.parse(
        JSON.stringify(this.state.history[this.state.historyIndex]),
      );
      this.renderWidgets();
      this.renderDropZones();
      this.saveLayout();
      this.updateHistoryButtons();
      this.announceChange("Undo applied");
    }
  }

  redo() {
    if (this.state.historyIndex < this.state.history.length - 1) {
      this.state.historyIndex++;
      this.state.layout = JSON.parse(
        JSON.stringify(this.state.history[this.state.historyIndex]),
      );
      this.renderWidgets();
      this.renderDropZones();
      this.saveLayout();
      this.updateHistoryButtons();
      this.announceChange("Redo applied");
    }
  }

  updateHistoryButtons() {
    const undoBtn = this.container.querySelector(".undo-btn");
    const redoBtn = this.container.querySelector(".redo-btn");

    if (undoBtn) undoBtn.disabled = this.state.historyIndex <= 0;
    if (redoBtn)
      redoBtn.disabled =
        this.state.historyIndex >= this.state.history.length - 1;
  }

  resetLayout() {
    if (
      confirm(
        "Are you sure you want to reset the layout? This cannot be undone.",
      )
    ) {
      this.state.layout = [...this.options.widgets];
      this.saveToHistory();
      this.renderWidgets();
      this.renderDropZones();
      this.saveLayout();
      this.announceChange("Layout reset to default");
    }
  }

  saveTemplate() {
    const templateName = prompt("Enter a name for this layout template:");
    if (templateName) {
      const templates = JSON.parse(
        localStorage.getItem("dashboard-templates") || "{}",
      );
      templates[templateName] = JSON.parse(JSON.stringify(this.state.layout));
      localStorage.setItem("dashboard-templates", JSON.stringify(templates));
      this.announceChange(`Template "${templateName}" saved`);
    }
  }

  loadTemplate(templateName) {
    const templates = JSON.parse(
      localStorage.getItem("dashboard-templates") || "{}",
    );
    if (templates[templateName]) {
      this.state.layout = JSON.parse(JSON.stringify(templates[templateName]));
      this.saveToHistory();
      this.renderWidgets();
      this.renderDropZones();
      this.saveLayout();
      this.announceChange(`Template "${templateName}" loaded`);
    }
  }

  saveLayout() {
    if (this.options.saveLayout) {
      const layoutData = {
        layout: this.state.layout,
        breakpoint: this.state.currentBreakpoint,
        timestamp: Date.now(),
      };
      localStorage.setItem(this.options.layoutKey, JSON.stringify(layoutData));
    }
  }

  loadLayout() {
    if (this.options.saveLayout) {
      const saved = localStorage.getItem(this.options.layoutKey);
      if (saved) {
        try {
          const layoutData = JSON.parse(saved);
          this.state.layout = layoutData.layout || [];
          this.saveToHistory();
        } catch (error) {
          console.warn("Failed to load saved layout:", error);
          this.state.layout = [...this.options.widgets];
        }
      } else {
        this.state.layout = [...this.options.widgets];
      }
    } else {
      this.state.layout = [...this.options.widgets];
    }
  }

  updateBreakpoint() {
    const width = this.containerWidth;
    let newBreakpoint = "xs";

    if (width >= this.options.breakpoints.lg) newBreakpoint = "lg";
    else if (width >= this.options.breakpoints.md) newBreakpoint = "md";
    else if (width >= this.options.breakpoints.sm) newBreakpoint = "sm";

    if (newBreakpoint !== this.state.currentBreakpoint) {
      this.state.currentBreakpoint = newBreakpoint;
      this.updateLayoutInfo();
      this.adjustLayoutForBreakpoint();
    }
  }

  adjustLayoutForBreakpoint() {
    // Adjust column count for smaller screens
    const breakpointColumns = {
      lg: 12,
      md: 8,
      sm: 6,
      xs: 4,
    };

    const newColumns = breakpointColumns[this.state.currentBreakpoint];
    if (newColumns !== this.options.columns) {
      this.options.columns = newColumns;
      this.calculateDimensions();

      // Adjust widget positions to fit new column count
      this.state.layout.forEach((widget) => {
        if (widget.x + widget.w > this.options.columns) {
          widget.x = Math.max(0, this.options.columns - widget.w);
        }
        if (widget.w > this.options.columns) {
          widget.w = this.options.columns;
        }
      });

      this.renderWidgets();
      this.renderGridBackground();
      this.renderDropZones();
    }
  }

  updateLayoutInfo() {
    const breakpointNames = {
      lg: "Desktop Layout",
      md: "Tablet Layout",
      sm: "Mobile Layout",
      xs: "Small Mobile Layout",
    };

    const breakpointElement = this.container.querySelector(
      ".current-breakpoint",
    );
    if (breakpointElement) {
      breakpointElement.textContent =
        breakpointNames[this.state.currentBreakpoint];
    }
  }

  updateWidgetFocus() {
    this.container.querySelectorAll(".dashboard-widget").forEach((element) => {
      element.classList.toggle(
        "focused",
        element.dataset.id === this.state.focusedWidget,
      );
    });
  }

  setupAccessibility() {
    // Add ARIA live region for announcements
    const liveRegion = document.createElement("div");
    liveRegion.id = "dashboard-announcements";
    liveRegion.className = "sr-only";
    liveRegion.setAttribute("aria-live", "polite");
    liveRegion.setAttribute("aria-atomic", "true");
    this.container.appendChild(liveRegion);
  }

  announceChange(message) {
    const liveRegion = this.container.querySelector("#dashboard-announcements");
    if (liveRegion) {
      liveRegion.textContent = message;
    }
  }

  // Public API methods
  getLayout() {
    return JSON.parse(JSON.stringify(this.state.layout));
  }

  setLayout(layout) {
    this.state.layout = JSON.parse(JSON.stringify(layout));
    this.saveToHistory();
    this.renderWidgets();
    this.renderDropZones();
    this.saveLayout();
  }

  getWidget(widgetId) {
    return this.state.layout.find((w) => w.id === widgetId);
  }

  updateWidget(widgetId, updates) {
    const widget = this.state.layout.find((w) => w.id === widgetId);
    if (widget) {
      Object.assign(widget, updates);
      this.renderWidgets();
      this.saveLayout();
    }
  }

  exportLayout() {
    return {
      layout: this.getLayout(),
      breakpoint: this.state.currentBreakpoint,
      options: {
        columns: this.options.columns,
        rowHeight: this.options.rowHeight,
        margin: this.options.margin,
      },
      exported: new Date().toISOString(),
    };
  }

  importLayout(data) {
    if (data.layout) {
      this.setLayout(data.layout);
      this.announceChange("Layout imported successfully");
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
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }

    this.observers.forEach((observer) => observer.disconnect());
    this.eventHandlers.clear();

    this.container.innerHTML = "";
  }
}

// Export for module systems
if (typeof module !== "undefined" && module.exports) {
  module.exports = DragDropDashboard;
}

// Global access
window.DragDropDashboard = DragDropDashboard;
