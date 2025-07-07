/**
 * Advanced Dashboard Layout System
 *
 * Drag-and-drop dashboard configuration with responsive grid layouts
 * Features widget management, layout persistence, and real-time updates
 */

export class DashboardLayout {
  constructor(container, options = {}) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;
    this.options = {
      columns: 12,
      rowHeight: 60,
      margin: [10, 10],
      containerPadding: [10, 10],
      maxRows: Infinity,
      isDraggable: true,
      isResizable: true,
      preventCollision: false,
      autoSize: true,
      compactType: "vertical",
      layouts: {},
      breakpoints: {
        lg: 1200,
        md: 996,
        sm: 768,
        xs: 480,
        xxs: 0,
      },
      responsiveLayouts: {
        lg: [],
        md: [],
        sm: [],
        xs: [],
        xxs: [],
      },
      ...options,
    };

    this.widgets = new Map();
    this.layout = [];
    this.currentBreakpoint = "lg";
    this.isDragging = false;
    this.isResizing = false;
    this.draggedWidget = null;
    this.placeholder = null;

    this.init();
  }

  init() {
    this.setupContainer();
    this.detectBreakpoint();
    this.bindEvents();
    this.render();
  }

  setupContainer() {
    this.container.classList.add("dashboard-layout");
    this.container.innerHTML = "";

    // Create grid overlay for visual guidance
    this.gridOverlay = document.createElement("div");
    this.gridOverlay.className = "grid-overlay";
    this.gridOverlay.style.display = "none";
    this.container.appendChild(this.gridOverlay);

    // Set container styles
    Object.assign(this.container.style, {
      position: "relative",
      minHeight: "100vh",
      padding: `${this.options.containerPadding[1]}px ${this.options.containerPadding[0]}px`,
    });
  }

  detectBreakpoint() {
    const width = this.container.clientWidth;

    for (const [breakpoint, minWidth] of Object.entries(
      this.options.breakpoints,
    )) {
      if (width >= minWidth) {
        this.currentBreakpoint = breakpoint;
        break;
      }
    }

    // Load layout for current breakpoint
    this.loadLayout(this.currentBreakpoint);
  }

  bindEvents() {
    // Resize observer for responsive behavior
    if (window.ResizeObserver) {
      this.resizeObserver = new ResizeObserver(() => {
        this.detectBreakpoint();
        this.render();
      });
      this.resizeObserver.observe(this.container);
    } else {
      window.addEventListener(
        "resize",
        this.debounce(() => {
          this.detectBreakpoint();
          this.render();
        }, 250),
      );
    }

    // Drag and drop events
    this.container.addEventListener("mousedown", this.onMouseDown.bind(this));
    this.container.addEventListener("mousemove", this.onMouseMove.bind(this));
    this.container.addEventListener("mouseup", this.onMouseUp.bind(this));

    // Touch events for mobile
    this.container.addEventListener(
      "touchstart",
      this.onTouchStart.bind(this),
      { passive: false },
    );
    this.container.addEventListener("touchmove", this.onTouchMove.bind(this), {
      passive: false,
    });
    this.container.addEventListener("touchend", this.onTouchEnd.bind(this));

    // Keyboard navigation
    this.container.addEventListener("keydown", this.onKeyDown.bind(this));
  }

  addWidget(widgetConfig) {
    const widget = {
      id: widgetConfig.id || this.generateId(),
      type: widgetConfig.type || "default",
      title: widgetConfig.title || "Widget",
      content: widgetConfig.content || "",
      component: widgetConfig.component || null,
      props: widgetConfig.props || {},
      x: widgetConfig.x || 0,
      y: widgetConfig.y || 0,
      w: widgetConfig.w || 2,
      h: widgetConfig.h || 2,
      minW: widgetConfig.minW || 1,
      minH: widgetConfig.minH || 1,
      maxW: widgetConfig.maxW || Infinity,
      maxH: widgetConfig.maxH || Infinity,
      static: widgetConfig.static || false,
      isDraggable: widgetConfig.isDraggable !== false,
      isResizable: widgetConfig.isResizable !== false,
      moved: false,
      resizeHandles: widgetConfig.resizeHandles || ["se"],
      ...widgetConfig,
    };

    // Find suitable position if not specified
    if (widgetConfig.x === undefined || widgetConfig.y === undefined) {
      const position = this.findSuitablePosition(widget.w, widget.h);
      widget.x = position.x;
      widget.y = position.y;
    }

    this.widgets.set(widget.id, widget);
    this.layout.push(widget);

    // Compact layout if needed
    if (this.options.compactType) {
      this.compactLayout();
    }

    this.render();
    this.saveLayout();

    // Emit event
    this.emitEvent("widgetAdded", { widget });

    return widget;
  }

  removeWidget(widgetId) {
    const widget = this.widgets.get(widgetId);
    if (!widget) return false;

    this.widgets.delete(widgetId);
    this.layout = this.layout.filter((w) => w.id !== widgetId);

    // Remove DOM element
    const element = this.container.querySelector(
      `[data-widget-id="${widgetId}"]`,
    );
    if (element) {
      element.remove();
    }

    // Compact layout
    if (this.options.compactType) {
      this.compactLayout();
    }

    this.render();
    this.saveLayout();

    // Emit event
    this.emitEvent("widgetRemoved", { widget });

    return true;
  }

  updateWidget(widgetId, updates) {
    const widget = this.widgets.get(widgetId);
    if (!widget) return false;

    Object.assign(widget, updates);

    // Update in layout array
    const layoutIndex = this.layout.findIndex((w) => w.id === widgetId);
    if (layoutIndex >= 0) {
      this.layout[layoutIndex] = widget;
    }

    this.render();
    this.saveLayout();

    // Emit event
    this.emitEvent("widgetUpdated", { widget, updates });

    return true;
  }

  findSuitablePosition(width, height) {
    const columns = this.options.columns;

    // Try to find a position without collisions
    for (let y = 0; y < this.options.maxRows; y++) {
      for (let x = 0; x <= columns - width; x++) {
        if (!this.hasCollision({ x, y, w: width, h: height })) {
          return { x, y };
        }
      }
    }

    // If no space found, place at bottom
    const maxY = Math.max(0, ...this.layout.map((w) => w.y + w.h));
    return { x: 0, y: maxY };
  }

  hasCollision(widget, excludeId = null) {
    return this.layout.some((w) => {
      if (w.id === excludeId) return false;

      return !(
        widget.x >= w.x + w.w ||
        widget.x + widget.w <= w.x ||
        widget.y >= w.y + w.h ||
        widget.y + widget.h <= w.y
      );
    });
  }

  compactLayout() {
    if (this.options.compactType === "vertical") {
      this.compactVertical();
    } else if (this.options.compactType === "horizontal") {
      this.compactHorizontal();
    }
  }

  compactVertical() {
    // Sort by y position, then x
    const sortedLayout = [...this.layout].sort((a, b) => {
      if (a.y === b.y) return a.x - b.x;
      return a.y - b.y;
    });

    sortedLayout.forEach((widget) => {
      if (widget.static) return;

      // Find the highest position this widget can move to
      let targetY = 0;
      for (let y = 0; y < widget.y; y++) {
        const testWidget = { ...widget, y };
        if (!this.hasCollision(testWidget, widget.id)) {
          targetY = y;
        } else {
          break;
        }
      }

      widget.y = targetY;
    });
  }

  compactHorizontal() {
    // Similar to vertical but moves widgets left
    const sortedLayout = [...this.layout].sort((a, b) => {
      if (a.x === b.x) return a.y - b.y;
      return a.x - b.x;
    });

    sortedLayout.forEach((widget) => {
      if (widget.static) return;

      let targetX = 0;
      for (let x = 0; x < widget.x; x++) {
        const testWidget = { ...widget, x };
        if (!this.hasCollision(testWidget, widget.id)) {
          targetX = x;
        } else {
          break;
        }
      }

      widget.x = targetX;
    });
  }

  render() {
    // Clear existing widgets (except overlay)
    const existingWidgets =
      this.container.querySelectorAll(".dashboard-widget");
    existingWidgets.forEach((w) => w.remove());

    // Calculate grid dimensions
    const containerWidth =
      this.container.clientWidth - this.options.containerPadding[0] * 2;
    const colWidth =
      (containerWidth - this.options.margin[0] * (this.options.columns - 1)) /
      this.options.columns;

    // Render each widget
    this.layout.forEach((widget) => {
      const element = this.createWidgetElement(widget, colWidth);
      this.container.appendChild(element);
    });

    // Update container height
    if (this.options.autoSize) {
      const maxY = Math.max(0, ...this.layout.map((w) => w.y + w.h));
      const containerHeight =
        maxY * (this.options.rowHeight + this.options.margin[1]) +
        this.options.containerPadding[1] * 2;
      this.container.style.minHeight = `${containerHeight}px`;
    }
  }

  createWidgetElement(widget, colWidth) {
    const element = document.createElement("div");
    element.className = "dashboard-widget";
    element.setAttribute("data-widget-id", widget.id);
    element.setAttribute("tabindex", "0");
    element.setAttribute("role", "article");
    element.setAttribute("aria-label", `Widget: ${widget.title}`);

    // Calculate position and size
    const x = widget.x * (colWidth + this.options.margin[0]);
    const y = widget.y * (this.options.rowHeight + this.options.margin[1]);
    const width = widget.w * colWidth + (widget.w - 1) * this.options.margin[0];
    const height =
      widget.h * this.options.rowHeight +
      (widget.h - 1) * this.options.margin[1];

    // Apply styles
    Object.assign(element.style, {
      position: "absolute",
      left: `${x}px`,
      top: `${y}px`,
      width: `${width}px`,
      height: `${height}px`,
      backgroundColor: "white",
      border: "1px solid #e2e8f0",
      borderRadius: "8px",
      boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
      overflow: "hidden",
      transition: this.isDragging || this.isResizing ? "none" : "all 0.2s ease",
      cursor: widget.isDraggable ? "move" : "default",
      zIndex: widget.static ? 1 : 2,
    });

    // Create widget content
    const header = document.createElement("div");
    header.className = "widget-header";
    header.style.cssText = `
            padding: 12px 16px;
            border-bottom: 1px solid #e2e8f0;
            background: #f8fafc;
            display: flex;
            justify-content: space-between;
            align-items: center;
            min-height: 44px;
        `;

    const title = document.createElement("h3");
    title.className = "widget-title";
    title.textContent = widget.title;
    title.style.cssText = `
            margin: 0;
            font-size: 14px;
            font-weight: 600;
            color: #1f2937;
        `;

    const actions = document.createElement("div");
    actions.className = "widget-actions";
    actions.style.cssText = `
            display: flex;
            gap: 8px;
        `;

    // Add action buttons
    if (!widget.static) {
      const removeBtn = this.createActionButton("×", () =>
        this.removeWidget(widget.id),
      );
      removeBtn.setAttribute("aria-label", "Remove widget");
      actions.appendChild(removeBtn);
    }

    header.appendChild(title);
    header.appendChild(actions);

    const content = document.createElement("div");
    content.className = "widget-content";
    content.style.cssText = `
            padding: 16px;
            height: calc(100% - 44px);
            overflow: auto;
        `;

    // Render widget content
    if (widget.component && typeof widget.component === "function") {
      const componentElement = widget.component(widget.props);
      content.appendChild(componentElement);
    } else if (widget.content) {
      content.innerHTML = widget.content;
    } else {
      content.innerHTML = `<p style="color: #6b7280;">No content available</p>`;
    }

    element.appendChild(header);
    element.appendChild(content);

    // Add resize handles if resizable
    if (widget.isResizable && !widget.static) {
      this.addResizeHandles(element, widget);
    }

    return element;
  }

  createActionButton(text, onClick) {
    const button = document.createElement("button");
    button.textContent = text;
    button.style.cssText = `
            background: transparent;
            border: none;
            color: #6b7280;
            cursor: pointer;
            font-size: 16px;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            transition: all 0.2s ease;
        `;

    button.addEventListener("mouseenter", () => {
      button.style.backgroundColor = "#e5e7eb";
      button.style.color = "#374151";
    });

    button.addEventListener("mouseleave", () => {
      button.style.backgroundColor = "transparent";
      button.style.color = "#6b7280";
    });

    button.addEventListener("click", (e) => {
      e.stopPropagation();
      onClick();
    });

    return button;
  }

  addResizeHandles(element, widget) {
    widget.resizeHandles.forEach((position) => {
      const handle = document.createElement("div");
      handle.className = `resize-handle resize-handle-${position}`;
      handle.style.cssText = this.getResizeHandleStyles(position);
      handle.setAttribute("data-resize-direction", position);

      handle.addEventListener("mousedown", (e) => {
        e.stopPropagation();
        this.startResize(e, widget, position);
      });

      element.appendChild(handle);
    });
  }

  getResizeHandleStyles(position) {
    const baseStyles = `
            position: absolute;
            background: #3b82f6;
            opacity: 0;
            transition: opacity 0.2s ease;
            cursor: ${this.getResizeCursor(position)};
        `;

    switch (position) {
      case "se":
        return (
          baseStyles +
          `
                    bottom: 0;
                    right: 0;
                    width: 12px;
                    height: 12px;
                    border-radius: 12px 0 0 0;
                `
        );
      case "s":
        return (
          baseStyles +
          `
                    bottom: 0;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 24px;
                    height: 4px;
                `
        );
      case "e":
        return (
          baseStyles +
          `
                    right: 0;
                    top: 50%;
                    transform: translateY(-50%);
                    width: 4px;
                    height: 24px;
                `
        );
      default:
        return baseStyles;
    }
  }

  getResizeCursor(position) {
    const cursors = {
      se: "nw-resize",
      s: "ns-resize",
      e: "ew-resize",
      ne: "ne-resize",
      nw: "nw-resize",
      sw: "sw-resize",
      n: "ns-resize",
      w: "ew-resize",
    };
    return cursors[position] || "default";
  }

  // Event handlers
  onMouseDown(e) {
    const widgetElement = e.target.closest(".dashboard-widget");
    if (!widgetElement) return;

    const widgetId = widgetElement.getAttribute("data-widget-id");
    const widget = this.widgets.get(widgetId);

    if (!widget || widget.static || !widget.isDraggable) return;

    // Check if clicking on resize handle
    if (e.target.classList.contains("resize-handle")) return;

    this.startDrag(e, widget);
  }

  startDrag(e, widget) {
    this.isDragging = true;
    this.draggedWidget = widget;

    const widgetElement = this.container.querySelector(
      `[data-widget-id="${widget.id}"]`,
    );
    const rect = widgetElement.getBoundingClientRect();
    const containerRect = this.container.getBoundingClientRect();

    this.dragOffset = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };

    // Show grid overlay
    this.showGridOverlay();

    // Add dragging class
    widgetElement.classList.add("dragging");

    // Create placeholder
    this.createPlaceholder(widget);

    this.emitEvent("dragStart", { widget });
  }

  startResize(e, widget, direction) {
    this.isResizing = true;
    this.draggedWidget = widget;
    this.resizeDirection = direction;

    const widgetElement = this.container.querySelector(
      `[data-widget-id="${widget.id}"]`,
    );
    widgetElement.classList.add("resizing");

    this.emitEvent("resizeStart", { widget, direction });
  }

  onMouseMove(e) {
    if (this.isDragging) {
      this.handleDrag(e);
    } else if (this.isResizing) {
      this.handleResize(e);
    }
  }

  handleDrag(e) {
    if (!this.draggedWidget) return;

    const containerRect = this.container.getBoundingClientRect();
    const colWidth =
      (this.container.clientWidth -
        this.options.containerPadding[0] * 2 -
        this.options.margin[0] * (this.options.columns - 1)) /
      this.options.columns;

    // Calculate grid position
    const x = Math.round(
      (e.clientX - containerRect.left - this.dragOffset.x) /
        (colWidth + this.options.margin[0]),
    );
    const y = Math.round(
      (e.clientY - containerRect.top - this.dragOffset.y) /
        (this.options.rowHeight + this.options.margin[1]),
    );

    // Constrain to grid bounds
    const constrainedX = Math.max(
      0,
      Math.min(x, this.options.columns - this.draggedWidget.w),
    );
    const constrainedY = Math.max(0, y);

    // Update placeholder position
    if (this.placeholder) {
      this.placeholder.x = constrainedX;
      this.placeholder.y = constrainedY;
      this.updatePlaceholderPosition();
    }
  }

  handleResize(e) {
    if (!this.draggedWidget) return;

    const containerRect = this.container.getBoundingClientRect();
    const colWidth =
      (this.container.clientWidth -
        this.options.containerPadding[0] * 2 -
        this.options.margin[0] * (this.options.columns - 1)) /
      this.options.columns;

    const widget = this.draggedWidget;
    const direction = this.resizeDirection;

    // Calculate new size based on direction
    let newW = widget.w;
    let newH = widget.h;

    if (direction.includes("e")) {
      const newWidth =
        e.clientX -
        containerRect.left -
        widget.x * (colWidth + this.options.margin[0]);
      newW = Math.max(
        widget.minW,
        Math.min(
          widget.maxW,
          Math.round(newWidth / (colWidth + this.options.margin[0])),
        ),
      );
    }

    if (direction.includes("s")) {
      const newHeight =
        e.clientY -
        containerRect.top -
        widget.y * (this.options.rowHeight + this.options.margin[1]);
      newH = Math.max(
        widget.minH,
        Math.min(
          widget.maxH,
          Math.round(
            newHeight / (this.options.rowHeight + this.options.margin[1]),
          ),
        ),
      );
    }

    // Update widget size temporarily for visual feedback
    const widgetElement = this.container.querySelector(
      `[data-widget-id="${widget.id}"]`,
    );
    const width = newW * colWidth + (newW - 1) * this.options.margin[0];
    const height =
      newH * this.options.rowHeight + (newH - 1) * this.options.margin[1];

    widgetElement.style.width = `${width}px`;
    widgetElement.style.height = `${height}px`;
  }

  onMouseUp(e) {
    if (this.isDragging) {
      this.endDrag();
    } else if (this.isResizing) {
      this.endResize();
    }
  }

  endDrag() {
    if (!this.draggedWidget || !this.placeholder) return;

    const widget = this.draggedWidget;

    // Update widget position
    widget.x = this.placeholder.x;
    widget.y = this.placeholder.y;
    widget.moved = true;

    // Remove visual feedback
    const widgetElement = this.container.querySelector(
      `[data-widget-id="${widget.id}"]`,
    );
    widgetElement.classList.remove("dragging");

    this.removePlaceholder();
    this.hideGridOverlay();

    // Handle collisions if needed
    if (!this.options.preventCollision) {
      this.resolveCollisions(widget);
    }

    // Compact layout
    if (this.options.compactType) {
      this.compactLayout();
    }

    // Re-render and save
    this.render();
    this.saveLayout();

    this.emitEvent("dragEnd", { widget });

    this.isDragging = false;
    this.draggedWidget = null;
  }

  endResize() {
    if (!this.draggedWidget) return;

    const widget = this.draggedWidget;
    const widgetElement = this.container.querySelector(
      `[data-widget-id="${widget.id}"]`,
    );

    // Get final size from element
    const containerWidth =
      this.container.clientWidth - this.options.containerPadding[0] * 2;
    const colWidth =
      (containerWidth - this.options.margin[0] * (this.options.columns - 1)) /
      this.options.columns;

    const elementWidth = parseInt(widgetElement.style.width);
    const elementHeight = parseInt(widgetElement.style.height);

    widget.w = Math.round(elementWidth / (colWidth + this.options.margin[0]));
    widget.h = Math.round(
      elementHeight / (this.options.rowHeight + this.options.margin[1]),
    );

    widgetElement.classList.remove("resizing");

    // Handle collisions
    if (!this.options.preventCollision) {
      this.resolveCollisions(widget);
    }

    // Compact and re-render
    if (this.options.compactType) {
      this.compactLayout();
    }

    this.render();
    this.saveLayout();

    this.emitEvent("resizeEnd", { widget });

    this.isResizing = false;
    this.draggedWidget = null;
    this.resizeDirection = null;
  }

  resolveCollisions(movedWidget) {
    // Find colliding widgets and move them
    const collisions = this.layout.filter(
      (w) =>
        w.id !== movedWidget.id &&
        !w.static &&
        this.hasCollision(movedWidget, w.id),
    );

    collisions.forEach((widget) => {
      // Move widget down to resolve collision
      widget.y = movedWidget.y + movedWidget.h;
      widget.moved = true;
    });
  }

  createPlaceholder(widget) {
    this.placeholder = {
      x: widget.x,
      y: widget.y,
      w: widget.w,
      h: widget.h,
    };

    const element = document.createElement("div");
    element.className = "widget-placeholder";
    element.style.cssText = `
            position: absolute;
            background: rgba(59, 130, 246, 0.2);
            border: 2px dashed #3b82f6;
            border-radius: 8px;
            pointer-events: none;
            z-index: 1000;
        `;

    this.placeholderElement = element;
    this.container.appendChild(element);
    this.updatePlaceholderPosition();
  }

  updatePlaceholderPosition() {
    if (!this.placeholderElement || !this.placeholder) return;

    const containerWidth =
      this.container.clientWidth - this.options.containerPadding[0] * 2;
    const colWidth =
      (containerWidth - this.options.margin[0] * (this.options.columns - 1)) /
      this.options.columns;

    const x = this.placeholder.x * (colWidth + this.options.margin[0]);
    const y =
      this.placeholder.y * (this.options.rowHeight + this.options.margin[1]);
    const width =
      this.placeholder.w * colWidth +
      (this.placeholder.w - 1) * this.options.margin[0];
    const height =
      this.placeholder.h * this.options.rowHeight +
      (this.placeholder.h - 1) * this.options.margin[1];

    Object.assign(this.placeholderElement.style, {
      left: `${x}px`,
      top: `${y}px`,
      width: `${width}px`,
      height: `${height}px`,
    });
  }

  removePlaceholder() {
    if (this.placeholderElement) {
      this.placeholderElement.remove();
      this.placeholderElement = null;
    }
    this.placeholder = null;
  }

  showGridOverlay() {
    // Implementation for grid overlay
    this.gridOverlay.style.display = "block";
  }

  hideGridOverlay() {
    this.gridOverlay.style.display = "none";
  }

  // Touch event handlers
  onTouchStart(e) {
    if (e.touches.length === 1) {
      e.preventDefault();
      const touch = e.touches[0];
      this.onMouseDown({
        ...touch,
        target: touch.target,
        stopPropagation: () => e.stopPropagation(),
        preventDefault: () => e.preventDefault(),
      });
    }
  }

  onTouchMove(e) {
    if (e.touches.length === 1 && (this.isDragging || this.isResizing)) {
      e.preventDefault();
      const touch = e.touches[0];
      this.onMouseMove(touch);
    }
  }

  onTouchEnd(e) {
    this.onMouseUp(e);
  }

  // Keyboard navigation
  onKeyDown(e) {
    const widgetElement = e.target.closest(".dashboard-widget");
    if (!widgetElement) return;

    const widgetId = widgetElement.getAttribute("data-widget-id");
    const widget = this.widgets.get(widgetId);

    if (!widget || widget.static) return;

    let moved = false;
    const step = e.shiftKey ? 5 : 1;

    switch (e.key) {
      case "ArrowLeft":
        widget.x = Math.max(0, widget.x - step);
        moved = true;
        break;
      case "ArrowRight":
        widget.x = Math.min(this.options.columns - widget.w, widget.x + step);
        moved = true;
        break;
      case "ArrowUp":
        widget.y = Math.max(0, widget.y - step);
        moved = true;
        break;
      case "ArrowDown":
        widget.y = widget.y + step;
        moved = true;
        break;
      case "Delete":
      case "Backspace":
        this.removeWidget(widgetId);
        e.preventDefault();
        return;
    }

    if (moved) {
      e.preventDefault();

      // Check for collisions
      if (this.hasCollision(widget, widget.id)) {
        // Revert move
        switch (e.key) {
          case "ArrowLeft":
            widget.x += step;
            break;
          case "ArrowRight":
            widget.x -= step;
            break;
          case "ArrowUp":
            widget.y += step;
            break;
          case "ArrowDown":
            widget.y -= step;
            break;
        }
        return;
      }

      this.render();
      this.saveLayout();
      this.emitEvent("widgetMoved", { widget });
    }
  }

  // Layout persistence
  saveLayout() {
    const layoutData = {
      [this.currentBreakpoint]: this.layout.map((w) => ({
        i: w.id,
        x: w.x,
        y: w.y,
        w: w.w,
        h: w.h,
        static: w.static,
      })),
    };

    try {
      localStorage.setItem("dashboard-layout", JSON.stringify(layoutData));
      this.emitEvent("layoutSaved", { layout: layoutData });
    } catch (error) {
      console.error("Failed to save layout:", error);
    }
  }

  loadLayout(breakpoint) {
    try {
      const saved = localStorage.getItem("dashboard-layout");
      if (saved) {
        const layoutData = JSON.parse(saved);
        const breakpointLayout = layoutData[breakpoint];

        if (breakpointLayout) {
          breakpointLayout.forEach((item) => {
            const widget = this.widgets.get(item.i);
            if (widget) {
              widget.x = item.x;
              widget.y = item.y;
              widget.w = item.w;
              widget.h = item.h;
              widget.static = item.static;
            }
          });

          this.emitEvent("layoutLoaded", { layout: breakpointLayout });
        }
      }
    } catch (error) {
      console.error("Failed to load layout:", error);
    }
  }

  // Utility methods
  generateId() {
    return "widget_" + Math.random().toString(36).substr(2, 9);
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

  emitEvent(eventName, detail) {
    const event = new CustomEvent(`dashboard:${eventName}`, {
      detail,
      bubbles: true,
      cancelable: true,
    });
    this.container.dispatchEvent(event);
  }

  // Public API methods
  getLayout() {
    return this.layout.map((w) => ({ ...w }));
  }

  setLayout(layout) {
    this.layout = layout.map((w) => ({ ...w }));
    this.widgets.clear();

    this.layout.forEach((widget) => {
      this.widgets.set(widget.id, widget);
    });

    this.render();
    this.saveLayout();
  }

  exportLayout() {
    return {
      layout: this.getLayout(),
      breakpoint: this.currentBreakpoint,
      options: { ...this.options },
    };
  }

  importLayout(data) {
    if (data.layout) {
      this.setLayout(data.layout);
    }
  }

  destroy() {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }

    this.container.innerHTML = "";
    this.widgets.clear();
    this.layout = [];
  }
}

// Widget registry for reusable components
export class WidgetRegistry {
  constructor() {
    this.widgets = new Map();
  }

  register(type, config) {
    this.widgets.set(type, config);
  }

  create(type, props = {}) {
    const config = this.widgets.get(type);
    if (!config) {
      throw new Error(`Widget type '${type}' not found`);
    }

    return {
      ...config,
      props: { ...config.defaultProps, ...props },
    };
  }

  getTypes() {
    return Array.from(this.widgets.keys());
  }
}

// Default widget types
export const defaultWidgets = {
  metric: {
    type: "metric",
    title: "Metric Widget",
    w: 2,
    h: 2,
    component: (props) => {
      const div = document.createElement("div");
      div.className = "metric-widget";
      div.innerHTML = `
                <div class="metric-value">${props.value || 0}</div>
                <div class="metric-label">${props.label || "Metric"}</div>
                <div class="metric-change ${props.change >= 0 ? "positive" : "negative"}">
                    ${props.change >= 0 ? "+" : ""}${props.change || 0}%
                </div>
            `;
      return div;
    },
    defaultProps: {
      value: 0,
      label: "Metric",
      change: 0,
    },
  },

  chart: {
    type: "chart",
    title: "Chart Widget",
    w: 4,
    h: 3,
    component: (props) => {
      const div = document.createElement("div");
      div.className = "chart-widget";
      div.innerHTML = `
                <div class="chart-placeholder">
                    <div class="chart-icon">📊</div>
                    <p>${props.chartType || "Chart"} visualization</p>
                </div>
            `;
      return div;
    },
    defaultProps: {
      chartType: "Line",
    },
  },

  table: {
    type: "table",
    title: "Data Table",
    w: 6,
    h: 4,
    component: (props) => {
      const div = document.createElement("div");
      div.className = "table-widget";

      const table = document.createElement("table");
      table.className = "widget-table";

      // Create header
      if (props.columns) {
        const thead = document.createElement("thead");
        const headerRow = document.createElement("tr");
        props.columns.forEach((col) => {
          const th = document.createElement("th");
          th.textContent = col;
          headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);
      }

      // Create body
      const tbody = document.createElement("tbody");
      if (props.data && props.data.length > 0) {
        props.data.forEach((row) => {
          const tr = document.createElement("tr");
          Object.values(row).forEach((cell) => {
            const td = document.createElement("td");
            td.textContent = cell;
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
      } else {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = props.columns?.length || 1;
        td.textContent = "No data available";
        td.style.textAlign = "center";
        td.style.color = "#6b7280";
        tr.appendChild(td);
        tbody.appendChild(tr);
      }

      table.appendChild(tbody);
      div.appendChild(table);

      return div;
    },
    defaultProps: {
      columns: ["Column 1", "Column 2"],
      data: [],
    },
  },
};

// CSS styles
export const dashboardStyles = `
.dashboard-layout {
    position: relative;
    background: #f8fafc;
}

.dashboard-widget {
    box-sizing: border-box;
    user-select: none;
}

.dashboard-widget:hover .resize-handle {
    opacity: 1;
}

.dashboard-widget.dragging {
    z-index: 1000;
    opacity: 0.8;
}

.dashboard-widget.resizing {
    z-index: 1000;
}

.widget-header {
    cursor: move;
}

.widget-content {
    position: relative;
}

.widget-placeholder {
    animation: pulse 1s ease-in-out infinite alternate;
}

@keyframes pulse {
    from { opacity: 0.3; }
    to { opacity: 0.7; }
}

.metric-widget {
    text-align: center;
    padding: 20px;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #1f2937;
}

.metric-label {
    font-size: 0.875rem;
    color: #6b7280;
    margin: 8px 0;
}

.metric-change {
    font-size: 0.875rem;
    font-weight: 500;
}

.metric-change.positive {
    color: #059669;
}

.metric-change.negative {
    color: #dc2626;
}

.chart-widget {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
}

.chart-placeholder {
    text-align: center;
    color: #6b7280;
}

.chart-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.widget-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}

.widget-table th,
.widget-table td {
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
}

.widget-table th {
    background: #f9fafb;
    font-weight: 600;
    color: #374151;
}

.grid-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    pointer-events: none;
    z-index: 999;
    background-image: 
        linear-gradient(to right, rgba(0,0,0,0.1) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(0,0,0,0.1) 1px, transparent 1px);
    background-size: 60px 60px;
}
`;
