/**
 * Mobile-Optimized UI Components
 * Touch-friendly interfaces with gesture support and responsive dashboard layouts
 * Provides native-like mobile experience for anomaly detection platform
 */

/**
 * Touch Gesture Recognition System
 * Handles swipe, pinch, pan, and tap gestures for mobile interactions
 */
class TouchGestureManager {
  constructor(element, options = {}) {
    this.element = element;
    this.options = {
      swipeThreshold: 50, // Minimum distance for swipe
      tapTimeout: 300, // Maximum time for tap
      doubleTapTimeout: 300, // Time between taps for double tap
      pinchThreshold: 10, // Minimum distance change for pinch
      enablePinch: true,
      enableSwipe: true,
      enableTap: true,
      enablePan: true,
      preventDefault: true,
      ...options,
    };

    // Touch state tracking
    this.touches = new Map();
    this.lastTap = null;
    this.isGesturing = false;
    this.gestureStartDistance = 0;
    this.gestureStartScale = 1;

    // Event listeners
    this.listeners = new Map();

    this.init();
  }

  init() {
    this.bindEvents();
  }

  bindEvents() {
    // Touch events
    this.element.addEventListener(
      "touchstart",
      this.handleTouchStart.bind(this),
      { passive: false },
    );
    this.element.addEventListener(
      "touchmove",
      this.handleTouchMove.bind(this),
      { passive: false },
    );
    this.element.addEventListener("touchend", this.handleTouchEnd.bind(this), {
      passive: false,
    });
    this.element.addEventListener(
      "touchcancel",
      this.handleTouchCancel.bind(this),
      { passive: false },
    );

    // Mouse events for desktop testing
    this.element.addEventListener("mousedown", this.handleMouseDown.bind(this));
    this.element.addEventListener("mousemove", this.handleMouseMove.bind(this));
    this.element.addEventListener("mouseup", this.handleMouseUp.bind(this));

    // Prevent default context menu on long press
    this.element.addEventListener("contextmenu", (e) => {
      if (this.options.preventDefault) {
        e.preventDefault();
      }
    });
  }

  handleTouchStart(event) {
    if (this.options.preventDefault) {
      event.preventDefault();
    }

    const touches = Array.from(event.changedTouches);
    touches.forEach((touch) => {
      this.touches.set(touch.identifier, {
        id: touch.identifier,
        startX: touch.clientX,
        startY: touch.clientY,
        currentX: touch.clientX,
        currentY: touch.clientY,
        startTime: Date.now(),
        element: touch.target,
      });
    });

    if (event.touches.length === 2 && this.options.enablePinch) {
      this.startPinchGesture(event.touches);
    }

    this.emit("touchstart", {
      touches: Array.from(this.touches.values()),
      originalEvent: event,
    });
  }

  handleTouchMove(event) {
    if (this.options.preventDefault) {
      event.preventDefault();
    }

    const touches = Array.from(event.changedTouches);
    touches.forEach((touch) => {
      const touchData = this.touches.get(touch.identifier);
      if (touchData) {
        touchData.currentX = touch.clientX;
        touchData.currentY = touch.clientY;
      }
    });

    if (event.touches.length === 2 && this.options.enablePinch) {
      this.handlePinchGesture(event.touches);
    } else if (event.touches.length === 1 && this.options.enablePan) {
      this.handlePanGesture(Array.from(this.touches.values())[0]);
    }

    this.emit("touchmove", {
      touches: Array.from(this.touches.values()),
      originalEvent: event,
    });
  }

  handleTouchEnd(event) {
    const touches = Array.from(event.changedTouches);

    touches.forEach((touch) => {
      const touchData = this.touches.get(touch.identifier);
      if (touchData) {
        this.processTouchEnd(touchData);
        this.touches.delete(touch.identifier);
      }
    });

    if (event.touches.length === 0) {
      this.isGesturing = false;
    }

    this.emit("touchend", {
      touches: Array.from(this.touches.values()),
      originalEvent: event,
    });
  }

  handleTouchCancel(event) {
    event.changedTouches.forEach((touch) => {
      this.touches.delete(touch.identifier);
    });
    this.isGesturing = false;
  }

  processTouchEnd(touchData) {
    const duration = Date.now() - touchData.startTime;
    const deltaX = touchData.currentX - touchData.startX;
    const deltaY = touchData.currentY - touchData.startY;
    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

    // Check for tap
    if (
      this.options.enableTap &&
      duration < this.options.tapTimeout &&
      distance < 10
    ) {
      this.handleTap(touchData);
    }

    // Check for swipe
    if (this.options.enableSwipe && distance > this.options.swipeThreshold) {
      this.handleSwipe(touchData, deltaX, deltaY, distance);
    }
  }

  handleTap(touchData) {
    const now = Date.now();
    const tapEvent = {
      x: touchData.currentX,
      y: touchData.currentY,
      element: touchData.element,
      timestamp: now,
    };

    // Check for double tap
    if (
      this.lastTap &&
      now - this.lastTap.timestamp < this.options.doubleTapTimeout &&
      Math.abs(this.lastTap.x - tapEvent.x) < 25 &&
      Math.abs(this.lastTap.y - tapEvent.y) < 25
    ) {
      this.emit("doubletap", tapEvent);
      this.lastTap = null;
    } else {
      this.emit("tap", tapEvent);
      this.lastTap = tapEvent;

      // Clear last tap after timeout
      setTimeout(() => {
        if (this.lastTap === tapEvent) {
          this.lastTap = null;
        }
      }, this.options.doubleTapTimeout);
    }
  }

  handleSwipe(touchData, deltaX, deltaY, distance) {
    const direction = this.getSwipeDirection(deltaX, deltaY);

    this.emit("swipe", {
      direction,
      deltaX,
      deltaY,
      distance,
      velocity: distance / (Date.now() - touchData.startTime),
      startX: touchData.startX,
      startY: touchData.startY,
      endX: touchData.currentX,
      endY: touchData.currentY,
    });
  }

  getSwipeDirection(deltaX, deltaY) {
    const absDeltaX = Math.abs(deltaX);
    const absDeltaY = Math.abs(deltaY);

    if (absDeltaX > absDeltaY) {
      return deltaX > 0 ? "right" : "left";
    } else {
      return deltaY > 0 ? "down" : "up";
    }
  }

  startPinchGesture(touches) {
    const touch1 = touches[0];
    const touch2 = touches[1];

    this.gestureStartDistance = this.calculateDistance(
      touch1.clientX,
      touch1.clientY,
      touch2.clientX,
      touch2.clientY,
    );
    this.isGesturing = true;
  }

  handlePinchGesture(touches) {
    if (!this.isGesturing) return;

    const touch1 = touches[0];
    const touch2 = touches[1];

    const currentDistance = this.calculateDistance(
      touch1.clientX,
      touch1.clientY,
      touch2.clientX,
      touch2.clientY,
    );

    const scale = currentDistance / this.gestureStartDistance;
    const centerX = (touch1.clientX + touch2.clientX) / 2;
    const centerY = (touch1.clientY + touch2.clientY) / 2;

    this.emit("pinch", {
      scale,
      centerX,
      centerY,
      distance: currentDistance,
      startDistance: this.gestureStartDistance,
    });
  }

  handlePanGesture(touchData) {
    const deltaX = touchData.currentX - touchData.startX;
    const deltaY = touchData.currentY - touchData.startY;

    this.emit("pan", {
      deltaX,
      deltaY,
      currentX: touchData.currentX,
      currentY: touchData.currentY,
      startX: touchData.startX,
      startY: touchData.startY,
    });
  }

  calculateDistance(x1, y1, x2, y2) {
    return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
  }

  // Mouse event handlers for desktop testing
  handleMouseDown(event) {
    this.touches.set("mouse", {
      id: "mouse",
      startX: event.clientX,
      startY: event.clientY,
      currentX: event.clientX,
      currentY: event.clientY,
      startTime: Date.now(),
      element: event.target,
    });
  }

  handleMouseMove(event) {
    const touchData = this.touches.get("mouse");
    if (touchData) {
      touchData.currentX = event.clientX;
      touchData.currentY = event.clientY;
      this.handlePanGesture(touchData);
    }
  }

  handleMouseUp(event) {
    const touchData = this.touches.get("mouse");
    if (touchData) {
      this.processTouchEnd(touchData);
      this.touches.delete("mouse");
    }
  }

  // Event system
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
    return () => this.off(event, callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).delete(callback);
    }
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach((callback) => {
        try {
          callback(data);
        } catch (error) {
          console.error("Touch gesture callback error:", error);
        }
      });
    }
  }

  destroy() {
    this.touches.clear();
    this.listeners.clear();
    this.lastTap = null;
  }
}

/**
 * Mobile Dashboard Layout Manager
 * Responsive layout system optimized for mobile screens
 */
class MobileDashboardManager {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      enableSwipeNavigation: true,
      enablePullToRefresh: true,
      enableCollapsiblePanels: true,
      tabBarHeight: 60,
      headerHeight: 56,
      minPanelHeight: 200,
      maxColumns: { mobile: 1, tablet: 2, desktop: 3 },
      breakpoints: {
        mobile: 768,
        tablet: 1024,
        desktop: 1200,
      },
      ...options,
    };

    this.currentLayout = "mobile";
    this.widgets = new Map();
    this.panels = new Map();
    this.activeTab = 0;
    this.isRefreshing = false;

    // UI elements
    this.header = null;
    this.tabBar = null;
    this.contentArea = null;
    this.pullToRefreshIndicator = null;

    this.init();
  }

  init() {
    this.detectLayout();
    this.createMobileStructure();
    this.setupGestureHandling();
    this.setupResizeListener();
    this.setupPullToRefresh();
  }

  detectLayout() {
    const width = window.innerWidth;
    if (width <= this.options.breakpoints.mobile) {
      this.currentLayout = "mobile";
    } else if (width <= this.options.breakpoints.tablet) {
      this.currentLayout = "tablet";
    } else {
      this.currentLayout = "desktop";
    }
  }

  createMobileStructure() {
    this.container.className = `mobile-dashboard ${this.currentLayout}`;

    this.container.innerHTML = `
      <header class="mobile-header">
        <div class="header-content">
          <button class="menu-button" aria-label="Menu">
            <span class="hamburger"></span>
          </button>
          <h1 class="header-title">Pynomaly</h1>
          <div class="header-actions">
            <button class="refresh-button" aria-label="Refresh">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
              </svg>
            </button>
            <button class="settings-button" aria-label="Settings">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M19.14,12.94c0.04-0.3,0.06-0.61,0.06-0.94c0-0.32-0.02-0.64-0.07-0.94l2.03-1.58c0.18-0.14,0.23-0.41,0.12-0.61 l-1.92-3.32c-0.12-0.22-0.37-0.29-0.59-0.22l-2.39,0.96c-0.5-0.38-1.03-0.7-1.62-0.94L14.4,2.81c-0.04-0.24-0.24-0.41-0.48-0.41 h-3.84c-0.24,0-0.43,0.17-0.47,0.41L9.25,5.35C8.66,5.59,8.12,5.92,7.63,6.29L5.24,5.33c-0.22-0.08-0.47,0-0.59,0.22L2.74,8.87 C2.62,9.08,2.66,9.34,2.86,9.48l2.03,1.58C4.84,11.36,4.8,11.69,4.8,12s0.02,0.64,0.07,0.94l-2.03,1.58 c-0.18,0.14-0.23,0.41-0.12,0.61l1.92,3.32c0.12,0.22,0.37,0.29,0.59,0.22l2.39-0.96c0.5,0.38,1.03,0.7,1.62,0.94l0.36,2.54 c0.05,0.24,0.24,0.41,0.48,0.41h3.84c0.24,0,0.44-0.17,0.47-0.41l0.36-2.54c0.59-0.24,1.13-0.56,1.62-0.94l2.39,0.96 c0.22,0.08,0.47,0,0.59-0.22l1.92-3.32c0.12-0.22,0.07-0.47-0.12-0.61L19.14,12.94z M12,15.6c-1.98,0-3.6-1.62-3.6-3.6 s1.62-3.6,3.6-3.6s3.6,1.62,3.6,3.6S13.98,15.6,12,15.6z"/>
              </svg>
            </button>
          </div>
        </div>
      </header>

      <div class="pull-to-refresh-indicator">
        <div class="refresh-spinner"></div>
        <span class="refresh-text">Pull to refresh</span>
      </div>

      <main class="content-area">
        <div class="dashboard-tabs" role="tablist">
          <!-- Tabs will be dynamically generated -->
        </div>
        
        <div class="tab-panels">
          <!-- Panel content will be dynamically generated -->
        </div>
      </main>

      <nav class="tab-bar" role="tablist">
        <button class="tab-button active" data-tab="0" role="tab" aria-selected="true">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
          </svg>
          <span>Dashboard</span>
        </button>
        <button class="tab-button" data-tab="1" role="tab" aria-selected="false">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/>
          </svg>
          <span>Analytics</span>
        </button>
        <button class="tab-button" data-tab="2" role="tab" aria-selected="false">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
          </svg>
          <span>Alerts</span>
        </button>
        <button class="tab-button" data-tab="3" role="tab" aria-selected="false">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
          </svg>
          <span>Models</span>
        </button>
      </nav>
    `;

    // Cache DOM elements
    this.header = this.container.querySelector(".mobile-header");
    this.tabBar = this.container.querySelector(".tab-bar");
    this.contentArea = this.container.querySelector(".content-area");
    this.pullToRefreshIndicator = this.container.querySelector(
      ".pull-to-refresh-indicator",
    );

    this.setupTabNavigation();
  }

  setupTabNavigation() {
    const tabButtons = this.tabBar.querySelectorAll(".tab-button");

    tabButtons.forEach((button, index) => {
      button.addEventListener("click", () => {
        this.switchTab(index);
      });
    });

    // Initialize first tab
    this.switchTab(0);
  }

  switchTab(tabIndex) {
    const tabButtons = this.tabBar.querySelectorAll(".tab-button");
    const panels = this.contentArea.querySelectorAll(".tab-panel");

    // Update tab buttons
    tabButtons.forEach((button, index) => {
      const isActive = index === tabIndex;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-selected", isActive);
    });

    // Update panels
    panels.forEach((panel, index) => {
      panel.classList.toggle("active", index === tabIndex);
    });

    this.activeTab = tabIndex;
    this.emit("tab-changed", { activeTab: tabIndex });
  }

  setupGestureHandling() {
    const gestureManager = new TouchGestureManager(this.contentArea, {
      enableSwipe: this.options.enableSwipeNavigation,
      enablePinch: false, // Disable pinch on main container
    });

    // Swipe navigation between tabs
    if (this.options.enableSwipeNavigation) {
      gestureManager.on("swipe", (gesture) => {
        if (Math.abs(gesture.deltaY) < 50) {
          // Horizontal swipes only
          if (gesture.direction === "left" && this.activeTab < 3) {
            this.switchTab(this.activeTab + 1);
          } else if (gesture.direction === "right" && this.activeTab > 0) {
            this.switchTab(this.activeTab - 1);
          }
        }
      });
    }
  }

  setupPullToRefresh() {
    if (!this.options.enablePullToRefresh) return;

    let pullStartY = 0;
    let pullCurrentY = 0;
    let pullDeltaY = 0;
    let isPulling = false;

    const gestureManager = new TouchGestureManager(this.contentArea);

    gestureManager.on("touchstart", (event) => {
      if (this.contentArea.scrollTop === 0) {
        pullStartY = event.touches[0].currentY;
        isPulling = true;
      }
    });

    gestureManager.on("touchmove", (event) => {
      if (!isPulling) return;

      pullCurrentY = event.touches[0].currentY;
      pullDeltaY = pullCurrentY - pullStartY;

      if (pullDeltaY > 0 && this.contentArea.scrollTop === 0) {
        const pullDistance = Math.min(pullDeltaY, 100);
        const opacity = Math.min(pullDistance / 60, 1);

        this.pullToRefreshIndicator.style.transform = `translateY(${pullDistance}px)`;
        this.pullToRefreshIndicator.style.opacity = opacity;

        if (pullDistance > 60) {
          this.pullToRefreshIndicator.classList.add("ready");
          this.pullToRefreshIndicator.querySelector(
            ".refresh-text",
          ).textContent = "Release to refresh";
        } else {
          this.pullToRefreshIndicator.classList.remove("ready");
          this.pullToRefreshIndicator.querySelector(
            ".refresh-text",
          ).textContent = "Pull to refresh";
        }
      }
    });

    gestureManager.on("touchend", () => {
      if (!isPulling) return;

      isPulling = false;

      if (pullDeltaY > 60 && !this.isRefreshing) {
        this.triggerRefresh();
      } else {
        this.resetPullToRefresh();
      }
    });
  }

  triggerRefresh() {
    this.isRefreshing = true;
    this.pullToRefreshIndicator.classList.add("refreshing");
    this.pullToRefreshIndicator.querySelector(".refresh-text").textContent =
      "Refreshing...";

    this.emit("refresh-requested");

    // Auto-reset after 3 seconds if not manually reset
    setTimeout(() => {
      if (this.isRefreshing) {
        this.resetPullToRefresh();
      }
    }, 3000);
  }

  resetPullToRefresh() {
    this.isRefreshing = false;
    this.pullToRefreshIndicator.classList.remove("ready", "refreshing");
    this.pullToRefreshIndicator.style.transform = "translateY(-100%)";
    this.pullToRefreshIndicator.style.opacity = "0";
    this.pullToRefreshIndicator.querySelector(".refresh-text").textContent =
      "Pull to refresh";
  }

  setupResizeListener() {
    let resizeTimeout;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        const oldLayout = this.currentLayout;
        this.detectLayout();

        if (oldLayout !== this.currentLayout) {
          this.container.className = `mobile-dashboard ${this.currentLayout}`;
          this.updateLayoutForScreen();
        }
      }, 250);
    });
  }

  updateLayoutForScreen() {
    const maxCols = this.options.maxColumns[this.currentLayout];

    // Update widget layouts based on screen size
    this.panels.forEach((panel) => {
      panel.updateLayout(maxCols);
    });

    this.emit("layout-changed", { layout: this.currentLayout });
  }

  /**
   * Widget and Panel Management
   */
  createPanel(id, title, content, tabIndex = 0) {
    const panelElement = document.createElement("div");
    panelElement.className = `tab-panel ${tabIndex === this.activeTab ? "active" : ""}`;
    panelElement.setAttribute("role", "tabpanel");
    panelElement.innerHTML = `
      <div class="panel-header">
        <h2 class="panel-title">${title}</h2>
        <div class="panel-actions">
          <button class="panel-collapse-btn" aria-label="Collapse panel">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M7.41 8.84L12 13.42l4.59-4.58L18 10.25l-6 6-6-6z"/>
            </svg>
          </button>
        </div>
      </div>
      <div class="panel-content">
        ${content}
      </div>
    `;

    const panel = {
      id,
      element: panelElement,
      title,
      tabIndex,
      isCollapsed: false,
      updateLayout: (maxCols) => {
        panelElement.style.gridColumn = `span ${Math.min(1, maxCols)}`;
      },
    };

    // Add collapse functionality
    const collapseBtn = panelElement.querySelector(".panel-collapse-btn");
    const panelContent = panelElement.querySelector(".panel-content");

    collapseBtn.addEventListener("click", () => {
      panel.isCollapsed = !panel.isCollapsed;
      panelElement.classList.toggle("collapsed", panel.isCollapsed);

      if (panel.isCollapsed) {
        panelContent.style.height = "0";
        collapseBtn.style.transform = "rotate(-90deg)";
      } else {
        panelContent.style.height = "auto";
        collapseBtn.style.transform = "rotate(0deg)";
      }
    });

    this.panels.set(id, panel);

    // Add to appropriate tab
    const tabPanels = this.contentArea.querySelector(".tab-panels");
    tabPanels.appendChild(panelElement);

    return panel;
  }

  createWidget(id, type, config, panelId) {
    const widget = {
      id,
      type,
      config,
      panelId,
      element: null,
      touchOptimized: true,
    };

    // Create widget element based on type
    switch (type) {
      case "chart":
        widget.element = this.createChartWidget(config);
        break;
      case "metric":
        widget.element = this.createMetricWidget(config);
        break;
      case "list":
        widget.element = this.createListWidget(config);
        break;
      case "form":
        widget.element = this.createFormWidget(config);
        break;
      default:
        widget.element = this.createDefaultWidget(config);
    }

    // Add touch optimizations
    this.optimizeWidgetForTouch(widget);

    this.widgets.set(id, widget);

    // Add to panel
    const panel = this.panels.get(panelId);
    if (panel) {
      panel.element.querySelector(".panel-content").appendChild(widget.element);
    }

    return widget;
  }

  createChartWidget(config) {
    const element = document.createElement("div");
    element.className = "widget chart-widget touch-optimized";
    element.innerHTML = `
      <div class="widget-header">
        <h3 class="widget-title">${config.title || "Chart"}</h3>
        <div class="widget-controls">
          <button class="zoom-out-btn" aria-label="Zoom out">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
              <path d="M7 9h5v1H7z"/>
            </svg>
          </button>
          <button class="fullscreen-btn" aria-label="Fullscreen">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
            </svg>
          </button>
        </div>
      </div>
      <div class="widget-content chart-container" style="height: ${config.height || "250px"};">
        <!-- Chart will be rendered here -->
      </div>
    `;

    return element;
  }

  createMetricWidget(config) {
    const element = document.createElement("div");
    element.className = "widget metric-widget touch-optimized";
    element.innerHTML = `
      <div class="metric-display">
        <div class="metric-value ${config.trend || ""}">${config.value || "0"}</div>
        <div class="metric-label">${config.label || "Metric"}</div>
        <div class="metric-change">${config.change || "+0%"}</div>
      </div>
    `;

    return element;
  }

  createListWidget(config) {
    const element = document.createElement("div");
    element.className = "widget list-widget touch-optimized";
    element.innerHTML = `
      <div class="widget-header">
        <h3 class="widget-title">${config.title || "List"}</h3>
        <button class="refresh-widget-btn" aria-label="Refresh">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
          </svg>
        </button>
      </div>
      <div class="widget-content">
        <div class="list-container">
          <!-- List items will be dynamically added -->
        </div>
      </div>
    `;

    return element;
  }

  createFormWidget(config) {
    const element = document.createElement("div");
    element.className = "widget form-widget touch-optimized";
    element.innerHTML = `
      <div class="widget-header">
        <h3 class="widget-title">${config.title || "Form"}</h3>
      </div>
      <div class="widget-content">
        <form class="mobile-form">
          <!-- Form fields will be dynamically added -->
        </form>
      </div>
    `;

    return element;
  }

  createDefaultWidget(config) {
    const element = document.createElement("div");
    element.className = "widget default-widget touch-optimized";
    element.innerHTML = `
      <div class="widget-content">
        ${config.content || "Widget content"}
      </div>
    `;

    return element;
  }

  optimizeWidgetForTouch(widget) {
    const element = widget.element;

    // Add touch-friendly tap targets
    const buttons = element.querySelectorAll("button");
    buttons.forEach((button) => {
      button.style.minHeight = "44px";
      button.style.minWidth = "44px";
      button.classList.add("touch-target");
    });

    // Add gesture support for charts
    if (widget.type === "chart") {
      const chartContainer = element.querySelector(".chart-container");
      const gestureManager = new TouchGestureManager(chartContainer, {
        enablePinch: true,
        enablePan: true,
      });

      gestureManager.on("pinch", (gesture) => {
        // Handle chart zoom
        this.emit("chart-zoom", {
          widgetId: widget.id,
          scale: gesture.scale,
          centerX: gesture.centerX,
          centerY: gesture.centerY,
        });
      });

      gestureManager.on("pan", (gesture) => {
        // Handle chart pan
        this.emit("chart-pan", {
          widgetId: widget.id,
          deltaX: gesture.deltaX,
          deltaY: gesture.deltaY,
        });
      });

      gestureManager.on("doubletap", () => {
        // Reset chart zoom
        this.emit("chart-reset", { widgetId: widget.id });
      });
    }

    // Add touch feedback
    element.addEventListener("touchstart", () => {
      element.classList.add("touch-active");
    });

    element.addEventListener("touchend", () => {
      setTimeout(() => {
        element.classList.remove("touch-active");
      }, 150);
    });
  }

  /**
   * Mobile-specific features
   */
  enableHapticFeedback() {
    if ("vibrate" in navigator) {
      return {
        light: () => navigator.vibrate(10),
        medium: () => navigator.vibrate(20),
        heavy: () => navigator.vibrate(50),
        success: () => navigator.vibrate([50, 50, 50]),
        error: () => navigator.vibrate([100, 50, 100]),
      };
    }
    return {
      light: () => {},
      medium: () => {},
      heavy: () => {},
      success: () => {},
      error: () => {},
    };
  }

  showToast(message, type = "info", duration = 3000) {
    const toast = document.createElement("div");
    toast.className = `mobile-toast ${type}`;
    toast.innerHTML = `
      <div class="toast-content">
        <span class="toast-message">${message}</span>
        <button class="toast-close" aria-label="Close">×</button>
      </div>
    `;

    document.body.appendChild(toast);

    // Add touch handling
    const gestureManager = new TouchGestureManager(toast);
    gestureManager.on("swipe", (gesture) => {
      if (gesture.direction === "up" || gesture.direction === "right") {
        this.dismissToast(toast);
      }
    });

    toast.querySelector(".toast-close").addEventListener("click", () => {
      this.dismissToast(toast);
    });

    // Auto-dismiss
    setTimeout(() => {
      this.dismissToast(toast);
    }, duration);

    // Show animation
    requestAnimationFrame(() => {
      toast.classList.add("show");
    });
  }

  dismissToast(toast) {
    toast.classList.add("dismiss");
    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
    }, 300);
  }

  // Event system
  on(event, callback) {
    if (!this.listeners) {
      this.listeners = new Map();
    }
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
    return () => this.off(event, callback);
  }

  off(event, callback) {
    if (this.listeners && this.listeners.has(event)) {
      this.listeners.get(event).delete(callback);
    }
  }

  emit(event, data) {
    if (this.listeners && this.listeners.has(event)) {
      this.listeners.get(event).forEach((callback) => {
        try {
          callback(data);
        } catch (error) {
          console.error("Mobile dashboard event error:", error);
        }
      });
    }
  }

  /**
   * Cleanup
   */
  destroy() {
    if (this.listeners) {
      this.listeners.clear();
    }
    this.widgets.clear();
    this.panels.clear();
  }
}

// Export classes
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    TouchGestureManager,
    MobileDashboardManager,
  };
} else {
  // Browser environment
  window.TouchGestureManager = TouchGestureManager;
  window.MobileDashboardManager = MobileDashboardManager;
}
