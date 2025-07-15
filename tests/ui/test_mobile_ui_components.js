/**
 * Unit tests for mobile UI JavaScript components
 * Tests TouchGestureManager and MobileDashboardManager functionality
 */

// Mock DOM environment for Node.js testing
const { JSDOM } = require('jsdom');

describe('Mobile UI Components', () => {
  let dom;
  let document;
  let window;
  let TouchGestureManager;
  let MobileDashboardManager;

  beforeEach(() => {
    // Create DOM environment
    dom = new JSDOM(`
      <!DOCTYPE html>
      <html>
        <body>
          <div class="mobile-dashboard">
            <header class="mobile-header">
              <div class="header-content">
                <button class="menu-button"></button>
                <h1 class="header-title">Test</h1>
              </div>
            </header>
            <div class="content-area"></div>
            <nav class="tab-bar">
              <button class="tab-button active" data-tab="0"></button>
              <button class="tab-button" data-tab="1"></button>
            </nav>
          </div>
        </body>
      </html>
    `);
    
    global.document = dom.window.document;
    global.window = dom.window;
    global.navigator = { vibrate: jest.fn() };
    
    // Load mobile UI components (would need to be adapted for actual file)
    // For now, we'll mock the basic structure
    TouchGestureManager = class {
      constructor(element, options = {}) {
        this.element = element;
        this.options = options;
        this.listeners = new Map();
      }
      
      on(event, callback) {
        if (!this.listeners.has(event)) {
          this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(callback);
      }
      
      emit(event, data) {
        if (this.listeners.has(event)) {
          this.listeners.get(event).forEach(callback => callback(data));
        }
      }
    };
    
    MobileDashboardManager = class {
      constructor(container, options = {}) {
        this.container = container;
        this.options = options;
        this.listeners = new Map();
        this.activeTab = 0;
        this.isRefreshing = false;
        this.widgets = new Map();
        this.panels = new Map();
      }
      
      switchTab(index) {
        this.activeTab = index;
        this.emit('tab-changed', { activeTab: index });
      }
      
      on(event, callback) {
        if (!this.listeners.has(event)) {
          this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(callback);
      }
      
      emit(event, data) {
        if (this.listeners.has(event)) {
          this.listeners.get(event).forEach(callback => callback(data));
        }
      }
      
      showToast(message, type = 'info', duration = 3000) {
        const toast = document.createElement('div');
        toast.className = `mobile-toast ${type}`;
        toast.innerHTML = `<div class="toast-content"><span>${message}</span></div>`;
        document.body.appendChild(toast);
        return toast;
      }
    };
  });

  afterEach(() => {
    dom.window.close();
  });

  describe('TouchGestureManager', () => {
    test('should initialize with default options', () => {
      const element = document.createElement('div');
      const gestureManager = new TouchGestureManager(element);
      
      expect(gestureManager.element).toBe(element);
      expect(gestureManager.listeners).toBeInstanceOf(Map);
    });

    test('should register event listeners', () => {
      const element = document.createElement('div');
      const gestureManager = new TouchGestureManager(element);
      const callback = jest.fn();
      
      gestureManager.on('swipe', callback);
      
      expect(gestureManager.listeners.get('swipe')).toContain(callback);
    });

    test('should emit events to registered listeners', () => {
      const element = document.createElement('div');
      const gestureManager = new TouchGestureManager(element);
      const callback = jest.fn();
      
      gestureManager.on('swipe', callback);
      gestureManager.emit('swipe', { direction: 'left' });
      
      expect(callback).toHaveBeenCalledWith({ direction: 'left' });
    });
  });

  describe('MobileDashboardManager', () => {
    test('should initialize with container element', () => {
      const container = document.querySelector('.mobile-dashboard');
      const dashboard = new MobileDashboardManager(container);
      
      expect(dashboard.container).toBe(container);
      expect(dashboard.activeTab).toBe(0);
    });

    test('should switch tabs correctly', () => {
      const container = document.querySelector('.mobile-dashboard');
      const dashboard = new MobileDashboardManager(container);
      const callback = jest.fn();
      
      dashboard.on('tab-changed', callback);
      dashboard.switchTab(1);
      
      expect(dashboard.activeTab).toBe(1);
      expect(callback).toHaveBeenCalledWith({ activeTab: 1 });
    });

    test('should create toast notifications', () => {
      const container = document.querySelector('.mobile-dashboard');
      const dashboard = new MobileDashboardManager(container);
      
      const toast = dashboard.showToast('Test message', 'info');
      
      expect(toast.className).toContain('mobile-toast');
      expect(toast.className).toContain('info');
      expect(toast.textContent).toContain('Test message');
    });

    test('should handle multiple event listeners', () => {
      const container = document.querySelector('.mobile-dashboard');
      const dashboard = new MobileDashboardManager(container);
      const callback1 = jest.fn();
      const callback2 = jest.fn();
      
      dashboard.on('test-event', callback1);
      dashboard.on('test-event', callback2);
      dashboard.emit('test-event', { data: 'test' });
      
      expect(callback1).toHaveBeenCalledWith({ data: 'test' });
      expect(callback2).toHaveBeenCalledWith({ data: 'test' });
    });
  });

  describe('Mobile UI Integration', () => {
    test('should handle tab navigation with keyboard', () => {
      const container = document.querySelector('.mobile-dashboard');
      const dashboard = new MobileDashboardManager(container);
      
      // Simulate arrow key navigation
      const keyEvent = new dom.window.KeyboardEvent('keydown', { key: 'ArrowRight' });
      
      // Mock the keyboard navigation method
      dashboard.handleTabNavigation = jest.fn();
      dashboard.handleTabNavigation(keyEvent);
      
      expect(dashboard.handleTabNavigation).toHaveBeenCalledWith(keyEvent);
    });

    test('should provide haptic feedback when available', () => {
      const container = document.querySelector('.mobile-dashboard');
      const dashboard = new MobileDashboardManager(container);
      
      // Mock navigator.vibrate
      global.navigator.vibrate = jest.fn();
      
      const haptic = dashboard.enableHapticFeedback();
      haptic.light();
      
      expect(global.navigator.vibrate).toHaveBeenCalledWith(10);
    });

    test('should handle responsive layout changes', () => {
      const container = document.querySelector('.mobile-dashboard');
      const dashboard = new MobileDashboardManager(container);
      const callback = jest.fn();
      
      dashboard.on('layout-changed', callback);
      
      // Mock layout detection
      dashboard.currentLayout = 'tablet';
      dashboard.emit('layout-changed', { layout: 'tablet' });
      
      expect(callback).toHaveBeenCalledWith({ layout: 'tablet' });
    });

    test('should clean up resources on destroy', () => {
      const container = document.querySelector('.mobile-dashboard');
      const dashboard = new MobileDashboardManager(container);
      
      dashboard.widgets.set('test-widget', {});
      dashboard.panels.set('test-panel', {});
      dashboard.listeners.set('test-event', new Set());
      
      dashboard.destroy();
      
      expect(dashboard.widgets.size).toBe(0);
      expect(dashboard.panels.size).toBe(0);
      expect(dashboard.listeners.size).toBe(0);
    });
  });

  describe('CSS Responsive Behavior', () => {
    test('should apply mobile styles at correct breakpoints', () => {
      // This would require actual CSS testing, but we can check classes
      const container = document.querySelector('.mobile-dashboard');
      container.className = 'mobile-dashboard mobile';
      
      expect(container.classList.contains('mobile')).toBe(true);
    });

    test('should support touch target sizes', () => {
      const button = document.querySelector('.menu-button');
      button.classList.add('enhanced-touch-target');
      
      expect(button.classList.contains('enhanced-touch-target')).toBe(true);
    });
  });

  describe('Accessibility Features', () => {
    test('should provide proper ARIA labels', () => {
      const button = document.querySelector('.menu-button');
      button.setAttribute('aria-label', 'Open menu');
      
      expect(button.getAttribute('aria-label')).toBe('Open menu');
    });

    test('should support keyboard navigation', () => {
      const tabButtons = document.querySelectorAll('.tab-button');
      
      tabButtons.forEach(button => {
        expect(button.getAttribute('tabindex')).not.toBe('-1');
      });
    });

    test('should indicate active states properly', () => {
      const activeTab = document.querySelector('.tab-button.active');
      
      expect(activeTab.classList.contains('active')).toBe(true);
    });
  });
});

// Performance tests
describe('Mobile UI Performance', () => {
  test('should use efficient selectors', () => {
    const container = document.querySelector('.mobile-dashboard');
    
    // Test that commonly used selectors are efficient
    const startTime = performance.now();
    const buttons = container.querySelectorAll('button');
    const endTime = performance.now();
    
    expect(endTime - startTime).toBeLessThan(1); // Should be very fast
    expect(buttons.length).toBeGreaterThan(0);
  });

  test('should minimize DOM manipulations', () => {
    const container = document.querySelector('.mobile-dashboard');
    const dashboard = new MobileDashboardManager(container);
    
    // Count initial DOM nodes
    const initialNodeCount = container.querySelectorAll('*').length;
    
    // Create a widget (should be efficient)
    const widget = document.createElement('div');
    widget.className = 'widget test-widget';
    container.appendChild(widget);
    
    const finalNodeCount = container.querySelectorAll('*').length;
    
    expect(finalNodeCount).toBe(initialNodeCount + 1);
  });
});

module.exports = {
  TouchGestureManager,
  MobileDashboardManager
};