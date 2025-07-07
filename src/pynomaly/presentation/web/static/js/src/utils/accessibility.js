// Accessibility Utilities
export class AccessibilityManager {
  constructor() {
    this.init();
  }

  init() {
    this.setupFocusManagement();
    this.setupKeyboardNavigation();
    this.setupScreenReaderSupport();
  }

  setupFocusManagement() {
    // Enhanced focus management
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        document.body.classList.add('using-keyboard');
      }
    });

    document.addEventListener('mousedown', () => {
      document.body.classList.remove('using-keyboard');
    });
  }

  setupKeyboardNavigation() {
    // Arrow key navigation for grids
    document.addEventListener('keydown', (e) => {
      const grid = e.target.closest('[role="grid"]');
      if (grid && ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        this.handleGridNavigation(e, grid);
      }
    });
  }

  setupScreenReaderSupport() {
    // Dynamic content announcements
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'childList') {
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType === Node.ELEMENT_NODE && node.hasAttribute('aria-live')) {
              this.announceChange(node);
            }
          });
        }
      });
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }

  handleGridNavigation(event, grid) {
    // Grid navigation implementation
    event.preventDefault();
    // Implementation would handle arrow key navigation
  }

  announceChange(element) {
    // Announce dynamic content changes
    console.log('Content changed:', element.textContent);
  }
}

// Initialize Accessibility Manager
if (typeof window !== 'undefined') {
  window.AccessibilityManager = new AccessibilityManager();
}
