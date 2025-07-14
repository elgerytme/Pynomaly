/**
 * Pynomaly Accessibility Enhancement Module
 * Provides comprehensive accessibility features including:
 * - Keyboard navigation
 * - Screen reader support  
 * - High contrast mode
 * - Voice commands
 * - Motor accessibility features
 */

class AccessibilityManager {
  constructor() {
    this.isInitialized = false;
    this.settings = {
      highContrast: false,
      fontSize: 100, // percentage
      reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
      voiceCommands: false,
      keyboardNavigation: true,
      announcements: true,
    };
    
    this.focusStack = [];
    this.modalFocusStack = [];
    this.init();
  }

  init() {
    if (this.isInitialized) return;
    
    this.loadSettings();
    this.initKeyboardNavigation();
    this.initHighContrastMode();
    this.initFontSizeControls();
    this.initModalFocusTrapping();
    this.initLiveRegions();
    this.initAccessibilityToolbar();
    this.initARIAEnhancements();
    
    // Initialize voice commands if supported
    if ('speechSynthesis' in window && 'webkitSpeechRecognition' in window) {
      this.initVoiceCommands();
    }
    
    this.isInitialized = true;
    this.announce('Accessibility features loaded and ready');
  }

  // =============================================
  // KEYBOARD NAVIGATION
  // =============================================
  
  initKeyboardNavigation() {
    // Track keyboard vs mouse usage
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        document.body.classList.add('keyboard-nav-active');
      }
      
      // Handle escape key for modals and dropdowns
      if (e.key === 'Escape') {
        this.handleEscape();
      }
      
      // Handle arrow keys for menu navigation
      if (e.target.closest('[role="menu"], [role="menubar"]')) {
        this.handleMenuNavigation(e);
      }
    });

    document.addEventListener('mousedown', () => {
      document.body.classList.remove('keyboard-nav-active');
    });

    // Add skip links functionality
    this.initSkipLinks();
    
    // Enhanced focus management
    this.initFocusManagement();
  }

  initSkipLinks() {
    const skipLinks = document.querySelectorAll('.skip-link');
    skipLinks.forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href').substring(1);
        const target = document.getElementById(targetId);
        
        if (target) {
          target.focus();
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });
  }

  initFocusManagement() {
    // Enhanced focus indicators
    document.addEventListener('focusin', (e) => {
      if (document.body.classList.contains('keyboard-nav-active')) {
        e.target.classList.add('focus-visible');
      }
    });

    document.addEventListener('focusout', (e) => {
      e.target.classList.remove('focus-visible');
    });
  }

  handleEscape() {
    // Close modals
    const openModal = document.querySelector('.modal.show');
    if (openModal) {
      this.closeModal(openModal);
      return;
    }

    // Close dropdowns
    const openDropdowns = document.querySelectorAll('[aria-expanded="true"]');
    openDropdowns.forEach(dropdown => {
      dropdown.setAttribute('aria-expanded', 'false');
      dropdown.focus();
    });
  }

  handleMenuNavigation(e) {
    const menu = e.target.closest('[role="menu"], [role="menubar"]');
    const items = menu.querySelectorAll('[role="menuitem"]');
    const currentIndex = Array.from(items).indexOf(e.target);

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        const nextIndex = (currentIndex + 1) % items.length;
        items[nextIndex].focus();
        break;
        
      case 'ArrowUp':
        e.preventDefault();
        const prevIndex = (currentIndex - 1 + items.length) % items.length;
        items[prevIndex].focus();
        break;
        
      case 'Home':
        e.preventDefault();
        items[0].focus();
        break;
        
      case 'End':
        e.preventDefault();
        items[items.length - 1].focus();
        break;
    }
  }

  // =============================================
  // HIGH CONTRAST MODE
  // =============================================
  
  initHighContrastMode() {
    const toggle = document.getElementById('high-contrast-toggle');
    if (!toggle) return;

    toggle.addEventListener('click', () => {
      this.toggleHighContrast();
    });

    // Detect system high contrast preference
    const highContrastMedia = window.matchMedia('(prefers-contrast: high)');
    highContrastMedia.addListener(() => {
      this.updateHighContrast(highContrastMedia.matches);
    });

    // Apply saved setting
    if (this.settings.highContrast) {
      this.updateHighContrast(true);
    }
  }

  toggleHighContrast() {
    this.settings.highContrast = !this.settings.highContrast;
    this.updateHighContrast(this.settings.highContrast);
    this.saveSettings();
    
    const message = this.settings.highContrast ? 
      'High contrast mode enabled' : 
      'High contrast mode disabled';
    this.announce(message);
  }

  updateHighContrast(enabled) {
    const toggle = document.getElementById('high-contrast-toggle');
    
    if (enabled) {
      document.documentElement.classList.add('high-contrast');
      toggle?.setAttribute('aria-pressed', 'true');
    } else {
      document.documentElement.classList.remove('high-contrast');
      toggle?.setAttribute('aria-pressed', 'false');
    }
    
    this.settings.highContrast = enabled;
  }

  // =============================================
  // FONT SIZE CONTROLS
  // =============================================
  
  initFontSizeControls() {
    const increaseBtn = document.getElementById('font-size-increase');
    const decreaseBtn = document.getElementById('font-size-decrease');

    increaseBtn?.addEventListener('click', () => {
      this.adjustFontSize(10);
    });

    decreaseBtn?.addEventListener('click', () => {
      this.adjustFontSize(-10);
    });

    // Apply saved font size
    this.updateFontSize(this.settings.fontSize);
  }

  adjustFontSize(delta) {
    const newSize = Math.max(80, Math.min(200, this.settings.fontSize + delta));
    this.updateFontSize(newSize);
    this.saveSettings();
    
    this.announce(`Font size ${delta > 0 ? 'increased' : 'decreased'} to ${newSize}%`);
  }

  updateFontSize(percentage) {
    document.documentElement.style.fontSize = `${percentage}%`;
    this.settings.fontSize = percentage;
  }

  // =============================================
  // MODAL FOCUS TRAPPING
  // =============================================
  
  initModalFocusTrapping() {
    // Listen for modal events
    document.addEventListener('modal:open', (e) => {
      this.trapFocus(e.detail.modal);
    });

    document.addEventListener('modal:close', (e) => {
      this.releaseFocus();
    });
  }

  trapFocus(modal) {
    const focusableElements = modal.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    
    if (focusableElements.length === 0) return;

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    // Store the currently focused element
    this.modalFocusStack.push(document.activeElement);

    // Focus the first element
    firstElement.focus();

    const trapListener = (e) => {
      if (e.key === 'Tab') {
        if (e.shiftKey) {
          // Shift + Tab
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
          }
        } else {
          // Tab
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
          }
        }
      }
    };

    modal.addEventListener('keydown', trapListener);
    modal.trapListener = trapListener; // Store for cleanup
  }

  releaseFocus() {
    const previousFocus = this.modalFocusStack.pop();
    if (previousFocus) {
      previousFocus.focus();
    }
  }

  closeModal(modal) {
    modal.classList.remove('show');
    modal.setAttribute('aria-hidden', 'true');
    
    // Remove focus trap listener
    if (modal.trapListener) {
      modal.removeEventListener('keydown', modal.trapListener);
      delete modal.trapListener;
    }
    
    this.releaseFocus();
    
    // Dispatch close event
    document.dispatchEvent(new CustomEvent('modal:close', {
      detail: { modal }
    }));
  }

  // =============================================
  // LIVE REGIONS AND ANNOUNCEMENTS
  // =============================================
  
  initLiveRegions() {
    // Create live regions if they don't exist
    if (!document.getElementById('live-region-polite')) {
      const politeRegion = document.createElement('div');
      politeRegion.id = 'live-region-polite';
      politeRegion.className = 'sr-only';
      politeRegion.setAttribute('aria-live', 'polite');
      politeRegion.setAttribute('aria-atomic', 'true');
      document.body.appendChild(politeRegion);
    }

    if (!document.getElementById('live-region-assertive')) {
      const assertiveRegion = document.createElement('div');
      assertiveRegion.id = 'live-region-assertive';
      assertiveRegion.className = 'sr-only';
      assertiveRegion.setAttribute('aria-live', 'assertive');
      assertiveRegion.setAttribute('aria-atomic', 'true');
      document.body.appendChild(assertiveRegion);
    }
  }

  announce(message, priority = 'polite') {
    if (!this.settings.announcements) return;

    const regionId = priority === 'assertive' ? 
      'live-region-assertive' : 
      'live-region-polite';
    
    const region = document.getElementById(regionId);
    if (region) {
      // Clear the region first to ensure the announcement is made
      region.textContent = '';
      
      // Use setTimeout to ensure the screen reader notices the change
      setTimeout(() => {
        region.textContent = message;
      }, 100);
      
      // Clear after a delay to prevent accumulation
      setTimeout(() => {
        region.textContent = '';
      }, 5000);
    }
  }

  // =============================================
  // VOICE COMMANDS
  // =============================================
  
  initVoiceCommands() {
    if (!('webkitSpeechRecognition' in window)) return;

    this.recognition = new webkitSpeechRecognition();
    this.recognition.continuous = true;
    this.recognition.interimResults = false;
    this.recognition.lang = 'en-US';

    this.recognition.onresult = (event) => {
      const command = event.results[event.results.length - 1][0].transcript.toLowerCase().trim();
      this.processVoiceCommand(command);
    };

    this.recognition.onerror = (event) => {
      console.warn('Voice recognition error:', event.error);
    };

    const voiceToggle = document.getElementById('voice-commands-toggle');
    voiceToggle?.addEventListener('click', () => {
      this.toggleVoiceCommands();
    });
  }

  toggleVoiceCommands() {
    this.settings.voiceCommands = !this.settings.voiceCommands;
    
    if (this.settings.voiceCommands) {
      this.recognition.start();
      this.announce('Voice commands enabled');
    } else {
      this.recognition.stop();
      this.announce('Voice commands disabled');
    }
    
    const toggle = document.getElementById('voice-commands-toggle');
    toggle?.setAttribute('aria-pressed', this.settings.voiceCommands.toString());
    
    this.saveSettings();
  }

  processVoiceCommand(command) {
    const commands = {
      'go to dashboard': () => window.location.href = '/',
      'go to datasets': () => window.location.href = '/datasets',
      'go to detectors': () => window.location.href = '/detectors',
      'run detection': () => window.location.href = '/detection',
      'toggle high contrast': () => this.toggleHighContrast(),
      'increase font size': () => this.adjustFontSize(10),
      'decrease font size': () => this.adjustFontSize(-10),
      'help': () => this.showVoiceHelp(),
    };

    if (commands[command]) {
      commands[command]();
      this.announce(`Voice command executed: ${command}`);
    } else {
      this.announce('Voice command not recognized. Say "help" for available commands.');
    }
  }

  showVoiceHelp() {
    const helpText = `
      Available voice commands:
      - Go to dashboard
      - Go to datasets
      - Go to detectors
      - Run detection
      - Toggle high contrast
      - Increase font size
      - Decrease font size
      - Help
    `;
    
    this.announce(helpText, 'assertive');
  }

  // =============================================
  // ACCESSIBILITY TOOLBAR
  // =============================================
  
  initAccessibilityToolbar() {
    const toolbar = document.getElementById('accessibility-toolbar');
    if (!toolbar) return;

    // Show toolbar on focus
    toolbar.addEventListener('focusin', () => {
      toolbar.classList.remove('sr-only-focusable');
      toolbar.classList.add('visible');
    });

    toolbar.addEventListener('focusout', (e) => {
      // Hide if focus moves outside toolbar
      if (!toolbar.contains(e.relatedTarget)) {
        toolbar.classList.add('sr-only-focusable');
        toolbar.classList.remove('visible');
      }
    });

    // Keyboard shortcut to show toolbar
    document.addEventListener('keydown', (e) => {
      if (e.altKey && e.key === 'a') {
        e.preventDefault();
        toolbar.querySelector('button').focus();
      }
    });
  }

  // =============================================
  // ARIA ENHANCEMENTS
  // =============================================
  
  initARIAEnhancements() {
    // Auto-add ARIA labels to form controls without labels
    this.enhanceFormLabels();
    
    // Add role and ARIA attributes to dynamic content
    this.enhanceDynamicContent();
    
    // Improve table accessibility
    this.enhanceTableAccessibility();
  }

  enhanceFormLabels() {
    const inputs = document.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
      if (!input.getAttribute('aria-label') && !input.getAttribute('aria-labelledby')) {
        const label = document.querySelector(`label[for="${input.id}"]`);
        if (!label && input.placeholder) {
          input.setAttribute('aria-label', input.placeholder);
        }
      }
    });
  }

  enhanceDynamicContent() {
    // Add ARIA live regions to content that updates dynamically
    const dynamicContainers = document.querySelectorAll('[data-dynamic]');
    dynamicContainers.forEach(container => {
      if (!container.getAttribute('aria-live')) {
        container.setAttribute('aria-live', 'polite');
      }
    });
  }

  enhanceTableAccessibility() {
    const tables = document.querySelectorAll('table');
    tables.forEach(table => {
      // Add scope attributes to headers
      const headers = table.querySelectorAll('th');
      headers.forEach(header => {
        if (!header.getAttribute('scope')) {
          const parent = header.parentElement;
          const thead = table.querySelector('thead');
          const scope = thead && thead.contains(parent) ? 'col' : 'row';
          header.setAttribute('scope', scope);
        }
      });

      // Add table caption if missing
      if (!table.querySelector('caption')) {
        const caption = document.createElement('caption');
        caption.className = 'sr-only';
        caption.textContent = table.getAttribute('aria-label') || 'Data table';
        table.insertBefore(caption, table.firstChild);
      }
    });
  }

  // =============================================
  // SETTINGS PERSISTENCE
  // =============================================
  
  saveSettings() {
    try {
      localStorage.setItem('pynomaly-accessibility-settings', JSON.stringify(this.settings));
    } catch (e) {
      console.warn('Could not save accessibility settings:', e);
    }
  }

  loadSettings() {
    try {
      const saved = localStorage.getItem('pynomaly-accessibility-settings');
      if (saved) {
        this.settings = { ...this.settings, ...JSON.parse(saved) };
      }
    } catch (e) {
      console.warn('Could not load accessibility settings:', e);
    }
  }

  // =============================================
  // PUBLIC API
  // =============================================
  
  // Method to announce messages from other components
  static announce(message, priority = 'polite') {
    if (window.accessibilityManager) {
      window.accessibilityManager.announce(message, priority);
    }
  }

  // Method to check if reduced motion is preferred
  static prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  // Method to get current accessibility settings
  static getSettings() {
    return window.accessibilityManager?.settings || {};
  }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.accessibilityManager = new AccessibilityManager();
  });
} else {
  window.accessibilityManager = new AccessibilityManager();
}

// Export for use in other modules
window.AccessibilityManager = AccessibilityManager;