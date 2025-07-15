// Enhanced Accessibility Utilities
export class AccessibilityManager {
  constructor() {
    this.preferences = this.loadPreferences();
    this.voiceCommands = new VoiceCommandManager();
    this.highContrastMode = false;
    this.reducedMotion = false;
    this.keyboardShortcuts = new Map();
    this.init();
  }

  init() {
    this.setupFocusManagement();
    this.setupKeyboardNavigation();
    this.setupScreenReaderSupport();
    this.setupVoiceCommands();
    this.setupHighContrastMode();
    this.setupMotorAccessibility();
    this.setupKeyboardShortcuts();
    this.applyUserPreferences();
  }

  loadPreferences() {
    try {
      const stored = localStorage.getItem('pynomaly-accessibility-preferences');
      return stored ? JSON.parse(stored) : {
        highContrast: false,
        reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
        fontSize: 'normal',
        keyboardShortcuts: true,
        voiceCommands: false,
        announcements: true
      };
    } catch {
      return {};
    }
  }

  savePreferences() {
    try {
      localStorage.setItem('pynomaly-accessibility-preferences', JSON.stringify(this.preferences));
    } catch (error) {
      console.warn('Could not save accessibility preferences:', error);
    }
  }

  setupFocusManagement() {
    // Enhanced focus management with trap support
    document.addEventListener("keydown", (e) => {
      if (e.key === "Tab") {
        document.body.classList.add("using-keyboard");
      }

      // Handle Escape key for modal dismissal
      if (e.key === "Escape") {
        this.handleEscapeKey(e);
      }
    });

    document.addEventListener("mousedown", () => {
      document.body.classList.remove("using-keyboard");
    });

    // Focus trap for modals
    this.setupFocusTrapping();
  }

  setupFocusTrapping() {
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        const modal = document.querySelector('[role="dialog"][aria-modal="true"]');
        if (modal) {
          this.trapFocusInModal(e, modal);
        }
      }
    });
  }

  trapFocusInModal(event, modal) {
    const focusableElements = modal.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    if (event.shiftKey) {
      if (document.activeElement === firstElement) {
        lastElement.focus();
        event.preventDefault();
      }
    } else {
      if (document.activeElement === lastElement) {
        firstElement.focus();
        event.preventDefault();
      }
    }
  }

  setupKeyboardNavigation() {
    // Enhanced keyboard navigation
    document.addEventListener("keydown", (e) => {
      // Grid navigation
      const grid = e.target.closest('[role="grid"]');
      if (grid && ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) {
        this.handleGridNavigation(e, grid);
        return;
      }

      // Skip links (Alt + S)
      if (e.altKey && e.key === 's') {
        this.showSkipLinks();
        e.preventDefault();
        return;
      }

      // Handle global keyboard shortcuts
      this.handleKeyboardShortcuts(e);
    });
  }

  setupKeyboardShortcuts() {
    // Register global keyboard shortcuts
    this.keyboardShortcuts.set('Alt+d', () => this.focusDashboard());
    this.keyboardShortcuts.set('Alt+n', () => this.focusNavigation());
    this.keyboardShortcuts.set('Alt+m', () => this.focusMainContent());
    this.keyboardShortcuts.set('Alt+h', () => this.showKeyboardHelp());
    this.keyboardShortcuts.set('Alt+c', () => this.toggleHighContrast());
    this.keyboardShortcuts.set('Alt+v', () => this.toggleVoiceCommands());
  }

  handleKeyboardShortcuts(event) {
    if (!this.preferences.keyboardShortcuts) return;

    const combo = [];
    if (event.ctrlKey) combo.push('Ctrl');
    if (event.altKey) combo.push('Alt');
    if (event.shiftKey) combo.push('Shift');
    combo.push(event.key);

    const shortcut = combo.join('+');
    const handler = this.keyboardShortcuts.get(shortcut);
    if (handler) {
      handler();
      event.preventDefault();
    }
  }

  setupScreenReaderSupport() {
    // Enhanced screen reader support with better announcements
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === "childList") {
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              // Announce live region changes
              if (node.hasAttribute("aria-live")) {
                this.announceChange(node);
              }

              // Announce new content with headings
              const newHeadings = node.querySelectorAll('h1, h2, h3, h4, h5, h6');
              newHeadings.forEach(heading => this.announceNewSection(heading));
            }
          });
        }
      });
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });

    // Setup live regions for dynamic content
    this.createLiveRegions();
  }

  createLiveRegions() {
    // Create polite live region for status updates
    if (!document.getElementById('aria-live-polite')) {
      const politeRegion = document.createElement('div');
      politeRegion.id = 'aria-live-polite';
      politeRegion.setAttribute('aria-live', 'polite');
      politeRegion.setAttribute('aria-atomic', 'true');
      politeRegion.className = 'sr-only';
      document.body.appendChild(politeRegion);
    }

    // Create assertive live region for urgent updates
    if (!document.getElementById('aria-live-assertive')) {
      const assertiveRegion = document.createElement('div');
      assertiveRegion.id = 'aria-live-assertive';
      assertiveRegion.setAttribute('aria-live', 'assertive');
      assertiveRegion.setAttribute('aria-atomic', 'true');
      assertiveRegion.className = 'sr-only';
      document.body.appendChild(assertiveRegion);
    }
  }

  setupVoiceCommands() {
    // Initialize voice command support
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      this.voiceCommands.init();
    }
  }

  setupHighContrastMode() {
    // High contrast mode support
    const mediaQuery = window.matchMedia('(prefers-contrast: high)');
    this.handleContrastChange(mediaQuery);
    mediaQuery.addEventListener('change', (e) => this.handleContrastChange(e));
  }

  setupMotorAccessibility() {
    // Enhanced motor accessibility features
    this.setupLargerTouchTargets();
    this.setupStickyHover();
    this.setupAdjustableTiming();
  }

  handleGridNavigation(event, grid) {
    event.preventDefault();
    const currentCell = event.target.closest('[role="gridcell"]');
    if (!currentCell) return;

    const row = currentCell.closest('[role="row"]');
    const allRows = Array.from(grid.querySelectorAll('[role="row"]'));
    const allCellsInRow = Array.from(row.querySelectorAll('[role="gridcell"]'));

    const rowIndex = allRows.indexOf(row);
    const cellIndex = allCellsInRow.indexOf(currentCell);

    let targetCell = null;

    switch (event.key) {
      case 'ArrowUp':
        if (rowIndex > 0) {
          const targetRow = allRows[rowIndex - 1];
          const targetCells = targetRow.querySelectorAll('[role="gridcell"]');
          targetCell = targetCells[Math.min(cellIndex, targetCells.length - 1)];
        }
        break;
      case 'ArrowDown':
        if (rowIndex < allRows.length - 1) {
          const targetRow = allRows[rowIndex + 1];
          const targetCells = targetRow.querySelectorAll('[role="gridcell"]');
          targetCell = targetCells[Math.min(cellIndex, targetCells.length - 1)];
        }
        break;
      case 'ArrowLeft':
        if (cellIndex > 0) {
          targetCell = allCellsInRow[cellIndex - 1];
        }
        break;
      case 'ArrowRight':
        if (cellIndex < allCellsInRow.length - 1) {
          targetCell = allCellsInRow[cellIndex + 1];
        }
        break;
    }

    if (targetCell) {
      targetCell.focus();
      this.announceGridPosition(targetCell, allRows, allCellsInRow);
    }
  }

  announceChange(element) {
    if (!this.preferences.announcements) return;

    const message = element.textContent.trim();
    if (message) {
      this.announce(message, element.getAttribute('aria-live') === 'assertive');
    }
  }

  announce(message, urgent = false) {
    const regionId = urgent ? 'aria-live-assertive' : 'aria-live-polite';
    const region = document.getElementById(regionId);
    if (region) {
      region.textContent = message;
      // Clear after announcement to allow repeat announcements
      setTimeout(() => { region.textContent = ''; }, 1000);
    }
  }

  announceGridPosition(cell, allRows, allCellsInRow) {
    const row = cell.closest('[role="row"]');
    const rowIndex = allRows.indexOf(row) + 1;
    const cellIndex = Array.from(row.querySelectorAll('[role="gridcell"]')).indexOf(cell) + 1;
    const totalRows = allRows.length;
    const totalCells = allCellsInRow.length;

    this.announce(`Row ${rowIndex} of ${totalRows}, Column ${cellIndex} of ${totalCells}. ${cell.textContent.trim()}`);
  }

  announceNewSection(heading) {
    const level = heading.tagName.toLowerCase();
    const text = heading.textContent.trim();
    this.announce(`New ${level}: ${text}`);
  }

  handleEscapeKey(event) {
    // Close modals, dropdowns, etc.
    const modal = document.querySelector('[role="dialog"][aria-modal="true"]');
    if (modal) {
      const closeButton = modal.querySelector('[data-dismiss="modal"], .modal-close, .close');
      if (closeButton) {
        closeButton.click();
      }
    }

    // Close dropdowns
    const openDropdowns = document.querySelectorAll('[aria-expanded="true"]');
    openDropdowns.forEach(dropdown => {
      dropdown.setAttribute('aria-expanded', 'false');
    });
  }

  showSkipLinks() {
    const skipLinks = document.querySelector('.skip-links');
    if (skipLinks) {
      skipLinks.classList.add('visible');
      const firstLink = skipLinks.querySelector('a');
      if (firstLink) firstLink.focus();
    }
  }

  focusDashboard() {
    const dashboard = document.querySelector('[data-page="dashboard"]') || document.querySelector('#dashboard');
    if (dashboard) {
      dashboard.focus();
      this.announce('Dashboard focused');
    }
  }

  focusNavigation() {
    const nav = document.querySelector('nav[role="navigation"]') || document.querySelector('#navigation');
    if (nav) {
      const firstLink = nav.querySelector('a, button');
      if (firstLink) {
        firstLink.focus();
        this.announce('Navigation focused');
      }
    }
  }

  focusMainContent() {
    const main = document.querySelector('main') || document.querySelector('#main-content');
    if (main) {
      main.focus();
      this.announce('Main content focused');
    }
  }

  showKeyboardHelp() {
    // Show keyboard shortcuts help modal
    const existingModal = document.getElementById('keyboard-help-modal');
    if (existingModal) {
      existingModal.style.display = 'block';
      existingModal.focus();
    } else {
      this.createKeyboardHelpModal();
    }
  }

  createKeyboardHelpModal() {
    const modal = document.createElement('div');
    modal.id = 'keyboard-help-modal';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.setAttribute('aria-labelledby', 'keyboard-help-title');
    modal.className = 'modal keyboard-help-modal';

    modal.innerHTML = `
      <div class="modal-content">
        <div class="modal-header">
          <h2 id="keyboard-help-title">Keyboard Shortcuts</h2>
          <button class="modal-close" aria-label="Close keyboard help">&times;</button>
        </div>
        <div class="modal-body">
          <dl class="keyboard-shortcuts">
            <dt>Alt + D</dt><dd>Focus Dashboard</dd>
            <dt>Alt + N</dt><dd>Focus Navigation</dd>
            <dt>Alt + M</dt><dd>Focus Main Content</dd>
            <dt>Alt + S</dt><dd>Show Skip Links</dd>
            <dt>Alt + C</dt><dd>Toggle High Contrast</dd>
            <dt>Alt + V</dt><dd>Toggle Voice Commands</dd>
            <dt>Alt + H</dt><dd>Show This Help</dd>
            <dt>Escape</dt><dd>Close Modals/Dropdowns</dd>
            <dt>Arrow Keys</dt><dd>Navigate Data Tables</dd>
            <dt>Tab</dt><dd>Navigate Focusable Elements</dd>
          </dl>
        </div>
      </div>
    `;

    document.body.appendChild(modal);
    modal.style.display = 'block';
    modal.focus();

    // Close button handler
    modal.querySelector('.modal-close').addEventListener('click', () => {
      modal.style.display = 'none';
      modal.remove();
    });
  }

  toggleHighContrast() {
    this.preferences.highContrast = !this.preferences.highContrast;
    this.applyHighContrast();
    this.savePreferences();
    this.announce(`High contrast ${this.preferences.highContrast ? 'enabled' : 'disabled'}`);
  }

  toggleVoiceCommands() {
    this.preferences.voiceCommands = !this.preferences.voiceCommands;
    if (this.preferences.voiceCommands) {
      this.voiceCommands.start();
    } else {
      this.voiceCommands.stop();
    }
    this.savePreferences();
    this.announce(`Voice commands ${this.preferences.voiceCommands ? 'enabled' : 'disabled'}`);
  }

  handleContrastChange(mediaQuery) {
    if (mediaQuery.matches && !this.preferences.highContrast) {
      this.preferences.highContrast = true;
      this.applyHighContrast();
    }
  }

  applyHighContrast() {
    document.body.classList.toggle('high-contrast', this.preferences.highContrast);
  }

  applyUserPreferences() {
    // Apply all saved preferences
    this.applyHighContrast();

    if (this.preferences.reducedMotion) {
      document.body.classList.add('reduced-motion');
    }

    if (this.preferences.fontSize !== 'normal') {
      document.body.classList.add(`font-size-${this.preferences.fontSize}`);
    }
  }

  setupLargerTouchTargets() {
    // Add touch-friendly class on touch devices
    if ('ontouchstart' in window) {
      document.body.classList.add('touch-device');
    }
  }

  setupStickyHover() {
    // Implement sticky hover for better motor accessibility
    let stickyHoverTimeout;

    document.addEventListener('mouseenter', (e) => {
      if (e.target.matches('button, a, [role="button"]')) {
        clearTimeout(stickyHoverTimeout);
        e.target.classList.add('sticky-hover');
      }
    }, true);

    document.addEventListener('mouseleave', (e) => {
      if (e.target.matches('button, a, [role="button"]')) {
        stickyHoverTimeout = setTimeout(() => {
          e.target.classList.remove('sticky-hover');
        }, 300); // 300ms delay for sticky hover
      }
    }, true);
  }

  setupAdjustableTiming() {
    // Extend timeouts for users who need more time
    const originalTimeout = window.setTimeout;
    window.setTimeout = function(callback, delay, ...args) {
      // Increase timeout by 50% for accessibility
      const adjustedDelay = delay * 1.5;
      return originalTimeout.call(this, callback, adjustedDelay, ...args);
    };
  }

  // Public API for external control
  setPreference(key, value) {
    this.preferences[key] = value;
    this.savePreferences();
    this.applyUserPreferences();
  }

  getPreference(key) {
    return this.preferences[key];
  }
}

// Voice Command Manager
class VoiceCommandManager {
  constructor() {
    this.recognition = null;
    this.isListening = false;
    this.commands = new Map();
    this.setupCommands();
  }

  init() {
    if ('webkitSpeechRecognition' in window) {
      this.recognition = new webkitSpeechRecognition();
    } else if ('SpeechRecognition' in window) {
      this.recognition = new SpeechRecognition();
    }

    if (this.recognition) {
      this.recognition.continuous = true;
      this.recognition.interimResults = false;
      this.recognition.lang = 'en-US';

      this.recognition.onresult = (event) => {
        this.handleVoiceCommand(event);
      };

      this.recognition.onerror = (event) => {
        console.warn('Voice recognition error:', event.error);
      };
    }
  }

  setupCommands() {
    this.commands.set('go to dashboard', () => this.navigateTo('/dashboard'));
    this.commands.set('go to datasets', () => this.navigateTo('/datasets'));
    this.commands.set('go to detection', () => this.navigateTo('/detection'));
    this.commands.set('show help', () => window.AccessibilityManager.showKeyboardHelp());
    this.commands.set('high contrast on', () => window.AccessibilityManager.setPreference('highContrast', true));
    this.commands.set('high contrast off', () => window.AccessibilityManager.setPreference('highContrast', false));
    this.commands.set('stop listening', () => this.stop());
  }

  start() {
    if (this.recognition && !this.isListening) {
      this.recognition.start();
      this.isListening = true;
      if (window.AccessibilityManager) {
        window.AccessibilityManager.announce('Voice commands activated');
      }
    }
  }

  stop() {
    if (this.recognition && this.isListening) {
      this.recognition.stop();
      this.isListening = false;
      if (window.AccessibilityManager) {
        window.AccessibilityManager.announce('Voice commands deactivated');
      }
    }
  }

  handleVoiceCommand(event) {
    const command = event.results[event.results.length - 1][0].transcript.toLowerCase().trim();
    const handler = this.commands.get(command);

    if (handler) {
      handler();
      if (window.AccessibilityManager) {
        window.AccessibilityManager.announce(`Command executed: ${command}`);
      }
    } else {
      if (window.AccessibilityManager) {
        window.AccessibilityManager.announce('Command not recognized');
      }
    }
  }

  navigateTo(path) {
    if (window.location.pathname !== path) {
      window.location.href = path;
    }
  }
}

// Initialize Accessibility Manager
if (typeof window !== "undefined") {
  window.AccessibilityManager = new AccessibilityManager();
}
