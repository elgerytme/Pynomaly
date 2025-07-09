/**
 * Theme Manager for Dark Mode and Accessibility
 * Handles theme switching, persistence, and accessibility features
 */

class ThemeManager {
  constructor() {
    this.currentTheme = 'light';
    this.themes = {
      light: {
        name: 'Light',
        colors: {
          primary: '#3B82F6',
          secondary: '#64748B',
          background: '#FFFFFF',
          surface: '#F8FAFC',
          text: '#1E293B',
          textSecondary: '#64748B',
          border: '#E2E8F0',
          success: '#10B981',
          warning: '#F59E0B',
          error: '#EF4444',
          info: '#3B82F6'
        }
      },
      dark: {
        name: 'Dark',
        colors: {
          primary: '#60A5FA',
          secondary: '#94A3B8',
          background: '#0F172A',
          surface: '#1E293B',
          text: '#F1F5F9',
          textSecondary: '#94A3B8',
          border: '#334155',
          success: '#34D399',
          warning: '#FBBF24',
          error: '#F87171',
          info: '#60A5FA'
        }
      },
      highContrast: {
        name: 'High Contrast',
        colors: {
          primary: '#000000',
          secondary: '#666666',
          background: '#FFFFFF',
          surface: '#FFFFFF',
          text: '#000000',
          textSecondary: '#000000',
          border: '#000000',
          success: '#008000',
          warning: '#FF8C00',
          error: '#FF0000',
          info: '#0000FF'
        }
      }
    };
    
    this.mediaQueries = {
      prefersColorScheme: window.matchMedia('(prefers-color-scheme: dark)'),
      prefersContrast: window.matchMedia('(prefers-contrast: high)'),
      prefersReducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)')
    };
    
    this.init();
  }

  init() {
    this.loadThemePreference();
    this.setupSystemThemeDetection();
    this.setupThemeToggle();
    this.applyTheme();
    this.setupAccessibilityFeatures();
  }

  loadThemePreference() {
    const savedTheme = localStorage.getItem('pynomaly-theme');
    const systemPrefersDark = this.mediaQueries.prefersColorScheme.matches;
    const systemPrefersHighContrast = this.mediaQueries.prefersContrast.matches;
    
    if (savedTheme && this.themes[savedTheme]) {
      this.currentTheme = savedTheme;
    } else if (systemPrefersHighContrast) {
      this.currentTheme = 'highContrast';
    } else if (systemPrefersDark) {
      this.currentTheme = 'dark';
    } else {
      this.currentTheme = 'light';
    }
  }

  setupSystemThemeDetection() {
    this.mediaQueries.prefersColorScheme.addEventListener('change', (e) => {
      if (!localStorage.getItem('pynomaly-theme')) {
        this.currentTheme = e.matches ? 'dark' : 'light';
        this.applyTheme();
      }
    });

    this.mediaQueries.prefersContrast.addEventListener('change', (e) => {
      if (e.matches) {
        this.setTheme('highContrast');
      } else {
        this.setTheme('light');
      }
    });

    this.mediaQueries.prefersReducedMotion.addEventListener('change', (e) => {
      this.toggleReducedMotion(e.matches);
    });
  }

  setupThemeToggle() {
    // Create theme toggle button
    const themeToggle = document.createElement('button');
    themeToggle.id = 'theme-toggle';
    themeToggle.className = 'theme-toggle-btn';
    themeToggle.setAttribute('aria-label', 'Toggle theme');
    themeToggle.innerHTML = this.getThemeIcon();
    
    // Add to header or create header section
    const header = document.querySelector('header') || document.querySelector('nav');
    if (header) {
      header.appendChild(themeToggle);
    } else {
      document.body.appendChild(themeToggle);
    }

    // Add click handler
    themeToggle.addEventListener('click', () => {
      this.toggleTheme();
    });

    // Add keyboard navigation
    themeToggle.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        this.toggleTheme();
      }
    });
  }

  getThemeIcon() {
    const icons = {
      light: `
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/>
        </svg>
      `,
      dark: `
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/>
        </svg>
      `,
      highContrast: `
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/>
        </svg>
      `
    };
    
    return icons[this.currentTheme] || icons.light;
  }

  toggleTheme() {
    const themes = Object.keys(this.themes);
    const currentIndex = themes.indexOf(this.currentTheme);
    const nextIndex = (currentIndex + 1) % themes.length;
    
    this.setTheme(themes[nextIndex]);
  }

  setTheme(themeName) {
    if (!this.themes[themeName]) {
      console.warn(`Theme "${themeName}" not found`);
      return;
    }

    const previousTheme = this.currentTheme;
    this.currentTheme = themeName;
    
    this.applyTheme();
    this.saveThemePreference();
    this.updateThemeToggle();
    
    // Emit theme change event
    this.emitThemeChangeEvent(previousTheme, themeName);
  }

  applyTheme() {
    const theme = this.themes[this.currentTheme];
    const root = document.documentElement;
    
    // Apply CSS custom properties
    Object.entries(theme.colors).forEach(([key, value]) => {
      root.style.setProperty(`--color-${key}`, value);
    });
    
    // Add theme class to body
    document.body.className = document.body.className.replace(/theme-\w+/g, '');
    document.body.classList.add(`theme-${this.currentTheme}`);
    
    // Update meta theme-color
    let metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (!metaThemeColor) {
      metaThemeColor = document.createElement('meta');
      metaThemeColor.name = 'theme-color';
      document.head.appendChild(metaThemeColor);
    }
    metaThemeColor.content = theme.colors.primary;
    
    // Apply theme-specific styles
    this.applyThemeStyles();
  }

  applyThemeStyles() {
    // Create or update theme stylesheet
    let themeStylesheet = document.getElementById('theme-stylesheet');
    if (!themeStylesheet) {
      themeStylesheet = document.createElement('style');
      themeStylesheet.id = 'theme-stylesheet';
      document.head.appendChild(themeStylesheet);
    }
    
    const theme = this.themes[this.currentTheme];
    const css = this.generateThemeCSS(theme);
    themeStylesheet.textContent = css;
  }

  generateThemeCSS(theme) {
    return `
      /* Theme: ${theme.name} */
      
      .theme-${this.currentTheme} {
        background-color: var(--color-background);
        color: var(--color-text);
        transition: background-color 0.3s ease, color 0.3s ease;
      }
      
      /* Button styles */
      .theme-${this.currentTheme} .btn-primary {
        background-color: var(--color-primary);
        color: var(--color-background);
        border: 1px solid var(--color-primary);
      }
      
      .theme-${this.currentTheme} .btn-secondary {
        background-color: var(--color-secondary);
        color: var(--color-background);
        border: 1px solid var(--color-secondary);
      }
      
      /* Input styles */
      .theme-${this.currentTheme} input,
      .theme-${this.currentTheme} textarea,
      .theme-${this.currentTheme} select {
        background-color: var(--color-surface);
        color: var(--color-text);
        border: 1px solid var(--color-border);
      }
      
      /* Card styles */
      .theme-${this.currentTheme} .card {
        background-color: var(--color-surface);
        border: 1px solid var(--color-border);
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
      }
      
      /* Navigation styles */
      .theme-${this.currentTheme} nav {
        background-color: var(--color-surface);
        border-bottom: 1px solid var(--color-border);
      }
      
      /* Alert styles */
      .theme-${this.currentTheme} .alert-success {
        background-color: var(--color-success);
        color: var(--color-background);
      }
      
      .theme-${this.currentTheme} .alert-warning {
        background-color: var(--color-warning);
        color: var(--color-background);
      }
      
      .theme-${this.currentTheme} .alert-error {
        background-color: var(--color-error);
        color: var(--color-background);
      }
      
      /* Table styles */
      .theme-${this.currentTheme} table {
        background-color: var(--color-surface);
      }
      
      .theme-${this.currentTheme} th {
        background-color: var(--color-background);
        color: var(--color-text);
        border-bottom: 2px solid var(--color-border);
      }
      
      .theme-${this.currentTheme} td {
        border-bottom: 1px solid var(--color-border);
      }
      
      /* Focus styles */
      .theme-${this.currentTheme} *:focus {
        outline: 2px solid var(--color-primary);
        outline-offset: 2px;
      }
      
      /* High contrast specific styles */
      ${this.currentTheme === 'highContrast' ? this.getHighContrastStyles() : ''}
    `;
  }

  getHighContrastStyles() {
    return `
      /* High contrast theme specific styles */
      .theme-highContrast * {
        border-width: 2px !important;
      }
      
      .theme-highContrast button {
        border: 2px solid var(--color-text) !important;
        font-weight: bold;
      }
      
      .theme-highContrast a {
        text-decoration: underline !important;
        font-weight: bold;
      }
      
      .theme-highContrast .card {
        border: 3px solid var(--color-text) !important;
      }
    `;
  }

  updateThemeToggle() {
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
      themeToggle.innerHTML = this.getThemeIcon();
      themeToggle.setAttribute('aria-label', `Current theme: ${this.themes[this.currentTheme].name}`);
    }
  }

  saveThemePreference() {
    localStorage.setItem('pynomaly-theme', this.currentTheme);
  }

  emitThemeChangeEvent(previousTheme, newTheme) {
    const event = new CustomEvent('theme-changed', {
      detail: {
        previousTheme,
        newTheme,
        theme: this.themes[newTheme]
      }
    });
    
    document.dispatchEvent(event);
  }

  setupAccessibilityFeatures() {
    // Setup reduced motion
    this.toggleReducedMotion(this.mediaQueries.prefersReducedMotion.matches);
    
    // Setup focus management
    this.setupFocusManagement();
    
    // Setup keyboard navigation
    this.setupKeyboardNavigation();
  }

  toggleReducedMotion(enabled) {
    const root = document.documentElement;
    
    if (enabled) {
      root.style.setProperty('--animation-duration', '0s');
      root.style.setProperty('--transition-duration', '0s');
      document.body.classList.add('reduced-motion');
    } else {
      root.style.setProperty('--animation-duration', '0.3s');
      root.style.setProperty('--transition-duration', '0.3s');
      document.body.classList.remove('reduced-motion');
    }
  }

  setupFocusManagement() {
    // Trap focus in modals
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        const modal = document.querySelector('.modal.active');
        if (modal) {
          this.trapFocus(modal, e);
        }
      }
    });
  }

  trapFocus(container, event) {
    const focusableElements = container.querySelectorAll(
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
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + Shift + T to toggle theme
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'T') {
        e.preventDefault();
        this.toggleTheme();
      }
      
      // Escape to close modals
      if (e.key === 'Escape') {
        const activeModal = document.querySelector('.modal.active');
        if (activeModal) {
          this.closeModal(activeModal);
        }
      }
    });
  }

  closeModal(modal) {
    modal.classList.remove('active');
    
    // Return focus to trigger element
    const trigger = modal.dataset.trigger;
    if (trigger) {
      const triggerElement = document.querySelector(`[data-modal-trigger="${trigger}"]`);
      if (triggerElement) {
        triggerElement.focus();
      }
    }
  }

  // Public API methods
  getCurrentTheme() {
    return this.currentTheme;
  }

  getAvailableThemes() {
    return Object.keys(this.themes);
  }

  getThemeColors(themeName = this.currentTheme) {
    return this.themes[themeName]?.colors || {};
  }

  // Chart theme integration
  getChartTheme() {
    const colors = this.getThemeColors();
    return {
      backgroundColor: colors.background,
      textColor: colors.text,
      gridColor: colors.border,
      colors: [
        colors.primary,
        colors.secondary,
        colors.success,
        colors.warning,
        colors.error,
        colors.info
      ]
    };
  }

  // Utility methods
  isDarkTheme() {
    return this.currentTheme === 'dark';
  }

  isHighContrastTheme() {
    return this.currentTheme === 'highContrast';
  }

  // CSS-in-JS helper
  getThemeAwareStyles(lightStyles, darkStyles, highContrastStyles) {
    switch (this.currentTheme) {
      case 'dark':
        return darkStyles || lightStyles;
      case 'highContrast':
        return highContrastStyles || lightStyles;
      default:
        return lightStyles;
    }
  }
}

// Global theme manager instance
const themeManager = new ThemeManager();

// Listen for theme changes and update charts
document.addEventListener('theme-changed', (event) => {
  // Update chart themes
  const chartContainers = document.querySelectorAll('[data-chart]');
  chartContainers.forEach(container => {
    const chart = container._chart;
    if (chart && chart.updateTheme) {
      chart.updateTheme(themeManager.getChartTheme());
    }
  });
});

export default themeManager;