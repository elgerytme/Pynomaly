/**
 * Lazy Loading Utility for Heavy Components
 * Implements dynamic imports and intersection observer for performance
 */

class LazyLoader {
  constructor() {
    this.loadedComponents = new Set();
    this.observer = null;
    this.initIntersectionObserver();
  }

  initIntersectionObserver() {
    if ('IntersectionObserver' in window) {
      this.observer = new IntersectionObserver(
        (entries) => {
          entries.forEach(entry => {
            if (entry.isIntersecting) {
              this.loadComponent(entry.target);
              this.observer.unobserve(entry.target);
            }
          });
        },
        {
          rootMargin: '50px 0px',
          threshold: 0.1
        }
      );
    }
  }

  async loadComponent(element) {
    const componentName = element.dataset.component;

    if (this.loadedComponents.has(componentName)) {
      return;
    }

    try {
      // Show loading state
      this.showLoadingState(element);

      // Dynamic import based on component type
      let component;
      switch (componentName) {
        case 'chart':
          component = await import('../components/chart-components.js');
          break;
        case 'dashboard':
          component = await import('../components/dashboard-layout.js');
          break;
        case 'visualization':
          component = await import('../components/d3-charts-demo.js');
          break;
        case 'echarts':
          component = await import('../components/echarts-dashboard.js');
          break;
        case 'real-time':
          component = await import('../components/real-time-dashboard.js');
          break;
        default:
          console.warn(`Unknown component: ${componentName}`);
          return;
      }

      // Initialize component
      if (component.default) {
        await component.default.init(element);
      }

      this.loadedComponents.add(componentName);
      this.hideLoadingState(element);

      // Dispatch loaded event
      element.dispatchEvent(new CustomEvent('component-loaded', {
        detail: { componentName }
      }));

    } catch (error) {
      console.error(`Failed to load component ${componentName}:`, error);
      this.showErrorState(element);
    }
  }

  showLoadingState(element) {
    element.innerHTML = `
      <div class="flex items-center justify-center p-8">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span class="ml-2 text-gray-600">Loading...</span>
      </div>
    `;
  }

  hideLoadingState(element) {
    const loadingElement = element.querySelector('.animate-spin');
    if (loadingElement) {
      loadingElement.parentElement.remove();
    }
  }

  showErrorState(element) {
    element.innerHTML = `
      <div class="flex items-center justify-center p-8 text-red-600">
        <svg class="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span>Failed to load component</span>
      </div>
    `;
  }

  observe(element) {
    if (this.observer) {
      this.observer.observe(element);
    } else {
      // Fallback for browsers without IntersectionObserver
      this.loadComponent(element);
    }
  }

  // Preload critical components
  async preloadCritical(componentNames) {
    const loadPromises = componentNames.map(async (componentName) => {
      if (!this.loadedComponents.has(componentName)) {
        try {
          await this.loadComponentModule(componentName);
          this.loadedComponents.add(componentName);
        } catch (error) {
          console.warn(`Failed to preload ${componentName}:`, error);
        }
      }
    });

    await Promise.all(loadPromises);
  }

  async loadComponentModule(componentName) {
    switch (componentName) {
      case 'chart':
        return import('../components/chart-components.js');
      case 'dashboard':
        return import('../components/dashboard-layout.js');
      case 'visualization':
        return import('../components/d3-charts-demo.js');
      case 'echarts':
        return import('../components/echarts-dashboard.js');
      case 'real-time':
        return import('../components/real-time-dashboard.js');
      default:
        throw new Error(`Unknown component: ${componentName}`);
    }
  }
}

// Global lazy loader instance
const lazyLoader = new LazyLoader();

// Initialize lazy loading on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  // Find all lazy-load elements
  const lazyElements = document.querySelectorAll('[data-component]');

  lazyElements.forEach(element => {
    lazyLoader.observe(element);
  });

  // Preload critical components
  lazyLoader.preloadCritical(['dashboard', 'chart']);
});

export default lazyLoader;
