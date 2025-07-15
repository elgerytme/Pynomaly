/**
 * Progressive Enhancement Module for Mobile Features
 * Issue #18: Mobile-Responsive UI Enhancements
 * 
 * Adds advanced mobile features based on device capabilities
 * Gracefully degrades for older devices and browsers
 */

class ProgressiveEnhancementManager {
  constructor() {
    this.capabilities = this.detectCapabilities();
    this.features = new Map();
    
    this.init();
  }

  init() {
    this.enhanceBasedOnCapabilities();
    this.addServiceWorkerSupport();
    this.addOfflineCapabilities();
    this.addInstallPrompt();
    this.addAdvancedGestures();
    this.addPerformanceOptimizations();
  }

  detectCapabilities() {
    return {
      // Device capabilities
      touch: 'ontouchstart' in window,
      vibration: 'vibrate' in navigator,
      orientation: 'orientation' in window,
      deviceMotion: 'DeviceMotionEvent' in window,
      
      // Network capabilities
      connection: 'connection' in navigator,
      onLine: 'onLine' in navigator,
      
      // Storage capabilities
      localStorage: 'localStorage' in window,
      sessionStorage: 'sessionStorage' in window,
      indexedDB: 'indexedDB' in window,
      
      // API capabilities
      serviceWorker: 'serviceWorker' in navigator,
      pushManager: 'PushManager' in window,
      notification: 'Notification' in window,
      geolocation: 'geolocation' in navigator,
      
      // Browser features
      intersectionObserver: 'IntersectionObserver' in window,
      resizeObserver: 'ResizeObserver' in window,
      webGL: this.hasWebGL(),
      webAssembly: typeof WebAssembly === 'object',
      
      // CSS features
      cssGrid: CSS.supports('display', 'grid'),
      cssFlexbox: CSS.supports('display', 'flex'),
      cssCustomProperties: CSS.supports('--test', 'test'),
      
      // Hardware
      hardwareConcurrency: navigator.hardwareConcurrency || 1,
      memory: navigator.deviceMemory || 1,
      maxTouchPoints: navigator.maxTouchPoints || 0
    };
  }

  hasWebGL() {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      return !!gl;
    } catch (e) {
      return false;
    }
  }

  enhanceBasedOnCapabilities() {
    // Add capability classes to body for CSS targeting
    Object.entries(this.capabilities).forEach(([key, supported]) => {
      document.body.classList.add(supported ? `supports-${key}` : `no-${key}`);
    });

    // Add device information
    document.body.classList.add(`cores-${this.capabilities.hardwareConcurrency}`);
    document.body.classList.add(`memory-${Math.floor(this.capabilities.memory)}gb`);
    
    if (this.capabilities.maxTouchPoints > 0) {
      document.body.classList.add(`touch-points-${this.capabilities.maxTouchPoints}`);
    }
  }

  addServiceWorkerSupport() {
    if (!this.capabilities.serviceWorker) return;

    // Register service worker for offline functionality
    navigator.serviceWorker.register('/sw.js')
      .then(registration => {
        console.log('Service Worker registered:', registration);
        this.features.set('serviceWorker', registration);
        
        // Listen for updates
        registration.addEventListener('updatefound', () => {
          this.showUpdateAvailable();
        });
      })
      .catch(error => {
        console.warn('Service Worker registration failed:', error);
      });
  }

  addOfflineCapabilities() {
    if (!this.capabilities.onLine) return;

    // Monitor online/offline status
    window.addEventListener('online', () => {
      this.showConnectedNotification();
      this.syncOfflineData();
    });

    window.addEventListener('offline', () => {
      this.showOfflineNotification();
    });

    // Add offline indicator
    this.createOfflineIndicator();
  }

  addInstallPrompt() {
    if (!this.capabilities.serviceWorker) return;

    let deferredPrompt;
    
    // Listen for install prompt
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      deferredPrompt = e;
      this.showInstallButton(deferredPrompt);
    });

    // Handle successful installation
    window.addEventListener('appinstalled', () => {
      this.hideInstallButton();
      this.showInstalledNotification();
    });
  }

  addAdvancedGestures() {
    if (!this.capabilities.touch) return;

    // Add long press gesture support
    this.addLongPressGesture();
    
    // Add force touch support (if available)
    if ('onforcetouch' in window) {
      this.addForceTouchGesture();
    }

    // Add device orientation support
    if (this.capabilities.orientation) {
      this.addOrientationSupport();
    }

    // Add shake detection
    if (this.capabilities.deviceMotion) {
      this.addShakeDetection();
    }
  }

  addPerformanceOptimizations() {
    // Optimize based on device performance
    const isLowEnd = this.capabilities.hardwareConcurrency < 4 || this.capabilities.memory < 2;
    
    if (isLowEnd) {
      this.enableLowEndOptimizations();
    } else {
      this.enableHighEndFeatures();
    }

    // Monitor performance
    if ('PerformanceObserver' in window) {
      this.monitorPerformance();
    }
  }

  addLongPressGesture() {
    let pressTimer;
    const longPressDelay = 500;

    document.addEventListener('touchstart', (e) => {
      const target = e.target.closest('[data-long-press]');
      if (!target) return;

      pressTimer = setTimeout(() => {
        const event = new CustomEvent('longpress', {
          detail: { target, originalEvent: e },
          bubbles: true
        });
        target.dispatchEvent(event);
        
        if (this.capabilities.vibration) {
          navigator.vibrate(50);
        }
      }, longPressDelay);
    });

    document.addEventListener('touchend', () => {
      clearTimeout(pressTimer);
    });

    document.addEventListener('touchmove', () => {
      clearTimeout(pressTimer);
    });
  }

  addForceTouchGesture() {
    document.addEventListener('touchstart', (e) => {
      if (e.touches[0].force > 0.75) {
        const event = new CustomEvent('forcetouch', {
          detail: { force: e.touches[0].force, originalEvent: e },
          bubbles: true
        });
        e.target.dispatchEvent(event);
      }
    });
  }

  addOrientationSupport() {
    window.addEventListener('orientationchange', () => {
      // Handle orientation changes
      setTimeout(() => {
        this.adjustForOrientation();
      }, 100);
    });

    // Initial orientation setup
    this.adjustForOrientation();
  }

  addShakeDetection() {
    let lastX, lastY, lastZ;
    let lastUpdate = 0;
    const shakeThreshold = 10;

    window.addEventListener('devicemotion', (e) => {
      const current = Date.now();
      if ((current - lastUpdate) > 100) {
        const diffTime = current - lastUpdate;
        lastUpdate = current;

        const x = e.accelerationIncludingGravity.x;
        const y = e.accelerationIncludingGravity.y;
        const z = e.accelerationIncludingGravity.z;

        if (lastX !== undefined) {
          const speed = Math.abs(x + y + z - lastX - lastY - lastZ) / diffTime * 10000;

          if (speed > shakeThreshold) {
            const event = new CustomEvent('shake', {
              detail: { speed, acceleration: e.accelerationIncludingGravity },
              bubbles: true
            });
            document.dispatchEvent(event);
          }
        }

        lastX = x;
        lastY = y;
        lastZ = z;
      }
    });
  }

  enableLowEndOptimizations() {
    document.body.classList.add('low-end-device');
    
    // Reduce animation complexity
    const style = document.createElement('style');
    style.textContent = `
      .low-end-device * {
        animation-duration: 0.2s !important;
        transition-duration: 0.2s !important;
      }
      .low-end-device .chart-container canvas {
        image-rendering: pixelated;
      }
    `;
    document.head.appendChild(style);

    // Disable complex animations
    const complexElements = document.querySelectorAll('.animate-pulse, .animate-spin');
    complexElements.forEach(el => el.classList.remove('animate-pulse', 'animate-spin'));
  }

  enableHighEndFeatures() {
    document.body.classList.add('high-end-device');
    
    // Enable advanced visual effects
    if (this.capabilities.webGL) {
      this.enableWebGLEffects();
    }
    
    // Enable complex animations
    this.enableAdvancedAnimations();
  }

  enableWebGLEffects() {
    // Add WebGL-based chart enhancements
    const charts = document.querySelectorAll('.chart-container');
    charts.forEach(chart => {
      chart.classList.add('webgl-enhanced');
    });
  }

  enableAdvancedAnimations() {
    // Add sophisticated micro-interactions
    const style = document.createElement('style');
    style.textContent = `
      .high-end-device .metric-card {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      }
      .high-end-device .metric-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
      }
    `;
    document.head.appendChild(style);
  }

  monitorPerformance() {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.entryType === 'paint') {
          console.log(`${entry.name}: ${entry.startTime}ms`);
        }
      }
    });

    observer.observe({ entryTypes: ['paint', 'navigation'] });
  }

  adjustForOrientation() {
    const orientation = window.orientation || 0;
    document.body.classList.remove('portrait', 'landscape');
    
    if (Math.abs(orientation) === 90) {
      document.body.classList.add('landscape');
    } else {
      document.body.classList.add('portrait');
    }
  }

  createOfflineIndicator() {
    const indicator = document.createElement('div');
    indicator.id = 'offline-indicator';
    indicator.className = 'fixed top-0 left-0 right-0 bg-red-500 text-white text-center py-2 text-sm z-50 transform -translate-y-full transition-transform duration-300';
    indicator.textContent = 'You are offline. Some features may not be available.';
    document.body.appendChild(indicator);

    if (!navigator.onLine) {
      this.showOfflineNotification();
    }
  }

  showOfflineNotification() {
    const indicator = document.getElementById('offline-indicator');
    if (indicator) {
      indicator.style.transform = 'translateY(0)';
    }
  }

  showConnectedNotification() {
    const indicator = document.getElementById('offline-indicator');
    if (indicator) {
      indicator.style.transform = 'translateY(-100%)';
    }
  }

  showInstallButton(deferredPrompt) {
    const installButton = document.createElement('button');
    installButton.id = 'install-app-button';
    installButton.className = 'fixed bottom-20 right-4 bg-blue-600 text-white px-4 py-2 rounded-lg shadow-lg z-50 text-sm font-medium';
    installButton.textContent = 'Install App';
    
    installButton.addEventListener('click', async () => {
      deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;
      console.log(`User response to install prompt: ${outcome}`);
      
      if (outcome === 'accepted') {
        this.hideInstallButton();
      }
    });

    document.body.appendChild(installButton);
  }

  hideInstallButton() {
    const button = document.getElementById('install-app-button');
    if (button) {
      button.remove();
    }
  }

  showInstalledNotification() {
    const notification = document.createElement('div');
    notification.className = 'fixed top-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg z-50 text-sm';
    notification.textContent = 'App installed successfully!';
    document.body.appendChild(notification);

    setTimeout(() => {
      notification.remove();
    }, 3000);
  }

  showUpdateAvailable() {
    const notification = document.createElement('div');
    notification.className = 'fixed top-4 right-4 bg-blue-500 text-white px-4 py-2 rounded-lg shadow-lg z-50 text-sm';
    notification.innerHTML = `
      App update available! 
      <button onclick="window.location.reload()" class="ml-2 underline">Update</button>
    `;
    document.body.appendChild(notification);
  }

  syncOfflineData() {
    // Sync any offline data when connection is restored
    if (this.capabilities.indexedDB) {
      // Implementation would sync with backend
      console.log('Syncing offline data...');
    }
  }

  // Public API
  getCapabilities() {
    return { ...this.capabilities };
  }

  hasFeature(feature) {
    return this.features.has(feature);
  }

  enableFeature(name, implementation) {
    this.features.set(name, implementation);
  }
}

// Initialize progressive enhancement
document.addEventListener('DOMContentLoaded', () => {
  window.progressiveEnhancement = new ProgressiveEnhancementManager();
  
  // Add global event listeners for custom gestures
  document.addEventListener('longpress', (e) => {
    console.log('Long press detected on:', e.detail.target);
    // Add context menu or additional actions
  });

  document.addEventListener('shake', (e) => {
    console.log('Shake detected with speed:', e.detail.speed);
    // Add refresh or undo functionality
  });

  document.addEventListener('swipeLeft', (e) => {
    console.log('Swipe left detected');
  });

  document.addEventListener('swipeRight', (e) => {
    console.log('Swipe right detected');
  });
});

// Export for external use
window.ProgressiveEnhancementManager = ProgressiveEnhancementManager;