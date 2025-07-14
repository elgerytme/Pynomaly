/**
 * Enhanced Mobile Touch Interactions for Pynomaly Dashboard
 * Issue #18: Mobile-Responsive UI Enhancements
 * 
 * Comprehensive touch gesture system with swipe, pinch, tap, and pan support
 * Optimized for anomaly detection dashboard interactions
 */

class EnhancedTouchGestureManager {
  constructor() {
    this.isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    this.activeGestures = new Map();
    this.swipeThreshold = 50;
    this.tapTimeout = 300;
    this.longTapTimeout = 500;
    this.doubleTapTimeout = 300;
    this.pinchThreshold = 10;
    
    this.init();
  }

  init() {
    if (!this.isTouch) return;
    
    this.setupPullToRefresh();
    this.setupSwipeNavigation();
    this.setupTouchOptimizations();
    this.setupChartInteractions();
    this.setupTableSwipe();
    this.setupModalGestures();
  }

  // Pull to refresh functionality
  setupPullToRefresh() {
    let startY = 0;
    let currentY = 0;
    let isPulling = false;
    let refreshThreshold = 60;

    const refreshContainer = document.querySelector('.pull-to-refresh');
    if (!refreshContainer) return;

    const indicator = document.createElement('div');
    indicator.className = 'pull-to-refresh-indicator';
    indicator.innerHTML = `
      <svg class="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
      </svg>
    `;
    refreshContainer.appendChild(indicator);

    refreshContainer.addEventListener('touchstart', (e) => {
      if (refreshContainer.scrollTop === 0) {
        startY = e.touches[0].clientY;
        isPulling = true;
      }
    });

    refreshContainer.addEventListener('touchmove', (e) => {
      if (!isPulling) return;
      
      currentY = e.touches[0].clientY;
      const pullDistance = currentY - startY;
      
      if (pullDistance > 0) {
        e.preventDefault();
        const opacity = Math.min(pullDistance / refreshThreshold, 1);
        indicator.style.opacity = opacity;
        
        if (pullDistance > refreshThreshold) {
          refreshContainer.classList.add('pulling');
        } else {
          refreshContainer.classList.remove('pulling');
        }
      }
    });

    refreshContainer.addEventListener('touchend', (e) => {
      if (!isPulling) return;
      
      const pullDistance = currentY - startY;
      isPulling = false;
      
      if (pullDistance > refreshThreshold) {
        this.triggerRefresh();
      }
      
      refreshContainer.classList.remove('pulling');
      indicator.style.opacity = 0;
    });
  }

  // Swipe navigation between tabs/pages
  setupSwipeNavigation() {
    const swipeElements = document.querySelectorAll('.swipe-navigation');
    
    swipeElements.forEach(element => {
      let startX = 0;
      let startY = 0;
      let currentX = 0;
      let currentY = 0;
      let isSwipping = false;

      element.addEventListener('touchstart', (e) => {
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;
        isSwipping = true;
      });

      element.addEventListener('touchmove', (e) => {
        if (!isSwipping) return;
        
        currentX = e.touches[0].clientX;
        currentY = e.touches[0].clientY;
        
        const deltaX = currentX - startX;
        const deltaY = currentY - startY;
        
        // Horizontal swipe
        if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > this.swipeThreshold) {
          e.preventDefault();
          
          if (deltaX > 0) {
            this.handleSwipeRight(element);
          } else {
            this.handleSwipeLeft(element);
          }
          
          isSwipping = false;
        }
      });

      element.addEventListener('touchend', () => {
        isSwipping = false;
      });
    });
  }

  // Enhanced touch targets and haptic feedback
  setupTouchOptimizations() {
    // Add touch feedback to buttons
    const touchElements = document.querySelectorAll('.btn, .mobile-touch-target, button, [role="button"]');
    
    touchElements.forEach(element => {
      element.addEventListener('touchstart', () => {
        element.style.transform = 'scale(0.95)';
        element.style.transition = 'transform 0.1s ease';
        
        // Haptic feedback (if supported)
        if (navigator.vibrate) {
          navigator.vibrate(10);
        }
      });

      element.addEventListener('touchend', () => {
        setTimeout(() => {
          element.style.transform = '';
        }, 100);
      });

      element.addEventListener('touchcancel', () => {
        element.style.transform = '';
      });
    });

    // Enhanced tap handling with double-tap prevention
    this.setupDoubleTapPrevention();
  }

  // Chart and visualization touch interactions
  setupChartInteractions() {
    const charts = document.querySelectorAll('.chart-container');
    
    charts.forEach(chart => {
      let initialDistance = 0;
      let currentScale = 1;
      let isPinching = false;

      chart.addEventListener('touchstart', (e) => {
        if (e.touches.length === 2) {
          isPinching = true;
          initialDistance = this.getDistance(e.touches[0], e.touches[1]);
          chart.style.transition = 'none';
        }
      });

      chart.addEventListener('touchmove', (e) => {
        if (isPinching && e.touches.length === 2) {
          e.preventDefault();
          
          const currentDistance = this.getDistance(e.touches[0], e.touches[1]);
          const scale = currentDistance / initialDistance;
          currentScale = Math.max(0.5, Math.min(3, scale));
          
          chart.style.transform = `scale(${currentScale})`;
        }
      });

      chart.addEventListener('touchend', (e) => {
        if (isPinching) {
          isPinching = false;
          chart.style.transition = 'transform 0.3s ease';
          
          // Snap to reasonable scale values
          if (currentScale < 0.8) {
            currentScale = 0.5;
          } else if (currentScale > 2.5) {
            currentScale = 3;
          } else if (currentScale > 0.8 && currentScale < 1.2) {
            currentScale = 1;
          }
          
          chart.style.transform = `scale(${currentScale})`;
        }
      });
    });
  }

  // Table horizontal swipe for mobile
  setupTableSwipe() {
    const tables = document.querySelectorAll('.table-container');
    
    tables.forEach(container => {
      let startX = 0;
      let scrollLeft = 0;
      let isScrolling = false;

      container.addEventListener('touchstart', (e) => {
        startX = e.touches[0].clientX;
        scrollLeft = container.scrollLeft;
        isScrolling = true;
        container.style.scrollBehavior = 'auto';
      });

      container.addEventListener('touchmove', (e) => {
        if (!isScrolling) return;
        
        e.preventDefault();
        const x = e.touches[0].clientX;
        const walk = (x - startX) * 2;
        container.scrollLeft = scrollLeft - walk;
      });

      container.addEventListener('touchend', () => {
        isScrolling = false;
        container.style.scrollBehavior = 'smooth';
      });
    });
  }

  // Modal swipe-to-dismiss
  setupModalGestures() {
    const modals = document.querySelectorAll('.modal');
    
    modals.forEach(modal => {
      const modalContent = modal.querySelector('.modal-content');
      if (!modalContent) return;

      let startY = 0;
      let currentY = 0;
      let isDragging = false;

      modalContent.addEventListener('touchstart', (e) => {
        startY = e.touches[0].clientY;
        isDragging = true;
        modalContent.style.transition = 'none';
      });

      modalContent.addEventListener('touchmove', (e) => {
        if (!isDragging) return;
        
        currentY = e.touches[0].clientY;
        const deltaY = currentY - startY;
        
        if (deltaY > 0) {
          modalContent.style.transform = `translateY(${deltaY}px)`;
        }
      });

      modalContent.addEventListener('touchend', () => {
        if (!isDragging) return;
        
        const deltaY = currentY - startY;
        isDragging = false;
        
        modalContent.style.transition = 'transform 0.3s ease';
        
        if (deltaY > 100) {
          // Dismiss modal
          modalContent.style.transform = 'translateY(100%)';
          setTimeout(() => {
            modal.classList.remove('open');
            modalContent.style.transform = '';
          }, 300);
        } else {
          // Snap back
          modalContent.style.transform = '';
        }
      });
    });
  }

  // Prevent accidental double-tap zoom
  setupDoubleTapPrevention() {
    let lastTouchEnd = 0;
    
    document.addEventListener('touchend', (e) => {
      const now = new Date().getTime();
      if (now - lastTouchEnd <= 300) {
        e.preventDefault();
      }
      lastTouchEnd = now;
    }, false);
  }

  // Utility functions
  getDistance(touch1, touch2) {
    const dx = touch1.clientX - touch2.clientX;
    const dy = touch1.clientY - touch2.clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }

  handleSwipeRight(element) {
    // Navigate to previous tab/page
    const event = new CustomEvent('swipeRight', { 
      detail: { element },
      bubbles: true 
    });
    element.dispatchEvent(event);
    
    // Default behavior for tabs
    const tabContainer = element.closest('.tab-container');
    if (tabContainer) {
      this.navigateTabs(tabContainer, 'prev');
    }
  }

  handleSwipeLeft(element) {
    // Navigate to next tab/page
    const event = new CustomEvent('swipeLeft', { 
      detail: { element },
      bubbles: true 
    });
    element.dispatchEvent(event);
    
    // Default behavior for tabs
    const tabContainer = element.closest('.tab-container');
    if (tabContainer) {
      this.navigateTabs(tabContainer, 'next');
    }
  }

  navigateTabs(container, direction) {
    const tabs = container.querySelectorAll('.tab-button');
    const activeTab = container.querySelector('.tab-button.active');
    
    if (!activeTab) return;
    
    const currentIndex = Array.from(tabs).indexOf(activeTab);
    let newIndex;
    
    if (direction === 'next') {
      newIndex = (currentIndex + 1) % tabs.length;
    } else {
      newIndex = (currentIndex - 1 + tabs.length) % tabs.length;
    }
    
    tabs[newIndex].click();
  }

  triggerRefresh() {
    // Trigger refresh action
    const event = new CustomEvent('pullToRefresh', { bubbles: true });
    document.dispatchEvent(event);
    
    // Default refresh behavior
    if (typeof window.refreshDashboard === 'function') {
      window.refreshDashboard();
    } else {
      // Fallback: reload page after delay
      setTimeout(() => {
        window.location.reload();
      }, 1000);
    }
  }
}

// Enhanced mobile keyboard handling
class MobileKeyboardManager {
  constructor() {
    this.isKeyboardOpen = false;
    this.viewportHeight = window.innerHeight;
    
    this.init();
  }

  init() {
    // Handle viewport changes for virtual keyboard
    window.addEventListener('resize', () => {
      const currentHeight = window.innerHeight;
      const heightDifference = this.viewportHeight - currentHeight;
      
      if (heightDifference > 150) {
        // Keyboard likely opened
        this.handleKeyboardOpen();
      } else {
        // Keyboard likely closed
        this.handleKeyboardClose();
      }
    });

    // Smooth scroll to focused inputs
    document.addEventListener('focusin', (e) => {
      if (e.target.matches('input, textarea, select')) {
        setTimeout(() => {
          e.target.scrollIntoView({
            behavior: 'smooth',
            block: 'center'
          });
        }, 300);
      }
    });

    // Handle Enter key on mobile
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.target.matches('input[type="search"], input[type="text"]')) {
        e.target.blur();
      }
    });
  }

  handleKeyboardOpen() {
    this.isKeyboardOpen = true;
    document.body.classList.add('keyboard-open');
    
    // Adjust fixed elements
    const fixedElements = document.querySelectorAll('.fab, .mobile-nav-header');
    fixedElements.forEach(el => {
      el.style.display = 'none';
    });
  }

  handleKeyboardClose() {
    this.isKeyboardOpen = false;
    document.body.classList.remove('keyboard-open');
    
    // Restore fixed elements
    const fixedElements = document.querySelectorAll('.fab, .mobile-nav-header');
    fixedElements.forEach(el => {
      el.style.display = '';
    });
  }
}

// Performance optimizations for mobile
class MobilePerformanceOptimizer {
  constructor() {
    this.init();
  }

  init() {
    // Passive touch listeners for better performance
    this.setupPassiveListeners();
    
    // Optimize animations
    this.optimizeAnimations();
    
    // Lazy load images
    this.setupLazyLoading();
    
    // Optimize scroll performance
    this.optimizeScrolling();
  }

  setupPassiveListeners() {
    // Add passive listeners to improve scroll performance
    const passiveEvents = ['touchstart', 'touchmove', 'wheel'];
    
    passiveEvents.forEach(eventName => {
      document.addEventListener(eventName, () => {}, { passive: true });
    });
  }

  optimizeAnimations() {
    // Reduce animations on slower devices
    if (navigator.hardwareConcurrency < 4) {
      document.documentElement.style.setProperty('--mobile-transition', 'all 0.1s ease');
    }
    
    // Disable animations if reduced motion is preferred
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      const style = document.createElement('style');
      style.textContent = `
        *, *::before, *::after {
          animation-duration: 0.01ms !important;
          animation-iteration-count: 1 !important;
          transition-duration: 0.01ms !important;
        }
      `;
      document.head.appendChild(style);
    }
  }

  setupLazyLoading() {
    // Lazy load images using Intersection Observer
    const images = document.querySelectorAll('img[data-src]');
    
    if ('IntersectionObserver' in window) {
      const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src;
            img.classList.remove('lazy');
            imageObserver.unobserve(img);
          }
        });
      });

      images.forEach(img => imageObserver.observe(img));
    } else {
      // Fallback for older browsers
      images.forEach(img => {
        img.src = img.dataset.src;
      });
    }
  }

  optimizeScrolling() {
    // Use passive scrolling where possible
    let ticking = false;
    
    const updateScrollElements = () => {
      // Update scroll-dependent elements
      const scrollTop = window.pageYOffset;
      
      // Update header background opacity
      const header = document.querySelector('.mobile-nav-header');
      if (header) {
        const opacity = Math.min(scrollTop / 100, 1);
        header.style.backgroundColor = `rgba(255, 255, 255, ${opacity})`;
      }
      
      ticking = false;
    };

    window.addEventListener('scroll', () => {
      if (!ticking) {
        requestAnimationFrame(updateScrollElements);
        ticking = true;
      }
    }, { passive: true });
  }
}

// Initialize mobile enhancements when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  // Only initialize on mobile devices
  if (window.innerWidth <= 1024) {
    new EnhancedTouchGestureManager();
    new MobileKeyboardManager();
    new MobilePerformanceOptimizer();
    
    console.log('Mobile touch enhancements initialized');
  }
});

// Export for use in other modules
window.EnhancedTouchGestureManager = EnhancedTouchGestureManager;
window.MobileKeyboardManager = MobileKeyboardManager;
window.MobilePerformanceOptimizer = MobilePerformanceOptimizer;