// PWA Manager - Install prompts and update handling
export class PWAManager {
  constructor() {
    this.deferredPrompt = null;
    this.isInstalled = false;
    
    this.init();
  }

  init() {
    this.setupInstallPrompt();
    this.setupUpdateHandling();
    this.checkInstallStatus();
  }

  setupInstallPrompt() {
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      this.deferredPrompt = e;
      this.showInstallButton();
    });
  }

  setupUpdateHandling() {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.addEventListener('controllerchange', () => {
        window.location.reload();
      });
    }
  }

  checkInstallStatus() {
    if (window.matchMedia('(display-mode: standalone)').matches) {
      this.isInstalled = true;
    }
  }

  showInstallButton() {
    // Show install prompt button
    console.log('PWA install available');
  }

  async installPWA() {
    if (this.deferredPrompt) {
      this.deferredPrompt.prompt();
      const result = await this.deferredPrompt.userChoice;
      
      if (result.outcome === 'accepted') {
        console.log('PWA installed');
      }
      
      this.deferredPrompt = null;
    }
  }
}

// Initialize PWA Manager
if (typeof window !== 'undefined') {
  window.PWAManager = new PWAManager();
}