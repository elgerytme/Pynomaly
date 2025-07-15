/**
 * App Update Manager
 *
 * Enhanced update management for PWA with version checking,
 * update notifications, and automatic update handling.
 */

export class AppUpdateManager {
  constructor(options = {}) {
    this.options = {
      checkInterval: 5 * 60 * 1000, // 5 minutes
      enableAutoCheck: true,
      enableBackgroundUpdate: true,
      showUpdatePrompt: true,
      forceUpdate: false,
      skipWaitingOnUpdate: false,
      updateNotificationDuration: 0, // 0 = persistent
      cacheFirst: true,
      enableVersionApi: true,
      versionEndpoint: '/api/version',
      releaseNotesUrl: '/release-notes',
      enableReleaseNotes: true,
      enableLogging: true,
      updateCheckRetries: 3,
      notificationPosition: 'top-right',
      ...options
    };

    this.serviceWorker = null;
    this.currentVersion = null;
    this.latestVersion = null;
    this.isUpdateAvailable = false;
    this.isUpdateReady = false;
    this.updatePrompt = null;
    this.checkTimer = null;
    this.lastUpdateCheck = null;
    this.updateMetadata = null;

    this.init();
  }

  async init() {
    this.log('Initializing App Update Manager...');

    // Get current version
    await this.getCurrentVersion();

    // Setup service worker
    await this.setupServiceWorker();

    // Setup update checking
    if (this.options.enableAutoCheck) {
      this.startUpdateChecking();
    }

    // Setup event listeners
    this.setupEventListeners();

    // Initial update check
    await this.checkForUpdates();

    this.log('App Update Manager initialized');
  }

  /**
   * Service Worker Setup
   */
  async setupServiceWorker() {
    if (!('serviceWorker' in navigator)) {
      this.log('Service Worker not supported');
      return;
    }

    try {
      const registration = await navigator.serviceWorker.getRegistration();
      if (registration) {
        this.serviceWorker = registration;
        this.setupServiceWorkerListeners();
      }

      // Listen for new registrations
      navigator.serviceWorker.addEventListener('controllerchange', () => {
        this.handleControllerChange();
      });

    } catch (error) {
      this.log('Failed to setup service worker:', error);
    }
  }

  setupServiceWorkerListeners() {
    if (!this.serviceWorker) return;

    // Check for waiting service worker
    if (this.serviceWorker.waiting) {
      this.handleUpdateReady();
    }

    // Listen for installing worker
    this.serviceWorker.addEventListener('updatefound', () => {
      const newWorker = this.serviceWorker.installing;
      if (newWorker) {
        this.handleUpdateFound(newWorker);
      }
    });
  }

  handleUpdateFound(newWorker) {
    this.log('New service worker found, installing...');
    this.notifyUpdateStatus('installing');

    newWorker.addEventListener('statechange', () => {
      if (newWorker.state === 'installed') {
        if (navigator.serviceWorker.controller) {
          // New update available
          this.handleUpdateAvailable();
        } else {
          // First install
          this.log('Service worker installed for first time');
        }
      }
    });
  }

  handleUpdateAvailable() {
    this.log('Update available');
    this.isUpdateAvailable = true;
    this.notifyUpdateStatus('available');

    if (this.options.showUpdatePrompt) {
      this.showUpdateNotification();
    }

    if (this.options.enableBackgroundUpdate) {
      // Optionally trigger background update
      setTimeout(() => {
        if (this.isUpdateAvailable) {
          this.applyUpdate();
        }
      }, 10000); // Wait 10 seconds before auto-update
    }
  }

  handleUpdateReady() {
    this.log('Update ready to apply');
    this.isUpdateReady = true;
    this.notifyUpdateStatus('ready');

    if (this.options.forceUpdate || this.options.skipWaitingOnUpdate) {
      this.applyUpdate();
    } else if (this.options.showUpdatePrompt) {
      this.showUpdateReadyNotification();
    }
  }

  handleControllerChange() {
    this.log('Service worker controller changed');
    if (this.isUpdateReady) {
      // Update has been applied
      this.notifyUpdateStatus('applied');
      this.showUpdateCompletedNotification();
    }
  }

  /**
   * Version Management
   */
  async getCurrentVersion() {
    try {
      // Try to get version from meta tag
      const versionMeta = document.querySelector('meta[name="app-version"]');
      if (versionMeta) {
        this.currentVersion = versionMeta.content;
        this.log('Current version from meta:', this.currentVersion);
        return this.currentVersion;
      }

      // Try to get from API
      if (this.options.enableVersionApi) {
        const response = await fetch(this.options.versionEndpoint);
        if (response.ok) {
          const data = await response.json();
          this.currentVersion = data.version || data.app_version;
          this.log('Current version from API:', this.currentVersion);
          return this.currentVersion;
        }
      }

      // Fallback to localStorage
      this.currentVersion = localStorage.getItem('app_version') || 'unknown';
      this.log('Current version from storage:', this.currentVersion);

    } catch (error) {
      this.log('Failed to get current version:', error);
      this.currentVersion = 'unknown';
    }

    return this.currentVersion;
  }

  async getLatestVersion() {
    try {
      let retries = 0;

      while (retries < this.options.updateCheckRetries) {
        try {
          const response = await fetch(`${this.options.versionEndpoint}?cache-bust=${Date.now()}`, {
            method: 'GET',
            headers: {
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache'
            }
          });

          if (response.ok) {
            const data = await response.json();
            this.latestVersion = data.version || data.app_version;
            this.updateMetadata = {
              version: this.latestVersion,
              releaseDate: data.release_date,
              releaseNotes: data.release_notes,
              critical: data.critical,
              size: data.size,
              features: data.features || [],
              bugFixes: data.bug_fixes || [],
              security: data.security_fixes || []
            };

            this.log('Latest version:', this.latestVersion);
            return this.latestVersion;
          }
        } catch (error) {
          retries++;
          if (retries >= this.options.updateCheckRetries) {
            throw error;
          }

          // Wait before retry
          await new Promise(resolve => setTimeout(resolve, 1000 * retries));
        }
      }
    } catch (error) {
      this.log('Failed to get latest version:', error);
      this.latestVersion = this.currentVersion;
    }

    return this.latestVersion;
  }

  compareVersions(current, latest) {
    if (!current || !latest) return false;

    // Simple version comparison (you may want to use a library like semver)
    const currentParts = current.split('.').map(Number);
    const latestParts = latest.split('.').map(Number);

    for (let i = 0; i < Math.max(currentParts.length, latestParts.length); i++) {
      const currentPart = currentParts[i] || 0;
      const latestPart = latestParts[i] || 0;

      if (latestPart > currentPart) {
        return true; // Update available
      } else if (latestPart < currentPart) {
        return false; // Current is newer
      }
    }

    return false; // Same version
  }

  /**
   * Update Checking
   */
  startUpdateChecking() {
    if (this.checkTimer) {
      clearInterval(this.checkTimer);
    }

    this.checkTimer = setInterval(() => {
      this.checkForUpdates();
    }, this.options.checkInterval);

    this.log(`Started automatic update checking every ${this.options.checkInterval / 1000}s`);
  }

  stopUpdateChecking() {
    if (this.checkTimer) {
      clearInterval(this.checkTimer);
      this.checkTimer = null;
      this.log('Stopped automatic update checking');
    }
  }

  async checkForUpdates() {
    this.log('Checking for updates...');
    this.lastUpdateCheck = Date.now();

    try {
      // Check service worker updates
      if (this.serviceWorker) {
        await this.serviceWorker.update();
      }

      // Check version API
      if (this.options.enableVersionApi) {
        await this.getLatestVersion();

        if (this.currentVersion && this.latestVersion) {
          const hasUpdate = this.compareVersions(this.currentVersion, this.latestVersion);

          if (hasUpdate && !this.isUpdateAvailable) {
            this.log(`Version update available: ${this.currentVersion} → ${this.latestVersion}`);
            this.handleUpdateAvailable();
          }
        }
      }

      this.notifyUpdateStatus('checked');

    } catch (error) {
      this.log('Update check failed:', error);
      this.notifyUpdateStatus('check_failed', { error: error.message });
    }
  }

  /**
   * Update Application
   */
  async applyUpdate() {
    this.log('Applying update...');
    this.notifyUpdateStatus('applying');

    try {
      if (this.serviceWorker && this.serviceWorker.waiting) {
        // Tell the waiting service worker to skip waiting
        this.serviceWorker.waiting.postMessage({ type: 'SKIP_WAITING' });
      } else {
        // Force page reload if no waiting worker
        this.reloadApp();
      }
    } catch (error) {
      this.log('Failed to apply update:', error);
      this.notifyUpdateStatus('apply_failed', { error: error.message });
    }
  }

  reloadApp() {
    this.log('Reloading application...');
    window.location.reload();
  }

  /**
   * User Interface
   */
  showUpdateNotification() {
    if (this.updatePrompt) {
      this.hideUpdateNotification();
    }

    const notification = this.createUpdateNotification();
    this.updatePrompt = notification;
    document.body.appendChild(notification);

    // Auto-hide if duration is set
    if (this.options.updateNotificationDuration > 0) {
      setTimeout(() => {
        this.hideUpdateNotification();
      }, this.options.updateNotificationDuration);
    }
  }

  createUpdateNotification() {
    const notification = document.createElement('div');
    notification.className = `app-update-notification ${this.options.notificationPosition}`;

    const isCritical = this.updateMetadata?.critical;
    const hasReleaseNotes = this.options.enableReleaseNotes && this.updateMetadata?.releaseNotes;

    notification.innerHTML = `
      <div class="update-notification-content ${isCritical ? 'critical' : ''}">
        <div class="update-icon">
          ${isCritical ? '🚨' : '📱'}
        </div>
        <div class="update-details">
          <h3 class="update-title">
            ${isCritical ? 'Critical Update Available' : 'App Update Available'}
          </h3>
          <p class="update-message">
            Version ${this.latestVersion} is ready to install.
            ${isCritical ? 'This is a critical security update.' : ''}
          </p>
          ${this.updateMetadata?.features?.length > 0 ? `
            <div class="update-features">
              <strong>New features:</strong>
              <ul>
                ${this.updateMetadata.features.slice(0, 3).map(feature => `<li>${feature}</li>`).join('')}
              </ul>
            </div>
          ` : ''}
        </div>
        <div class="update-actions">
          <button class="update-btn update-primary" onclick="this.closest('.app-update-notification').updateManager.applyUpdate()">
            ${isCritical ? 'Update Now' : 'Update'}
          </button>
          ${!isCritical ? `
            <button class="update-btn update-secondary" onclick="this.closest('.app-update-notification').updateManager.hideUpdateNotification()">
              Later
            </button>
          ` : ''}
          ${hasReleaseNotes ? `
            <button class="update-btn update-link" onclick="this.closest('.app-update-notification').updateManager.showReleaseNotes()">
              What's New
            </button>
          ` : ''}
        </div>
        ${!isCritical ? `
          <button class="update-close" onclick="this.closest('.app-update-notification').updateManager.hideUpdateNotification()">
            ×
          </button>
        ` : ''}
      </div>
    `;

    // Attach reference for button handlers
    notification.updateManager = this;

    return notification;
  }

  showUpdateReadyNotification() {
    const notification = document.createElement('div');
    notification.className = `app-update-ready ${this.options.notificationPosition}`;

    notification.innerHTML = `
      <div class="update-ready-content">
        <div class="update-icon">✅</div>
        <div class="update-details">
          <h3>Update Ready</h3>
          <p>Restart the app to apply the update.</p>
        </div>
        <div class="update-actions">
          <button class="update-btn update-primary" onclick="this.closest('.app-update-ready').updateManager.reloadApp()">
            Restart Now
          </button>
          <button class="update-btn update-secondary" onclick="this.closest('.app-update-ready').updateManager.hideUpdateReadyNotification()">
            Later
          </button>
        </div>
      </div>
    `;

    notification.updateManager = this;
    document.body.appendChild(notification);
    this.updatePrompt = notification;
  }

  showUpdateCompletedNotification() {
    const notification = document.createElement('div');
    notification.className = `app-update-completed ${this.options.notificationPosition}`;

    notification.innerHTML = `
      <div class="update-completed-content">
        <div class="update-icon">🎉</div>
        <div class="update-details">
          <h3>Update Complete</h3>
          <p>Successfully updated to version ${this.latestVersion}</p>
        </div>
        <div class="update-actions">
          <button class="update-btn update-primary" onclick="this.closest('.app-update-completed').remove()">
            Got it
          </button>
        </div>
      </div>
    `;

    document.body.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (notification.parentNode) {
        notification.remove();
      }
    }, 5000);

    // Update stored version
    localStorage.setItem('app_version', this.latestVersion);
    this.currentVersion = this.latestVersion;
  }

  hideUpdateNotification() {
    if (this.updatePrompt && this.updatePrompt.parentNode) {
      this.updatePrompt.remove();
      this.updatePrompt = null;
    }
  }

  hideUpdateReadyNotification() {
    const notification = document.querySelector('.app-update-ready');
    if (notification) {
      notification.remove();
    }
  }

  showReleaseNotes() {
    if (this.updateMetadata?.releaseNotes) {
      // Open release notes in modal or new tab
      const modal = this.createReleaseNotesModal();
      document.body.appendChild(modal);
    } else if (this.options.releaseNotesUrl) {
      window.open(this.options.releaseNotesUrl, '_blank');
    }
  }

  createReleaseNotesModal() {
    const modal = document.createElement('div');
    modal.className = 'release-notes-modal';

    modal.innerHTML = `
      <div class="modal-overlay" onclick="this.closest('.release-notes-modal').remove()"></div>
      <div class="modal-content">
        <div class="modal-header">
          <h2>What's New in ${this.latestVersion}</h2>
          <button class="modal-close" onclick="this.closest('.release-notes-modal').remove()">×</button>
        </div>
        <div class="modal-body">
          <div class="release-content">
            ${this.updateMetadata.features?.length > 0 ? `
              <section>
                <h3>🚀 New Features</h3>
                <ul>
                  ${this.updateMetadata.features.map(feature => `<li>${feature}</li>`).join('')}
                </ul>
              </section>
            ` : ''}

            ${this.updateMetadata.bugFixes?.length > 0 ? `
              <section>
                <h3>🐛 Bug Fixes</h3>
                <ul>
                  ${this.updateMetadata.bugFixes.map(fix => `<li>${fix}</li>`).join('')}
                </ul>
              </section>
            ` : ''}

            ${this.updateMetadata.security?.length > 0 ? `
              <section>
                <h3>🔒 Security Updates</h3>
                <ul>
                  ${this.updateMetadata.security.map(fix => `<li>${fix}</li>`).join('')}
                </ul>
              </section>
            ` : ''}
          </div>
        </div>
        <div class="modal-footer">
          <button class="update-btn update-primary" onclick="this.closest('.release-notes-modal').updateManager.applyUpdate()">
            Update Now
          </button>
          <button class="update-btn update-secondary" onclick="this.closest('.release-notes-modal').remove()">
            Close
          </button>
        </div>
      </div>
    `;

    modal.updateManager = this;
    return modal;
  }

  /**
   * Event Listeners
   */
  setupEventListeners() {
    // Page visibility change
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && this.options.enableAutoCheck) {
        // Check for updates when page becomes visible
        const timeSinceLastCheck = Date.now() - (this.lastUpdateCheck || 0);
        if (timeSinceLastCheck > this.options.checkInterval) {
          this.checkForUpdates();
        }
      }
    });

    // Network status change
    window.addEventListener('online', () => {
      if (this.options.enableAutoCheck) {
        this.checkForUpdates();
      }
    });
  }

  /**
   * Status and Events
   */
  notifyUpdateStatus(status, data = {}) {
    const event = new CustomEvent('app-update', {
      detail: {
        status,
        currentVersion: this.currentVersion,
        latestVersion: this.latestVersion,
        metadata: this.updateMetadata,
        ...data
      }
    });

    window.dispatchEvent(event);
    this.log(`Update status: ${status}`, data);
  }

  /**
   * Public API
   */
  async forceUpdateCheck() {
    this.log('Force checking for updates...');
    return await this.checkForUpdates();
  }

  async updateNow() {
    if (this.isUpdateAvailable || this.isUpdateReady) {
      await this.applyUpdate();
    } else {
      await this.checkForUpdates();
      if (this.isUpdateAvailable) {
        await this.applyUpdate();
      }
    }
  }

  dismissUpdate() {
    this.hideUpdateNotification();
    this.isUpdateAvailable = false;
    this.notifyUpdateStatus('dismissed');
  }

  getUpdateStatus() {
    return {
      currentVersion: this.currentVersion,
      latestVersion: this.latestVersion,
      isUpdateAvailable: this.isUpdateAvailable,
      isUpdateReady: this.isUpdateReady,
      lastUpdateCheck: this.lastUpdateCheck,
      metadata: this.updateMetadata
    };
  }

  /**
   * Utility Methods
   */
  log(...args) {
    if (this.options.enableLogging) {
      console.log('[AppUpdate]', ...args);
    }
  }

  destroy() {
    this.stopUpdateChecking();
    this.hideUpdateNotification();

    // Remove event listeners would go here if needed

    this.log('App Update Manager destroyed');
  }
}

// Global instance
let globalUpdateManager = null;

export function getAppUpdateManager(options = {}) {
  if (!globalUpdateManager) {
    globalUpdateManager = new AppUpdateManager(options);
  }
  return globalUpdateManager;
}

export default AppUpdateManager;
