/**
 * Progressive Web App Service
 * 
 * Comprehensive PWA service with background sync, push notifications,
 * offline capabilities, and app installation management
 */

export class PWAService {
    constructor(options = {}) {
        this.options = {
            enableServiceWorker: true,
            enablePushNotifications: true,
            enableBackgroundSync: true,
            enablePeriodicBackgroundSync: false,
            swPath: '/sw.js',
            vapidPublicKey: options.vapidPublicKey || null,
            notificationIcon: '/static/icons/notification.png',
            notificationBadge: '/static/icons/badge.png',
            enableLogging: false,
            syncTags: {
                backgroundSync: 'background-sync',
                periodicSync: 'periodic-background-sync'
            },
            ...options
        };
        
        this.serviceWorker = null;
        this.pushSubscription = null;
        this.isOnline = navigator.onLine;
        this.isInstallPromptAvailable = false;
        this.installPrompt = null;
        this.listeners = new Map();
        this.syncQueue = [];
        this.notificationPermission = 'default';
        
        this.init();
    }
    
    async init() {
        this.log('Initializing PWA Service...');
        
        // Check browser support
        this.checkBrowserSupport();
        
        // Register service worker
        if (this.options.enableServiceWorker && 'serviceWorker' in navigator) {
            await this.registerServiceWorker();
        }
        
        // Setup event listeners
        this.bindEvents();
        
        // Initialize push notifications
        if (this.options.enablePushNotifications) {
            await this.initializePushNotifications();
        }
        
        // Setup background sync
        if (this.options.enableBackgroundSync && this.serviceWorker) {
            this.initializeBackgroundSync();
        }
        
        // Check for app install prompt
        this.setupInstallPrompt();
        
        this.log('PWA Service initialized successfully');
    }
    
    checkBrowserSupport() {
        const support = {
            serviceWorker: 'serviceWorker' in navigator,
            pushManager: 'PushManager' in window,
            notifications: 'Notification' in window,
            backgroundSync: 'serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype,
            periodicBackgroundSync: 'serviceWorker' in navigator && 'periodicSync' in window.ServiceWorkerRegistration.prototype,
            indexedDB: 'indexedDB' in window,
            cacheAPI: 'caches' in window
        };
        
        this.log('Browser support:', support);
        this.emit('support_check', support);
        
        return support;
    }
    
    async registerServiceWorker() {
        try {
            this.log('Registering service worker...');
            
            const registration = await navigator.serviceWorker.register(this.options.swPath, {
                scope: '/'
            });
            
            this.serviceWorker = registration;
            
            // Handle service worker updates
            registration.addEventListener('updatefound', () => {
                this.handleServiceWorkerUpdate(registration);
            });
            
            // Check for existing service worker
            if (registration.active) {
                this.log('Service worker already active');
                this.emit('sw_ready', registration);
            }
            
            // Listen for service worker messages
            navigator.serviceWorker.addEventListener('message', (event) => {
                this.handleServiceWorkerMessage(event);
            });
            
            this.log('Service worker registered successfully');
            this.emit('sw_registered', registration);
            
            return registration;
            
        } catch (error) {
            this.log('Service worker registration failed:', error);
            this.emit('sw_registration_failed', error);
            throw error;
        }
    }
    
    handleServiceWorkerUpdate(registration) {
        const newWorker = registration.installing;
        
        newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                // New service worker is available
                this.log('New service worker available');
                this.emit('sw_update_available', newWorker);
                
                this.showUpdateNotification();
            }
        });
    }
    
    showUpdateNotification() {
        const notification = {
            title: 'App Update Available',
            message: 'A new version of the app is available. Refresh to update.',
            actions: [
                { text: 'Update Now', action: 'update' },
                { text: 'Later', action: 'dismiss' }
            ]
        };
        
        this.emit('update_notification', notification);
    }
    
    async updateServiceWorker() {
        if (this.serviceWorker && this.serviceWorker.waiting) {
            this.serviceWorker.waiting.postMessage({ type: 'SKIP_WAITING' });
            window.location.reload();
        }
    }
    
    handleServiceWorkerMessage(event) {
        const { type, payload } = event.data;
        
        switch (type) {
            case 'SYNC_COMPLETE':
                this.handleSyncComplete(payload);
                break;
                
            case 'PUSH_RECEIVED':
                this.handlePushReceived(payload);
                break;
                
            case 'NOTIFICATION_CLICK':
                this.handleNotificationClick(payload);
                break;
                
            case 'CACHE_UPDATE':
                this.handleCacheUpdate(payload);
                break;
                
            case 'ERROR':
                this.handleServiceWorkerError(payload);
                break;
                
            default:
                this.log('Unknown service worker message:', type, payload);
        }
    }
    
    bindEvents() {
        // Network status
        window.addEventListener('online', () => {
            this.isOnline = true;
            this.log('App is online');
            this.emit('online');
            this.processSyncQueue();
        });
        
        window.addEventListener('offline', () => {
            this.isOnline = false;
            this.log('App is offline');
            this.emit('offline');
        });
        
        // App install prompt
        window.addEventListener('beforeinstallprompt', (event) => {
            event.preventDefault();
            this.installPrompt = event;
            this.isInstallPromptAvailable = true;
            this.emit('install_prompt_available');
        });
        
        // App installed
        window.addEventListener('appinstalled', () => {
            this.log('App installed successfully');
            this.installPrompt = null;
            this.isInstallPromptAvailable = false;
            this.emit('app_installed');
        });
        
        // Visibility change
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.isOnline) {
                this.syncInBackground();
            }
        });
    }
    
    // Push Notifications
    async initializePushNotifications() {
        if (!('PushManager' in window) || !('Notification' in window)) {
            this.log('Push notifications not supported');
            return;
        }
        
        this.notificationPermission = Notification.permission;
        
        if (this.notificationPermission === 'granted') {
            await this.subscribeToPush();
        }
        
        this.log('Push notifications initialized');
    }
    
    async requestNotificationPermission() {
        if (!('Notification' in window)) {
            throw new Error('Notifications not supported');
        }
        
        if (this.notificationPermission === 'granted') {
            return 'granted';
        }
        
        const permission = await Notification.requestPermission();
        this.notificationPermission = permission;
        
        if (permission === 'granted') {
            await this.subscribeToPush();
        }
        
        this.emit('notification_permission', permission);
        return permission;
    }
    
    async subscribeToPush() {
        if (!this.serviceWorker || !this.options.vapidPublicKey) {
            this.log('Cannot subscribe to push: missing service worker or VAPID key');
            return;
        }
        
        try {
            const subscription = await this.serviceWorker.pushManager.subscribe({
                userVisibleOnly: true,
                applicationServerKey: this.urlBase64ToUint8Array(this.options.vapidPublicKey)
            });
            
            this.pushSubscription = subscription;
            this.log('Push subscription created:', subscription);
            
            // Send subscription to server
            await this.sendSubscriptionToServer(subscription);
            
            this.emit('push_subscribed', subscription);
            return subscription;
            
        } catch (error) {
            this.log('Push subscription failed:', error);
            this.emit('push_subscription_failed', error);
            throw error;
        }
    }
    
    async sendSubscriptionToServer(subscription) {
        try {
            const response = await fetch('/api/push/subscribe', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    subscription,
                    userAgent: navigator.userAgent,
                    timestamp: Date.now()
                })
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}`);
            }
            
            this.log('Subscription sent to server successfully');
            
        } catch (error) {
            this.log('Failed to send subscription to server:', error);
            throw error;
        }
    }
    
    async unsubscribeFromPush() {
        if (!this.pushSubscription) {
            return;
        }
        
        try {
            await this.pushSubscription.unsubscribe();
            
            // Notify server
            await fetch('/api/push/unsubscribe', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    endpoint: this.pushSubscription.endpoint
                })
            });
            
            this.pushSubscription = null;
            this.log('Unsubscribed from push notifications');
            this.emit('push_unsubscribed');
            
        } catch (error) {
            this.log('Failed to unsubscribe from push:', error);
            throw error;
        }
    }
    
    showLocalNotification(title, options = {}) {
        if (this.notificationPermission !== 'granted') {
            this.log('Cannot show notification: permission not granted');
            return;
        }
        
        const notificationOptions = {
            body: options.body || '',
            icon: options.icon || this.options.notificationIcon,
            badge: options.badge || this.options.notificationBadge,
            tag: options.tag || 'pynomaly-notification',
            data: options.data || {},
            actions: options.actions || [],
            requireInteraction: options.requireInteraction || false,
            silent: options.silent || false,
            ...options
        };
        
        if (this.serviceWorker && this.serviceWorker.active) {
            // Show notification through service worker
            this.serviceWorker.active.postMessage({
                type: 'SHOW_NOTIFICATION',
                payload: { title, options: notificationOptions }
            });
        } else {
            // Fallback to regular notification
            const notification = new Notification(title, notificationOptions);
            
            notification.onclick = () => {
                this.handleNotificationClick({ 
                    notification: { tag: notificationOptions.tag, data: notificationOptions.data },
                    action: 'click'
                });
            };
        }
    }
    
    handlePushReceived(payload) {
        this.log('Push notification received:', payload);
        this.emit('push_received', payload);
    }
    
    handleNotificationClick(payload) {
        this.log('Notification clicked:', payload);
        this.emit('notification_click', payload);
        
        // Focus app window
        if ('clients' in self) {
            // This runs in service worker context
            self.clients.openWindow('/');
        } else {
            window.focus();
        }
    }
    
    // Background Sync
    initializeBackgroundSync() {
        if (!('sync' in window.ServiceWorkerRegistration.prototype)) {
            this.log('Background sync not supported');
            return;
        }
        
        this.log('Background sync initialized');
        this.emit('background_sync_ready');
    }
    
    async scheduleBackgroundSync(tag = null, data = null) {
        if (!this.serviceWorker) {
            this.log('Cannot schedule background sync: no service worker');
            return;
        }
        
        const syncTag = tag || this.options.syncTags.backgroundSync;
        
        try {
            await this.serviceWorker.sync.register(syncTag);
            this.log('Background sync scheduled:', syncTag);
            
            if (data) {
                // Store data for sync
                await this.storeSyncData(syncTag, data);
            }
            
            this.emit('sync_scheduled', { tag: syncTag, data });
            
        } catch (error) {
            this.log('Failed to schedule background sync:', error);
            this.addToSyncQueue({ tag: syncTag, data });
        }
    }
    
    async schedulePeriodicBackgroundSync(tag = null, minInterval = 24 * 60 * 60 * 1000) {
        if (!this.options.enablePeriodicBackgroundSync || 
            !('periodicSync' in window.ServiceWorkerRegistration.prototype)) {
            this.log('Periodic background sync not supported or disabled');
            return;
        }
        
        const syncTag = tag || this.options.syncTags.periodicSync;
        
        try {
            await this.serviceWorker.periodicSync.register(syncTag, {
                minInterval
            });
            
            this.log('Periodic background sync scheduled:', syncTag, minInterval);
            this.emit('periodic_sync_scheduled', { tag: syncTag, minInterval });
            
        } catch (error) {
            this.log('Failed to schedule periodic background sync:', error);
            throw error;
        }
    }
    
    async storeSyncData(tag, data) {
        try {
            if ('indexedDB' in window) {
                // Store in IndexedDB
                const db = await this.openSyncDatabase();
                const transaction = db.transaction(['sync_data'], 'readwrite');
                const store = transaction.objectStore('sync_data');
                
                await store.put({
                    tag,
                    data,
                    timestamp: Date.now()
                });
                
                this.log('Sync data stored:', tag);
            } else {
                // Fallback to localStorage
                localStorage.setItem(`sync_data_${tag}`, JSON.stringify({
                    data,
                    timestamp: Date.now()
                }));
            }
        } catch (error) {
            this.log('Failed to store sync data:', error);
        }
    }
    
    async openSyncDatabase() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open('PWASyncDB', 1);
            
            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve(request.result);
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains('sync_data')) {
                    const store = db.createObjectStore('sync_data', { keyPath: 'tag' });
                    store.createIndex('timestamp', 'timestamp');
                }
            };
        });
    }
    
    addToSyncQueue(syncData) {
        this.syncQueue.push({
            ...syncData,
            timestamp: Date.now()
        });
        
        this.log('Added to sync queue:', syncData);
    }
    
    async processSyncQueue() {
        if (!this.isOnline || this.syncQueue.length === 0) {
            return;
        }
        
        this.log('Processing sync queue:', this.syncQueue.length, 'items');
        
        const queue = [...this.syncQueue];
        this.syncQueue = [];
        
        for (const item of queue) {
            try {
                await this.scheduleBackgroundSync(item.tag, item.data);
            } catch (error) {
                this.log('Failed to process sync queue item:', error);
                this.syncQueue.push(item); // Re-add failed items
            }
        }
    }
    
    handleSyncComplete(payload) {
        this.log('Background sync completed:', payload);
        this.emit('sync_complete', payload);
    }
    
    async syncInBackground() {
        if (!this.isOnline || !this.serviceWorker) {
            return;
        }
        
        this.log('Syncing in background...');
        
        // Schedule sync for various data types
        await this.scheduleBackgroundSync('anomaly_data');
        await this.scheduleBackgroundSync('user_preferences');
        await this.scheduleBackgroundSync('offline_actions');
        
        this.emit('background_sync_initiated');
    }
    
    // App Installation
    setupInstallPrompt() {
        this.log('Setting up install prompt...');
    }
    
    async promptAppInstall() {
        if (!this.isInstallPromptAvailable || !this.installPrompt) {
            throw new Error('Install prompt not available');
        }
        
        try {
            const result = await this.installPrompt.prompt();
            this.log('Install prompt result:', result);
            
            const choiceResult = await result.userChoice;
            this.log('User choice:', choiceResult);
            
            if (choiceResult.outcome === 'accepted') {
                this.emit('install_accepted');
            } else {
                this.emit('install_dismissed');
            }
            
            this.installPrompt = null;
            this.isInstallPromptAvailable = false;
            
            return choiceResult;
            
        } catch (error) {
            this.log('Install prompt failed:', error);
            this.emit('install_failed', error);
            throw error;
        }
    }
    
    isAppInstalled() {
        return window.matchMedia('(display-mode: standalone)').matches ||
               window.navigator.standalone === true;
    }
    
    // Cache Management
    async clearCache(cacheName = null) {
        if (!('caches' in window)) {
            this.log('Cache API not supported');
            return;
        }
        
        try {
            if (cacheName) {
                await caches.delete(cacheName);
                this.log('Cache cleared:', cacheName);
            } else {
                const cacheNames = await caches.keys();
                await Promise.all(cacheNames.map(name => caches.delete(name)));
                this.log('All caches cleared');
            }
            
            this.emit('cache_cleared', { cacheName });
            
        } catch (error) {
            this.log('Failed to clear cache:', error);
            throw error;
        }
    }
    
    async getCacheSize() {
        if (!('caches' in window) || !('storage' in navigator) || !('estimate' in navigator.storage)) {
            return null;
        }
        
        try {
            const estimate = await navigator.storage.estimate();
            return {
                usage: estimate.usage,
                quota: estimate.quota,
                usageDetails: estimate.usageDetails
            };
        } catch (error) {
            this.log('Failed to get cache size:', error);
            return null;
        }
    }
    
    handleCacheUpdate(payload) {
        this.log('Cache updated:', payload);
        this.emit('cache_updated', payload);
    }
    
    // Offline Support
    async enableOfflineMode() {
        if (!this.serviceWorker) {
            throw new Error('Service worker required for offline mode');
        }
        
        this.serviceWorker.active.postMessage({
            type: 'ENABLE_OFFLINE_MODE'
        });
        
        this.log('Offline mode enabled');
        this.emit('offline_mode_enabled');
    }
    
    async disableOfflineMode() {
        if (!this.serviceWorker) {
            return;
        }
        
        this.serviceWorker.active.postMessage({
            type: 'DISABLE_OFFLINE_MODE'
        });
        
        this.log('Offline mode disabled');
        this.emit('offline_mode_disabled');
    }
    
    // Utility methods
    urlBase64ToUint8Array(base64String) {
        const padding = '='.repeat((4 - base64String.length % 4) % 4);
        const base64 = (base64String + padding)
            .replace(/-/g, '+')
            .replace(/_/g, '/');
            
        const rawData = window.atob(base64);
        const outputArray = new Uint8Array(rawData.length);
        
        for (let i = 0; i < rawData.length; ++i) {
            outputArray[i] = rawData.charCodeAt(i);
        }
        
        return outputArray;
    }
    
    handleServiceWorkerError(payload) {
        this.log('Service worker error:', payload);
        this.emit('sw_error', payload);
    }
    
    // Status and Information
    getStatus() {
        return {
            isOnline: this.isOnline,
            isInstalled: this.isAppInstalled(),
            isInstallPromptAvailable: this.isInstallPromptAvailable,
            serviceWorkerRegistered: !!this.serviceWorker,
            pushSubscribed: !!this.pushSubscription,
            notificationPermission: this.notificationPermission,
            syncQueueLength: this.syncQueue.length
        };
    }
    
    async getCapabilities() {
        const support = this.checkBrowserSupport();
        const cacheSize = await this.getCacheSize();
        
        return {
            ...support,
            cacheSize,
            isOnline: this.isOnline,
            isInstalled: this.isAppInstalled()
        };
    }
    
    // Event management
    on(event, listener) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(listener);
        
        return () => this.off(event, listener);
    }
    
    off(event, listener) {
        const eventListeners = this.listeners.get(event);
        if (eventListeners) {
            eventListeners.delete(listener);
            if (eventListeners.size === 0) {
                this.listeners.delete(event);
            }
        }
    }
    
    emit(event, data) {
        const eventListeners = this.listeners.get(event);
        if (eventListeners) {
            eventListeners.forEach(listener => {
                try {
                    listener(data);
                } catch (error) {
                    this.log('Error in event listener:', error);
                }
            });
        }
    }
    
    log(...args) {
        if (this.options.enableLogging) {
            console.log('[PWAService]', ...args);
        }
    }
    
    // Cleanup
    destroy() {
        this.listeners.clear();
        this.syncQueue = [];
        
        if (this.pushSubscription) {
            this.unsubscribeFromPush().catch(() => {});
        }
        
        // Remove event listeners
        window.removeEventListener('online', this.handleOnline);
        window.removeEventListener('offline', this.handleOffline);
        window.removeEventListener('beforeinstallprompt', this.handleBeforeInstallPrompt);
        window.removeEventListener('appinstalled', this.handleAppInstalled);
        document.removeEventListener('visibilitychange', this.handleVisibilityChange);
    }
}

/**
 * PWA Installation Helper Component
 */
export class PWAInstallPrompt {
    constructor(pwaService, options = {}) {
        this.pwaService = pwaService;
        this.options = {
            showBanner: true,
            showButton: true,
            bannerDismissible: true,
            buttonText: 'Install App',
            bannerText: 'Install this app for a better experience',
            position: 'bottom',
            ...options
        };
        
        this.isVisible = false;
        this.banner = null;
        this.button = null;
        
        this.init();
    }
    
    init() {
        this.pwaService.on('install_prompt_available', () => {
            this.show();
        });
        
        this.pwaService.on('app_installed', () => {
            this.hide();
        });
        
        if (this.options.showBanner) {
            this.createBanner();
        }
        
        if (this.options.showButton) {
            this.createButton();
        }
    }
    
    createBanner() {
        this.banner = document.createElement('div');
        this.banner.className = `pwa-install-banner pwa-banner-${this.options.position}`;
        this.banner.style.display = 'none';
        this.banner.innerHTML = `
            <div class="pwa-banner-content">
                <div class="pwa-banner-icon">📱</div>
                <div class="pwa-banner-text">${this.options.bannerText}</div>
                <div class="pwa-banner-actions">
                    <button class="pwa-install-btn">Install</button>
                    ${this.options.bannerDismissible ? '<button class="pwa-dismiss-btn">×</button>' : ''}
                </div>
            </div>
        `;
        
        // Bind events
        this.banner.querySelector('.pwa-install-btn').onclick = () => {
            this.install();
        };
        
        if (this.options.bannerDismissible) {
            this.banner.querySelector('.pwa-dismiss-btn').onclick = () => {
                this.hide();
            };
        }
        
        document.body.appendChild(this.banner);
    }
    
    createButton() {
        this.button = document.createElement('button');
        this.button.className = 'pwa-install-button';
        this.button.textContent = this.options.buttonText;
        this.button.style.display = 'none';
        this.button.onclick = () => this.install();
        
        // You can append this button to any container
        // For example: document.querySelector('.header-actions').appendChild(this.button);
    }
    
    show() {
        if (this.pwaService.isAppInstalled()) {
            return;
        }
        
        this.isVisible = true;
        
        if (this.banner) {
            this.banner.style.display = 'block';
            setTimeout(() => {
                this.banner.classList.add('pwa-banner-visible');
            }, 100);
        }
        
        if (this.button) {
            this.button.style.display = 'inline-block';
        }
    }
    
    hide() {
        this.isVisible = false;
        
        if (this.banner) {
            this.banner.classList.remove('pwa-banner-visible');
            setTimeout(() => {
                this.banner.style.display = 'none';
            }, 300);
        }
        
        if (this.button) {
            this.button.style.display = 'none';
        }
    }
    
    async install() {
        try {
            await this.pwaService.promptAppInstall();
        } catch (error) {
            console.error('Installation failed:', error);
        }
    }
    
    getButton() {
        return this.button;
    }
    
    destroy() {
        if (this.banner && this.banner.parentNode) {
            this.banner.parentNode.removeChild(this.banner);
        }
        
        if (this.button && this.button.parentNode) {
            this.button.parentNode.removeChild(this.button);
        }
    }
}

export default PWAService;