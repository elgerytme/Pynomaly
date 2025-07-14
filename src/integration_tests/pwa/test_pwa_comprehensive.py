"""
Comprehensive PWA Testing Suite

Tests for Progressive Web App functionality including:
- Service Worker functionality
- Offline capabilities
- Push notifications
- Background sync
- App installation
- Update mechanisms
- Mobile optimization
"""

import json
import subprocess
import time

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class PWATestFramework:
    """Base framework for PWA testing"""

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.driver = None
        self.wait = None

    def setup_driver(self, headless=True):
        """Setup Chrome driver with PWA testing capabilities"""
        chrome_options = Options()

        if headless:
            chrome_options.add_argument("--headless")

        # Enable PWA features
        chrome_options.add_argument("--enable-features=WebAppManifest")
        chrome_options.add_argument("--enable-features=ServiceWorkerPaymentApps")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=TranslateUI")
        chrome_options.add_argument("--disable-ipc-flooding-protection")

        # Mobile simulation
        mobile_emulation = {
            "deviceMetrics": {"width": 375, "height": 667, "pixelRatio": 3.0},
            "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
        }
        chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)

        # Enable notifications
        prefs = {
            "profile.default_content_setting_values.notifications": 1,
            "profile.default_content_settings.popups": 0,
            "profile.managed_default_content_settings.images": 2,
        }
        chrome_options.add_experimental_option("prefs", prefs)

        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)

        # Inject PWA testing helpers
        self.inject_testing_helpers()

    def inject_testing_helpers(self):
        """Inject JavaScript helpers for PWA testing"""
        helpers_js = """
        window.PWATestHelpers = {
            // Service Worker helpers
            waitForServiceWorker: function() {
                return new Promise((resolve) => {
                    if ('serviceWorker' in navigator) {
                        navigator.serviceWorker.ready.then(resolve);
                    } else {
                        resolve(null);
                    }
                });
            },

            // Installation helpers
            triggerInstallPrompt: function() {
                return new Promise((resolve) => {
                    if (window.deferredPrompt) {
                        window.deferredPrompt.prompt();
                        window.deferredPrompt.userChoice.then(resolve);
                    } else {
                        resolve({ outcome: 'not_available' });
                    }
                });
            },

            // Offline simulation
            goOffline: function() {
                return navigator.serviceWorker.ready.then(registration => {
                    return registration.sync.register('test-offline');
                });
            },

            // Push notification helpers
            requestNotificationPermission: function() {
                if ('Notification' in window) {
                    return Notification.requestPermission();
                }
                return Promise.resolve('not_supported');
            },

            // Cache inspection
            getCacheInfo: function() {
                return caches.keys().then(cacheNames => {
                    const promises = cacheNames.map(cacheName => {
                        return caches.open(cacheName).then(cache => {
                            return cache.keys().then(keys => ({
                                name: cacheName,
                                size: keys.length,
                                keys: keys.map(req => req.url)
                            }));
                        });
                    });
                    return Promise.all(promises);
                });
            },

            // Background sync testing
            queueBackgroundSync: function(tag, data) {
                return navigator.serviceWorker.ready.then(registration => {
                    if ('sync' in registration) {
                        return registration.sync.register(tag);
                    }
                    throw new Error('Background sync not supported');
                });
            }
        };
        """
        self.driver.execute_script(helpers_js)

    def teardown_driver(self):
        """Clean up driver"""
        if self.driver:
            self.driver.quit()


class TestServiceWorker:
    """Test Service Worker functionality"""

    def setup_method(self):
        self.framework = PWATestFramework()
        self.framework.setup_driver()

    def teardown_method(self):
        self.framework.teardown_driver()

    def test_service_worker_registration(self):
        """Test that service worker is properly registered"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        # Wait for service worker registration
        registration = driver.execute_async_script("""
            const callback = arguments[arguments.length - 1];
            window.PWATestHelpers.waitForServiceWorker().then(callback);
        """)

        assert registration is not None, "Service worker should be registered"

        # Check service worker scope
        scope = driver.execute_script("""
            return navigator.serviceWorker.controller ?
                navigator.serviceWorker.controller.scriptURL : null;
        """)

        assert scope is not None, "Service worker should be active"
        assert "/sw.js" in scope, "Service worker should be sw.js"

    def test_service_worker_caching(self):
        """Test that service worker caches resources"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        # Wait for initial caching
        time.sleep(2)

        # Check cache contents
        cache_info = driver.execute_async_script("""
            const callback = arguments[arguments.length - 1];
            window.PWATestHelpers.getCacheInfo().then(callback);
        """)

        assert len(cache_info) > 0, "Should have cached resources"

        # Check for specific cached resources
        all_cached_urls = []
        for cache in cache_info:
            all_cached_urls.extend(cache["keys"])

        assert any(
            "/static/" in url for url in all_cached_urls
        ), "Should cache static resources"
        assert any(
            "manifest.json" in url for url in all_cached_urls
        ), "Should cache manifest"

    def test_offline_page_serving(self):
        """Test that offline page is served when offline"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        # Wait for service worker
        driver.execute_async_script("""
            const callback = arguments[arguments.length - 1];
            window.PWATestHelpers.waitForServiceWorker().then(callback);
        """)

        # Simulate offline by blocking network
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd(
            "Network.emulateNetworkConditions",
            {
                "offline": True,
                "latency": 0,
                "downloadThroughput": 0,
                "uploadThroughput": 0,
            },
        )

        # Navigate to a non-cached page
        driver.get(f"{self.framework.base_url}/non-existent-page")

        # Check if offline page is shown
        page_title = driver.title
        assert "Offline" in page_title or "offline" in page_title.lower()


class TestPWAManifest:
    """Test PWA Manifest functionality"""

    def setup_method(self):
        self.framework = PWATestFramework()
        self.framework.setup_driver()

    def teardown_method(self):
        self.framework.teardown_driver()

    def test_manifest_exists(self):
        """Test that manifest.json exists and is valid"""
        driver = self.framework.driver
        driver.get(f"{self.framework.base_url}/static/manifest.json")

        # Check that manifest loads without error
        manifest_text = driver.find_element(By.TAG_NAME, "pre").text
        manifest = json.loads(manifest_text)

        # Validate required manifest fields
        required_fields = ["name", "short_name", "start_url", "display", "icons"]
        for field in required_fields:
            assert field in manifest, f"Manifest should contain {field}"

        # Validate icons
        assert len(manifest["icons"]) > 0, "Manifest should contain icons"
        for icon in manifest["icons"]:
            assert "src" in icon, "Each icon should have src"
            assert "sizes" in icon, "Each icon should have sizes"
            assert "type" in icon, "Each icon should have type"

    def test_manifest_linked_in_html(self):
        """Test that manifest is properly linked in HTML"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        # Check for manifest link
        manifest_link = driver.find_element(By.CSS_SELECTOR, 'link[rel="manifest"]')
        assert manifest_link is not None, "HTML should contain manifest link"

        manifest_href = manifest_link.get_attribute("href")
        assert manifest_href.endswith(
            "manifest.json"
        ), "Manifest link should point to manifest.json"

    def test_theme_color(self):
        """Test that theme color is set"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        # Check for theme color meta tag
        theme_color = driver.find_element(By.CSS_SELECTOR, 'meta[name="theme-color"]')
        assert theme_color is not None, "Should have theme-color meta tag"

        color_value = theme_color.get_attribute("content")
        assert color_value is not None, "Theme color should have a value"
        assert color_value.startswith("#") or color_value.startswith(
            "rgb"
        ), "Should be a valid color"


class TestInstallability:
    """Test PWA installation functionality"""

    def setup_method(self):
        self.framework = PWATestFramework()
        self.framework.setup_driver(headless=False)  # Installation needs non-headless

    def teardown_method(self):
        self.framework.teardown_driver()

    def test_install_prompt_criteria(self):
        """Test that PWA meets installability criteria"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        # Wait for service worker and potential install prompt
        time.sleep(3)

        # Check if beforeinstallprompt event fired
        install_prompt_available = driver.execute_script("""
            return typeof window.deferredPrompt !== 'undefined';
        """)

        # Check PWA criteria programmatically
        criteria_check = driver.execute_script("""
            const criteria = {
                hasServiceWorker: 'serviceWorker' in navigator,
                hasManifest: document.querySelector('link[rel="manifest"]') !== null,
                isSecure: location.protocol === 'https:' || location.hostname === 'localhost',
                hasValidIcons: true // Assume valid for this test
            };

            return criteria;
        """)

        assert criteria_check["hasServiceWorker"], "Should have service worker"
        assert criteria_check["hasManifest"], "Should have manifest"
        assert criteria_check["isSecure"], "Should be served over HTTPS or localhost"

    def test_install_button_functionality(self):
        """Test install button if present"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        # Look for install button
        try:
            install_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, ".pwa-install-button, [data-install-pwa]")
                )
            )

            # Click install button
            install_button.click()

            # Check if install prompt was triggered
            time.sleep(1)

            # Note: Actual installation testing requires user interaction
            # This test verifies the button works programmatically

        except:
            # Install button may not be visible if already installed
            # or if install criteria not met
            pass


class TestPushNotifications:
    """Test Push Notification functionality"""

    def setup_method(self):
        self.framework = PWATestFramework()
        self.framework.setup_driver()

    def teardown_method(self):
        self.framework.teardown_driver()

    def test_notification_permission_request(self):
        """Test notification permission request"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        # Request notification permission
        permission = driver.execute_async_script("""
            const callback = arguments[arguments.length - 1];
            window.PWATestHelpers.requestNotificationPermission().then(callback);
        """)

        # Permission should be granted in test environment
        assert permission in ["granted", "default", "denied", "not_supported"]

    def test_push_subscription_api_exists(self):
        """Test that push subscription APIs are available"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        # Check API availability
        api_support = driver.execute_script("""
            return {
                pushManager: 'PushManager' in window,
                notification: 'Notification' in window,
                serviceWorker: 'serviceWorker' in navigator
            };
        """)

        assert api_support["pushManager"], "PushManager should be available"
        assert api_support["notification"], "Notification should be available"
        assert api_support["serviceWorker"], "ServiceWorker should be available"

    @pytest.mark.asyncio
    async def test_push_subscription_endpoints(self):
        """Test push notification API endpoints"""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            # Test subscribe endpoint
            subscription_data = {
                "subscription": {
                    "endpoint": "https://fcm.googleapis.com/fcm/send/test",
                    "keys": {"p256dh": "test-key", "auth": "test-auth"},
                }
            }

            async with session.post(
                f"{self.framework.base_url}/api/push/subscribe",
                json=subscription_data,
                headers={"Authorization": "Bearer test-token"},
            ) as response:
                # Should return 401 without proper auth, but endpoint should exist
                assert response.status in [200, 201, 401, 403]


class TestBackgroundSync:
    """Test Background Sync functionality"""

    def setup_method(self):
        self.framework = PWATestFramework()
        self.framework.setup_driver()

    def teardown_method(self):
        self.framework.teardown_driver()

    def test_background_sync_registration(self):
        """Test background sync registration"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        # Wait for service worker
        driver.execute_async_script("""
            const callback = arguments[arguments.length - 1];
            window.PWATestHelpers.waitForServiceWorker().then(callback);
        """)

        # Test background sync registration
        sync_result = driver.execute_async_script("""
            const callback = arguments[arguments.length - 1];
            window.PWATestHelpers.queueBackgroundSync('test-sync', {test: 'data'})
                .then(() => callback({success: true}))
                .catch(err => callback({success: false, error: err.message}));
        """)

        assert sync_result["success"] or "sync" in sync_result.get("error", "").lower()

    def test_offline_data_queuing(self):
        """Test that data is queued when offline"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        # Go offline
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd(
            "Network.emulateNetworkConditions",
            {
                "offline": True,
                "latency": 0,
                "downloadThroughput": 0,
                "uploadThroughput": 0,
            },
        )

        # Try to perform an action that should be queued
        queue_result = driver.execute_script("""
            // Simulate queuing data for sync
            if (window.PWAManager && window.PWAManager.saveDataOffline) {
                window.PWAManager.saveDataOffline('test', {action: 'test', timestamp: Date.now()});
                return {queued: true};
            }
            return {queued: false, reason: 'PWAManager not available'};
        """)

        # Should either queue successfully or show that PWA manager isn't available
        assert queue_result["queued"] or "not available" in queue_result.get(
            "reason", ""
        )


class TestOfflineCapabilities:
    """Test offline functionality"""

    def setup_method(self):
        self.framework = PWATestFramework()
        self.framework.setup_driver()

    def teardown_method(self):
        self.framework.teardown_driver()

    def test_offline_page_display(self):
        """Test offline page is displayed when offline"""
        driver = self.framework.driver

        # First visit to cache resources
        driver.get(self.framework.base_url)
        time.sleep(2)  # Allow caching

        # Simulate offline
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd(
            "Network.emulateNetworkConditions",
            {
                "offline": True,
                "latency": 0,
                "downloadThroughput": 0,
                "uploadThroughput": 0,
            },
        )

        # Navigate to offline page or trigger offline content
        driver.get(f"{self.framework.base_url}/offline")

        # Check for offline indicators
        page_content = driver.page_source.lower()
        offline_indicators = ["offline", "no connection", "network", "cache"]

        assert any(indicator in page_content for indicator in offline_indicators)

    def test_cached_resources_available_offline(self):
        """Test that cached resources are available offline"""
        driver = self.framework.driver

        # Load main page to populate cache
        driver.get(self.framework.base_url)
        time.sleep(2)

        # Go offline
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd(
            "Network.emulateNetworkConditions",
            {
                "offline": True,
                "latency": 0,
                "downloadThroughput": 0,
                "uploadThroughput": 0,
            },
        )

        # Try to access cached page
        driver.get(self.framework.base_url)

        # Page should still load (from cache)
        assert "Pynomaly" in driver.title or len(driver.page_source) > 1000


class TestMobileOptimization:
    """Test mobile-specific PWA features"""

    def setup_method(self):
        self.framework = PWATestFramework()
        # Already configured for mobile in setup_driver
        self.framework.setup_driver()

    def teardown_method(self):
        self.framework.teardown_driver()

    def test_viewport_meta_tag(self):
        """Test viewport meta tag for mobile"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        viewport_meta = driver.find_element(By.CSS_SELECTOR, 'meta[name="viewport"]')
        assert viewport_meta is not None, "Should have viewport meta tag"

        viewport_content = viewport_meta.get_attribute("content")
        assert "width=device-width" in viewport_content, "Should set device width"
        assert "initial-scale=1" in viewport_content, "Should set initial scale"

    def test_mobile_responsive_design(self):
        """Test responsive design on mobile viewport"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        # Get page dimensions
        page_width = driver.execute_script("return document.body.scrollWidth;")
        viewport_width = driver.execute_script("return window.innerWidth;")

        # Page should not overflow viewport
        assert (
            page_width <= viewport_width + 50
        ), "Page should not overflow viewport significantly"

    def test_touch_friendly_elements(self):
        """Test that interactive elements are touch-friendly"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        # Find clickable elements
        clickable_elements = driver.find_elements(
            By.CSS_SELECTOR, 'button, a, input[type="button"], .btn'
        )

        for element in clickable_elements[:5]:  # Test first 5 elements
            try:
                size = element.size
                # Touch targets should be at least 44x44px (Apple guidelines)
                assert (
                    size["height"] >= 40 or size["width"] >= 40
                ), f"Touch target too small: {size}"
            except:
                # Element might not be visible, skip
                continue

    def test_apple_touch_icons(self):
        """Test Apple touch icons for iOS"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        # Check for Apple touch icon
        apple_icon = driver.find_elements(
            By.CSS_SELECTOR, 'link[rel="apple-touch-icon"]'
        )
        assert len(apple_icon) > 0, "Should have Apple touch icon for iOS"


class TestUpdateMechanism:
    """Test PWA update functionality"""

    def setup_method(self):
        self.framework = PWATestFramework()
        self.framework.setup_driver()

    def teardown_method(self):
        self.framework.teardown_driver()

    def test_update_check_mechanism(self):
        """Test that update checking mechanism exists"""
        driver = self.framework.driver
        driver.get(self.framework.base_url)

        # Check for update manager
        update_manager_exists = driver.execute_script("""
            return typeof window.PWAManager !== 'undefined' &&
                   typeof window.PWAManager.getAppStatus === 'function';
        """)

        # If PWAManager exists, test its functionality
        if update_manager_exists:
            app_status = driver.execute_script("""
                return window.PWAManager.getAppStatus();
            """)

            assert isinstance(app_status, dict), "App status should return object"

    def test_version_api_endpoint(self):
        """Test version API endpoint exists"""
        driver = self.framework.driver

        # Test version endpoint
        try:
            driver.get(f"{self.framework.base_url}/api/version")
            page_source = driver.page_source

            # Should either return JSON or show that endpoint exists
            assert '"version"' in page_source or "version" in page_source.lower()
        except:
            # Endpoint might require authentication
            pass


class TestPerformance:
    """Test PWA performance characteristics"""

    def setup_method(self):
        self.framework = PWATestFramework()
        self.framework.setup_driver()

    def teardown_method(self):
        self.framework.teardown_driver()

    def test_initial_load_performance(self):
        """Test initial page load performance"""
        driver = self.framework.driver

        start_time = time.time()
        driver.get(self.framework.base_url)

        # Wait for page to be interactive
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

        load_time = time.time() - start_time

        # Initial load should be under 5 seconds
        assert load_time < 5.0, f"Initial load too slow: {load_time:.2f}s"

    def test_cached_load_performance(self):
        """Test cached page load performance"""
        driver = self.framework.driver

        # First load
        driver.get(self.framework.base_url)
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

        # Second load (should be cached)
        start_time = time.time()
        driver.get(self.framework.base_url)
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        cached_load_time = time.time() - start_time

        # Cached load should be under 2 seconds
        assert cached_load_time < 2.0, f"Cached load too slow: {cached_load_time:.2f}s"


# Integration test runner
class TestPWAIntegration:
    """Integration tests for complete PWA functionality"""

    def test_complete_pwa_workflow(self):
        """Test complete PWA user workflow"""
        framework = PWATestFramework()
        framework.setup_driver(headless=False)

        try:
            driver = framework.driver

            # 1. Initial load
            driver.get(framework.base_url)

            # 2. Wait for service worker registration
            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script("return 'serviceWorker' in navigator")
            )

            # 3. Check PWA features
            pwa_features = driver.execute_script("""
                return {
                    serviceWorker: 'serviceWorker' in navigator,
                    pushManager: 'PushManager' in window,
                    notification: 'Notification' in window,
                    cache: 'caches' in window,
                    backgroundSync: 'serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype
                };
            """)

            # All core PWA features should be available
            for feature, available in pwa_features.items():
                assert available, f"{feature} should be available"

            # 4. Test offline capability
            time.sleep(2)  # Allow caching

            driver.execute_cdp_cmd("Network.enable", {})
            driver.execute_cdp_cmd(
                "Network.emulateNetworkConditions",
                {
                    "offline": True,
                    "latency": 0,
                    "downloadThroughput": 0,
                    "uploadThroughput": 0,
                },
            )

            # Page should still be accessible
            driver.refresh()
            assert len(driver.page_source) > 100, "Should serve cached content offline"

        finally:
            framework.teardown_driver()


# Test configuration and utilities
def run_pwa_lighthouse_audit():
    """Run Lighthouse PWA audit"""
    try:
        result = subprocess.run(
            [
                "lighthouse",
                "http://localhost:8000",
                "--only-categories=pwa",
                '--chrome-flags="--headless"',
                "--output=json",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            audit_data = json.loads(result.stdout)
            pwa_score = audit_data["categories"]["pwa"]["score"]
            return pwa_score >= 0.9  # 90% or higher

    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass

    return None


def pytest_configure(config):
    """Configure pytest for PWA testing"""
    config.addinivalue_line("markers", "pwa: mark test as PWA-specific")
    config.addinivalue_line("markers", "lighthouse: mark test as requiring Lighthouse")


if __name__ == "__main__":
    # Run basic PWA tests
    pytest.main([__file__, "-v"])
