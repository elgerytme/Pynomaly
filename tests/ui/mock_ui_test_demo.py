#!/usr/bin/env python3
"""
Demo UI test to showcase the testing infrastructure without requiring the full web server.
This creates a simple HTML page and demonstrates UI automation capabilities.
"""

import os
import tempfile
import time
from pathlib import Path


def create_mock_ui():
    """Create a mock Pynomaly UI for testing demonstration."""

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pynomaly - State-of-the-art Anomaly Detection</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        colors: {
                            primary: '#3B82F6',
                            secondary: '#10B981',
                            accent: '#F59E0B'
                        }
                    }
                }
            }
        </script>
        <style>
            .fade-in { animation: fadeIn 0.5s ease-in; }
            @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        </style>
    </head>
    <body class="bg-gray-50 min-h-screen">
        <!-- Navigation -->
        <nav class="bg-white shadow-lg">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div class="flex justify-between h-16">
                    <div class="flex items-center">
                        <a href="#" class="text-2xl font-bold text-primary">
                            🔍 Pynomaly
                        </a>
                    </div>
                    <div class="hidden sm:flex sm:space-x-8 items-center">
                        <a href="#dashboard" class="text-gray-900 border-b-2 border-primary px-3 py-2 text-sm font-medium">Dashboard</a>
                        <a href="#detectors" class="text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium">Detectors</a>
                        <a href="#datasets" class="text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium">Datasets</a>
                        <a href="#detection" class="text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium">Detection</a>
                        <a href="#visualizations" class="text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium">Visualizations</a>
                        <a href="#exports" class="text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium">Exports</a>
                    </div>
                    <!-- Mobile menu button -->
                    <div class="sm:hidden flex items-center">
                        <button id="mobile-menu-btn" class="text-gray-500 hover:text-gray-700">
                            <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
            <!-- Mobile menu -->
            <div id="mobile-menu" class="hidden sm:hidden bg-white border-t border-gray-200">
                <div class="px-2 pt-2 pb-3 space-y-1">
                    <a href="#dashboard" class="block px-3 py-2 text-gray-900 font-medium">Dashboard</a>
                    <a href="#detectors" class="block px-3 py-2 text-gray-500">Detectors</a>
                    <a href="#datasets" class="block px-3 py-2 text-gray-500">Datasets</a>
                    <a href="#detection" class="block px-3 py-2 text-gray-500">Detection</a>
                    <a href="#visualizations" class="block px-3 py-2 text-gray-500">Visualizations</a>
                    <a href="#exports" class="block px-3 py-2 text-gray-500">Exports</a>
                </div>
            </div>
        </nav>

        <!-- Main Content -->
        <main class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
            <!-- Dashboard Header -->
            <div class="fade-in">
                <h1 class="text-3xl font-bold text-gray-900 mb-8">Dashboard</h1>

                <!-- Statistics Cards -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <div class="bg-white overflow-hidden shadow rounded-lg">
                        <div class="p-5">
                            <div class="flex items-center">
                                <div class="flex-shrink-0">
                                    <div class="w-8 h-8 bg-primary rounded-md flex items-center justify-center">
                                        <span class="text-white font-bold">D</span>
                                    </div>
                                </div>
                                <div class="ml-5 w-0 flex-1">
                                    <dl>
                                        <dt class="text-sm font-medium text-gray-500 truncate">Detectors</dt>
                                        <dd class="text-lg font-medium text-gray-900">12</dd>
                                    </dl>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="bg-white overflow-hidden shadow rounded-lg">
                        <div class="p-5">
                            <div class="flex items-center">
                                <div class="flex-shrink-0">
                                    <div class="w-8 h-8 bg-secondary rounded-md flex items-center justify-center">
                                        <span class="text-white font-bold">S</span>
                                    </div>
                                </div>
                                <div class="ml-5 w-0 flex-1">
                                    <dl>
                                        <dt class="text-sm font-medium text-gray-500 truncate">Datasets</dt>
                                        <dd class="text-lg font-medium text-gray-900">8</dd>
                                    </dl>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="bg-white overflow-hidden shadow rounded-lg">
                        <div class="p-5">
                            <div class="flex items-center">
                                <div class="flex-shrink-0">
                                    <div class="w-8 h-8 bg-accent rounded-md flex items-center justify-center">
                                        <span class="text-white font-bold">R</span>
                                    </div>
                                </div>
                                <div class="ml-5 w-0 flex-1">
                                    <dl>
                                        <dt class="text-sm font-medium text-gray-500 truncate">Results</dt>
                                        <dd class="text-lg font-medium text-gray-900">156</dd>
                                    </dl>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Recent Results Table -->
                <div class="bg-white shadow overflow-hidden sm:rounded-md">
                    <div class="px-4 py-5 sm:px-6">
                        <h3 class="text-lg leading-6 font-medium text-gray-900">Recent Detection Results</h3>
                        <p class="mt-1 max-w-2xl text-sm text-gray-500">Latest anomaly detection executions</p>
                    </div>
                    <ul class="divide-y divide-gray-200">
                        <li class="px-4 py-4 hover:bg-gray-50">
                            <div class="flex items-center justify-between">
                                <div class="flex items-center">
                                    <div class="text-sm font-medium text-gray-900">IsolationForest Detection</div>
                                </div>
                                <div class="flex items-center space-x-2">
                                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                                        Completed
                                    </span>
                                    <span class="text-sm text-gray-500">23 anomalies (2.3%)</span>
                                </div>
                            </div>
                            <div class="mt-2 text-sm text-gray-500">
                                <span>Dataset: financial_data.csv</span> •
                                <span>2024-06-24 14:30</span>
                            </div>
                        </li>
                        <li class="px-4 py-4 hover:bg-gray-50">
                            <div class="flex items-center justify-between">
                                <div class="flex items-center">
                                    <div class="text-sm font-medium text-gray-900">LOF Detection</div>
                                </div>
                                <div class="flex items-center space-x-2">
                                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                                        Completed
                                    </span>
                                    <span class="text-sm text-gray-500">15 anomalies (1.5%)</span>
                                </div>
                            </div>
                            <div class="mt-2 text-sm text-gray-500">
                                <span>Dataset: network_logs.csv</span> •
                                <span>2024-06-24 13:45</span>
                            </div>
                        </li>
                        <li class="px-4 py-4 hover:bg-gray-50">
                            <div class="flex items-center justify-between">
                                <div class="flex items-center">
                                    <div class="text-sm font-medium text-gray-900">AutoEncoder Detection</div>
                                </div>
                                <div class="flex items-center space-x-2">
                                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                                        Running
                                    </span>
                                    <span class="text-sm text-gray-500">Processing...</span>
                                </div>
                            </div>
                            <div class="mt-2 text-sm text-gray-500">
                                <span>Dataset: sensor_data.csv</span> •
                                <span>2024-06-24 15:00</span>
                            </div>
                        </li>
                    </ul>
                </div>

                <!-- Quick Actions -->
                <div class="mt-8 flex flex-col sm:flex-row gap-4">
                    <button class="bg-primary text-white px-4 py-2 rounded-md hover:bg-blue-600 transition-colors">
                        🚀 Quick Detection
                    </button>
                    <button class="bg-secondary text-white px-4 py-2 rounded-md hover:bg-green-600 transition-colors">
                        📊 Upload Dataset
                    </button>
                    <button class="bg-accent text-white px-4 py-2 rounded-md hover:bg-yellow-600 transition-colors">
                        🤖 Autonomous Mode
                    </button>
                </div>
            </div>
        </main>

        <!-- Footer -->
        <footer class="bg-white border-t border-gray-200 mt-12">
            <div class="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
                <p class="text-center text-sm text-gray-500">
                    Pynomaly - State-of-the-art Anomaly Detection Platform
                </p>
            </div>
        </footer>

        <script>
            // Mobile menu toggle
            document.getElementById('mobile-menu-btn').addEventListener('click', function() {
                const menu = document.getElementById('mobile-menu');
                menu.classList.toggle('hidden');
            });

            // Simulate real-time updates
            setTimeout(() => {
                const runningStatus = document.querySelector('.bg-yellow-100');
                if (runningStatus) {
                    runningStatus.className = 'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800';
                    runningStatus.textContent = 'Completed';
                    runningStatus.parentElement.querySelector('.text-gray-500').textContent = '42 anomalies (4.2%)';
                }
            }, 3000);
        </script>
    </body>
    </html>
    """

    # Create temporary HTML file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False)
    temp_file.write(html_content)
    temp_file.close()

    return temp_file.name


def run_ui_demo():
    """Run UI automation demo with mock interface."""

    print("🎭 Pynomaly UI Automation Demo")
    print("=" * 50)

    # Create mock UI
    mock_html_path = create_mock_ui()
    file_url = f"file://{mock_html_path}"

    print(f"📄 Mock UI created: {file_url}")

    try:
        # Try to install and use playwright for the demo
        import subprocess
        import sys

        print("📦 Installing Playwright...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "playwright"],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            check=True,
        )

        print("🚀 Running UI automation demo...")

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Navigate to mock UI
            page.goto(file_url)

            # Take screenshots
            screenshots_dir = Path("tests/ui/screenshots")
            screenshots_dir.mkdir(parents=True, exist_ok=True)

            print("📸 Capturing screenshots...")

            # Desktop screenshot
            page.set_viewport_size({"width": 1920, "height": 1080})
            page.screenshot(
                path=str(screenshots_dir / "demo_desktop_dashboard.png"), full_page=True
            )

            # Test mobile menu
            page.set_viewport_size({"width": 375, "height": 667})
            page.screenshot(
                path=str(screenshots_dir / "demo_mobile_dashboard.png"), full_page=True
            )

            # Click mobile menu
            page.click("#mobile-menu-btn")
            page.screenshot(
                path=str(screenshots_dir / "demo_mobile_menu_open.png"), full_page=True
            )

            # Test navigation hover states
            page.set_viewport_size({"width": 1920, "height": 1080})
            page.hover('nav a:has-text("Detectors")')
            page.screenshot(path=str(screenshots_dir / "demo_navigation_hover.png"))

            # Test button interactions
            page.hover('button:has-text("Quick Detection")')
            page.screenshot(path=str(screenshots_dir / "demo_button_hover.png"))

            # Wait for the status update animation
            print("⏳ Waiting for status update animation...")
            time.sleep(4)
            page.screenshot(
                path=str(screenshots_dir / "demo_status_updated.png"), full_page=True
            )

            browser.close()

            print("✅ Demo completed successfully!")
            print(f"📸 Screenshots saved to: {screenshots_dir}")

            # List captured screenshots
            screenshots = list(screenshots_dir.glob("demo_*.png"))
            print(f"📋 Captured {len(screenshots)} screenshots:")
            for screenshot in screenshots:
                print(f"   • {screenshot.name}")

    except ImportError:
        print("⚠️ Playwright not available for demo")
        print("💡 To run the full demo, install Playwright:")
        print("   pip install playwright")
        print("   playwright install")

    except Exception as e:
        print(f"❌ Demo failed: {e}")

    finally:
        # Cleanup
        try:
            os.unlink(mock_html_path)
        except:
            pass


if __name__ == "__main__":
    run_ui_demo()
