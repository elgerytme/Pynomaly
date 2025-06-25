#!/usr/bin/env python3
"""
Simple working FastAPI server for UI testing.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Pynomaly UI Test Server", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pynomaly - State-of-the-art Anomaly Detection</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-50 min-h-screen">
        <!-- Navigation -->
        <nav class="bg-white shadow-lg" id="main-navigation">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div class="flex justify-between h-16">
                    <div class="flex items-center">
                        <a href="/" class="text-2xl font-bold text-blue-600" id="logo">
                            🔍 Pynomaly
                        </a>
                    </div>
                    <div class="hidden sm:flex sm:space-x-8 items-center">
                        <a href="/" class="text-gray-900 border-b-2 border-blue-600 px-3 py-2 text-sm font-medium">Dashboard</a>
                        <a href="/detectors" class="text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium">Detectors</a>
                        <a href="/datasets" class="text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium">Datasets</a>
                        <a href="/detection" class="text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium">Detection</a>
                        <a href="/visualizations" class="text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium">Visualizations</a>
                        <a href="/exports" class="text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium">Exports</a>
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
                    <a href="/" class="block px-3 py-2 text-gray-900 font-medium">Dashboard</a>
                    <a href="/detectors" class="block px-3 py-2 text-gray-500">Detectors</a>
                    <a href="/datasets" class="block px-3 py-2 text-gray-500">Datasets</a>
                </div>
            </div>
        </nav>

        <!-- Main Content -->
        <main class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
            <div id="main-content">
                <h1 class="text-3xl font-bold text-gray-900 mb-8" id="page-title">Dashboard</h1>

                <!-- Statistics Cards -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8" id="stats-cards">
                    <div class="bg-white overflow-hidden shadow rounded-lg">
                        <div class="p-5">
                            <div class="flex items-center">
                                <div class="flex-shrink-0">
                                    <div class="w-8 h-8 bg-blue-600 rounded-md flex items-center justify-center">
                                        <span class="text-white font-bold">D</span>
                                    </div>
                                </div>
                                <div class="ml-5 w-0 flex-1">
                                    <dl>
                                        <dt class="text-sm font-medium text-gray-500 truncate">Detectors</dt>
                                        <dd class="text-lg font-medium text-gray-900" id="detector-count">12</dd>
                                    </dl>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="bg-white overflow-hidden shadow rounded-lg">
                        <div class="p-5">
                            <div class="flex items-center">
                                <div class="flex-shrink-0">
                                    <div class="w-8 h-8 bg-green-600 rounded-md flex items-center justify-center">
                                        <span class="text-white font-bold">S</span>
                                    </div>
                                </div>
                                <div class="ml-5 w-0 flex-1">
                                    <dl>
                                        <dt class="text-sm font-medium text-gray-500 truncate">Datasets</dt>
                                        <dd class="text-lg font-medium text-gray-900" id="dataset-count">8</dd>
                                    </dl>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="bg-white overflow-hidden shadow rounded-lg">
                        <div class="p-5">
                            <div class="flex items-center">
                                <div class="flex-shrink-0">
                                    <div class="w-8 h-8 bg-yellow-600 rounded-md flex items-center justify-center">
                                        <span class="text-white font-bold">R</span>
                                    </div>
                                </div>
                                <div class="ml-5 w-0 flex-1">
                                    <dl>
                                        <dt class="text-sm font-medium text-gray-500 truncate">Results</dt>
                                        <dd class="text-lg font-medium text-gray-900" id="results-count">156</dd>
                                    </dl>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Quick Actions -->
                <div class="flex flex-col sm:flex-row gap-4" id="quick-actions">
                    <button class="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors" id="quick-detection-btn">
                        🚀 Quick Detection
                    </button>
                    <button class="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition-colors" id="upload-dataset-btn">
                        📊 Upload Dataset
                    </button>
                    <button class="bg-yellow-600 text-white px-4 py-2 rounded-md hover:bg-yellow-700 transition-colors" id="autonomous-mode-btn">
                        🤖 Autonomous Mode
                    </button>
                </div>
            </div>
        </main>

        <script>
            // Mobile menu toggle
            document.getElementById('mobile-menu-btn')?.addEventListener('click', function() {
                const menu = document.getElementById('mobile-menu');
                if (menu) {
                    menu.classList.toggle('hidden');
                }
            });
        </script>
    </body>
    </html>
    """


@app.get("/detectors", response_class=HTMLResponse)
async def detectors():
    return """
    <!DOCTYPE html>
    <html><head><title>Pynomaly - Detectors</title><script src="https://cdn.tailwindcss.com"></script></head>
    <body class="bg-gray-50"><nav class="bg-white shadow-lg" id="main-navigation">
    <div class="max-w-7xl mx-auto px-4"><div class="flex justify-between h-16">
    <div class="flex items-center"><a href="/" id="logo" class="text-2xl font-bold text-blue-600">🔍 Pynomaly</a></div>
    </div></div></nav>
    <main class="max-w-7xl mx-auto py-6 px-4"><div id="main-content">
    <h1 id="page-title" class="text-3xl font-bold text-gray-900 mb-8">Detectors</h1>
    <div class="bg-white shadow rounded-lg p-6" id="detectors-section">
    <h2 class="text-xl font-semibold mb-4">Anomaly Detectors</h2>
    <div class="space-y-4"><div class="border border-gray-200 rounded-lg p-4">
    <h3 class="font-medium">IsolationForest</h3></div></div></div>
    </div></main></body></html>
    """


@app.get("/datasets", response_class=HTMLResponse)
async def datasets():
    return """
    <!DOCTYPE html>
    <html><head><title>Pynomaly - Datasets</title><script src="https://cdn.tailwindcss.com"></script></head>
    <body class="bg-gray-50"><nav class="bg-white shadow-lg" id="main-navigation">
    <div class="max-w-7xl mx-auto px-4"><div class="flex justify-between h-16">
    <div class="flex items-center"><a href="/" id="logo" class="text-2xl font-bold text-blue-600">🔍 Pynomaly</a></div>
    </div></div></nav>
    <main class="max-w-7xl mx-auto py-6 px-4"><div id="main-content">
    <h1 id="page-title" class="text-3xl font-bold text-gray-900 mb-8">Datasets</h1>
    <div class="bg-white shadow rounded-lg p-6" id="datasets-section">
    <button id="upload-btn" class="px-4 py-2 bg-blue-600 text-white rounded">Upload Files</button>
    </div></div></main></body></html>
    """


@app.get("/detection", response_class=HTMLResponse)
async def detection():
    return """
    <!DOCTYPE html>
    <html><head><title>Pynomaly - Detection</title><script src="https://cdn.tailwindcss.com"></script></head>
    <body class="bg-gray-50"><nav class="bg-white shadow-lg" id="main-navigation">
    <div class="max-w-7xl mx-auto px-4"><div class="flex justify-between h-16">
    <div class="flex items-center"><a href="/" id="logo" class="text-2xl font-bold text-blue-600">🔍 Pynomaly</a></div>
    </div></div></nav>
    <main class="max-w-7xl mx-auto py-6 px-4"><div id="main-content">
    <h1 id="page-title" class="text-3xl font-bold text-gray-900 mb-8">Detection</h1>
    <div class="bg-white shadow rounded-lg p-6" id="detection-section">
    <form id="detection-form"><select id="detector-select" class="block w-full rounded-md">
    <option>IsolationForest</option></select>
    <button id="run-detection-btn" class="bg-blue-600 text-white px-4 py-2 rounded-md">Run Detection</button>
    </form></div></div></main></body></html>
    """


@app.get("/visualizations", response_class=HTMLResponse)
async def visualizations():
    return """
    <!DOCTYPE html>
    <html><head><title>Pynomaly - Visualizations</title><script src="https://cdn.tailwindcss.com"></script></head>
    <body class="bg-gray-50"><nav class="bg-white shadow-lg" id="main-navigation">
    <div class="max-w-7xl mx-auto px-4"><div class="flex justify-between h-16">
    <div class="flex items-center"><a href="/" id="logo" class="text-2xl font-bold text-blue-600">🔍 Pynomaly</a></div>
    </div></div></nav>
    <main class="max-w-7xl mx-auto py-6 px-4"><div id="main-content">
    <h1 id="page-title" class="text-3xl font-bold text-gray-900 mb-8">Visualizations</h1>
    <div class="bg-white shadow rounded-lg p-6" id="visualizations-section">
    <h2 class="text-xl font-semibold mb-4">Visualizations</h2>
    </div></div></main></body></html>
    """


@app.get("/exports", response_class=HTMLResponse)
async def exports():
    return """
    <!DOCTYPE html>
    <html><head><title>Pynomaly - Exports</title><script src="https://cdn.tailwindcss.com"></script></head>
    <body class="bg-gray-50"><nav class="bg-white shadow-lg" id="main-navigation">
    <div class="max-w-7xl mx-auto px-4"><div class="flex justify-between h-16">
    <div class="flex items-center"><a href="/" id="logo" class="text-2xl font-bold text-blue-600">🔍 Pynomaly</a></div>
    </div></div></nav>
    <main class="max-w-7xl mx-auto py-6 px-4"><div id="main-content">
    <h1 id="page-title" class="text-3xl font-bold text-gray-900 mb-8">Exports</h1>
    <div class="bg-white shadow rounded-lg p-6" id="exports-section">
    <select id="export-format" class="block w-full rounded-md"><option>CSV</option></select>
    <button id="export-btn" class="bg-green-600 text-white px-4 py-2 rounded-md">Export Data</button>
    </div></div></main></body></html>
    """


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Pynomaly UI Test Server"}


if __name__ == "__main__":
    print("🚀 Starting Pynomaly UI Test Server...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
