#!/usr/bin/env python3
"""
Simple FastAPI server for UI testing demonstration.
This creates a minimal Pynomaly-like web interface for testing purposes.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Pynomaly UI Test Server", version="1.0.0")

# Minimal HTML template for testing
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pynomaly - State-of-the-art Anomaly Detection</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .fade-in { animation: fadeIn 0.5s ease-in; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
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
                <a href="/detection" class="block px-3 py-2 text-gray-500">Detection</a>
                <a href="/visualizations" class="block px-3 py-2 text-gray-500">Visualizations</a>
                <a href="/exports" class="block px-3 py-2 text-gray-500">Exports</a>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div class="fade-in" id="main-content">
            <h1 class="text-3xl font-bold text-gray-900 mb-8" id="page-title">{page_title}</h1>
            
            {content}
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

DASHBOARD_CONTENT = """
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

<!-- Recent Results Table -->
<div class="bg-white shadow overflow-hidden sm:rounded-md mb-8" id="recent-results">
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
        </li>
    </ul>
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
"""

# Route handlers
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTML_TEMPLATE.format(
        page_title="Dashboard",
        content=DASHBOARD_CONTENT
    )

@app.get("/detectors", response_class=HTMLResponse)
async def detectors():
    content = """
    <div class="bg-white shadow rounded-lg p-6" id="detectors-section">
        <h2 class="text-xl font-semibold mb-4">Anomaly Detectors</h2>
        <div class="space-y-4">
            <div class="border border-gray-200 rounded-lg p-4">
                <h3 class="font-medium">IsolationForest</h3>
                <p class="text-sm text-gray-500">Isolation-based anomaly detection</p>
                <div class="mt-2 flex space-x-2">
                    <button class="px-3 py-1 bg-blue-100 text-blue-800 rounded text-sm">Configure</button>
                    <button class="px-3 py-1 bg-green-100 text-green-800 rounded text-sm">Train</button>
                </div>
            </div>
        </div>
    </div>
    """
    return HTML_TEMPLATE.format(
        page_title="Detectors",
        content=content
    )

@app.get("/datasets", response_class=HTMLResponse)
async def datasets():
    content = """
    <div class="bg-white shadow rounded-lg p-6" id="datasets-section">
        <h2 class="text-xl font-semibold mb-4">Datasets</h2>
        <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <svg class="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
            </svg>
            <h3 class="mt-2 text-sm font-medium text-gray-900">Upload Dataset</h3>
            <p class="mt-1 text-sm text-gray-500">Drag and drop files or click to browse</p>
            <button class="mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700" id="upload-btn">
                Upload Files
            </button>
        </div>
    </div>
    """
    return HTML_TEMPLATE.format(
        page_title="Datasets",
        content=content
    )

@app.get("/detection", response_class=HTMLResponse)
async def detection():
    content = """
    <div class="bg-white shadow rounded-lg p-6" id="detection-section">
        <h2 class="text-xl font-semibold mb-4">Anomaly Detection</h2>
        <form class="space-y-4" id="detection-form">
            <div>
                <label class="block text-sm font-medium text-gray-700">Select Detector</label>
                <select class="mt-1 block w-full rounded-md border-gray-300 shadow-sm" id="detector-select">
                    <option>IsolationForest</option>
                    <option>LOF</option>
                    <option>OneClassSVM</option>
                </select>
            </div>
            <div>
                <label class="block text-sm font-medium text-gray-700">Select Dataset</label>
                <select class="mt-1 block w-full rounded-md border-gray-300 shadow-sm" id="dataset-select">
                    <option>financial_data.csv</option>
                    <option>network_logs.csv</option>
                </select>
            </div>
            <button type="submit" class="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700" id="run-detection-btn">
                Run Detection
            </button>
        </form>
    </div>
    """
    return HTML_TEMPLATE.format(
        page_title="Detection",
        content=content
    )

@app.get("/visualizations", response_class=HTMLResponse)
async def visualizations():
    content = """
    <div class="bg-white shadow rounded-lg p-6" id="visualizations-section">
        <h2 class="text-xl font-semibold mb-4">Visualizations</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="border border-gray-200 rounded-lg p-4">
                <h3 class="font-medium mb-2">Anomaly Score Distribution</h3>
                <div class="h-48 bg-gray-100 rounded flex items-center justify-center">
                    <span class="text-gray-500">Chart Placeholder</span>
                </div>
            </div>
            <div class="border border-gray-200 rounded-lg p-4">
                <h3 class="font-medium mb-2">Detection Timeline</h3>
                <div class="h-48 bg-gray-100 rounded flex items-center justify-center">
                    <span class="text-gray-500">Chart Placeholder</span>
                </div>
            </div>
        </div>
    </div>
    """
    return HTML_TEMPLATE.format(
        page_title="Visualizations",
        content=content
    )

@app.get("/exports", response_class=HTMLResponse)
async def exports():
    content = """
    <div class="bg-white shadow rounded-lg p-6" id="exports-section">
        <h2 class="text-xl font-semibold mb-4">Export Results</h2>
        <div class="space-y-4">
            <div>
                <label class="block text-sm font-medium text-gray-700">Export Format</label>
                <select class="mt-1 block w-full rounded-md border-gray-300 shadow-sm" id="export-format">
                    <option>CSV</option>
                    <option>JSON</option>
                    <option>Excel</option>
                    <option>PDF Report</option>
                </select>
            </div>
            <button class="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700" id="export-btn">
                Export Data
            </button>
        </div>
    </div>
    """
    return HTML_TEMPLATE.format(
        page_title="Exports",
        content=content
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Pynomaly UI Test Server"}

if __name__ == "__main__":
    print("🚀 Starting Pynomaly UI Test Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("🏠 Dashboard: http://localhost:8000/")
    print("🛠️ Health check: http://localhost:8000/health")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")