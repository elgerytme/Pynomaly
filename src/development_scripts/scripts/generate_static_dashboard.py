#!/usr/bin/env python3
"""
Generate static dashboard for GitHub Pages deployment.

This script creates a static version of the anomaly_detection dashboard that can be
deployed to GitHub Pages without requiring a backend server.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def generate_sample_data() -> dict[str, Any]:
    """Generate sample data for the static dashboard."""

    # Generate sample anomaly detection results
    np.random.seed(42)
    n_points = 100

    # Time series data
    timestamps = [datetime.now().timestamp() + i * 3600 for i in range(n_points)]
    normal_data = np.random.normal(0, 1, size=n_points - 10)
    anomaly_data = np.random.normal(3, 0.5, size=10)  # Clear anomalies

    # Combine normal and anomaly data
    all_data = np.concatenate([normal_data, anomaly_data])
    np.random.shuffle(all_data)

    # Anomaly scores (higher for anomalies)
    anomaly_scores = []
    for value in all_data:
        if abs(value) > 2:  # Anomaly threshold
            score = min(0.9, abs(value) / 4)
        else:
            score = max(0.1, abs(value) / 10)
        anomaly_scores.append(score)

    # Create sample dashboard data
    dashboard_data = {
        "summary": {
            "total_data_points": n_points,
            "anomalies_detected": sum(1 for score in anomaly_scores if score > 0.5),
            "accuracy": 0.94,
            "last_updated": datetime.now().isoformat(),
            "system_status": "healthy",
        },
        "time_series": {
            "timestamps": timestamps,
            "values": all_data.tolist(),
            "anomaly_scores": anomaly_scores,
        },
        "detectors": [
            {
                "id": "isolation_forest_1",
                "name": "Isolation Forest",
                "status": "active",
                "accuracy": 0.94,
                "last_trained": "2024-07-10T10:30:00Z",
                "anomalies_detected": 8,
            },
            {
                "id": "one_class_svm_1",
                "name": "One-Class SVM",
                "status": "active",
                "accuracy": 0.89,
                "last_trained": "2024-07-10T09:15:00Z",
                "anomalies_detected": 12,
            },
            {
                "id": "local_outlier_factor_1",
                "name": "Local Outlier Factor",
                "status": "training",
                "accuracy": 0.92,
                "last_trained": "2024-07-10T08:45:00Z",
                "anomalies_detected": 6,
            },
        ],
        "recent_alerts": [
            {
                "id": "alert_001",
                "timestamp": "2024-07-10T14:23:15Z",
                "severity": "high",
                "message": "Unusual pattern detected in sensor data",
                "detector": "isolation_forest_1",
                "anomaly_score": 0.87,
            },
            {
                "id": "alert_002",
                "timestamp": "2024-07-10T14:18:42Z",
                "severity": "medium",
                "message": "Statistical outlier in network traffic",
                "detector": "one_class_svm_1",
                "anomaly_score": 0.63,
            },
            {
                "id": "alert_003",
                "timestamp": "2024-07-10T14:12:08Z",
                "severity": "low",
                "message": "Minor deviation from baseline behavior",
                "detector": "local_outlier_factor_1",
                "anomaly_score": 0.52,
            },
        ],
        "feature_importance": [
            {"feature": "sensor_temperature", "importance": 0.34, "impact": "high"},
            {"feature": "network_latency", "importance": 0.28, "impact": "medium"},
            {"feature": "cpu_utilization", "importance": 0.21, "impact": "medium"},
            {"feature": "memory_usage", "importance": 0.17, "impact": "low"},
        ],
        "performance_metrics": {
            "precision": 0.91,
            "recall": 0.88,
            "f1_score": 0.895,
            "auc_roc": 0.93,
            "processing_time_ms": 12.5,
        },
    }

    return dashboard_data


def create_static_html() -> str:
    """Create the static HTML dashboard."""

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>anomaly_detection - Real-time Anomaly Detection Dashboard</title>
    <meta name="description" content="Interactive dashboard for monitoring anomaly detection in real-time">

    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>

    <!-- Chart.js for visualizations -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <!-- Alpine.js for interactivity -->
    <script defer src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"></script>

    <style>
        [x-cloak] { display: none !important; }
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .card-hover {
            transition: transform 0.2s ease-in-out;
        }
        .card-hover:hover {
            transform: translateY(-2px);
        }
        .status-healthy { color: #10b981; }
        .status-warning { color: #f59e0b; }
        .status-critical { color: #ef4444; }
        .metric-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
        }
    </style>
</head>
<body class="bg-gray-50" x-data="dashboardApp()" x-init="init()">
    <!-- Header -->
    <header class="gradient-bg text-white shadow-lg">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between items-center py-6">
                <div class="flex items-center">
                    <h1 class="text-3xl font-bold">🔍 anomaly_detection</h1>
                    <span class="ml-4 text-lg opacity-90">Anomaly Detection Dashboard</span>
                </div>
                <div class="flex items-center space-x-4">
                    <div class="text-sm opacity-90">
                        <span x-text="lastUpdated"></span>
                    </div>
                    <div class="flex items-center">
                        <div class="w-3 h-3 rounded-full mr-2"
                             :class="systemStatus === 'healthy' ? 'bg-green-400' : 'bg-yellow-400'"></div>
                        <span class="text-sm font-medium" x-text="systemStatus.toUpperCase()"></span>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Dashboard -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

        <!-- Summary Cards -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">

            <!-- Total Data Points -->
            <div class="metric-card rounded-lg shadow-lg p-6 card-hover">
                <div class="flex items-center">
                    <div class="flex-shrink-0">
                        <div class="w-8 h-8 bg-blue-500 rounded-md flex items-center justify-center">
                            <span class="text-white text-lg">📊</span>
                        </div>
                    </div>
                    <div class="ml-5 w-0 flex-1">
                        <dl>
                            <dt class="text-sm font-medium text-gray-500 truncate">Total Data Points</dt>
                            <dd class="text-lg font-medium text-gray-900" x-text="summary.total_data_points"></dd>
                        </dl>
                    </div>
                </div>
            </div>

            <!-- Anomalies Detected -->
            <div class="metric-card rounded-lg shadow-lg p-6 card-hover">
                <div class="flex items-center">
                    <div class="flex-shrink-0">
                        <div class="w-8 h-8 bg-red-500 rounded-md flex items-center justify-center">
                            <span class="text-white text-lg">⚠️</span>
                        </div>
                    </div>
                    <div class="ml-5 w-0 flex-1">
                        <dl>
                            <dt class="text-sm font-medium text-gray-500 truncate">Anomalies Detected</dt>
                            <dd class="text-lg font-medium text-gray-900" x-text="summary.anomalies_detected"></dd>
                        </dl>
                    </div>
                </div>
            </div>

            <!-- Detection Accuracy -->
            <div class="metric-card rounded-lg shadow-lg p-6 card-hover">
                <div class="flex items-center">
                    <div class="flex-shrink-0">
                        <div class="w-8 h-8 bg-green-500 rounded-md flex items-center justify-center">
                            <span class="text-white text-lg">🎯</span>
                        </div>
                    </div>
                    <div class="ml-5 w-0 flex-1">
                        <dl>
                            <dt class="text-sm font-medium text-gray-500 truncate">Detection Accuracy</dt>
                            <dd class="text-lg font-medium text-gray-900" x-text="(summary.accuracy * 100).toFixed(1) + '%'"></dd>
                        </dl>
                    </div>
                </div>
            </div>

            <!-- System Status -->
            <div class="metric-card rounded-lg shadow-lg p-6 card-hover">
                <div class="flex items-center">
                    <div class="flex-shrink-0">
                        <div class="w-8 h-8 bg-purple-500 rounded-md flex items-center justify-center">
                            <span class="text-white text-lg">🖥️</span>
                        </div>
                    </div>
                    <div class="ml-5 w-0 flex-1">
                        <dl>
                            <dt class="text-sm font-medium text-gray-500 truncate">System Status</dt>
                            <dd class="text-lg font-medium"
                                :class="systemStatus === 'healthy' ? 'text-green-600' : 'text-yellow-600'"
                                x-text="systemStatus.toUpperCase()"></dd>
                        </dl>
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">

            <!-- Time Series Chart -->
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h3 class="text-lg font-medium text-gray-900 mb-4">Anomaly Detection Over Time</h3>
                <div class="h-64">
                    <canvas id="timeSeriesChart"></canvas>
                </div>
            </div>

            <!-- Feature Importance Chart -->
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h3 class="text-lg font-medium text-gray-900 mb-4">Feature Importance</h3>
                <div class="h-64">
                    <canvas id="featureChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Detectors and Alerts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">

            <!-- Active Detectors -->
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h3 class="text-lg font-medium text-gray-900 mb-4">Active Detectors</h3>
                <div class="space-y-4">
                    <template x-for="detector in detectors" :key="detector.id">
                        <div class="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                            <div class="flex items-center">
                                <div class="w-3 h-3 rounded-full mr-3"
                                     :class="detector.status === 'active' ? 'bg-green-400' : 'bg-yellow-400'"></div>
                                <div>
                                    <p class="text-sm font-medium text-gray-900" x-text="detector.name"></p>
                                    <p class="text-xs text-gray-500" x-text="'Accuracy: ' + (detector.accuracy * 100).toFixed(1) + '%'"></p>
                                </div>
                            </div>
                            <div class="text-right">
                                <p class="text-sm font-medium text-gray-900" x-text="detector.anomalies_detected"></p>
                                <p class="text-xs text-gray-500">anomalies</p>
                            </div>
                        </div>
                    </template>
                </div>
            </div>

            <!-- Recent Alerts -->
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h3 class="text-lg font-medium text-gray-900 mb-4">Recent Alerts</h3>
                <div class="space-y-4">
                    <template x-for="alert in recentAlerts" :key="alert.id">
                        <div class="flex items-start space-x-3 p-4 bg-gray-50 rounded-lg">
                            <div class="flex-shrink-0">
                                <div class="w-6 h-6 rounded-full flex items-center justify-center"
                                     :class="{
                                         'bg-red-100 text-red-600': alert.severity === 'high',
                                         'bg-yellow-100 text-yellow-600': alert.severity === 'medium',
                                         'bg-blue-100 text-blue-600': alert.severity === 'low'
                                     }">
                                    <span class="text-xs font-bold" x-text="alert.severity.charAt(0).toUpperCase()"></span>
                                </div>
                            </div>
                            <div class="flex-1 min-w-0">
                                <p class="text-sm font-medium text-gray-900" x-text="alert.message"></p>
                                <div class="flex items-center space-x-2 text-xs text-gray-500">
                                    <span x-text="formatTimestamp(alert.timestamp)"></span>
                                    <span>•</span>
                                    <span x-text="'Score: ' + alert.anomaly_score.toFixed(2)"></span>
                                </div>
                            </div>
                        </div>
                    </template>
                </div>
            </div>
        </div>

        <!-- Performance Metrics -->
        <div class="mt-8 bg-white rounded-lg shadow-lg p-6">
            <h3 class="text-lg font-medium text-gray-900 mb-4">Performance Metrics</h3>
            <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div class="text-center">
                    <div class="text-2xl font-bold text-blue-600" x-text="(performanceMetrics.precision * 100).toFixed(1) + '%'"></div>
                    <div class="text-sm text-gray-500">Precision</div>
                </div>
                <div class="text-center">
                    <div class="text-2xl font-bold text-green-600" x-text="(performanceMetrics.recall * 100).toFixed(1) + '%'"></div>
                    <div class="text-sm text-gray-500">Recall</div>
                </div>
                <div class="text-center">
                    <div class="text-2xl font-bold text-purple-600" x-text="(performanceMetrics.f1_score * 100).toFixed(1) + '%'"></div>
                    <div class="text-sm text-gray-500">F1 Score</div>
                </div>
                <div class="text-center">
                    <div class="text-2xl font-bold text-indigo-600" x-text="(performanceMetrics.auc_roc * 100).toFixed(1) + '%'"></div>
                    <div class="text-sm text-gray-500">AUC-ROC</div>
                </div>
                <div class="text-center">
                    <div class="text-2xl font-bold text-red-600" x-text="performanceMetrics.processing_time_ms.toFixed(1) + 'ms'"></div>
                    <div class="text-sm text-gray-500">Processing Time</div>
                </div>
            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="bg-white border-t mt-16">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div class="text-center text-gray-500">
                <p>anomaly_detection - Advanced Anomaly Detection Platform</p>
                <p class="mt-2">🔗 <a href="https://github.com/anomaly_detection/anomaly_detection" class="text-blue-600 hover:text-blue-800">GitHub</a> |
                   📖 <a href="./docs/api/" class="text-blue-600 hover:text-blue-800">API Documentation</a> |
                   🚀 <a href="./examples/" class="text-blue-600 hover:text-blue-800">Examples</a></p>
            </div>
        </div>
    </footer>

    <script>
        function dashboardApp() {
            return {
                summary: {},
                detectors: [],
                recentAlerts: [],
                featureImportance: [],
                performanceMetrics: {},
                timeSeriesData: {},
                systemStatus: 'healthy',
                lastUpdated: '',

                async init() {
                    // Load dashboard data
                    try {
                        const response = await fetch('./dashboard-data.json');
                        const data = await response.json();

                        this.summary = data.summary;
                        this.detectors = data.detectors;
                        this.recentAlerts = data.recent_alerts;
                        this.featureImportance = data.feature_importance;
                        this.performanceMetrics = data.performance_metrics;
                        this.timeSeriesData = data.time_series;
                        this.systemStatus = data.summary.system_status;
                        this.lastUpdated = this.formatTimestamp(data.summary.last_updated);

                        // Initialize charts
                        this.$nextTick(() => {
                            this.initTimeSeriesChart();
                            this.initFeatureChart();
                        });
                    } catch (error) {
                        console.error('Failed to load dashboard data:', error);
                        // Use fallback data
                        this.loadFallbackData();
                    }
                },

                loadFallbackData() {
                    this.summary = {
                        total_data_points: 100,
                        anomalies_detected: 8,
                        accuracy: 0.94,
                        system_status: 'healthy'
                    };
                    this.systemStatus = 'healthy';
                    this.lastUpdated = 'Just now';
                },

                initTimeSeriesChart() {
                    const ctx = document.getElementById('timeSeriesChart');
                    if (!ctx || !this.timeSeriesData.timestamps) return;

                    new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: this.timeSeriesData.timestamps.slice(-20).map(ts =>
                                new Date(ts * 1000).toLocaleTimeString()
                            ),
                            datasets: [{
                                label: 'Anomaly Score',
                                data: this.timeSeriesData.anomaly_scores.slice(-20),
                                borderColor: 'rgb(239, 68, 68)',
                                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                                tension: 0.4,
                                fill: true
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    max: 1
                                }
                            },
                            plugins: {
                                legend: {
                                    display: false
                                }
                            }
                        }
                    });
                },

                initFeatureChart() {
                    const ctx = document.getElementById('featureChart');
                    if (!ctx || !this.featureImportance.length) return;

                    new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: this.featureImportance.map(f => f.feature),
                            datasets: [{
                                label: 'Importance',
                                data: this.featureImportance.map(f => f.importance),
                                backgroundColor: [
                                    'rgba(59, 130, 246, 0.8)',
                                    'rgba(16, 185, 129, 0.8)',
                                    'rgba(245, 158, 11, 0.8)',
                                    'rgba(139, 92, 246, 0.8)'
                                ],
                                borderWidth: 0
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            },
                            plugins: {
                                legend: {
                                    display: false
                                }
                            }
                        }
                    });
                },

                formatTimestamp(timestamp) {
                    try {
                        const date = new Date(timestamp);
                        return date.toLocaleString();
                    } catch {
                        return 'Recently';
                    }
                }
            }
        }
    </script>
</body>
</html>"""

    return html_content


def create_github_pages_index() -> str:
    """Create the main index.html for GitHub Pages."""

    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>anomaly_detection - Advanced Anomaly Detection Platform</title>
    <meta name="description" content="Production-ready anomaly domain-bounded monorepo with real-time monitoring, ML governance, and advanced analytics">
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .card-hover {
            transition: all 0.3s ease;
        }
        .card-hover:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body class="bg-gray-50">
    <!-- Hero Section -->
    <header class="gradient-bg text-white">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
            <div class="text-center">
                <h1 class="text-5xl font-bold mb-6">🔍 anomaly_detection</h1>
                <p class="text-xl mb-8 opacity-90">Advanced Anomaly Detection Platform</p>
                <p class="text-lg mb-12 max-w-3xl mx-auto opacity-80">
                    Production-ready anomaly detection with real-time streaming, ML governance,
                    multi-tenant architecture, and advanced explainability features.
                </p>
                <div class="flex flex-col sm:flex-row gap-4 justify-center">
                    <a href="./dashboard.html"
                       class="bg-white text-blue-600 px-8 py-3 rounded-lg font-medium hover:bg-gray-100 transition-colors">
                        🚀 View Live Dashboard
                    </a>
                    <a href="./docs/api/"
                       class="border-2 border-white text-white px-8 py-3 rounded-lg font-medium hover:bg-white hover:text-blue-600 transition-colors">
                        📖 API Documentation
                    </a>
                </div>
            </div>
        </div>
    </header>

    <!-- Features Section -->
    <section class="py-16">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="text-center mb-12">
                <h2 class="text-3xl font-bold text-gray-900 mb-4">Production-Ready Features</h2>
                <p class="text-lg text-gray-600">Enterprise-grade anomaly detection with comprehensive ML operations</p>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                <!-- Real-time Detection -->
                <div class="bg-white rounded-xl shadow-lg p-6 card-hover">
                    <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                        <span class="text-2xl">⚡</span>
                    </div>
                    <h3 class="text-xl font-semibold text-gray-900 mb-2">Real-time Streaming</h3>
                    <p class="text-gray-600">High-throughput anomaly detection with adaptive thresholds and intelligent alerting for live data streams.</p>
                </div>

                <!-- ML Governance -->
                <div class="bg-white rounded-xl shadow-lg p-6 card-hover">
                    <div class="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
                        <span class="text-2xl">🏛️</span>
                    </div>
                    <h3 class="text-xl font-semibold text-gray-900 mb-2">ML Governance</h3>
                    <p class="text-gray-600">Comprehensive model lifecycle management with automated drift detection and intelligent retraining.</p>
                </div>

                <!-- Multi-tenant Architecture -->
                <div class="bg-white rounded-xl shadow-lg p-6 card-hover">
                    <div class="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
                        <span class="text-2xl">🏢</span>
                    </div>
                    <h3 class="text-xl font-semibold text-gray-900 mb-2">Multi-tenant</h3>
                    <p class="text-gray-600">Secure isolation with resource management, federated learning, and tenant-specific configurations.</p>
                </div>

                <!-- Advanced Explainability -->
                <div class="bg-white rounded-xl shadow-lg p-6 card-hover">
                    <div class="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center mb-4">
                        <span class="text-2xl">🔬</span>
                    </div>
                    <h3 class="text-xl font-semibold text-gray-900 mb-2">Explainable AI</h3>
                    <p class="text-gray-600">SHAP, LIME, counterfactuals, and advanced interpretability features for transparent anomaly detection.</p>
                </div>

                <!-- Performance Monitoring -->
                <div class="bg-white rounded-xl shadow-lg p-6 card-hover">
                    <div class="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center mb-4">
                        <span class="text-2xl">📊</span>
                    </div>
                    <h3 class="text-xl font-semibold text-gray-900 mb-2">Monitoring</h3>
                    <p class="text-gray-600">Real-time dashboards, performance metrics, and comprehensive alerting with Prometheus integration.</p>
                </div>

                <!-- Enterprise Security -->
                <div class="bg-white rounded-xl shadow-lg p-6 card-hover">
                    <div class="w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center mb-4">
                        <span class="text-2xl">🔒</span>
                    </div>
                    <h3 class="text-xl font-semibold text-gray-900 mb-2">Enterprise Security</h3>
                    <p class="text-gray-600">MFA, role-based access control, audit logging, and comprehensive security hardening.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- Quick Start Section -->
    <section class="py-16 bg-white">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="text-center mb-12">
                <h2 class="text-3xl font-bold text-gray-900 mb-4">Quick Start</h2>
                <p class="text-lg text-gray-600">Get started with anomaly_detection in minutes</p>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-12">
                <!-- Installation -->
                <div>
                    <h3 class="text-xl font-semibold text-gray-900 mb-4">📦 Installation</h3>
                    <div class="bg-gray-900 rounded-lg p-4 text-green-400 font-mono text-sm">
                        <div># Install anomaly_detection</div>
                        <div>pip install anomaly_detection</div>
                        <div><br># Start the web UI</div>
                        <div>anomaly_detection server start</div>
                        <div><br># Open dashboard</div>
                        <div>open http://localhost:8000</div>
                    </div>
                </div>

                <!-- Python API -->
                <div>
                    <h3 class="text-xl font-semibold text-gray-900 mb-4">🐍 Python API</h3>
                    <div class="bg-gray-900 rounded-lg p-4 text-green-400 font-mono text-sm">
                        <div>from anomaly_detection import AnomalyDetector</div>
                        <div><br># Create detector</div>
                        <div>detector = AnomalyDetector()</div>
                        <div><br># Detect anomalies</div>
                        <div>anomalies = detector.detect(data)</div>
                        <div>print(f"Found {len(anomalies)} anomalies")</div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Resources Section -->
    <section class="py-16">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="text-center mb-12">
                <h2 class="text-3xl font-bold text-gray-900 mb-4">Resources</h2>
                <p class="text-lg text-gray-600">Documentation, examples, and guides</p>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <a href="./dashboard.html" class="block bg-white rounded-lg shadow-lg p-6 card-hover text-center">
                    <div class="text-3xl mb-3">📊</div>
                    <h3 class="font-semibold text-gray-900 mb-2">Live Dashboard</h3>
                    <p class="text-sm text-gray-600">Interactive monitoring dashboard</p>
                </a>

                <a href="./docs/api/" class="block bg-white rounded-lg shadow-lg p-6 card-hover text-center">
                    <div class="text-3xl mb-3">📖</div>
                    <h3 class="font-semibold text-gray-900 mb-2">API Docs</h3>
                    <p class="text-sm text-gray-600">Complete API reference</p>
                </a>

                <a href="./examples/" class="block bg-white rounded-lg shadow-lg p-6 card-hover text-center">
                    <div class="text-3xl mb-3">🚀</div>
                    <h3 class="font-semibold text-gray-900 mb-2">Examples</h3>
                    <p class="text-sm text-gray-600">Code samples and tutorials</p>
                </a>

                <a href="./docs/user-guide/" class="block bg-white rounded-lg shadow-lg p-6 card-hover text-center">
                    <div class="text-3xl mb-3">📚</div>
                    <h3 class="font-semibold text-gray-900 mb-2">User Guide</h3>
                    <p class="text-sm text-gray-600">Getting started guide</p>
                </a>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="bg-gray-900 text-white py-12">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="text-center">
                <h3 class="text-xl font-bold mb-4">🔍 anomaly_detection</h3>
                <p class="text-gray-400 mb-6">Advanced Anomaly Detection Platform</p>
                <div class="flex justify-center space-x-6">
                    <a href="https://github.com/anomaly_detection/anomaly_detection" class="text-gray-400 hover:text-white transition-colors">GitHub</a>
                    <a href="./docs/api/" class="text-gray-400 hover:text-white transition-colors">Documentation</a>
                    <a href="./examples/" class="text-gray-400 hover:text-white transition-colors">Examples</a>
                </div>
                <div class="mt-8 text-sm text-gray-500">
                    <p>&copy; 2024 anomaly_detection. Built with ❤️ for production anomaly detection.</p>
                </div>
            </div>
        </div>
    </footer>
</body>
</html>"""


def main():
    """Main function to generate static dashboard files."""

    print("🚀 Generating static dashboard for GitHub Pages...")

    # Create output directory
    output_dir = Path("static_dashboard")
    output_dir.mkdir(exist_ok=True)

    # Generate sample data
    print("📊 Generating sample dashboard data...")
    dashboard_data = generate_sample_data()

    # Write dashboard data as JSON
    with open(output_dir / "dashboard-data.json", "w") as f:
        json.dump(dashboard_data, f, indent=2)

    # Generate dashboard HTML
    print("🎨 Creating dashboard HTML...")
    dashboard_html = create_static_html()
    with open(output_dir / "dashboard.html", "w") as f:
        f.write(dashboard_html)

    # Create docs directory for GitHub Pages
    docs_dir = Path("docs/github-pages")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Generate main index.html
    print("📄 Creating main index.html...")
    index_html = create_github_pages_index()
    with open(docs_dir / "index.html", "w") as f:
        f.write(index_html)

    # Copy static assets if they exist
    static_src = Path("src/anomaly_detection/presentation/web/static")
    if static_src.exists():
        print("📁 Copying static assets...")
        static_dest = output_dir / "static"
        if static_dest.exists():
            shutil.rmtree(static_dest)
        shutil.copytree(static_src, static_dest)

    print("✅ Static dashboard generated successfully!")
    print(f"📂 Output directory: {output_dir.absolute()}")
    print(f"🌐 Main page: {docs_dir}/index.html")
    print(f"📊 Dashboard: {output_dir}/dashboard.html")

    # Generate deployment report
    report = {
        "timestamp": datetime.now().isoformat(),
        "files_generated": [
            str(output_dir / "dashboard.html"),
            str(output_dir / "dashboard-data.json"),
            str(docs_dir / "index.html"),
        ],
        "dashboard_data_points": dashboard_data["summary"]["total_data_points"],
        "anomalies_detected": dashboard_data["summary"]["anomalies_detected"],
        "detectors_configured": len(dashboard_data["detectors"]),
        "ready_for_deployment": True,
    }

    with open(output_dir / "generation-report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("📋 Generation report saved to generation-report.json")


if __name__ == "__main__":
    main()
