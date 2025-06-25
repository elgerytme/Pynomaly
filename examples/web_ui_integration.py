#!/usr/bin/env python3
"""
Web UI Integration Example
=========================

This example demonstrates how to integrate with the Pynomaly Progressive Web App,
including real-time monitoring, dashboard creation, and interactive visualization.
"""

import asyncio
import os
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Any

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from pynomaly.infrastructure.config import create_container


class WebUIDataProvider:
    """Provides data for web UI demonstration."""

    def __init__(self):
        self.container = None
        self.detectors = {}
        self.datasets = {}

    async def initialize(self):
        """Initialize the data provider with container and sample data."""
        self.container = create_container()
        await self._setup_sample_detectors()
        await self._setup_sample_datasets()

    async def _setup_sample_detectors(self):
        """Create sample detectors for web UI demonstration."""
        detection_service = self.container.detection_service()

        # Create detectors for different use cases
        detector_configs = [
            {
                "name": "Credit Card Fraud Detector",
                "algorithm": "IsolationForest",
                "parameters": {"contamination": 0.1, "n_estimators": 100},
                "use_case": "fraud_detection",
            },
            {
                "name": "Network Intrusion Detector",
                "algorithm": "LOF",
                "parameters": {"contamination": 0.05, "n_neighbors": 20},
                "use_case": "network_security",
            },
            {
                "name": "IoT Sensor Monitor",
                "algorithm": "OCSVM",
                "parameters": {"contamination": 0.08, "kernel": "rbf"},
                "use_case": "iot_monitoring",
            },
            {
                "name": "Manufacturing Quality Control",
                "algorithm": "COPOD",
                "parameters": {"contamination": 0.15},
                "use_case": "quality_control",
            },
        ]

        for config in detector_configs:
            detector = await detection_service.create_detector(
                name=config["name"],
                algorithm=config["algorithm"],
                parameters=config["parameters"],
            )
            self.detectors[config["use_case"]] = detector

    async def _setup_sample_datasets(self):
        """Create sample datasets for different domains."""
        dataset_service = self.container.dataset_service()

        # Fraud detection dataset
        fraud_data = self._generate_fraud_data(1000)
        fraud_dataset = await dataset_service.create_from_data(
            data=fraud_data,
            name="Credit Card Transactions",
            description="Sample credit card transaction data for fraud detection",
        )
        self.datasets["fraud_detection"] = fraud_dataset

        # Network security dataset
        network_data = self._generate_network_data(800)
        network_dataset = await dataset_service.create_from_data(
            data=network_data,
            name="Network Traffic",
            description="Sample network traffic data for intrusion detection",
        )
        self.datasets["network_security"] = network_dataset

        # IoT sensor data
        iot_data = self._generate_iot_data(1200)
        iot_dataset = await dataset_service.create_from_data(
            data=iot_data,
            name="IoT Sensor Readings",
            description="Sample IoT sensor data for anomaly monitoring",
        )
        self.datasets["iot_monitoring"] = iot_dataset

        # Manufacturing data
        manufacturing_data = self._generate_manufacturing_data(600)
        manufacturing_dataset = await dataset_service.create_from_data(
            data=manufacturing_data,
            name="Manufacturing Metrics",
            description="Sample manufacturing quality control data",
        )
        self.datasets["quality_control"] = manufacturing_dataset

    def _generate_fraud_data(self, n_samples: int) -> list[dict[str, Any]]:
        """Generate synthetic fraud detection data."""
        data = []
        for i in range(n_samples):
            # Normal transactions (90%)
            if random.random() < 0.9:
                amount = random.lognormal(3, 1)  # Log-normal distribution
                merchant_risk = random.uniform(0.1, 0.3)
                time_since_last = random.exponential(2)
                location_risk = random.uniform(0.1, 0.4)
            else:
                # Fraudulent transactions (10%)
                amount = random.lognormal(5, 1.5)  # Higher amounts
                merchant_risk = random.uniform(0.6, 0.9)
                time_since_last = random.exponential(0.5)  # More frequent
                location_risk = random.uniform(0.7, 1.0)

            data.append(
                {
                    "transaction_id": f"txn_{i:06d}",
                    "amount": round(amount, 2),
                    "merchant_risk_score": round(merchant_risk, 3),
                    "hours_since_last_txn": round(time_since_last, 2),
                    "location_risk_score": round(location_risk, 3),
                    "is_weekend": random.choice([0, 1]),
                    "hour_of_day": random.randint(0, 23),
                }
            )
        return data

    def _generate_network_data(self, n_samples: int) -> list[dict[str, Any]]:
        """Generate synthetic network traffic data."""
        data = []
        for i in range(n_samples):
            # Normal traffic (95%)
            if random.random() < 0.95:
                packet_size = random.normal(1500, 300)
                connection_duration = random.exponential(30)
                bytes_transferred = random.lognormal(8, 2)
                unique_ports = random.randint(1, 5)
            else:
                # Anomalous traffic (5%)
                packet_size = random.normal(3000, 500)  # Larger packets
                connection_duration = random.exponential(5)  # Shorter connections
                bytes_transferred = random.lognormal(12, 3)  # More data
                unique_ports = random.randint(10, 50)  # Port scanning

            data.append(
                {
                    "connection_id": f"conn_{i:06d}",
                    "avg_packet_size": max(64, round(packet_size)),
                    "connection_duration_sec": round(connection_duration, 2),
                    "total_bytes": int(max(0, bytes_transferred)),
                    "unique_destination_ports": unique_ports,
                    "protocol_variety": random.randint(1, 3),
                    "failed_connection_attempts": random.randint(0, 2),
                }
            )
        return data

    def _generate_iot_data(self, n_samples: int) -> list[dict[str, Any]]:
        """Generate synthetic IoT sensor data."""
        data = []
        for i in range(n_samples):
            # Normal sensor readings (92%)
            if random.random() < 0.92:
                temperature = random.normal(22, 2)
                humidity = random.normal(45, 5)
                pressure = random.normal(1013, 10)
                vibration = random.exponential(0.1)
            else:
                # Sensor anomalies (8%)
                temperature = random.choice(
                    [
                        random.normal(50, 5),  # Overheating
                        random.normal(-10, 3),  # Cooling failure
                    ]
                )
                humidity = random.choice(
                    [
                        random.normal(80, 5),  # High humidity
                        random.normal(10, 2),  # Low humidity
                    ]
                )
                pressure = random.normal(1013, 50)  # Pressure variations
                vibration = random.exponential(1.0)  # High vibration

            data.append(
                {
                    "sensor_id": f"sensor_{i % 10:02d}",
                    "timestamp": (
                        datetime.now() - timedelta(hours=i // 10)
                    ).isoformat(),
                    "temperature_celsius": round(temperature, 2),
                    "humidity_percent": max(0, min(100, round(humidity, 1))),
                    "pressure_hpa": round(pressure, 1),
                    "vibration_level": round(vibration, 4),
                    "battery_level": max(0, min(100, random.normal(85, 10))),
                }
            )
        return data

    def _generate_manufacturing_data(self, n_samples: int) -> list[dict[str, Any]]:
        """Generate synthetic manufacturing quality data."""
        data = []
        for i in range(n_samples):
            # Normal production (85%)
            if random.random() < 0.85:
                dimension_1 = random.normal(10.0, 0.1)
                dimension_2 = random.normal(5.0, 0.05)
                surface_roughness = random.exponential(0.2)
                hardness = random.normal(60, 2)
                weight = random.normal(100, 1)
            else:
                # Quality defects (15%)
                dimension_1 = random.normal(10.0, 0.5)  # Higher variance
                dimension_2 = random.normal(5.0, 0.3)  # Higher variance
                surface_roughness = random.exponential(1.0)  # Rougher surface
                hardness = random.choice(
                    [
                        random.normal(45, 5),  # Too soft
                        random.normal(80, 5),  # Too hard
                    ]
                )
                weight = random.normal(100, 5)  # Weight variations

            data.append(
                {
                    "part_id": f"part_{i:06d}",
                    "batch_number": f"batch_{i // 50:03d}",
                    "dimension_1_mm": round(dimension_1, 3),
                    "dimension_2_mm": round(dimension_2, 3),
                    "surface_roughness_ra": round(surface_roughness, 4),
                    "hardness_hrc": round(hardness, 1),
                    "weight_grams": round(weight, 2),
                    "production_line": random.randint(1, 5),
                    "shift": random.choice(["A", "B", "C"]),
                }
            )
        return data


async def create_web_dashboard_config():
    """Create configuration for web dashboard."""
    print("🎛️ Creating Web Dashboard Configuration")
    print("=" * 50)

    provider = WebUIDataProvider()
    await provider.initialize()

    # Dashboard configuration
    dashboard_config = {
        "dashboard_name": "Pynomaly Multi-Domain Monitoring",
        "refresh_interval_seconds": 30,
        "tiles": [
            {
                "id": "fraud_tile",
                "title": "Credit Card Fraud Detection",
                "type": "anomaly_monitor",
                "detector_id": provider.detectors["fraud_detection"].id,
                "dataset_id": provider.datasets["fraud_detection"].id,
                "visualization": "real_time_alerts",
                "alert_threshold": 0.8,
                "position": {"row": 1, "col": 1, "width": 2, "height": 1},
            },
            {
                "id": "network_tile",
                "title": "Network Intrusion Detection",
                "type": "anomaly_monitor",
                "detector_id": provider.detectors["network_security"].id,
                "dataset_id": provider.datasets["network_security"].id,
                "visualization": "time_series_chart",
                "alert_threshold": 0.7,
                "position": {"row": 1, "col": 3, "width": 2, "height": 1},
            },
            {
                "id": "iot_tile",
                "title": "IoT Sensor Monitoring",
                "type": "multi_sensor_view",
                "detector_id": provider.detectors["iot_monitoring"].id,
                "dataset_id": provider.datasets["iot_monitoring"].id,
                "visualization": "sensor_grid",
                "alert_threshold": 0.6,
                "position": {"row": 2, "col": 1, "width": 3, "height": 1},
            },
            {
                "id": "quality_tile",
                "title": "Manufacturing Quality Control",
                "type": "quality_control",
                "detector_id": provider.detectors["quality_control"].id,
                "dataset_id": provider.datasets["quality_control"].id,
                "visualization": "control_chart",
                "alert_threshold": 0.9,
                "position": {"row": 2, "col": 4, "width": 1, "height": 1},
            },
            {
                "id": "summary_tile",
                "title": "System Overview",
                "type": "summary_statistics",
                "visualization": "metrics_grid",
                "position": {"row": 3, "col": 1, "width": 4, "height": 1},
                "metrics": [
                    "total_detectors",
                    "active_datasets",
                    "anomalies_last_hour",
                    "system_health",
                ],
            },
        ],
        "alerts": {
            "enabled": True,
            "notification_channels": ["web_notification", "email"],
            "severity_levels": {
                "low": {"threshold": 0.6, "color": "#FFA500"},
                "medium": {"threshold": 0.8, "color": "#FF6B35"},
                "high": {"threshold": 0.9, "color": "#DC143C"},
            },
        },
        "data_retention": {
            "real_time_hours": 24,
            "historical_days": 30,
            "archive_months": 12,
        },
    }

    print("✅ Dashboard configuration created!")
    print(f"   - {len(dashboard_config['tiles'])} dashboard tiles")
    print(f"   - {len(provider.detectors)} detectors configured")
    print(f"   - {len(provider.datasets)} datasets available")

    return dashboard_config, provider


async def simulate_real_time_monitoring():
    """Simulate real-time monitoring for web UI."""
    print("\n📡 Simulating Real-Time Monitoring")
    print("=" * 50)

    dashboard_config, provider = await create_web_dashboard_config()
    detection_service = provider.container.detection_service()

    # Train detectors
    print("🎯 Training detectors...")
    for use_case, detector in provider.detectors.items():
        dataset = provider.datasets[use_case]
        await detection_service.train_detector(detector.id, dataset.id)
        print(f"   ✅ {detector.name} trained")

    # Simulate real-time data stream
    print("\n📊 Starting real-time simulation (60 seconds)...")

    monitoring_data = {
        "timestamp": [],
        "fraud_alerts": [],
        "network_alerts": [],
        "iot_alerts": [],
        "quality_alerts": [],
        "total_alerts": [],
    }

    start_time = time.time()
    simulation_duration = 60

    while (time.time() - start_time) < simulation_duration:
        current_time = datetime.now()
        alerts_this_cycle = {"fraud": 0, "network": 0, "iot": 0, "quality": 0}

        # Simulate incoming data for each domain
        for use_case, detector in provider.detectors.items():
            # Generate new data points
            if use_case == "fraud_detection":
                new_data = provider._generate_fraud_data(5)
            elif use_case == "network_security":
                new_data = provider._generate_network_data(3)
            elif use_case == "iot_monitoring":
                new_data = provider._generate_iot_data(8)
            else:  # quality_control
                new_data = provider._generate_manufacturing_data(2)

            # Run detection
            try:
                results = await detection_service.detect_batch(detector.id, new_data)
                anomalies = sum(1 for r in results if r.is_anomaly)
                alerts_this_cycle[use_case.split("_")[0]] = anomalies

                # Log significant alerts
                if anomalies > 0:
                    high_score_results = [
                        r
                        for r in results
                        if r.is_anomaly and r.anomaly_score.value > 0.8
                    ]
                    if high_score_results:
                        timestamp_str = current_time.strftime("%H:%M:%S")
                        print(
                            f"   🚨 [{timestamp_str}] {detector.name}: {len(high_score_results)} high-confidence alerts"
                        )

            except Exception as e:
                print(f"   ❌ Error in {use_case}: {e}")

        # Record monitoring data
        monitoring_data["timestamp"].append(current_time)
        monitoring_data["fraud_alerts"].append(alerts_this_cycle["fraud"])
        monitoring_data["network_alerts"].append(alerts_this_cycle["network"])
        monitoring_data["iot_alerts"].append(alerts_this_cycle["iot"])
        monitoring_data["quality_alerts"].append(alerts_this_cycle["quality"])
        monitoring_data["total_alerts"].append(sum(alerts_this_cycle.values()))

        # Wait for next cycle
        await asyncio.sleep(3)  # 3-second cycles

    # Summary statistics
    total_cycles = len(monitoring_data["timestamp"])
    total_fraud_alerts = sum(monitoring_data["fraud_alerts"])
    total_network_alerts = sum(monitoring_data["network_alerts"])
    total_iot_alerts = sum(monitoring_data["iot_alerts"])
    total_quality_alerts = sum(monitoring_data["quality_alerts"])
    total_all_alerts = sum(monitoring_data["total_alerts"])

    print(f"\n📊 Real-Time Monitoring Summary ({total_cycles} cycles)")
    print("=" * 50)
    print(f"   Fraud Detection Alerts:     {total_fraud_alerts}")
    print(f"   Network Security Alerts:    {total_network_alerts}")
    print(f"   IoT Monitoring Alerts:      {total_iot_alerts}")
    print(f"   Quality Control Alerts:     {total_quality_alerts}")
    print(f"   Total Alerts:               {total_all_alerts}")
    print(f"   Average Alerts per Cycle:   {total_all_alerts / total_cycles:.1f}")

    return monitoring_data


async def generate_web_api_endpoints():
    """Generate example API endpoints for web UI integration."""
    print("\n🌐 Web API Endpoints for UI Integration")
    print("=" * 50)

    # Example REST API endpoints that the web UI would call
    api_endpoints = {
        "dashboard": {
            "GET /api/dashboard/config": {
                "description": "Get dashboard configuration",
                "response_example": {
                    "dashboard_name": "Pynomaly Multi-Domain Monitoring",
                    "tiles": ["fraud_tile", "network_tile", "iot_tile"],
                    "refresh_interval": 30,
                },
            },
            "GET /api/dashboard/status": {
                "description": "Get real-time dashboard status",
                "response_example": {
                    "active_detectors": 4,
                    "total_alerts_last_hour": 23,
                    "system_health": "healthy",
                    "last_updated": "2024-01-01T12:00:00Z",
                },
            },
        },
        "detectors": {
            "GET /api/detectors": {
                "description": "List all detectors",
                "response_example": [
                    {
                        "id": "detector_123",
                        "name": "Credit Card Fraud Detector",
                        "algorithm": "IsolationForest",
                        "status": "trained",
                        "last_used": "2024-01-01T11:30:00Z",
                    }
                ],
            },
            "POST /api/detectors/{id}/detect": {
                "description": "Run detection on new data",
                "request_example": {
                    "data": [
                        {"amount": 150.0, "merchant_risk_score": 0.3},
                        {"amount": 5000.0, "merchant_risk_score": 0.9},
                    ]
                },
                "response_example": {
                    "results": [
                        {
                            "is_anomaly": False,
                            "score": 0.3,
                            "explanation": "Normal transaction",
                        },
                        {
                            "is_anomaly": True,
                            "score": 0.95,
                            "explanation": "High-risk transaction",
                        },
                    ]
                },
            },
        },
        "alerts": {
            "GET /api/alerts/recent": {
                "description": "Get recent alerts",
                "query_params": ["limit", "severity", "detector_id"],
                "response_example": [
                    {
                        "id": "alert_456",
                        "detector_name": "Network Intrusion Detector",
                        "severity": "high",
                        "message": "Potential port scanning detected",
                        "timestamp": "2024-01-01T12:00:00Z",
                        "anomaly_score": 0.92,
                    }
                ],
            },
            "POST /api/alerts/{id}/acknowledge": {
                "description": "Acknowledge an alert",
                "request_example": {
                    "acknowledged_by": "user@example.com",
                    "notes": "Investigated - false positive",
                },
            },
        },
        "datasets": {
            "GET /api/datasets/{id}/stats": {
                "description": "Get dataset statistics",
                "response_example": {
                    "total_samples": 1000,
                    "features": ["amount", "merchant_risk_score"],
                    "anomaly_rate": 0.08,
                    "last_updated": "2024-01-01T10:00:00Z",
                },
            },
            "POST /api/datasets/{id}/upload": {
                "description": "Upload new data to dataset",
                "content_type": "multipart/form-data",
                "response_example": {
                    "samples_added": 100,
                    "processing_status": "completed",
                },
            },
        },
        "visualization": {
            "GET /api/visualization/time-series": {
                "description": "Get time series data for charts",
                "query_params": [
                    "detector_id",
                    "start_time",
                    "end_time",
                    "granularity",
                ],
                "response_example": {
                    "timestamps": ["2024-01-01T10:00:00Z", "2024-01-01T11:00:00Z"],
                    "anomaly_scores": [0.2, 0.8],
                    "anomaly_flags": [False, True],
                },
            },
            "GET /api/visualization/distribution": {
                "description": "Get data distribution for histograms",
                "response_example": {
                    "feature": "amount",
                    "bins": [0, 100, 200, 500, 1000],
                    "counts": [450, 300, 180, 70],
                    "anomaly_counts": [2, 5, 8, 15],
                },
            },
        },
        "websocket": {
            "WS /ws/real-time-alerts": {
                "description": "WebSocket for real-time alert notifications",
                "message_example": {
                    "type": "alert",
                    "detector_id": "detector_123",
                    "severity": "high",
                    "data": {
                        "anomaly_score": 0.95,
                        "explanation": "Suspicious transaction pattern",
                    },
                },
            },
            "WS /ws/dashboard-updates": {
                "description": "WebSocket for dashboard tile updates",
                "message_example": {
                    "type": "tile_update",
                    "tile_id": "fraud_tile",
                    "data": {
                        "alerts_count": 3,
                        "last_alert_time": "2024-01-01T12:00:00Z",
                    },
                },
            },
        },
    }

    print("📋 Generated API Endpoints:")
    for category, endpoints in api_endpoints.items():
        print(f"\n   {category.upper()}:")
        for endpoint, details in endpoints.items():
            print(f"     {endpoint}")
            print(f"       → {details['description']}")

    return api_endpoints


async def create_htmx_examples():
    """Create examples of HTMX integration patterns."""
    print("\n⚡ HTMX Integration Examples")
    print("=" * 50)

    htmx_examples = {
        "real_time_alerts": """
<!-- Real-time alert panel that updates every 5 seconds -->
<div id="alert-panel" 
     hx-get="/api/alerts/recent?limit=5" 
     hx-trigger="every 5s" 
     hx-swap="innerHTML">
    <div class="loading">Loading alerts...</div>
</div>

<!-- Alert template (server-rendered) -->
<!-- This would be returned by the /api/alerts/recent endpoint -->
<div class="alert alert-{{severity}}">
    <div class="alert-header">
        <span class="detector-name">{{detector_name}}</span>
        <span class="timestamp">{{timestamp}}</span>
        <span class="score">Score: {{anomaly_score}}</span>
    </div>
    <div class="alert-message">{{message}}</div>
    <button hx-post="/api/alerts/{{id}}/acknowledge" 
            hx-swap="outerHTML" 
            hx-confirm="Acknowledge this alert?">
        Acknowledge
    </button>
</div>
        """,
        "detector_status": """
<!-- Detector status cards with live updates -->
<div class="detector-grid">
    <div hx-get="/api/detectors" 
         hx-trigger="load, every 30s" 
         hx-swap="innerHTML">
        Loading detectors...
    </div>
</div>

<!-- Detector card template -->
<div class="detector-card {{status}}">
    <h3>{{name}}</h3>
    <div class="status-indicator">
        <span class="status-dot {{status}}"></span>
        {{status}}
    </div>
    <div class="metrics">
        <span>Algorithm: {{algorithm}}</span>
        <span>Last used: {{last_used}}</span>
    </div>
    <button hx-post="/api/detectors/{{id}}/retrain" 
            hx-swap="closest .detector-card"
            hx-confirm="Retrain this detector?">
        Retrain
    </button>
</div>
        """,
        "data_upload": """
<!-- Drag and drop file upload with progress -->
<div id="upload-zone" 
     hx-encoding="multipart/form-data"
     hx-post="/api/datasets/{{dataset_id}}/upload"
     hx-swap="innerHTML"
     class="upload-zone">
    
    <form>
        <input type="file" name="datafile" accept=".csv,.json" required>
        <div class="upload-progress" style="display: none;">
            <div class="progress-bar"></div>
            <span class="progress-text">Uploading...</span>
        </div>
        <button type="submit">Upload Data</button>
    </form>
</div>

<!-- Upload success response -->
<div class="upload-success">
    <h4>✅ Upload Successful</h4>
    <p>Added {{samples_added}} samples to dataset</p>
    <button hx-get="/dashboard" hx-target="body">
        Return to Dashboard
    </button>
</div>
        """,
        "anomaly_details": """
<!-- Expandable anomaly details -->
<div class="anomaly-item">
    <div class="anomaly-summary" 
         hx-get="/api/anomalies/{{id}}/details" 
         hx-trigger="click"
         hx-target="next .anomaly-details"
         hx-swap="innerHTML">
        <span class="score">{{score}}</span>
        <span class="timestamp">{{timestamp}}</span>
        <span class="detector">{{detector_name}}</span>
        <span class="expand-icon">▶</span>
    </div>
    <div class="anomaly-details">
        <!-- Details loaded on click -->
    </div>
</div>

<!-- Detailed view template -->
<div class="anomaly-detail-content">
    <h4>Anomaly Analysis</h4>
    <div class="explanation">{{explanation}}</div>
    <div class="feature-contributions">
        {{#each feature_contributions}}
        <div class="feature-item">
            <span class="feature-name">{{name}}</span>
            <div class="contribution-bar" style="width: {{percentage}}%"></div>
            <span class="contribution-value">{{value}}</span>
        </div>
        {{/each}}
    </div>
    <div class="actions">
        <button hx-post="/api/anomalies/{{id}}/flag-false-positive">
            Mark as False Positive
        </button>
    </div>
</div>
        """,
        "live_chart_updates": """
<!-- Live updating chart with server-sent events -->
<div id="anomaly-chart" 
     hx-ext="sse" 
     sse-connect="/sse/anomaly-scores/{{detector_id}}"
     sse-swap="chart-update">
    
    <canvas id="chart-canvas" width="800" height="400"></canvas>
    
    <!-- Chart data template (updates via SSE) -->
    <script type="application/json" id="chart-data">
    {
        "timestamps": {{timestamps}},
        "scores": {{anomaly_scores}},
        "alerts": {{alert_flags}}
    }
    </script>
</div>

<!-- JavaScript to update chart -->
<script>
document.body.addEventListener('htmx:sseMessage', function(e) {
    if (e.detail.type === 'chart-update') {
        const data = JSON.parse(e.detail.data);
        updateAnomalyChart(data);
    }
});
</script>
        """,
    }

    print("⚡ HTMX Integration Patterns:")
    for pattern, code in htmx_examples.items():
        print(f"\n   📝 {pattern.replace('_', ' ').title()}:")
        print("      - Server-side rendering with dynamic updates")
        print("      - Minimal JavaScript required")
        print("      - Progressive enhancement")

    return htmx_examples


if __name__ == "__main__":
    print("🌐 Pynomaly Web UI Integration Examples")
    print("=" * 60)

    # Run examples
    asyncio.run(create_web_dashboard_config())
    asyncio.run(simulate_real_time_monitoring())
    asyncio.run(generate_web_api_endpoints())
    asyncio.run(create_htmx_examples())

    print("\n✅ Web UI integration examples completed!")
    print("\nKey features demonstrated:")
    print("- Dashboard configuration for multi-domain monitoring")
    print("- Real-time anomaly detection simulation")
    print("- REST API endpoints for web UI integration")
    print("- HTMX patterns for progressive web app functionality")
    print("- Server-side rendering with dynamic updates")
    print("\nNext steps:")
    print("- Implement WebSocket connections for real-time updates")
    print("- Add authentication and authorization")
    print("- Create responsive CSS with Tailwind")
    print("- Implement offline capability with service workers")
    print("- Add push notifications for critical alerts")
