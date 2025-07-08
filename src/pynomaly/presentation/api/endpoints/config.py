"""FastAPI endpoints for managing anomaly detection configuration.

This module exposes CRUD endpoints for managing the configuration of:
- Default thresholds per metric
- Detection algorithm selection
- Alert escalation rules
"""

from fastapi import FastAPI, HTTPException
from pynomaly.infrastructure.config.anomaly_config import (
    AnomalyDetectionConfig,
    MetricThreshold,
    AlgorithmConfig,
    AlertRuleConfig,
    create_default_config,
)

app = FastAPI()
# Initialize with default configuration
config = create_default_config()


@app.get("/config")
def get_config() -> AnomalyDetectionConfig:
    """Retrieve the current configuration."""
    return config


@app.get("/config/thresholds/{metric_name}")
def get_metric_threshold(metric_name: str) -> MetricThreshold:
    """Retrieve threshold for a specific metric."""
    if metric_name in config.default_thresholds:
        return config.default_thresholds[metric_name]
    raise HTTPException(status_code=404, detail="Metric threshold not found")


@app.put("/config/thresholds/{metric_name}")
def update_metric_threshold(metric_name: str, threshold: MetricThreshold) -> str:
    """Update threshold for a specific metric."""
    if metric_name in config.default_thresholds:
        config.default_thresholds[metric_name] = threshold
        return "Metric threshold updated"
    raise HTTPException(status_code=404, detail="Metric threshold not found")


@app.get("/config/algorithms/{algorithm_name}")
def get_algorithm_config(algorithm_name: str) -> AlgorithmConfig:
    """Retrieve algorithm configuration."""
    if algorithm_name in config.algorithms:
        return config.algorithms[algorithm_name]
    raise HTTPException(status_code=404, detail="Algorithm not found")


@app.put("/config/algorithms/{algorithm_name}")
def update_algorithm_config(algorithm_name: str, algorithm: AlgorithmConfig) -> str:
    """Update algorithm configuration."""
    if algorithm_name in config.algorithms:
        config.algorithms[algorithm_name] = algorithm
        return "Algorithm configuration updated"
    raise HTTPException(status_code=404, detail="Algorithm not found")


@app.get("/config/alerts/{rule_id}")
def get_alert_rule(rule_id: str) -> AlertRuleConfig:
    """Retrieve alert rule configuration."""
    if rule_id in config.alert_rules:
        return config.alert_rules[rule_id]
    raise HTTPException(status_code=404, detail="Alert rule not found")


@app.put("/config/alerts/{rule_id}")
def update_alert_rule(rule_id: str, alert_rule: AlertRuleConfig) -> str:
    """Update alert rule configuration."""
    if rule_id in config.alert_rules:
        config.alert_rules[rule_id] = alert_rule
        return "Alert rule configuration updated"
    raise HTTPException(status_code=404, detail="Alert rule not found")

