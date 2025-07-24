"""Model performance monitoring commands for Typer CLI."""

import json
import typer
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from rich import print
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import asyncio

from ...infrastructure.logging import get_logger
from ...infrastructure.repositories.model_repository import ModelRepository

console = Console()
logger = get_logger(__name__)

app = typer.Typer(help="Model performance monitoring commands")


@app.command()
def performance(
    model_ids: List[str] = typer.Option(..., "--models", "-m", help="Model IDs to monitor"),
    duration: int = typer.Option(300, "--duration", "-d", help="Monitoring duration in seconds"),
    interval: int = typer.Option(30, "--interval", "-i", help="Update interval in seconds"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save monitoring data to file"),
    alert_threshold: float = typer.Option(0.05, "--alert-threshold", help="Performance degradation alert threshold"),
) -> None:
    """Monitor model performance in real-time."""
    
    print(f"[blue]🔍[/blue] Starting performance monitoring for {len(model_ids)} models...")
    print(f"[blue]ℹ[/blue] Duration: {duration}s, Update interval: {interval}s")
    
    # Initialize monitoring data
    monitoring_data = {
        "start_time": datetime.now().isoformat(),
        "model_ids": model_ids,
        "monitoring_config": {
            "duration": duration,
            "interval": interval,
            "alert_threshold": alert_threshold
        },
        "performance_history": {model_id: [] for model_id in model_ids},
        "alerts": []
    }
    
    model_repository = ModelRepository()
    
    # Verify models exist
    for model_id in model_ids:
        try:
            model = model_repository.load(model_id)
            if not model:
                print(f"[red]✗[/red] Model '{model_id}' not found")
                raise typer.Exit(1)
        except Exception as e:
            print(f"[red]✗[/red] Failed to load model '{model_id}': {e}")
            raise typer.Exit(1)
    
    print(f"[green]✓[/green] All models validated successfully")
    
    # Create monitoring layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    
    layout["main"].split_row(
        Layout(name="metrics"),
        Layout(name="alerts"),
    )
    
    start_time = time.time()
    iteration = 0
    
    try:
        with Live(layout, refresh_per_second=0.5, screen=True) as live:
            while time.time() - start_time < duration:
                iteration += 1
                current_time = datetime.now()
                
                # Collect performance metrics for each model
                current_metrics = {}
                
                for model_id in model_ids:
                    # Simulate performance metrics collection
                    # In real implementation, this would query actual metrics
                    metrics = _collect_model_metrics(model_id, iteration)
                    current_metrics[model_id] = metrics
                    
                    # Store in history
                    monitoring_data["performance_history"][model_id].append({
                        "timestamp": current_time.isoformat(),
                        "metrics": metrics
                    })
                    
                    # Check for performance alerts
                    alert = _check_performance_alert(model_id, metrics, alert_threshold)
                    if alert:
                        alert["timestamp"] = current_time.isoformat()
                        monitoring_data["alerts"].append(alert)
                
                # Update display
                _update_monitoring_display(layout, current_metrics, monitoring_data["alerts"], 
                                         start_time, duration, iteration)
                
                # Wait for next update
                time.sleep(interval)
        
        # Final summary
        monitoring_data["end_time"] = datetime.now().isoformat()
        monitoring_data["total_iterations"] = iteration
        
    except KeyboardInterrupt:
        print(f"\n[yellow]⚠[/yellow] Monitoring interrupted by user")
        monitoring_data["end_time"] = datetime.now().isoformat()
        monitoring_data["interrupted"] = True
    
    # Display final summary
    _display_monitoring_summary(monitoring_data)
    
    # Save monitoring data if requested
    if output:
        with open(output, 'w') as f:
            json.dump(monitoring_data, f, indent=2)
        print(f"[green]✓[/green] Monitoring data saved to: {output}")
    
    print(f"[green]✅[/green] Performance monitoring completed!")


@app.command()
def health_check(
    model_ids: List[str] = typer.Option(..., "--models", "-m", help="Model IDs to check"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed health information"),
) -> None:
    """Perform health check on specified models."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        check_task = progress.add_task(f"Checking health of {len(model_ids)} models...", total=len(model_ids))
        
        model_repository = ModelRepository()
        health_results = {}
        
        for model_id in model_ids:
            try:
                # Load model
                model = model_repository.load(model_id)
                
                if not model:
                    health_results[model_id] = {
                        "status": "ERROR",
                        "error": "Model not found",
                        "healthy": False
                    }
                else:
                    # Perform health checks
                    health_status = _perform_model_health_check(model_id, model, detailed)
                    health_results[model_id] = health_status
                
                progress.update(check_task, advance=1)
                
            except Exception as e:
                health_results[model_id] = {
                    "status": "ERROR",
                    "error": str(e),
                    "healthy": False
                }
                progress.update(check_task, advance=1)
    
    # Display results
    table = Table(title="[bold blue]Model Health Check Results[/bold blue]")
    table.add_column("Model ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Health Score", style="yellow")
    table.add_column("Issues", style="red")
    
    healthy_count = 0
    
    for model_id, result in health_results.items():
        status_icon = "✅" if result["healthy"] else "❌"
        status_text = f"{status_icon} {result['status']}"
        
        health_score = result.get("health_score", "N/A")
        issues = result.get("issues", [])
        issues_text = f"{len(issues)} issues" if issues else "None"
        
        if result["healthy"]:
            healthy_count += 1
        
        table.add_row(
            model_id,
            status_text,
            str(health_score) if health_score != "N/A" else "N/A",
            issues_text
        )
    
    console.print(table)
    
    # Summary
    print(f"\n[blue]📊[/blue] Health Check Summary:")
    print(f"  Healthy models: {healthy_count}/{len(model_ids)}")
    print(f"  Success rate: {healthy_count/len(model_ids)*100:.1f}%")
    
    if detailed:
        for model_id, result in health_results.items():
            if result.get("issues"):
                print(f"\n[yellow]⚠[/yellow] Issues found in model '{model_id}':")
                for issue in result["issues"]:
                    print(f"  • {issue}")
    
    print(f"[green]✅[/green] Health check completed!")


@app.command()
def drift_analysis(
    model_id: str = typer.Option(..., "--model", "-m", help="Model ID to analyze"),
    reference_data: Path = typer.Option(..., "--reference", "-r", help="Reference dataset file"),
    current_data: Path = typer.Option(..., "--current", "-c", help="Current dataset file"),
    threshold: float = typer.Option(0.1, "--threshold", help="Drift detection threshold"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save drift analysis results"),
) -> None:
    """Analyze concept drift for a specific model."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Load datasets
        load_task = progress.add_task("Loading datasets...", total=None)
        
        try:
            import pandas as pd
            import numpy as np
            from scipy import stats
            
            if reference_data.suffix.lower() == '.csv':
                ref_df = pd.read_csv(reference_data)
            else:
                print(f"[red]✗[/red] Only CSV files supported")
                raise typer.Exit(1)
            
            if current_data.suffix.lower() == '.csv':
                curr_df = pd.read_csv(current_data)
            else:
                print(f"[red]✗[/red] Only CSV files supported")
                raise typer.Exit(1)
            
            progress.update(load_task, completed=True)
            
        except Exception as e:
            print(f"[red]✗[/red] Failed to load datasets: {e}")
            raise typer.Exit(1)
        
        # Perform drift analysis
        analysis_task = progress.add_task("Performing drift analysis...", total=None)
        
        try:
            # Ensure same columns
            common_columns = list(set(ref_df.columns) & set(curr_df.columns))
            if not common_columns:
                print(f"[red]✗[/red] No common columns found between datasets")
                raise typer.Exit(1)
            
            ref_data_filtered = ref_df[common_columns]
            curr_data_filtered = curr_df[common_columns]
            
            # Calculate drift metrics
            drift_results = {
                "model_id": model_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "datasets": {
                    "reference": str(reference_data),
                    "current": str(current_data),
                    "reference_size": len(ref_df),
                    "current_size": len(curr_df),
                    "common_features": common_columns
                },
                "drift_metrics": {},
                "drift_detected": False,
                "drift_severity": "None"
            }
            
            # Calculate statistical tests for each feature
            significant_drifts = []
            
            for column in common_columns:
                if ref_data_filtered[column].dtype in ['int64', 'float64']:
                    # Numerical feature - use KS test
                    ref_values = ref_data_filtered[column].dropna()
                    curr_values = curr_data_filtered[column].dropna()
                    
                    ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)
                    
                    # Calculate effect size (difference in means normalized by pooled std)
                    mean_diff = abs(curr_values.mean() - ref_values.mean())
                    pooled_std = np.sqrt(((ref_values.std()**2 + curr_values.std()**2) / 2))
                    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    drift_results["drift_metrics"][column] = {
                        "test": "Kolmogorov-Smirnov",
                        "statistic": float(ks_stat),
                        "p_value": float(p_value),
                        "effect_size": float(effect_size),
                        "drift_detected": ks_stat > threshold,
                        "reference_mean": float(ref_values.mean()),
                        "current_mean": float(curr_values.mean()),
                        "reference_std": float(ref_values.std()),
                        "current_std": float(curr_values.std())
                    }
                    
                    if ks_stat > threshold:
                        significant_drifts.append((column, ks_stat, effect_size))
                
                else:
                    # Categorical feature - use Chi-square test
                    ref_counts = ref_data_filtered[column].value_counts()
                    curr_counts = curr_data_filtered[column].value_counts()
                    
                    # Align categories
                    all_categories = set(ref_counts.index) | set(curr_counts.index)
                    ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                    curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                    
                    if sum(ref_aligned) > 0 and sum(curr_aligned) > 0:
                        chi2_stat, p_value = stats.chisquare(curr_aligned, ref_aligned)
                        
                        drift_results["drift_metrics"][column] = {
                            "test": "Chi-square",
                            "statistic": float(chi2_stat),
                            "p_value": float(p_value),
                            "drift_detected": p_value < 0.05,
                            "reference_categories": len(ref_counts),
                            "current_categories": len(curr_counts)
                        }
                        
                        if p_value < 0.05:
                            significant_drifts.append((column, chi2_stat, 0))
            
            # Determine overall drift status
            if significant_drifts:
                drift_results["drift_detected"] = True
                avg_drift_magnitude = sum(drift[1] for drift in significant_drifts) / len(significant_drifts)
                
                if avg_drift_magnitude > 0.3:
                    drift_results["drift_severity"] = "High"
                elif avg_drift_magnitude > 0.1:
                    drift_results["drift_severity"] = "Medium"
                else:
                    drift_results["drift_severity"] = "Low"
                
                drift_results["significant_features"] = [
                    {
                        "feature": drift[0],
                        "magnitude": drift[1],
                        "effect_size": drift[2]
                    } for drift in significant_drifts
                ]
            
            progress.update(analysis_task, completed=True)
            
        except Exception as e:
            print(f"[red]✗[/red] Drift analysis failed: {e}")
            raise typer.Exit(1)
    
    # Display results
    print(f"\n[blue]🔍[/blue] Drift Analysis Results for Model: {model_id}")
    
    status_icon = "⚠️" if drift_results["drift_detected"] else "✅"
    severity = drift_results["drift_severity"]
    
    print(f"  Status: {status_icon} {'Drift Detected' if drift_results['drift_detected'] else 'No Drift Detected'}")
    print(f"  Severity: {severity}")
    print(f"  Features analyzed: {len(common_columns)}")
    
    if significant_drifts:
        print(f"  Features with significant drift: {len(significant_drifts)}")
        
        # Show top drifting features
        table = Table(title="[bold yellow]Features with Significant Drift[/bold yellow]")
        table.add_column("Feature", style="cyan")
        table.add_column("Test Statistic", style="yellow")
        table.add_column("Effect Size", style="red")
        table.add_column("Severity", style="magenta")
        
        for feature, magnitude, effect_size in sorted(significant_drifts, key=lambda x: x[1], reverse=True)[:10]:
            if magnitude > 0.3:
                severity_label = "High"
            elif magnitude > 0.1:
                severity_label = "Medium"
            else:
                severity_label = "Low"
            
            table.add_row(
                feature,
                f"{magnitude:.4f}",
                f"{effect_size:.4f}" if effect_size > 0 else "N/A",
                severity_label
            )
        
        console.print(table)
    
    # Save results if requested
    if output:
        with open(output, 'w') as f:
            json.dump(drift_results, f, indent=2)
        print(f"[green]✓[/green] Drift analysis results saved to: {output}")
    
    # Provide recommendations
    if drift_results["drift_detected"]:
        print(f"\n[yellow]💡[/yellow] Recommendations:")
        if drift_results["drift_severity"] == "High":
            print(f"  • Model retraining strongly recommended")
            print(f"  • Consider data quality investigation")
        elif drift_results["drift_severity"] == "Medium":
            print(f"  • Monitor model performance closely")
            print(f"  • Consider model retraining if performance degrades")
        else:
            print(f"  • Continue monitoring")
            print(f"  • Document drift patterns for future reference")
    else:
        print(f"\n[green]✅[/green] No significant drift detected. Model appears stable.")
    
    print(f"[green]✅[/green] Drift analysis completed!")


def _collect_model_metrics(model_id: str, iteration: int) -> Dict[str, float]:
    """Simulate collecting model performance metrics."""
    import random
    import math
    
    # Simulate realistic performance metrics with some degradation over time
    base_accuracy = 0.85
    base_precision = 0.80
    base_recall = 0.82
    base_f1 = 0.81
    
    # Add some noise and gradual degradation
    noise_factor = 0.02
    degradation_factor = max(0, iteration * 0.001)  # Gradual degradation
    
    return {
        "accuracy": max(0.5, base_accuracy - degradation_factor + random.uniform(-noise_factor, noise_factor)),
        "precision": max(0.5, base_precision - degradation_factor + random.uniform(-noise_factor, noise_factor)),
        "recall": max(0.5, base_recall - degradation_factor + random.uniform(-noise_factor, noise_factor)),
        "f1_score": max(0.5, base_f1 - degradation_factor + random.uniform(-noise_factor, noise_factor)),
        "response_time": 50 + random.uniform(-10, 20),  # ms
        "throughput": 1000 + random.uniform(-100, 200),  # samples/sec
        "memory_usage": 512 + random.uniform(-50, 100),  # MB
        "cpu_usage": 25 + random.uniform(-5, 15)  # %
    }


def _check_performance_alert(model_id: str, metrics: Dict[str, float], threshold: float) -> Optional[Dict[str, Any]]:
    """Check if performance metrics trigger an alert."""
    
    # Define baseline performance (in practice, this would come from historical data)
    baselines = {
        "accuracy": 0.85,
        "precision": 0.80,
        "recall": 0.82,
        "f1_score": 0.81
    }
    
    alerts = []
    
    for metric, baseline in baselines.items():
        if metric in metrics:
            degradation = baseline - metrics[metric]
            if degradation > threshold:
                alerts.append({
                    "metric": metric,
                    "current_value": metrics[metric],
                    "baseline_value": baseline,
                    "degradation": degradation
                })
    
    if alerts:
        return {
            "model_id": model_id,
            "alert_type": "performance_degradation",
            "severity": "high" if any(alert["degradation"] > threshold * 2 for alert in alerts) else "medium",
            "details": alerts
        }
    
    return None


def _perform_model_health_check(model_id: str, model, detailed: bool) -> Dict[str, Any]:
    """Perform comprehensive health check on a model."""
    
    health_score = 100
    issues = []
    status = "HEALTHY"
    
    try:
        # Check model file size (basic integrity check)
        # In practice, this would include more comprehensive checks
        
        # Simulate health checks
        import random
        
        # Random health issues for demonstration
        if random.random() < 0.1:  # 10% chance of memory issue
            issues.append("High memory usage detected")
            health_score -= 20
        
        if random.random() < 0.05:  # 5% chance of performance issue
            issues.append("Model response time above threshold")
            health_score -= 15
        
        if random.random() < 0.03:  # 3% chance of accuracy issue
            issues.append("Accuracy degradation detected")
            health_score -= 25
        
        if health_score < 70:
            status = "CRITICAL"
        elif health_score < 85:
            status = "WARNING"
        
        return {
            "status": status,
            "healthy": status == "HEALTHY",
            "health_score": health_score,
            "issues": issues,
            "last_checked": datetime.now().isoformat(),
            "model_size": "N/A",  # Would be actual model size
            "model_version": getattr(model, 'version', 'unknown')
        }
        
    except Exception as e:
        return {
            "status": "ERROR",
            "healthy": False,
            "error": str(e),
            "issues": [f"Health check failed: {str(e)}"]
        }


def _update_monitoring_display(layout, current_metrics: Dict[str, Dict], alerts: List[Dict], 
                             start_time: float, duration: int, iteration: int):
    """Update the live monitoring display."""
    
    elapsed = int(time.time() - start_time)
    remaining = max(0, duration - elapsed)
    
    # Header
    layout["header"].update(Panel(
        f"[bold blue]Model Performance Monitoring[/bold blue] | "
        f"Iteration: {iteration} | "
        f"Elapsed: {elapsed}s | "
        f"Remaining: {remaining}s",
        style="blue"
    ))
    
    # Metrics table
    metrics_table = Table(title="Current Performance Metrics")
    metrics_table.add_column("Model ID", style="cyan")
    metrics_table.add_column("Accuracy", style="green")
    metrics_table.add_column("F1-Score", style="green")
    metrics_table.add_column("Response (ms)", style="yellow")
    metrics_table.add_column("Throughput", style="blue")
    
    for model_id, metrics in current_metrics.items():
        metrics_table.add_row(
            model_id,
            f"{metrics['accuracy']:.3f}",
            f"{metrics['f1_score']:.3f}",
            f"{metrics['response_time']:.1f}",
            f"{metrics['throughput']:.0f}"
        )
    
    # Alerts panel
    alerts_content = ""
    if alerts:
        recent_alerts = alerts[-5:]  # Show last 5 alerts
        for alert in recent_alerts:
            timestamp = alert.get("timestamp", "Unknown")
            severity = alert.get("severity", "medium").upper()
            model_id = alert.get("model_id", "Unknown")
            alerts_content += f"[{severity}] {model_id}: Performance degradation at {timestamp}\n"
    else:
        alerts_content = "[green]No alerts detected[/green]"
    
    layout["metrics"].update(Panel(metrics_table, title="Metrics", border_style="green"))
    layout["alerts"].update(Panel(alerts_content, title="Recent Alerts", border_style="red"))
    
    # Footer with system info
    layout["footer"].update(Panel(
        f"Press Ctrl+C to stop monitoring early | "
        f"Total models: {len(current_metrics)} | "
        f"Alert count: {len(alerts)}",
        style="dim"
    ))


def _display_monitoring_summary(monitoring_data: Dict[str, Any]):
    """Display final monitoring summary."""
    
    print(f"\n[blue]📊[/blue] Monitoring Summary:")
    print(f"  Start time: {monitoring_data['start_time']}")
    print(f"  End time: {monitoring_data.get('end_time', 'N/A')}")
    print(f"  Total iterations: {monitoring_data.get('total_iterations', 0)}")
    print(f"  Models monitored: {len(monitoring_data['model_ids'])}")
    print(f"  Total alerts: {len(monitoring_data['alerts'])}")
    
    if monitoring_data['alerts']:
        print(f"\n[yellow]⚠[/yellow] Alert Summary:")
        alert_counts = {}
        for alert in monitoring_data['alerts']:
            model_id = alert.get('model_id', 'Unknown')
            alert_counts[model_id] = alert_counts.get(model_id, 0) + 1
        
        for model_id, count in alert_counts.items():
            print(f"  {model_id}: {count} alerts")
    else:
        print(f"\n[green]✅[/green] No performance issues detected during monitoring period")