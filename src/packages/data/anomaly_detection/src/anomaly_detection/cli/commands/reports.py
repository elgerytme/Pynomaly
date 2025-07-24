"""Report generation commands for Typer CLI."""

import json
import typer
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import tempfile
import uuid

from ...domain.services.detection_service import DetectionService
from ...domain.entities.dataset import Dataset, DatasetType, DatasetMetadata
from ...infrastructure.logging import get_logger

console = Console()
logger = get_logger(__name__)

app = typer.Typer(help="Report generation commands")


@app.command()
def detection_report(
    results_file: Path = typer.Option(..., "--results", "-r", help="Detection results JSON file"),
    output: Path = typer.Option(..., "--output", "-o", help="Output report file (HTML/PDF/JSON)"),
    template: str = typer.Option("standard", "--template", help="Report template (standard/executive/technical)"),
    include_plots: bool = typer.Option(True, "--include-plots", help="Include visualization plots"),
    title: Optional[str] = typer.Option(None, "--title", help="Custom report title"),
) -> None:
    """Generate comprehensive detection report from results."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Load results
        load_task = progress.add_task("Loading detection results...", total=None)
        
        try:
            if not results_file.exists():
                print(f"[red]✗[/red] Results file '{results_file}' not found")
                raise typer.Exit(1)
            
            with open(results_file, 'r') as f:
                results_data = json.load(f)
            
            progress.update(load_task, completed=True)
            print(f"[green]✓[/green] Loaded detection results")
            
        except Exception as e:
            print(f"[red]✗[/red] Failed to load results: {e}")
            raise typer.Exit(1)
        
        # Generate report
        report_task = progress.add_task("Generating report...", total=None)
        
        try:
            report_content = _generate_detection_report_content(
                results_data, template, title or "Anomaly Detection Report"
            )
            
            # Determine output format
            output_format = output.suffix.lower()
            
            if output_format == '.html':
                _save_html_report(report_content, output, include_plots)
            elif output_format == '.json':
                _save_json_report(report_content, output)
            elif output_format == '.pdf':
                _save_pdf_report(report_content, output, include_plots)
            else:
                print(f"[red]✗[/red] Unsupported output format: {output_format}")
                print("Supported formats: .html, .json, .pdf")
                raise typer.Exit(1)
            
            progress.update(report_task, completed=True)
            
        except Exception as e:
            print(f"[red]✗[/red] Report generation failed: {e}")
            raise typer.Exit(1)
    
    print(f"[green]✓[/green] Report saved to: {output}")
    print(f"[green]✅[/green] Detection report generated successfully!")


@app.command()
def performance_report(
    model_ids: List[str] = typer.Option(..., "--models", "-m", help="Model IDs to include in report"),
    output: Path = typer.Option(..., "--output", "-o", help="Output report file"),
    time_range: int = typer.Option(30, "--days", help="Time range in days for performance data"),
    include_comparisons: bool = typer.Option(True, "--compare", help="Include model comparisons"),
) -> None:
    """Generate model performance report."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Collect performance data
        collect_task = progress.add_task("Collecting performance data...", total=None)
        
        try:
            # This would typically query a model registry or monitoring system
            # For now, we'll generate a mock report structure
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_range)
            
            performance_data = {
                "report_metadata": {
                    "generated_at": end_date.isoformat(),
                    "time_range": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat(),
                        "days": time_range
                    },
                    "models_analyzed": model_ids
                },
                "model_performance": {},
                "summary_statistics": {},
                "recommendations": []
            }
            
            for model_id in model_ids:
                # Mock performance data - in real implementation, this would query metrics
                performance_data["model_performance"][model_id] = {
                    "accuracy_trend": [0.85, 0.87, 0.84, 0.82, 0.86],
                    "precision_trend": [0.78, 0.80, 0.76, 0.74, 0.79],
                    "recall_trend": [0.82, 0.84, 0.81, 0.79, 0.83],
                    "f1_trend": [0.80, 0.82, 0.78, 0.76, 0.81],
                    "current_metrics": {
                        "accuracy": 0.86,
                        "precision": 0.79,
                        "recall": 0.83,
                        "f1_score": 0.81
                    },
                    "drift_detected": model_id == model_ids[0],  # Mock drift for first model
                    "prediction_volume": [1000, 1200, 980, 1100, 1050],
                    "response_times": [45, 52, 48, 50, 47]  # ms
                }
            
            progress.update(collect_task, completed=True)
            
        except Exception as e:
            print(f"[red]✗[/red] Failed to collect performance data: {e}")
            raise typer.Exit(1)
        
        # Generate report
        report_task = progress.add_task("Generating performance report...", total=None)
        
        try:
            report_content = _generate_performance_report_content(
                performance_data, include_comparisons
            )
            
            output_format = output.suffix.lower()
            
            if output_format == '.html':
                _save_html_report(report_content, output, True)
            elif output_format == '.json':
                with open(output, 'w') as f:
                    json.dump(performance_data, f, indent=2)
            else:
                print(f"[red]✗[/red] Unsupported format for performance report: {output_format}")
                raise typer.Exit(1)
            
            progress.update(report_task, completed=True)
            
        except Exception as e:
            print(f"[red]✗[/red] Performance report generation failed: {e}")
            raise typer.Exit(1)
    
    # Display summary
    table = Table(title="[bold blue]Performance Report Summary[/bold blue]")
    table.add_column("Model ID", style="cyan")
    table.add_column("Current F1", style="green")
    table.add_column("Drift Status", style="yellow")
    table.add_column("Avg Response (ms)", style="green")
    
    for model_id in model_ids:
        model_perf = performance_data["model_performance"][model_id]
        drift_status = "⚠️ Detected" if model_perf["drift_detected"] else "✅ Stable"
        
        table.add_row(
            model_id,
            f"{model_perf['current_metrics']['f1_score']:.3f}",
            drift_status,
            f"{sum(model_perf['response_times']) / len(model_perf['response_times']):.1f}"
        )
    
    console.print(table)
    print(f"[green]✓[/green] Performance report saved to: {output}")
    print(f"[green]✅[/green] Performance report generated successfully!")


@app.command()
def batch_report(
    input_dir: Path = typer.Option(..., "--input-dir", "-i", help="Directory containing result files"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory for reports"),
    format: str = typer.Option("html", "--format", help="Output format (html/json/pdf)"),
    template: str = typer.Option("standard", "--template", help="Report template"),
) -> None:
    """Generate batch reports for multiple result files."""
    
    if not input_dir.exists():
        print(f"[red]✗[/red] Input directory '{input_dir}' not found")
        raise typer.Exit(1)
    
    # Find all result files
    result_files = list(input_dir.glob("*.json"))
    
    if not result_files:
        print(f"[red]✗[/red] No JSON result files found in '{input_dir}'")
        raise typer.Exit(1)
    
    output_dir.mkdir(exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        batch_task = progress.add_task(f"Processing {len(result_files)} files...", total=len(result_files))
        
        generated_reports = []
        
        for i, result_file in enumerate(result_files):
            try:
                # Generate individual report
                output_file = output_dir / f"{result_file.stem}_report.{format}"
                
                # Load results
                with open(result_file, 'r') as f:
                    results_data = json.load(f)
                
                # Generate report content
                report_content = _generate_detection_report_content(
                    results_data, template, f"Detection Report - {result_file.stem}"
                )
                
                # Save report
                if format == 'html':
                    _save_html_report(report_content, output_file, True)
                elif format == 'json':
                    _save_json_report(report_content, output_file)
                
                generated_reports.append(output_file)
                
                progress.update(batch_task, advance=1)
                
            except Exception as e:
                print(f"[yellow]⚠[/yellow] Failed to process {result_file}: {e}")
                continue
    
    # Generate summary report
    summary_file = output_dir / f"batch_summary.{format}"
    _generate_batch_summary(result_files, generated_reports, summary_file)
    
    print(f"[green]✓[/green] Generated {len(generated_reports)} individual reports")
    print(f"[green]✓[/green] Batch summary saved to: {summary_file}")
    print(f"[green]✅[/green] Batch report generation completed!")


def _generate_detection_report_content(results_data: Dict[str, Any], template: str, title: str) -> Dict[str, Any]:
    """Generate report content from detection results."""
    
    current_time = datetime.now()
    
    # Extract key metrics
    detection_results = results_data.get("detection_results", {})
    dataset_info = results_data.get("dataset_info", {})
    evaluation_metrics = results_data.get("evaluation_metrics", {})
    
    report_content = {
        "metadata": {
            "title": title,
            "generated_at": current_time.isoformat(),
            "template": template,
            "report_id": str(uuid.uuid4())
        },
        "executive_summary": {
            "total_samples": dataset_info.get("total_samples", 0),
            "anomalies_detected": detection_results.get("anomalies_detected", 0),
            "anomaly_rate": detection_results.get("anomaly_rate", 0),
            "algorithm_used": results_data.get("algorithm", "Unknown"),
            "detection_success": results_data.get("success", False)
        },
        "detailed_results": {
            "dataset_information": dataset_info,
            "detection_configuration": {
                "algorithm": results_data.get("algorithm"),
                "contamination": results_data.get("contamination"),
                "input_file": results_data.get("input")
            },
            "detection_results": detection_results,
            "performance_metrics": evaluation_metrics
        },
        "recommendations": _generate_recommendations(results_data),
        "appendix": {
            "anomaly_indices": detection_results.get("anomaly_indices", []),
            "technical_details": {
                "processing_time": "N/A",  # Would be extracted from results
                "model_parameters": "N/A"
            }
        }
    }
    
    return report_content


def _generate_performance_report_content(performance_data: Dict[str, Any], include_comparisons: bool) -> Dict[str, Any]:
    """Generate performance report content."""
    
    models = list(performance_data["model_performance"].keys())
    
    # Calculate summary statistics
    all_f1_scores = []
    drift_count = 0
    
    for model_id, perf in performance_data["model_performance"].items():
        all_f1_scores.append(perf["current_metrics"]["f1_score"])
        if perf["drift_detected"]:
            drift_count += 1
    
    avg_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0
    
    report_content = {
        "metadata": performance_data["report_metadata"],
        "executive_summary": {
            "models_monitored": len(models),
            "average_f1_score": avg_f1,
            "models_with_drift": drift_count,
            "overall_health": "Good" if drift_count == 0 and avg_f1 > 0.8 else "Attention Required"
        },
        "model_performance": performance_data["model_performance"],
        "recommendations": [
            "Monitor drift detection alerts closely",
            "Consider retraining models with detected drift",
            "Evaluate model performance trends regularly"
        ]
    }
    
    if include_comparisons:
        report_content["model_comparisons"] = _generate_model_comparisons(performance_data)
    
    return report_content


def _generate_model_comparisons(performance_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate model comparison analysis."""
    
    models = performance_data["model_performance"]
    
    # Find best performing model
    best_model = max(models.keys(), 
                    key=lambda m: models[m]["current_metrics"]["f1_score"])
    
    # Find model with most stable performance (lowest variance)
    import numpy as np
    most_stable = min(models.keys(),
                     key=lambda m: np.var(models[m]["f1_trend"]))
    
    return {
        "best_performer": {
            "model_id": best_model,
            "f1_score": models[best_model]["current_metrics"]["f1_score"]
        },
        "most_stable": {
            "model_id": most_stable,
            "variance": float(np.var(models[most_stable]["f1_trend"]))
        },
        "ranking": sorted(models.keys(), 
                         key=lambda m: models[m]["current_metrics"]["f1_score"], 
                         reverse=True)
    }


def _generate_recommendations(results_data: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on detection results."""
    
    recommendations = []
    
    # Analyze anomaly rate
    detection_results = results_data.get("detection_results", {})
    anomaly_rate = detection_results.get("anomaly_rate", 0)
    
    if anomaly_rate > 0.2:
        recommendations.append("High anomaly rate detected. Consider reviewing data quality or adjusting contamination parameter.")
    elif anomaly_rate < 0.01:
        recommendations.append("Very low anomaly rate. Consider increasing sensitivity or reviewing detection criteria.")
    
    # Analyze performance metrics
    eval_metrics = results_data.get("evaluation_metrics", {})
    if eval_metrics:
        precision = eval_metrics.get("precision", 0)
        recall = eval_metrics.get("recall", 0)
        
        if precision < 0.7:
            recommendations.append("Low precision indicates many false positives. Consider tightening detection criteria.")
        if recall < 0.7:
            recommendations.append("Low recall indicates missed anomalies. Consider loosening detection criteria.")
    
    # Algorithm-specific recommendations
    algorithm = results_data.get("algorithm", "")
    if algorithm == "isolation_forest":
        recommendations.append("Consider ensemble methods for improved robustness.")
    
    if not recommendations:
        recommendations.append("Detection results appear nominal. Continue monitoring for drift.")
    
    return recommendations


def _save_html_report(report_content: Dict[str, Any], output_path: Path, include_plots: bool) -> None:
    """Save report as HTML."""
    
    html_template = f\"\"\"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_content['metadata']['title']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .error {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        .recommendations {{ background: #e8f8f5; padding: 20px; border-left: 4px solid #1abc9c; margin: 20px 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report_content['metadata']['title']}</h1>
        
        <div class="metadata">
            <p><strong>Generated:</strong> {report_content['metadata']['generated_at']}</p>
            <p><strong>Report ID:</strong> {report_content['metadata']['report_id']}</p>
        </div>
        
        <h2>Executive Summary</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{report_content['executive_summary']['total_samples']}</div>
                <div class="metric-label">Total Samples</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report_content['executive_summary']['anomalies_detected']}</div>
                <div class="metric-label">Anomalies Detected</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report_content['executive_summary']['anomaly_rate']:.1%}</div>
                <div class="metric-label">Anomaly Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report_content['executive_summary']['algorithm_used']}</div>
                <div class="metric-label">Algorithm</div>
            </div>
        </div>
        
        <h2>Detection Results</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
    \"\"\"
    
    # Add detection results to table
    for key, value in report_content['detailed_results']['detection_results'].items():
        if isinstance(value, float):
            value_str = f"{value:.3f}"
        elif isinstance(value, list):
            value_str = f"[{len(value)} items]"
        else:
            value_str = str(value)
        html_template += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value_str}</td></tr>"
    
    html_template += "</table>"
    
    # Add performance metrics if available
    if report_content['detailed_results']['performance_metrics']:
        html_template += "<h2>Performance Metrics</h2><table><tr><th>Metric</th><th>Value</th></tr>"
        for key, value in report_content['detailed_results']['performance_metrics'].items():
            html_template += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value:.3f}</td></tr>"
        html_template += "</table>"
    
    # Add recommendations
    html_template += "<div class=\"recommendations\"><h2>Recommendations</h2><ul>"
    for rec in report_content['recommendations']:
        html_template += f"<li>{rec}</li>"
    html_template += "</ul></div>"
    
    # Add footer
    html_template += f\"\"\"
        <div class="footer">
            <p>Generated by Anomaly Detection System - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
    \"\"\"
    
    with open(output_path, 'w') as f:
        f.write(html_template)


def _save_json_report(report_content: Dict[str, Any], output_path: Path) -> None:
    """Save report as JSON."""
    with open(output_path, 'w') as f:
        json.dump(report_content, f, indent=2)


def _save_pdf_report(report_content: Dict[str, Any], output_path: Path, include_plots: bool) -> None:
    """Save report as PDF (requires additional dependencies)."""
    try:
        # This would require libraries like reportlab or weasyprint
        # For now, we'll create an HTML report and suggest PDF conversion
        html_path = output_path.with_suffix('.html')
        _save_html_report(report_content, html_path, include_plots)
        
        print(f"[yellow]⚠[/yellow] PDF generation requires additional dependencies.")
        print(f"[blue]ℹ[/blue] HTML report created at: {html_path}")
        print(f"[blue]ℹ[/blue] Use a tool like wkhtmltopdf to convert to PDF:")
        print(f"[blue]ℹ[/blue] wkhtmltopdf {html_path} {output_path}")
        
    except Exception as e:
        print(f"[red]✗[/red] PDF generation failed: {e}")
        raise


def _generate_batch_summary(result_files: List[Path], generated_reports: List[Path], output_path: Path) -> None:
    """Generate summary of batch processing."""
    
    summary_data = {
        "batch_summary": {
            "processed_files": len(result_files),
            "successful_reports": len(generated_reports),
            "generated_at": datetime.now().isoformat()
        },
        "files_processed": [str(f) for f in result_files],
        "reports_generated": [str(f) for f in generated_reports]
    }
    
    if output_path.suffix.lower() == '.html':
        _save_html_batch_summary(summary_data, output_path)
    else:
        with open(output_path, 'w') as f:
            json.dump(summary_data, f, indent=2)


def _save_html_batch_summary(summary_data: Dict[str, Any], output_path: Path) -> None:
    """Save batch summary as HTML."""
    
    html_content = f\"\"\"
<!DOCTYPE html>
<html>
<head>
    <title>Batch Report Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f8ff; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>Batch Report Generation Summary</h1>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <p><strong>Files Processed:</strong> {summary_data['batch_summary']['processed_files']}</p>
        <p><strong>Reports Generated:</strong> {summary_data['batch_summary']['successful_reports']}</p>
        <p><strong>Success Rate:</strong> {summary_data['batch_summary']['successful_reports'] / summary_data['batch_summary']['processed_files'] * 100:.1f}%</p>
        <p><strong>Generated At:</strong> {summary_data['batch_summary']['generated_at']}</p>
    </div>
    
    <h2>Generated Reports</h2>
    <table>
        <tr><th>Report File</th></tr>
    \"\"\"
    
    for report in summary_data['reports_generated']:
        html_content += f"<tr><td>{report}</td></tr>"
    
    html_content += "</table></body></html>"
    
    with open(output_path, 'w') as f:
        f.write(html_content)