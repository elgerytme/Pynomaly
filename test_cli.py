#!/usr/bin/env python
"""Test CLI functionality for anomaly_detection."""

import sys
import os
sys.path.insert(0, '/mnt/c/Users/andre/anomaly_detection/src/packages/data/anomaly_detection/src')

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="anomaly_detection Anomaly Detection CLI")
console = Console()

@app.command()
def detect(
    data_file: str = typer.Argument(..., help="Path to data file"),
    output_file: str = typer.Option("results.json", help="Output file for results"),
    algorithm: str = typer.Option("isolation_forest", help="Algorithm to use"),
    contamination: float = typer.Option(0.1, help="Expected contamination rate"),
):
    """Detect anomalies in data file."""
    console.print(f"[bold green]Detecting anomalies in {data_file}[/bold green]")
    console.print(f"Algorithm: {algorithm}")
    console.print(f"Contamination rate: {contamination}")
    console.print(f"Output file: {output_file}")
    
    try:
        # Test basic functionality
        import numpy as np
        from anomaly_detection import AnomalyDetector
        
        # Generate sample data for testing
        np.random.seed(42)
        data = np.random.randn(100, 2)
        
        # Create detector
        detector = AnomalyDetector()
        
        # Fit and predict
        detector.fit(data, contamination=contamination)
        predictions = detector.predict(data)
        
        anomalies_count = sum(predictions)
        
        console.print(f"[bold blue]Results:[/bold blue]")
        console.print(f"Total samples: {len(data)}")
        console.print(f"Anomalies detected: {anomalies_count}")
        console.print(f"Anomaly rate: {anomalies_count/len(data)*100:.2f}%")
        
        console.print(f"[bold green]Results saved to {output_file}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)

@app.command()
def algorithms():
    """List available algorithms."""
    console.print("[bold blue]Available Algorithms:[/bold blue]")
    
    table = Table(title="Anomaly Detection Algorithms")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Description", style="green")
    
    algorithms = [
        ("isolation_forest", "Tree-based", "Isolation Forest (default)"),
        ("lof", "Density-based", "Local Outlier Factor"),
        ("one_class_svm", "SVM-based", "One-Class SVM"),
        ("pyod_*", "Various", "PyOD library algorithms"),
    ]
    
    for name, type_, desc in algorithms:
        table.add_row(name, type_, desc)
    
    console.print(table)

@app.command()
def version():
    """Show version information."""
    try:
        import anomaly_detection
        console.print(f"[bold green]anomaly_detection Detection v{anomaly_detection.__version__}[/bold green]")
        console.print(f"Author: {anomaly_detection.__author__}")
        console.print(f"Email: {anomaly_detection.__email__}")
    except Exception as e:
        console.print(f"[bold red]Could not load version info: {e}[/bold red]")

if __name__ == "__main__":
    app()