#!/usr/bin/env python3
"""
Pynomaly CLI - Command Line Interface for Anomaly Detection

This module provides a command-line interface for the Pynomaly
anomaly detection platform.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

try:
    from pynomaly_detection import AnomalyDetector, __version__
except ImportError:
    # Fallback for development - add the package to Python path
    import sys
    from pathlib import Path
    package_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(package_root))
    from pynomaly_detection import AnomalyDetector, __version__

app = typer.Typer(
    name="pynomaly",
    help="Pynomaly - Advanced Anomaly Detection Platform",
    no_args_is_help=True,
)

console = Console()


@app.command()
def version():
    """Show version information."""
    console.print(f"Pynomaly version: {__version__}")


@app.command()
def algorithms():
    """List available algorithms."""
    table = Table(title="Available Algorithms")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Status", style="yellow")

    # Core algorithms
    table.add_row("isolation_forest", "Isolation Forest (default)", "✓ Working")
    table.add_row("lof", "Local Outlier Factor", "⚠ Requires sklearn")
    table.add_row("ocsvm", "One-Class SVM", "⚠ Requires sklearn")
    table.add_row("autoencoder", "Neural Network Autoencoder", "✗ Not implemented")
    table.add_row("ensemble", "Ensemble Methods", "✗ Not implemented")

    console.print(table)


@app.command()
def detect(
    file: str = typer.Argument(..., help="Path to data file (CSV, JSON, or NPY)"),
    algorithm: str = typer.Option("isolation_forest", help="Algorithm to use"),
    contamination: float = typer.Option(0.1, help="Expected contamination rate"),
    output: str | None = typer.Option(None, help="Output file path"),
    format: str = typer.Option("json", help="Output format (json, csv, or text)"),
):
    """Detect anomalies in a data file."""
    try:
        # Load data
        console.print(f"Loading data from: {file}")
        data = load_data(file)

        # Create detector
        detector = AnomalyDetector()

        # Detect anomalies
        console.print(f"Running anomaly detection with {algorithm}...")
        predictions = detector.detect(data, contamination=contamination)

        # Process results
        anomaly_count = np.sum(predictions)
        total_count = len(predictions)
        anomaly_rate = anomaly_count / total_count

        # Show results
        console.print("\n📊 Results:")
        console.print(f"  Total samples: {total_count}")
        console.print(f"  Anomalies detected: {anomaly_count}")
        console.print(f"  Anomaly rate: {anomaly_rate:.2%}")

        # Save results if requested
        if output:
            save_results(predictions, output, format)
            console.print(f"  Results saved to: {output}")

    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="red")
        raise typer.Exit(1) from e


@app.command()
def train(
    file: str = typer.Argument(..., help="Path to training data file"),
    algorithm: str = typer.Option("isolation_forest", help="Algorithm to use"),
    contamination: float = typer.Option(0.1, help="Expected contamination rate"),
    model_path: str | None = typer.Option(None, help="Path to save trained model"),
):
    """Train a detector on data."""
    try:
        # Load data
        console.print(f"Loading training data from: {file}")
        data = load_data(file)

        # Create and train detector
        detector = AnomalyDetector()
        detector.fit(data, contamination=contamination)

        console.print(f"✅ Model trained successfully with {algorithm}")

        # Save model if requested
        if model_path:
            # Note: Model saving would need to be implemented
            console.print("Model saving not yet implemented")

    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="red")
        raise typer.Exit(1) from e


@app.command()
def validate(
    file: str = typer.Argument(..., help="Path to data file"),
):
    """Validate data format and structure."""
    try:
        data = load_data(file)

        console.print("✅ Data validation results:")
        console.print(f"  Shape: {data.shape}")
        console.print(f"  Data type: {data.dtype}")
        console.print(f"  Has NaN values: {np.isnan(data).any()}")
        console.print(f"  Has infinite values: {np.isinf(data).any()}")

    except Exception as e:
        console.print(f"❌ Validation failed: {str(e)}", style="red")
        raise typer.Exit(1) from e


def load_data(file_path: str) -> np.ndarray:
    """Load data from various file formats."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
        return df.select_dtypes(include=[np.number]).values
    elif path.suffix.lower() == '.json':
        with open(path) as f:
            data = json.load(f)
        return np.array(data)
    elif path.suffix.lower() == '.npy':
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_results(predictions: np.ndarray, output_path: str, format: str):
    """Save detection results to file."""
    path = Path(output_path)

    if format == "json":
        results = {
            "predictions": predictions.tolist(),
            "anomaly_count": int(np.sum(predictions)),
            "total_count": int(len(predictions)),
            "anomaly_rate": float(np.sum(predictions) / len(predictions))
        }
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == "csv":
        pd.DataFrame({"prediction": predictions}).to_csv(path, index=False)
    elif format == "text":
        with open(path, 'w') as f:
            f.write("Anomaly Detection Results\n")
            f.write("=" * 25 + "\n")
            f.write(f"Total samples: {len(predictions)}\n")
            f.write(f"Anomalies detected: {np.sum(predictions)}\n")
            f.write(f"Anomaly rate: {np.sum(predictions) / len(predictions):.2%}\n")
            f.write("\nPredictions:\n")
            for i, pred in enumerate(predictions):
                f.write(f"Sample {i}: {'ANOMALY' if pred else 'NORMAL'}\n")
    else:
        raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    app()
