"""Data generation and management commands for Typer CLI."""

import typer
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

app = typer.Typer(help="Data generation and management commands")


@app.command()
def generate(
    output: Path = typer.Option(..., "--output", "-o", help="Output dataset file path"),
    samples: int = typer.Option(1000, "--samples", "-n", help="Number of samples to generate"),
    features: int = typer.Option(5, "--features", "-f", help="Number of features"),
    contamination: float = typer.Option(0.1, "--contamination", "-c", help="Contamination rate (anomaly ratio)"),
    anomaly_type: str = typer.Option("point", "--anomaly-type", help="Type of anomalies to generate"),
    random_state: int = typer.Option(42, "--random-state", help="Random seed for reproducibility"),
) -> None:
    """Generate synthetic anomaly detection dataset."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        generate_task = progress.add_task("Generating synthetic dataset...", total=None)
        
        try:
            from sklearn.datasets import make_blobs
            from sklearn.preprocessing import StandardScaler
            
            np.random.seed(random_state)
            
            # Generate base dataset
            X, _ = make_blobs(n_samples=samples, centers=1, n_features=features, 
                             random_state=random_state, cluster_std=1.0)
            
            # Standardize data
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            # Generate anomalies
            n_anomalies = int(samples * contamination)
            n_normal = samples - n_anomalies
            
            # Create labels (1 for normal, -1 for anomaly)
            labels = np.ones(samples)
            
            if anomaly_type == 'point':
                # Point anomalies: random outliers
                anomaly_indices = np.random.choice(samples, n_anomalies, replace=False)
                X[anomaly_indices] += np.random.normal(0, 3, (n_anomalies, features))
                labels[anomaly_indices] = -1
                
            elif anomaly_type == 'contextual':
                # Contextual anomalies: anomalies in specific feature combinations
                anomaly_indices = np.random.choice(samples, n_anomalies, replace=False)
                for idx in anomaly_indices:
                    anomalous_features = np.random.choice(features, np.random.randint(1, min(3, features+1)), replace=False)
                    X[idx, anomalous_features] += np.random.normal(0, 2, len(anomalous_features))
                labels[anomaly_indices] = -1
                
            elif anomaly_type == 'collective':
                # Collective anomalies: groups of anomalous points
                n_groups = max(1, n_anomalies // 10)
                group_size = n_anomalies // n_groups
                
                start_idx = 0
                for _ in range(n_groups):
                    end_idx = min(start_idx + group_size, samples)
                    group_indices = range(start_idx, end_idx)
                    
                    group_shift = np.random.normal(0, 2, features)
                    X[group_indices] += group_shift
                    labels[group_indices] = -1
                    
                    start_idx = end_idx
            
            # Create dataset with feature names
            feature_names = [f"feature_{i}" for i in range(features)]
            df = pd.DataFrame(X, columns=feature_names)
            df['label'] = labels.astype(int)
            
            progress.update(generate_task, completed=True)
            
        except Exception as e:
            print(f"[red]✗[/red] Failed to generate dataset: {e}")
            raise typer.Exit(1)
        
        # Save dataset
        save_task = progress.add_task("Saving dataset...", total=None)
        
        try:
            if output.suffix.lower() == '.csv':
                df.to_csv(output, index=False)
            elif output.suffix.lower() == '.json':
                df.to_json(output, orient='records', indent=2)
            elif output.suffix.lower() == '.parquet':
                df.to_parquet(output, index=False)
            else:
                # Default to CSV
                output = output.with_suffix('.csv')
                df.to_csv(output, index=False)
            
            progress.update(save_task, completed=True)
            
        except Exception as e:
            print(f"[red]✗[/red] Failed to save dataset: {e}")
            raise typer.Exit(1)
    
    print(f"[green]✅[/green] Generated synthetic dataset:")
    print(f"   File: [cyan]{output}[/cyan]")
    print(f"   Samples: [green]{samples}[/green] ({n_normal} normal, {n_anomalies} anomalies)")
    print(f"   Features: [green]{features}[/green]")
    print(f"   Contamination: [green]{contamination:.1%}[/green]")
    print(f"   Anomaly type: [green]{anomaly_type}[/green]")
    print(f"   Random seed: [green]{random_state}[/green]")


@app.command()
def validate(
    input_file: Path = typer.Argument(..., help="Input dataset file to validate"),
    label_column: Optional[str] = typer.Option(None, "--label-column", help="Name of label column if present"),
) -> None:
    """Validate dataset format and contents."""
    
    try:
        if not input_file.exists():
            print(f"[red]✗[/red] Input file '{input_file}' not found")
            raise typer.Exit(1)
        
        # Load data
        if input_file.suffix.lower() == '.csv':
            df = pd.read_csv(input_file)
        elif input_file.suffix.lower() == '.json':
            df = pd.read_json(input_file)
        elif input_file.suffix.lower() == '.parquet':
            df = pd.read_parquet(input_file)
        else:
            print(f"[red]✗[/red] Unsupported file format '{input_file.suffix}'")
            raise typer.Exit(1)
        
        print(f"[green]✓[/green] Successfully loaded dataset")
        print(f"   Rows: [cyan]{len(df)}[/cyan]")
        print(f"   Columns: [cyan]{len(df.columns)}[/cyan]")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"[yellow]⚠[/yellow] Found {missing_values} missing values")
        else:
            print(f"[green]✓[/green] No missing values found")
        
        # Check data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if label_column and label_column in non_numeric_cols:
            non_numeric_cols.remove(label_column)
        
        print(f"[green]✓[/green] Numeric columns: {len(numeric_cols)}")
        if non_numeric_cols:
            print(f"[yellow]⚠[/yellow] Non-numeric columns found: {non_numeric_cols}")
        
        # Check labels if present
        if label_column and label_column in df.columns:
            labels = df[label_column]
            unique_labels = labels.unique()
            print(f"[green]✓[/green] Label column '{label_column}' found")
            print(f"   Unique labels: {unique_labels}")
            
            if set(unique_labels).issubset({-1, 1}):
                anomaly_count = (labels == -1).sum()
                normal_count = (labels == 1).sum()
                anomaly_rate = anomaly_count / len(labels)
                print(f"   Normal samples: [green]{normal_count}[/green]")
                print(f"   Anomaly samples: [red]{anomaly_count}[/red]")
                print(f"   Anomaly rate: [blue]{anomaly_rate:.1%}[/blue]")
        
        print(f"[green]✅[/green] Dataset validation completed")
        
    except Exception as e:
        print(f"[red]✗[/red] Dataset validation failed: {e}")
        raise typer.Exit(1)


@app.command()
def convert(
    input_file: Path = typer.Argument(..., help="Input dataset file"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file path"),
    format: str = typer.Option("csv", "--format", "-f", help="Output format (csv, json, parquet)"),
) -> None:
    """Convert dataset between different formats."""
    
    try:
        # Load input file
        if input_file.suffix.lower() == '.csv':
            df = pd.read_csv(input_file)
        elif input_file.suffix.lower() == '.json':
            df = pd.read_json(input_file)
        elif input_file.suffix.lower() == '.parquet':
            df = pd.read_parquet(input_file)
        else:
            print(f"[red]✗[/red] Unsupported input format '{input_file.suffix}'")
            raise typer.Exit(1)
        
        # Save in new format
        if format.lower() == 'csv':
            df.to_csv(output, index=False)
        elif format.lower() == 'json':
            df.to_json(output, orient='records', indent=2)
        elif format.lower() == 'parquet':
            df.to_parquet(output, index=False)
        else:
            print(f"[red]✗[/red] Unsupported output format '{format}'")
            raise typer.Exit(1)
        
        print(f"[green]✅[/green] Converted dataset from {input_file.suffix} to {format}")
        print(f"   Input: [cyan]{input_file}[/cyan] ({len(df)} rows)")
        print(f"   Output: [cyan]{output}[/cyan]")
        
    except Exception as e:
        print(f"[red]✗[/red] Conversion failed: {e}")
        raise typer.Exit(1)