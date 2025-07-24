"""Streaming detection commands for Typer CLI."""

import typer
import asyncio
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import time
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel

from ...domain.services.streaming_service import StreamingService
from ...domain.services.detection_service import DetectionService
from ...infrastructure.logging import get_logger

console = Console()
logger = get_logger(__name__)

app = typer.Typer(help="Streaming anomaly detection commands")


@app.command()
def monitor(
    input_source: str = typer.Option("random", "--input", "-i", help="Input source: 'random', 'file:<path>', or 'kafka:<topic>'"),
    algorithm: str = typer.Option("isolation_forest", "--algorithm", "-a", help="Detection algorithm"),
    window_size: int = typer.Option(1000, "--window-size", "-w", help="Sliding window size"),
    update_frequency: int = typer.Option(100, "--update-freq", "-u", help="Model update frequency"),
    contamination: float = typer.Option(0.1, "--contamination", "-c", help="Contamination rate"),
    duration: int = typer.Option(60, "--duration", "-d", help="Monitoring duration in seconds"),
    sample_rate: float = typer.Option(1.0, "--sample-rate", "-r", help="Samples per second"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results"),
) -> None:
    """Monitor real-time anomaly detection on streaming data."""
    
    print(f"[blue]🔄[/blue] Starting streaming anomaly detection monitor")
    print(f"[dim]Input:[/dim] {input_source}")
    print(f"[dim]Algorithm:[/dim] {algorithm}")
    print(f"[dim]Window Size:[/dim] {window_size}")
    print(f"[dim]Duration:[/dim] {duration}s")
    
    # Initialize services
    detection_service = DetectionService()
    streaming_service = StreamingService(
        detection_service=detection_service,
        window_size=window_size,
        update_frequency=update_frequency
    )
    
    # Algorithm mapping
    algorithm_map = {
        'isolation_forest': 'iforest',
        'one_class_svm': 'ocsvm',
        'lof': 'lof'
    }
    mapped_algorithm = algorithm_map.get(algorithm, algorithm)
    
    # Setup data generator
    data_generator = _create_data_generator(input_source, sample_rate)
    
    # Setup monitoring display
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=5)
    )
    
    # Monitoring state
    start_time = datetime.utcnow()
    results = []
    anomaly_count = 0
    total_samples = 0
    
    try:
        with Live(layout, refresh_per_second=2, console=console) as live:
            for elapsed in range(duration):
                current_time = datetime.utcnow()
                
                # Process samples for this second
                samples_this_second = int(sample_rate)
                if samples_this_second > 0:
                    batch_data = []
                    for _ in range(samples_this_second):
                        try:
                            sample = next(data_generator)
                            batch_data.append(sample)
                        except StopIteration:
                            break
                    
                    if batch_data:
                        batch_array = np.array(batch_data)
                        result = streaming_service.process_batch(batch_array, mapped_algorithm)
                        
                        # Update counters
                        batch_anomalies = int(np.sum(result.predictions == -1))
                        anomaly_count += batch_anomalies
                        total_samples += len(batch_data)
                        
                        # Store result
                        result_record = {
                            'timestamp': current_time.isoformat(),
                            'batch_size': len(batch_data),
                            'anomalies': batch_anomalies,
                            'anomaly_rate': batch_anomalies / len(batch_data) if len(batch_data) > 0 else 0,
                            'algorithm': algorithm
                        }
                        results.append(result_record)
                
                # Update display
                _update_monitoring_display(
                    layout, 
                    streaming_service, 
                    total_samples, 
                    anomaly_count, 
                    elapsed, 
                    duration,
                    algorithm
                )
                
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n[yellow]⚠[/yellow] Monitoring interrupted by user")
    
    # Final results
    end_time = datetime.utcnow()
    total_duration = (end_time - start_time).total_seconds()
    
    print(f"\n[green]✅[/green] Monitoring completed")
    
    # Results table
    results_table = Table(title="[bold blue]Streaming Detection Results[/bold blue]")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    
    results_table.add_row("Total Samples Processed", str(total_samples))
    results_table.add_row("Total Anomalies Detected", str(anomaly_count))
    results_table.add_row("Overall Anomaly Rate", f"{anomaly_count/total_samples:.1%}" if total_samples > 0 else "0%")
    results_table.add_row("Monitoring Duration", f"{total_duration:.1f}s")
    results_table.add_row("Average Sample Rate", f"{total_samples/total_duration:.1f}/s" if total_duration > 0 else "0/s")
    
    console.print(results_table)
    
    # Get final streaming stats
    stats = streaming_service.get_streaming_stats()
    stats_table = Table(title="[bold blue]Streaming Service Statistics[/bold blue]")
    stats_table.add_column("Property", style="cyan")
    stats_table.add_column("Value", style="green")
    
    for key, value in stats.items():
        stats_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(stats_table)
    
    # Save results if requested
    if output_file:
        output_data = {
            'monitoring_session': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': total_duration,
                'algorithm': algorithm,
                'window_size': window_size,
                'update_frequency': update_frequency
            },
            'summary': {
                'total_samples': total_samples,
                'total_anomalies': anomaly_count,
                'anomaly_rate': anomaly_count/total_samples if total_samples > 0 else 0
            },
            'streaming_stats': stats,
            'batch_results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"[green]✅[/green] Results saved to: {output_file}")


@app.command()
def stats(
    window_size: int = typer.Option(1000, "--window-size", "-w", help="Window size for streaming service"),
    update_frequency: int = typer.Option(100, "--update-freq", "-u", help="Update frequency"),
) -> None:
    """Show streaming service statistics."""
    
    # Initialize streaming service
    streaming_service = StreamingService(
        window_size=window_size,
        update_frequency=update_frequency
    )
    
    stats = streaming_service.get_streaming_stats()
    
    # Display stats
    stats_table = Table(title="[bold blue]Streaming Service Statistics[/bold blue]")
    stats_table.add_column("Property", style="cyan")
    stats_table.add_column("Value", style="green")
    
    for key, value in stats.items():
        stats_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(stats_table)


@app.command()
def drift(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input data file"),
    window_size: int = typer.Option(200, "--window-size", "-w", help="Window size for drift detection"),
    algorithm: str = typer.Option("isolation_forest", "--algorithm", "-a", help="Detection algorithm"),
) -> None:
    """Detect concept drift in streaming data."""
    
    print(f"[blue]🔍[/blue] Analyzing concept drift in data stream")
    
    try:
        # Load data
        if not input_file.exists():
            print(f"[red]✗[/red] Input file '{input_file}' not found")
            raise typer.Exit(1)
        
        if input_file.suffix.lower() == '.csv':
            df = pd.read_csv(input_file)
        elif input_file.suffix.lower() == '.json':
            df = pd.read_json(input_file)
        else:
            print(f"[red]✗[/red] Unsupported file format '{input_file.suffix}'")
            raise typer.Exit(1)
        
        data_array = df.values.astype(np.float64)
        print(f"[green]✓[/green] Loaded {len(data_array)} samples with {data_array.shape[1]} features")
        
        # Initialize streaming service
        streaming_service = StreamingService(window_size=len(data_array))
        
        # Process data sequentially
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        mapped_algorithm = algorithm_map.get(algorithm, algorithm)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task("Processing data stream...", total=len(data_array))
            
            for i, sample in enumerate(data_array):
                streaming_service.process_sample(sample, mapped_algorithm)
                progress.update(task, advance=1)
                
                # Check for drift periodically
                if i > 2 * window_size and i % (window_size // 2) == 0:
                    drift_result = streaming_service.detect_concept_drift(window_size)
                    if drift_result['drift_detected']:
                        print(f"[yellow]⚠[/yellow] Concept drift detected at sample {i}")
        
        # Final drift analysis
        drift_result = streaming_service.detect_concept_drift(window_size)
        
        # Display results
        drift_table = Table(title="[bold blue]Concept Drift Analysis[/bold blue]")
        drift_table.add_column("Property", style="cyan")
        drift_table.add_column("Value", style="green")
        
        for key, value in drift_result.items():
            if key == 'drift_detected':
                value_str = "[red]Yes[/red]" if value else "[green]No[/green]"
            else:
                value_str = str(value)
            drift_table.add_row(key.replace('_', ' ').title(), value_str)
        
        console.print(drift_table)
        
        if drift_result['drift_detected']:
            print(f"[yellow]⚠[/yellow] Concept drift detected! Consider retraining your models.")
        else:
            print(f"[green]✅[/green] No significant concept drift detected.")
            
    except Exception as e:
        print(f"[red]✗[/red] Drift analysis failed: {e}")
        raise typer.Exit(1)


def _create_data_generator(input_source: str, sample_rate: float):
    """Create a data generator based on input source."""
    if input_source == "random":
        # Generate random data with occasional anomalies
        def random_generator():
            while True:
                if np.random.random() < 0.1:  # 10% anomalies
                    yield np.random.normal(3, 1, 5)  # Anomalous data
                else:
                    yield np.random.normal(0, 1, 5)  # Normal data
        return random_generator()
    
    elif input_source.startswith("file:"):
        # Load from file and stream
        file_path = Path(input_source[5:])
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        data_array = df.values.astype(np.float64)
        
        def file_generator():
            i = 0
            while True:
                yield data_array[i % len(data_array)]
                i += 1
        return file_generator()
    
    else:
        raise ValueError(f"Unsupported input source: {input_source}")


def _update_monitoring_display(layout, streaming_service, total_samples, anomaly_count, elapsed, duration, algorithm):
    """Update the live monitoring display."""
    
    # Header
    progress_pct = (elapsed / duration) * 100
    layout["header"].update(
        Panel(
            f"[bold blue]Streaming Anomaly Detection Monitor[/bold blue] | "
            f"Progress: {progress_pct:.1f}% | "
            f"Time: {elapsed}/{duration}s",
            style="blue"
        )
    )
    
    # Get streaming stats
    stats = streaming_service.get_streaming_stats()
    
    # Main content - split into two columns
    main_layout = Layout()
    main_layout.split_row(
        Layout(name="left"),
        Layout(name="right")
    )
    
    # Left panel - Current stats
    current_stats = Table(title="Current Statistics", show_header=False)
    current_stats.add_column("Metric", style="cyan")
    current_stats.add_column("Value", style="green")
    
    current_stats.add_row("Algorithm", algorithm)
    current_stats.add_row("Total Samples", str(total_samples))
    current_stats.add_row("Anomalies Found", str(anomaly_count))
    current_stats.add_row("Anomaly Rate", f"{anomaly_count/total_samples:.1%}" if total_samples > 0 else "0%")
    current_stats.add_row("Sample Rate", f"{total_samples/elapsed:.1f}/s" if elapsed > 0 else "0/s")
    
    layout_left = Panel(current_stats, title="Detection Stats", border_style="green")
    
    # Right panel - Streaming service stats  
    streaming_stats = Table(title="Streaming Service", show_header=False)
    streaming_stats.add_column("Property", style="cyan")
    streaming_stats.add_column("Value", style="green")
    
    streaming_stats.add_row("Buffer Size", f"{stats['buffer_size']}/{stats['buffer_capacity']}")
    streaming_stats.add_row("Model Fitted", "Yes" if stats['model_fitted'] else "No")
    streaming_stats.add_row("Since Update", str(stats['samples_since_update']))
    streaming_stats.add_row("Update Freq", str(stats['update_frequency']))
    
    layout_right = Panel(streaming_stats, title="Service Status", border_style="blue")
    
    main_layout["left"].update(layout_left)
    main_layout["right"].update(layout_right)
    layout["main"].update(main_layout)
    
    # Footer
    layout["footer"].update(
        Panel(
            f"[dim]Press Ctrl+C to stop monitoring[/dim]",
            style="dim"
        )
    )