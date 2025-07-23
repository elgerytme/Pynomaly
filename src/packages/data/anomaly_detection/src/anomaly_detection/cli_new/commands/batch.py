"""Batch processing commands for anomaly detection."""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel

console = Console()

app = typer.Typer(help="Batch processing commands for large-scale anomaly detection")


@app.command()
def detect(
    input_files: List[Path] = typer.Argument(..., help="Input files to process"),
    output_dir: Path = typer.Option(Path("./batch_results"), "--output-dir", "-o", help="Output directory for results"),
    algorithms: List[str] = typer.Option(["isolation_forest"], "--algorithm", "-a", help="Algorithms to use"),
    contamination: float = typer.Option(0.1, "--contamination", "-c", help="Expected contamination rate"),
    parallel_jobs: int = typer.Option(4, "--parallel", "-p", help="Number of parallel processing jobs"),
    chunk_size: int = typer.Option(1000, "--chunk-size", help="Chunk size for processing large files"),
    output_format: str = typer.Option("json", "--format", "-f", help="Output format (json, csv, parquet)"),
    save_models: bool = typer.Option(False, "--save-models", help="Save trained models for each file"),
    resume: bool = typer.Option(False, "--resume", help="Resume from previous incomplete run"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run batch anomaly detection on multiple files."""
    
    try:
        from ...domain.services.detection_service import DetectionService
        from ...domain.services.batch_processing_service import BatchProcessingService
        from ...infrastructure.repositories.model_repository import ModelRepository
        
        # Validate input files
        valid_files = []
        for file_path in input_files:
            if file_path.exists() and file_path.is_file():
                valid_files.append(file_path)
            else:
                print(f"[yellow]⚠[/yellow] Warning: File not found: {file_path}")
        
        if not valid_files:
            print("[red]✗[/red] No valid input files found")
            raise typer.Exit(1)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        detection_service = DetectionService()
        model_repository = ModelRepository() if save_models else None
        batch_service = BatchProcessingService(
            detection_service=detection_service,
            model_repository=model_repository,
            parallel_jobs=parallel_jobs,
            chunk_size=chunk_size
        )
        
        print(f"[blue]🔄 Starting batch anomaly detection[/blue]")
        print(f"   Files to process: [cyan]{len(valid_files)}[/cyan]")
        print(f"   Algorithms: [cyan]{', '.join(algorithms)}[/cyan]")
        print(f"   Parallel jobs: [cyan]{parallel_jobs}[/cyan]")
        print(f"   Output directory: [cyan]{output_dir}[/cyan]")
        print()
        
        async def run_batch():
            # Configure batch job
            batch_config = {
                'algorithms': algorithms,
                'contamination': contamination,
                'output_format': output_format,
                'save_models': save_models,
                'resume': resume,
                'verbose': verbose
            }
            
            results = []
            failed_files = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                # Create main progress task
                main_task = progress.add_task("Processing files...", total=len(valid_files))
                
                # Process files
                for i, file_path in enumerate(valid_files):
                    try:
                        file_task = progress.add_task(f"Processing {file_path.name}", total=100)
                        
                        # Update progress callback
                        def progress_callback(percent: float):
                            progress.update(file_task, completed=percent)
                        
                        # Process single file
                        result = await batch_service.process_file(
                            input_file=file_path,
                            output_dir=output_dir,
                            config=batch_config,
                            progress_callback=progress_callback
                        )
                        
                        results.append(result)
                        progress.update(file_task, completed=100)
                        progress.update(main_task, completed=i + 1)
                        
                        if verbose:
                            print(f"[green]✓[/green] Completed: {file_path.name}")
                            print(f"   Anomalies detected: {result.get('anomaly_count', 0)}")
                            print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
                        
                    except Exception as e:
                        failed_files.append((file_path, str(e)))
                        progress.update(file_task, completed=100)
                        progress.update(main_task, completed=i + 1)
                        
                        if verbose:
                            print(f"[red]✗[/red] Failed: {file_path.name} - {str(e)}")
            
            # Generate summary report
            summary_report = await batch_service.generate_summary_report(
                results=results,
                failed_files=failed_files,
                output_dir=output_dir
            )
            
            # Display results
            print(f"\\n[bold green]📊 Batch Processing Complete[/bold green]")
            print(f"   Total files: [cyan]{len(valid_files)}[/cyan]")
            print(f"   Successful: [green]{len(results)}[/green]")
            print(f"   Failed: [red]{len(failed_files)}[/red]")
            print(f"   Total anomalies detected: [yellow]{summary_report['total_anomalies']}[/yellow]")
            print(f"   Average processing time: [blue]{summary_report['avg_processing_time']:.2f}s[/blue]")
            print(f"   Summary report: [cyan]{summary_report['report_path']}[/cyan]")
            
            if failed_files:
                print(f"\\n[red]Failed Files:[/red]")
                for file_path, error in failed_files:
                    print(f"   • {file_path.name}: {error}")
        
        asyncio.run(run_batch())
        
    except ImportError:
        print("[red]✗[/red] Batch processing components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Batch processing failed: {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    input_files: List[Path] = typer.Argument(..., help="Files to validate"),
    schema_file: Optional[Path] = typer.Option(None, "--schema", "-s", help="JSON schema file for validation"),
    data_types: bool = typer.Option(True, "--check-types", help="Validate data types"),
    missing_values: bool = typer.Option(True, "--check-missing", help="Check for missing values"),
    outliers: bool = typer.Option(True, "--check-outliers", help="Check for statistical outliers"),
    duplicates: bool = typer.Option(True, "--check-duplicates", help="Check for duplicate rows"),
    output_report: Optional[Path] = typer.Option(None, "--report", "-r", help="Output validation report"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Validate data files for quality and consistency."""
    
    try:
        from ...domain.services.data_validation_service import DataValidationService
        
        validation_service = DataValidationService()
        
        print(f"[blue]🔍 Starting data validation[/blue]")
        print(f"   Files to validate: [cyan]{len(input_files)}[/cyan]")
        print()
        
        async def run_validation():
            validation_results = []
            
            for file_path in input_files:
                if not file_path.exists():
                    print(f"[red]✗[/red] File not found: {file_path}")
                    continue
                
                print(f"[blue]Validating:[/blue] {file_path.name}")
                
                # Run validation
                result = await validation_service.validate_file(
                    file_path=file_path,
                    schema_file=schema_file,
                    check_types=data_types,
                    check_missing=missing_values,
                    check_outliers=outliers,
                    check_duplicates=duplicates
                )
                
                validation_results.append(result)
                
                # Display results
                if result['is_valid']:
                    print(f"   [green]✓ Valid[/green]")
                else:
                    print(f"   [red]✗ Invalid[/red] ({len(result['errors'])} errors)")
                
                if verbose or not result['is_valid']:
                    for error in result['errors']:
                        print(f"     • {error}")
                    
                    for warning in result['warnings']:
                        print(f"     [yellow]⚠[/yellow] {warning}")
                
                print()
            
            # Generate summary report
            if output_report:
                summary = {
                    'validation_date': datetime.utcnow().isoformat(),
                    'total_files': len(input_files),
                    'valid_files': sum(1 for r in validation_results if r['is_valid']),
                    'invalid_files': sum(1 for r in validation_results if not r['is_valid']),
                    'results': validation_results
                }
                
                with open(output_report, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                
                print(f"[blue]📄 Validation report saved:[/blue] {output_report}")
            
            # Summary
            valid_count = sum(1 for r in validation_results if r['is_valid'])
            print(f"[bold blue]📊 Validation Summary[/bold blue]")
            print(f"   Total files: [cyan]{len(validation_results)}[/cyan]")
            print(f"   Valid: [green]{valid_count}[/green]")
            print(f"   Invalid: [red]{len(validation_results) - valid_count}[/red]")
        
        asyncio.run(run_validation())
        
    except ImportError:
        print("[red]✗[/red] Data validation components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Data validation failed: {e}")
        raise typer.Exit(1)


@app.command()
def convert(
    input_files: List[Path] = typer.Argument(..., help="Files to convert"),
    output_format: str = typer.Option(..., "--format", "-f", help="Target format (csv, json, parquet, hdf5)"),
    output_dir: Path = typer.Option(Path("./converted"), "--output-dir", "-o", help="Output directory"),
    compression: Optional[str] = typer.Option(None, "--compression", "-c", help="Compression (gzip, bz2, xz)"),
    chunk_size: int = typer.Option(10000, "--chunk-size", help="Chunk size for large files"),
    preserve_dtypes: bool = typer.Option(True, "--preserve-types", help="Preserve data types during conversion"),
) -> None:
    """Convert data files between different formats."""
    
    try:
        from ...domain.services.data_conversion_service import DataConversionService
        
        conversion_service = DataConversionService()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[blue]🔄 Starting batch file conversion[/blue]")
        print(f"   Files to convert: [cyan]{len(input_files)}[/cyan]")
        print(f"   Target format: [cyan]{output_format}[/cyan]")
        print(f"   Output directory: [cyan]{output_dir}[/cyan]")
        print()
        
        async def run_conversion():
            successful = 0
            failed = 0
            
            for file_path in input_files:
                if not file_path.exists():
                    print(f"[red]✗[/red] File not found: {file_path}")
                    failed += 1
                    continue
                
                try:
                    start_time = time.time()
                    
                    output_file = await conversion_service.convert_file(
                        input_file=file_path,
                        output_format=output_format,
                        output_dir=output_dir,
                        compression=compression,
                        chunk_size=chunk_size,
                        preserve_dtypes=preserve_dtypes
                    )
                    
                    processing_time = time.time() - start_time
                    
                    print(f"[green]✓[/green] Converted: {file_path.name} → {output_file.name} ({processing_time:.2f}s)")
                    successful += 1
                    
                except Exception as e:
                    print(f"[red]✗[/red] Failed: {file_path.name} - {str(e)}")
                    failed += 1
            
            print(f"\\n[bold blue]📊 Conversion Summary[/bold blue]")
            print(f"   Successful: [green]{successful}[/green]")
            print(f"   Failed: [red]{failed}[/red]")
        
        asyncio.run(run_conversion())
        
    except ImportError:
        print("[red]✗[/red] Data conversion components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] File conversion failed: {e}")
        raise typer.Exit(1)


@app.command()
def profile(
    input_files: List[Path] = typer.Argument(..., help="Files to profile"),
    output_dir: Path = typer.Option(Path("./profiles"), "--output-dir", "-o", help="Output directory for profiles"),
    include_correlations: bool = typer.Option(True, "--correlations", help="Include correlation analysis"),
    include_distributions: bool = typer.Option(True, "--distributions", help="Include distribution analysis"),
    generate_plots: bool = typer.Option(False, "--plots", help="Generate visualization plots"),
    sample_size: Optional[int] = typer.Option(None, "--sample", "-n", help="Sample size for large files"),
) -> None:
    """Generate comprehensive data profiles and statistics."""
    
    try:
        from ...domain.services.data_profiling_service import DataProfilingService
        
        profiling_service = DataProfilingService()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[blue]📊 Starting data profiling[/blue]")
        print(f"   Files to profile: [cyan]{len(input_files)}[/cyan]")
        print(f"   Output directory: [cyan]{output_dir}[/cyan]")
        print()
        
        async def run_profiling():
            for file_path in input_files:
                if not file_path.exists():
                    print(f"[red]✗[/red] File not found: {file_path}")
                    continue
                
                try:
                    print(f"[blue]Profiling:[/blue] {file_path.name}")
                    
                    profile = await profiling_service.profile_file(
                        file_path=file_path,
                        include_correlations=include_correlations,
                        include_distributions=include_distributions,
                        generate_plots=generate_plots,
                        sample_size=sample_size
                    )
                    
                    # Save profile report
                    profile_file = output_dir / f"{file_path.stem}_profile.json"
                    with open(profile_file, 'w') as f:
                        json.dump(profile, f, indent=2, default=str)
                    
                    # Display summary
                    print(f"   [green]✓[/green] Rows: {profile['dataset_info']['row_count']:,}")
                    print(f"   [green]✓[/green] Columns: {profile['dataset_info']['column_count']}")
                    print(f"   [green]✓[/green] Missing values: {profile['data_quality']['missing_values_total']:,}")
                    print(f"   [green]✓[/green] Profile saved: {profile_file}")
                    
                    if generate_plots:
                        plots_dir = output_dir / f"{file_path.stem}_plots"
                        print(f"   [green]✓[/green] Plots saved: {plots_dir}")
                    
                    print()
                    
                except Exception as e:
                    print(f"   [red]✗[/red] Failed to profile: {str(e)}")
                    print()
        
        asyncio.run(run_profiling())
        
    except ImportError:
        print("[red]✗[/red] Data profiling components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Data profiling failed: {e}")
        raise typer.Exit(1)


@app.command()
def sample(
    input_files: List[Path] = typer.Argument(..., help="Files to sample"),
    sample_size: int = typer.Option(1000, "--size", "-n", help="Number of samples to extract"),
    method: str = typer.Option("random", "--method", "-m", help="Sampling method (random, systematic, stratified)"),
    output_dir: Path = typer.Option(Path("./samples"), "--output-dir", "-o", help="Output directory"),
    stratify_column: Optional[str] = typer.Option(None, "--stratify", help="Column for stratified sampling"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility"),
) -> None:
    """Sample data from files using various sampling methods."""
    
    try:
        from ...domain.services.data_sampling_service import DataSamplingService
        
        sampling_service = DataSamplingService()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[blue]🎯 Starting data sampling[/blue]")
        print(f"   Files to sample: [cyan]{len(input_files)}[/cyan]")
        print(f"   Sample size: [cyan]{sample_size}[/cyan]")
        print(f"   Method: [cyan]{method}[/cyan]")
        print(f"   Output directory: [cyan]{output_dir}[/cyan]")
        print()
        
        async def run_sampling():
            for file_path in input_files:
                if not file_path.exists():
                    print(f"[red]✗[/red] File not found: {file_path}")
                    continue
                
                try:
                    print(f"[blue]Sampling:[/blue] {file_path.name}")
                    
                    sample_data = await sampling_service.sample_file(
                        file_path=file_path,
                        sample_size=sample_size,
                        method=method,
                        stratify_column=stratify_column,
                        seed=seed
                    )
                    
                    # Save sample
                    sample_file = output_dir / f"{file_path.stem}_sample_{method}_{sample_size}.csv"
                    sample_data.to_csv(sample_file, index=False)
                    
                    print(f"   [green]✓[/green] Sample created: {len(sample_data)} rows")
                    print(f"   [green]✓[/green] Saved: {sample_file}")
                    print()
                    
                except Exception as e:
                    print(f"   [red]✗[/red] Failed to sample: {str(e)}")
                    print()
        
        asyncio.run(run_sampling())
        
    except ImportError:
        print("[red]✗[/red] Data sampling components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Data sampling failed: {e}")
        raise typer.Exit(1)