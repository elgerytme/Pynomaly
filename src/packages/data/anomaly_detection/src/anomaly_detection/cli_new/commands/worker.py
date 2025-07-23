"""Worker management commands for Typer CLI."""

import typer
import asyncio
from pathlib import Path
from typing import Optional
from rich import print
from rich.console import Console
from rich.table import Table

console = Console()

app = typer.Typer(help="Background worker management commands")


@app.command()
def start(
    models_dir: Path = typer.Option(Path("models"), "--models-dir", "-d", help="Models directory"),
    max_jobs: int = typer.Option(3, "--max-jobs", help="Maximum concurrent jobs"),
    enable_monitoring: bool = typer.Option(True, "--enable-monitoring/--disable-monitoring", help="Enable monitoring"),
) -> None:
    """Start the background worker service."""
    
    print("[blue]🚀 Starting Anomaly Detection Worker[/blue]")
    print(f"   Models directory: [cyan]{models_dir}[/cyan]")
    print(f"   Max concurrent jobs: [cyan]{max_jobs}[/cyan]")
    print(f"   Monitoring enabled: [cyan]{enable_monitoring}[/cyan]")
    print("   Press Ctrl+C to stop worker\n")
    
    try:
        from ...worker import AnomalyDetectionWorker
        
        worker_instance = AnomalyDetectionWorker(
            models_dir=str(models_dir),
            max_concurrent_jobs=max_jobs,
            enable_monitoring=enable_monitoring
        )
        
        async def run_worker():
            try:
                await worker_instance.start()
            except KeyboardInterrupt:
                print("\n[yellow]🛑 Shutting down worker...[/yellow]")
                await worker_instance.stop()
                print("[green]✅ Worker stopped successfully[/green]")
        
        asyncio.run(run_worker())
        
    except KeyboardInterrupt:
        print("\n[green]✅ Worker shutdown complete[/green]")
    except ImportError:
        print("[red]✗[/red] Worker components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Failed to start worker: {e}")
        raise typer.Exit(1)


@app.command()
def submit(
    job_type: str = typer.Option(..., "--job-type", "-t", help="Type of job to submit"),
    data_source: Optional[str] = typer.Option(None, "--data-source", help="Data source (file path or direct data)"),
    algorithm: str = typer.Option("isolation_forest", "--algorithm", "-a", help="Algorithm to use"),
    contamination: float = typer.Option(0.1, "--contamination", "-c", help="Contamination rate"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    priority: str = typer.Option("normal", "--priority", "-p", help="Job priority"),
    models_dir: Path = typer.Option(Path("models"), "--models-dir", "-d", help="Models directory"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for job completion"),
) -> None:
    """Submit a job to the background worker."""
    
    try:
        from ...worker import AnomalyDetectionWorker, JobType, JobPriority
        
        # Map string values to enums
        job_type_map = {
            'detection': JobType.DETECTION,
            'ensemble': JobType.ENSEMBLE,
            'batch_training': JobType.BATCH_TRAINING,
            'stream_monitoring': JobType.STREAM_MONITORING,
            'model_validation': JobType.MODEL_VALIDATION,
            'data_preprocessing': JobType.DATA_PREPROCESSING,
            'explanation_generation': JobType.EXPLANATION_GENERATION,
            'scheduled_analysis': JobType.SCHEDULED_ANALYSIS
        }
        
        priority_map = {
            'low': JobPriority.LOW,
            'normal': JobPriority.NORMAL,
            'high': JobPriority.HIGH,
            'critical': JobPriority.CRITICAL
        }
        
        if job_type not in job_type_map:
            print(f"[red]✗[/red] Invalid job type: {job_type}")
            print(f"Valid types: {', '.join(job_type_map.keys())}")
            raise typer.Exit(1)
        
        if priority not in priority_map:
            print(f"[red]✗[/red] Invalid priority: {priority}")
            print(f"Valid priorities: {', '.join(priority_map.keys())}")
            raise typer.Exit(1)
        
        # Build job payload
        payload = {
            'algorithm': algorithm,
            'contamination': contamination
        }
        
        if data_source:
            payload['data_source'] = data_source
        if output:
            payload['output_path'] = str(output)
        
        worker_instance = AnomalyDetectionWorker(
            models_dir=str(models_dir),
            enable_monitoring=False
        )
        
        async def submit_and_wait():
            # Submit job
            job_id = await worker_instance.submit_job(
                job_type_map[job_type],
                payload,
                priority=priority_map[priority]
            )
            
            print(f"[green]✅ Job submitted successfully[/green]")
            print(f"   Job ID: [cyan]{job_id}[/cyan]")
            print(f"   Type: [cyan]{job_type}[/cyan]")
            print(f"   Priority: [cyan]{priority}[/cyan]")
            
            if wait:
                print("\n[blue]⏳ Waiting for job completion...[/blue]")
                
                # Start worker in background
                worker_task = asyncio.create_task(worker_instance.start())
                
                try:
                    # Monitor job progress
                    max_wait = 300  # 5 minutes
                    wait_time = 0
                    
                    while wait_time < max_wait:
                        status = await worker_instance.get_job_status(job_id)
                        if status:
                            job_status = status['status']
                            progress = status.get('progress', 0)
                            
                            print(f"   Status: [cyan]{job_status}[/cyan] ({progress:.1f}%)")
                            
                            if job_status in ['completed', 'failed', 'cancelled']:
                                break
                        
                        await asyncio.sleep(3)
                        wait_time += 3
                    
                    # Get final result
                    final_status = await worker_instance.get_job_status(job_id)
                    if final_status:
                        if final_status['status'] == 'completed':
                            print(f"\n[green]✅ Job completed successfully![/green]")
                        else:
                            print(f"\n[red]❌ Job {final_status['status']}[/red]")
                            if final_status.get('error_message'):
                                print(f"   Error: {final_status['error_message']}")
                    
                finally:
                    await worker_instance.stop()
                    worker_task.cancel()
                    try:
                        await worker_task
                    except asyncio.CancelledError:
                        pass
        
        asyncio.run(submit_and_wait())
        
    except ImportError:
        print("[red]✗[/red] Worker components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Failed to submit job: {e}")
        raise typer.Exit(1)


@app.command()
def status(
    job_id: str = typer.Argument(..., help="Job ID to check"),
    models_dir: Path = typer.Option(Path("models"), "--models-dir", "-d", help="Models directory"),
) -> None:
    """Check the status of a specific job."""
    
    try:
        from ...worker import AnomalyDetectionWorker
        
        worker_instance = AnomalyDetectionWorker(
            models_dir=str(models_dir),
            enable_monitoring=False
        )
        
        async def check_status():
            job_status = await worker_instance.get_job_status(job_id)
            
            if job_status:
                table = Table(title=f"[bold blue]Job Status: {job_id}[/bold blue]")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Status", job_status['status'])
                table.add_row("Type", job_status['job_type'])
                table.add_row("Priority", job_status['priority'])
                table.add_row("Progress", f"{job_status.get('progress', 0):.1f}%")
                table.add_row("Created", job_status['created_at'])
                
                if job_status.get('started_at'):
                    table.add_row("Started", job_status['started_at'])
                
                if job_status.get('completed_at'):
                    table.add_row("Completed", job_status['completed_at'])
                
                if job_status.get('error_message'):
                    table.add_row("Error", job_status['error_message'])
                
                console.print(table)
            else:
                print(f"[red]❌ Job not found: {job_id}[/red]")
        
        asyncio.run(check_status())
        
    except ImportError:
        print("[red]✗[/red] Worker components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Failed to check job status: {e}")
        raise typer.Exit(1)


@app.command()
def info(
    models_dir: Path = typer.Option(Path("models"), "--models-dir", "-d", help="Models directory"),
) -> None:
    """Show worker system information."""
    
    try:
        from ...worker import AnomalyDetectionWorker
        
        worker_instance = AnomalyDetectionWorker(
            models_dir=str(models_dir),
            enable_monitoring=False
        )
        
        async def show_info():
            worker_status = await worker_instance.get_worker_status()
            queue_status = worker_status['queue_status']
            
            # Worker info table
            worker_table = Table(title="[bold blue]Worker System Information[/bold blue]")
            worker_table.add_column("Property", style="cyan")
            worker_table.add_column("Value", style="green")
            
            worker_table.add_row("Running", str(worker_status['is_running']))
            worker_table.add_row("Max Concurrent Jobs", str(worker_status['max_concurrent_jobs']))
            worker_table.add_row("Currently Running", str(worker_status['currently_running_jobs']))
            worker_table.add_row("Monitoring Enabled", str(worker_status['monitoring_enabled']))
            
            console.print(worker_table)
            
            # Queue status table
            queue_table = Table(title="[bold blue]Job Queue Status[/bold blue]")
            queue_table.add_column("Metric", style="cyan")
            queue_table.add_column("Value", style="green", justify="right")
            
            queue_table.add_row("Pending Jobs", str(queue_status['pending_jobs']))
            queue_table.add_row("Total Jobs", str(queue_status['total_jobs']))
            
            console.print(queue_table)
            
            # Status counts
            if queue_status.get('status_counts'):
                status_table = Table(title="[bold blue]Job Status Counts[/bold blue]")
                status_table.add_column("Status", style="cyan")
                status_table.add_column("Count", style="green", justify="right")
                
                for status, count in queue_status['status_counts'].items():
                    status_table.add_row(status, str(count))
                
                console.print(status_table)
        
        asyncio.run(show_info())
        
    except ImportError:
        print("[red]✗[/red] Worker components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Failed to get worker info: {e}")
        raise typer.Exit(1)