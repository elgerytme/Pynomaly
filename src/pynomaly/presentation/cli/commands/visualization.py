"""
Data Visualization CLI Commands

Provides comprehensive data visualization capabilities through command-line interface.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import json

from ....shared.error_handling import handle_cli_errors
from ....shared.logging import get_logger
from ....infrastructure.config import get_cli_container

logger = get_logger(__name__)
console = Console()

# Create the viz command group
app = typer.Typer(
    name="viz",
    help="📊 Data visualization and charting tools",
    rich_markup_mode="rich"
)


@app.command("plot")
@handle_cli_errors
def create_plot(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    plot_type: str = typer.Argument(..., help="Plot type: scatter, line, bar, histogram, box, heatmap"),
    x_column: str = typer.Option(..., "--x", help="X-axis column name"),
    y_column: Optional[str] = typer.Option(None, "--y", help="Y-axis column name"),
    color_column: Optional[str] = typer.Option(None, "--color", help="Column for color grouping"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    title: Optional[str] = typer.Option(None, "--title", help="Plot title"),
    theme: str = typer.Option("default", "--theme", help="Plot theme: default, dark, light, minimal"),
    width: int = typer.Option(800, "--width", help="Plot width in pixels"),
    height: int = typer.Option(600, "--height", help="Plot height in pixels"),
    format: str = typer.Option("html", "--format", help="Output format: html, png, svg, pdf"),
    interactive: bool = typer.Option(True, "--interactive/--static", help="Create interactive plot"),
    show: bool = typer.Option(True, "--show/--no-show", help="Display plot after creation")
):
    """
    Create interactive data visualizations from dataset features.
    
    Supports multiple chart types with customization options and export capabilities.
    
    Examples:
        pynomaly viz plot data.csv scatter --x age --y income
        pynomaly viz plot data.csv histogram --x score --output plot.html
        pynomaly viz plot data.csv bar --x category --y value --color group
    """
    console.print(f"[bold blue]📊 Creating {plot_type} plot from {data_path}[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading dataset...", total=None)
            
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.offline import plot
            
            # Load dataset
            if not data_path.exists():
                console.print(f"[red]❌ Dataset file not found: {data_path}[/red]")
                raise typer.Exit(1)
            
            try:
                if data_path.suffix.lower() == '.csv':
                    df = pd.read_csv(data_path)
                elif data_path.suffix.lower() in ['.json']:
                    df = pd.read_json(data_path)
                elif data_path.suffix.lower() in ['.parquet']:
                    df = pd.read_parquet(data_path)
                else:
                    console.print(f"[red]❌ Unsupported file format: {data_path.suffix}[/red]")
                    raise typer.Exit(1)
            except Exception as e:
                console.print(f"[red]❌ Error loading dataset: {e}[/red]")
                raise typer.Exit(1)
            
            # Validate columns
            required_cols = [x_column]
            if y_column:
                required_cols.append(y_column)
            if color_column:
                required_cols.append(color_column)
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                console.print(f"[red]❌ Missing columns: {missing_cols}[/red]")
                available_cols = list(df.columns)
                console.print(f"Available columns: {', '.join(available_cols)}")
                raise typer.Exit(1)
            
            progress.update(task, description="Creating visualization...")
            
            # Set plot title
            if not title:
                title = f"{plot_type.title()} Plot: {x_column}"
                if y_column:
                    title += f" vs {y_column}"
            
            # Create plot based on type
            fig = None
            
            if plot_type == "scatter":
                if not y_column:
                    console.print("[red]❌ Scatter plot requires both x and y columns[/red]")
                    raise typer.Exit(1)
                fig = px.scatter(
                    df, x=x_column, y=y_column, color=color_column,
                    title=title, template=theme,
                    width=width, height=height
                )
            
            elif plot_type == "line":
                if not y_column:
                    console.print("[red]❌ Line plot requires both x and y columns[/red]")
                    raise typer.Exit(1)
                fig = px.line(
                    df, x=x_column, y=y_column, color=color_column,
                    title=title, template=theme,
                    width=width, height=height
                )
            
            elif plot_type == "bar":
                if y_column:
                    fig = px.bar(
                        df, x=x_column, y=y_column, color=color_column,
                        title=title, template=theme,
                        width=width, height=height
                    )
                else:
                    # Count plot
                    value_counts = df[x_column].value_counts()
                    fig = px.bar(
                        x=value_counts.index, y=value_counts.values,
                        title=title, template=theme,
                        width=width, height=height
                    )
                    fig.update_xaxes(title=x_column)
                    fig.update_yaxes(title="Count")
            
            elif plot_type == "histogram":
                fig = px.histogram(
                    df, x=x_column, color=color_column,
                    title=title, template=theme,
                    width=width, height=height
                )
            
            elif plot_type == "box":
                if y_column:
                    fig = px.box(
                        df, x=x_column, y=y_column, color=color_column,
                        title=title, template=theme,
                        width=width, height=height
                    )
                else:
                    fig = px.box(
                        df, y=x_column, color=color_column,
                        title=title, template=theme,
                        width=width, height=height
                    )
            
            elif plot_type == "heatmap":
                # Create correlation heatmap for numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) < 2:
                    console.print("[red]❌ Heatmap requires at least 2 numeric columns[/red]")
                    raise typer.Exit(1)
                
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    title=title or "Correlation Heatmap",
                    template=theme,
                    width=width, height=height,
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                fig.update_layout(
                    xaxis_title="Features",
                    yaxis_title="Features"
                )
            
            else:
                console.print(f"[red]❌ Unsupported plot type: {plot_type}[/red]")
                console.print("Supported types: scatter, line, bar, histogram, box, heatmap")
                raise typer.Exit(1)
            
            if not fig:
                console.print("[red]❌ Failed to create plot[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description="Saving plot...")
            
            # Set output path
            if not output:
                output = Path(f"{plot_type}_plot.{format}")
            
            # Save plot
            if format == "html":
                plot(fig, filename=str(output), auto_open=show)
                console.print(f"[green]✅ Interactive plot saved to {output}[/green]")
            elif format in ["png", "svg", "pdf"]:
                try:
                    fig.write_image(str(output), format=format, width=width, height=height)
                    console.print(f"[green]✅ Static plot saved to {output}[/green]")
                except Exception as e:
                    console.print(f"[yellow]⚠️  Error saving as {format}: {e}[/yellow]")
                    console.print("Note: Static image export requires kaleido package")
                    # Fallback to HTML
                    html_output = output.with_suffix('.html')
                    plot(fig, filename=str(html_output), auto_open=show)
                    console.print(f"[green]✅ Plot saved as HTML to {html_output}[/green]")
            else:
                console.print(f"[red]❌ Unsupported format: {format}[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description="Complete", completed=True)
        
        console.print("[green]✅ Visualization created successfully[/green]")
        
        # Display plot metadata
        _display_plot_info(df, x_column, y_column, color_column, plot_type)
        
    except Exception as e:
        logger.error(f"Error creating plot: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("dashboard")
@handle_cli_errors
def create_dashboard(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Dashboard configuration file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output HTML file"),
    title: str = typer.Option("Data Dashboard", "--title", help="Dashboard title"),
    theme: str = typer.Option("default", "--theme", help="Dashboard theme"),
    auto_layout: bool = typer.Option(True, "--auto/--manual", help="Automatic layout generation"),
    show: bool = typer.Option(True, "--show/--no-show", help="Open dashboard in browser")
):
    """
    Create interactive dashboards with multiple visualizations.
    
    Automatically generates comprehensive dashboards or uses custom configuration files.
    
    Examples:
        pynomaly viz dashboard data.csv
        pynomaly viz dashboard data.csv --config dashboard_config.json
        pynomaly viz dashboard data.csv --title "Sales Dashboard" --output sales.html
    """
    console.print(f"[bold blue]🏗️  Creating dashboard from {data_path}[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading dataset...", total=None)
            
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            from plotly.offline import plot
            
            # Load dataset
            if not data_path.exists():
                console.print(f"[red]❌ Dataset file not found: {data_path}[/red]")
                raise typer.Exit(1)
            
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix.lower() in ['.json']:
                df = pd.read_json(data_path)
            elif data_path.suffix.lower() in ['.parquet']:
                df = pd.read_parquet(data_path)
            else:
                console.print(f"[red]❌ Unsupported file format: {data_path.suffix}[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description="Analyzing data structure...")
            
            # Analyze data types
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            if len(numeric_cols) == 0:
                console.print("[red]❌ Dataset contains no numeric columns for visualization[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description="Creating dashboard layout...")
            
            # Load configuration or create auto layout
            if config_path and config_path.exists():
                with open(config_path) as f:
                    dashboard_config = json.load(f)
                plots = dashboard_config.get('plots', [])
            elif auto_layout:
                plots = _generate_auto_layout(df, numeric_cols, categorical_cols, datetime_cols)
            else:
                console.print("[red]❌ No configuration provided and auto-layout disabled[/red]")
                raise typer.Exit(1)
            
            if not plots:
                console.print("[red]❌ No plots configured for dashboard[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description="Generating visualizations...")
            
            # Create subplots
            n_plots = len(plots)
            cols = min(2, n_plots)
            rows = (n_plots + cols - 1) // cols
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[plot.get('title', f'Plot {i+1}') for i, plot in enumerate(plots)],
                specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
            )
            
            # Generate each plot
            for i, plot_config in enumerate(plots):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                try:
                    _add_subplot_to_dashboard(fig, df, plot_config, row, col)
                except Exception as e:
                    console.print(f"[yellow]⚠️  Skipping plot {i+1}: {e}[/yellow]")
            
            # Update layout
            fig.update_layout(
                title=title,
                template=theme,
                height=400 * rows,
                showlegend=True,
                margin=dict(t=100, b=50, l=50, r=50)
            )
            
            progress.update(task, description="Saving dashboard...")
            
            # Set output path
            if not output:
                output = Path("dashboard.html")
            
            # Save dashboard
            plot(fig, filename=str(output), auto_open=show)
            
            progress.update(task, description="Complete", completed=True)
        
        console.print(f"[green]✅ Dashboard created successfully: {output}[/green]")
        
        # Display dashboard info
        _display_dashboard_info(df, plots, output)
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("explore")
@handle_cli_errors
def exploratory_analysis(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    format: str = typer.Option("html", "--format", help="Output format: html, png, svg"),
    include_correlations: bool = typer.Option(True, "--correlations/--no-correlations", help="Include correlation analysis"),
    include_distributions: bool = typer.Option(True, "--distributions/--no-distributions", help="Include distribution plots"),
    include_outliers: bool = typer.Option(True, "--outliers/--no-outliers", help="Include outlier analysis"),
    sample_size: Optional[int] = typer.Option(None, "--sample", help="Sample size for large datasets"),
    show: bool = typer.Option(True, "--show/--no-show", help="Open plots in browser")
):
    """
    Generate comprehensive exploratory data analysis (EDA) visualizations.
    
    Creates a complete set of visualizations for understanding dataset characteristics.
    
    Examples:
        pynomaly viz explore data.csv
        pynomaly viz explore data.csv --output eda_plots --format png
        pynomaly viz explore large_data.csv --sample 10000
    """
    console.print(f"[bold blue]🔍 Performing exploratory analysis of {data_path}[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading and sampling data...", total=None)
            
            import pandas as pd
            import plotly.express as px
            import plotly.figure_factory as ff
            import numpy as np
            from plotly.offline import plot
            
            # Load dataset
            if not data_path.exists():
                console.print(f"[red]❌ Dataset file not found: {data_path}[/red]")
                raise typer.Exit(1)
            
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix.lower() in ['.json']:
                df = pd.read_json(data_path)
            elif data_path.suffix.lower() in ['.parquet']:
                df = pd.read_parquet(data_path)
            else:
                console.print(f"[red]❌ Unsupported file format: {data_path.suffix}[/red]")
                raise typer.Exit(1)
            
            # Sample data if needed
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                console.print(f"[yellow]📊 Sampled {sample_size} rows from {len(df)} total[/yellow]")
            
            # Analyze data structure
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(numeric_cols) == 0:
                console.print("[red]❌ No numeric columns found for analysis[/red]")
                raise typer.Exit(1)
            
            # Set output directory
            if not output_dir:
                output_dir = Path("eda_output")
            output_dir.mkdir(exist_ok=True)
            
            plots_created = []
            
            # 1. Distribution plots
            if include_distributions and numeric_cols:
                progress.update(task, description="Creating distribution plots...")
                
                for col in numeric_cols[:6]:  # Limit to first 6 columns
                    try:
                        fig = px.histogram(
                            df, x=col, 
                            title=f"Distribution of {col}",
                            template="plotly_white"
                        )
                        
                        output_file = output_dir / f"distribution_{col}.{format}"
                        if format == "html":
                            plot(fig, filename=str(output_file), auto_open=False)
                        else:
                            fig.write_image(str(output_file))
                        
                        plots_created.append(output_file)
                    except Exception as e:
                        console.print(f"[yellow]⚠️  Skipping distribution for {col}: {e}[/yellow]")
            
            # 2. Correlation heatmap
            if include_correlations and len(numeric_cols) > 1:
                progress.update(task, description="Creating correlation heatmap...")
                
                try:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(
                        corr_matrix,
                        title="Feature Correlation Heatmap",
                        template="plotly_white",
                        color_continuous_scale="RdBu",
                        aspect="auto"
                    )
                    
                    output_file = output_dir / f"correlation_heatmap.{format}"
                    if format == "html":
                        plot(fig, filename=str(output_file), auto_open=False)
                    else:
                        fig.write_image(str(output_file))
                    
                    plots_created.append(output_file)
                except Exception as e:
                    console.print(f"[yellow]⚠️  Skipping correlation heatmap: {e}[/yellow]")
            
            # 3. Box plots for outlier detection
            if include_outliers and numeric_cols:
                progress.update(task, description="Creating outlier analysis...")
                
                for col in numeric_cols[:4]:  # Limit to first 4 columns
                    try:
                        fig = px.box(
                            df, y=col,
                            title=f"Outlier Analysis: {col}",
                            template="plotly_white"
                        )
                        
                        output_file = output_dir / f"outliers_{col}.{format}"
                        if format == "html":
                            plot(fig, filename=str(output_file), auto_open=False)
                        else:
                            fig.write_image(str(output_file))
                        
                        plots_created.append(output_file)
                    except Exception as e:
                        console.print(f"[yellow]⚠️  Skipping outlier analysis for {col}: {e}[/yellow]")
            
            # 4. Categorical value counts
            if categorical_cols:
                progress.update(task, description="Creating categorical analysis...")
                
                for col in categorical_cols[:3]:  # Limit to first 3 columns
                    try:
                        value_counts = df[col].value_counts().head(20)  # Top 20 values
                        
                        fig = px.bar(
                            x=value_counts.index, y=value_counts.values,
                            title=f"Value Counts: {col}",
                            template="plotly_white"
                        )
                        fig.update_xaxes(title=col)
                        fig.update_yaxes(title="Count")
                        
                        output_file = output_dir / f"categorical_{col}.{format}"
                        if format == "html":
                            plot(fig, filename=str(output_file), auto_open=False)
                        else:
                            fig.write_image(str(output_file))
                        
                        plots_created.append(output_file)
                    except Exception as e:
                        console.print(f"[yellow]⚠️  Skipping categorical analysis for {col}: {e}[/yellow]")
            
            # 5. Pairwise scatter plots (if not too many numeric columns)
            if len(numeric_cols) >= 2 and len(numeric_cols) <= 5:
                progress.update(task, description="Creating pairwise plots...")
                
                try:
                    fig = px.scatter_matrix(
                        df[numeric_cols],
                        title="Pairwise Relationships",
                        template="plotly_white"
                    )
                    
                    output_file = output_dir / f"pairwise_plots.{format}"
                    if format == "html":
                        plot(fig, filename=str(output_file), auto_open=False)
                    else:
                        fig.write_image(str(output_file))
                    
                    plots_created.append(output_file)
                except Exception as e:
                    console.print(f"[yellow]⚠️  Skipping pairwise plots: {e}[/yellow]")
            
            progress.update(task, description="Complete", completed=True)
        
        console.print(f"[green]✅ Exploratory analysis complete: {len(plots_created)} plots created[/green]")
        
        # Display summary
        _display_eda_summary(df, numeric_cols, categorical_cols, plots_created, output_dir)
        
        if show and plots_created:
            # Open first HTML plot if available
            html_plots = [p for p in plots_created if p.suffix == '.html']
            if html_plots:
                import webbrowser
                webbrowser.open(f"file://{html_plots[0].absolute()}")
        
    except Exception as e:
        logger.error(f"Error in exploratory analysis: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


# Helper functions

def _display_plot_info(df: "pd.DataFrame", x_col: str, y_col: Optional[str], 
                      color_col: Optional[str], plot_type: str):
    """Display information about the created plot."""
    table = Table(title="📊 Plot Information", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Plot Type", plot_type.title())
    table.add_row("Dataset Size", f"{len(df)} rows, {len(df.columns)} columns")
    table.add_row("X-axis", x_col)
    if y_col:
        table.add_row("Y-axis", y_col)
    if color_col:
        table.add_row("Color", color_col)
    
    # Data type info
    if x_col in df.columns:
        x_dtype = str(df[x_col].dtype)
        table.add_row("X-axis Type", x_dtype)
    
    if y_col and y_col in df.columns:
        y_dtype = str(df[y_col].dtype)
        table.add_row("Y-axis Type", y_dtype)
    
    console.print(table)


def _generate_auto_layout(df: "pd.DataFrame", numeric_cols: List[str], 
                         categorical_cols: List[str], datetime_cols: List[str]) -> List[Dict]:
    """Generate automatic dashboard layout based on data characteristics."""
    plots = []
    
    # 1. Distribution plots for first few numeric columns
    for col in numeric_cols[:3]:
        plots.append({
            'type': 'histogram',
            'x': col,
            'title': f'Distribution of {col}'
        })
    
    # 2. Correlation heatmap if enough numeric columns
    if len(numeric_cols) > 2:
        plots.append({
            'type': 'heatmap',
            'title': 'Feature Correlations'
        })
    
    # 3. Categorical value counts
    for col in categorical_cols[:2]:
        plots.append({
            'type': 'bar',
            'x': col,
            'title': f'Value Counts: {col}'
        })
    
    # 4. Scatter plot if we have at least 2 numeric columns
    if len(numeric_cols) >= 2:
        color_col = categorical_cols[0] if categorical_cols else None
        plots.append({
            'type': 'scatter',
            'x': numeric_cols[0],
            'y': numeric_cols[1],
            'color': color_col,
            'title': f'{numeric_cols[0]} vs {numeric_cols[1]}'
        })
    
    # 5. Box plots for outlier detection
    for col in numeric_cols[:2]:
        plots.append({
            'type': 'box',
            'y': col,
            'title': f'Outliers: {col}'
        })
    
    return plots


def _add_subplot_to_dashboard(fig, df: "pd.DataFrame", plot_config: Dict, row: int, col: int):
    """Add a subplot to the dashboard figure."""
    import plotly.express as px
    import plotly.graph_objects as go
    
    plot_type = plot_config.get('type', 'scatter')
    
    if plot_type == 'scatter':
        x_col = plot_config['x']
        y_col = plot_config['y']
        color_col = plot_config.get('color')
        
        if color_col and color_col in df.columns:
            for i, category in enumerate(df[color_col].unique()):
                subset = df[df[color_col] == category]
                fig.add_trace(
                    go.Scatter(
                        x=subset[x_col], y=subset[y_col],
                        mode='markers',
                        name=str(category),
                        showlegend=(row == 1 and col == 1)  # Only show legend for first plot
                    ),
                    row=row, col=col
                )
        else:
            fig.add_trace(
                go.Scatter(x=df[x_col], y=df[y_col], mode='markers'),
                row=row, col=col
            )
    
    elif plot_type == 'histogram':
        x_col = plot_config['x']
        fig.add_trace(
            go.Histogram(x=df[x_col], showlegend=False),
            row=row, col=col
        )
    
    elif plot_type == 'bar':
        x_col = plot_config['x']
        value_counts = df[x_col].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=value_counts.index, y=value_counts.values, showlegend=False),
            row=row, col=col
        )
    
    elif plot_type == 'box':
        y_col = plot_config['y']
        fig.add_trace(
            go.Box(y=df[y_col], showlegend=False),
            row=row, col=col
        )
    
    elif plot_type == 'heatmap':
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    showlegend=False
                ),
                row=row, col=col
            )


def _display_dashboard_info(df: "pd.DataFrame", plots: List[Dict], output_path: Path):
    """Display dashboard creation summary."""
    table = Table(title="🏗️  Dashboard Summary", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Output File", str(output_path))
    table.add_row("Dataset Size", f"{len(df)} rows, {len(df.columns)} columns")
    table.add_row("Number of Plots", str(len(plots)))
    
    plot_types = [plot.get('type', 'unknown') for plot in plots]
    type_counts = {t: plot_types.count(t) for t in set(plot_types)}
    table.add_row("Plot Types", ", ".join([f"{t}: {c}" for t, c in type_counts.items()]))
    
    console.print(table)


def _display_eda_summary(df: "pd.DataFrame", numeric_cols: List[str], 
                        categorical_cols: List[str], plots_created: List[Path], 
                        output_dir: Path):
    """Display EDA summary information."""
    table = Table(title="🔍 EDA Summary", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Output Directory", str(output_dir))
    table.add_row("Dataset Size", f"{len(df)} rows, {len(df.columns)} columns")
    table.add_row("Numeric Features", str(len(numeric_cols)))
    table.add_row("Categorical Features", str(len(categorical_cols)))
    table.add_row("Plots Created", str(len(plots_created)))
    
    if plots_created:
        file_types = {}
        for plot_path in plots_created:
            ext = plot_path.suffix[1:]  # Remove the dot
            file_types[ext] = file_types.get(ext, 0) + 1
        
        formats = ", ".join([f"{ext}: {count}" for ext, count in file_types.items()])
        table.add_row("File Formats", formats)
    
    console.print(table)
    
    # Show missing values summary
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    
    if not missing_data.empty:
        console.print("\n[yellow]⚠️  Missing Values Detected:[/yellow]")
        for col, count in missing_data.items():
            pct = (count / len(df)) * 100
            console.print(f"  {col}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    app()