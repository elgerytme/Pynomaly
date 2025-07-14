"""
Statistical Analysis CLI Commands

Provides comprehensive statistical analysis capabilities through command-line interface.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import json
import csv

from ....shared.error_handling import handle_cli_errors
from ....shared.logging import get_logger
from ....infrastructure.config import get_cli_container

logger = get_logger(__name__)
console = Console()

# Create the stats command group
app = typer.Typer(
    name="stats",
    help="📊 Statistical analysis and data exploration tools",
    rich_markup_mode="rich"
)


@app.command("describe")
@handle_cli_errors
def descriptive_statistics(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv"),
    features: Optional[str] = typer.Option(None, "--features", help="Comma-separated feature names"),
    include_percentiles: bool = typer.Option(True, "--percentiles/--no-percentiles", help="Include percentile calculations"),
    detect_outliers: bool = typer.Option(True, "--outliers/--no-outliers", help="Detect outliers"),
    missing_analysis: bool = typer.Option(True, "--missing/--no-missing", help="Analyze missing values")
):
    """
    Generate comprehensive descriptive statistics for dataset features.
    
    Calculates measures of central tendency, dispersion, and distribution shape
    including mean, median, mode, standard deviation, variance, skewness, kurtosis,
    and percentile distributions.
    
    Examples:
        pynomaly stats describe data.csv
        pynomaly stats describe data.csv --output stats.json --format json
        pynomaly stats describe data.csv --features "age,income,score"
    """
    console.print(f"[bold blue]📊 Generating descriptive statistics for {data_path}[/bold blue]")
    
    try:
        container = get_cli_container()
        
        # Load and validate dataset
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading dataset...", total=None)
            
            # Load dataset (implementation would use actual data loading service)
            import pandas as pd
            
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
            
            progress.update(task, description="Analyzing features...")
            
            # Filter features if specified
            if features:
                feature_list = [f.strip() for f in features.split(",")]
                missing_features = [f for f in feature_list if f not in df.columns]
                if missing_features:
                    console.print(f"[yellow]⚠️  Missing features: {missing_features}[/yellow]")
                available_features = [f for f in feature_list if f in df.columns]
                if not available_features:
                    console.print("[red]❌ No valid features found[/red]")
                    raise typer.Exit(1)
                df = df[available_features]
            
            # Generate descriptive statistics
            progress.update(task, description="Computing statistics...")
            
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.empty:
                console.print("[yellow]⚠️  No numeric features found for statistical analysis[/yellow]")
                raise typer.Exit(1)
            
            # Basic descriptive statistics
            desc_stats = numeric_df.describe()
            
            # Additional statistics
            stats_dict = {}
            for column in numeric_df.columns:
                col_stats = {
                    'count': float(desc_stats.loc['count', column]),
                    'mean': float(desc_stats.loc['mean', column]),
                    'std': float(desc_stats.loc['std', column]),
                    'min': float(desc_stats.loc['min', column]),
                    'max': float(desc_stats.loc['max', column]),
                    'median': float(desc_stats.loc['50%', column]),
                    'q1': float(desc_stats.loc['25%', column]),
                    'q3': float(desc_stats.loc['75%', column]),
                }
                
                # Additional metrics
                col_data = numeric_df[column].dropna()
                if len(col_data) > 0:
                    stats_dict[column] = {
                        **col_stats,
                        'skewness': float(col_data.skew()),
                        'kurtosis': float(col_data.kurtosis()),
                        'variance': float(col_data.var()),
                        'range': float(col_data.max() - col_data.min()),
                        'iqr': float(desc_stats.loc['75%', column] - desc_stats.loc['25%', column]),
                        'missing_count': int(df[column].isna().sum()),
                        'missing_percentage': float(df[column].isna().sum() / len(df) * 100),
                    }
                    
                    # Outlier detection using IQR method
                    if detect_outliers:
                        q1, q3 = stats_dict[column]['q1'], stats_dict[column]['q3']
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                        stats_dict[column]['outlier_count'] = len(outliers)
                        stats_dict[column]['outlier_percentage'] = float(len(outliers) / len(col_data) * 100)
            
            progress.update(task, description="Formatting output...", completed=True)
        
        # Display results
        if format == "table":
            _display_stats_table(stats_dict, include_percentiles, detect_outliers, missing_analysis)
        elif format == "json":
            if output:
                with open(output, 'w') as f:
                    json.dump(stats_dict, f, indent=2)
                console.print(f"[green]✅ Statistics saved to {output}[/green]")
            else:
                console.print(json.dumps(stats_dict, indent=2))
        elif format == "csv":
            _export_stats_csv(stats_dict, output)
        
        console.print("[green]✅ Descriptive statistics analysis completed[/green]")
        
    except Exception as e:
        logger.error(f"Error in descriptive statistics: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("correlate")
@handle_cli_errors
def correlation_analysis(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    method: str = typer.Option("pearson", "--method", "-m", help="Correlation method: pearson, spearman, kendall"),
    features: Optional[str] = typer.Option(None, "--features", help="Comma-separated feature names"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Correlation threshold for highlighting"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv"),
    show_significance: bool = typer.Option(True, "--significance/--no-significance", help="Show statistical significance")
):
    """
    Perform correlation analysis between dataset features.
    
    Supports multiple correlation methods including Pearson, Spearman, and Kendall
    correlations with statistical significance testing.
    
    Examples:
        pynomaly stats correlate data.csv
        pynomaly stats correlate data.csv --method spearman --threshold 0.7
        pynomaly stats correlate data.csv --features "age,income,score" --output corr.json
    """
    console.print(f"[bold blue]🔗 Analyzing correlations in {data_path}[/bold blue]")
    
    try:
        # Load and process dataset
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading dataset...", total=None)
            
            import pandas as pd
            import numpy as np
            from scipy.stats import pearsonr, spearmanr, kendalltau
            
            if not data_path.exists():
                console.print(f"[red]❌ Dataset file not found: {data_path}[/red]")
                raise typer.Exit(1)
            
            # Load dataset
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix.lower() in ['.json']:
                df = pd.read_json(data_path)
            elif data_path.suffix.lower() in ['.parquet']:
                df = pd.read_parquet(data_path)
            else:
                console.print(f"[red]❌ Unsupported file format: {data_path.suffix}[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description="Processing features...")
            
            # Filter features if specified
            if features:
                feature_list = [f.strip() for f in features.split(",")]
                missing_features = [f for f in feature_list if f not in df.columns]
                if missing_features:
                    console.print(f"[yellow]⚠️  Missing features: {missing_features}[/yellow]")
                available_features = [f for f in feature_list if f in df.columns]
                if len(available_features) < 2:
                    console.print("[red]❌ Need at least 2 valid features for correlation analysis[/red]")
                    raise typer.Exit(1)
                df = df[available_features]
            
            # Select numeric features
            numeric_df = df.select_dtypes(include=['number'])
            if len(numeric_df.columns) < 2:
                console.print("[red]❌ Need at least 2 numeric features for correlation analysis[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description=f"Computing {method} correlations...")
            
            # Calculate correlation matrix
            if method == "pearson":
                corr_matrix = numeric_df.corr(method='pearson')
            elif method == "spearman":
                corr_matrix = numeric_df.corr(method='spearman')
            elif method == "kendall":
                corr_matrix = numeric_df.corr(method='kendall')
            else:
                console.print(f"[red]❌ Unsupported correlation method: {method}[/red]")
                raise typer.Exit(1)
            
            # Calculate significance if requested
            significance_matrix = None
            if show_significance:
                progress.update(task, description="Computing significance tests...")
                significance_matrix = _calculate_correlation_significance(numeric_df, method)
            
            progress.update(task, description="Formatting results...", completed=True)
        
        # Display results
        if format == "table":
            _display_correlation_table(corr_matrix, significance_matrix, threshold, method)
        elif format == "json":
            result = {
                "correlation_matrix": corr_matrix.to_dict(),
                "method": method,
                "threshold": threshold
            }
            if significance_matrix is not None:
                result["significance_matrix"] = significance_matrix.to_dict()
            
            if output:
                with open(output, 'w') as f:
                    json.dump(result, f, indent=2)
                console.print(f"[green]✅ Correlation analysis saved to {output}[/green]")
            else:
                console.print(json.dumps(result, indent=2))
        elif format == "csv":
            if output:
                corr_matrix.to_csv(output)
                console.print(f"[green]✅ Correlation matrix saved to {output}[/green]")
            else:
                console.print(corr_matrix.to_string())
        
        # Show insights
        _show_correlation_insights(corr_matrix, threshold, method)
        
        console.print("[green]✅ Correlation analysis completed[/green]")
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("distribution")
@handle_cli_errors
def distribution_analysis(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    feature: str = typer.Argument(..., help="Feature name to analyze"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    distributions: str = typer.Option("normal,exponential,gamma", "--distributions", "-d", 
                                    help="Comma-separated distribution types to test"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json"),
    significance_level: float = typer.Option(0.05, "--alpha", help="Significance level for tests"),
    plot: bool = typer.Option(False, "--plot", help="Generate distribution plots")
):
    """
    Analyze statistical distribution of a feature with goodness-of-fit testing.
    
    Tests multiple distribution types and provides parameter estimation with
    confidence intervals and goodness-of-fit statistics.
    
    Examples:
        pynomaly stats distribution data.csv age
        pynomaly stats distribution data.csv income --distributions "normal,lognormal,gamma"
        pynomaly stats distribution data.csv score --plot --output dist_analysis.json
    """
    console.print(f"[bold blue]📈 Analyzing distribution of '{feature}' in {data_path}[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading data...", total=None)
            
            import pandas as pd
            import numpy as np
            from scipy import stats
            
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
            
            if feature not in df.columns:
                console.print(f"[red]❌ Feature '{feature}' not found in dataset[/red]")
                available_features = list(df.columns)
                console.print(f"Available features: {', '.join(available_features)}")
                raise typer.Exit(1)
            
            progress.update(task, description="Preparing data...")
            
            # Extract and clean feature data
            data = df[feature].dropna()
            if len(data) == 0:
                console.print(f"[red]❌ No valid data found for feature '{feature}'[/red]")
                raise typer.Exit(1)
            
            if not pd.api.types.is_numeric_dtype(data):
                console.print(f"[red]❌ Feature '{feature}' is not numeric[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description="Testing distributions...")
            
            # Parse distribution types
            dist_types = [d.strip() for d in distributions.split(",")]
            
            # Test distributions
            results = {}
            for dist_name in dist_types:
                try:
                    result = _test_distribution(data, dist_name, significance_level)
                    results[dist_name] = result
                except Exception as e:
                    console.print(f"[yellow]⚠️  Error testing {dist_name} distribution: {e}[/yellow]")
            
            progress.update(task, description="Analyzing results...", completed=True)
        
        if not results:
            console.print("[red]❌ No valid distribution tests completed[/red]")
            raise typer.Exit(1)
        
        # Display results
        if format == "table":
            _display_distribution_table(results, feature, significance_level)
        elif format == "json":
            result_data = {
                "feature": feature,
                "data_summary": {
                    "count": len(data),
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max())
                },
                "distribution_tests": results,
                "significance_level": significance_level
            }
            
            if output:
                with open(output, 'w') as f:
                    json.dump(result_data, f, indent=2)
                console.print(f"[green]✅ Distribution analysis saved to {output}[/green]")
            else:
                console.print(json.dumps(result_data, indent=2))
        
        # Find best fitting distribution
        best_dist = min(results.keys(), key=lambda k: results[k].get('p_value', 0))
        console.print(f"\n[bold green]🏆 Best fitting distribution: {best_dist}[/bold green]")
        
        console.print("[green]✅ Distribution analysis completed[/green]")
        
    except Exception as e:
        logger.error(f"Error in distribution analysis: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


# Helper functions

def _display_stats_table(stats_dict: Dict[str, Dict], include_percentiles: bool, 
                        detect_outliers: bool, missing_analysis: bool):
    """Display descriptive statistics in a rich table."""
    table = Table(title="📊 Descriptive Statistics", show_header=True, header_style="bold magenta")
    
    table.add_column("Feature", style="cyan", no_wrap=True)
    table.add_column("Count", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Median", justify="right")
    
    if include_percentiles:
        table.add_column("Q1", justify="right")
        table.add_column("Q3", justify="right")
    
    if detect_outliers:
        table.add_column("Outliers", justify="right")
    
    if missing_analysis:
        table.add_column("Missing %", justify="right")
    
    for feature, stats in stats_dict.items():
        row = [
            feature,
            f"{stats['count']:.0f}",
            f"{stats['mean']:.3f}",
            f"{stats['std']:.3f}",
            f"{stats['min']:.3f}",
            f"{stats['max']:.3f}",
            f"{stats['median']:.3f}"
        ]
        
        if include_percentiles:
            row.extend([f"{stats['q1']:.3f}", f"{stats['q3']:.3f}"])
        
        if detect_outliers:
            outlier_pct = stats.get('outlier_percentage', 0)
            color = "red" if outlier_pct > 5 else "yellow" if outlier_pct > 2 else "green"
            row.append(f"[{color}]{outlier_pct:.1f}%[/{color}]")
        
        if missing_analysis:
            missing_pct = stats.get('missing_percentage', 0)
            color = "red" if missing_pct > 10 else "yellow" if missing_pct > 5 else "green"
            row.append(f"[{color}]{missing_pct:.1f}%[/{color}]")
        
        table.add_row(*row)
    
    console.print(table)


def _display_correlation_table(corr_matrix, significance_matrix, threshold: float, method: str):
    """Display correlation matrix in a rich table."""
    table = Table(title=f"🔗 {method.title()} Correlation Matrix", show_header=True, header_style="bold magenta")
    
    # Add columns
    table.add_column("Feature", style="cyan", no_wrap=True)
    for col in corr_matrix.columns:
        table.add_column(col, justify="right")
    
    # Add rows
    for idx, row in corr_matrix.iterrows():
        row_data = [idx]
        for col in corr_matrix.columns:
            corr_val = row[col]
            
            # Color code based on correlation strength
            if abs(corr_val) >= threshold:
                if corr_val > 0:
                    color = "green"
                else:
                    color = "red"
            else:
                color = "white"
            
            # Add significance indicator if available
            if significance_matrix is not None:
                p_val = significance_matrix.loc[idx, col]
                sig_marker = "*" if p_val < 0.05 else ""
                row_data.append(f"[{color}]{corr_val:.3f}{sig_marker}[/{color}]")
            else:
                row_data.append(f"[{color}]{corr_val:.3f}[/{color}]")
        
        table.add_row(*row_data)
    
    console.print(table)
    
    if significance_matrix is not None:
        console.print("[dim]* indicates statistical significance (p < 0.05)[/dim]")


def _display_distribution_table(results: Dict, feature: str, alpha: float):
    """Display distribution test results in a rich table."""
    table = Table(title=f"📈 Distribution Analysis for '{feature}'", show_header=True, header_style="bold magenta")
    
    table.add_column("Distribution", style="cyan")
    table.add_column("Test Statistic", justify="right")
    table.add_column("P-Value", justify="right")
    table.add_column("Result", justify="center")
    table.add_column("Parameters", justify="left")
    
    for dist_name, result in results.items():
        p_value = result.get('p_value', 0)
        test_stat = result.get('test_statistic', 0)
        parameters = result.get('parameters', {})
        
        # Determine result
        if p_value > alpha:
            result_text = "[green]✓ Fits[/green]"
        else:
            result_text = "[red]✗ Rejected[/red]"
        
        # Format parameters
        param_str = ", ".join([f"{k}={v:.3f}" for k, v in parameters.items()])
        
        table.add_row(
            dist_name.title(),
            f"{test_stat:.4f}",
            f"{p_value:.4f}",
            result_text,
            param_str
        )
    
    console.print(table)


def _calculate_correlation_significance(df, method: str):
    """Calculate statistical significance for correlations."""
    import pandas as pd
    import numpy as np
    from scipy.stats import pearsonr, spearmanr, kendalltau
    
    n_features = len(df.columns)
    p_matrix = pd.DataFrame(np.ones((n_features, n_features)), 
                           index=df.columns, columns=df.columns)
    
    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i != j:
                try:
                    if method == "pearson":
                        _, p_val = pearsonr(df[col1].dropna(), df[col2].dropna())
                    elif method == "spearman":
                        _, p_val = spearmanr(df[col1].dropna(), df[col2].dropna())
                    elif method == "kendall":
                        _, p_val = kendalltau(df[col1].dropna(), df[col2].dropna())
                    
                    p_matrix.loc[col1, col2] = p_val
                except:
                    p_matrix.loc[col1, col2] = 1.0
    
    return p_matrix


def _test_distribution(data, dist_name: str, alpha: float):
    """Test if data fits a specific distribution."""
    from scipy import stats
    import numpy as np
    
    # Distribution mapping
    distributions = {
        'normal': stats.norm,
        'exponential': stats.expon,
        'gamma': stats.gamma,
        'lognormal': stats.lognorm,
        'uniform': stats.uniform,
        'beta': stats.beta
    }
    
    if dist_name not in distributions:
        raise ValueError(f"Unsupported distribution: {dist_name}")
    
    dist = distributions[dist_name]
    
    # Fit distribution parameters
    if dist_name == 'uniform':
        params = (data.min(), data.max() - data.min())
    elif dist_name == 'exponential':
        params = (0, 1/data.mean())
    else:
        params = dist.fit(data)
    
    # Perform Kolmogorov-Smirnov test
    ks_stat, p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))
    
    # Format parameters
    param_names = ['loc', 'scale'] if len(params) == 2 else ['shape', 'loc', 'scale']
    if dist_name == 'uniform':
        param_names = ['min', 'range']
    elif dist_name == 'exponential':
        param_names = ['loc', 'scale']
    
    param_dict = dict(zip(param_names[:len(params)], params))
    
    return {
        'test_statistic': ks_stat,
        'p_value': p_value,
        'parameters': param_dict,
        'fitted_params': params
    }


def _show_correlation_insights(corr_matrix, threshold: float, method: str):
    """Display correlation insights."""
    import numpy as np
    
    # Find strong correlations
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                strong_corrs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    if strong_corrs:
        console.print(f"\n[bold yellow]🔍 Strong Correlations (|r| >= {threshold}):[/bold yellow]")
        for feat1, feat2, corr in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True):
            direction = "positive" if corr > 0 else "negative"
            color = "green" if corr > 0 else "red"
            console.print(f"  [{color}]{feat1} ↔ {feat2}: {corr:.3f}[/{color}] ({direction})")
    else:
        console.print(f"\n[dim]No strong correlations found (threshold: {threshold})[/dim]")


def _export_stats_csv(stats_dict: Dict[str, Dict], output: Optional[Path]):
    """Export statistics to CSV format."""
    if not output:
        console.print("[yellow]⚠️  No output file specified for CSV export[/yellow]")
        return
    
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(stats_dict, orient='index')
    df.to_csv(output)
    console.print(f"[green]✅ Statistics exported to {output}[/green]")


if __name__ == "__main__":
    app()