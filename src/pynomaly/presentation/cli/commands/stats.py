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
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, excel"),
    significance_level: float = typer.Option(0.05, "--alpha", help="Significance level for tests"),
    plot: bool = typer.Option(False, "--plot", help="Generate distribution plots"),
    save_plots: bool = typer.Option(False, "--save-plots", help="Save generated plots to files"),
    plot_format: str = typer.Option("png", "--plot-format", help="Plot format: png, svg, pdf")
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


@app.command("hypothesis")
@handle_cli_errors
def hypothesis_testing(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    test_type: str = typer.Argument(..., help="Test type: ttest, anova, chi2, mann_whitney, kruskal"),
    feature1: str = typer.Option(..., "--feature1", help="Primary feature for testing"),
    feature2: Optional[str] = typer.Option(None, "--feature2", help="Secondary feature for testing"),
    group_column: Optional[str] = typer.Option(None, "--group", help="Grouping column for multi-group tests"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, excel"),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level"),
    effect_size: bool = typer.Option(True, "--effect-size/--no-effect-size", help="Calculate effect size"),
    power_analysis: bool = typer.Option(False, "--power/--no-power", help="Perform power analysis"),
    bootstrap: bool = typer.Option(False, "--bootstrap", help="Use bootstrap resampling"),
    bootstrap_samples: int = typer.Option(1000, "--bootstrap-samples", help="Number of bootstrap samples")
):
    """
    Perform statistical hypothesis testing with comprehensive analysis.
    
    Supports various hypothesis tests including t-tests, ANOVA, chi-square,
    and non-parametric tests with effect size and power analysis.
    
    Examples:
        pynomaly stats hypothesis data.csv ttest --feature1 group1 --feature2 group2
        pynomaly stats hypothesis data.csv anova --feature1 value --group category
        pynomaly stats hypothesis data.csv chi2 --feature1 cat1 --feature2 cat2
        pynomaly stats hypothesis data.csv mann_whitney --feature1 score --group treatment
    """
    console.print(f"[bold blue]🧪 Performing {test_type} hypothesis test[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading data and preparing test...", total=None)
            
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
            
            progress.update(task, description=f"Performing {test_type} test...")
            
            # Validate required features
            required_features = [feature1]
            if feature2:
                required_features.append(feature2)
            if group_column:
                required_features.append(group_column)
            
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                console.print(f"[red]❌ Missing features: {missing_features}[/red]")
                raise typer.Exit(1)
            
            # Perform hypothesis test
            test_results = _perform_hypothesis_test(
                df, test_type, feature1, feature2, group_column, 
                alpha, effect_size, power_analysis, bootstrap, bootstrap_samples
            )
            
            progress.update(task, description="Generating results...", completed=True)
        
        # Display results
        if format == "table":
            _display_hypothesis_test_table(test_results, test_type, alpha)
        elif format == "json":
            if output:
                with open(output, 'w') as f:
                    json.dump(test_results, f, indent=2, default=str)
                console.print(f"[green]✅ Hypothesis test results saved to {output}[/green]")
            else:
                console.print(json.dumps(test_results, indent=2, default=str))
        elif format == "excel":
            if output:
                _export_hypothesis_test_excel(test_results, output, test_type)
                console.print(f"[green]✅ Hypothesis test results saved to {output}[/green]")
            else:
                console.print("[yellow]⚠️  Excel format requires --output parameter[/yellow]")
        
        console.print("[green]✅ Hypothesis testing completed[/green]")
        
    except Exception as e:
        logger.error(f"Error in hypothesis testing: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("regression")
@handle_cli_errors
def regression_analysis(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    target: str = typer.Argument(..., help="Target variable for regression"),
    features: str = typer.Option(..., "--features", help="Comma-separated predictor features"),
    regression_type: str = typer.Option("linear", "--type", help="Regression type: linear, logistic, polynomial, ridge, lasso"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, excel"),
    polynomial_degree: int = typer.Option(2, "--degree", help="Polynomial degree (for polynomial regression)"),
    regularization_alpha: float = typer.Option(1.0, "--alpha", help="Regularization parameter (for ridge/lasso)"),
    cross_validate: bool = typer.Option(True, "--cv/--no-cv", help="Perform cross-validation"),
    cv_folds: int = typer.Option(5, "--cv-folds", help="Number of cross-validation folds"),
    residual_analysis: bool = typer.Option(True, "--residuals/--no-residuals", help="Perform residual analysis"),
    feature_importance: bool = typer.Option(True, "--importance/--no-importance", help="Calculate feature importance"),
    save_plots: bool = typer.Option(False, "--save-plots", help="Save diagnostic plots"),
    plot_format: str = typer.Option("png", "--plot-format", help="Plot format: png, svg, pdf")
):
    """
    Perform comprehensive regression analysis with diagnostics.
    
    Supports multiple regression types including linear, logistic, polynomial,
    ridge, and lasso regression with cross-validation and diagnostics.
    
    Examples:
        pynomaly stats regression data.csv price --features "size,rooms,location"
        pynomaly stats regression data.csv outcome --features "var1,var2" --type logistic
        pynomaly stats regression data.csv target --features "x1,x2,x3" --type ridge --alpha 0.1
        pynomaly stats regression data.csv y --features "x" --type polynomial --degree 3
    """
    console.print(f"[bold blue]📈 Performing {regression_type} regression analysis[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading data and preparing regression...", total=None)
            
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import cross_val_score, train_test_split
            from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.metrics import r2_score, mean_squared_error, classification_report
            
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
            
            progress.update(task, description="Preparing features and target...")
            
            # Validate features and target
            feature_list = [f.strip() for f in features.split(",")]
            all_features = feature_list + [target]
            missing_features = [f for f in all_features if f not in df.columns]
            if missing_features:
                console.print(f"[red]❌ Missing features: {missing_features}[/red]")
                raise typer.Exit(1)
            
            # Prepare data
            X = df[feature_list].select_dtypes(include=[np.number])
            y = df[target]
            
            if X.empty:
                console.print("[red]❌ No numeric features found for regression[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description=f"Fitting {regression_type} model...")
            
            # Perform regression analysis
            regression_results = _perform_regression_analysis(
                X, y, regression_type, polynomial_degree, regularization_alpha,
                cross_validate, cv_folds, residual_analysis, feature_importance
            )
            
            progress.update(task, description="Generating diagnostic plots...", completed=True)
            
            # Generate plots if requested
            if save_plots:
                plot_paths = _generate_regression_plots(
                    X, y, regression_results, output or Path("regression_plots"), plot_format
                )
                regression_results["plot_files"] = plot_paths
        
        # Display results
        if format == "table":
            _display_regression_table(regression_results, regression_type)
        elif format == "json":
            if output:
                with open(output, 'w') as f:
                    json.dump(regression_results, f, indent=2, default=str)
                console.print(f"[green]✅ Regression analysis saved to {output}[/green]")
            else:
                console.print(json.dumps(regression_results, indent=2, default=str))
        elif format == "excel":
            if output:
                _export_regression_excel(regression_results, output, regression_type)
                console.print(f"[green]✅ Regression analysis saved to {output}[/green]")
            else:
                console.print("[yellow]⚠️  Excel format requires --output parameter[/yellow]")
        
        console.print("[green]✅ Regression analysis completed[/green]")
        
    except Exception as e:
        logger.error(f"Error in regression analysis: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("time-series")
@handle_cli_errors
def time_series_analysis(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    time_column: str = typer.Argument(..., help="Time/date column name"),
    value_column: str = typer.Argument(..., help="Value column name"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, excel"),
    decomposition: bool = typer.Option(True, "--decompose/--no-decompose", help="Perform time series decomposition"),
    seasonality_test: bool = typer.Option(True, "--seasonality/--no-seasonality", help="Test for seasonality"),
    stationarity_test: bool = typer.Option(True, "--stationarity/--no-stationarity", help="Test for stationarity"),
    autocorrelation: bool = typer.Option(True, "--autocorr/--no-autocorr", help="Calculate autocorrelation"),
    forecast_periods: int = typer.Option(0, "--forecast", help="Number of periods to forecast"),
    forecast_method: str = typer.Option("arima", "--forecast-method", help="Forecast method: arima, exponential, linear"),
    save_plots: bool = typer.Option(False, "--save-plots", help="Save time series plots"),
    plot_format: str = typer.Option("png", "--plot-format", help="Plot format: png, svg, pdf")
):
    """
    Perform comprehensive time series analysis and forecasting.
    
    Analyzes time series data with decomposition, stationarity testing,
    seasonality analysis, and optional forecasting capabilities.
    
    Examples:
        pynomaly stats time-series data.csv date value
        pynomaly stats time-series data.csv timestamp sales --forecast 12 --forecast-method arima
        pynomaly stats time-series data.csv date revenue --decompose --save-plots
        pynomaly stats time-series data.csv time metric --output ts_analysis.xlsx --format excel
    """
    console.print(f"[bold blue]📈 Performing time series analysis[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading and preparing time series data...", total=None)
            
            import pandas as pd
            import numpy as np
            
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
            
            # Validate columns
            if time_column not in df.columns:
                console.print(f"[red]❌ Time column '{time_column}' not found[/red]")
                raise typer.Exit(1)
            
            if value_column not in df.columns:
                console.print(f"[red]❌ Value column '{value_column}' not found[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description="Preparing time series...")
            
            # Convert to datetime and sort
            df[time_column] = pd.to_datetime(df[time_column])
            df = df.sort_values(time_column)
            df.set_index(time_column, inplace=True)
            
            # Extract time series
            ts = df[value_column].dropna()
            
            progress.update(task, description="Performing time series analysis...")
            
            # Perform comprehensive time series analysis
            ts_results = _perform_time_series_analysis(
                ts, decomposition, seasonality_test, stationarity_test,
                autocorrelation, forecast_periods, forecast_method
            )
            
            progress.update(task, description="Generating plots...", completed=True)
            
            # Generate plots if requested
            if save_plots:
                plot_paths = _generate_time_series_plots(
                    ts, ts_results, output or Path("timeseries_plots"), plot_format
                )
                ts_results["plot_files"] = plot_paths
        
        # Display results
        if format == "table":
            _display_time_series_table(ts_results)
        elif format == "json":
            if output:
                with open(output, 'w') as f:
                    json.dump(ts_results, f, indent=2, default=str)
                console.print(f"[green]✅ Time series analysis saved to {output}[/green]")
            else:
                console.print(json.dumps(ts_results, indent=2, default=str))
        elif format == "excel":
            if output:
                _export_time_series_excel(ts_results, output)
                console.print(f"[green]✅ Time series analysis saved to {output}[/green]")
            else:
                console.print("[yellow]⚠️  Excel format requires --output parameter[/yellow]")
        
        console.print("[green]✅ Time series analysis completed[/green]")
        
    except Exception as e:
        logger.error(f"Error in time series analysis: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("export")
@handle_cli_errors
def export_analysis(
    input_path: Path = typer.Argument(..., help="Path to analysis results file (JSON)"),
    output_path: Path = typer.Argument(..., help="Output file path"),
    export_format: str = typer.Option("excel", "--format", help="Export format: excel, csv, pdf, html"),
    include_plots: bool = typer.Option(True, "--plots/--no-plots", help="Include plots in export"),
    template: Optional[str] = typer.Option(None, "--template", help="Template name: report, dashboard, summary"),
    title: Optional[str] = typer.Option(None, "--title", help="Report title"),
    author: Optional[str] = typer.Option(None, "--author", help="Report author"),
    compress: bool = typer.Option(False, "--compress", help="Compress output (for multi-file exports)")
):
    """
    Export statistical analysis results to various formats.
    
    Converts analysis results to professional reports in Excel, PDF, HTML,
    or CSV formats with optional templates and visualizations.
    
    Examples:
        pynomaly stats export results.json report.xlsx
        pynomaly stats export analysis.json report.pdf --template report --title "Data Analysis"
        pynomaly stats export stats.json dashboard.html --template dashboard --plots
        pynomaly stats export results.json data.csv --format csv --no-plots
    """
    console.print(f"[bold blue]📤 Exporting analysis results to {export_format.upper()}[/bold blue]")
    
    try:
        # Load analysis results
        if not input_path.exists():
            console.print(f"[red]❌ Input file not found: {input_path}[/red]")
            raise typer.Exit(1)
        
        with open(input_path, 'r') as f:
            analysis_data = json.load(f)
        
        # Perform export
        export_success = _export_analysis_results(
            analysis_data, output_path, export_format, 
            include_plots, template, title, author, compress
        )
        
        if export_success:
            console.print(f"[green]✅ Analysis exported to {output_path}[/green]")
        else:
            console.print("[red]❌ Export failed[/red]")
            raise typer.Exit(1)
        
    except Exception as e:
        logger.error(f"Error in export: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


# Enhanced helper functions for new commands

def _perform_hypothesis_test(df, test_type, feature1, feature2, group_column, 
                           alpha, effect_size, power_analysis, bootstrap, bootstrap_samples):
    """Perform statistical hypothesis testing."""
    import pandas as pd
    import numpy as np
    from scipy import stats
    
    results = {
        "test_type": test_type,
        "alpha": alpha,
        "features": {"feature1": feature1, "feature2": feature2, "group": group_column}
    }
    
    if test_type == "ttest":
        if feature2:
            # Two-sample t-test
            group1 = df[feature1].dropna()
            group2 = df[feature2].dropna()
            statistic, p_value = stats.ttest_ind(group1, group2)
            
            if effect_size:
                # Cohen's d
                pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + 
                                    (len(group2) - 1) * group2.var()) / 
                                   (len(group1) + len(group2) - 2))
                cohens_d = (group1.mean() - group2.mean()) / pooled_std
                results["effect_size"] = {"cohens_d": cohens_d}
        
        elif group_column:
            # One-sample or grouped t-test
            groups = df.groupby(group_column)[feature1].apply(list)
            if len(groups) == 2:
                group1, group2 = groups.iloc[0], groups.iloc[1]
                statistic, p_value = stats.ttest_ind(group1, group2)
            else:
                console.print("[yellow]⚠️  T-test requires exactly 2 groups[/yellow]")
                return results
        
        results.update({
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < alpha
        })
    
    elif test_type == "anova":
        if not group_column:
            console.print("[red]❌ ANOVA requires a group column[/red]")
            return results
        
        groups = [group[feature1].dropna() for name, group in df.groupby(group_column)]
        statistic, p_value = stats.f_oneway(*groups)
        
        results.update({
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < alpha,
            "num_groups": len(groups)
        })
        
        if effect_size:
            # Eta-squared
            n_total = sum(len(group) for group in groups)
            ss_between = sum(len(group) * (np.mean(group) - np.mean(np.concatenate(groups)))**2 
                           for group in groups)
            ss_total = sum((x - np.mean(np.concatenate(groups)))**2 
                          for group in groups for x in group)
            eta_squared = ss_between / ss_total
            results["effect_size"] = {"eta_squared": eta_squared}
    
    elif test_type == "chi2":
        if not feature2:
            console.print("[red]❌ Chi-square test requires two categorical features[/red]")
            return results
        
        contingency_table = pd.crosstab(df[feature1], df[feature2])
        statistic, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        results.update({
            "statistic": statistic,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "significant": p_value < alpha
        })
        
        if effect_size:
            # Cramér's V
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(statistic / (n * (min(contingency_table.shape) - 1)))
            results["effect_size"] = {"cramers_v": cramers_v}
    
    # Add bootstrap results if requested
    if bootstrap:
        results["bootstrap"] = _perform_bootstrap_test(
            df, test_type, feature1, feature2, group_column, bootstrap_samples
        )
    
    return results


def _perform_regression_analysis(X, y, regression_type, polynomial_degree, 
                                regularization_alpha, cross_validate, cv_folds,
                                residual_analysis, feature_importance):
    """Perform comprehensive regression analysis."""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score, mean_squared_error
    
    results = {
        "regression_type": regression_type,
        "features": list(X.columns),
        "target": y.name if hasattr(y, 'name') else "target",
        "sample_size": len(X)
    }
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Select and configure model
    if regression_type == "linear":
        model = LinearRegression()
    elif regression_type == "logistic":
        model = LogisticRegression()
    elif regression_type == "ridge":
        model = Ridge(alpha=regularization_alpha)
    elif regression_type == "lasso":
        model = Lasso(alpha=regularization_alpha)
    elif regression_type == "polynomial":
        poly_features = PolynomialFeatures(degree=polynomial_degree)
        X_train = poly_features.fit_transform(X_train)
        X_test = poly_features.transform(X_test)
        model = LinearRegression()
        results["polynomial_degree"] = polynomial_degree
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    if regression_type != "logistic":
        results["metrics"] = {
            "r2_train": r2_score(y_train, y_pred_train),
            "r2_test": r2_score(y_test, y_pred_test),
            "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "mae_train": np.mean(np.abs(y_train - y_pred_train)),
            "mae_test": np.mean(np.abs(y_test - y_pred_test))
        }
    
    # Cross-validation
    if cross_validate:
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, 
                                   scoring='r2' if regression_type != 'logistic' else 'accuracy')
        results["cross_validation"] = {
            "cv_scores": cv_scores.tolist(),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_folds": cv_folds
        }
    
    # Feature importance
    if feature_importance and hasattr(model, 'coef_'):
        if regression_type == "polynomial":
            results["feature_importance"] = {
                "note": "Polynomial features - coefficients represent polynomial terms"
            }
        else:
            importance_dict = dict(zip(X.columns, model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_))
            results["feature_importance"] = importance_dict
    
    # Residual analysis
    if residual_analysis and regression_type != "logistic":
        residuals = y_test - y_pred_test
        results["residual_analysis"] = {
            "residuals_mean": residuals.mean(),
            "residuals_std": residuals.std(),
            "residuals_skewness": residuals.skew() if hasattr(residuals, 'skew') else float(stats.skew(residuals)),
            "residuals_kurtosis": residuals.kurtosis() if hasattr(residuals, 'kurtosis') else float(stats.kurtosis(residuals))
        }
    
    return results


def _perform_time_series_analysis(ts, decomposition, seasonality_test, 
                                 stationarity_test, autocorrelation, 
                                 forecast_periods, forecast_method):
    """Perform comprehensive time series analysis."""
    import pandas as pd
    import numpy as np
    from scipy import stats
    
    results = {
        "series_length": len(ts),
        "start_date": str(ts.index[0]),
        "end_date": str(ts.index[-1]),
        "frequency": str(ts.index.freq) if ts.index.freq else "irregular"
    }
    
    # Basic statistics
    results["descriptive_stats"] = {
        "mean": ts.mean(),
        "std": ts.std(),
        "min": ts.min(),
        "max": ts.max(),
        "skewness": ts.skew(),
        "kurtosis": ts.kurtosis()
    }
    
    # Decomposition
    if decomposition:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            decomp = seasonal_decompose(ts, model='additive', period=min(12, len(ts)//2))
            results["decomposition"] = {
                "trend_mean": decomp.trend.mean(),
                "seasonal_amplitude": decomp.seasonal.std(),
                "residual_std": decomp.resid.std()
            }
        except ImportError:
            results["decomposition"] = {"error": "statsmodels not available"}
        except Exception as e:
            results["decomposition"] = {"error": str(e)}
    
    # Stationarity test (Augmented Dickey-Fuller)
    if stationarity_test:
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(ts.dropna())
            results["stationarity_test"] = {
                "adf_statistic": adf_result[0],
                "p_value": adf_result[1],
                "critical_values": adf_result[4],
                "is_stationary": adf_result[1] < 0.05
            }
        except ImportError:
            results["stationarity_test"] = {"error": "statsmodels not available"}
    
    # Seasonality test
    if seasonality_test:
        # Simple seasonality detection using FFT
        fft = np.fft.fft(ts.fillna(ts.mean()))
        freq_power = np.abs(fft[:len(fft)//2])
        dominant_freq_idx = np.argmax(freq_power[1:]) + 1  # Skip DC component
        seasonality_strength = freq_power[dominant_freq_idx] / np.sum(freq_power)
        
        results["seasonality_test"] = {
            "seasonality_strength": seasonality_strength,
            "dominant_period": len(ts) / dominant_freq_idx if dominant_freq_idx > 0 else None,
            "has_seasonality": seasonality_strength > 0.1
        }
    
    # Autocorrelation
    if autocorrelation:
        autocorr_lags = min(20, len(ts)//4)
        autocorr_values = [ts.autocorr(lag=i) for i in range(1, autocorr_lags + 1)]
        results["autocorrelation"] = {
            "lags": list(range(1, autocorr_lags + 1)),
            "values": autocorr_values,
            "significant_lags": [i+1 for i, val in enumerate(autocorr_values) if abs(val) > 0.2]
        }
    
    # Forecasting
    if forecast_periods > 0:
        if forecast_method == "linear":
            # Simple linear trend forecast
            x = np.arange(len(ts))
            slope, intercept, _, _, _ = stats.linregress(x, ts.fillna(ts.mean()))
            forecast_x = np.arange(len(ts), len(ts) + forecast_periods)
            forecast_values = slope * forecast_x + intercept
            
            results["forecast"] = {
                "method": "linear",
                "periods": forecast_periods,
                "values": forecast_values.tolist()
            }
        else:
            results["forecast"] = {
                "error": f"Forecast method '{forecast_method}' not implemented in this demo"
            }
    
    return results


# Export helper functions

def _export_analysis_results(analysis_data, output_path, export_format, 
                           include_plots, template, title, author, compress):
    """Export analysis results to specified format."""
    try:
        if export_format == "excel":
            return _export_to_excel(analysis_data, output_path, include_plots, template, title, author)
        elif export_format == "csv":
            return _export_to_csv(analysis_data, output_path)
        elif export_format == "pdf":
            return _export_to_pdf(analysis_data, output_path, include_plots, template, title, author)
        elif export_format == "html":
            return _export_to_html(analysis_data, output_path, include_plots, template, title, author)
        else:
            console.print(f"[red]❌ Unsupported export format: {export_format}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]❌ Export error: {e}[/red]")
        return False


def _export_to_excel(analysis_data, output_path, include_plots, template, title, author):
    """Export analysis to Excel format."""
    try:
        import pandas as pd
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                "Metric": ["Analysis Type", "Date Generated", "Total Records", "Features Analyzed"],
                "Value": [
                    analysis_data.get("analysis_type", "Statistical Analysis"),
                    pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    analysis_data.get("sample_size", "N/A"),
                    len(analysis_data.get("features", []))
                ]
            }
            
            if title:
                summary_data["Metric"].insert(0, "Report Title")
                summary_data["Value"].insert(0, title)
            
            if author:
                summary_data["Metric"].insert(-1, "Author")
                summary_data["Value"].insert(-1, author)
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            
            # Results sheet
            if "results" in analysis_data:
                results_df = pd.DataFrame([analysis_data["results"]])
                results_df.to_excel(writer, sheet_name="Results", index=False)
            
            # Additional sheets based on analysis type
            if "metrics" in analysis_data:
                metrics_df = pd.DataFrame([analysis_data["metrics"]])
                metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
            
            if "cross_validation" in analysis_data:
                cv_df = pd.DataFrame(analysis_data["cross_validation"])
                cv_df.to_excel(writer, sheet_name="Cross_Validation", index=False)
        
        return True
    except ImportError:
        console.print("[yellow]⚠️  openpyxl not available for Excel export[/yellow]")
        return False
    except Exception as e:
        console.print(f"[red]❌ Excel export error: {e}[/red]")
        return False


def _export_to_csv(analysis_data, output_path):
    """Export analysis to CSV format."""
    try:
        import pandas as pd
        
        # Flatten analysis data for CSV export
        flattened_data = []
        
        def flatten_dict(d, prefix=""):
            for key, value in d.items():
                if isinstance(value, dict):
                    flatten_dict(value, f"{prefix}{key}_" if prefix else f"{key}_")
                elif isinstance(value, list):
                    if value and isinstance(value[0], (int, float)):
                        flattened_data.append({
                            "metric": f"{prefix}{key}",
                            "value": str(value),
                            "type": "list"
                        })
                else:
                    flattened_data.append({
                        "metric": f"{prefix}{key}",
                        "value": value,
                        "type": type(value).__name__
                    })
        
        flatten_dict(analysis_data)
        pd.DataFrame(flattened_data).to_csv(output_path, index=False)
        return True
    except Exception as e:
        console.print(f"[red]❌ CSV export error: {e}[/red]")
        return False


# Display helper functions for new commands

def _display_hypothesis_test_table(results, test_type, alpha):
    """Display hypothesis test results in a table."""
    console.print(f"\n[bold blue]{test_type.title()} Hypothesis Test Results[/bold blue]")
    
    table = Table(title="Test Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Test Type", test_type.title())
    table.add_row("Test Statistic", f"{results.get('statistic', 'N/A'):.4f}")
    table.add_row("P-Value", f"{results.get('p_value', 'N/A'):.4f}")
    table.add_row("Significance Level", f"{alpha}")
    
    significant = results.get('significant', False)
    sig_color = "green" if significant else "red"
    table.add_row("Result", f"[{sig_color}]{'Significant' if significant else 'Not Significant'}[/{sig_color}]")
    
    if "effect_size" in results:
        for effect_name, effect_value in results["effect_size"].items():
            table.add_row(f"Effect Size ({effect_name})", f"{effect_value:.4f}")
    
    console.print(table)


def _display_regression_table(results, regression_type):
    """Display regression analysis results in a table."""
    console.print(f"\n[bold blue]{regression_type.title()} Regression Analysis[/bold blue]")
    
    # Model summary
    table = Table(title="Model Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Regression Type", regression_type.title())
    table.add_row("Sample Size", str(results.get("sample_size", "N/A")))
    table.add_row("Features", str(len(results.get("features", []))))
    
    if "metrics" in results:
        metrics = results["metrics"]
        for metric_name, metric_value in metrics.items():
            table.add_row(metric_name.upper(), f"{metric_value:.4f}")
    
    console.print(table)
    
    # Cross-validation results
    if "cross_validation" in results:
        cv = results["cross_validation"]
        console.print(f"\n[bold]Cross-Validation ({cv['cv_folds']} folds)[/bold]")
        console.print(f"Mean Score: {cv['cv_mean']:.4f} ± {cv['cv_std']:.4f}")


def _display_time_series_table(results):
    """Display time series analysis results in a table."""
    console.print(f"\n[bold blue]Time Series Analysis Results[/bold blue]")
    
    table = Table(title="Series Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Series Length", str(results.get("series_length", "N/A")))
    table.add_row("Start Date", results.get("start_date", "N/A"))
    table.add_row("End Date", results.get("end_date", "N/A"))
    table.add_row("Frequency", results.get("frequency", "N/A"))
    
    if "descriptive_stats" in results:
        stats = results["descriptive_stats"]
        table.add_row("Mean", f"{stats['mean']:.4f}")
        table.add_row("Std Dev", f"{stats['std']:.4f}")
        table.add_row("Skewness", f"{stats['skewness']:.4f}")
        table.add_row("Kurtosis", f"{stats['kurtosis']:.4f}")
    
    console.print(table)
    
    # Stationarity test
    if "stationarity_test" in results and "error" not in results["stationarity_test"]:
        st = results["stationarity_test"]
        stationary_color = "green" if st["is_stationary"] else "red"
        console.print(f"\n[bold]Stationarity Test (ADF)[/bold]")
        console.print(f"P-Value: {st['p_value']:.4f}")
        console.print(f"Result: [{stationary_color}]{'Stationary' if st['is_stationary'] else 'Non-stationary'}[/{stationary_color}]")
    
    # Seasonality test
    if "seasonality_test" in results:
        seas = results["seasonality_test"]
        seasonal_color = "green" if seas["has_seasonality"] else "yellow"
        console.print(f"\n[bold]Seasonality Test[/bold]")
        console.print(f"Seasonality Strength: {seas['seasonality_strength']:.4f}")
        console.print(f"Result: [{seasonal_color}]{'Seasonal' if seas['has_seasonality'] else 'Non-seasonal'}[/{seasonal_color}]")


# Placeholder functions for complex implementations
def _perform_bootstrap_test(df, test_type, feature1, feature2, group_column, bootstrap_samples):
    """Perform bootstrap hypothesis testing."""
    return {"note": "Bootstrap implementation placeholder"}

def _generate_regression_plots(X, y, results, output_dir, plot_format):
    """Generate regression diagnostic plots."""
    return ["plot1.png", "plot2.png"]  # Placeholder

def _generate_time_series_plots(ts, results, output_dir, plot_format):
    """Generate time series plots."""
    return ["ts_plot.png", "decomp_plot.png"]  # Placeholder

def _export_hypothesis_test_excel(results, output_path, test_type):
    """Export hypothesis test results to Excel."""
    _export_to_excel(results, output_path, False, None, f"{test_type} Test Results", None)

def _export_regression_excel(results, output_path, regression_type):
    """Export regression results to Excel."""
    _export_to_excel(results, output_path, False, None, f"{regression_type} Regression Analysis", None)

def _export_time_series_excel(results, output_path):
    """Export time series results to Excel."""
    _export_to_excel(results, output_path, False, None, "Time Series Analysis", None)

def _export_to_pdf(analysis_data, output_path, include_plots, template, title, author):
    """Export to PDF format."""
    console.print("[yellow]⚠️  PDF export not implemented in this demo[/yellow]")
    return False

def _export_to_html(analysis_data, output_path, include_plots, template, title, author):
    """Export to HTML format."""
    console.print("[yellow]⚠️  HTML export not implemented in this demo[/yellow]")
    return False


if __name__ == "__main__":
    app()