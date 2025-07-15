"""
Data Profiling CLI Commands

Provides comprehensive data profiling capabilities through command-line interface.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.columns import Columns
from rich.text import Text
import json
import csv

from ....shared.error_handling import handle_cli_errors
from ....shared.logging import get_logger
from ....infrastructure.config import get_cli_container

logger = get_logger(__name__)
console = Console()

# Create the profiling command group
app = typer.Typer(
    name="profile",
    help="🔍 Data profiling and schema discovery tools",
    rich_markup_mode="rich"
)


@app.command("discover")
@handle_cli_errors
def schema_discovery(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, yaml"),
    infer_types: bool = typer.Option(True, "--infer-types/--no-infer-types", help="Infer detailed data types"),
    detect_patterns: bool = typer.Option(True, "--patterns/--no-patterns", help="Detect data patterns"),
    sample_size: Optional[int] = typer.Option(None, "--sample", help="Sample size for analysis"),
    include_examples: bool = typer.Option(True, "--examples/--no-examples", help="Include example values"),
    include_stats: bool = typer.Option(True, "--stats/--no-stats", help="Include basic statistics")
):
    """
    Discover and analyze dataset schema with comprehensive data profiling.
    
    Automatically detects column types, patterns, constraints, and relationships
    to provide insights into data structure and quality.
    
    Examples:
        pynomaly profile discover data.csv
        pynomaly profile discover data.csv --output schema.json --format json
        pynomaly profile discover data.csv --sample 10000 --no-patterns
    """
    console.print(f"[bold blue]🔍 Discovering schema for {data_path}[/bold blue]")
    
    try:
        container = get_cli_container()
        
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
                    df = pd.read_csv(data_path, nrows=sample_size)
                elif data_path.suffix.lower() in ['.json']:
                    df = pd.read_json(data_path, lines=True)
                    if sample_size:
                        df = df.head(sample_size)
                elif data_path.suffix.lower() in ['.parquet']:
                    df = pd.read_parquet(data_path)
                    if sample_size:
                        df = df.head(sample_size)
                else:
                    console.print(f"[red]❌ Unsupported file format: {data_path.suffix}[/red]")
                    raise typer.Exit(1)
            except Exception as e:
                console.print(f"[red]❌ Error loading dataset: {e}[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description="Analyzing schema...")
            
            # Perform schema discovery
            schema_info = _discover_schema(df, infer_types, detect_patterns, include_examples, include_stats)
            
            progress.update(task, description="Formatting results...", completed=True)
        
        # Display results
        if format == "table":
            _display_schema_table(schema_info, data_path.name)
        elif format == "json":
            if output:
                with open(output, 'w') as f:
                    json.dump(schema_info, f, indent=2, default=str)
                console.print(f"[green]✅ Schema information saved to {output}[/green]")
            else:
                console.print(json.dumps(schema_info, indent=2, default=str))
        elif format == "yaml":
            try:
                import yaml
                yaml_output = yaml.dump(schema_info, default_flow_style=False, sort_keys=False)
                if output:
                    with open(output, 'w') as f:
                        f.write(yaml_output)
                    console.print(f"[green]✅ Schema information saved to {output}[/green]")
                else:
                    console.print(yaml_output)
            except ImportError:
                console.print("[red]❌ PyYAML not installed. Use 'pip install pyyaml' for YAML support[/red]")
                raise typer.Exit(1)
        
        console.print("[green]✅ Schema discovery completed[/green]")
        
    except Exception as e:
        logger.error(f"Error in schema discovery: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("quality")
@handle_cli_errors
def data_quality_assessment(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, html"),
    features: Optional[str] = typer.Option(None, "--features", help="Comma-separated feature names"),
    checks: str = typer.Option("all", "--checks", help="Quality checks: missing,duplicates,outliers,consistency,all"),
    threshold_missing: float = typer.Option(0.05, "--missing-threshold", help="Missing value threshold"),
    threshold_outliers: float = typer.Option(0.05, "--outlier-threshold", help="Outlier detection threshold"),
    generate_report: bool = typer.Option(True, "--report/--no-report", help="Generate detailed quality report")
):
    """
    Assess data quality with comprehensive checks and recommendations.
    
    Performs multiple quality assessments including missing values, duplicates,
    outliers, consistency checks, and data integrity validation.
    
    Examples:
        pynomaly profile quality data.csv
        pynomaly profile quality data.csv --checks missing,duplicates
        pynomaly profile quality data.csv --output quality_report.html --format html
    """
    console.print(f"[bold blue]🔍 Assessing data quality for {data_path}[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading dataset...", total=None)
            
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
            
            progress.update(task, description="Performing quality checks...")
            
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
            
            # Perform quality checks
            check_list = checks.split(",") if checks != "all" else ["missing", "duplicates", "outliers", "consistency"]
            quality_results = _perform_quality_checks(df, check_list, threshold_missing, threshold_outliers)
            
            progress.update(task, description="Generating quality report...", completed=True)
        
        # Display results
        if format == "table":
            _display_quality_table(quality_results, data_path.name)
        elif format == "json":
            if output:
                with open(output, 'w') as f:
                    json.dump(quality_results, f, indent=2, default=str)
                console.print(f"[green]✅ Quality assessment saved to {output}[/green]")
            else:
                console.print(json.dumps(quality_results, indent=2, default=str))
        elif format == "html":
            if output:
                _generate_quality_html_report(quality_results, output, data_path.name)
                console.print(f"[green]✅ Quality report saved to {output}[/green]")
            else:
                console.print("[yellow]⚠️  HTML format requires --output parameter[/yellow]")
        
        console.print("[green]✅ Data quality assessment completed[/green]")
        
    except Exception as e:
        logger.error(f"Error in data quality assessment: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("summary")
@handle_cli_errors
def data_summary(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json"),
    include_distribution: bool = typer.Option(True, "--distribution/--no-distribution", help="Include distribution analysis"),
    include_correlations: bool = typer.Option(True, "--correlations/--no-correlations", help="Include correlation analysis"),
    include_nulls: bool = typer.Option(True, "--nulls/--no-nulls", help="Include null value analysis"),
    sample_size: Optional[int] = typer.Option(None, "--sample", help="Sample size for analysis")
):
    """
    Generate comprehensive data summary with key statistics and insights.
    
    Provides an overview of dataset characteristics including size, types,
    distributions, correlations, and quality metrics.
    
    Examples:
        pynomaly profile summary data.csv
        pynomaly profile summary data.csv --output summary.json --format json
        pynomaly profile summary data.csv --sample 5000 --no-correlations
    """
    console.print(f"[bold blue]📊 Generating data summary for {data_path}[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading and analyzing dataset...", total=None)
            
            import pandas as pd
            import numpy as np
            
            # Load dataset
            if not data_path.exists():
                console.print(f"[red]❌ Dataset file not found: {data_path}[/red]")
                raise typer.Exit(1)
            
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path, nrows=sample_size)
            elif data_path.suffix.lower() in ['.json']:
                df = pd.read_json(data_path)
                if sample_size:
                    df = df.head(sample_size)
            elif data_path.suffix.lower() in ['.parquet']:
                df = pd.read_parquet(data_path)
                if sample_size:
                    df = df.head(sample_size)
            else:
                console.print(f"[red]❌ Unsupported file format: {data_path.suffix}[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description="Computing summary statistics...")
            
            # Generate comprehensive summary
            summary_results = _generate_data_summary(
                df, include_distribution, include_correlations, include_nulls
            )
            
            progress.update(task, description="Formatting results...", completed=True)
        
        # Display results
        if format == "table":
            _display_summary_table(summary_results, data_path.name)
        elif format == "json":
            if output:
                with open(output, 'w') as f:
                    json.dump(summary_results, f, indent=2, default=str)
                console.print(f"[green]✅ Data summary saved to {output}[/green]")
            else:
                console.print(json.dumps(summary_results, indent=2, default=str))
        
        console.print("[green]✅ Data summary completed[/green]")
        
    except Exception as e:
        logger.error(f"Error generating data summary: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("relationships")
@handle_cli_errors
def analyze_relationships(
    data_path: Path = typer.Argument(..., help="Path to dataset file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    method: str = typer.Option("all", "--method", help="Analysis method: correlation, association, dependency, all"),
    threshold: float = typer.Option(0.5, "--threshold", help="Relationship strength threshold"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, graph"),
    features: Optional[str] = typer.Option(None, "--features", help="Comma-separated feature names"),
    create_graph: bool = typer.Option(True, "--graph/--no-graph", help="Create relationship graph visualization")
):
    """
    Analyze relationships and dependencies between dataset features.
    
    Discovers correlations, associations, and dependencies using various
    statistical methods to understand feature interactions.
    
    Examples:
        pynomaly profile relationships data.csv
        pynomaly profile relationships data.csv --method correlation --threshold 0.7
        pynomaly profile relationships data.csv --output relationships.json --format json
    """
    console.print(f"[bold blue]🔗 Analyzing relationships in {data_path}[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading dataset...", total=None)
            
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
            
            progress.update(task, description="Analyzing relationships...")
            
            # Filter features if specified
            if features:
                feature_list = [f.strip() for f in features.split(",")]
                missing_features = [f for f in feature_list if f not in df.columns]
                if missing_features:
                    console.print(f"[yellow]⚠️  Missing features: {missing_features}[/yellow]")
                available_features = [f for f in feature_list if f in df.columns]
                if len(available_features) < 2:
                    console.print("[red]❌ Need at least 2 valid features for relationship analysis[/red]")
                    raise typer.Exit(1)
                df = df[available_features]
            
            # Perform relationship analysis
            method_list = method.split(",") if method != "all" else ["correlation", "association", "dependency"]
            relationship_results = _analyze_relationships(df, method_list, threshold)
            
            progress.update(task, description="Generating visualization...", completed=True)
        
        # Display results
        if format == "table":
            _display_relationships_table(relationship_results, data_path.name)
        elif format == "json":
            if output:
                with open(output, 'w') as f:
                    json.dump(relationship_results, f, indent=2, default=str)
                console.print(f"[green]✅ Relationship analysis saved to {output}[/green]")
            else:
                console.print(json.dumps(relationship_results, indent=2, default=str))
        elif format == "graph":
            if output:
                _create_relationship_graph(relationship_results, output, threshold)
                console.print(f"[green]✅ Relationship graph saved to {output}[/green]")
            else:
                console.print("[yellow]⚠️  Graph format requires --output parameter[/yellow]")
        
        console.print("[green]✅ Relationship analysis completed[/green]")
        
    except Exception as e:
        logger.error(f"Error in relationship analysis: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


# Helper functions

def _discover_schema(df: "pd.DataFrame", infer_types: bool, detect_patterns: bool, 
                    include_examples: bool, include_stats: bool) -> Dict[str, Any]:
    """Discover and analyze dataset schema."""
    import pandas as pd
    import numpy as np
    
    schema_info = {
        "dataset_info": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "estimated_size_mb": df.size * 8 / 1024 / 1024  # Rough estimate
        },
        "columns": {}
    }
    
    for column in df.columns:
        col_info = {
            "name": column,
            "data_type": str(df[column].dtype),
            "non_null_count": int(df[column].count()),
            "null_count": int(df[column].isnull().sum()),
            "null_percentage": float(df[column].isnull().sum() / len(df) * 100),
            "unique_count": int(df[column].nunique()),
            "unique_percentage": float(df[column].nunique() / len(df) * 100)
        }
        
        # Infer detailed types
        if infer_types:
            col_info["inferred_type"] = _infer_column_type(df[column])
        
        # Include examples
        if include_examples:
            valid_values = df[column].dropna()
            if len(valid_values) > 0:
                col_info["sample_values"] = valid_values.head(5).tolist()
        
        # Include basic statistics for numeric columns
        if include_stats and pd.api.types.is_numeric_dtype(df[column]):
            col_stats = df[column].describe()
            col_info["statistics"] = {
                "mean": float(col_stats["mean"]),
                "median": float(col_stats["50%"]),
                "std": float(col_stats["std"]),
                "min": float(col_stats["min"]),
                "max": float(col_stats["max"]),
                "q1": float(col_stats["25%"]),
                "q3": float(col_stats["75%"])
            }
        
        # Detect patterns
        if detect_patterns:
            col_info["patterns"] = _detect_column_patterns(df[column])
        
        schema_info["columns"][column] = col_info
    
    return schema_info


def _infer_column_type(series: "pd.Series") -> str:
    """Infer detailed column type beyond pandas dtype."""
    import pandas as pd
    import re
    
    # Check for common patterns in string columns
    if pd.api.types.is_object_dtype(series):
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            return "unknown"
        
        sample = non_null.head(100)
        
        # Email pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if sample.str.match(email_pattern).sum() / len(sample) > 0.8:
            return "email"
        
        # URL pattern
        url_pattern = r'^https?://'
        if sample.str.match(url_pattern).sum() / len(sample) > 0.8:
            return "url"
        
        # Phone pattern
        phone_pattern = r'^\+?[\d\s\-\(\)]{10,}$'
        if sample.str.match(phone_pattern).sum() / len(sample) > 0.8:
            return "phone"
        
        # Date pattern
        try:
            pd.to_datetime(sample.head(10))
            return "datetime_string"
        except:
            pass
        
        # Categorical (low cardinality)
        if series.nunique() / len(series) < 0.1:
            return "categorical"
        
        return "text"
    
    elif pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series):
            if series.nunique() < 10:
                return "ordinal"
            return "integer"
        return "float"
    
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    elif pd.api.types.is_bool_dtype(series):
        return "boolean"
    
    return str(series.dtype)


def _detect_column_patterns(series: "pd.Series") -> Dict[str, Any]:
    """Detect common patterns in column data."""
    import pandas as pd
    import re
    
    patterns = {
        "has_missing": series.isnull().any(),
        "all_unique": series.nunique() == len(series),
        "has_duplicates": series.duplicated().any(),
        "constant_value": series.nunique() <= 1
    }
    
    if pd.api.types.is_object_dtype(series):
        non_null = series.dropna().astype(str)
        if len(non_null) > 0:
            patterns.update({
                "min_length": int(non_null.str.len().min()),
                "max_length": int(non_null.str.len().max()),
                "avg_length": float(non_null.str.len().mean()),
                "has_leading_zeros": (non_null.str.match(r'^0+\d').sum() > 0),
                "has_special_chars": (non_null.str.contains(r'[^a-zA-Z0-9\s]').sum() > 0),
                "all_uppercase": (non_null.str.isupper().sum() == len(non_null)),
                "all_lowercase": (non_null.str.islower().sum() == len(non_null))
            })
    
    elif pd.api.types.is_numeric_dtype(series):
        patterns.update({
            "has_negative": (series < 0).any(),
            "has_zero": (series == 0).any(),
            "all_positive": (series > 0).all(),
            "has_decimals": not pd.api.types.is_integer_dtype(series) and (series % 1 != 0).any()
        })
    
    return patterns


def _perform_quality_checks(df: "pd.DataFrame", checks: List[str], 
                           missing_threshold: float, outlier_threshold: float) -> Dict[str, Any]:
    """Perform comprehensive data quality checks."""
    import pandas as pd
    import numpy as np
    from scipy import stats
    
    results = {
        "overview": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "checks_performed": checks
        },
        "detailed_results": {}
    }
    
    # Missing values check
    if "missing" in checks:
        missing_analysis = {}
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_pct = missing_count / len(df)
            missing_analysis[column] = {
                "missing_count": int(missing_count),
                "missing_percentage": float(missing_pct * 100),
                "severity": "high" if missing_pct > missing_threshold else "low"
            }
        
        results["detailed_results"]["missing_values"] = {
            "summary": {
                "columns_with_missing": sum(1 for col in missing_analysis.values() if col["missing_count"] > 0),
                "total_missing_values": sum(col["missing_count"] for col in missing_analysis.values()),
                "avg_missing_percentage": np.mean([col["missing_percentage"] for col in missing_analysis.values()])
            },
            "by_column": missing_analysis
        }
    
    # Duplicates check
    if "duplicates" in checks:
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = duplicate_rows / len(df) * 100
        
        # Check for duplicate columns
        duplicate_columns = []
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                if df[col1].equals(df[col2]):
                    duplicate_columns.append((col1, col2))
        
        results["detailed_results"]["duplicates"] = {
            "duplicate_rows": int(duplicate_rows),
            "duplicate_percentage": float(duplicate_percentage),
            "duplicate_columns": duplicate_columns,
            "severity": "high" if duplicate_percentage > 5 else "low"
        }
    
    # Outliers check
    if "outliers" in checks:
        outlier_analysis = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            col_data = df[column].dropna()
            if len(col_data) > 0:
                # IQR method
                q1, q3 = col_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers_iqr = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                outliers_iqr_pct = outliers_iqr / len(col_data) * 100
                
                # Z-score method
                z_scores = np.abs(stats.zscore(col_data))
                outliers_zscore = (z_scores > 3).sum()
                outliers_zscore_pct = outliers_zscore / len(col_data) * 100
                
                outlier_analysis[column] = {
                    "outliers_iqr": int(outliers_iqr),
                    "outliers_iqr_percentage": float(outliers_iqr_pct),
                    "outliers_zscore": int(outliers_zscore),
                    "outliers_zscore_percentage": float(outliers_zscore_pct),
                    "severity": "high" if max(outliers_iqr_pct, outliers_zscore_pct) > outlier_threshold * 100 else "low"
                }
        
        results["detailed_results"]["outliers"] = {
            "summary": {
                "columns_analyzed": len(outlier_analysis),
                "columns_with_outliers": sum(1 for col in outlier_analysis.values() 
                                           if col["outliers_iqr"] > 0 or col["outliers_zscore"] > 0)
            },
            "by_column": outlier_analysis
        }
    
    # Consistency check
    if "consistency" in checks:
        consistency_issues = []
        
        # Check for mixed data types in object columns
        for column in df.select_dtypes(include=['object']).columns:
            non_null = df[column].dropna()
            if len(non_null) > 0:
                # Check for mixed numeric/string
                try:
                    numeric_values = pd.to_numeric(non_null, errors='coerce').notna().sum()
                    if 0 < numeric_values < len(non_null):
                        consistency_issues.append({
                            "column": column,
                            "issue": "mixed_types",
                            "description": f"Column contains both numeric and text values"
                        })
                except:
                    pass
                
                # Check for inconsistent formatting
                if non_null.dtype == 'object':
                    sample = non_null.head(100)
                    if sample.str.len().std() > sample.str.len().mean() * 0.5:
                        consistency_issues.append({
                            "column": column,
                            "issue": "inconsistent_formatting",
                            "description": f"Column has highly variable string lengths"
                        })
        
        results["detailed_results"]["consistency"] = {
            "issues_found": len(consistency_issues),
            "issues": consistency_issues
        }
    
    return results


def _generate_data_summary(df: "pd.DataFrame", include_distribution: bool, 
                          include_correlations: bool, include_nulls: bool) -> Dict[str, Any]:
    """Generate comprehensive data summary."""
    import pandas as pd
    import numpy as np
    
    summary = {
        "basic_info": {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": len(df.select_dtypes(include=['datetime64']).columns)
        }
    }
    
    # Column types summary
    summary["column_types"] = {}
    for dtype in df.dtypes.value_counts().index:
        summary["column_types"][str(dtype)] = int(df.dtypes.value_counts()[dtype])
    
    # Null values summary
    if include_nulls:
        null_summary = {}
        for column in df.columns:
            null_count = df[column].isnull().sum()
            if null_count > 0:
                null_summary[column] = {
                    "count": int(null_count),
                    "percentage": float(null_count / len(df) * 100)
                }
        summary["null_values"] = null_summary
    
    # Distribution summary for numeric columns
    if include_distribution:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            distribution_summary = {}
            for column in numeric_cols:
                col_data = df[column].dropna()
                if len(col_data) > 0:
                    distribution_summary[column] = {
                        "mean": float(col_data.mean()),
                        "median": float(col_data.median()),
                        "std": float(col_data.std()),
                        "skewness": float(col_data.skew()),
                        "kurtosis": float(col_data.kurtosis()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "range": float(col_data.max() - col_data.min())
                    }
            summary["distributions"] = distribution_summary
    
    # Correlation summary
    if include_correlations:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # Strong correlation threshold
                        strong_correlations.append({
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": float(corr_val)
                        })
            
            summary["correlations"] = {
                "strong_correlations": strong_correlations,
                "correlation_matrix_shape": corr_matrix.shape
            }
    
    return summary


def _analyze_relationships(df: "pd.DataFrame", methods: List[str], threshold: float) -> Dict[str, Any]:
    """Analyze relationships between features."""
    import pandas as pd
    import numpy as np
    from scipy.stats import chi2_contingency, pearsonr, spearmanr
    
    results = {
        "methods_used": methods,
        "threshold": threshold,
        "relationships": {}
    }
    
    # Correlation analysis
    if "correlation" in methods:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlations = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    corr_pearson, p_pearson = pearsonr(df[col1].dropna(), df[col2].dropna())
                    corr_spearman, p_spearman = spearmanr(df[col1].dropna(), df[col2].dropna())
                    
                    if abs(corr_pearson) >= threshold or abs(corr_spearman) >= threshold:
                        correlations.append({
                            "feature1": col1,
                            "feature2": col2,
                            "pearson_correlation": float(corr_pearson),
                            "pearson_p_value": float(p_pearson),
                            "spearman_correlation": float(corr_spearman),
                            "spearman_p_value": float(p_spearman),
                            "strength": "strong" if max(abs(corr_pearson), abs(corr_spearman)) > 0.8 else "moderate"
                        })
            
            results["relationships"]["correlations"] = correlations
    
    # Association analysis (categorical variables)
    if "association" in methods:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 1:
            associations = []
            for i in range(len(categorical_cols)):
                for j in range(i+1, len(categorical_cols)):
                    col1, col2 = categorical_cols[i], categorical_cols[j]
                    
                    # Create contingency table
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    
                    # Chi-square test
                    try:
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        # Cramér's V
                        n = contingency_table.sum().sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                        
                        if cramers_v >= threshold:
                            associations.append({
                                "feature1": col1,
                                "feature2": col2,
                                "cramers_v": float(cramers_v),
                                "chi2_statistic": float(chi2),
                                "p_value": float(p_value),
                                "strength": "strong" if cramers_v > 0.8 else "moderate"
                            })
                    except ValueError:
                        # Skip if contingency table is invalid
                        continue
            
            results["relationships"]["associations"] = associations
    
    # Dependency analysis (mixed types)
    if "dependency" in methods:
        dependencies = []
        all_cols = df.columns.tolist()
        
        for target_col in all_cols:
            for feature_col in all_cols:
                if target_col != feature_col:
                    # Calculate dependency strength (simplified mutual information)
                    try:
                        dependency_strength = _calculate_dependency_strength(df[feature_col], df[target_col])
                        if dependency_strength >= threshold:
                            dependencies.append({
                                "feature": feature_col,
                                "target": target_col,
                                "dependency_strength": float(dependency_strength),
                                "strength": "strong" if dependency_strength > 0.8 else "moderate"
                            })
                    except:
                        continue
        
        results["relationships"]["dependencies"] = dependencies
    
    return results


def _calculate_dependency_strength(feature_series: "pd.Series", target_series: "pd.Series") -> float:
    """Calculate dependency strength between two variables (simplified mutual information)."""
    import pandas as pd
    import numpy as np
    
    # For simplicity, we'll use correlation for numeric and Cramér's V for categorical
    feature_numeric = pd.api.types.is_numeric_dtype(feature_series)
    target_numeric = pd.api.types.is_numeric_dtype(target_series)
    
    if feature_numeric and target_numeric:
        # Use correlation for numeric-numeric
        return abs(feature_series.corr(target_series))
    
    elif not feature_numeric and not target_numeric:
        # Use Cramér's V for categorical-categorical
        try:
            from scipy.stats import chi2_contingency
            contingency_table = pd.crosstab(feature_series, target_series)
            chi2, _, _, _ = chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            return cramers_v
        except:
            return 0.0
    
    else:
        # Mixed types - use ANOVA F-statistic (simplified)
        try:
            if feature_numeric:
                numeric_col, cat_col = feature_series, target_series
            else:
                numeric_col, cat_col = target_series, feature_series
            
            # Group numeric values by categories
            grouped = numeric_col.groupby(cat_col)
            group_means = grouped.mean()
            overall_mean = numeric_col.mean()
            
            # Calculate between-group variance
            between_var = sum(len(group) * (group.mean() - overall_mean)**2 for _, group in grouped)
            within_var = sum(group.var() * (len(group) - 1) for _, group in grouped if len(group) > 1)
            
            if within_var > 0:
                f_stat = between_var / within_var
                # Normalize to [0,1] range (simplified)
                return min(f_stat / (f_stat + 100), 1.0)
            else:
                return 0.0
        except:
            return 0.0


# Display functions

def _display_schema_table(schema_info: Dict[str, Any], filename: str):
    """Display schema information in a rich table."""
    # Dataset overview
    dataset_panel = Panel(
        f"[bold blue]Dataset Overview[/bold blue]\n"
        f"Rows: {schema_info['dataset_info']['total_rows']:,}\n"
        f"Columns: {schema_info['dataset_info']['total_columns']}\n"
        f"Memory Usage: {schema_info['dataset_info']['memory_usage_mb']:.2f} MB",
        title=f"Schema Discovery - {filename}"
    )
    console.print(dataset_panel)
    
    # Column details table
    table = Table(title="Column Details", show_header=True, header_style="bold magenta")
    table.add_column("Column", style="cyan", no_wrap=True)
    table.add_column("Type", style="yellow")
    table.add_column("Inferred", style="green")
    table.add_column("Non-Null", justify="right")
    table.add_column("Unique", justify="right")
    table.add_column("Missing %", justify="right")
    table.add_column("Examples", style="dim")
    
    for column, info in schema_info["columns"].items():
        examples = ", ".join(str(x) for x in info.get("sample_values", [])[:3])
        if len(examples) > 30:
            examples = examples[:27] + "..."
        
        missing_pct = info.get("null_percentage", 0)
        missing_color = "red" if missing_pct > 10 else "yellow" if missing_pct > 5 else "green"
        
        table.add_row(
            column,
            info["data_type"],
            info.get("inferred_type", ""),
            f"{info['non_null_count']:,}",
            f"{info['unique_count']:,}",
            f"[{missing_color}]{missing_pct:.1f}%[/{missing_color}]",
            examples
        )
    
    console.print(table)


def _display_quality_table(quality_results: Dict[str, Any], filename: str):
    """Display data quality results in rich tables."""
    overview = quality_results["overview"]
    
    # Overview panel
    overview_panel = Panel(
        f"[bold blue]Data Quality Assessment[/bold blue]\n"
        f"Dataset: {filename}\n"
        f"Rows: {overview['total_rows']:,}\n"
        f"Columns: {overview['total_columns']}\n"
        f"Checks: {', '.join(overview['checks_performed'])}",
        title="Quality Assessment Overview"
    )
    console.print(overview_panel)
    
    detailed = quality_results["detailed_results"]
    
    # Missing values
    if "missing_values" in detailed:
        missing_data = detailed["missing_values"]
        console.print("\n[bold]Missing Values Analysis[/bold]")
        
        missing_table = Table(show_header=True, header_style="bold magenta")
        missing_table.add_column("Column", style="cyan")
        missing_table.add_column("Missing Count", justify="right")
        missing_table.add_column("Missing %", justify="right")
        missing_table.add_column("Severity", justify="center")
        
        for column, info in missing_data["by_column"].items():
            if info["missing_count"] > 0:
                severity_color = "red" if info["severity"] == "high" else "yellow"
                missing_table.add_row(
                    column,
                    f"{info['missing_count']:,}",
                    f"{info['missing_percentage']:.1f}%",
                    f"[{severity_color}]{info['severity']}[/{severity_color}]"
                )
        
        if missing_table.rows:
            console.print(missing_table)
        else:
            console.print("[green]✅ No missing values found[/green]")
    
    # Duplicates
    if "duplicates" in detailed:
        dup_data = detailed["duplicates"]
        console.print(f"\n[bold]Duplicate Analysis[/bold]")
        console.print(f"Duplicate rows: {dup_data['duplicate_rows']:,} ({dup_data['duplicate_percentage']:.1f}%)")
        
        if dup_data["duplicate_columns"]:
            console.print("Duplicate columns found:")
            for col1, col2 in dup_data["duplicate_columns"]:
                console.print(f"  • {col1} ≡ {col2}")
        else:
            console.print("[green]✅ No duplicate columns found[/green]")
    
    # Outliers
    if "outliers" in detailed:
        outlier_data = detailed["outliers"]
        if outlier_data["by_column"]:
            console.print(f"\n[bold]Outlier Analysis[/bold]")
            
            outlier_table = Table(show_header=True, header_style="bold magenta")
            outlier_table.add_column("Column", style="cyan")
            outlier_table.add_column("IQR Outliers", justify="right")
            outlier_table.add_column("Z-Score Outliers", justify="right")
            outlier_table.add_column("Severity", justify="center")
            
            for column, info in outlier_data["by_column"].items():
                severity_color = "red" if info["severity"] == "high" else "yellow"
                outlier_table.add_row(
                    column,
                    f"{info['outliers_iqr']} ({info['outliers_iqr_percentage']:.1f}%)",
                    f"{info['outliers_zscore']} ({info['outliers_zscore_percentage']:.1f}%)",
                    f"[{severity_color}]{info['severity']}[/{severity_color}]"
                )
            
            console.print(outlier_table)


def _display_summary_table(summary_results: Dict[str, Any], filename: str):
    """Display data summary in rich format."""
    basic_info = summary_results["basic_info"]
    
    # Basic info panel
    info_panel = Panel(
        f"[bold blue]Dataset Summary[/bold blue]\n"
        f"File: {filename}\n"
        f"Rows: {basic_info['rows']:,}\n"
        f"Columns: {basic_info['columns']}\n"
        f"Memory: {basic_info['memory_usage_mb']:.2f} MB\n"
        f"Numeric: {basic_info['numeric_columns']}, "
        f"Categorical: {basic_info['categorical_columns']}, "
        f"DateTime: {basic_info['datetime_columns']}",
        title="Data Summary"
    )
    console.print(info_panel)
    
    # Column types
    if "column_types" in summary_results:
        console.print("\n[bold]Column Types Distribution[/bold]")
        types_table = Table(show_header=True, header_style="bold magenta")
        types_table.add_column("Data Type", style="cyan")
        types_table.add_column("Count", justify="right")
        
        for dtype, count in summary_results["column_types"].items():
            types_table.add_row(dtype, str(count))
        
        console.print(types_table)
    
    # Null values summary
    if "null_values" in summary_results and summary_results["null_values"]:
        console.print("\n[bold]Columns with Missing Values[/bold]")
        null_table = Table(show_header=True, header_style="bold magenta")
        null_table.add_column("Column", style="cyan")
        null_table.add_column("Missing Count", justify="right")
        null_table.add_column("Missing %", justify="right")
        
        for column, info in summary_results["null_values"].items():
            null_table.add_row(
                column,
                f"{info['count']:,}",
                f"{info['percentage']:.1f}%"
            )
        
        console.print(null_table)
    
    # Strong correlations
    if "correlations" in summary_results and summary_results["correlations"]["strong_correlations"]:
        console.print("\n[bold]Strong Correlations (>0.7)[/bold]")
        corr_table = Table(show_header=True, header_style="bold magenta")
        corr_table.add_column("Feature 1", style="cyan")
        corr_table.add_column("Feature 2", style="cyan")
        corr_table.add_column("Correlation", justify="right")
        
        for corr in summary_results["correlations"]["strong_correlations"]:
            corr_val = corr["correlation"]
            color = "green" if corr_val > 0 else "red"
            corr_table.add_row(
                corr["feature1"],
                corr["feature2"],
                f"[{color}]{corr_val:.3f}[/{color}]"
            )
        
        console.print(corr_table)


def _display_relationships_table(relationship_results: Dict[str, Any], filename: str):
    """Display relationship analysis results."""
    # Overview panel
    overview_panel = Panel(
        f"[bold blue]Relationship Analysis[/bold blue]\n"
        f"Dataset: {filename}\n"
        f"Methods: {', '.join(relationship_results['methods_used'])}\n"
        f"Threshold: {relationship_results['threshold']}",
        title="Relationship Analysis"
    )
    console.print(overview_panel)
    
    relationships = relationship_results["relationships"]
    
    # Correlations
    if "correlations" in relationships and relationships["correlations"]:
        console.print("\n[bold]Numeric Correlations[/bold]")
        corr_table = Table(show_header=True, header_style="bold magenta")
        corr_table.add_column("Feature 1", style="cyan")
        corr_table.add_column("Feature 2", style="cyan")
        corr_table.add_column("Pearson", justify="right")
        corr_table.add_column("Spearman", justify="right")
        corr_table.add_column("Strength", justify="center")
        
        for corr in relationships["correlations"]:
            pearson_color = "green" if corr["pearson_correlation"] > 0 else "red"
            spearman_color = "green" if corr["spearman_correlation"] > 0 else "red"
            strength_color = "red" if corr["strength"] == "strong" else "yellow"
            
            corr_table.add_row(
                corr["feature1"],
                corr["feature2"],
                f"[{pearson_color}]{corr['pearson_correlation']:.3f}[/{pearson_color}]",
                f"[{spearman_color}]{corr['spearman_correlation']:.3f}[/{spearman_color}]",
                f"[{strength_color}]{corr['strength']}[/{strength_color}]"
            )
        
        console.print(corr_table)
    
    # Associations
    if "associations" in relationships and relationships["associations"]:
        console.print("\n[bold]Categorical Associations[/bold]")
        assoc_table = Table(show_header=True, header_style="bold magenta")
        assoc_table.add_column("Feature 1", style="cyan")
        assoc_table.add_column("Feature 2", style="cyan")
        assoc_table.add_column("Cramér's V", justify="right")
        assoc_table.add_column("P-Value", justify="right")
        assoc_table.add_column("Strength", justify="center")
        
        for assoc in relationships["associations"]:
            strength_color = "red" if assoc["strength"] == "strong" else "yellow"
            p_color = "green" if assoc["p_value"] < 0.05 else "red"
            
            assoc_table.add_row(
                assoc["feature1"],
                assoc["feature2"],
                f"{assoc['cramers_v']:.3f}",
                f"[{p_color}]{assoc['p_value']:.4f}[/{p_color}]",
                f"[{strength_color}]{assoc['strength']}[/{strength_color}]"
            )
        
        console.print(assoc_table)


def _generate_quality_html_report(quality_results: Dict[str, Any], output_path: Path, filename: str):
    """Generate HTML quality report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Quality Report - {filename}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .high {{ color: red; }}
            .low {{ color: green; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Data Quality Report</h1>
            <p><strong>Dataset:</strong> {filename}</p>
            <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Overview</h2>
            <p>Rows: {quality_results['overview']['total_rows']:,}</p>
            <p>Columns: {quality_results['overview']['total_columns']}</p>
            <p>Checks Performed: {', '.join(quality_results['overview']['checks_performed'])}</p>
        </div>
        
        <!-- Add more sections based on detailed_results -->
        
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)


def _create_relationship_graph(relationship_results: Dict[str, Any], output_path: Path, threshold: float):
    """Create relationship graph visualization."""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.Graph()
        
        # Add nodes and edges from relationships
        relationships = relationship_results["relationships"]
        
        if "correlations" in relationships:
            for corr in relationships["correlations"]:
                G.add_edge(corr["feature1"], corr["feature2"], 
                          weight=abs(corr["pearson_correlation"]),
                          type="correlation")
        
        if "associations" in relationships:
            for assoc in relationships["associations"]:
                G.add_edge(assoc["feature1"], assoc["feature2"],
                          weight=assoc["cramers_v"],
                          type="association")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.8)
        
        # Draw edges with different styles
        correlation_edges = [(u, v) for u, v, d in G.edges(data=True) 
                           if d.get('type') == 'correlation']
        association_edges = [(u, v) for u, v, d in G.edges(data=True) 
                           if d.get('type') == 'association']
        
        nx.draw_networkx_edges(G, pos, edgelist=correlation_edges, 
                              edge_color='blue', alpha=0.6, label='Correlation')
        nx.draw_networkx_edges(G, pos, edgelist=association_edges,
                              edge_color='red', alpha=0.6, style='dashed', 
                              label='Association')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title(f"Feature Relationships (threshold: {threshold})")
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        console.print("[yellow]⚠️  NetworkX or Matplotlib not available for graph visualization[/yellow]")


if __name__ == "__main__":
    app()