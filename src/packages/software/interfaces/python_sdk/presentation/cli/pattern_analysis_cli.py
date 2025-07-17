"""
Processing CLI Interface

Command-line interface for pattern analysis processing operations.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional
import click

from ...application.services.pattern_analysis_service import PatternAnalysisService
from ...application.dto.pattern_analysis_dto import PatternAnalysisRequestDTO
from ...domain.value_objects.algorithm_config import AlgorithmType
from ...infrastructure.adapters.pyod_algorithm_adapter import PyODAlgorithmAdapter


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """Software Python SDK CLI - Pattern Analysis Processing Tools"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@cli.command()
@click.option('--data', '-d', required=True, help='Path to data file or comma-separated values')
@click.option('--algorithm', '-a', default='isolation_forest',
              type=click.Choice(['isolation_forest', 'local_outlier_factor', 'one_class_svm', 
                               'elliptic_envelope', 'autoencoder']),
              help='Algorithm to use for pattern analysis')
@click.option('--contamination', '-c', default=0.1, type=float,
              help='Expected proportion of outliers')
@click.option('--output', '-o', help='Output file for results (JSON format)')
@click.option('--format', 'output_format', default='json',
              type=click.Choice(['json', 'csv', 'text']),
              help='Output format')
@click.pass_context
def detect(ctx, data, algorithm, contamination, output, output_format):
    """
    Analyze patterns in the provided data.
    
    DATA can be either:
    - Path to a CSV/JSON file containing numeric data
    - Comma-separated numeric values (e.g., "1.0,2.0,3.0,100.0,4.0")
    
    Examples:
        software-sdk detect --data "1,2,3,100,4,5" --algorithm isolation_forest
        software-sdk detect --data data.csv --algorithm lof --contamination 0.05
    """
    try:
        # Parse input data
        data_values = _parse_data_input(data)
        
        if ctx.obj['verbose']:
            click.echo(f"Loaded {len(data_values)} data points")
            click.echo(f"Using algorithm: {algorithm}")
            click.echo(f"Contamination rate: {contamination}")
        
        # Create processing request
        algorithm_config = {
            "algorithm_type": algorithm,
            "parameters": {},
            "contamination": contamination
        }
        
        request_dto = PatternAnalysisRequestDTO(
            data=data_values,
            algorithm_config=algorithm_config
        )
        
        # Execute processing
        if ctx.obj['verbose']:
            click.echo("Starting pattern analysis processing...")
        
        result = asyncio.run(_execute_processing(request_dto))
        
        # Format and output results
        _output_results(result, output, output_format, ctx.obj['verbose'])
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--data', '-d', required=True, help='Path to data file or comma-separated values')
@click.option('--top', '-t', default=3, type=int, help='Number of top recommendations to show')
@click.pass_context
def recommend(ctx, data, top):
    """
    Get algorithm recommendations for the provided data.
    
    Analyzes the data characteristics and suggests the most suitable
    pattern analysis algorithms.
    """
    try:
        # Parse input data
        data_values = _parse_data_input(data)
        
        if ctx.obj['verbose']:
            click.echo(f"Analyzing {len(data_values)} data points")
        
        # Get recommendations
        recommendations = asyncio.run(_get_recommendations(data_values))
        
        click.echo("\nRecommended algorithms:")
        for i, algo in enumerate(recommendations[:top], 1):
            click.echo(f"{i}. {algo}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def algorithms():
    """List all supported algorithms and their descriptions."""
    algorithms_info = {
        "isolation_forest": {
            "description": "Isolation Forest - Fast tree-based algorithm",
            "best_for": "Large datasets, general-purpose analysis",
            "complexity": "O(n log n)"
        },
        "local_outlier_factor": {
            "description": "Local Outlier Factor - Density-based processing",
            "best_for": "Local patterns, varying density datasets",
            "complexity": "O(n²)"
        },
        "one_class_svm": {
            "description": "One-Class SVM - Support vector machine approach",
            "best_for": "High-dimensional data, small datasets",
            "complexity": "O(n³)"
        },
        "elliptic_envelope": {
            "description": "Elliptic Envelope - Assumes Gaussian distribution",
            "best_for": "Gaussian-distributed data",
            "complexity": "O(n³)"
        },
        "autoencoder": {
            "description": "Autoencoder - Deep learning approach",
            "best_for": "Complex patterns, large datasets",
            "complexity": "Depends on architecture"
        }
    }
    
    click.echo("Supported Pattern Analysis Algorithms:\n")
    
    for algo_name, info in algorithms_info.items():
        click.echo(f"🔍 {algo_name}")
        click.echo(f"   Description: {info['description']}")
        click.echo(f"   Best for: {info['best_for']}")
        click.echo(f"   Complexity: {info['complexity']}")
        click.echo()


def _parse_data_input(data_input: str) -> List[float]:
    """
    Parse data input from file or comma-separated values.
    
    Args:
        data_input: File path or comma-separated values.
        
    Returns:
        List[float]: Parsed numeric data.
    """
    # Check if it's a file path
    if ',' not in data_input and Path(data_input).exists():
        file_path = Path(data_input)
        
        if file_path.suffix.lower() == '.csv':
            import pandas as pd
            df = pd.read_csv(file_path)
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=[float, int]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in CSV file")
            return df[numeric_cols[0]].tolist()
            
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [float(x) for x in data]
                elif isinstance(data, dict) and 'data' in data:
                    return [float(x) for x in data['data']]
                else:
                    raise ValueError("JSON file must contain a list or object with 'data' key")
        else:
            # Try to read as text file with one value per line
            with open(file_path, 'r') as f:
                lines = f.readlines()
                return [float(line.strip()) for line in lines if line.strip()]
    
    # Parse as comma-separated values
    try:
        return [float(x.strip()) for x in data_input.split(',')]
    except ValueError:
        raise ValueError("Invalid data format. Provide comma-separated numbers or a valid file path.")


async def _execute_processing(request_dto: PatternAnalysisRequestDTO) -> dict:
    """
    Execute pattern analysis using the application service.
    
    Args:
        request_dto: Processing request data.
        
    Returns:
        dict: Processing results.
    """
    # For CLI usage, we'll create a simple in-memory setup
    # In a real application, this would be injected via dependency injection
    
    algorithm_adapter = PyODAlgorithmAdapter()
    
    from ...domain.value_objects.algorithm_config import AlgorithmConfig
    
    # Convert DTO to domain objects
    algorithm_config = AlgorithmConfig.from_dict(request_dto.algorithm_config)
    
    # Execute processing directly
    result = await algorithm_adapter.analyze_patterns(
        data=request_dto.data,
        algorithm_config=algorithm_config
    )
    
    return result.to_dict()


async def _get_recommendations(data: List[float]) -> List[str]:
    """
    Get algorithm recommendations for the data.
    
    Args:
        data: Input data to analyze.
        
    Returns:
        List[str]: Recommended algorithm names.
    """
    data_size = len(data)
    
    # Simple heuristic-based recommendations
    recommendations = []
    
    if data_size < 1000:
        recommendations.extend(["local_outlier_factor", "elliptic_envelope", "one_class_svm"])
    elif data_size < 10000:
        recommendations.extend(["isolation_forest", "local_outlier_factor", "elliptic_envelope"])
    else:
        recommendations.extend(["isolation_forest", "autoencoder"])
    
    return recommendations


def _output_results(result: dict, output_file: Optional[str], format_type: str, verbose: bool):
    """
    Output processing results in the specified format.
    
    Args:
        result: Processing results dictionary.
        output_file: Optional output file path.
        format_type: Output format (json, csv, text).
        verbose: Whether to include verbose output.
    """
    if format_type == 'json':
        output_data = json.dumps(result, indent=2)
    elif format_type == 'csv':
        import pandas as pd
        df = pd.DataFrame({
            'index': range(len(result['anomalies'])),
            'is_pattern': result['patterns'],
            'score': result['scores']
        })
        output_data = df.to_csv(index=False)
    elif format_type == 'text':
        pattern_indices = [i for i, is_pattern in enumerate(result['patterns']) if is_pattern]
        output_data = f"""Pattern Analysis Results
Algorithm: {result['algorithm_type']}
Execution Time: {result['execution_time_ms']}ms
Total Data Points: {len(result['patterns'])}
Patterns Found: {sum(result['patterns'])}

Pattern Indices: {pattern_indices}
"""
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_data)
        if verbose:
            click.echo(f"Results saved to {output_file}")
    else:
        click.echo(output_data)


if __name__ == '__main__':
    cli()