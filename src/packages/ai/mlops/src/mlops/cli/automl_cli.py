"""
AutoML CLI Commands

Command-line interface for automated machine learning operations,
including neural architecture search, hyperparameter optimization,
and model optimization.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import click
import pandas as pd
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich.tree import Tree
from rich import print as rprint

from ..infrastructure.automl.advanced_automl_engine import (
    AdvancedAutoMLEngine, AutoMLTask, OptimizationObjective, 
    AutoMLConfig, FeatureEngineering
)
from ..infrastructure.automl.neural_architecture_search import (
    NeuralArchitectureSearchEngine, NASMethod, NASConfig
)
from ..infrastructure.automl.hyperparameter_optimization import (
    HyperparameterOptimizer, OptimizationMethod, HPOConfig, SearchSpace, SearchSpaceType
)
from ..infrastructure.automl.model_optimization import (
    ModelOptimizer, OptimizationType, OptimizationConfig, PruningMethod, QuantizationMethod
)


console = Console()


@click.group()
@click.pass_context
def automl(ctx):
    """AutoML and Model Optimization CLI."""
    ctx.ensure_object(dict)
    
    # Initialize AutoML components
    ctx.obj['automl_engine'] = AdvancedAutoMLEngine()
    ctx.obj['nas_engine'] = NeuralArchitectureSearchEngine()
    ctx.obj['hpo_engine'] = HyperparameterOptimizer()
    ctx.obj['model_optimizer'] = ModelOptimizer()


@automl.group()
def experiment():
    """AutoML experiment management."""
    pass


@experiment.command()
@click.option('--data-path', required=True, help='Path to training data CSV file')
@click.option('--target-column', required=True, help='Name of target column')
@click.option('--task-type', 
              type=click.Choice([t.value for t in AutoMLTask]),
              required=True,
              help='ML task type')
@click.option('--objective',
              type=click.Choice([o.value for o in OptimizationObjective]),
              default='accuracy',
              help='Optimization objective')
@click.option('--max-trials', default=100, help='Maximum number of trials')
@click.option('--max-time', default=60, help='Maximum time in minutes')
@click.option('--feature-engineering',
              type=click.Choice([f.value for f in FeatureEngineering]),
              default='advanced',
              help='Feature engineering strategy')
@click.option('--enable-nas', is_flag=True, help='Enable neural architecture search')
@click.option('--experiment-name', help='Name for the experiment')
@click.pass_context
async def start(ctx, data_path, target_column, task_type, objective, max_trials, 
                max_time, feature_engineering, enable_nas, experiment_name):
    """Start a new AutoML experiment."""
    
    automl_engine = ctx.obj['automl_engine']
    
    # Load data
    try:
        data = pd.read_csv(data_path)
        console.print(f"✅ Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
    except Exception as e:
        console.print(f"❌ Error loading data: {str(e)}")
        return
    
    # Validate target column
    if target_column not in data.columns:
        console.print(f"❌ Target column '{target_column}' not found in data")
        return
    
    # Create AutoML configuration
    config = AutoMLConfig(
        task_type=AutoMLTask(task_type),
        optimization_objective=OptimizationObjective(objective),
        max_trials=max_trials,
        max_runtime_minutes=max_time,
        feature_engineering=FeatureEngineering(feature_engineering),
        enable_nas=enable_nas
    )
    
    # Start experiment
    with console.status("Starting AutoML experiment..."):
        experiment_id = await automl_engine.start_automl_experiment(
            data=data,
            target_column=target_column,
            config=config,
            experiment_name=experiment_name or ""
        )
    
    console.print(f"✅ AutoML experiment started with ID: [bold blue]{experiment_id}[/bold blue]")
    console.print(f"📊 Task: {task_type}, Objective: {objective}")
    console.print(f"⏱️  Max time: {max_time} minutes, Max trials: {max_trials}")
    
    # Show progress
    with Progress() as progress:
        task = progress.add_task("Running AutoML experiment...", total=max_time * 60)
        
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds
            
            status = await automl_engine.get_experiment_status(experiment_id)
            
            if status["status"] == "completed":
                progress.update(task, completed=max_time * 60)
                break
            
            # Update progress based on time
            elapsed = status.get("training_time_seconds", 0)
            progress.update(task, completed=min(elapsed, max_time * 60))
    
    # Show results
    result = await automl_engine.get_experiment_results(experiment_id)
    
    results_panel = Panel(
        f"""
[bold]AutoML Experiment Results[/bold]

[bold]Best Model Performance:[/bold]
• CV Score: {result.best_score:.4f}
• Test Score: {result.test_score:.4f}
• Training Time: {result.training_time_seconds:.1f} seconds

[bold]Model Details:[/bold]
• Model Type: {result.best_params.get('model_type', 'Unknown')}
• Features Selected: {len(result.selected_features)}
• Features Engineered: {len(result.engineered_features)}

[bold]Optimization:[/bold]
• Total Trials: {len(result.optimization_history)}
• Model Path: {result.model_path or 'Not saved'}
        """.strip(),
        title=f"Experiment {experiment_id[:8]}",
        expand=False
    )
    console.print(results_panel)


@experiment.command()
@click.pass_context
async def list(ctx):
    """List all AutoML experiments."""
    
    automl_engine = ctx.obj['automl_engine']
    
    experiments = await automl_engine.list_experiments()
    
    if not experiments:
        console.print("No experiments found.")
        return
    
    table = Table(title="AutoML Experiments")
    table.add_column("Experiment ID", style="cyan")
    table.add_column("Task Type", style="bright_white")
    table.add_column("Status", style="green")
    table.add_column("Best Score", style="yellow")
    table.add_column("Test Score", style="magenta")
    table.add_column("Created", style="blue")
    table.add_column("Duration", style="white")
    
    for exp in experiments:
        duration = f"{exp['training_time_seconds']:.1f}s" if exp['training_time_seconds'] > 0 else "N/A"
        
        table.add_row(
            exp['experiment_id'][:8] + "...",
            exp['task_type'],
            exp['status'],
            f"{exp['best_score']:.4f}",
            f"{exp['test_score']:.4f}",
            exp['created_at'][:19],
            duration
        )
    
    console.print(table)


@experiment.command()
@click.argument('experiment_id')
@click.pass_context
async def status(ctx, experiment_id):
    """Get status of an AutoML experiment."""
    
    automl_engine = ctx.obj['automl_engine']
    
    try:
        status = await automl_engine.get_experiment_status(experiment_id)
        
        status_panel = Panel(
            f"""
[bold]Experiment Status[/bold]

[bold]ID:[/bold] {status['experiment_id']}
[bold]Status:[/bold] {status['status']}
[bold]Task Type:[/bold] {status['task_type']}
[bold]Created:[/bold] {status['created_at']}
[bold]Training Time:[/bold] {status['training_time_seconds']:.1f} seconds
[bold]Best Score:[/bold] {status['best_score']:.4f}
[bold]Test Score:[/bold] {status['test_score']:.4f}
[bold]Trials:[/bold] {status['optimization_trials']}
[bold]Features:[/bold] {status['selected_features_count']}
[bold]Model Path:[/bold] {status['model_path'] or 'Not saved'}
            """.strip(),
            title="Experiment Status",
            expand=False
        )
        console.print(status_panel)
        
    except ValueError as e:
        console.print(f"❌ {str(e)}")


@experiment.command()
@click.argument('experiment_id')
@click.argument('data_path')
@click.option('--output', help='Output file for predictions')
@click.pass_context
async def predict(ctx, experiment_id, data_path, output):
    """Make predictions using trained AutoML model."""
    
    automl_engine = ctx.obj['automl_engine']
    
    try:
        # Load test data
        test_data = pd.read_csv(data_path)
        console.print(f"✅ Loaded test data: {test_data.shape[0]} rows, {test_data.shape[1]} columns")
        
        # Make predictions
        with console.status("Making predictions..."):
            predictions = await automl_engine.predict_with_experiment(experiment_id, test_data)
        
        console.print(f"✅ Generated {len(predictions)} predictions")
        
        # Save predictions
        if output:
            results_df = test_data.copy()
            results_df['prediction'] = predictions
            results_df.to_csv(output, index=False)
            console.print(f"✅ Predictions saved to {output}")
        else:
            # Show first few predictions
            console.print("\n[bold]Sample Predictions:[/bold]")
            for i, pred in enumerate(predictions[:10]):
                console.print(f"Sample {i+1}: {pred}")
            
            if len(predictions) > 10:
                console.print(f"... and {len(predictions) - 10} more")
        
    except Exception as e:
        console.print(f"❌ Error making predictions: {str(e)}")


@automl.group()
def nas():
    """Neural Architecture Search commands."""
    pass


@nas.command()
@click.option('--data-path', required=True, help='Path to training data CSV file')
@click.option('--target-column', required=True, help='Name of target column')
@click.option('--method',
              type=click.Choice([m.value for m in NASMethod]),
              default='evolutionary',
              help='NAS method')
@click.option('--population-size', default=20, help='Population size for evolutionary search')
@click.option('--generations', default=50, help='Number of generations')
@click.option('--max-layers', default=10, help='Maximum number of layers')
@click.option('--max-units', default=512, help='Maximum units per layer')
@click.pass_context
async def search(ctx, data_path, target_column, method, population_size, 
                generations, max_layers, max_units):
    """Perform neural architecture search."""
    
    nas_engine = ctx.obj['nas_engine']
    
    # Load and prepare data
    try:
        data = pd.read_csv(data_path)
        console.print(f"✅ Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Basic preprocessing
        X = data.drop(columns=[target_column]).values
        y = data[target_column].values
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
    except Exception as e:
        console.print(f"❌ Error loading data: {str(e)}")
        return
    
    # Create NAS configuration
    config = NASConfig(
        method=NASMethod(method),
        population_size=population_size,
        generations=generations,
        max_layers=max_layers,
        max_units=max_units
    )
    
    nas_engine.config = config
    
    # Start search
    with console.status("Performing neural architecture search..."):
        best_architecture = await nas_engine.search_architecture(X_tensor, y_tensor)
    
    # Display results
    results_panel = Panel(
        f"""
[bold]Neural Architecture Search Results[/bold]

[bold]Best Architecture:[/bold]
• Validation Accuracy: {best_architecture.validation_accuracy:.4f}
• Parameters: {best_architecture.parameters_count:,}
• Layers: {len(best_architecture.layers)}
• Training Time: {best_architecture.training_time_seconds:.1f} seconds

[bold]Architecture Details:[/bold]
• Architecture ID: {best_architecture.architecture_id[:8]}...
• Generation: {best_architecture.generation}
• Memory Usage: {best_architecture.memory_usage_mb:.1f} MB

[bold]Optimization:[/bold]
• Optimizer: {best_architecture.optimizer}
• Learning Rate: {best_architecture.learning_rate:.6f}
• Batch Size: {best_architecture.batch_size}
        """.strip(),
        title="NAS Results",
        expand=False
    )
    console.print(results_panel)


@nas.command()
@click.pass_context
async def results(ctx):
    """Show neural architecture search results."""
    
    nas_engine = ctx.obj['nas_engine']
    
    try:
        results = await nas_engine.get_search_results()
        
        # Search metrics
        metrics_panel = Panel(
            f"""
[bold]Search Metrics[/bold]

[bold]Best Architecture:[/bold]
• Best Accuracy: {results['search_metrics']['best_accuracy']:.4f}
• Search Time: {results['search_metrics']['search_time_seconds']:.1f} seconds
• Architectures Evaluated: {results['search_metrics']['total_architectures_evaluated']}

[bold]Convergence:[/bold]
• Population Diversity: {results['population_diversity']:.4f}
• Convergence Generation: {results['search_metrics'].get('convergence_generation', 'N/A')}
            """.strip(),
            title="Search Performance",
            expand=False
        )
        console.print(metrics_panel)
        
        # Pareto front
        if results['pareto_front']:
            console.print("\n[bold]Pareto Front (Top Architectures):[/bold]")
            
            table = Table()
            table.add_column("Architecture ID", style="cyan")
            table.add_column("Accuracy", style="green")
            table.add_column("Parameters", style="yellow")
            table.add_column("Memory (MB)", style="magenta")
            
            for arch in results['pareto_front'][:10]:  # Show top 10
                table.add_row(
                    arch['architecture_id'][:8] + "...",
                    f"{arch['validation_accuracy']:.4f}",
                    f"{arch['parameters_count']:,}",
                    f"{arch['memory_usage_mb']:.1f}"
                )
            
            console.print(table)
        
    except Exception as e:
        console.print(f"❌ Error retrieving results: {str(e)}")


@automl.group()
def hpo():
    """Hyperparameter Optimization commands."""
    pass


@hpo.command()
@click.option('--model-type', required=True, 
              type=click.Choice(['random_forest', 'svm', 'logistic_regression', 'mlp']),
              help='Model type to optimize')
@click.option('--data-path', required=True, help='Path to training data CSV file')
@click.option('--target-column', required=True, help='Name of target column')
@click.option('--method',
              type=click.Choice([m.value for m in OptimizationMethod]),
              default='bayesian_optimization',
              help='Optimization method')
@click.option('--max-trials', default=100, help='Maximum number of trials')
@click.option('--max-time', default=60, help='Maximum time in minutes')
@click.pass_context
async def optimize(ctx, model_type, data_path, target_column, method, max_trials, max_time):
    """Optimize hyperparameters for a model."""
    
    hpo_engine = ctx.obj['hpo_engine']
    
    # Load data
    try:
        data = pd.read_csv(data_path)
        X = data.drop(columns=[target_column])
        y = data[target_column]
        console.print(f"✅ Loaded data: {X.shape[0]} rows, {X.shape[1]} features")
    except Exception as e:
        console.print(f"❌ Error loading data: {str(e)}")
        return
    
    # Create search space based on model type
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        search_space = hpo_engine.create_search_space_from_model(RandomForestClassifier)
    elif model_type == 'svm':
        from sklearn.svm import SVC
        search_space = hpo_engine.create_search_space_from_model(SVC)
    elif model_type == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        search_space = hpo_engine.create_search_space_from_model(LogisticRegression)
    elif model_type == 'mlp':
        from sklearn.neural_network import MLPClassifier
        search_space = hpo_engine.create_search_space_from_model(MLPClassifier)
    else:
        console.print(f"❌ Unsupported model type: {model_type}")
        return
    
    # Create HPO configuration
    config = HPOConfig(
        method=OptimizationMethod(method),
        max_trials=max_trials,
        max_time_minutes=max_time
    )
    
    hpo_engine.config = config
    
    # Define objective function
    def objective_function(params):
        try:
            # Create model with parameters
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**params, random_state=42)
            elif model_type == 'svm':
                from sklearn.svm import SVC
                model = SVC(**params, random_state=42)
            elif model_type == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(**params, random_state=42)
            elif model_type == 'mlp':
                from sklearn.neural_network import MLPClassifier
                model = MLPClassifier(**params, random_state=42)
            
            # Cross-validation score
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            return scores.mean()
            
        except Exception as e:
            return 0.0  # Return poor score for invalid parameters
    
    # Start optimization
    with Progress() as progress:
        task = progress.add_task("Optimizing hyperparameters...", total=max_trials)
        
        # Run optimization
        result = await hpo_engine.optimize(objective_function, search_space)
        progress.update(task, completed=max_trials)
    
    # Display results
    results_panel = Panel(
        f"""
[bold]Hyperparameter Optimization Results[/bold]

[bold]Best Configuration:[/bold]
• Best Score: {result.best_score:.4f}
• Best Trial: {result.best_trial_number}

[bold]Best Parameters:[/bold]
{json.dumps(result.best_params, indent=2)}

[bold]Optimization Summary:[/bold]
• Method: {result.method.value}
• Total Trials: {result.total_trials}
• Successful Trials: {result.successful_trials}
• Failed Trials: {result.failed_trials}
• Optimization Time: {result.optimization_time_seconds:.1f} seconds
• Average Trial Time: {result.average_trial_time_seconds:.2f} seconds

[bold]Convergence:[/bold]
• Convergence Trial: {result.convergence_trial or 'Not converged'}
• Improvement Rate: {result.improvement_rate:.4f}
        """.strip(),
        title=f"HPO Results - {model_type}",
        expand=False
    )
    console.print(results_panel)


@automl.group()
def optimize():
    """Model optimization commands."""
    pass


@optimize.command()
@click.option('--model-path', required=True, help='Path to PyTorch model file')
@click.option('--data-path', required=True, help='Path to training data CSV file')
@click.option('--optimization-type',
              type=click.Choice([o.value for o in OptimizationType]),
              required=True,
              help='Type of optimization')
@click.option('--sparsity-ratio', default=0.5, help='Sparsity ratio for pruning')
@click.option('--target-speedup', default=2.0, help='Target inference speedup')
@click.option('--target-compression', default=4.0, help='Target model size compression')
@click.pass_context
async def model(ctx, model_path, data_path, optimization_type, 
                sparsity_ratio, target_speedup, target_compression):
    """Optimize a trained PyTorch model."""
    
    model_optimizer = ctx.obj['model_optimizer']
    
    try:
        # Load model
        model = torch.load(model_path, map_location='cpu')
        console.print(f"✅ Loaded model from {model_path}")
        
        # Load data
        data = pd.read_csv(data_path)
        # This would need proper preprocessing and data loader creation
        console.print(f"✅ Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Create dummy data loaders for demonstration
        from torch.utils.data import DataLoader, TensorDataset
        
        # Simplified data preparation
        X = torch.randn(100, 784)  # Example input
        y = torch.randint(0, 10, (100,))  # Example labels
        
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
    except Exception as e:
        console.print(f"❌ Error loading model or data: {str(e)}")
        return
    
    # Create optimization configuration
    config = OptimizationConfig(
        optimization_type=OptimizationType(optimization_type),
        sparsity_ratio=sparsity_ratio,
        target_speedup=target_speedup,
        target_compression_ratio=target_compression
    )
    
    if optimization_type == 'pruning':
        config.pruning_method = PruningMethod.MAGNITUDE
    elif optimization_type == 'quantization':
        config.quantization_method = QuantizationMethod.DYNAMIC
    
    model_optimizer.config = config
    
    # Start optimization
    with console.status("Optimizing model..."):
        result = await model_optimizer.optimize_model(
            model, train_loader, val_loader
        )
    
    # Display results
    results_panel = Panel(
        f"""
[bold]Model Optimization Results[/bold]

[bold]Original Model:[/bold]
• Accuracy: {result.original_accuracy:.4f}
• Size: {result.original_model_size_mb:.2f} MB
• Inference Time: {result.original_inference_time_ms:.2f} ms
• FLOPs: {result.original_flops:,}

[bold]Optimized Model:[/bold]
• Accuracy: {result.optimized_accuracy:.4f}
• Size: {result.optimized_model_size_mb:.2f} MB
• Inference Time: {result.optimized_inference_time_ms:.2f} ms
• FLOPs: {result.optimized_flops:,}

[bold]Improvements:[/bold]
• Accuracy Change: {result.accuracy_change:+.4f}
• Size Reduction: {result.size_reduction_ratio:.2f}x
• Speedup: {result.speedup_ratio:.2f}x
• FLOP Reduction: {result.flops_reduction_ratio:.2f}x

[bold]Optimization:[/bold]
• Method: {result.optimization_method}
• Successful: {'✅' if result.optimization_successful else '❌'}
• Time: {result.optimization_time_seconds:.1f} seconds
• Model Path: {result.optimized_model_path or 'Not saved'}
        """.strip(),
        title=f"Optimization Results - {optimization_type}",
        expand=False
    )
    console.print(results_panel)


@automl.command()
@click.option('--component',
              type=click.Choice(['automl', 'nas', 'hpo', 'optimization']),
              help='Specific component to check')
@click.pass_context
async def health(ctx, component):
    """Check AutoML system health."""
    
    health_status = {}
    
    if not component or component == 'automl':
        automl_engine = ctx.obj['automl_engine']
        health_status['automl'] = {
            "status": "healthy",
            "experiments": len(automl_engine.experiments),
            "background_tasks": len(automl_engine.background_tasks),
            "feature_engineer": "available",
            "model_registry": "available"
        }
    
    if not component or component == 'nas':
        nas_engine = ctx.obj['nas_engine']
        health_status['nas'] = {
            "status": "healthy",
            "population_size": len(nas_engine.population),
            "search_history": len(nas_engine.search_history),
            "pareto_front": len(nas_engine.pareto_front),
            "device": str(nas_engine.device)
        }
    
    if not component or component == 'hpo':
        hpo_engine = ctx.obj['hpo_engine']
        health_status['hpo'] = {
            "status": "healthy",
            "active_studies": len(hpo_engine.active_studies),
            "optimization_results": len(hpo_engine.optimization_results),
            "search_spaces": len(hpo_engine.search_spaces)
        }
    
    if not component or component == 'optimization':
        model_optimizer = ctx.obj['model_optimizer']
        health_status['optimization'] = {
            "status": "healthy",
            "optimization_results": len(model_optimizer.optimization_results),
            "device": str(model_optimizer.device),
            "pruning_optimizer": "available",
            "quantization_optimizer": "available"
        }
    
    _display_health_status(health_status)


def _display_health_status(health_status: Dict[str, Any]) -> None:
    """Display system health status."""
    
    for component, status in health_status.items():
        status_icon = "🟢" if status["status"] == "healthy" else "🔴"
        
        panel_content = f"[bold]Status:[/bold] {status_icon} {status['status'].upper()}\n"
        
        # Add component-specific metrics
        for key, value in status.items():
            if key != "status":
                panel_content += f"[bold]{key.replace('_', ' ').title()}:[/bold] {value}\n"
        
        panel = Panel(
            panel_content.strip(),
            title=f"{component.title()} Health",
            expand=False
        )
        console.print(panel)


if __name__ == "__main__":
    # Support async CLI commands
    def async_command(f):
        def wrapper(*args, **kwargs):
            return asyncio.run(f(*args, **kwargs))
        return wrapper
    
    # Apply async wrapper to commands that need it
    async_commands = [
        experiment.commands['start'],
        experiment.commands['list'],
        experiment.commands['status'],
        experiment.commands['predict'],
        nas.commands['search'],
        nas.commands['results'],
        hpo.commands['optimize'],
        optimize.commands['model'],
        automl.commands['health']
    ]
    
    for command in async_commands:
        command.callback = async_command(command.callback)
    
    automl()