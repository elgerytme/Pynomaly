"""Enhanced AutoML CLI commands with advanced optimization features."""

import asyncio
import json

import click
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pynomaly_detection.application.services.enhanced_automl_service import EnhancedAutoMLConfig
from pynomaly_detection.infrastructure.config.container import create_container
from pynomaly_detection.infrastructure.config.feature_flags import require_feature

console = Console()


@click.group(name="enhanced-automl")
def enhanced_automl_cli():
    """Enhanced AutoML commands with advanced optimization techniques."""
    pass


@enhanced_automl_cli.command()
@require_feature("advanced_automl")
@click.argument("dataset_id")
@click.argument("algorithm")
@click.option(
    "--objectives",
    "-o",
    multiple=True,
    default=["auc"],
    help="Optimization objectives (auc, precision, recall, training_time)",
)
@click.option(
    "--strategy",
    "-s",
    default="bayesian",
    type=click.Choice(
        ["bayesian", "hyperband", "bohb", "multi_objective", "evolutionary"]
    ),
    help="Optimization strategy",
)
@click.option(
    "--acquisition",
    "-a",
    default="expected_improvement",
    type=click.Choice(
        ["expected_improvement", "probability_improvement", "upper_confidence_bound"]
    ),
    help="Acquisition function for Bayesian optimization",
)
@click.option("--n-trials", "-n", default=100, help="Number of optimization trials")
@click.option("--timeout", "-t", default=3600, help="Optimization timeout in seconds")
@click.option(
    "--enable-meta-learning", is_flag=True, default=True, help="Enable meta-learning"
)
@click.option(
    "--enable-parallel", is_flag=True, default=True, help="Enable parallel optimization"
)
@click.option("--output", "-f", help="Output file for results (JSON)")
def optimize(
    dataset_id: str,
    algorithm: str,
    objectives: list[str],
    strategy: str,
    acquisition: str,
    n_trials: int,
    timeout: int,
    enable_meta_learning: bool,
    enable_parallel: bool,
    output: str | None,
):
    """Run advanced hyperparameter optimization for a specific algorithm."""

    async def run_optimization():
        console.print(
            "[bold green]Starting advanced hyperparameter optimization[/bold green]"
        )
        console.print(f"Dataset: {dataset_id}")
        console.print(f"Algorithm: {algorithm}")
        console.print(f"Objectives: {', '.join(objectives)}")
        console.print(f"Strategy: {strategy}")
        console.print(f"Trials: {n_trials}")
        console.print()

        # Create container and get enhanced AutoML service
        container = create_container()

        if not hasattr(container, "enhanced_automl_service"):
            console.print(
                "[red]Enhanced AutoML service not available. Using basic AutoML.[/red]"
            )
            if hasattr(container, "automl_service"):
                automl_service = container.automl_service()
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task("Optimizing hyperparameters...", total=None)
                    result = await automl_service.optimize_hyperparameters(
                        dataset_id, algorithm
                    )
            else:
                console.print("[red]No AutoML service available[/red]")
                return
        else:
            # Configure enhanced AutoML
            from pynomaly_detection.infrastructure.automl import (
                AcquisitionFunction,
                OptimizationStrategy,
            )

            strategy_enum = OptimizationStrategy(strategy)
            acquisition_enum = AcquisitionFunction(acquisition)

            config = EnhancedAutoMLConfig(
                optimization_strategy=strategy_enum,
                acquisition_function=acquisition_enum,
                n_trials=n_trials,
                max_optimization_time=timeout,
                enable_meta_learning=enable_meta_learning,
                enable_multi_objective=len(objectives) > 1,
                objectives=list(objectives),
                enable_parallel=enable_parallel,
            )

            enhanced_service = container.enhanced_automl_service()
            enhanced_service.config = config

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Running advanced optimization...", total=None)
                result = await enhanced_service.advanced_optimize_hyperparameters(
                    dataset_id, algorithm, list(objectives)
                )

        # Display results
        _display_optimization_result(result)

        # Save to file if requested
        if output:
            _save_result_to_file(result, output)
            console.print(f"[green]Results saved to {output}[/green]")

    asyncio.run(run_optimization())


@enhanced_automl_cli.command()
@require_feature("advanced_automl")
@click.argument("dataset_id")
@click.option(
    "--objectives",
    "-o",
    multiple=True,
    default=["auc", "training_time"],
    help="Optimization objectives",
)
@click.option("--max-algorithms", "-m", default=3, help="Maximum algorithms to try")
@click.option(
    "--strategy",
    "-s",
    default="bayesian",
    type=click.Choice(["bayesian", "hyperband", "bohb", "multi_objective"]),
    help="Optimization strategy",
)
@click.option("--n-trials", "-n", default=100, help="Number of trials per algorithm")
@click.option("--timeout", "-t", default=3600, help="Total optimization timeout")
@click.option(
    "--enable-ensemble", is_flag=True, default=True, help="Enable ensemble creation"
)
@click.option(
    "--enable-meta-learning", is_flag=True, default=True, help="Enable meta-learning"
)
@click.option("--output", "-f", help="Output file for results (JSON)")
def auto_optimize(
    dataset_id: str,
    objectives: list[str],
    max_algorithms: int,
    strategy: str,
    n_trials: int,
    timeout: int,
    enable_ensemble: bool,
    enable_meta_learning: bool,
    output: str | None,
):
    """Automatically select and optimize the best algorithms."""

    async def run_auto_optimization():
        console.print(
            "[bold green]Starting automatic algorithm selection and optimization[/bold green]"
        )
        console.print(f"Dataset: {dataset_id}")
        console.print(f"Objectives: {', '.join(objectives)}")
        console.print(f"Max algorithms: {max_algorithms}")
        console.print(f"Strategy: {strategy}")
        console.print()

        # Create container and get enhanced AutoML service
        container = create_container()

        if not hasattr(container, "enhanced_automl_service"):
            console.print(
                "[red]Enhanced AutoML service not available. Using basic AutoML.[/red]"
            )
            if hasattr(container, "automl_service"):
                automl_service = container.automl_service()
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task("Running AutoML...", total=None)
                    result = await automl_service.auto_select_and_optimize(
                        dataset_id, enable_ensemble=enable_ensemble
                    )
            else:
                console.print("[red]No AutoML service available[/red]")
                return
        else:
            # Configure enhanced AutoML
            from pynomaly_detection.infrastructure.automl import OptimizationStrategy

            strategy_enum = OptimizationStrategy(strategy)

            config = EnhancedAutoMLConfig(
                optimization_strategy=strategy_enum,
                n_trials=n_trials,
                max_optimization_time=timeout,
                enable_meta_learning=enable_meta_learning,
                enable_multi_objective=len(objectives) > 1,
                objectives=list(objectives),
                enable_ensemble_optimization=enable_ensemble,
            )

            enhanced_service = container.enhanced_automl_service()
            enhanced_service.config = config

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Running advanced AutoML...", total=None)
                result = await enhanced_service.auto_select_and_optimize_advanced(
                    dataset_id, list(objectives), max_algorithms, enable_ensemble
                )

        # Display results
        _display_automl_result(result)

        # Save to file if requested
        if output:
            _save_result_to_file(result, output)
            console.print(f"[green]Results saved to {output}[/green]")

    asyncio.run(run_auto_optimization())


@enhanced_automl_cli.command()
@require_feature("advanced_automl")
@click.argument("dataset_id")
@click.option(
    "--objectives",
    "-o",
    multiple=True,
    default=["auc", "precision", "training_time"],
    help="Multiple objectives to optimize",
)
@click.option(
    "--weights",
    "-w",
    help='Objective weights as JSON (e.g., \'{"auc": 0.5, "precision": 0.3, "training_time": 0.2}\')',
)
@click.option(
    "--algorithms", "-a", multiple=True, help="Specific algorithms to include"
)
@click.option("--n-trials", "-n", default=150, help="Number of optimization trials")
@click.option("--timeout", "-t", default=7200, help="Optimization timeout in seconds")
@click.option("--output", "-f", help="Output file for Pareto front (JSON)")
def multi_objective(
    dataset_id: str,
    objectives: list[str],
    weights: str | None,
    algorithms: list[str],
    n_trials: int,
    timeout: int,
    output: str | None,
):
    """Run multi-objective optimization to find Pareto optimal solutions."""

    async def run_multi_objective():
        console.print("[bold green]Starting multi-objective optimization[/bold green]")
        console.print(f"Dataset: {dataset_id}")
        console.print(f"Objectives: {', '.join(objectives)}")

        if weights:
            try:
                weight_dict = json.loads(weights)
                console.print(f"Weights: {weight_dict}")
            except json.JSONDecodeError:
                console.print("[red]Invalid weights JSON format[/red]")
                return
        else:
            # Equal weights
            weight_dict = {obj: 1.0 / len(objectives) for obj in objectives}
            console.print(f"Using equal weights: {weight_dict}")

        console.print()

        # Create container and get enhanced AutoML service
        container = create_container()

        if not hasattr(container, "enhanced_automl_service"):
            console.print(
                "[red]Enhanced AutoML service required for multi-objective optimization[/red]"
            )
            return

        # Configure for multi-objective optimization
        config = EnhancedAutoMLConfig(
            optimization_strategy="multi_objective",
            n_trials=n_trials,
            max_optimization_time=timeout,
            enable_multi_objective=True,
            objectives=list(objectives),
            objective_weights=weight_dict,
            enable_meta_learning=True,
        )

        enhanced_service = container.enhanced_automl_service()
        enhanced_service.config = config

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Finding Pareto optimal solutions...", total=None)

            if algorithms:
                # Optimize specified algorithms
                results = []
                for algorithm in algorithms:
                    try:
                        result = (
                            await enhanced_service.advanced_optimize_hyperparameters(
                                dataset_id, algorithm, list(objectives)
                            )
                        )
                        results.append(result)
                    except Exception as e:
                        console.print(
                            f"[yellow]Warning: Failed to optimize {algorithm}: {e}[/yellow]"
                        )

                if results:
                    best_result = max(results, key=lambda x: x.best_score)
                else:
                    console.print("[red]No algorithms could be optimized[/red]")
                    return
            else:
                # Auto-select and optimize
                best_result = await enhanced_service.auto_select_and_optimize_advanced(
                    dataset_id, list(objectives), max_algorithms=5
                )

        # Display multi-objective results
        _display_multi_objective_result(best_result, objectives)

        # Save Pareto front if requested
        if output and hasattr(best_result, "pareto_front") and best_result.pareto_front:
            _save_pareto_front(best_result.pareto_front, output)
            console.print(f"[green]Pareto front saved to {output}[/green]")

    asyncio.run(run_multi_objective())


@enhanced_automl_cli.command()
@require_feature("advanced_automl")
@click.argument("result_file", type=click.Path(exists=True))
def analyze(result_file: str):
    """Analyze optimization results and provide insights."""

    try:
        with open(result_file) as f:
            result_data = json.load(f)

        console.print("[bold green]Optimization Result Analysis[/bold green]")
        console.print()

        # Basic information
        info_table = Table(title="Optimization Summary")
        info_table.add_column("Metric", style="cyan")
        info_table.add_column("Value", style="magenta")

        info_table.add_row("Best Algorithm", result_data.get("best_algorithm", "N/A"))
        info_table.add_row("Best Score", f"{result_data.get('best_score', 0):.4f}")
        info_table.add_row(
            "Optimization Time", f"{result_data.get('optimization_time', 0):.2f}s"
        )
        info_table.add_row(
            "Trials Completed", str(result_data.get("trials_completed", 0))
        )

        if "optimization_strategy_used" in result_data:
            info_table.add_row(
                "Strategy Used", result_data["optimization_strategy_used"]
            )

        console.print(info_table)
        console.print()

        # Performance metrics
        if any(
            key in result_data
            for key in [
                "exploration_score",
                "exploitation_score",
                "convergence_stability",
            ]
        ):
            perf_table = Table(title="Optimization Quality Metrics")
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Score", style="magenta")
            perf_table.add_column("Interpretation", style="green")

            if "exploration_score" in result_data:
                score = result_data["exploration_score"]
                interpretation = _interpret_exploration_score(score)
                perf_table.add_row("Exploration", f"{score:.3f}", interpretation)

            if "exploitation_score" in result_data:
                score = result_data["exploitation_score"]
                interpretation = _interpret_exploitation_score(score)
                perf_table.add_row("Exploitation", f"{score:.3f}", interpretation)

            if "convergence_stability" in result_data:
                score = result_data["convergence_stability"]
                interpretation = _interpret_convergence_stability(score)
                perf_table.add_row(
                    "Convergence Stability", f"{score:.3f}", interpretation
                )

            console.print(perf_table)
            console.print()

        # Recommendations
        if "optimization_recommendations" in result_data:
            recommendations = result_data["optimization_recommendations"]
            if recommendations:
                console.print(
                    Panel(
                        "\n".join(f"• {rec}" for rec in recommendations),
                        title="[bold yellow]Optimization Recommendations[/bold yellow]",
                        border_style="yellow",
                    )
                )
                console.print()

        # Next steps
        if "next_steps" in result_data:
            next_steps = result_data["next_steps"]
            if next_steps:
                console.print(
                    Panel(
                        "\n".join(f"• {step}" for step in next_steps),
                        title="[bold blue]Suggested Next Steps[/bold blue]",
                        border_style="blue",
                    )
                )

    except Exception as e:
        console.print(f"[red]Error analyzing results: {e}[/red]")


def _display_optimization_result(result):
    """Display optimization result in a formatted table."""
    console.print("[bold green]✓ Optimization Complete[/bold green]")
    console.print()

    # Main results table
    table = Table(title="Optimization Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Best Algorithm", result.best_algorithm)
    table.add_row("Best Score", f"{result.best_score:.4f}")
    table.add_row("Optimization Time", f"{result.optimization_time:.2f}s")
    table.add_row("Trials Completed", str(result.trials_completed))

    if hasattr(result, "optimization_strategy_used"):
        table.add_row("Strategy", result.optimization_strategy_used)

    console.print(table)
    console.print()

    # Best parameters
    if result.best_params:
        console.print("[bold cyan]Best Parameters:[/bold cyan]")
        params_json = JSON.from_data(result.best_params)
        console.print(params_json)
        console.print()

    # Enhanced metrics if available
    if hasattr(result, "exploration_score"):
        metrics_table = Table(title="Optimization Quality")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Score", style="magenta")

        metrics_table.add_row("Exploration", f"{result.exploration_score:.3f}")
        metrics_table.add_row("Exploitation", f"{result.exploitation_score:.3f}")

        if hasattr(result, "convergence_stability"):
            metrics_table.add_row(
                "Convergence Stability", f"{result.convergence_stability:.3f}"
            )

        console.print(metrics_table)
        console.print()


def _display_automl_result(result):
    """Display AutoML result with algorithm rankings."""
    _display_optimization_result(result)

    # Algorithm rankings
    if result.algorithm_rankings:
        ranking_table = Table(title="Algorithm Performance Ranking")
        ranking_table.add_column("Rank", style="cyan")
        ranking_table.add_column("Algorithm", style="magenta")
        ranking_table.add_column("Score", style="green")

        for i, (algorithm, score) in enumerate(result.algorithm_rankings, 1):
            ranking_table.add_row(str(i), algorithm, f"{score:.4f}")

        console.print(ranking_table)
        console.print()

    # Ensemble information
    if result.ensemble_config:
        console.print("[bold cyan]Ensemble Configuration Available[/bold cyan]")
        console.print(f"Method: {result.ensemble_config.get('method', 'N/A')}")
        console.print(
            f"Algorithms: {len(result.ensemble_config.get('algorithms', []))}"
        )
        console.print()


def _display_multi_objective_result(result, objectives):
    """Display multi-objective optimization results."""
    console.print("[bold green]✓ Multi-Objective Optimization Complete[/bold green]")
    console.print()

    # Pareto front
    if hasattr(result, "pareto_front") and result.pareto_front:
        pareto_table = Table(title="Pareto Optimal Solutions")
        pareto_table.add_column("Solution", style="cyan")

        for obj in objectives:
            pareto_table.add_column(obj.upper(), style="magenta")

        for i, solution in enumerate(result.pareto_front[:5], 1):  # Show top 5
            row = [f"#{i}"]
            objectives_data = solution.get("objectives", {})
            for obj in objectives:
                value = objectives_data.get(obj, 0)
                row.append(f"{value:.4f}")
            pareto_table.add_row(*row)

        console.print(pareto_table)
        console.print()

        if len(result.pareto_front) > 5:
            console.print(
                f"[yellow]Showing top 5 of {len(result.pareto_front)} Pareto optimal solutions[/yellow]"
            )
            console.print()

    _display_optimization_result(result)


def _save_result_to_file(result, filename):
    """Save optimization result to file."""
    result_dict = {
        "best_algorithm": result.best_algorithm,
        "best_params": result.best_params,
        "best_score": result.best_score,
        "optimization_time": result.optimization_time,
        "trials_completed": result.trials_completed,
        "algorithm_rankings": result.algorithm_rankings,
    }

    # Add enhanced fields if available
    if hasattr(result, "optimization_strategy_used"):
        result_dict["optimization_strategy_used"] = result.optimization_strategy_used
    if hasattr(result, "exploration_score"):
        result_dict["exploration_score"] = result.exploration_score
    if hasattr(result, "exploitation_score"):
        result_dict["exploitation_score"] = result.exploitation_score
    if hasattr(result, "convergence_stability"):
        result_dict["convergence_stability"] = result.convergence_stability
    if hasattr(result, "optimization_recommendations"):
        result_dict["optimization_recommendations"] = (
            result.optimization_recommendations
        )
    if hasattr(result, "next_steps"):
        result_dict["next_steps"] = result.next_steps
    if hasattr(result, "pareto_front"):
        result_dict["pareto_front"] = result.pareto_front

    with open(filename, "w") as f:
        json.dump(result_dict, f, indent=2)


def _save_pareto_front(pareto_front, filename):
    """Save Pareto front to file."""
    with open(filename, "w") as f:
        json.dump(pareto_front, f, indent=2)


def _interpret_exploration_score(score):
    """Interpret exploration score."""
    if score >= 0.7:
        return "Excellent exploration"
    elif score >= 0.5:
        return "Good exploration"
    elif score >= 0.3:
        return "Moderate exploration"
    else:
        return "Poor exploration"


def _interpret_exploitation_score(score):
    """Interpret exploitation score."""
    if score >= 0.7:
        return "Strong convergence"
    elif score >= 0.5:
        return "Good convergence"
    elif score >= 0.3:
        return "Weak convergence"
    else:
        return "Poor convergence"


def _interpret_convergence_stability(score):
    """Interpret convergence stability score."""
    if score >= 0.8:
        return "Very stable"
    elif score >= 0.6:
        return "Stable"
    elif score >= 0.4:
        return "Moderately stable"
    else:
        return "Unstable"


if __name__ == "__main__":
    enhanced_automl_cli()
