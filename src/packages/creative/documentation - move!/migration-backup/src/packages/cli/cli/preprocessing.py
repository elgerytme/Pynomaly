"""Data preprocessing CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from monorepo.infrastructure.preprocessing.data_cleaner import (
    DataCleaner,
    MissingValueStrategy,
    OutlierStrategy,
)
from monorepo.infrastructure.preprocessing.data_transformer import (
    DataTransformer,
    EncodingStrategy,
    FeatureSelectionStrategy,
    ScalingStrategy,
)
from monorepo.infrastructure.preprocessing.preprocessing_pipeline import (
    PreprocessingPipeline,
)
from monorepo.presentation.cli.container import get_cli_container

app = typer.Typer()
console = Console()


@app.command("clean")
def clean_data(
    dataset_id: str = typer.Argument(..., help="Dataset ID to clean"),
    missing_strategy: str | None = typer.Option(
        "drop_rows", "--missing", help="Missing value strategy"
    ),
    outlier_strategy: str | None = typer.Option(
        "clip", "--outliers", help="Outlier handling strategy"
    ),
    remove_duplicates: bool = typer.Option(
        True, "--duplicates/--keep-duplicates", help="Remove duplicate rows"
    ),
    handle_zeros: str | None = typer.Option(
        "keep", "--zeros", help="Zero value handling: keep, remove, replace"
    ),
    handle_infinite: str | None = typer.Option(
        "remove", "--infinite", help="Infinite value handling: remove, replace, clip"
    ),
    fill_value: str | None = typer.Option(
        None, "--fill-value", help="Value for constant fill strategy"
    ),
    outlier_threshold: float | None = typer.Option(
        3.0, "--outlier-threshold", help="Z-score threshold for outliers"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done without applying changes"
    ),
    save_as: str | None = typer.Option(
        None, "--save-as", help="Save cleaned data as new dataset"
    ),
    output_format: str = typer.Option(
        "csv", "--format", help="Output format: csv, parquet, json"
    ),
):
    """Clean dataset by handling missing values, outliers, and duplicates."""
    container = get_cli_container()
    dataset_repo = container.dataset_repository()

    # Get dataset
    dataset = dataset_repo.find_by_id(dataset_id)
    if not dataset:
        console.print(f"[red]Error:[/red] Dataset with ID '{dataset_id}' not found")
        raise typer.Exit(1)

    console.print(f"[blue]Cleaning dataset:[/blue] {dataset.name}")
    console.print(
        f"Original shape: {dataset.n_samples:,} rows × {dataset.n_features} columns"
    )

    # Initialize cleaner
    cleaner = DataCleaner()

    # Get data
    df = dataset.to_dataframe()
    original_shape = df.shape

    if dry_run:
        console.print("\n[yellow]DRY RUN MODE - No changes will be applied[/yellow]")

    # Apply cleaning steps
    cleaning_steps = []

    try:
        # Handle missing values
        if missing_strategy and missing_strategy != "none":
            try:
                strategy = MissingValueStrategy(missing_strategy)
                if dry_run:
                    missing_count = df.isnull().sum().sum()
                    console.print(
                        f"Would handle {missing_count:,} missing values using {strategy.value}"
                    )
                    cleaning_steps.append(f"Missing values: {strategy.value}")
                else:
                    with console.status("Handling missing values..."):
                        df = cleaner.handle_missing_values(
                            df, strategy=strategy, fill_value=fill_value
                        )
                    console.print(f"✓ Handled missing values using {strategy.value}")
            except ValueError:
                console.print(
                    f"[red]Error:[/red] Invalid missing value strategy: {missing_strategy}"
                )
                raise typer.Exit(1)

        # Handle outliers
        if outlier_strategy and outlier_strategy != "none":
            try:
                strategy = OutlierStrategy(outlier_strategy)
                if dry_run:
                    # Estimate outliers
                    numeric_cols = df.select_dtypes(include=["number"]).columns
                    if len(numeric_cols) > 0:
                        z_scores = abs(
                            (df[numeric_cols] - df[numeric_cols].mean())
                            / df[numeric_cols].std()
                        )
                        outlier_count = (z_scores > outlier_threshold).sum().sum()
                        console.print(
                            f"Would handle ~{outlier_count:,} outliers using {strategy.value}"
                        )
                        cleaning_steps.append(f"Outliers: {strategy.value}")
                else:
                    with console.status("Handling outliers..."):
                        df = cleaner.handle_outliers(
                            df, strategy=strategy, threshold=outlier_threshold
                        )
                    console.print(f"✓ Handled outliers using {strategy.value}")
            except ValueError:
                console.print(
                    f"[red]Error:[/red] Invalid outlier strategy: {outlier_strategy}"
                )
                raise typer.Exit(1)

        # Remove duplicates
        if remove_duplicates:
            if dry_run:
                duplicate_count = df.duplicated().sum()
                console.print(f"Would remove {duplicate_count:,} duplicate rows")
                cleaning_steps.append(f"Duplicates: remove {duplicate_count:,} rows")
            else:
                original_rows = len(df)
                df = cleaner.remove_duplicates(df)
                removed = original_rows - len(df)
                if removed > 0:
                    console.print(f"✓ Removed {removed:,} duplicate rows")

        # Handle zero values
        if handle_zeros and handle_zeros != "keep":
            if dry_run:
                zero_count = (df == 0).sum().sum()
                console.print(
                    f"Would handle {zero_count:,} zero values using {handle_zeros}"
                )
                cleaning_steps.append(f"Zero values: {handle_zeros}")
            else:
                with console.status("Handling zero values..."):
                    df = cleaner.handle_zero_values(df, strategy=handle_zeros)
                console.print(f"✓ Handled zero values using {handle_zeros}")

        # Handle infinite values
        if handle_infinite and handle_infinite != "keep":
            if dry_run:
                inf_count = np.isinf(df.select_dtypes(include=["number"])).sum().sum()
                console.print(
                    f"Would handle {inf_count:,} infinite values using {handle_infinite}"
                )
                cleaning_steps.append(f"Infinite values: {handle_infinite}")
            else:
                with console.status("Handling infinite values..."):
                    df = cleaner.handle_infinite_values(df, strategy=handle_infinite)
                console.print(f"✓ Handled infinite values using {handle_infinite}")

        # Show results
        final_shape = df.shape
        console.print("\n[green]Cleaning completed![/green]")
        console.print(
            f"Final shape: {final_shape[0]:,} rows × {final_shape[1]} columns"
        )

        if not dry_run:
            rows_removed = original_shape[0] - final_shape[0]
            cols_removed = original_shape[1] - final_shape[1]

            if rows_removed > 0:
                console.print(
                    f"Rows removed: {rows_removed:,} ({rows_removed / original_shape[0] * 100:.1f}%)"
                )
            if cols_removed > 0:
                console.print(f"Columns removed: {cols_removed}")

            # Save results
            if save_as:
                # Create new dataset
                import uuid
                from datetime import datetime

                from monorepo.domain.entities import Dataset

                new_dataset = Dataset(
                    id=str(uuid.uuid4()),
                    name=save_as,
                    description=f"Cleaned version of {dataset.name}",
                    data=df,
                    target_column=dataset.target_column,
                    created_at=datetime.now(),
                )

                dataset_repo.save(new_dataset)
                console.print(
                    f"✓ Saved cleaned dataset as '{save_as}' (ID: {new_dataset.id[:8]})"
                )
            else:
                # Update existing dataset
                dataset.data = df
                dataset.n_samples = len(df)
                dataset.n_features = len(df.columns)
                if dataset.target_column and dataset.target_column not in df.columns:
                    dataset.target_column = None
                    dataset.has_target = False

                dataset_repo.save(dataset)
                console.print(f"✓ Updated dataset '{dataset.name}'")
        else:
            console.print("\n[yellow]Summary of planned changes:[/yellow]")
            for step in cleaning_steps:
                console.print(f"  • {step}")

    except Exception as e:
        console.print(f"[red]Error during cleaning:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("transform")
def transform_data(
    dataset_id: str = typer.Argument(..., help="Dataset ID to transform"),
    scaling: str | None = typer.Option(
        None, "--scaling", help="Feature scaling strategy"
    ),
    encoding: str | None = typer.Option(
        None, "--encoding", help="Categorical encoding strategy"
    ),
    feature_selection: str | None = typer.Option(
        None, "--feature-selection", help="Feature selection strategy"
    ),
    polynomial_degree: int | None = typer.Option(
        None, "--polynomial", help="Generate polynomial features"
    ),
    normalize_column_names: bool = typer.Option(
        False, "--normalize-names", help="Normalize column names"
    ),
    optimize_dtypes: bool = typer.Option(
        False, "--optimize-dtypes", help="Optimize data types"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done without applying changes"
    ),
    save_as: str | None = typer.Option(
        None, "--save-as", help="Save transformed data as new dataset"
    ),
    exclude_columns: str | None = typer.Option(
        None, "--exclude", help="Comma-separated columns to exclude"
    ),
):
    """Transform dataset features with scaling, encoding, and feature engineering."""
    container = get_cli_container()
    dataset_repo = container.dataset_repository()

    # Get dataset
    dataset = dataset_repo.find_by_id(dataset_id)
    if not dataset:
        console.print(f"[red]Error:[/red] Dataset with ID '{dataset_id}' not found")
        raise typer.Exit(1)

    console.print(f"[blue]Transforming dataset:[/blue] {dataset.name}")
    console.print(
        f"Original shape: {dataset.n_samples:,} rows × {dataset.n_features} columns"
    )

    # Initialize transformer
    transformer = DataTransformer()

    # Get data
    df = dataset.to_dataframe()
    original_shape = df.shape

    # Parse excluded columns
    excluded_cols = []
    if exclude_columns:
        excluded_cols = [col.strip() for col in exclude_columns.split(",")]
        console.print(f"Excluding columns: {excluded_cols}")

    if dry_run:
        console.print("\n[yellow]DRY RUN MODE - No changes will be applied[/yellow]")

    # Apply transformations
    transformation_steps = []

    try:
        # Feature scaling
        if scaling and scaling != "none":
            try:
                strategy = ScalingStrategy(scaling)
                numeric_cols = df.select_dtypes(include=["number"]).columns
                target_cols = [col for col in numeric_cols if col not in excluded_cols]

                if dry_run:
                    console.print(
                        f"Would apply {strategy.value} scaling to {len(target_cols)} numeric columns"
                    )
                    transformation_steps.append(
                        f"Scaling: {strategy.value} on {len(target_cols)} columns"
                    )
                else:
                    with console.status(f"Applying {strategy.value} scaling..."):
                        df = transformer.scale_features(
                            df, strategy=strategy, columns=target_cols
                        )
                    console.print(
                        f"✓ Applied {strategy.value} scaling to {len(target_cols)} columns"
                    )
            except ValueError:
                console.print(f"[red]Error:[/red] Invalid scaling strategy: {scaling}")
                raise typer.Exit(1)

        # Categorical encoding
        if encoding and encoding != "none":
            try:
                strategy = EncodingStrategy(encoding)
                categorical_cols = df.select_dtypes(
                    include=["object", "category"]
                ).columns
                target_cols = [
                    col for col in categorical_cols if col not in excluded_cols
                ]

                if dry_run:
                    console.print(
                        f"Would apply {strategy.value} encoding to {len(target_cols)} categorical columns"
                    )
                    transformation_steps.append(
                        f"Encoding: {strategy.value} on {len(target_cols)} columns"
                    )
                else:
                    with console.status(f"Applying {strategy.value} encoding..."):
                        df = transformer.encode_categorical(
                            df, strategy=strategy, columns=target_cols
                        )
                    console.print(
                        f"✓ Applied {strategy.value} encoding to {len(target_cols)} columns"
                    )
            except ValueError:
                console.print(
                    f"[red]Error:[/red] Invalid encoding strategy: {encoding}"
                )
                raise typer.Exit(1)

        # Feature selection
        if feature_selection and feature_selection != "none":
            try:
                strategy = FeatureSelectionStrategy(feature_selection)
                if dry_run:
                    console.print(f"Would apply {strategy.value} feature selection")
                    transformation_steps.append(f"Feature selection: {strategy.value}")
                else:
                    original_features = len(df.columns)
                    with console.status(
                        f"Applying {strategy.value} feature selection..."
                    ):
                        df = transformer.select_features(df, strategy=strategy)
                    selected_features = len(df.columns)
                    console.print(
                        f"✓ Selected {selected_features}/{original_features} features using {strategy.value}"
                    )
            except ValueError:
                console.print(
                    f"[red]Error:[/red] Invalid feature selection strategy: {feature_selection}"
                )
                raise typer.Exit(1)

        # Polynomial features
        if polynomial_degree and polynomial_degree > 1:
            if dry_run:
                numeric_cols = df.select_dtypes(include=["number"]).columns
                estimated_features = len(numeric_cols) ** polynomial_degree
                console.print(
                    f"Would generate polynomial features (degree {polynomial_degree})"
                )
                console.print(f"Estimated new feature count: ~{estimated_features}")
                transformation_steps.append(
                    f"Polynomial features: degree {polynomial_degree}"
                )
            else:
                original_features = len(df.columns)
                with console.status(
                    f"Generating polynomial features (degree {polynomial_degree})..."
                ):
                    df = transformer.create_polynomial_features(
                        df, degree=polynomial_degree
                    )
                new_features = len(df.columns)
                console.print(
                    f"✓ Generated {new_features - original_features} polynomial features"
                )

        # Normalize column names
        if normalize_column_names:
            if dry_run:
                console.print(
                    "Would normalize column names (lowercase, underscore-separated)"
                )
                transformation_steps.append("Column names: normalize")
            else:
                df.columns = [
                    col.lower().replace(" ", "_").replace("-", "_")
                    for col in df.columns
                ]
                console.print("✓ Normalized column names")

        # Optimize data types
        if optimize_dtypes:
            if dry_run:
                console.print("Would optimize data types for memory efficiency")
                transformation_steps.append("Data types: optimize")
            else:
                original_memory = df.memory_usage(deep=True).sum()
                with console.status("Optimizing data types..."):
                    df = transformer.optimize_dtypes(df)
                new_memory = df.memory_usage(deep=True).sum()
                memory_saved = (original_memory - new_memory) / original_memory * 100
                console.print(
                    f"✓ Optimized data types (saved {memory_saved:.1f}% memory)"
                )

        # Show results
        final_shape = df.shape
        console.print("\n[green]Transformation completed![/green]")
        console.print(
            f"Final shape: {final_shape[0]:,} rows × {final_shape[1]} columns"
        )

        if not dry_run:
            features_added = final_shape[1] - original_shape[1]
            if features_added > 0:
                console.print(f"Features added: {features_added}")
            elif features_added < 0:
                console.print(f"Features removed: {abs(features_added)}")

            # Save results
            if save_as:
                # Create new dataset
                import uuid
                from datetime import datetime

                from monorepo.domain.entities import Dataset

                new_dataset = Dataset(
                    id=str(uuid.uuid4()),
                    name=save_as,
                    description=f"Transformed version of {dataset.name}",
                    data=df,
                    target_column=(
                        dataset.target_column
                        if dataset.target_column in df.columns
                        else None
                    ),
                    created_at=datetime.now(),
                )

                dataset_repo.save(new_dataset)
                console.print(
                    f"✓ Saved transformed dataset as '{save_as}' (ID: {new_dataset.id[:8]})"
                )
            else:
                # Update existing dataset
                dataset.data = df
                dataset.n_samples = len(df)
                dataset.n_features = len(df.columns)
                if dataset.target_column and dataset.target_column not in df.columns:
                    dataset.target_column = None
                    dataset.has_target = False

                dataset_repo.save(dataset)
                console.print(f"✓ Updated dataset '{dataset.name}'")
        else:
            console.print("\n[yellow]Summary of planned transformations:[/yellow]")
            for step in transformation_steps:
                console.print(f"  • {step}")

    except Exception as e:
        console.print(f"[red]Error during transformation:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("pipeline")
def manage_pipeline(
    action: str = typer.Argument(
        ..., help="Action: create, list, show, apply, save, load, delete"
    ),
    name: str | None = typer.Option(None, "--name", help="Pipeline name"),
    dataset_id: str | None = typer.Option(None, "--dataset", help="Dataset ID"),
    config_file: Path | None = typer.Option(
        None, "--config", help="Pipeline configuration file"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", help="Output file for save operations"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done without applying changes"
    ),
):
    """Manage preprocessing pipelines."""
    get_cli_container()

    if action == "create":
        _create_pipeline(name, config_file)
    elif action == "list":
        _list_pipelines()
    elif action == "show":
        _show_pipeline(name)
    elif action == "apply":
        _apply_pipeline(name, dataset_id, dry_run)
    elif action == "save":
        _save_pipeline(name, output_file)
    elif action == "load":
        _load_pipeline(config_file, name)
    elif action == "delete":
        _delete_pipeline(name)
    else:
        console.print(f"[red]Error:[/red] Unknown action '{action}'")
        console.print(
            "Available actions: create, list, show, apply, save, load, delete"
        )
        raise typer.Exit(1)


def _create_pipeline(name: str | None, config_file: Path | None):
    """Create a new preprocessing pipeline."""
    if not name:
        name = typer.prompt("Pipeline name")

    pipeline = PreprocessingPipeline(name=name)

    if config_file and config_file.exists():
        # Load from config file
        try:
            with open(config_file) as f:
                config = json.load(f)

            for step_config in config.get("steps", []):
                pipeline.add_step(
                    name=step_config["name"],
                    operation=step_config["operation"],
                    parameters=step_config.get("parameters", {}),
                    enabled=step_config.get("enabled", True),
                    description=step_config.get("description"),
                )

            console.print(f"✓ Created pipeline '{name}' from {config_file}")
            console.print(f"Steps loaded: {len(pipeline.steps)}")

        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to load config: {str(e)}")
            raise typer.Exit(1)
    else:
        # Interactive creation
        console.print(f"Creating preprocessing pipeline: {name}")
        console.print("Add preprocessing steps (press Enter without input to finish):")

        while True:
            step_name = typer.prompt("Step name (or Enter to finish)", default="")
            if not step_name:
                break

            operation = typer.prompt("Operation")
            description = typer.prompt("Description (optional)", default="")

            pipeline.add_step(
                name=step_name, operation=operation, description=description or None
            )

            console.print(f"✓ Added step: {step_name}")

        console.print(f"✓ Created pipeline '{name}' with {len(pipeline.steps)} steps")

    # Store pipeline (in a real app, would persist this)
    _store_pipeline(pipeline)


def _list_pipelines():
    """List all available pipelines."""
    pipelines = _get_stored_pipelines()

    if not pipelines:
        console.print("No pipelines found.")
        return

    table = Table(title="Preprocessing Pipelines")
    table.add_column("Name", style="cyan")
    table.add_column("Steps", style="yellow")
    table.add_column("Status", style="green")

    for pipeline in pipelines:
        enabled_steps = sum(1 for step in pipeline.steps if step.enabled)
        status = "✓ Fitted" if pipeline._fitted else "○ Not fitted"

        table.add_row(pipeline.name, f"{enabled_steps}/{len(pipeline.steps)}", status)

    console.print(table)


def _show_pipeline(name: str | None):
    """Show details of a specific pipeline."""
    if not name:
        console.print("[red]Error:[/red] Pipeline name required")
        raise typer.Exit(1)

    pipeline = _get_pipeline_by_name(name)
    if not pipeline:
        console.print(f"[red]Error:[/red] Pipeline '{name}' not found")
        raise typer.Exit(1)

    console.print(f"[bold blue]Pipeline:[/bold blue] {pipeline.name}")
    console.print(f"Status: {'✓ Fitted' if pipeline._fitted else '○ Not fitted'}")
    console.print(f"Steps: {len(pipeline.steps)}")

    if pipeline.steps:
        console.print("\n[bold]Steps:[/bold]")
        table = Table()
        table.add_column("#", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("Operation", style="green")
        table.add_column("Enabled", style="yellow")
        table.add_column("Description", style="dim")

        for i, step in enumerate(pipeline.steps, 1):
            table.add_row(
                str(i),
                step.name,
                step.operation,
                "✓" if step.enabled else "✗",
                step.description or "",
            )

        console.print(table)


def _apply_pipeline(name: str | None, dataset_id: str | None, dry_run: bool):
    """Apply a pipeline to a dataset."""
    if not name:
        console.print("[red]Error:[/red] Pipeline name required")
        raise typer.Exit(1)

    if not dataset_id:
        console.print("[red]Error:[/red] Dataset ID required")
        raise typer.Exit(1)

    pipeline = _get_pipeline_by_name(name)
    if not pipeline:
        console.print(f"[red]Error:[/red] Pipeline '{name}' not found")
        raise typer.Exit(1)

    container = get_cli_container()
    dataset_repo = container.dataset_repository()

    dataset = dataset_repo.find_by_id(dataset_id)
    if not dataset:
        console.print(f"[red]Error:[/red] Dataset with ID '{dataset_id}' not found")
        raise typer.Exit(1)

    console.print(
        f"[blue]Applying pipeline '{name}' to dataset '{dataset.name}'[/blue]"
    )

    if dry_run:
        console.print("\n[yellow]DRY RUN MODE - No changes will be applied[/yellow]")
        console.print("Pipeline steps that would be executed:")

        for i, step in enumerate(pipeline.steps, 1):
            if step.enabled:
                console.print(f"  {i}. {step.name} ({step.operation})")
                if step.description:
                    console.print(f"     {step.description}")
    else:
        try:
            df = dataset.to_dataframe()
            original_shape = df.shape

            # Apply pipeline
            with console.status("Applying preprocessing pipeline..."):
                transformed_df = pipeline.fit_transform(df)

            # Update dataset
            dataset.data = transformed_df
            dataset.n_samples = len(transformed_df)
            dataset.n_features = len(transformed_df.columns)

            dataset_repo.save(dataset)

            console.print("✓ Pipeline applied successfully")
            console.print(
                f"Shape: {original_shape[0]:,}×{original_shape[1]} → {len(transformed_df):,}×{len(transformed_df.columns)}"
            )

        except Exception as e:
            console.print(f"[red]Error:[/red] Pipeline failed: {str(e)}")
            raise typer.Exit(1)


def _save_pipeline(name: str | None, output_file: Path | None):
    """Save a pipeline to file."""
    if not name:
        console.print("[red]Error:[/red] Pipeline name required")
        raise typer.Exit(1)

    pipeline = _get_pipeline_by_name(name)
    if not pipeline:
        console.print(f"[red]Error:[/red] Pipeline '{name}' not found")
        raise typer.Exit(1)

    if not output_file:
        output_file = Path(f"{name}_pipeline.json")

    try:
        config = {
            "name": pipeline.name,
            "steps": [
                {
                    "name": step.name,
                    "operation": step.operation,
                    "parameters": step.parameters,
                    "enabled": step.enabled,
                    "description": step.description,
                }
                for step in pipeline.steps
            ],
        }

        with open(output_file, "w") as f:
            json.dump(config, f, indent=2)

        console.print(f"✓ Pipeline '{name}' saved to {output_file}")

    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to save pipeline: {str(e)}")
        raise typer.Exit(1)


def _load_pipeline(config_file: Path | None, name: str | None):
    """Load a pipeline from file."""
    if not config_file:
        console.print("[red]Error:[/red] Config file required")
        raise typer.Exit(1)

    if not config_file.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config_file}")
        raise typer.Exit(1)

    try:
        with open(config_file) as f:
            config = json.load(f)

        pipeline_name = name or config.get("name", "loaded_pipeline")
        pipeline = PreprocessingPipeline(name=pipeline_name)

        for step_config in config.get("steps", []):
            pipeline.add_step(
                name=step_config["name"],
                operation=step_config["operation"],
                parameters=step_config.get("parameters", {}),
                enabled=step_config.get("enabled", True),
                description=step_config.get("description"),
            )

        _store_pipeline(pipeline)

        console.print(f"✓ Loaded pipeline '{pipeline_name}' from {config_file}")
        console.print(f"Steps loaded: {len(pipeline.steps)}")

    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to load pipeline: {str(e)}")
        raise typer.Exit(1)


def _delete_pipeline(name: str | None):
    """Delete a pipeline."""
    if not name:
        console.print("[red]Error:[/red] Pipeline name required")
        raise typer.Exit(1)

    pipeline = _get_pipeline_by_name(name)
    if not pipeline:
        console.print(f"[red]Error:[/red] Pipeline '{name}' not found")
        raise typer.Exit(1)

    if typer.confirm(f"Delete pipeline '{name}'?"):
        _remove_pipeline(name)
        console.print(f"✓ Pipeline '{name}' deleted")
    else:
        console.print("Deletion cancelled")


# Helper functions for pipeline storage (in a real app, would use proper persistence)
_pipelines = {}


def _store_pipeline(pipeline: PreprocessingPipeline):
    """Store a pipeline in memory."""
    _pipelines[pipeline.name] = pipeline


def _get_stored_pipelines() -> list[PreprocessingPipeline]:
    """Get all stored pipelines."""
    return list(_pipelines.values())


def _get_pipeline_by_name(name: str) -> PreprocessingPipeline | None:
    """Get a pipeline by name."""
    return _pipelines.get(name)


def _remove_pipeline(name: str):
    """Remove a pipeline."""
    if name in _pipelines:
        del _pipelines[name]


if __name__ == "__main__":
    app()
