"""Explainability commands for Typer CLI."""

import typer
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...domain.services.explainability_service import ExplainabilityService, ExplainerType
from ...domain.services.detection_service import DetectionService
from ...infrastructure.repositories.model_repository import ModelRepository
from ...infrastructure.logging import get_logger

console = Console()
logger = get_logger(__name__)

app = typer.Typer(help="Model explainability and interpretability commands")


@app.command()
def sample(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input data file"),
    sample_index: int = typer.Option(0, "--sample", "-s", help="Sample index to explain"),
    algorithm: str = typer.Option("isolation_forest", "--algorithm", "-a", help="Detection algorithm"),
    explainer: str = typer.Option("feature_importance", "--explainer", "-e", 
                                 help="Explainer type (shap, lime, permutation, feature_importance)"),
    model_id: Optional[str] = typer.Option(None, "--model-id", "-m", help="Use saved model by ID"),
    models_dir: Path = typer.Option(Path("models"), "--models-dir", "-d", help="Models directory"),
    contamination: float = typer.Option(0.1, "--contamination", "-c", help="Contamination rate"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for explanation"),
    top_features: int = typer.Option(5, "--top-features", "-t", help="Number of top features to show"),
) -> None:
    """Explain a single sample prediction."""
    
    print(f"[blue]🔍[/blue] Explaining prediction for sample {sample_index}")
    
    try:
        # Load data
        if not input_file.exists():
            print(f"[red]✗[/red] Input file '{input_file}' not found")
            raise typer.Exit(1)
        
        if input_file.suffix.lower() == '.csv':
            df = pd.read_csv(input_file)
        elif input_file.suffix.lower() == '.json':
            df = pd.read_json(input_file)
        elif input_file.suffix.lower() == '.parquet':
            df = pd.read_parquet(input_file)
        else:
            print(f"[red]✗[/red] Unsupported file format '{input_file.suffix}'")
            raise typer.Exit(1)
        
        if sample_index >= len(df):
            print(f"[red]✗[/red] Sample index {sample_index} out of range (0-{len(df)-1})")
            raise typer.Exit(1)
        
        print(f"[green]✓[/green] Loaded data with {len(df)} samples, {len(df.columns)} features")
        
        # Get sample to explain
        sample_data = df.iloc[sample_index].values.astype(np.float64)
        feature_names = df.columns.tolist()
        
        # Initialize services
        detection_service = DetectionService()
        explainability_service = ExplainabilityService(detection_service)
        
        # Algorithm mapping
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        mapped_algorithm = algorithm_map.get(algorithm, algorithm)
        
        # Load model or fit new one
        if model_id:
            try:
                repo = ModelRepository(str(models_dir))
                model = repo.load(model_id)
                detection_service._fitted_models[mapped_algorithm] = model.model_object
                print(f"[green]✓[/green] Loaded saved model: {model_id}")
            except Exception as e:
                print(f"[red]✗[/red] Failed to load model {model_id}: {e}")
                raise typer.Exit(1)
        else:
            # Fit model on the data
            print(f"[blue]ℹ[/blue] Fitting {algorithm} model...")
            data_array = df.values.astype(np.float64)
            detection_service.fit(data_array, mapped_algorithm, contamination=contamination)
            print(f"[green]✓[/green] Model fitted successfully")
        
        # Map explainer type
        explainer_type_map = {
            'shap': ExplainerType.SHAP,
            'lime': ExplainerType.LIME,
            'permutation': ExplainerType.PERMUTATION,
            'feature_importance': ExplainerType.FEATURE_IMPORTANCE
        }
        explainer_enum = explainer_type_map.get(explainer, ExplainerType.FEATURE_IMPORTANCE)
        
        # Generate explanation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Generating {explainer} explanation...", total=None)
            
            training_data = df.values.astype(np.float64) if explainer == 'lime' else None
            
            explanation = explainability_service.explain_prediction(
                sample=sample_data,
                algorithm=mapped_algorithm,
                explainer_type=explainer_enum,
                training_data=training_data,
                feature_names=feature_names
            )
            
            progress.update(task, completed=True)
        
        # Display results
        print(f"\n[green]✅[/green] Explanation generated successfully!")
        
        # Prediction info
        pred_table = Table(title="[bold blue]Prediction Information[/bold blue]")
        pred_table.add_column("Property", style="cyan")
        pred_table.add_column("Value", style="green")
        
        pred_table.add_row("Sample Index", str(sample_index))
        pred_table.add_row("Algorithm", algorithm)
        pred_table.add_row("Explainer", explainer)
        pred_table.add_row("Is Anomaly", "[red]Yes[/red]" if explanation.is_anomaly else "[green]No[/green]")
        if explanation.prediction_confidence is not None:
            pred_table.add_row("Confidence", f"{explanation.prediction_confidence:.3f}")
        if explanation.base_value is not None:
            pred_table.add_row("Base Value", f"{explanation.base_value:.3f}")
        
        console.print(pred_table)
        
        # Top features
        if explanation.top_features:
            features_table = Table(title=f"[bold blue]Top {len(explanation.top_features)} Contributing Features[/bold blue]")
            features_table.add_column("Rank", style="yellow", justify="center")
            features_table.add_column("Feature", style="cyan")
            features_table.add_column("Value", style="blue", justify="right")
            features_table.add_column("Importance", style="green", justify="right")
            
            for feature in explanation.top_features:
                features_table.add_row(
                    str(feature["rank"]),
                    feature["feature_name"],
                    f"{feature['value']:.3f}",
                    f"{feature['importance']:.3f}"
                )
            
            console.print(features_table)
        
        # Feature importance summary
        if explanation.feature_importance:
            print(f"\n[bold blue]All Feature Importance:[/bold blue]")
            
            # Sort by importance
            sorted_features = sorted(
                explanation.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for i, (feature_name, importance) in enumerate(sorted_features[:top_features]):
                feature_value = explanation.data_sample[i] if i < len(explanation.data_sample) else 0.0
                print(f"  {i+1:2d}. {feature_name:20s}: {importance:6.3f} (value: {feature_value:8.3f})")
        
        # Sample values
        print(f"\n[bold blue]Sample Values:[/bold blue]")
        sample_panel_content = ""
        for i, (name, value) in enumerate(zip(feature_names, explanation.data_sample)):
            sample_panel_content += f"{name}: {value:.3f}\n"
        
        console.print(Panel(sample_panel_content.strip(), title="Sample Data"))
        
        # Save explanation if requested
        if output_file:
            explanation_data = {
                'sample_index': sample_index,
                'algorithm': algorithm,
                'explainer_type': explainer,
                'prediction': {
                    'is_anomaly': explanation.is_anomaly,
                    'confidence': explanation.prediction_confidence
                },
                'feature_importance': explanation.feature_importance,
                'top_features': explanation.top_features,
                'sample_data': dict(zip(feature_names, explanation.data_sample)),
                'metadata': explanation.metadata
            }
            
            with open(output_file, 'w') as f:
                json.dump(explanation_data, f, indent=2)
            
            print(f"[green]✅[/green] Explanation saved to: {output_file}")
            
    except Exception as e:
        print(f"[red]✗[/red] Explanation failed: {e}")
        raise typer.Exit(1)


@app.command()
def batch(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input data file"),
    algorithm: str = typer.Option("isolation_forest", "--algorithm", "-a", help="Detection algorithm"),
    explainer: str = typer.Option("feature_importance", "--explainer", "-e", help="Explainer type"),
    max_samples: int = typer.Option(10, "--max-samples", "-n", help="Maximum samples to explain"),
    contamination: float = typer.Option(0.1, "--contamination", "-c", help="Contamination rate"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for explanations"),
    anomalies_only: bool = typer.Option(False, "--anomalies-only", help="Only explain detected anomalies"),
) -> None:
    """Explain multiple sample predictions."""
    
    print(f"[blue]🔍[/blue] Explaining batch predictions")
    
    try:
        # Load data
        if not input_file.exists():
            print(f"[red]✗[/red] Input file '{input_file}' not found")
            raise typer.Exit(1)
        
        if input_file.suffix.lower() == '.csv':
            df = pd.read_csv(input_file)
        elif input_file.suffix.lower() == '.json':
            df = pd.read_json(input_file)
        else:
            print(f"[red]✗[/red] Unsupported file format '{input_file.suffix}'")
            raise typer.Exit(1)
        
        print(f"[green]✓[/green] Loaded data with {len(df)} samples, {len(df.columns)} features")
        
        # Initialize services
        detection_service = DetectionService()
        explainability_service = ExplainabilityService(detection_service)
        
        # Algorithm mapping
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        mapped_algorithm = algorithm_map.get(algorithm, algorithm)
        
        # Fit model
        print(f"[blue]ℹ[/blue] Fitting {algorithm} model...")
        data_array = df.values.astype(np.float64)
        detection_service.fit(data_array, mapped_algorithm, contamination=contamination)
        print(f"[green]✓[/green] Model fitted successfully")
        
        # Get predictions to filter anomalies if requested
        samples_to_explain = data_array[:max_samples]
        sample_indices = list(range(min(max_samples, len(data_array))))
        
        if anomalies_only:
            # Get predictions for all samples
            result = detection_service.detect_anomalies(data_array, mapped_algorithm, contamination)
            anomaly_indices = np.where(result.predictions == -1)[0]
            
            # Filter to anomalies only
            sample_indices = anomaly_indices[:max_samples].tolist()
            samples_to_explain = data_array[sample_indices]
            
            print(f"[blue]ℹ[/blue] Found {len(anomaly_indices)} anomalies, explaining {len(sample_indices)}")
        
        if len(samples_to_explain) == 0:
            print(f"[yellow]⚠[/yellow] No samples to explain")
            return
        
        # Map explainer type
        explainer_type_map = {
            'shap': ExplainerType.SHAP,
            'lime': ExplainerType.LIME,
            'permutation': ExplainerType.PERMUTATION,
            'feature_importance': ExplainerType.FEATURE_IMPORTANCE
        }
        explainer_enum = explainer_type_map.get(explainer, ExplainerType.FEATURE_IMPORTANCE)
        
        # Generate explanations
        feature_names = df.columns.tolist()
        explanations = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Generating {explainer} explanations...", total=len(samples_to_explain))
            
            for i, (sample_idx, sample) in enumerate(zip(sample_indices, samples_to_explain)):
                training_data = data_array if explainer == 'lime' else None
                
                explanation = explainability_service.explain_prediction(
                    sample=sample,
                    algorithm=mapped_algorithm,
                    explainer_type=explainer_enum,
                    training_data=training_data,
                    feature_names=feature_names
                )
                
                explanations.append((sample_idx, explanation))
                progress.update(task, advance=1)
        
        # Display results
        print(f"\n[green]✅[/green] Generated {len(explanations)} explanations!")
        
        # Summary table
        summary_table = Table(title="[bold blue]Batch Explanation Summary[/bold blue]")
        summary_table.add_column("Sample", style="yellow", justify="center")
        summary_table.add_column("Anomaly", style="red", justify="center")
        summary_table.add_column("Confidence", style="blue", justify="right")
        summary_table.add_column("Top Feature", style="cyan")
        summary_table.add_column("Importance", style="green", justify="right")
        
        for sample_idx, explanation in explanations:
            is_anomaly = "Yes" if explanation.is_anomaly else "No"
            confidence = f"{explanation.prediction_confidence:.3f}" if explanation.prediction_confidence else "N/A"
            
            if explanation.top_features:
                top_feature = explanation.top_features[0]
                feature_name = top_feature["feature_name"]
                importance = f"{top_feature['importance']:.3f}"
            else:
                feature_name = "N/A"
                importance = "N/A"
            
            summary_table.add_row(
                str(sample_idx),
                is_anomaly,
                confidence,
                feature_name,
                importance
            )
        
        console.print(summary_table)
        
        # Global feature importance
        print(f"\n[bold blue]Global Feature Importance (Average):[/bold blue]")
        global_importance = {}
        for _, explanation in explanations:
            for feature_name, importance in explanation.feature_importance.items():
                if feature_name not in global_importance:
                    global_importance[feature_name] = []
                global_importance[feature_name].append(importance)
        
        # Average importance
        avg_importance = {}
        for feature_name, importance_list in global_importance.items():
            avg_importance[feature_name] = np.mean(importance_list)
        
        # Sort and display
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature_name, importance) in enumerate(sorted_features[:10]):
            print(f"  {i+1:2d}. {feature_name:20s}: {importance:6.3f}")
        
        # Save explanations if requested
        if output_file:
            batch_data = {
                'algorithm': algorithm,
                'explainer_type': explainer,
                'total_samples': len(explanations),
                'anomalies_only': anomalies_only,
                'global_feature_importance': avg_importance,
                'explanations': []
            }
            
            for sample_idx, explanation in explanations:
                batch_data['explanations'].append({
                    'sample_index': sample_idx,
                    'is_anomaly': explanation.is_anomaly,
                    'confidence': explanation.prediction_confidence,
                    'feature_importance': explanation.feature_importance,
                    'top_features': explanation.top_features,
                    'sample_data': dict(zip(feature_names, explanation.data_sample))
                })
            
            with open(output_file, 'w') as f:
                json.dump(batch_data, f, indent=2)
            
            print(f"[green]✅[/green] Batch explanations saved to: {output_file}")
            
    except Exception as e:
        print(f"[red]✗[/red] Batch explanation failed: {e}")
        raise typer.Exit(1)


@app.command()
def global_importance(
    input_file: Path = typer.Option(..., "--input", "-i", help="Training data file"),
    algorithm: str = typer.Option("isolation_forest", "--algorithm", "-a", help="Detection algorithm"),
    contamination: float = typer.Option(0.1, "--contamination", "-c", help="Contamination rate"),
    n_samples: int = typer.Option(100, "--samples", "-n", help="Number of samples to analyze"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for importance"),
) -> None:
    """Analyze global feature importance across the dataset."""
    
    print(f"[blue]🌍[/blue] Analyzing global feature importance")
    
    try:
        # Load data
        if not input_file.exists():
            print(f"[red]✗[/red] Input file '{input_file}' not found")
            raise typer.Exit(1)
        
        if input_file.suffix.lower() == '.csv':
            df = pd.read_csv(input_file)
        elif input_file.suffix.lower() == '.json':
            df = pd.read_json(input_file)
        else:
            print(f"[red]✗[/red] Unsupported file format '{input_file.suffix}'")
            raise typer.Exit(1)
        
        print(f"[green]✓[/green] Loaded data with {len(df)} samples, {len(df.columns)} features")
        
        # Initialize services
        detection_service = DetectionService()
        explainability_service = ExplainabilityService(detection_service)
        
        # Algorithm mapping
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        mapped_algorithm = algorithm_map.get(algorithm, algorithm)
        
        # Fit model
        print(f"[blue]ℹ[/blue] Fitting {algorithm} model...")
        data_array = df.values.astype(np.float64)
        detection_service.fit(data_array, mapped_algorithm, contamination=contamination)
        print(f"[green]✓[/green] Model fitted successfully")
        
        # Get global feature importance
        feature_names = df.columns.tolist()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Analyzing {n_samples} samples...", total=None)
            
            global_importance = explainability_service.get_global_feature_importance(
                algorithm=mapped_algorithm,
                training_data=data_array,
                feature_names=feature_names,
                n_samples=n_samples
            )
            
            progress.update(task, completed=True)
        
        # Display results
        print(f"\n[green]✅[/green] Global feature importance analysis complete!")
        
        # Create importance table
        importance_table = Table(title="[bold blue]Global Feature Importance[/bold blue]")
        importance_table.add_column("Rank", style="yellow", justify="center")
        importance_table.add_column("Feature", style="cyan")
        importance_table.add_column("Importance", style="green", justify="right")
        importance_table.add_column("Relative %", style="blue", justify="right")
        
        # Sort features by importance
        sorted_features = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
        max_importance = max(global_importance.values()) if global_importance else 1.0
        
        for i, (feature_name, importance) in enumerate(sorted_features):
            relative_pct = (importance / max_importance) * 100 if max_importance > 0 else 0
            
            importance_table.add_row(
                str(i + 1),
                feature_name,
                f"{importance:.4f}",
                f"{relative_pct:.1f}%"
            )
        
        console.print(importance_table)
        
        # Summary statistics
        stats_table = Table(title="[bold blue]Importance Statistics[/bold blue]")
        stats_table.add_column("Statistic", style="cyan")
        stats_table.add_column("Value", style="green")
        
        importance_values = list(global_importance.values())
        if importance_values:
            stats_table.add_row("Total Features", str(len(importance_values)))
            stats_table.add_row("Samples Analyzed", str(n_samples))
            stats_table.add_row("Max Importance", f"{max(importance_values):.4f}")
            stats_table.add_row("Min Importance", f"{min(importance_values):.4f}")
            stats_table.add_row("Mean Importance", f"{np.mean(importance_values):.4f}")
            stats_table.add_row("Std Importance", f"{np.std(importance_values):.4f}")
        
        console.print(stats_table)
        
        # Save results if requested
        if output_file:
            output_data = {
                'algorithm': algorithm,
                'samples_analyzed': n_samples,
                'total_features': len(global_importance),
                'global_feature_importance': global_importance,
                'importance_ranking': [
                    {'rank': i+1, 'feature': name, 'importance': importance}
                    for i, (name, importance) in enumerate(sorted_features)
                ],
                'statistics': {
                    'max_importance': max(importance_values) if importance_values else 0,
                    'min_importance': min(importance_values) if importance_values else 0,
                    'mean_importance': float(np.mean(importance_values)) if importance_values else 0,
                    'std_importance': float(np.std(importance_values)) if importance_values else 0
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"[green]✅[/green] Global importance analysis saved to: {output_file}")
            
    except Exception as e:
        print(f"[red]✗[/red] Global importance analysis failed: {e}")
        raise typer.Exit(1)


@app.command()
def available() -> None:
    """Show available explainer types."""
    
    detection_service = DetectionService()
    explainability_service = ExplainabilityService(detection_service)
    
    available_explainers = explainability_service.get_available_explainers()
    
    # Display available explainers
    explainers_table = Table(title="[bold blue]Available Explainer Types[/bold blue]")
    explainers_table.add_column("Explainer", style="cyan")
    explainers_table.add_column("Description", style="green")
    explainers_table.add_column("Status", style="blue")
    
    descriptions = {
        'shap': 'SHAP (SHapley Additive exPlanations) - Advanced model-agnostic explanations',
        'lime': 'LIME (Local Interpretable Model-agnostic Explanations) - Local linear approximations',
        'permutation': 'Permutation Importance - Feature importance via permutation testing',
        'feature_importance': 'Simple Feature Importance - Based on feature magnitude'
    }
    
    for explainer in available_explainers:
        description = descriptions.get(explainer, "Feature importance method")
        status = "[green]Available[/green]"
        
        explainers_table.add_row(explainer, description, status)
    
    console.print(explainers_table)
    
    # Installation notes
    if 'shap' not in available_explainers or 'lime' not in available_explainers:
        print("\n[yellow]💡 Note:[/yellow] To enable SHAP and LIME explainers:")
        print("   pip install shap lime")