"""Anomaly Detection CLI interface."""

import click
import structlog
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import sys

from .domain.services.detection_service import DetectionService
from .domain.services.ensemble_service import EnsembleService
from .domain.services.streaming_service import StreamingService
from .domain.entities.dataset import Dataset, DatasetType, DatasetMetadata
from .domain.entities.model import Model, ModelMetadata, ModelStatus, SerializationFormat
from .infrastructure.repositories.model_repository import ModelRepository
from .infrastructure.logging import get_logger, log_decorator
from .infrastructure.logging.error_handler import (
    ErrorHandler, 
    AnomalyDetectionError, 
    InputValidationError,
    default_error_handler
)
from .infrastructure.monitoring import get_health_checker, get_metrics_collector
import asyncio

logger = get_logger(__name__)


def handle_cli_error(func):
    """Decorator for CLI error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AnomalyDetectionError as e:
            logger.error("CLI operation failed", 
                        error_type=type(e).__name__,
                        error_message=str(e),
                        operation=func.__name__)
            
            click.echo(f"❌ Error: {e.message}", err=True)
            if e.details:
                click.echo(f"   Details: {e.details}", err=True)
            
            if e.recoverable:
                click.echo("   This error may be recoverable. Please check your input and try again.", err=True)
            
            sys.exit(1)
        except Exception as e:
            # Handle unexpected errors
            ad_error = default_error_handler.handle_error(
                error=e,
                context={"cli_command": func.__name__},
                operation="cli_command",
                reraise=False
            )
            
            click.echo(f"❌ Unexpected error: {str(e)}", err=True)
            logger.error("Unexpected CLI error", 
                        error_type=type(e).__name__,
                        error_message=str(e),
                        operation=func.__name__)
            sys.exit(1)
    
    return wrapper


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def main(ctx: click.Context, verbose: bool, config: Optional[str]) -> None:
    """Anomaly Detection CLI - ML-based outlier detection and analysis."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config
    
    if verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(20)  # INFO level
        )


@main.group()
def detect() -> None:
    """Anomaly detection commands."""
    pass


@detect.command()
@click.option('--input', '-i', required=True, help='Input data file path')
@click.option('--output', '-o', help='Output results file path')
@click.option('--algorithm', '-a', default='isolation_forest', 
              type=click.Choice(['isolation_forest', 'one_class_svm', 'lof', 'autoencoder']),
              help='Detection algorithm to use')
@click.option('--contamination', default=0.1, type=float, help='Expected contamination ratio')
@click.option('--has-labels', is_flag=True, help='Dataset includes ground truth labels')
@click.option('--label-column', default='label', help='Name of label column')
@handle_cli_error
def run(input: str, output: Optional[str], algorithm: str, contamination: float, 
        has_labels: bool, label_column: str) -> None:
    """Run anomaly detection on dataset."""
    logger.info("Running anomaly detection", 
                input=input, algorithm=algorithm, contamination=contamination)
    
    try:
        # Load dataset
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"Error: Input file '{input}' not found", err=True)
            return
        
        # Load data based on file extension
        if input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_path)
        elif input_path.suffix.lower() in ['.json']:
            df = pd.read_json(input_path)
        elif input_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(input_path)
        else:
            click.echo(f"Error: Unsupported file format '{input_path.suffix}'", err=True)
            return
        
        # Extract labels if present
        labels = None
        if has_labels and label_column in df.columns:
            labels = df[label_column].values
            df = df.drop(columns=[label_column])
        
        # Create dataset
        dataset = Dataset(
            data=df,
            dataset_type=DatasetType.INFERENCE,
            labels=labels,
            metadata=DatasetMetadata(
                name=input_path.stem,
                source=str(input_path),
                description=f"Dataset loaded from {input_path}"
            )
        )
        
        # Validate dataset
        validation_issues = dataset.validate()
        if validation_issues:
            click.echo("Warning: Dataset validation issues found:", err=True)
            for issue in validation_issues:
                click.echo(f"  - {issue}", err=True)
        
        # Run detection
        click.echo(f"Running {algorithm} detection on {dataset.n_samples} samples...")
        
        service = DetectionService()
        data_array = dataset.to_numpy()
        
        # Map algorithm names to service parameters
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm', 
            'lof': 'lof',
            'autoencoder': 'autoencoder'
        }
        
        detection_result = service.detect_anomalies(
            data=data_array,
            algorithm=algorithm_map.get(algorithm, algorithm),
            contamination=contamination
        )
        
        # Prepare results
        results = {
            "input": str(input_path),
            "algorithm": algorithm,
            "contamination": contamination,
            "dataset_info": {
                "total_samples": dataset.n_samples,
                "n_features": dataset.n_features,
                "feature_names": dataset.feature_names
            },
            "detection_results": {
                "anomalies_detected": detection_result.anomaly_count,
                "normal_samples": detection_result.normal_count,
                "anomaly_rate": detection_result.anomaly_rate,
                "anomaly_indices": detection_result.anomalies
            },
            "timestamp": detection_result.timestamp.isoformat() if detection_result.timestamp else None,
            "success": detection_result.success
        }
        
        # Add evaluation metrics if ground truth labels are available
        if has_labels and labels is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
            
            # Convert predictions to same format as labels
            pred_labels = detection_result.predictions
            
            results["evaluation_metrics"] = {
                "accuracy": float(accuracy_score(labels, pred_labels)),
                "precision": float(precision_score(labels, pred_labels, pos_label=-1, zero_division=0, average='binary')),
                "recall": float(recall_score(labels, pred_labels, pos_label=-1, zero_division=0, average='binary')),
                "f1_score": float(f1_score(labels, pred_labels, pos_label=-1, zero_division=0, average='binary')),
                "classification_report": classification_report(labels, pred_labels, target_names=['Normal', 'Anomaly'], output_dict=True)
            }
        
        # Output results
        if output:
            output_path = Path(output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            click.echo(f"Results saved to: {output_path}")
        else:
            click.echo(json.dumps(results, indent=2))
        
        # Summary
        click.echo(f"\n✅ Detection completed successfully")
        click.echo(f"   Anomalies detected: {detection_result.anomaly_count}/{dataset.n_samples} ({detection_result.anomaly_rate:.1%})")
        
    except Exception as e:
        logger.error("Detection failed", error=str(e))
        click.echo(f"Error: {e}", err=True)


@main.group()
def ensemble() -> None:
    """Ensemble detection commands."""
    pass


@ensemble.command()
@click.option('--input', '-i', required=True, help='Input data file path')
@click.option('--output', '-o', help='Output results file path')
@click.option('--algorithms', '-a', multiple=True, 
              default=['isolation_forest', 'one_class_svm', 'lof'],
              help='Algorithms to use in ensemble')
@click.option('--method', default='majority', 
              type=click.Choice(['majority', 'average', 'weighted_average', 'max']),
              help='Ensemble combination method')
@click.option('--contamination', default=0.1, type=float, help='Expected contamination ratio')
@handle_cli_error
def combine(input: str, output: Optional[str], algorithms: tuple, method: str, contamination: float) -> None:
    """Run ensemble anomaly detection."""
    logger.info("Running ensemble detection", 
                input=input, algorithms=list(algorithms), method=method)
    
    try:
        # Load dataset
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"Error: Input file '{input}' not found", err=True)
            return
        
        # Load data
        if input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_path)
        elif input_path.suffix.lower() in ['.json']:
            df = pd.read_json(input_path)
        else:
            click.echo(f"Error: Unsupported file format '{input_path.suffix}'", err=True)
            return
        
        data_array = df.select_dtypes(include=[np.number]).values.astype(np.float64)
        
        # Map algorithm names
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof',
            'autoencoder': 'autoencoder'
        }
        
        mapped_algorithms = [algorithm_map.get(alg, alg) for alg in algorithms]
        
        click.echo(f"Running ensemble with {len(algorithms)} algorithms using {method} method...")
        
        # Initialize services
        detectors = []
        for algorithm in mapped_algorithms:
            service = DetectionService()
            detectors.append((algorithm, service))
        
        # Run ensemble detection
        ensemble_service = EnsembleService()
        
        # Get individual results
        individual_results = {}
        predictions_list = []
        scores_list = []
        
        for algorithm, service in detectors:
            result = service.detect_anomalies(
                data=data_array,
                algorithm=algorithm,
                contamination=contamination
            )
            individual_results[algorithm] = result
            predictions_list.append(result.predictions)
            if result.confidence_scores is not None:
                scores_list.append(result.confidence_scores)
        
        # Combine predictions using ensemble method
        predictions_array = np.array(predictions_list)
        scores_array = np.array(scores_list) if scores_list else None
        
        if method == 'majority':
            ensemble_predictions = ensemble_service.majority_vote(predictions_array)
            ensemble_scores = None
        elif method == 'average' and scores_array is not None:
            ensemble_predictions, ensemble_scores = ensemble_service.average_combination(
                predictions_array, scores_array
            )
        elif method == 'weighted_average' and scores_array is not None:
            # Use equal weights for simplicity
            weights = np.ones(len(algorithms)) / len(algorithms)
            ensemble_predictions, ensemble_scores = ensemble_service.weighted_combination(
                predictions_array, scores_array, weights
            )
        elif method == 'max' and scores_array is not None:
            ensemble_predictions, ensemble_scores = ensemble_service.max_combination(
                predictions_array, scores_array
            )
        else:
            # Fall back to majority vote
            ensemble_predictions = ensemble_service.majority_vote(predictions_array)
            ensemble_scores = None
        
        # Calculate ensemble statistics
        ensemble_anomaly_count = int(np.sum(ensemble_predictions == -1))
        ensemble_normal_count = len(ensemble_predictions) - ensemble_anomaly_count
        ensemble_anomaly_rate = ensemble_anomaly_count / len(ensemble_predictions)
        
        # Prepare results
        results = {
            "input": str(input_path),
            "ensemble_config": {
                "algorithms": list(algorithms),
                "method": method,
                "contamination": contamination
            },
            "individual_results": {
                alg: {
                    "anomalies_detected": result.anomaly_count,
                    "anomaly_rate": result.anomaly_rate,
                    "success": result.success
                } for alg, result in individual_results.items()
            },
            "ensemble_results": {
                "anomalies_detected": ensemble_anomaly_count,
                "normal_samples": ensemble_normal_count,
                "anomaly_rate": ensemble_anomaly_rate,
                "anomaly_indices": np.where(ensemble_predictions == -1)[0].tolist()
            },
            "dataset_info": {
                "total_samples": len(data_array),
                "n_features": data_array.shape[1]
            },
            "success": True
        }
        
        # Output results
        if output:
            output_path = Path(output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            click.echo(f"Results saved to: {output_path}")
        else:
            click.echo(json.dumps(results, indent=2))
        
        # Summary
        click.echo(f"\n✅ Ensemble detection completed successfully")
        click.echo(f"   Ensemble anomalies: {ensemble_anomaly_count}/{len(data_array)} ({ensemble_anomaly_rate:.1%})")
        click.echo("   Individual algorithm results:")
        for alg, result in individual_results.items():
            click.echo(f"     {alg}: {result.anomaly_count} anomalies ({result.anomaly_rate:.1%})")
        
    except Exception as e:
        logger.error("Ensemble detection failed", error=str(e))
        click.echo(f"Error: {e}", err=True)


@main.group() 
def stream() -> None:
    """Streaming detection commands."""
    pass


@stream.command()
@click.option('--source', '-s', required=True, help='Data stream source (file path or URL)')
@click.option('--window-size', default=100, type=int, help='Stream window size')
@click.option('--threshold', default=0.5, type=float, help='Anomaly threshold')
@click.option('--algorithm', '-a', default='isolation_forest',
              type=click.Choice(['isolation_forest', 'one_class_svm', 'lof']),
              help='Streaming detection algorithm')
@click.option('--update-frequency', default=10, type=int, help='Model update frequency (samples)')
def monitor(source: str, window_size: int, threshold: float, algorithm: str, update_frequency: int) -> None:
    """Monitor data stream for anomalies."""
    logger.info("Starting stream monitoring", 
                source=source, window_size=window_size, threshold=threshold)
    
    try:
        # Initialize streaming service
        streaming_service = StreamingService(
            buffer_size=window_size,
            algorithm=algorithm,
            update_frequency=update_frequency
        )
        
        click.echo(f"🔍 Starting anomaly monitoring")
        click.echo(f"   Source: {source}")
        click.echo(f"   Algorithm: {algorithm}")
        click.echo(f"   Window size: {window_size}")
        click.echo(f"   Threshold: {threshold}")
        click.echo(f"   Update frequency: {update_frequency} samples")
        click.echo("   Press Ctrl+C to stop monitoring\n")
        
        # Simulate streaming from file (in real implementation, this would connect to actual stream)
        if Path(source).exists():
            df = pd.read_csv(source)
            data_array = df.select_dtypes(include=[np.number]).values.astype(np.float64)
            
            # Simulate streaming by processing data in chunks
            chunk_size = 10  # Process 10 samples at a time
            anomaly_count = 0
            total_processed = 0
            
            for i in range(0, len(data_array), chunk_size):
                chunk = data_array[i:i+chunk_size]
                
                for sample in chunk:
                    # Process single sample
                    is_anomaly, confidence = streaming_service.detect_streaming_anomaly(
                        sample.reshape(1, -1)
                    )
                    
                    total_processed += 1
                    
                    if is_anomaly and confidence >= threshold:
                        anomaly_count += 1
                        click.echo(f"🚨 ANOMALY DETECTED at sample {total_processed}: confidence={confidence:.3f}")
                    elif total_processed % 50 == 0:  # Progress update every 50 samples
                        click.echo(f"✅ Processed {total_processed} samples, {anomaly_count} anomalies detected")
                
                # Small delay to simulate real-time streaming
                import time
                time.sleep(0.1)
            
            click.echo(f"\n📊 Streaming monitoring completed:")
            click.echo(f"   Total samples processed: {total_processed}")
            click.echo(f"   Anomalies detected: {anomaly_count}")
            click.echo(f"   Detection rate: {anomaly_count/total_processed:.1%}")
            
        else:
            click.echo(f"Error: Source file '{source}' not found", err=True)
            click.echo("Note: Real streaming from URLs/APIs not implemented yet", err=True)
        
    except KeyboardInterrupt:
        click.echo("\n⏹️  Monitoring stopped by user")
    except Exception as e:
        logger.error("Stream monitoring failed", error=str(e))
        click.echo(f"Error: {e}", err=True)


@main.group()
def data() -> None:
    """Data generation and management commands."""
    pass


@data.command()
@click.option('--output', '-o', required=True, help='Output dataset file path')
@click.option('--samples', '-n', default=1000, type=int, help='Number of samples to generate')
@click.option('--features', '-f', default=5, type=int, help='Number of features')
@click.option('--contamination', '-c', default=0.1, type=float, help='Contamination rate (anomaly ratio)')
@click.option('--anomaly-type', default='point', 
              type=click.Choice(['point', 'contextual', 'collective']),
              help='Type of anomalies to generate')
@click.option('--random-state', default=42, type=int, help='Random seed for reproducibility')
def generate(output: str, samples: int, features: int, contamination: float, 
             anomaly_type: str, random_state: int) -> None:
    """Generate synthetic anomaly detection dataset."""
    try:
        from sklearn.datasets import make_blobs
        from sklearn.preprocessing import StandardScaler
        
        np.random.seed(random_state)
        
        # Generate base dataset
        X, _ = make_blobs(n_samples=samples, centers=1, n_features=features, 
                         random_state=random_state, cluster_std=1.0)
        
        # Standardize data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Generate anomalies
        n_anomalies = int(samples * contamination)
        n_normal = samples - n_anomalies
        
        # Create labels (1 for normal, -1 for anomaly)
        labels = np.ones(samples)
        
        if anomaly_type == 'point':
            # Point anomalies: random outliers
            anomaly_indices = np.random.choice(samples, n_anomalies, replace=False)
            # Make these points far from the center
            X[anomaly_indices] += np.random.normal(0, 3, (n_anomalies, features))
            labels[anomaly_indices] = -1
            
        elif anomaly_type == 'contextual':
            # Contextual anomalies: anomalies in specific feature combinations
            anomaly_indices = np.random.choice(samples, n_anomalies, replace=False)
            # Modify specific features to create contextual anomalies
            for idx in anomaly_indices:
                # Make anomalous in one or two features
                anomalous_features = np.random.choice(features, np.random.randint(1, min(3, features+1)), replace=False)
                X[idx, anomalous_features] += np.random.normal(0, 2, len(anomalous_features))
            labels[anomaly_indices] = -1
            
        elif anomaly_type == 'collective':
            # Collective anomalies: groups of anomalous points
            n_groups = max(1, n_anomalies // 10)  # Create groups of ~10 anomalies
            group_size = n_anomalies // n_groups
            
            start_idx = 0
            for _ in range(n_groups):
                end_idx = min(start_idx + group_size, samples)
                group_indices = range(start_idx, end_idx)
                
                # Create a collective anomaly pattern
                group_shift = np.random.normal(0, 2, features)
                X[group_indices] += group_shift
                labels[group_indices] = -1
                
                start_idx = end_idx
        
        # Create dataset with feature names
        feature_names = [f"feature_{i}" for i in range(features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = labels.astype(int)
        
        # Save dataset
        output_path = Path(output)
        if output_path.suffix.lower() == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix.lower() == '.json':
            df.to_json(output_path, orient='records', indent=2)
        elif output_path.suffix.lower() == '.parquet':
            df.to_parquet(output_path, index=False)
        else:
            # Default to CSV
            output_path = output_path.with_suffix('.csv')
            df.to_csv(output_path, index=False)
        
        click.echo(f"✅ Generated synthetic dataset:")
        click.echo(f"   File: {output_path}")
        click.echo(f"   Samples: {samples} ({n_normal} normal, {n_anomalies} anomalies)")
        click.echo(f"   Features: {features}")
        click.echo(f"   Contamination: {contamination:.1%}")
        click.echo(f"   Anomaly type: {anomaly_type}")
        click.echo(f"   Random seed: {random_state}")
        
    except Exception as e:
        logger.error("Dataset generation failed", error=str(e))
        click.echo(f"Error: {e}", err=True)


@main.group()
def model() -> None:
    """Model management commands."""
    pass


@model.command()
@click.option('--input', '-i', required=True, help='Training dataset file path')
@click.option('--model-name', '-n', required=True, help='Name for the trained model')
@click.option('--algorithm', '-a', default='isolation_forest',
              type=click.Choice(['isolation_forest', 'one_class_svm', 'lof']),
              help='Algorithm to train')
@click.option('--contamination', '-c', default=0.1, type=float, help='Contamination rate')
@click.option('--output-dir', '-o', default='models', help='Output directory for saved model')
@click.option('--format', '-f', default='pickle',
              type=click.Choice(['pickle', 'joblib', 'json']),
              help='Model serialization format')
@click.option('--has-labels', is_flag=True, help='Dataset includes ground truth labels')
@click.option('--label-column', default='label', help='Name of label column')
@handle_cli_error
def train(input: str, model_name: str, algorithm: str, contamination: float,
          output_dir: str, format: str, has_labels: bool, label_column: str) -> None:
    """Train and save an anomaly detection model."""
    try:
        import uuid
        from datetime import datetime
        
        # Load dataset
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"Error: Input file '{input}' not found", err=True)
            return
        
        # Load data
        if input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_path)
        elif input_path.suffix.lower() == '.json':
            df = pd.read_json(input_path)
        else:
            click.echo(f"Error: Unsupported file format '{input_path.suffix}'", err=True)
            return
        
        # Extract labels if present
        labels = None
        if has_labels and label_column in df.columns:
            labels = df[label_column].values
            df = df.drop(columns=[label_column])
        
        # Create dataset
        dataset = Dataset(
            data=df,
            dataset_type=DatasetType.TRAINING,
            labels=labels,
            metadata=DatasetMetadata(
                name=input_path.stem,
                source=str(input_path),
                description=f"Training dataset loaded from {input_path}"
            )
        )
        
        click.echo(f"🏋️ Training {algorithm} model '{model_name}' on {dataset.n_samples} samples...")
        
        start_time = datetime.utcnow()
        
        # Train model
        service = DetectionService()
        data_array = dataset.to_numpy()
        
        # Algorithm mapping
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        
        # Fit the model
        service.fit(data_array, algorithm_map.get(algorithm, algorithm), contamination=contamination)
        
        # Get predictions for evaluation
        detection_result = service.detect_anomalies(
            data=data_array,
            algorithm=algorithm_map.get(algorithm, algorithm),
            contamination=contamination
        )
        
        end_time = datetime.utcnow()
        training_duration = (end_time - start_time).total_seconds()
        
        # Calculate metrics if labels available
        accuracy, precision, recall, f1_score = None, None, None, None
        if has_labels and labels is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score as f1
            pred_labels = detection_result.predictions
            accuracy = float(accuracy_score(labels, pred_labels))
            precision = float(precision_score(labels, pred_labels, pos_label=-1, zero_division=0, average='binary'))
            recall = float(recall_score(labels, pred_labels, pos_label=-1, zero_division=0, average='binary'))
            f1_score = float(f1(labels, pred_labels, pos_label=-1, zero_division=0, average='binary'))
        
        # Create model entity
        model_id = str(uuid.uuid4())
        metadata = ModelMetadata(
            model_id=model_id,
            name=model_name,
            algorithm=algorithm,
            status=ModelStatus.TRAINED,
            training_samples=dataset.n_samples,
            training_features=dataset.n_features,
            contamination_rate=contamination,
            training_duration_seconds=training_duration,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            feature_names=dataset.feature_names,
            hyperparameters={'contamination': contamination},
            description=f"Trained {algorithm} model on {dataset.n_samples} samples",
        )
        
        # Get the trained model object from the service
        trained_model_obj = service._fitted_models.get(algorithm_map.get(algorithm, algorithm))
        
        model = Model(
            metadata=metadata,
            model_object=trained_model_obj
        )
        
        # Save model using repository
        repo = ModelRepository(output_dir)
        format_map = {
            'pickle': SerializationFormat.PICKLE,
            'joblib': SerializationFormat.JOBLIB,
            'json': SerializationFormat.JSON
        }
        format_enum = format_map.get(format, SerializationFormat.PICKLE)
        saved_model_id = repo.save(model, format_enum)
        
        # Results summary
        click.echo(f"\n✅ Model training completed successfully!")
        click.echo(f"   Model ID: {saved_model_id}")
        click.echo(f"   Name: {model_name}")
        click.echo(f"   Algorithm: {algorithm}")
        click.echo(f"   Training time: {training_duration:.2f} seconds")
        click.echo(f"   Training samples: {dataset.n_samples}")
        
        if has_labels and labels is not None:
            click.echo(f"   Performance metrics:")
            click.echo(f"     Accuracy: {accuracy:.3f}")
            click.echo(f"     Precision: {precision:.3f}")
            click.echo(f"     Recall: {recall:.3f}")
            click.echo(f"     F1-score: {f1_score:.3f}")
        
        click.echo(f"   Model saved to: {output_dir}")
        
    except Exception as e:
        logger.error("Model training failed", error=str(e))
        click.echo(f"Error: {e}", err=True)


@model.command()
@click.option('--models-dir', '-d', default='models', help='Models directory')
@click.option('--algorithm', '-a', help='Filter by algorithm')
@click.option('--status', '-s', type=click.Choice(['training', 'trained', 'deployed', 'deprecated', 'failed']),
              help='Filter by status')
def list(models_dir: str, algorithm: Optional[str], status: Optional[str]) -> None:
    """List saved models."""
    try:
        repo = ModelRepository(models_dir)
        
        status_filter = ModelStatus(status) if status else None
        models = repo.list_models(status=status_filter, algorithm=algorithm)
        
        if not models:
            click.echo("No models found matching the criteria.")
            return
        
        click.echo(f"📋 Found {len(models)} model(s):\n")
        
        for model in models:
            click.echo(f"🤖 {model['name']} ({model['model_id'][:8]}...)")
            click.echo(f"   Algorithm: {model['algorithm']}")
            click.echo(f"   Status: {model['status']}")
            click.echo(f"   Created: {model['created_at']}")
            if model['accuracy']:
                click.echo(f"   Accuracy: {model['accuracy']:.3f}")
            if model['training_samples']:
                click.echo(f"   Training samples: {model['training_samples']}")
            if model['tags']:
                click.echo(f"   Tags: {', '.join(model['tags'])}")
            click.echo()
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@model.command()
@click.argument('model_id')
@click.option('--models-dir', '-d', default='models', help='Models directory')
def info(model_id: str, models_dir: str) -> None:
    """Show detailed information about a model."""
    try:
        repo = ModelRepository(models_dir)
        metadata = repo.get_model_metadata(model_id)
        
        click.echo(f"🤖 Model Information:")
        click.echo(f"   ID: {metadata['model_id']}")
        click.echo(f"   Name: {metadata['name']}")
        click.echo(f"   Algorithm: {metadata['algorithm']}")
        click.echo(f"   Version: {metadata['version']}")
        click.echo(f"   Status: {metadata['status']}")
        click.echo(f"   Created: {metadata['created_at']}")
        click.echo(f"   Updated: {metadata['updated_at']}")
        
        if metadata.get('description'):
            click.echo(f"   Description: {metadata['description']}")
        
        click.echo(f"\n📊 Training Information:")
        if metadata.get('training_samples'):
            click.echo(f"   Training samples: {metadata['training_samples']}")
        if metadata.get('training_features'):
            click.echo(f"   Features: {metadata['training_features']}")
        if metadata.get('contamination_rate'):
            click.echo(f"   Contamination rate: {metadata['contamination_rate']:.1%}")
        if metadata.get('training_duration_seconds'):
            click.echo(f"   Training time: {metadata['training_duration_seconds']:.2f} seconds")
        
        if any(metadata.get(metric) for metric in ['accuracy', 'precision', 'recall', 'f1_score']):
            click.echo(f"\n📈 Performance Metrics:")
            if metadata.get('accuracy'):
                click.echo(f"   Accuracy: {metadata['accuracy']:.3f}")
            if metadata.get('precision'):
                click.echo(f"   Precision: {metadata['precision']:.3f}")
            if metadata.get('recall'):
                click.echo(f"   Recall: {metadata['recall']:.3f}")
            if metadata.get('f1_score'):
                click.echo(f"   F1-score: {metadata['f1_score']:.3f}")
        
        if metadata.get('hyperparameters'):
            click.echo(f"\n⚙️ Hyperparameters:")
            for param, value in metadata['hyperparameters'].items():
                click.echo(f"   {param}: {value}")
        
        if metadata.get('tags'):
            click.echo(f"\n🏷️ Tags: {', '.join(metadata['tags'])}")
        
    except FileNotFoundError:
        click.echo(f"Error: Model with ID '{model_id}' not found", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@model.command()
@click.argument('model_id')
@click.option('--models-dir', '-d', default='models', help='Models directory')
@click.confirmation_option(prompt='Are you sure you want to delete this model?')
def delete(model_id: str, models_dir: str) -> None:
    """Delete a saved model."""
    try:
        repo = ModelRepository(models_dir)
        
        if repo.delete(model_id):
            click.echo(f"✅ Model '{model_id}' deleted successfully")
        else:
            click.echo(f"❌ Model '{model_id}' not found", err=True)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@model.command()
@click.option('--models-dir', '-d', default='models', help='Models directory')
def stats(models_dir: str) -> None:
    """Show repository statistics."""
    try:
        repo = ModelRepository(models_dir)
        stats = repo.get_repository_stats()
        
        click.echo(f"📊 Model Repository Statistics:")
        click.echo(f"   Total models: {stats['total_models']}")
        click.echo(f"   Storage size: {stats['storage_size_mb']} MB")
        click.echo(f"   Storage path: {stats['storage_path']}")
        
        if stats['by_status']:
            click.echo(f"\n📈 By Status:")
            for status, count in stats['by_status'].items():
                click.echo(f"   {status}: {count}")
        
        if stats['by_algorithm']:
            click.echo(f"\n🤖 By Algorithm:")
            for algorithm, count in stats['by_algorithm'].items():
                click.echo(f"   {algorithm}: {count}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@main.group()
def explain() -> None:
    """Explainability commands."""
    pass


@explain.command()
@click.option('--results', '-r', required=True, help='Detection results file')
@click.option('--method', default='shap', 
              type=click.Choice(['shap', 'lime', 'feature_importance']),
              help='Explanation method')
def analyze(results: str, method: str) -> None:
    """Generate explanations for detection results."""
    logger.info("Generating explanations", results=results, method=method)
    
    # Implementation would use ExplanationAnalyzers
    click.echo(f"Analyzing results from: {results} using {method}")


@main.group()
def worker() -> None:
    """Background worker management commands."""
    pass


@worker.command()
@click.option('--models-dir', '-d', default='models', help='Models directory')
@click.option('--max-jobs', default=3, type=int, help='Maximum concurrent jobs')
@click.option('--enable-monitoring', is_flag=True, default=True, help='Enable monitoring')
@handle_cli_error
def start(models_dir: str, max_jobs: int, enable_monitoring: bool) -> None:
    """Start the background worker service."""
    from .worker import AnomalyDetectionWorker
    
    click.echo(f"🚀 Starting Anomaly Detection Worker")
    click.echo(f"   Models directory: {models_dir}")
    click.echo(f"   Max concurrent jobs: {max_jobs}")
    click.echo(f"   Monitoring enabled: {enable_monitoring}")
    click.echo("   Press Ctrl+C to stop worker\n")
    
    worker_instance = AnomalyDetectionWorker(
        models_dir=models_dir,
        max_concurrent_jobs=max_jobs,
        enable_monitoring=enable_monitoring
    )
    
    async def run_worker():
        try:
            await worker_instance.start()
        except KeyboardInterrupt:
            click.echo("\n🛑 Shutting down worker...")
            await worker_instance.stop()
            click.echo("✅ Worker stopped successfully")
    
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        click.echo("\n✅ Worker shutdown complete")


@worker.command()
@click.option('--job-type', required=True,
              type=click.Choice(['detection', 'ensemble', 'batch_training', 'stream_monitoring', 
                               'model_validation', 'data_preprocessing', 'explanation_generation', 
                               'scheduled_analysis']),
              help='Type of job to submit')
@click.option('--data-source', help='Data source (file path or direct data)')
@click.option('--algorithm', default='isolation_forest', help='Algorithm to use')
@click.option('--contamination', default=0.1, type=float, help='Contamination rate')
@click.option('--output', help='Output file path')
@click.option('--priority', default='normal',
              type=click.Choice(['low', 'normal', 'high', 'critical']),
              help='Job priority')
@click.option('--models-dir', '-d', default='models', help='Models directory')
@click.option('--wait', is_flag=True, help='Wait for job completion')
@handle_cli_error
def submit(job_type: str, data_source: Optional[str], algorithm: str, 
           contamination: float, output: Optional[str], priority: str,
           models_dir: str, wait: bool) -> None:
    """Submit a job to the background worker."""
    from .worker import AnomalyDetectionWorker, JobType, JobPriority
    
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
    
    # Build job payload
    payload = {
        'algorithm': algorithm,
        'contamination': contamination
    }
    
    if data_source:
        payload['data_source'] = data_source
    if output:
        payload['output_path'] = output
    
    # Additional payload based on job type
    if job_type == 'ensemble':
        payload['algorithms'] = ['isolation_forest', 'lof']
        payload['method'] = 'majority'
    elif job_type == 'explanation_generation':
        payload['anomaly_indices'] = [1, 2, 3]  # Default indices
        payload['method'] = 'feature_importance'
    elif job_type == 'scheduled_analysis':
        payload['analysis_type'] = 'daily_summary'
        payload['data_sources'] = ['default']
    
    worker_instance = AnomalyDetectionWorker(
        models_dir=models_dir,
        enable_monitoring=False
    )
    
    async def submit_and_wait():
        # Submit job
        job_id = await worker_instance.submit_job(
            job_type_map[job_type],
            payload,
            priority=priority_map[priority]
        )
        
        click.echo(f"✅ Job submitted successfully")
        click.echo(f"   Job ID: {job_id}")
        click.echo(f"   Type: {job_type}")
        click.echo(f"   Priority: {priority}")
        
        if wait:
            click.echo("\n⏳ Waiting for job completion...")
            
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
                        
                        click.echo(f"   Status: {job_status} ({progress:.1f}%)")
                        
                        if job_status in ['completed', 'failed', 'cancelled']:
                            break
                    
                    await asyncio.sleep(3)
                    wait_time += 3
                
                # Get final result
                final_status = await worker_instance.get_job_status(job_id)
                if final_status:
                    if final_status['status'] == 'completed':
                        click.echo(f"\n✅ Job completed successfully!")
                        if final_status.get('result'):
                            click.echo("   Result:")
                            click.echo(f"   {json.dumps(final_status['result'], indent=2, default=str)}")
                    else:
                        click.echo(f"\n❌ Job {final_status['status']}")
                        if final_status.get('error_message'):
                            click.echo(f"   Error: {final_status['error_message']}")
                
            finally:
                await worker_instance.stop()
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass
    
    asyncio.run(submit_and_wait())


@worker.command()
@click.option('--job-id', required=True, help='Job ID to check')
@click.option('--models-dir', '-d', default='models', help='Models directory')
@handle_cli_error
def status(job_id: str, models_dir: str) -> None:
    """Check the status of a specific job."""
    from .worker import AnomalyDetectionWorker
    
    worker_instance = AnomalyDetectionWorker(
        models_dir=models_dir,
        enable_monitoring=False
    )
    
    async def check_status():
        job_status = await worker_instance.get_job_status(job_id)
        
        if job_status:
            click.echo(f"📋 Job Status: {job_id}")
            click.echo(f"   Status: {job_status['status']}")
            click.echo(f"   Type: {job_status['job_type']}")
            click.echo(f"   Priority: {job_status['priority']}")
            click.echo(f"   Progress: {job_status.get('progress', 0):.1f}%")
            click.echo(f"   Created: {job_status['created_at']}")
            
            if job_status.get('started_at'):
                click.echo(f"   Started: {job_status['started_at']}")
            
            if job_status.get('completed_at'):
                click.echo(f"   Completed: {job_status['completed_at']}")
            
            if job_status.get('error_message'):
                click.echo(f"   Error: {job_status['error_message']}")
            
            if job_status.get('result'):
                click.echo(f"\n📊 Result:")
                click.echo(f"{json.dumps(job_status['result'], indent=2, default=str)}")
        else:
            click.echo(f"❌ Job not found: {job_id}")
    
    asyncio.run(check_status())


@worker.command()
@click.option('--models-dir', '-d', default='models', help='Models directory')
@handle_cli_error
def info() -> None:
    """Show worker system information."""
    from .worker import AnomalyDetectionWorker
    
    worker_instance = AnomalyDetectionWorker(
        models_dir='models',
        enable_monitoring=False
    )
    
    async def show_info():
        worker_status = await worker_instance.get_worker_status()
        queue_status = worker_status['queue_status']
        
        click.echo(f"🔧 Worker System Information")
        click.echo(f"   Running: {worker_status['is_running']}")
        click.echo(f"   Max concurrent jobs: {worker_status['max_concurrent_jobs']}")
        click.echo(f"   Currently running: {worker_status['currently_running_jobs']}")
        click.echo(f"   Monitoring enabled: {worker_status['monitoring_enabled']}")
        
        click.echo(f"\n📋 Job Queue Status")
        click.echo(f"   Pending jobs: {queue_status['pending_jobs']}")
        click.echo(f"   Total jobs: {queue_status['total_jobs']}")
        
        if queue_status['status_counts']:
            click.echo(f"\n📊 Job Status Counts:")
            for status, count in queue_status['status_counts'].items():
                click.echo(f"   {status}: {count}")
        
        if worker_status['running_job_ids']:
            click.echo(f"\n🏃 Currently Running Jobs:")
            for job_id in worker_status['running_job_ids']:
                click.echo(f"   {job_id}")
    
    asyncio.run(show_info())


@worker.command()
@click.option('--job-id', required=True, help='Job ID to cancel')
@click.option('--models-dir', '-d', default='models', help='Models directory')
@handle_cli_error
def cancel(job_id: str, models_dir: str) -> None:
    """Cancel a pending or running job."""
    from .worker import AnomalyDetectionWorker
    
    worker_instance = AnomalyDetectionWorker(
        models_dir=models_dir,
        enable_monitoring=False
    )
    
    async def cancel_job():
        success = await worker_instance.cancel_job(job_id)
        
        if success:
            click.echo(f"✅ Job cancelled successfully: {job_id}")
        else:
            click.echo(f"❌ Failed to cancel job: {job_id}")
            click.echo("   Job may not exist or already be completed")
    
    asyncio.run(cancel_job())


@worker.command()
@handle_cli_error
def demo() -> None:
    """Run a demonstration of worker capabilities."""
    from .worker import run_worker_demo
    
    click.echo(f"🎭 Running Worker Demo")
    click.echo(f"   This will demonstrate various worker capabilities")
    click.echo(f"   Including detection, ensemble, and explanation jobs\n")
    
    try:
        asyncio.run(run_worker_demo())
    except KeyboardInterrupt:
        click.echo("\n🛑 Demo interrupted by user")


if __name__ == '__main__':
    main()