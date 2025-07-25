"""A/B testing adapter implementation."""

import json
import asyncio
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import statistics
from dataclasses import asdict

from machine_learning.domain.interfaces.advanced_ml_operations import ABTestingPort
from machine_learning.domain.entities.model_version import (
    ABTestExperiment, ABTestResult, ExperimentStatus
)

logger = logging.getLogger(__name__)

class FileBasedABTestingAdapter(ABTestingPort):
    """File-based A/B testing implementation."""
    
    def __init__(self, storage_root: str = "/tmp/ab_experiments"):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.experiments_dir = self.storage_root / "experiments"
        self.results_dir = self.storage_root / "results"
        self.predictions_dir = self.storage_root / "predictions"
        
        for directory in [self.experiments_dir, self.results_dir, self.predictions_dir]:
            directory.mkdir(exist_ok=True)
    
    async def create_experiment(self, experiment: ABTestExperiment) -> str:
        """Create new A/B testing experiment."""
        try:
            experiment_file = self.experiments_dir / f"{experiment.experiment_id}.json"
            
            with open(experiment_file, 'w') as f:
                json.dump(asdict(experiment), f, indent=2, default=str)
            
            # Create predictions tracking file
            predictions_file = self.predictions_dir / f"{experiment.experiment_id}_predictions.json"
            with open(predictions_file, 'w') as f:
                json.dump({"predictions": [], "outcomes": []}, f)
            
            logger.info(f"Created A/B test experiment {experiment.experiment_id}")
            return experiment.experiment_id
        
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start running A/B test experiment."""
        try:
            experiment = await self._load_experiment(experiment_id)
            if not experiment:
                return False
            
            experiment.status = ExperimentStatus.RUNNING
            experiment.start_date = datetime.utcnow().isoformat()
            
            await self._save_experiment(experiment)
            
            logger.info(f"Started A/B test experiment {experiment_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to start experiment {experiment_id}: {e}")
            return False
    
    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop running A/B test experiment."""
        try:
            experiment = await self._load_experiment(experiment_id)
            if not experiment:
                return False
            
            experiment.status = ExperimentStatus.COMPLETED
            experiment.end_date = datetime.utcnow().isoformat()
            
            await self._save_experiment(experiment)
            
            # Generate final results
            await self._generate_experiment_results(experiment_id)
            
            logger.info(f"Stopped A/B test experiment {experiment_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to stop experiment {experiment_id}: {e}")
            return False
    
    async def get_experiment_results(self, experiment_id: str) -> Optional[ABTestResult]:
        """Get results of A/B test experiment."""
        try:
            results_file = self.results_dir / f"{experiment_id}_results.json"
            if not results_file.exists():
                # Generate results if they don't exist
                await self._generate_experiment_results(experiment_id)
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                return ABTestResult(**data)
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to get experiment results {experiment_id}: {e}")
            return None
    
    async def route_prediction(
        self, 
        experiment_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route prediction request based on A/B test configuration."""
        try:
            experiment = await self._load_experiment(experiment_id)
            if not experiment or experiment.status != ExperimentStatus.RUNNING:
                return {"error": "Experiment not running", "model_used": "default"}
            
            # Determine which model to use based on traffic split
            use_model_b = random.random() < experiment.traffic_split
            model_version = experiment.model_b_version if use_model_b else experiment.model_a_version
            
            # Generate prediction ID
            prediction_id = f"pred_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            
            # Simulate prediction (in real implementation, call actual model)
            prediction_result = await self._simulate_prediction(model_version, input_data)
            
            # Record prediction for analysis
            await self._record_prediction(
                experiment_id,
                prediction_id,
                model_version,
                input_data,
                prediction_result
            )
            
            # Update experiment sample size
            experiment.current_sample_size += 1
            await self._save_experiment(experiment)
            
            return {
                "prediction_id": prediction_id,
                "model_version": model_version,
                "model_group": "B" if use_model_b else "A",
                "prediction": prediction_result,
                "experiment_id": experiment_id
            }
        
        except Exception as e:
            logger.error(f"Failed to route prediction for experiment {experiment_id}: {e}")
            return {"error": str(e), "model_used": "error"}
    
    async def record_experiment_outcome(
        self,
        experiment_id: str,
        prediction_id: str,
        outcome: Dict[str, Any]
    ) -> bool:
        """Record outcome for A/B test analysis."""
        try:
            predictions_file = self.predictions_dir / f"{experiment_id}_predictions.json"
            
            # Load existing data
            with open(predictions_file, 'r') as f:
                data = json.load(f)
            
            # Add outcome
            outcome_record = {
                "prediction_id": prediction_id,
                "outcome": outcome,
                "recorded_at": datetime.utcnow().isoformat()
            }
            
            data["outcomes"].append(outcome_record)
            
            # Save updated data
            with open(predictions_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to record outcome for {prediction_id}: {e}")
            return False
    
    async def _load_experiment(self, experiment_id: str) -> Optional[ABTestExperiment]:
        """Load experiment from file."""
        try:
            experiment_file = self.experiments_dir / f"{experiment_id}.json"
            if not experiment_file.exists():
                return None
            
            with open(experiment_file, 'r') as f:
                data = json.load(f)
            
            return ABTestExperiment(**data)
        
        except Exception as e:
            logger.error(f"Failed to load experiment {experiment_id}: {e}")
            return None
    
    async def _save_experiment(self, experiment: ABTestExperiment) -> bool:
        """Save experiment to file."""
        try:
            experiment_file = self.experiments_dir / f"{experiment.experiment_id}.json"
            
            with open(experiment_file, 'w') as f:
                json.dump(asdict(experiment), f, indent=2, default=str)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to save experiment {experiment.experiment_id}: {e}")
            return False
    
    async def _simulate_prediction(self, model_version: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate model prediction (replace with actual model inference)."""
        # Simulate different model performance
        base_accuracy = 0.85
        if "v2" in model_version or "B" in model_version:
            # Model B is slightly better
            accuracy_adjustment = 0.03
        else:
            accuracy_adjustment = 0.0
        
        # Simulate prediction with some randomness
        confidence = base_accuracy + accuracy_adjustment + random.uniform(-0.1, 0.1)
        confidence = max(0.0, min(1.0, confidence))
        
        prediction = "positive" if confidence > 0.5 else "negative"
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "model_version": model_version,
            "response_time_ms": random.uniform(50, 200)
        }
    
    async def _record_prediction(
        self,
        experiment_id: str,
        prediction_id: str,
        model_version: str,
        input_data: Dict[str, Any],
        prediction_result: Dict[str, Any]
    ) -> bool:
        """Record prediction for experiment tracking."""
        try:
            predictions_file = self.predictions_dir / f"{experiment_id}_predictions.json"
            
            # Load existing data
            with open(predictions_file, 'r') as f:
                data = json.load(f)
            
            # Add prediction
            prediction_record = {
                "prediction_id": prediction_id,
                "model_version": model_version,
                "model_group": "B" if "v2" in model_version or "B" in model_version else "A",
                "input_data": input_data,
                "prediction_result": prediction_result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            data["predictions"].append(prediction_record)
            
            # Save updated data
            with open(predictions_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to record prediction {prediction_id}: {e}")
            return False
    
    async def _generate_experiment_results(self, experiment_id: str) -> bool:
        """Generate statistical results for experiment."""
        try:
            experiment = await self._load_experiment(experiment_id)
            if not experiment:
                return False
            
            predictions_file = self.predictions_dir / f"{experiment_id}_predictions.json"
            if not predictions_file.exists():
                return False
            
            with open(predictions_file, 'r') as f:
                data = json.load(f)
            
            predictions = data["predictions"]
            outcomes = data["outcomes"]
            
            # Separate predictions by model group
            model_a_predictions = [p for p in predictions if p["model_group"] == "A"]
            model_b_predictions = [p for p in predictions if p["model_group"] == "B"]
            
            # Calculate performance metrics
            model_a_performance = self._calculate_model_performance(model_a_predictions, outcomes)
            model_b_performance = self._calculate_model_performance(model_b_predictions, outcomes)
            
            # Calculate statistical significance (simplified)
            significance = self._calculate_statistical_significance(
                model_a_performance, model_b_performance
            )
            
            # Determine winner
            winner = None
            if significance > experiment.significance_threshold:
                if model_b_performance["accuracy"] > model_a_performance["accuracy"]:
                    winner = "model_b"
                elif model_a_performance["accuracy"] > model_b_performance["accuracy"]:
                    winner = "model_a"
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                model_a_performance, model_b_performance, winner, significance
            )
            
            # Calculate test duration
            start_date = datetime.fromisoformat(experiment.start_date.replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(experiment.end_date.replace('Z', '+00:00')) if experiment.end_date else datetime.utcnow()
            duration_days = (end_date - start_date).days
            
            # Create results
            results = ABTestResult(
                experiment_id=experiment_id,
                model_a_performance=model_a_performance,
                model_b_performance=model_b_performance,
                statistical_significance=significance,
                confidence_interval={"lower": 0.95, "upper": 0.99},  # Simplified
                winner=winner,
                recommendation=recommendation,
                sample_size_a=len(model_a_predictions),
                sample_size_b=len(model_b_predictions),
                test_duration_days=duration_days
            )
            
            # Save results
            results_file = self.results_dir / f"{experiment_id}_results.json"
            with open(results_file, 'w') as f:
                json.dump(asdict(results), f, indent=2, default=str)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to generate experiment results {experiment_id}: {e}")
            return False
    
    def _calculate_model_performance(self, predictions: List[Dict], outcomes: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics for model."""
        if not predictions:
            return {"accuracy": 0.0, "response_time": 0.0, "confidence": 0.0}
        
        # Calculate average confidence and response time
        confidences = [p["prediction_result"]["confidence"] for p in predictions]
        response_times = [p["prediction_result"]["response_time_ms"] for p in predictions]
        
        # Simulate accuracy based on outcomes (in real implementation, match with actual outcomes)
        accuracy = statistics.mean(confidences) if confidences else 0.0
        
        return {
            "accuracy": accuracy,
            "response_time": statistics.mean(response_times) if response_times else 0.0,
            "confidence": statistics.mean(confidences) if confidences else 0.0,
            "prediction_count": len(predictions)
        }
    
    def _calculate_statistical_significance(self, perf_a: Dict, perf_b: Dict) -> float:
        """Calculate statistical significance (simplified implementation)."""
        # In real implementation, use proper statistical tests like t-test or chi-square
        sample_size_a = perf_a.get("prediction_count", 0)
        sample_size_b = perf_b.get("prediction_count", 0)
        
        if sample_size_a < 30 or sample_size_b < 30:
            return 0.0  # Not enough samples
        
        accuracy_diff = abs(perf_b["accuracy"] - perf_a["accuracy"])
        
        # Simplified significance calculation
        min_sample_size = min(sample_size_a, sample_size_b)
        significance = min(0.99, accuracy_diff * min_sample_size / 100)
        
        return significance
    
    def _generate_recommendation(
        self, 
        perf_a: Dict, 
        perf_b: Dict, 
        winner: Optional[str],
        significance: float
    ) -> str:
        """Generate recommendation based on results."""
        if significance < 0.95:
            return "Inconclusive: Statistical significance threshold not met. Consider running longer or increasing sample size."
        
        if winner == "model_b":
            improvement = (perf_b["accuracy"] - perf_a["accuracy"]) * 100
            return f"RECOMMEND Model B: Shows {improvement:.1f}% improvement in accuracy with statistical significance."
        elif winner == "model_a":
            improvement = (perf_a["accuracy"] - perf_b["accuracy"]) * 100
            return f"RECOMMEND Model A: Shows {improvement:.1f}% better performance than Model B."
        else:
            return "NEUTRAL: No significant difference between models. Consider business factors for decision."