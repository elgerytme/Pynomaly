"""Migration utilities and guide for transitioning from Phase 1 to Phase 2.

This module provides utilities to help users migrate from the old architecture
to the new Phase 2 simplified services and enhanced features.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass


@dataclass
class MigrationRecommendation:
    """Migration recommendation for upgrading to Phase 2."""
    old_usage: str
    new_usage: str
    description: str
    complexity: str  # "simple", "moderate", "complex"
    breaking_changes: List[str]
    benefits: List[str]


class MigrationHelper:
    """Helper class for migrating from Phase 1 to Phase 2."""
    
    def __init__(self):
        """Initialize migration helper."""
        self.recommendations: List[MigrationRecommendation] = []
        self._setup_recommendations()
    
    def _setup_recommendations(self) -> None:
        """Setup migration recommendations."""
        self.recommendations = [
            MigrationRecommendation(
                old_usage="from pynomaly_detection.services.* import multiple services",
                new_usage="from pynomaly_detection import CoreDetectionService",
                description="Replace multiple service imports with single CoreDetectionService",
                complexity="simple",
                breaking_changes=["Service interface changes", "Method name changes"],
                benefits=["Simplified API", "Better performance", "Unified interface"]
            ),
            MigrationRecommendation(
                old_usage="Manual algorithm selection and configuration",
                new_usage="from pynomaly_detection import AutoMLService",
                description="Use AutoMLService for intelligent algorithm selection",
                complexity="simple",
                breaking_changes=["Configuration format changes"],
                benefits=["Automatic algorithm selection", "Better performance", "Less configuration"]
            ),
            MigrationRecommendation(
                old_usage="Custom ensemble implementations",
                new_usage="from pynomaly_detection import EnsembleService",
                description="Replace custom ensemble code with EnsembleService",
                complexity="moderate",
                breaking_changes=["Ensemble interface changes"],
                benefits=["Multiple voting strategies", "Better ensemble methods", "Performance optimization"]
            ),
            MigrationRecommendation(
                old_usage="Manual model saving/loading",
                new_usage="from pynomaly_detection import ModelPersistence",
                description="Use ModelPersistence for enterprise-grade model management",
                complexity="moderate",
                breaking_changes=["Model storage format changes"],
                benefits=["Version control", "Metadata tracking", "Performance metrics"]
            ),
            MigrationRecommendation(
                old_usage="Basic logging for monitoring",
                new_usage="from pynomaly_detection import MonitoringAlertingSystem",
                description="Upgrade to comprehensive monitoring and alerting",
                complexity="complex",
                breaking_changes=["Monitoring interface changes"],
                benefits=["Real-time monitoring", "Configurable alerts", "Performance tracking"]
            )
        ]
    
    def analyze_code(self, code_snippet: str) -> List[MigrationRecommendation]:
        """Analyze code snippet and provide migration recommendations.
        
        Args:
            code_snippet: Python code to analyze
            
        Returns:
            List of applicable migration recommendations
        """
        applicable_recommendations = []
        
        # Check for old service imports
        if "from pynomaly_detection.services" in code_snippet:
            applicable_recommendations.append(self.recommendations[0])
        
        # Check for manual algorithm selection
        if "IsolationForest" in code_snippet or "LocalOutlierFactor" in code_snippet:
            applicable_recommendations.append(self.recommendations[1])
        
        # Check for ensemble usage
        if "ensemble" in code_snippet.lower() or "voting" in code_snippet.lower():
            applicable_recommendations.append(self.recommendations[2])
        
        # Check for model persistence
        if "pickle" in code_snippet or "joblib" in code_snippet:
            applicable_recommendations.append(self.recommendations[3])
        
        # Check for basic logging
        if "logging" in code_snippet and "anomaly" in code_snippet.lower():
            applicable_recommendations.append(self.recommendations[4])
        
        return applicable_recommendations
    
    def get_all_recommendations(self) -> List[MigrationRecommendation]:
        """Get all migration recommendations."""
        return self.recommendations
    
    def generate_migration_report(self, code_snippets: List[str]) -> str:
        """Generate a comprehensive migration report.
        
        Args:
            code_snippets: List of code snippets to analyze
            
        Returns:
            Formatted migration report
        """
        report = []
        report.append("=" * 60)
        report.append("PYNOMALY DETECTION - PHASE 2 MIGRATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Analyze all code snippets
        all_recommendations = set()
        for snippet in code_snippets:
            recommendations = self.analyze_code(snippet)
            all_recommendations.update(recommendations)
        
        if not all_recommendations:
            report.append("🎉 No migration needed! Your code appears to be compatible.")
            return "\n".join(report)
        
        # Group by complexity
        by_complexity = {"simple": [], "moderate": [], "complex": []}
        for rec in all_recommendations:
            by_complexity[rec.complexity].append(rec)
        
        # Report by complexity
        for complexity in ["simple", "moderate", "complex"]:
            if by_complexity[complexity]:
                report.append(f"## {complexity.upper()} MIGRATIONS")
                report.append("-" * 40)
                report.append("")
                
                for rec in by_complexity[complexity]:
                    report.append(f"### {rec.description}")
                    report.append("")
                    report.append(f"**Old Usage:**")
                    report.append(f"```python")
                    report.append(f"{rec.old_usage}")
                    report.append(f"```")
                    report.append("")
                    report.append(f"**New Usage:**")
                    report.append(f"```python")
                    report.append(f"{rec.new_usage}")
                    report.append(f"```")
                    report.append("")
                    report.append(f"**Benefits:**")
                    for benefit in rec.benefits:
                        report.append(f"- {benefit}")
                    report.append("")
                    report.append(f"**Breaking Changes:**")
                    for change in rec.breaking_changes:
                        report.append(f"- {change}")
                    report.append("")
                    report.append("-" * 40)
                    report.append("")
        
        return "\n".join(report)


class CompatibilityLayer:
    """Compatibility layer for Phase 1 to Phase 2 migration."""
    
    def __init__(self):
        """Initialize compatibility layer."""
        self._warned_methods = set()
    
    def _deprecation_warning(self, old_method: str, new_method: str) -> None:
        """Issue deprecation warning."""
        if old_method not in self._warned_methods:
            warnings.warn(
                f"{old_method} is deprecated and will be removed in future versions. "
                f"Please use {new_method} instead.",
                DeprecationWarning,
                stacklevel=3
            )
            self._warned_methods.add(old_method)
    
    def legacy_detect_anomalies(self, data: npt.NDArray[np.floating], **kwargs: Any) -> Any:
        """Legacy detect_anomalies method with compatibility warnings."""
        self._deprecation_warning("legacy_detect_anomalies", "CoreDetectionService.detect_anomalies")
        
        # Try to use Phase 2 services
        try:
            from pynomaly_detection import CoreDetectionService
            service = CoreDetectionService()
            return service.detect_anomalies(data, **kwargs)
        except ImportError:
            # Fallback to basic detection
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(contamination=kwargs.get('contamination', 0.1))
            predictions = model.fit_predict(data)
            return (predictions == -1).astype(int)
    
    def legacy_automl_detect(self, data: npt.NDArray[np.floating], **kwargs: Any) -> Any:
        """Legacy AutoML detection with compatibility warnings."""
        self._deprecation_warning("legacy_automl_detect", "AutoMLService.auto_detect")
        
        try:
            from pynomaly_detection import AutoMLService
            service = AutoMLService()
            return service.auto_detect(data, **kwargs)
        except ImportError:
            # Fallback to basic detection
            return self.legacy_detect_anomalies(data, **kwargs)
    
    def legacy_ensemble_detect(self, data: npt.NDArray[np.floating], **kwargs: Any) -> Any:
        """Legacy ensemble detection with compatibility warnings."""
        self._deprecation_warning("legacy_ensemble_detect", "EnsembleService.ensemble_detect")
        
        try:
            from pynomaly_detection import EnsembleService
            service = EnsembleService()
            return service.ensemble_detect(data, **kwargs)
        except ImportError:
            # Fallback to basic detection
            return self.legacy_detect_anomalies(data, **kwargs)


def check_migration_status() -> Dict[str, Any]:
    """Check migration status and provide recommendations.
    
    Returns:
        Dictionary with migration status and recommendations
    """
    from pynomaly_detection import check_phase2_availability, get_version_info
    
    phase2_status = check_phase2_availability()
    version_info = get_version_info()
    
    # Calculate migration score
    available_features = sum(1 for available in phase2_status.values() if available)
    total_features = len(phase2_status)
    migration_score = (available_features / total_features) * 100
    
    # Determine migration status
    if migration_score == 100:
        migration_status = "complete"
        message = "✅ Migration complete! All Phase 2 features are available."
    elif migration_score >= 75:
        migration_status = "mostly_complete"
        message = "🟡 Migration mostly complete. Some Phase 2 features are missing."
    elif migration_score >= 50:
        migration_status = "in_progress"
        message = "🔄 Migration in progress. Many Phase 2 features are available."
    else:
        migration_status = "not_started"
        message = "❌ Migration not started. Phase 2 features are not available."
    
    return {
        "migration_status": migration_status,
        "migration_score": migration_score,
        "message": message,
        "phase2_availability": phase2_status,
        "version": version_info["version"],
        "next_steps": _get_migration_next_steps(phase2_status)
    }


def _get_migration_next_steps(phase2_status: Dict[str, bool]) -> List[str]:
    """Get next steps for migration based on current status."""
    next_steps = []
    
    if not phase2_status.get("simplified_services", False):
        next_steps.append("Install Phase 2 simplified services")
    
    if not phase2_status.get("enhanced_features", False):
        next_steps.append("Install Phase 2 enhanced features (model persistence, explainability)")
    
    if not phase2_status.get("performance_features", False):
        next_steps.append("Install Phase 2 performance features (batch processing, streaming)")
    
    if not phase2_status.get("monitoring", False):
        next_steps.append("Install Phase 2 monitoring and alerting")
    
    if not phase2_status.get("integration", False):
        next_steps.append("Install Phase 2 integration adapters")
    
    if not next_steps:
        next_steps.append("All Phase 2 features are available! Consider updating your code to use new APIs.")
    
    return next_steps


def generate_migration_examples() -> str:
    """Generate comprehensive migration examples.
    
    Returns:
        Formatted migration examples
    """
    examples = []
    examples.append("# PHASE 2 MIGRATION EXAMPLES")
    examples.append("=" * 50)
    examples.append("")
    
    # Example 1: Basic Detection
    examples.append("## Example 1: Basic Anomaly Detection")
    examples.append("")
    examples.append("### Phase 1 (Old)")
    examples.append("```python")
    examples.append("from pynomaly_detection.services.detection_service import DetectionService")
    examples.append("from pynomaly_detection.algorithms.adapters.sklearn_adapter import SklearnAdapter")
    examples.append("")
    examples.append("# Complex setup required")
    examples.append("adapter = SklearnAdapter()")
    examples.append("service = DetectionService(adapter)")
    examples.append("result = service.detect_anomalies(data)")
    examples.append("```")
    examples.append("")
    examples.append("### Phase 2 (New)")
    examples.append("```python")
    examples.append("from pynomaly_detection import CoreDetectionService")
    examples.append("")
    examples.append("# Simple, unified interface")
    examples.append("detector = CoreDetectionService()")
    examples.append("result = detector.detect_anomalies(data, algorithm='iforest')")
    examples.append("```")
    examples.append("")
    
    # Example 2: AutoML
    examples.append("## Example 2: AutoML Detection")
    examples.append("")
    examples.append("### Phase 1 (Old)")
    examples.append("```python")
    examples.append("# Manual algorithm selection and tuning")
    examples.append("from sklearn.ensemble import IsolationForest")
    examples.append("from sklearn.neighbors import LocalOutlierFactor")
    examples.append("from sklearn.model_selection import GridSearchCV")
    examples.append("")
    examples.append("# Complex manual process")
    examples.append("algorithms = [IsolationForest(), LocalOutlierFactor()]")
    examples.append("# ... lots of manual tuning code ...")
    examples.append("```")
    examples.append("")
    examples.append("### Phase 2 (New)")
    examples.append("```python")
    examples.append("from pynomaly_detection import AutoMLService")
    examples.append("")
    examples.append("# Automatic algorithm selection and optimization")
    examples.append("automl = AutoMLService()")
    examples.append("result = automl.auto_detect(data)")
    examples.append("```")
    examples.append("")
    
    # Example 3: Model Persistence
    examples.append("## Example 3: Model Persistence")
    examples.append("")
    examples.append("### Phase 1 (Old)")
    examples.append("```python")
    examples.append("import pickle")
    examples.append("import json")
    examples.append("")
    examples.append("# Manual model saving")
    examples.append("with open('model.pkl', 'wb') as f:")
    examples.append("    pickle.dump(model, f)")
    examples.append("    ")
    examples.append("# Manual metadata tracking")
    examples.append("metadata = {'algorithm': 'iforest', 'created': '2024-01-01'}")
    examples.append("with open('metadata.json', 'w') as f:")
    examples.append("    json.dump(metadata, f)")
    examples.append("```")
    examples.append("")
    examples.append("### Phase 2 (New)")
    examples.append("```python")
    examples.append("from pynomaly_detection import ModelPersistence")
    examples.append("")
    examples.append("# Enterprise-grade model management")
    examples.append("persistence = ModelPersistence()")
    examples.append("model_id = persistence.save_model(")
    examples.append("    model_data=model,")
    examples.append("    training_data=data,")
    examples.append("    algorithm='iforest',")
    examples.append("    performance_metrics={'accuracy': 0.95}")
    examples.append(")")
    examples.append("```")
    examples.append("")
    
    # Example 4: Monitoring
    examples.append("## Example 4: Monitoring and Alerting")
    examples.append("")
    examples.append("### Phase 1 (Old)")
    examples.append("```python")
    examples.append("import logging")
    examples.append("")
    examples.append("# Basic logging")
    examples.append("logging.info(f'Detected {n_anomalies} anomalies')")
    examples.append("```")
    examples.append("")
    examples.append("### Phase 2 (New)")
    examples.append("```python")
    examples.append("from pynomaly_detection import MonitoringAlertingSystem")
    examples.append("")
    examples.append("# Comprehensive monitoring")
    examples.append("monitoring = MonitoringAlertingSystem()")
    examples.append("monitoring.record_detection_result(result, processing_time, 'production')")
    examples.append("monitoring.start_background_monitoring()")
    examples.append("```")
    examples.append("")
    
    return "\n".join(examples)


# Convenience functions for migration
def print_migration_status() -> None:
    """Print current migration status."""
    status = check_migration_status()
    print(f"Migration Status: {status['migration_status']}")
    print(f"Migration Score: {status['migration_score']:.1f}%")
    print(f"Message: {status['message']}")
    print("\nNext Steps:")
    for step in status['next_steps']:
        print(f"  - {step}")


def print_migration_examples() -> None:
    """Print migration examples."""
    examples = generate_migration_examples()
    print(examples)


if __name__ == "__main__":
    print("=== PYNOMALY DETECTION PHASE 2 MIGRATION UTILITY ===")
    print()
    print_migration_status()
    print()
    print_migration_examples()