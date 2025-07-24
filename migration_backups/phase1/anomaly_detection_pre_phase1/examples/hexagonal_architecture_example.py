"""Example demonstrating the new hexagonal architecture.

This example shows how to use the refactored anomaly detection package
with proper dependency injection and clean architecture principles.
"""

import asyncio
import logging
import numpy as np
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function demonstrating the new architecture."""
    
    # 1. Configure the dependency injection container
    from anomaly_detection.infrastructure.container.container import (
        configure_container, get_container
    )
    
    logger.info("Configuring dependency injection container...")
    
    # Configure with both ML and MLOps integration
    # In production, these would be actual configuration objects
    ml_config = {
        "auto_optimization": True,
        "default_timeout": 300,
        "parallel_jobs": 4,
    }
    
    mlops_config = {
        "tracking_uri": "http://localhost:5000",
        "experiment_prefix": "anomaly_detection",
        "auto_log_metrics": True,
    }
    
    # Configure the container - this will use stubs if packages aren't available
    container = configure_container(
        enable_ml=True,  # Will fall back to stubs if machine_learning package not available
        enable_mlops=True,  # Will fall back to stubs if mlops package not available
        ml_config=ml_config,
        mlops_config=mlops_config
    )
    
    logger.info("Container configured successfully")
    
    # 2. Get application service from container
    from anomaly_detection.application.services.model_training_service import (
        ModelTrainingApplicationService
    )
    from anomaly_detection.domain.interfaces.ml_operations import MLModelTrainingPort
    from anomaly_detection.domain.interfaces.mlops_operations import (
        MLOpsExperimentTrackingPort, MLOpsModelRegistryPort
    )
    
    # Resolve dependencies from container
    ml_training = container.get(MLModelTrainingPort)
    experiment_tracking = container.get(MLOpsExperimentTrackingPort)
    model_registry = container.get(MLOpsModelRegistryPort)
    
    # Create application service
    training_service = ModelTrainingApplicationService(
        ml_training=ml_training,
        experiment_tracking=experiment_tracking,
        model_registry=model_registry
    )
    
    logger.info("Application service created successfully")
    
    # 3. Create sample data for demonstration
    from anomaly_detection.domain.entities.dataset import Dataset
    
    # Generate synthetic anomaly detection data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (800, 5))  # 800 normal samples, 5 features
    anomaly_data = np.random.normal(3, 1, (200, 5))  # 200 anomalous samples
    
    # Combine and create dataset
    all_data = np.vstack([normal_data, anomaly_data])
    feature_names = [f"feature_{i}" for i in range(5)]
    
    training_data = Dataset(
        data=all_data[:700],  # First 700 samples for training
        feature_names=feature_names,
        metadata={
            "source": "synthetic",
            "normal_samples": 560,
            "anomaly_samples": 140,
        }
    )
    
    validation_data = Dataset(
        data=all_data[700:900],  # Next 200 samples for validation
        feature_names=feature_names,
        metadata={
            "source": "synthetic",
            "normal_samples": 160,
            "anomaly_samples": 40,
        }
    )
    
    test_data = Dataset(
        data=all_data[900:],  # Last 100 samples for testing
        feature_names=feature_names,
        metadata={
            "source": "synthetic",  
            "normal_samples": 80,
            "anomaly_samples": 20,
        }
    )
    
    logger.info("Sample datasets created")
    
    # 4. Get supported algorithms
    logger.info("Getting supported algorithms...")
    algorithms = await training_service.get_supported_algorithms()
    logger.info(f"Supported algorithms: {algorithms}")
    
    # 5. Train a model using the new architecture
    if algorithms:
        algorithm = algorithms[0]  # Use first available algorithm
        
        logger.info(f"Training model using algorithm: {algorithm}")
        
        # Get algorithm parameters
        params_schema = await training_service.get_algorithm_parameters(algorithm)
        logger.info(f"Algorithm parameters: {params_schema}")
        
        # Use default parameters for demonstration
        parameters = {
            "contamination": 0.2,  # Expect 20% anomalies
            "random_state": 42,
        }
        
        # Train the model
        from anomaly_detection.domain.interfaces.ml_operations import OptimizationObjective
        
        training_result = await training_service.train_anomaly_detection_model(
            algorithm_name=algorithm,
            training_data=training_data,
            validation_data=validation_data,
            parameters=parameters,
            experiment_name=f"hexagonal_architecture_demo_{algorithm}",
            optimization_objective=OptimizationObjective.MAXIMIZE_F1,
            register_model=True,
            created_by="demo_user"
        )
        
        logger.info("Training completed!")
        logger.info(f"Training success: {training_result['success']}")
        
        if training_result['success']:
            logger.info(f"Model ID: {training_result['model']['id']}")
            logger.info(f"Training metrics: {training_result['training']['metrics']}")
            logger.info(f"Experiment ID: {training_result['experiment']['experiment_id']}")
            logger.info(f"Model registered: {training_result['registry']['registered']}")
            
            # 6. Evaluate the model
            if training_result['registry']['registered']:
                model_id = training_result['registry']['model_id']
                
                logger.info(f"Evaluating model: {model_id}")
                
                evaluation_result = await training_service.evaluate_trained_model(
                    model_id=model_id,
                    test_data=test_data,
                    evaluation_metrics=["accuracy", "precision", "recall", "f1_score"],
                    experiment_name=f"evaluation_demo_{algorithm}",
                    created_by="demo_user"
                )
                
                logger.info(f"Evaluation success: {evaluation_result['success']}")
                if evaluation_result['success']:
                    logger.info(f"Evaluation metrics: {evaluation_result['metrics']}")
        else:
            logger.error(f"Training failed: {training_result.get('error', 'Unknown error')}")
    
    else:
        logger.warning("No algorithms available for training")
    
    # 7. Demonstrate the AB testing service with new architecture
    logger.info("Demonstrating AB testing service...")
    
    try:
        from anomaly_detection.domain.services.ab_testing_service import (
            ABTestingService, ABTestConfig, TestVariant, SplitType
        )
        
        # Create AB testing service with injected dependencies
        ab_testing_service = ABTestingService(
            experiment_tracking=experiment_tracking,
            model_registry=model_registry,
            ab_testing=None,  # Would be injected if analytics package available
            performance_analytics=None  # Would be injected if analytics package available
        )
        
        # Create a test configuration
        test_config = ABTestConfig(
            test_name="algorithm_comparison_demo",
            description="Compare two anomaly detection algorithms",
            variants=[
                TestVariant(
                    variant_id="variant_a",
                    name="Isolation Forest",
                    model_id="model_a",
                    model_version=1,
                    traffic_percentage=50.0,
                    description="Isolation Forest algorithm"
                ),
                TestVariant(
                    variant_id="variant_b", 
                    name="One Class SVM",
                    model_id="model_b",
                    model_version=1,
                    traffic_percentage=50.0,
                    description="One Class SVM algorithm"
                )
            ],
            split_type=SplitType.RANDOM,
            duration_days=7,
            significance_threshold=0.05,
            minimum_sample_size=1000
        )
        
        # Note: This would fail validation since the models don't exist,
        # but it demonstrates the new interface-based architecture
        logger.info("AB testing service created with dependency injection")
        
    except Exception as e:
        logger.info(f"AB testing demo skipped due to missing models: {str(e)}")
    
    logger.info("Demo completed successfully!")
    logger.info("The hexagonal architecture provides:")
    logger.info("- Clean separation of concerns")
    logger.info("- Dependency inversion through interfaces")
    logger.info("- Easy testing with stub implementations")
    logger.info("- Flexible integration with external packages")
    logger.info("- Maintainable and extensible codebase")


def demonstrate_architecture_benefits():
    """Demonstrate the benefits of the new architecture."""
    
    print("\n" + "="*60)
    print("HEXAGONAL ARCHITECTURE BENEFITS DEMONSTRATION")
    print("="*60)
    
    print("\n1. DEPENDENCY INVERSION:")
    print("   - Domain layer depends on abstractions (interfaces)")
    print("   - Infrastructure provides concrete implementations")
    print("   - Easy to swap implementations (e.g., different ML libraries)")
    
    print("\n2. TESTABILITY:")
    print("   - Can easily mock interfaces for unit testing")
    print("   - Stub implementations available when packages not installed")
    print("   - Clear contracts defined by interfaces")
    
    print("\n3. MAINTAINABILITY:")
    print("   - Clear separation between business logic and infrastructure")
    print("   - Changes to external packages don't affect domain logic")
    print("   - Single responsibility principle enforced")
    
    print("\n4. EXTENSIBILITY:")
    print("   - New algorithms can be added through adapters")
    print("   - Multiple MLOps platforms can be supported")
    print("   - Analytics capabilities can be plugged in")
    
    print("\n5. INTEGRATION:")
    print("   - Graceful degradation when external packages unavailable")
    print("   - Dependency injection container manages complexity")
    print("   - Application services orchestrate workflows")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    """Run the hexagonal architecture demonstration."""
    
    print("Anomaly Detection Package - Hexagonal Architecture Demo")
    print("This demo shows the new clean architecture implementation")
    
    demonstrate_architecture_benefits()
    
    print("\nRunning integration demo...")
    asyncio.run(main())