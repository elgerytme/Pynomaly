#!/usr/bin/env python3
"""
Hexagonal Architecture Demonstration

This example shows how to use the new hexagonal architecture
implementation in the machine_learning package.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add package to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from machine_learning.infrastructure.container import Container, ContainerConfig
from machine_learning.domain.services.refactored_automl_service import AutoMLService
from machine_learning.domain.interfaces.automl_operations import OptimizationConfig, AlgorithmType
from machine_learning.domain.interfaces.explainability_operations import (
    ExplanationRequest, ExplanationMethod, ExplanationScope
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockDataset:
    """Mock dataset for demonstration purposes."""
    
    def __init__(self, name: str, size: int = 1000):
        self.name = name
        self.size = size
        self.data = [[i, i*2, i*3] for i in range(size)]  # Mock numerical data
    
    def __len__(self):
        return self.size
    
    def __repr__(self):
        return f"MockDataset(name='{self.name}', size={self.size})"


async def demonstrate_basic_automl():
    """Demonstrate basic AutoML functionality with clean architecture."""
    logger.info("🚀 Starting Basic AutoML Demonstration")
    
    # 1. Configure the container with desired integrations
    config = ContainerConfig(
        enable_sklearn_automl=True,    # Try to use scikit-learn
        enable_optuna_optimization=True,  # Try to use Optuna
        enable_distributed_tracing=True,  # Enable tracing
        tracing_backend="local",       # Use local tracing backend
        log_level="INFO"
    )
    
    # 2. Create container and let it wire dependencies
    container = Container(config)
    logger.info("✅ Container configured with dependencies")
    
    # 3. Get the AutoML service (fully configured with all dependencies)
    automl_service = container.get(AutoMLService)
    logger.info("✅ AutoML service retrieved from container")
    
    # 4. Create mock dataset
    dataset = MockDataset("demo_dataset", 500)
    logger.info(f"📊 Created dataset: {dataset}")
    
    # 5. Configure optimization
    optimization_config = OptimizationConfig(
        max_trials=10,
        algorithms_to_test=[
            AlgorithmType.ISOLATION_FOREST,
            AlgorithmType.LOCAL_OUTLIER_FACTOR,
            AlgorithmType.ONE_CLASS_SVM
        ]
    )
    logger.info("⚙️ Optimization configured for 3 algorithms with 10 trials")
    
    # 6. Run optimization (this will use adapters or stubs depending on availability)
    logger.info("🔄 Starting model optimization...")
    result = await automl_service.optimize_prediction(
        dataset=dataset,
        optimization_config=optimization_config
    )
    
    # 7. Display results
    logger.info("✅ Optimization completed!")
    logger.info(f"   Best Algorithm: {result.best_algorithm_type.value}")
    logger.info(f"   Best Score: {result.best_score:.4f}")
    logger.info(f"   Total Trials: {result.total_trials}")
    logger.info(f"   Optimization Time: {result.optimization_time_seconds:.2f}s")
    
    if result.recommendations:
        logger.info("💡 Recommendations:")
        for category, recommendations in result.recommendations.items():
            logger.info(f"   {category.title()}:")
            for rec in recommendations:
                logger.info(f"     - {rec}")
    
    return result


async def demonstrate_algorithm_selection():
    """Demonstrate intelligent algorithm selection."""
    logger.info("🎯 Starting Algorithm Selection Demonstration")
    
    # Create container with minimal configuration
    container = Container(ContainerConfig())
    automl_service = container.get(AutoMLService)
    
    # Create different types of datasets
    datasets = [
        MockDataset("small_dataset", 100),
        MockDataset("medium_dataset", 5000),  
        MockDataset("large_dataset", 50000)
    ]
    
    for dataset in datasets:
        logger.info(f"📊 Analyzing dataset: {dataset.name} (size: {dataset.size})")
        
        # Quick algorithm selection
        algorithm, config = await automl_service.auto_select_algorithm(
            dataset, quick_mode=True
        )
        
        logger.info(f"   Recommended: {algorithm.value}")
        logger.info(f"   Configuration: {config.parameters}")
        
        # Get optimization recommendations
        recommendations = await automl_service.get_optimization_recommendations(dataset)
        logger.info(f"   Top recommendation: {recommendations.get('algorithms', ['None'])[0]}")


async def demonstrate_graceful_degradation():
    """Demonstrate graceful degradation when external libraries unavailable."""
    logger.info("🎭 Starting Graceful Degradation Demonstration")
    
    # Force use of stubs by disabling external integrations
    config = ContainerConfig(
        enable_sklearn_automl=False,    # Force use of stubs
        enable_optuna_optimization=False,
        enable_distributed_tracing=False,
        log_level="WARNING"  # Reduce noise from stub warnings
    )
    
    container = Container(config)
    automl_service = container.get(AutoMLService)
    
    logger.info("⚠️ External libraries disabled - using stubs")
    
    # Run same optimization as before
    dataset = MockDataset("stub_demo", 1000)
    result = await automl_service.optimize_prediction(dataset)
    
    logger.info("✅ Optimization completed with stubs!")
    logger.info(f"   Stub Result - Algorithm: {result.best_algorithm_type.value}")
    logger.info(f"   Stub Result - Score: {result.best_score:.4f}")
    logger.info("   Note: This used fallback implementations - real libraries would give better results")


async def demonstrate_monitoring_integration():
    """Demonstrate monitoring and observability integration."""
    logger.info("📊 Starting Monitoring Integration Demonstration")
    
    # Enable monitoring with tracing
    config = ContainerConfig(
        enable_distributed_tracing=True,
        tracing_backend="local",
        enable_prometheus_monitoring=True
    )
    
    container = Container(config)
    
    # Show configuration summary
    summary = container.get_configuration_summary()
    logger.info("📋 Container Configuration:")
    logger.info(f"   Environment: {summary['environment']}")
    logger.info(f"   AutoML: sklearn={summary['automl']['sklearn_enabled']}")
    logger.info(f"   Monitoring: tracing={summary['monitoring']['tracing_enabled']}")
    logger.info(f"   Services: {len(summary['registered_services']['singletons'])} singletons")
    
    # Run traced operation
    automl_service = container.get(AutoMLService)
    dataset = MockDataset("monitored_demo", 250)
    
    logger.info("🔍 Running monitored optimization (check logs for trace info)...")
    result = await automl_service.optimize_prediction(dataset)
    
    logger.info("✅ Monitored optimization completed")
    logger.info("   Trace information should appear in logs above")


async def main():
    """Main demonstration function."""
    print("=" * 80)
    print("🏗️  HEXAGONAL ARCHITECTURE DEMONSTRATION")
    print("   Machine Learning Package - Clean DDD Implementation")
    print("=" * 80)
    print()
    
    try:
        # Run demonstrations
        await demonstrate_basic_automl()
        print()
        
        await demonstrate_algorithm_selection()
        print()
        
        await demonstrate_graceful_degradation() 
        print()
        
        await demonstrate_monitoring_integration()
        print()
        
        print("=" * 80)
        print("✅ All demonstrations completed successfully!")
        print("   The hexagonal architecture provides:")
        print("   • Clean separation of domain logic and infrastructure")
        print("   • Easy dependency injection and configuration")
        print("   • Graceful degradation when external services unavailable")
        print("   • Comprehensive monitoring and observability")
        print("   • Technology independence and flexibility")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))