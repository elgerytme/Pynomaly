#!/usr/bin/env python3
"""
Basic ecosystem integration example.

This example demonstrates how to set up and use the ecosystem integration
framework with multiple partners including Databricks, Snowflake, and DataDog.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd
from ecosystem.management.registry import PartnerRegistry
from ecosystem.connectors.databricks import DatabricksIntegration
from ecosystem.connectors.snowflake import SnowflakeIntegration
from ecosystem.core.interfaces import (
    IntegrationConfig, PartnerTier, PartnerCapability,
    ConnectionHealth, Event, EventType, EventPriority
)


async def setup_databricks_integration() -> DatabricksIntegration:
    """Set up Databricks integration."""
    config = IntegrationConfig(
        name="databricks-prod",
        platform="databricks",
        description="Production Databricks workspace for ML training",
        endpoint=os.getenv("DATABRICKS_WORKSPACE_URL", "https://your-workspace.cloud.databricks.com"),
        credentials={
            "access_token": os.getenv("DATABRICKS_TOKEN", "your-token"),
            "cluster_id": os.getenv("DATABRICKS_CLUSTER_ID", "your-cluster-id")
        },
        timeout_seconds=120,
        retry_attempts=3,
        tags={
            "environment": "production",
            "team": "ml-platform",
            "cost_center": "engineering"
        }
    )
    
    integration = DatabricksIntegration(config)
    return integration


async def setup_snowflake_integration() -> SnowflakeIntegration:
    """Set up Snowflake integration."""
    config = IntegrationConfig(
        name="snowflake-warehouse",
        platform="snowflake",
        description="Production Snowflake data warehouse",
        credentials={
            "account": os.getenv("SNOWFLAKE_ACCOUNT", "your-account"),
            "user": os.getenv("SNOWFLAKE_USER", "your-user"),
            "password": os.getenv("SNOWFLAKE_PASSWORD", "your-password"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            "database": os.getenv("SNOWFLAKE_DATABASE", "MLOPS_DB"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
        },
        timeout_seconds=60,
        retry_attempts=3,
        tags={
            "environment": "production",
            "team": "data-platform",
            "cost_center": "engineering"
        }
    )
    
    integration = SnowflakeIntegration(config)
    return integration


async def demonstrate_basic_integration():
    """Demonstrate basic integration setup and operations."""
    print("🚀 Setting up Ecosystem Integration Framework")
    
    # Initialize partner registry
    registry = PartnerRegistry({
        "monitoring_interval_minutes": 5
    })
    
    try:
        # Set up integrations
        print("\n📋 Setting up integrations...")
        databricks = await setup_databricks_integration()
        snowflake = await setup_snowflake_integration()
        
        # Register partners in registry
        print("\n🤝 Registering partners...")
        
        await registry.register_partner(
            name="databricks-prod",
            partner=databricks,  # In real implementation, this would be wrapped
            integration=databricks,
            tier=PartnerTier.ENTERPRISE,
            capabilities={
                PartnerCapability.DATA_PROCESSING,
                PartnerCapability.ML_TRAINING,
                PartnerCapability.EXPERIMENT_TRACKING,
                PartnerCapability.MODEL_REGISTRY
            }
        )
        
        await registry.register_partner(
            name="snowflake-warehouse",
            partner=snowflake,  # In real implementation, this would be wrapped
            integration=snowflake,
            tier=PartnerTier.ENTERPRISE,
            capabilities={
                PartnerCapability.DATA_STORAGE,
                PartnerCapability.DATA_PROCESSING,
                PartnerCapability.FEATURE_STORE,
                PartnerCapability.DATA_CATALOG
            }
        )
        
        # Test connections
        print("\n🔗 Testing connections...")
        health_results = await registry.check_all_health()
        
        for partner_name, health in health_results.items():
            status_emoji = "✅" if health == ConnectionHealth.HEALTHY else "⚠️" if health == ConnectionHealth.DEGRADED else "❌"
            print(f"  {status_emoji} {partner_name}: {health.value}")
        
        # Get health summary
        health_summary = await registry.get_health_summary()
        print(f"\n📊 Overall Health: {health_summary['health_percentage']:.1f}%")
        print(f"   Healthy: {health_summary['healthy']}")
        print(f"   Degraded: {health_summary['degraded']}")
        print(f"   Unhealthy: {health_summary['unhealthy']}")
        
        return registry, databricks, snowflake
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        raise


async def demonstrate_data_pipeline(registry: PartnerRegistry, databricks: DatabricksIntegration, snowflake: SnowflakeIntegration):
    """Demonstrate a basic data pipeline using multiple integrations."""
    print("\n🔄 Demonstrating Data Pipeline")
    
    try:
        # Step 1: Create sample data
        print("\n1️⃣ Creating sample data...")
        sample_data = pd.DataFrame({
            'user_id': range(1, 101),
            'feature_1': pd.np.random.normal(0, 1, 100),
            'feature_2': pd.np.random.normal(0, 1, 100),
            'label': pd.np.random.choice([0, 1], 100),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H')
        })
        
        print(f"   Created dataset with {len(sample_data)} rows")
        
        # Step 2: Store data in Snowflake
        print("\n2️⃣ Storing data in Snowflake...")
        snowflake_success = await snowflake.send_data(
            data=sample_data,
            destination="ML_TRAINING_DATA",
            options={
                "auto_create_table": True,
                "overwrite": True
            }
        )
        
        if snowflake_success:
            print("   ✅ Data stored in Snowflake successfully")
        else:
            print("   ❌ Failed to store data in Snowflake")
        
        # Step 3: Upload data to Databricks for processing
        print("\n3️⃣ Uploading data to Databricks...")
        databricks_success = await databricks.send_data(
            data=sample_data.to_json(orient='records'),
            destination="dbfs:/tmp/ml_training_data.json",
            format_type="json"
        )
        
        if databricks_success:
            print("   ✅ Data uploaded to Databricks successfully")
        else:
            print("   ❌ Failed to upload data to Databricks")
        
        return sample_data
        
    except Exception as e:
        print(f"❌ Data pipeline failed: {e}")
        return None


async def demonstrate_ml_workflow(databricks: DatabricksIntegration, sample_data: pd.DataFrame):
    """Demonstrate ML workflow with experiment tracking."""
    print("\n🧪 Demonstrating ML Workflow")
    
    try:
        # Step 1: Create experiment
        print("\n1️⃣ Creating ML experiment...")
        experiment_id = await databricks.create_experiment(
            name="fraud-detection-demo",
            description="Demo fraud detection experiment",
            tags={
                "team": "ml-platform",
                "project": "fraud-detection",
                "demo": "true"
            }
        )
        
        if experiment_id:
            print(f"   ✅ Experiment created with ID: {experiment_id}")
        else:
            print("   ❌ Failed to create experiment")
            return
        
        # Step 2: Start training run
        print("\n2️⃣ Starting training run...")
        run_id = await databricks.create_run(
            experiment_id=experiment_id,
            run_name="demo-run-baseline",
            tags={
                "algorithm": "random_forest",
                "data_version": "v1.0"
            }
        )
        
        if run_id:
            print(f"   ✅ Training run started with ID: {run_id}")
        else:
            print("   ❌ Failed to start training run")
            return
        
        # Step 3: Log parameters
        print("\n3️⃣ Logging parameters...")
        parameters = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "random_state": 42,
            "data_size": len(sample_data)
        }
        
        param_success = await databricks.log_parameters(run_id, parameters)
        if param_success:
            print("   ✅ Parameters logged successfully")
        else:
            print("   ❌ Failed to log parameters")
        
        # Step 4: Simulate training and log metrics
        print("\n4️⃣ Simulating training and logging metrics...")
        
        # Simulate training epochs
        for epoch in range(1, 6):
            metrics = {
                "accuracy": 0.7 + (epoch * 0.05) + pd.np.random.normal(0, 0.01),
                "precision": 0.65 + (epoch * 0.06) + pd.np.random.normal(0, 0.01),
                "recall": 0.6 + (epoch * 0.07) + pd.np.random.normal(0, 0.01),
                "f1_score": 0.62 + (epoch * 0.065) + pd.np.random.normal(0, 0.01)
            }
            
            metrics_success = await databricks.log_metrics(run_id, metrics, step=epoch)
            if metrics_success:
                print(f"   📈 Epoch {epoch} metrics logged: accuracy={metrics['accuracy']:.3f}")
            else:
                print(f"   ❌ Failed to log metrics for epoch {epoch}")
            
            # Small delay to simulate training time
            await asyncio.sleep(1)
        
        # Step 5: Log final model artifact (simulated)
        print("\n5️⃣ Logging model artifact...")
        model_metadata = {
            "model_type": "RandomForestClassifier",
            "sklearn_version": "1.3.0",
            "training_time": "120 seconds",
            "final_accuracy": 0.95
        }
        
        artifact_success = await databricks.log_artifact(
            run_id=run_id,
            artifact_path="model_metadata.json",
            artifact_data=model_metadata,
            artifact_type="json"
        )
        
        if artifact_success:
            print("   ✅ Model artifact logged successfully")
        else:
            print("   ❌ Failed to log model artifact")
        
        # Step 6: Register model
        print("\n6️⃣ Registering model...")
        model_version = await databricks.register_model(
            model_name="fraud-detector-demo",
            run_id=run_id,
            artifact_path="model",
            description="Demo fraud detection model",
            tags={
                "algorithm": "random_forest",
                "performance": "baseline"
            }
        )
        
        if model_version:
            print(f"   ✅ Model registered with version: {model_version}")
        else:
            print("   ❌ Failed to register model")
        
        return experiment_id, run_id, model_version
        
    except Exception as e:
        print(f"❌ ML workflow failed: {e}")
        return None, None, None


async def demonstrate_monitoring(registry: PartnerRegistry):
    """Demonstrate monitoring and analytics capabilities."""
    print("\n📊 Demonstrating Monitoring & Analytics")
    
    try:
        # Check partnership capabilities
        print("\n1️⃣ Partnership Capabilities Analysis:")
        capabilities = await registry.get_available_capabilities()
        capability_coverage = await registry.get_capability_coverage()
        
        print(f"   Total capabilities available: {len(capabilities)}")
        for capability, partners in capability_coverage.items():
            print(f"   📋 {capability.value}: {', '.join(partners)}")
        
        # Generate usage report
        print("\n2️⃣ Usage Report:")
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        usage_report = await registry.generate_usage_report(
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"   Report Period: {usage_report['period']['start_date']} to {usage_report['period']['end_date']}")
        print(f"   Total Partners: {usage_report['summary']['total_partners']}")
        print(f"   Active Partners: {usage_report['summary']['active_partners']}")
        print(f"   Total API Calls: {usage_report['summary']['total_api_calls']}")
        print(f"   Total Cost: ${usage_report['summary']['total_cost_usd']:.2f}")
        
        # Check for expiring contracts
        print("\n3️⃣ Contract Management:")
        expiring_contracts = await registry.get_expiring_contracts(days_threshold=90)
        
        if expiring_contracts:
            print("   ⚠️ Contracts expiring soon:")
            for contract in expiring_contracts:
                print(f"   - {contract['partner_name']}: {contract['days_until_expiry']} days")
        else:
            print("   ✅ No contracts expiring within 90 days")
        
    except Exception as e:
        print(f"❌ Monitoring demonstration failed: {e}")


def setup_health_monitoring(registry: PartnerRegistry):
    """Set up health monitoring callbacks."""
    print("\n💓 Setting up Health Monitoring")
    
    def on_health_change(partner_name: str, health: ConnectionHealth):
        status_emoji = "✅" if health == ConnectionHealth.HEALTHY else "⚠️" if health == ConnectionHealth.DEGRADED else "❌"
        print(f"   {status_emoji} Health Alert: {partner_name} is now {health.value}")
    
    registry.add_health_callback(on_health_change)
    print("   Health monitoring callbacks configured")


async def cleanup_demo_resources(databricks: DatabricksIntegration, experiment_id: str):
    """Clean up demo resources."""
    print("\n🧹 Cleaning up demo resources...")
    
    try:
        if experiment_id:
            cleanup_success = await databricks.delete_experiment(experiment_id)
            if cleanup_success:
                print("   ✅ Demo experiment deleted")
            else:
                print("   ⚠️ Failed to delete demo experiment (may not exist)")
    except Exception as e:
        print(f"   ⚠️ Cleanup warning: {e}")


async def main():
    """Main demonstration function."""
    print("🎯 Ecosystem Integration Framework Demo")
    print("=" * 50)
    
    registry = None
    experiment_id = None
    
    try:
        # Setup phase
        registry, databricks, snowflake = await demonstrate_basic_integration()
        
        # Set up monitoring
        setup_health_monitoring(registry)
        
        # Data pipeline demonstration
        sample_data = await demonstrate_data_pipeline(registry, databricks, snowflake)
        
        if sample_data is not None:
            # ML workflow demonstration
            experiment_id, run_id, model_version = await demonstrate_ml_workflow(databricks, sample_data)
        
        # Monitoring demonstration
        await demonstrate_monitoring(registry)
        
        print("\n🎉 Demo completed successfully!")
        print("\nTo run this demo:")
        print("1. Set environment variables for your credentials")
        print("2. Ensure you have access to Databricks and Snowflake instances")
        print("3. Run: python examples/basic_integration.py")
        
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        
    finally:
        # Cleanup
        if registry and databricks and experiment_id:
            await cleanup_demo_resources(databricks, experiment_id)
        
        if registry:
            await registry.stop_monitoring()
            print("\n👋 Registry monitoring stopped")


if __name__ == "__main__":
    # Set up environment variables (in real usage, these would be set externally)
    demo_env_vars = {
        "DATABRICKS_WORKSPACE_URL": "https://demo-workspace.cloud.databricks.com",
        "DATABRICKS_TOKEN": "demo-token",
        "DATABRICKS_CLUSTER_ID": "demo-cluster",
        "SNOWFLAKE_ACCOUNT": "demo-account",
        "SNOWFLAKE_USER": "demo-user", 
        "SNOWFLAKE_PASSWORD": "demo-password",
        "SNOWFLAKE_WAREHOUSE": "DEMO_WH",
        "SNOWFLAKE_DATABASE": "DEMO_DB"
    }
    
    for key, value in demo_env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    
    # Run the demo
    asyncio.run(main())