#!/usr/bin/env python3
"""
Basic anomaly detection example using the Python SDK.

This example demonstrates:
- Simple anomaly detection
- Error handling
- Configuration options
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the SDK to the path (for local development)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "sdk_core" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "clients" / "anomaly_detection_client" / "src"))

from anomaly_detection_client import (
    AnomalyDetectionClient,
    AnomalyDetectionSyncClient,
    ClientConfig,
    Environment,
    ValidationError,
    RateLimitError,
    ServerError,
)


def create_sample_data():
    """Create sample data with some obvious anomalies."""
    import random
    
    # Normal data points (around 0-10 range)
    normal_data = []
    for _ in range(100):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10) 
        normal_data.append([x, y])
    
    # Add some clear anomalies (far from normal range)
    anomalies = [
        [100, 100],  # Far away anomaly
        [-50, -50],  # Another far away anomaly
        [200, 5],    # High X value
        [5, 200],    # High Y value
    ]
    
    # Combine normal data with anomalies
    all_data = normal_data + anomalies
    
    # Shuffle to make anomalies less obvious
    random.shuffle(all_data)
    
    return all_data


async def async_example():
    """Demonstrate async API usage."""
    print("🔍 Async Anomaly Detection Example")
    print("=" * 40)
    
    # Configure client
    config = ClientConfig.for_environment(
        Environment.LOCAL,  # Use local development environment
        api_key=os.getenv("ANOMALY_DETECTION_API_KEY", "demo-api-key"),
        timeout=30.0,
        max_retries=3
    )
    
    async with AnomalyDetectionClient(config=config) as client:
        try:
            # Check service health first
            print("📊 Checking service health...")
            health = await client.health_check()
            print(f"   Status: {health.get('status', 'unknown')}")
            print()
            
            # Get available algorithms
            print("🤖 Available algorithms:")
            try:
                algorithms = await client.get_algorithms()
                for algo in algorithms[:3]:  # Show first 3
                    print(f"   - {algo.display_name}: {algo.description}")
            except Exception as e:
                print(f"   Could not fetch algorithms: {e}")
            print()
            
            # Create sample data
            data = create_sample_data()
            print(f"📈 Generated {len(data)} data points")
            
            # Perform anomaly detection
            print("🔍 Detecting anomalies...")
            result = await client.detect(
                data=data,
                algorithm="isolation_forest",
                contamination=0.1,  # Expect 10% to be anomalies
                parameters={"n_estimators": 100}
            )
            
            # Display results
            print(f"✅ Detection completed in {result.processing_time_ms:.2f}ms")
            print(f"   Total samples: {result.total_samples}")
            print(f"   Anomalies found: {result.anomaly_count}")
            print(f"   Anomaly indices: {result.anomalies[:10]}...")  # Show first 10
            
            if result.scores:
                avg_score = sum(result.scores) / len(result.scores)
                max_score = max(result.scores)
                print(f"   Average anomaly score: {avg_score:.4f}")
                print(f"   Maximum anomaly score: {max_score:.4f}")
            print()
            
            # Try ensemble detection
            print("🎯 Ensemble detection with multiple algorithms...")
            try:
                ensemble_result = await client.detect_ensemble(
                    data=data[:50],  # Use smaller dataset for ensemble
                    algorithms=["isolation_forest", "one_class_svm"],
                    voting_strategy="majority",
                    contamination=0.1
                )
                
                print(f"✅ Ensemble detection completed in {ensemble_result.processing_time_ms:.2f}ms")
                print(f"   Algorithms used: {', '.join(ensemble_result.algorithms_used)}")
                print(f"   Anomalies found: {ensemble_result.anomaly_count}")
                print(f"   Voting strategy: {ensemble_result.voting_strategy}")
                
            except Exception as e:
                print(f"   Ensemble detection failed: {e}")
            print()
            
            # Try model training
            print("🎓 Training a custom model...")
            try:
                training_result = await client.train_model(
                    data=data[:80],  # Use 80% for training
                    algorithm="isolation_forest",
                    name=f"demo_model_{int(asyncio.get_event_loop().time())}",
                    description="Demo model trained from example script",
                    contamination=0.1,
                    parameters={"n_estimators": 50}
                )
                
                model = training_result.model
                print(f"✅ Model trained successfully")
                print(f"   Model ID: {model.id}")
                print(f"   Model name: {model.name}")
                print(f"   Algorithm: {model.algorithm}")
                print(f"   Training time: {training_result.training_time_ms:.2f}ms")
                
                # Use the trained model for prediction
                print("🔮 Making predictions with trained model...")
                prediction_result = await client.predict(
                    data=data[80:],  # Use remaining 20% for prediction
                    model_id=model.id
                )
                
                print(f"✅ Prediction completed")
                print(f"   Anomalies found: {prediction_result.anomaly_count}")
                print(f"   Processing time: {prediction_result.processing_time_ms:.2f}ms")
                
            except Exception as e:
                print(f"   Model training/prediction failed: {e}")
            
        except ValidationError as e:
            print(f"❌ Validation error: {e.message}")
            if e.details:
                for detail in e.details:
                    print(f"   - {detail}")
                    
        except RateLimitError as e:
            print(f"⏳ Rate limited: {e.message}")
            print(f"   Retry after: {e.retry_after} seconds")
            
        except ServerError as e:
            print(f"🚨 Server error: {e.message}")
            
        except Exception as e:
            print(f"💥 Unexpected error: {e}")


def sync_example():
    """Demonstrate synchronous API usage."""
    print("\n" + "🔍 Sync Anomaly Detection Example")
    print("=" * 40)
    
    with AnomalyDetectionSyncClient(
        api_key=os.getenv("ANOMALY_DETECTION_API_KEY", "demo-api-key"),
        environment=Environment.LOCAL
    ) as client:
        try:
            # Create sample data
            data = create_sample_data()[:20]  # Use smaller dataset for sync example
            print(f"📈 Generated {len(data)} data points")
            
            # Perform detection
            print("🔍 Detecting anomalies (synchronously)...")
            result = client.detect(
                data=data,
                algorithm="isolation_forest",
                contamination=0.15
            )
            
            print(f"✅ Detection completed")
            print(f"   Anomalies found: {result.anomaly_count}")
            print(f"   Total samples: {result.total_samples}")
            print(f"   Processing time: {result.processing_time_ms:.2f}ms")
            
        except Exception as e:
            print(f"💥 Error in sync example: {e}")


def main():
    """Run all examples."""
    print("🚀 Platform Anomaly Detection SDK Examples")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.getenv("ANOMALY_DETECTION_API_KEY")
    if not api_key:
        print("⚠️  No API key found in ANOMALY_DETECTION_API_KEY environment variable")
        print("   Using demo API key - some features may not work")
        print()
    
    # Run async example
    try:
        asyncio.run(async_example())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n💥 Async example failed: {e}")
    
    # Run sync example
    try:
        sync_example()
    except Exception as e:
        print(f"\n💥 Sync example failed: {e}")
    
    print("\n✨ Examples completed!")


if __name__ == "__main__":
    main()