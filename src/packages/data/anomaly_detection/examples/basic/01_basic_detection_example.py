#!/usr/bin/env python3
"""
Example: Basic Anomaly Detection
Description: Simple demonstration of anomaly detection using the domain-driven architecture
Industry: General purpose
Difficulty: Beginner

This example shows how to:
1. Load and prepare data
2. Use the DetectionService 
3. Interpret results
4. Handle errors gracefully
"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add the package to the path for examples
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.entities.dataset import Dataset, DatasetMetadata


def generate_sample_data(n_samples: int = 1000, n_features: int = 5, contamination: float = 0.1):
    """Generate sample data with known anomalies."""
    print(f"📊 Generating {n_samples} samples with {n_features} features...")
    
    # Generate normal data
    np.random.seed(42)
    normal_data = np.random.multivariate_normal(
        mean=[0] * n_features,
        cov=np.eye(n_features),
        size=int(n_samples * (1 - contamination))
    )
    
    # Generate anomalous data (outliers)
    n_anomalies = int(n_samples * contamination)
    anomalous_data = np.random.multivariate_normal(
        mean=[3] * n_features,  # Shifted mean for clear anomalies
        cov=np.eye(n_features) * 0.5,
        size=n_anomalies
    )
    
    # Combine and shuffle
    all_data = np.vstack([normal_data, anomalous_data])
    indices = np.random.permutation(len(all_data))
    shuffled_data = all_data[indices]
    
    # True labels (1 for normal, -1 for anomaly)
    true_labels = np.ones(len(all_data))
    true_labels[-n_anomalies:] = -1
    true_labels = true_labels[indices]
    
    print(f"✅ Generated data: {len(normal_data)} normal + {n_anomalies} anomalous samples")
    
    return shuffled_data, true_labels


def main():
    """Main example function."""
    print("🚀 Starting Basic Anomaly Detection Example")
    print("=" * 50)
    
    try:
        # 1. Setup and configuration
        print("\n1️⃣ Setting up detection service...")
        detection_service = DetectionService()
        print("✅ Detection service initialized")
        
        # 2. Data preparation
        print("\n2️⃣ Preparing sample data...")
        data, true_labels = generate_sample_data(
            n_samples=1000,
            n_features=4,
            contamination=0.05  # 5% anomalies
        )
        
        # Create dataset metadata
        metadata = DatasetMetadata(
            name="basic_example_dataset",
            feature_names=[f"feature_{i}" for i in range(data.shape[1])],
            description="Synthetic dataset for basic anomaly detection example"
        )
        
        # Create dataset object
        dataset = Dataset(data=data, metadata=metadata)
        print(f"✅ Dataset created: {dataset.shape[0]} samples, {dataset.shape[1]} features")
        
        # 3. Anomaly detection
        print("\n3️⃣ Running anomaly detection...")
        
        # Test different algorithms
        algorithms = ["iforest", "lof", "ocsvm"]
        results = {}
        
        for algorithm in algorithms:
            print(f"\n🔍 Testing {algorithm.upper()} algorithm...")
            
            start_time = datetime.now()
            result = detection_service.detect_anomalies(
                data=data,
                algorithm=algorithm,
                contamination=0.05
            )
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            if result.success:
                results[algorithm] = result
                print(f"✅ {algorithm.upper()} completed in {processing_time:.2f}ms")
                print(f"   - Detected {result.anomaly_count} anomalies ({result.anomaly_rate:.1%})")
                print(f"   - Total samples processed: {result.total_samples}")
            else:
                print(f"❌ {algorithm.upper()} failed: {result.message}")
        
        # 4. Results analysis
        print("\n4️⃣ Analyzing results...")
        
        if results:
            print("\n📊 Algorithm Performance Comparison:")
            print(f"{'Algorithm':<12} {'Anomalies':<10} {'Rate':<8} {'Top Scores'}")
            print("-" * 50)
            
            for algo_name, result in results.items():
                top_scores = np.sort(result.anomaly_scores)[-3:][::-1]
                scores_str = ", ".join([f"{score:.3f}" for score in top_scores])
                
                print(f"{algo_name.upper():<12} {result.anomaly_count:<10} "
                      f"{result.anomaly_rate:.1%}:<8 {scores_str}")
        
        # 5. Detailed analysis of best result
        print("\n5️⃣ Detailed analysis...")
        
        if results:
            # Use Isolation Forest result for detailed analysis
            best_result = results.get("iforest", list(results.values())[0])
            
            print(f"\n🔍 Detailed analysis using {best_result.algorithm.upper()}:")
            print(f"Algorithm: {best_result.algorithm}")
            print(f"Parameters: {best_result.parameters}")
            print(f"Success: {best_result.success}")
            print(f"Processing time: {best_result.metadata.get('processing_time_ms', 0):.2f}ms")
            
            # Show top anomalies
            if len(best_result.anomaly_indices) > 0:
                print(f"\n🚨 Top 5 anomalies (by score):")
                top_anomaly_indices = np.argsort(best_result.anomaly_scores)[-5:][::-1]
                
                for i, idx in enumerate(top_anomaly_indices, 1):
                    score = best_result.anomaly_scores[idx]
                    sample = data[idx]
                    print(f"   {i}. Index {idx}: score={score:.4f}, "
                          f"values=[{', '.join([f'{v:.3f}' for v in sample[:3]])}...]")
            
            # Calculate accuracy if we have true labels
            predicted_labels = best_result.predictions
            true_anomaly_count = np.sum(true_labels == -1)
            detected_correctly = np.sum((predicted_labels == -1) & (true_labels == -1))
            
            print(f"\n📈 Accuracy Assessment:")
            print(f"True anomalies in dataset: {true_anomaly_count}")
            print(f"Detected anomalies: {best_result.anomaly_count}")
            print(f"Correctly detected: {detected_correctly}")
            print(f"Detection rate: {detected_correctly/true_anomaly_count:.1%}")
        
        # 6. Visualization (optional)
        print("\n6️⃣ Generating visualization...")
        try:
            import matplotlib.pyplot as plt
            
            if results and data.shape[1] >= 2:
                plt.figure(figsize=(12, 4))
                
                # Plot original data with true labels
                plt.subplot(1, 3, 1)
                normal_mask = true_labels == 1
                anomaly_mask = true_labels == -1
                
                plt.scatter(data[normal_mask, 0], data[normal_mask, 1], 
                           c='blue', alpha=0.6, label='Normal', s=20)
                plt.scatter(data[anomaly_mask, 0], data[anomaly_mask, 1], 
                           c='red', alpha=0.8, label='True Anomalies', s=30)
                plt.title('True Labels')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot detection results
                if "iforest" in results:
                    result = results["iforest"]
                    plt.subplot(1, 3, 2)
                    
                    normal_detected = result.predictions == 1
                    anomaly_detected = result.predictions == -1
                    
                    plt.scatter(data[normal_detected, 0], data[normal_detected, 1], 
                               c='lightblue', alpha=0.6, label='Normal', s=20)
                    plt.scatter(data[anomaly_detected, 0], data[anomaly_detected, 1], 
                               c='orange', alpha=0.8, label='Detected Anomalies', s=30)
                    plt.title(f'Isolation Forest Results')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                # Plot anomaly scores
                if "iforest" in results:
                    result = results["iforest"]
                    plt.subplot(1, 3, 3)
                    
                    scatter = plt.scatter(data[:, 0], data[:, 1], 
                                        c=result.anomaly_scores, 
                                        cmap='viridis', alpha=0.7, s=20)
                    plt.colorbar(scatter, label='Anomaly Score')
                    plt.title('Anomaly Scores')
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                plot_filename = f"basic_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                print(f"✅ Visualization saved as: {plot_filename}")
                
                # Show plot if running interactively
                try:
                    plt.show()
                except:
                    print("   (Plot display not available in this environment)")
                
        except ImportError:
            print("📊 Matplotlib not available - skipping visualization")
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
        
        print("\n🎉 Basic anomaly detection example completed successfully!")
        print("\n💡 Key takeaways:")
        print("   - Different algorithms may detect different anomalies")
        print("   - Isolation Forest is good for high-dimensional data")
        print("   - Local Outlier Factor works well for density-based anomalies")
        print("   - Always validate results with domain knowledge")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)