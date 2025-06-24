#!/usr/bin/env python3
"""
Demonstration of Pynomaly Autonomous Mode functionality.
Shows autonomous anomaly detection with sample data.
"""

import sys
import tempfile
import csv
import random
import asyncio
from pathlib import Path

# Add source directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

def create_demo_data():
    """Create demonstration dataset with known patterns."""
    temp_dir = Path(tempfile.mkdtemp(prefix="pynomaly_demo_"))
    
    # Create multiple datasets to showcase different scenarios
    datasets = {}
    
    # Dataset 1: Tabular data with clear anomalies
    csv_file = temp_dir / "tabular_anomalies.csv"
    random.seed(42)
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature1', 'feature2', 'feature3', 'feature4', 'label'])
        
        # Normal data cluster
        for i in range(90):
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
            x3 = random.gauss(0, 1)
            x4 = random.gauss(0, 1)
            writer.writerow([x1, x2, x3, x4, 'normal'])
        
        # Clear anomalies
        for i in range(10):
            x1 = random.uniform(-5, 5)
            x2 = random.uniform(-5, 5)
            x3 = random.uniform(-5, 5)
            x4 = random.uniform(-5, 5)
            writer.writerow([x1, x2, x3, x4, 'anomaly'])
    
    datasets['tabular'] = csv_file
    
    # Dataset 2: High-dimensional data
    high_dim_file = temp_dir / "high_dimensional.csv"
    with open(high_dim_file, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = [f'dim_{i}' for i in range(20)]
        writer.writerow(headers)
        
        # Normal high-dimensional data
        for i in range(80):
            row = [random.gauss(0, 1) for _ in range(20)]
            writer.writerow(row)
        
        # High-dimensional anomalies
        for i in range(20):
            row = [random.gauss(3, 0.5) for _ in range(20)]
            writer.writerow(row)
    
    datasets['high_dimensional'] = high_dim_file
    
    # Dataset 3: Mixed data types (simplified for CSV)
    mixed_file = temp_dir / "mixed_data.csv"
    with open(mixed_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['numeric1', 'numeric2', 'category', 'binary', 'score'])
        
        categories = ['A', 'B', 'C']
        
        # Normal mixed data
        for i in range(85):
            num1 = random.gauss(50, 10)
            num2 = random.gauss(25, 5)
            cat = random.choice(categories)
            binary = random.choice([0, 1])
            score = random.random() * 0.5  # Skewed distribution
            writer.writerow([num1, num2, cat, binary, score])
        
        # Anomalous mixed data
        for i in range(15):
            num1 = random.gauss(100, 20)  # Different distribution
            num2 = random.gauss(0, 10)    # Different distribution
            cat = random.choice(categories)
            binary = random.choice([0, 1])
            score = 0.5 + random.random() * 0.5  # Different distribution
            writer.writerow([num1, num2, cat, binary, score])
    
    datasets['mixed'] = mixed_file
    
    return temp_dir, datasets

async def demonstrate_autonomous_detection():
    """Demonstrate autonomous detection capabilities."""
    print("🤖 PYNOMALY AUTONOMOUS MODE DEMONSTRATION")
    print("="*60)
    
    # Import required components
    from pynomaly.application.services.autonomous_service import AutonomousDetectionService, AutonomousConfig
    from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
    from pynomaly.infrastructure.repositories.in_memory_repositories import (
        InMemoryDetectorRepository,
        InMemoryResultRepository
    )
    
    # Create demo data
    temp_dir, datasets = create_demo_data()
    
    try:
        # Setup autonomous service
        print("\n📋 Setting up autonomous detection service...")
        data_loaders = {"csv": CSVLoader()}
        
        service = AutonomousDetectionService(
            detector_repository=InMemoryDetectorRepository(),
            result_repository=InMemoryResultRepository(),
            data_loaders=data_loaders
        )
        print("✓ Service initialized")
        
        # Define different configurations for demonstration
        configs = {
            "Quick Analysis": AutonomousConfig(
                max_algorithms=1,
                auto_tune_hyperparams=False,
                confidence_threshold=0.7,
                verbose=True
            ),
            "Balanced Analysis": AutonomousConfig(
                max_algorithms=3,
                auto_tune_hyperparams=True,
                confidence_threshold=0.8,
                verbose=True
            ),
            "Comprehensive Analysis": AutonomousConfig(
                max_algorithms=5,
                auto_tune_hyperparams=True,
                confidence_threshold=0.9,
                save_results=True,
                verbose=True
            )
        }
        
        # Process each dataset with different configurations
        for dataset_name, dataset_path in datasets.items():
            print(f"\n📊 ANALYZING: {dataset_name.upper()} DATASET")
            print("-" * 50)
            
            # Choose configuration based on dataset complexity
            if dataset_name == "high_dimensional":
                config_name = "Comprehensive Analysis"
                config = configs[config_name]
            elif dataset_name == "mixed":
                config_name = "Balanced Analysis" 
                config = configs[config_name]
            else:
                config_name = "Quick Analysis"
                config = configs[config_name]
            
            print(f"Configuration: {config_name}")
            print(f"Data file: {dataset_path.name}")
            
            try:
                # Step 1: Auto-load data
                print("\n🔄 Step 1: Auto-loading data...")
                dataset = await service._auto_load_data(str(dataset_path), config)
                print(f"  ✓ Dataset loaded: {dataset.name}")
                print(f"  ✓ Shape: {dataset.data.shape[0]} rows × {dataset.data.shape[1]} columns")
                
                # Step 2: Profile data
                print("\n🔍 Step 2: Profiling data characteristics...")
                profile = await service._profile_data(dataset, config)
                print(f"  ✓ Samples: {profile.n_samples:,}")
                print(f"  ✓ Features: {profile.n_features}")
                print(f"  ✓ Numeric features: {profile.numeric_features}")
                print(f"  ✓ Categorical features: {profile.categorical_features}")
                print(f"  ✓ Complexity score: {profile.complexity_score:.3f}")
                print(f"  ✓ Recommended contamination: {profile.recommended_contamination:.1%}")
                print(f"  ✓ Missing values: {profile.missing_values_ratio:.1%}")
                
                # Step 3: Get algorithm recommendations
                print("\n🧠 Step 3: Generating algorithm recommendations...")
                recommendations = await service._recommend_algorithms(profile, config)
                print(f"  ✓ Generated {len(recommendations)} algorithm recommendations")
                
                print("\n  📋 Top Algorithm Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"    {i}. {rec.algorithm}")
                    print(f"       Confidence: {rec.confidence:.1%}")
                    print(f"       Reasoning: {rec.reasoning}")
                    print(f"       Expected Performance: {rec.expected_performance:.1%}")
                    print()
                
                # Step 4: Show what full autonomous detection would do
                print("🎯 Step 4: Autonomous detection preview...")
                if recommendations:
                    best_rec = recommendations[0]
                    print(f"  ✓ Would run: {best_rec.algorithm}")
                    print(f"  ✓ With contamination: {profile.recommended_contamination:.1%}")
                    print(f"  ✓ Expected to find: ~{int(profile.n_samples * profile.recommended_contamination)} anomalies")
                    
                    # Simulate what results would look like
                    expected_anomalies = int(profile.n_samples * profile.recommended_contamination)
                    print(f"\n  📈 Expected Results:")
                    print(f"    • Total samples: {profile.n_samples}")
                    print(f"    • Predicted anomalies: ~{expected_anomalies}")
                    print(f"    • Anomaly rate: ~{profile.recommended_contamination:.1%}")
                    print(f"    • Algorithm: {best_rec.algorithm}")
                    print(f"    • Confidence: {best_rec.confidence:.1%}")
                
            except Exception as e:
                print(f"  ❌ Error processing {dataset_name}: {e}")
                continue
        
        # Show CLI usage examples
        print(f"\n{'='*60}")
        print("🖥️  CLI USAGE EXAMPLES")
        print("="*60)
        
        print("\n# Quick autonomous detection:")
        print(f"pynomaly auto quick {list(datasets.values())[0]}")
        
        print("\n# Comprehensive autonomous detection:")
        print(f"pynomaly auto detect {list(datasets.values())[0]} --output results.csv")
        
        print("\n# Data profiling only:")
        print(f"pynomaly auto profile {list(datasets.values())[0]} --verbose")
        
        print("\n# Advanced autonomous detection:")
        print(f"pynomaly auto detect {list(datasets.values())[0]} \\")
        print("    --max-algorithms 5 \\")
        print("    --auto-tune \\")
        print("    --confidence-threshold 0.9 \\")
        print("    --output comprehensive_results.json \\")
        print("    --save-models")
        
        print("\n# Batch processing:")
        print("pynomaly auto batch-detect data/ --pattern '*.csv' --output results/")
        
        # Show what autonomous mode provides
        print(f"\n{'='*60}")
        print("🌟 AUTONOMOUS MODE BENEFITS")
        print("="*60)
        
        benefits = [
            "🔍 Automatic data format detection and loading",
            "📊 Intelligent data profiling and characterization", 
            "🧠 Smart algorithm selection based on data properties",
            "⚙️ Automatic hyperparameter tuning",
            "🎯 Adaptive contamination rate estimation",
            "📈 Performance-based algorithm ranking",
            "💾 Automatic result saving and export",
            "📝 Detailed reasoning and recommendations",
            "🔄 Batch processing capabilities",
            "🛡️ Robust error handling and fallbacks"
        ]
        
        for benefit in benefits:
            print(f"  {benefit}")
        
        print(f"\n{'='*60}")
        print("✨ DEMONSTRATION COMPLETE")
        print("="*60)
        print("Autonomous mode successfully demonstrated with multiple data types!")
        print("The system automatically adapted its approach for each dataset.")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\n🧹 Cleaned up demo data: {temp_dir}")

def main():
    """Run the autonomous mode demonstration."""
    try:
        # Check if we can import required modules
        print("Checking autonomous mode dependencies...")
        
        import pandas as pd
        import numpy as np
        from pynomaly.application.services.autonomous_service import AutonomousDetectionService
        
        print("✓ All dependencies available")
        
        # Run the demonstration
        asyncio.run(demonstrate_autonomous_detection())
        
        return 0
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please ensure pynomaly is properly installed.")
        return 1
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())