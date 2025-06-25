#!/usr/bin/env python3
"""
Simple CLI Wrapper for Pynomaly
Bypasses Typer compatibility issues by providing direct function calls
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import error handling utilities
from pynomaly.shared.error_handling import (
    handle_cli_errors, validate_file_exists, validate_data_format,
    validate_contamination_rate, validate_algorithm_name, create_user_friendly_message
)

def show_help():
    """Show help information"""
    help_text = """
Pynomaly - State-of-the-art anomaly detection CLI

Usage: python pynomaly_cli.py <command> [options]

Commands:
  help               Show this help message
  version            Show version information
  detector-list      List available detectors
  dataset-info FILE  Show dataset information
  validate FILE      Validate data quality for anomaly detection
  detect FILE [ALGORITHM] [CONTAMINATION]  Run anomaly detection on dataset
  server-start       Start the API server
  test-imports       Test core system imports
  benchmark FILE     Run performance benchmark on dataset
  perf-stats         Show performance statistics and system info
  auto-select FILE   Automatically select best algorithm and parameters

Examples:
  python pynomaly_cli.py version
  python pynomaly_cli.py detector-list
  python pynomaly_cli.py dataset-info data.csv
  python pynomaly_cli.py validate data.csv
  python pynomaly_cli.py detect data.csv
  python pynomaly_cli.py detect data.csv IsolationForest 0.05
  python pynomaly_cli.py detect data.csv LocalOutlierFactor 0.1
  python pynomaly_cli.py benchmark data.csv
  python pynomaly_cli.py perf-stats
  python pynomaly_cli.py server-start
"""
    print(help_text)

def show_version():
    """Show version information"""
    try:
        from pynomaly.infrastructure.config import Settings
        settings = Settings()
        print(f"Pynomaly v{settings.app.version}")
        print(f"Python {sys.version.split()[0]}")
        print(f"Storage: {settings.storage_path}")
    except Exception as e:
        print(f"Pynomaly v0.1.0 (config error: {e})")
        print(f"Python {sys.version.split()[0]}")

def list_detectors():
    """List available detectors"""
    try:
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
        print("Available Detectors:")
        print("- IsolationForest (Isolation Forest)")
        print("- LocalOutlierFactor (Local Outlier Factor)")
        print("- OneClassSVM (One-Class SVM)")
        print("- EllipticEnvelope (Elliptic Envelope)")
        print("- SGDOneClassSVM (SGD One-Class SVM)")
        print("\nNote: More detectors available through other adapters")
    except Exception as e:
        print(f"Error listing detectors: {e}")

def show_dataset_info(file_path):
    """Show dataset information"""
    try:
        import pandas as pd
        
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found")
            return
        
        # Try to read the dataset
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                print(f"Unsupported file format. Supported: .csv, .json")
                return
        except Exception as e:
            print(f"Error reading file: {e}")
            return
        
        print(f"Dataset Information for: {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nBasic statistics:")
        print(df.describe())
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")

@handle_cli_errors 
def validate_dataset(file_path):
    """Validate data quality for anomaly detection"""
    import pandas as pd
    from pynomaly.infrastructure.data.validation_pipeline import DataValidationPipeline, ValidationSeverity
    
    # Validate file
    file_path = validate_file_exists(file_path)
    data_format = validate_data_format(file_path)
    
    print(f"Validating data quality for: {file_path}")
    
    # Load data
    if data_format == 'csv':
        data = pd.read_csv(file_path)
    elif data_format == 'json':
        data = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported format: {data_format}")
    
    print(f"Data loaded: {data.shape}")
    
    # Create validation pipeline
    validator = DataValidationPipeline()
    
    # Run validation
    report = validator.validate(data)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"DATA QUALITY ASSESSMENT")
    print(f"{'='*60}")
    print(f"Overall Quality: {report.overall_quality.value.upper()}")
    print(f"Quality Score: {report.score:.1f}/100")
    
    # Display issues by severity
    severities = [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR, 
                 ValidationSeverity.WARNING, ValidationSeverity.INFO]
    
    for severity in severities:
        issues = report.get_issues_by_severity(severity)
        if issues:
            print(f"\n{severity.value.upper()} Issues ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue.message}")
                if issue.recommendation:
                    print(f"     → {issue.recommendation}")
    
    # Display key statistics
    print(f"\n{'='*60}")
    print(f"KEY STATISTICS")
    print(f"{'='*60}")
    stats = report.statistics
    print(f"Samples: {stats.get('n_samples', 'N/A')}")
    print(f"Features: {stats.get('n_features', 'N/A')}")
    print(f"Numeric Features: {len(stats.get('numeric_columns', []))}")
    print(f"Missing Value Ratio: {stats.get('total_missing_ratio', 0):.2%}")
    print(f"Constant Features: {len(stats.get('constant_features', []))}")
    
    # Display recommendations
    if report.recommendations:
        print(f"\n{'='*60}")
        print(f"RECOMMENDATIONS")
        print(f"{'='*60}")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")
    
    # Final assessment
    print(f"\n{'='*60}")
    if report.has_critical_issues():
        print("❌ CRITICAL ISSUES FOUND - Data is not suitable for anomaly detection")
        return False
    elif report.has_errors():
        print("⚠️ ERRORS FOUND - Data quality issues should be addressed")
        return False
    elif report.score >= 75:
        print("✅ DATA QUALITY IS GOOD - Suitable for anomaly detection")
        return True
    else:
        print("⚠️ DATA QUALITY IS FAIR - Consider preprocessing improvements")
        return True

@handle_cli_errors
def run_detection(file_path, algorithm=None, contamination=None):
    """Run anomaly detection on dataset"""
    import pandas as pd
    from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
    from pynomaly.domain.entities import Dataset
    from pynomaly.domain.value_objects import ContaminationRate
    
    # Validate file
    file_path = validate_file_exists(file_path)
    data_format = validate_data_format(file_path)
    
    # Set defaults
    algorithm = algorithm or "IsolationForest"
    contamination = float(contamination) if contamination else 0.1
    
    # Validate parameters
    available_algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM", "EllipticEnvelope", "SGDOneClassSVM"]
    algorithm = validate_algorithm_name(algorithm, available_algorithms)
    contamination = validate_contamination_rate(contamination)
    
    print(f"Running anomaly detection on: {file_path}")
    print(f"Algorithm: {algorithm}")
    print(f"Contamination rate: {contamination}")
    
    # Load data based on format
    if data_format == 'csv':
        data = pd.read_csv(file_path)
    elif data_format == 'json':
        data = pd.read_json(file_path)
    else:
        # This shouldn't happen due to validation, but just in case
        raise ValueError(f"Unsupported format: {data_format}")
    
    print(f"Data loaded: {data.shape}")
    
    # Validate data shape and content
    from pynomaly.shared.error_handling import validate_data_shape
    validate_data_shape(data, min_samples=2, min_features=1)
    
    # Create dataset
    dataset = Dataset(
        name=f"dataset_{file_path.name}",
        data=data
    )
    
    # Create adapter and run detection
    adapter = SklearnAdapter(algorithm, contamination_rate=ContaminationRate(contamination))
    adapter.fit(dataset)
    result = adapter.detect(dataset)
    
    # Show results
    anomaly_count = len(result.anomalies)
    print(f"\nDetection Results:")
    print(f"Total samples: {len(result.labels)}")
    print(f"Anomalies detected: {anomaly_count}")
    print(f"Anomaly rate: {anomaly_count/len(result.labels)*100:.1f}%")
    print(f"Threshold: {result.threshold:.3f}")
    print(f"Execution time: {result.execution_time_ms:.1f}ms")
    
    if anomaly_count > 0:
        anomaly_indices = [i for i, label in enumerate(result.labels) if label == 1]
        print(f"\nAnomaly indices: {anomaly_indices}")
    
    return True

@handle_cli_errors
def run_benchmark(file_path):
    """Run performance benchmark on dataset"""
    import pandas as pd
    import time
    from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
    from pynomaly.domain.entities import Dataset
    from pynomaly.domain.value_objects import ContaminationRate
    
    # Validate file
    file_path = validate_file_exists(file_path)
    data_format = validate_data_format(file_path)
    
    print(f"Running performance benchmark on: {file_path}")
    
    # Load data based on format
    if data_format == 'csv':
        data = pd.read_csv(file_path)
    elif data_format == 'json':
        data = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported format: {data_format}")
    
    print(f"Data loaded: {data.shape}")
    
    # Validate data
    from pynomaly.shared.error_handling import validate_data_shape
    validate_data_shape(data, min_samples=2, min_features=1)
    
    # Create dataset
    dataset = Dataset(
        name=f"benchmark_{file_path.name}",
        data=data
    )
    
    # Test different algorithms
    algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM", "EllipticEnvelope"]
    results = []
    
    print(f"\n{'Algorithm':<20} {'Fit Time (ms)':<15} {'Detect Time (ms)':<18} {'Total Time (ms)':<16} {'Anomalies':<10}")
    print("-" * 85)
    
    for algorithm in algorithms:
        try:
            # Create adapter
            adapter = SklearnAdapter(algorithm, contamination_rate=ContaminationRate(0.1))
            
            # Measure fit time
            start_time = time.time()
            adapter.fit(dataset)
            fit_time = (time.time() - start_time) * 1000
            
            # Measure detect time
            start_time = time.time()
            result = adapter.detect(dataset)
            detect_time = (time.time() - start_time) * 1000
            
            total_time = fit_time + detect_time
            anomaly_count = len(result.anomalies)
            
            results.append({
                'algorithm': algorithm,
                'fit_time': fit_time,
                'detect_time': detect_time,
                'total_time': total_time,
                'anomalies': anomaly_count
            })
            
            print(f"{algorithm:<20} {fit_time:<15.1f} {detect_time:<18.1f} {total_time:<16.1f} {anomaly_count:<10}")
            
        except Exception as e:
            print(f"{algorithm:<20} ERROR: {str(e)[:50]}")
    
    # Summary
    if results:
        print("\n" + "=" * 85)
        fastest = min(results, key=lambda x: x['total_time'])
        print(f"Fastest algorithm: {fastest['algorithm']} ({fastest['total_time']:.1f}ms total)")
        
        avg_time = sum(r['total_time'] for r in results) / len(results)
        print(f"Average execution time: {avg_time:.1f}ms")
        
        # Data processing rate
        samples_per_second = (len(data) / (avg_time / 1000))
        print(f"Average processing rate: {samples_per_second:.0f} samples/second")
    
    return True

def show_performance_stats():
    """Show system performance statistics and monitoring info"""
    try:
        import psutil
        from datetime import datetime
        
        print("🖥️  SYSTEM PERFORMANCE STATISTICS")
        print("=" * 60)
        
        # System info
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Platform: {psutil.LINUX if hasattr(psutil, 'LINUX') else 'Unknown'}")
        
        # CPU information
        cpu_count = psutil.cpu_count()
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        
        print(f"\n⚡ CPU Information:")
        print(f"  Physical cores: {cpu_count}")
        print(f"  Logical cores: {cpu_count_logical}")
        print(f"  Current usage: {cpu_percent:.1f}%")
        if cpu_freq:
            print(f"  Current frequency: {cpu_freq.current:.0f} MHz")
            print(f"  Max frequency: {cpu_freq.max:.0f} MHz")
        
        # Memory information
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        print(f"\n💾 Memory Information:")
        print(f"  Total RAM: {memory.total / 1024**3:.1f} GB")
        print(f"  Available RAM: {memory.available / 1024**3:.1f} GB")
        print(f"  Used RAM: {memory.used / 1024**3:.1f} GB ({memory.percent:.1f}%)")
        print(f"  Free RAM: {memory.free / 1024**3:.1f} GB")
        if swap.total > 0:
            print(f"  Swap total: {swap.total / 1024**3:.1f} GB")
            print(f"  Swap used: {swap.used / 1024**3:.1f} GB ({swap.percent:.1f}%)")
        
        # Disk information
        disk = psutil.disk_usage('/')
        
        print(f"\n💿 Disk Information:")
        print(f"  Total space: {disk.total / 1024**3:.1f} GB")
        print(f"  Used space: {disk.used / 1024**3:.1f} GB ({disk.used/disk.total*100:.1f}%)")
        print(f"  Free space: {disk.free / 1024**3:.1f} GB")
        
        # Process information
        process = psutil.Process()
        process_memory = process.memory_info()
        
        print(f"\n🔧 Current Process (Pynomaly CLI):")
        print(f"  PID: {process.pid}")
        print(f"  Memory usage: {process_memory.rss / 1024**2:.1f} MB")
        print(f"  CPU percent: {process.cpu_percent():.1f}%")
        print(f"  Created: {datetime.fromtimestamp(process.create_time()).strftime('%H:%M:%S')}")
        
        # Network interfaces (if available)
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                print(f"\n🌐 Network I/O:")
                print(f"  Bytes sent: {net_io.bytes_sent / 1024**2:.1f} MB")
                print(f"  Bytes received: {net_io.bytes_recv / 1024**2:.1f} MB")
                print(f"  Packets sent: {net_io.packets_sent:,}")
                print(f"  Packets received: {net_io.packets_recv:,}")
        except:
            pass  # Network stats not available
        
        # Performance recommendations
        print(f"\n📊 Performance Assessment:")
        
        recommendations = []
        if cpu_percent > 80:
            recommendations.append("⚠️  High CPU usage detected - consider closing other applications")
        if memory.percent > 85:
            recommendations.append("⚠️  High memory usage detected - consider freeing up RAM")
        if disk.used/disk.total > 0.9:
            recommendations.append("⚠️  Low disk space - consider cleaning up files")
        
        if not recommendations:
            print("✅ System performance is good for anomaly detection tasks")
        else:
            for rec in recommendations:
                print(f"  {rec}")
        
        # Anomaly detection performance estimates
        print(f"\n🧠 Anomaly Detection Performance Estimates:")
        
        # Estimate based on available memory and CPU
        available_gb = memory.available / 1024**3
        
        if available_gb >= 8:
            dataset_size = "Large (>100k samples)"
            performance = "Excellent"
        elif available_gb >= 4:
            dataset_size = "Medium (10k-100k samples)"
            performance = "Good"
        elif available_gb >= 2:
            dataset_size = "Small (1k-10k samples)"
            performance = "Fair"
        else:
            dataset_size = "Very small (<1k samples)"
            performance = "Limited"
        
        print(f"  Recommended dataset size: {dataset_size}")
        print(f"  Expected performance: {performance}")
        print(f"  Concurrent algorithms: {min(cpu_count, 4)}")
        
        print(f"\n{'='*60}")
        
    except Exception as e:
        print(f"Error getting performance statistics: {e}")

def start_server():
    """Start the API server"""
    try:
        import uvicorn
        from pynomaly.presentation.api import create_app
        
        print("Starting Pynomaly API server...")
        print("Server will be available at: http://127.0.0.1:8000")
        print("API documentation: http://127.0.0.1:8000/docs")
        print("Press CTRL+C to stop")
        
        app = create_app()
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
        
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()

@handle_cli_errors
def auto_select_algorithm(file_path):
    """Automatically select best algorithm and parameters using AutoML"""
    import pandas as pd
    import asyncio
    import time
    from pynomaly.domain.entities import Dataset
    from pynomaly.application.services.automl_service import AutoMLService, OptimizationObjective
    from pynomaly.infrastructure.config import Container
    
    # Validate file
    file_path = validate_file_exists(file_path)
    data_format = validate_data_format(file_path)
    
    print(f"🧠 AutoML Algorithm Selection for: {file_path}")
    print("=" * 60)
    
    # Load data
    if data_format == 'csv':
        data = pd.read_csv(file_path)
    elif data_format == 'json':
        data = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported format: {data_format}")
    
    print(f"📊 Data loaded: {data.shape}")
    
    # Validate data
    from pynomaly.shared.error_handling import validate_data_shape
    validate_data_shape(data, min_samples=10, min_features=1)
    
    async def run_automl():
        try:
            # Create container and dependencies (simplified for CLI)
            print("🔧 Initializing AutoML service...")
            
            # For CLI usage, we'll create a simplified AutoML service
            # In production, this would use the full container setup
            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
            from collections import defaultdict
            
            class MockRepository:
                def __init__(self):
                    self.data = {}
                async def get(self, id): 
                    return self.data.get(id)
                async def save(self, obj):
                    self.data[obj.id] = obj
            
            class MockRegistry:
                def get_adapter(self, adapter_type):
                    if adapter_type == "sklearn":
                        return SklearnAdapter("IsolationForest")
                    return None
            
            # Create dataset
            dataset = Dataset(
                name=f"automl_dataset_{file_path.name}",
                data=data
            )
            
            # Mock repositories for CLI
            dataset_repo = MockRepository()
            detector_repo = MockRepository()
            adapter_registry = MockRegistry()
            
            # Save dataset
            dataset.id = "automl-dataset-1"
            await dataset_repo.save(dataset)
            
            # Create AutoML service
            automl_service = AutoMLService(
                detector_repository=detector_repo,
                dataset_repository=dataset_repo,
                adapter_registry=adapter_registry,
                max_optimization_time=300,  # 5 minutes for CLI
                n_trials=20,  # Reduced for CLI speed
                cv_folds=3
            )
            
            # Profile dataset
            print("🔍 Profiling dataset characteristics...")
            profile = await automl_service.profile_dataset("automl-dataset-1")
            
            print(f"\n📋 Dataset Profile:")
            print(f"  Samples: {profile.n_samples:,}")
            print(f"  Features: {profile.n_features}")
            print(f"  Estimated contamination: {profile.contamination_estimate:.2%}")
            print(f"  Missing values: {profile.missing_values_ratio:.2%}")
            print(f"  Complexity score: {profile.complexity_score:.3f}")
            print(f"  Data size: {profile.dataset_size_mb:.1f} MB")
            
            if profile.numerical_features:
                print(f"  Numerical features: {len(profile.numerical_features)}")
            if profile.categorical_features:
                print(f"  Categorical features: {len(profile.categorical_features)}")
            if profile.time_series_features:
                print(f"  Time series features: {len(profile.time_series_features)}")
            
            # Get algorithm recommendations
            print(f"\n🎯 Recommending algorithms...")
            recommended_algorithms = automl_service.recommend_algorithms(profile, max_algorithms=5)
            
            print(f"\n📊 Recommended Algorithms (in order of suitability):")
            for i, algorithm in enumerate(recommended_algorithms, 1):
                config = automl_service.algorithm_configs[algorithm]
                print(f"  {i}. {algorithm}")
                print(f"     Family: {config.family.value}")
                print(f"     Complexity: {config.complexity_score:.2f}")
                print(f"     Training time factor: {config.training_time_factor:.2f}")
                print(f"     Memory factor: {config.memory_factor:.2f}")
            
            # Run AutoML optimization (simplified for CLI)
            print(f"\n⚡ Running AutoML optimization...")
            print(f"   Max algorithms to test: 3")
            print(f"   Max trials per algorithm: 20")
            print(f"   Timeout: 5 minutes")
            print(f"   Objective: AUC")
            
            start_time = time.time()
            
            try:
                automl_result = await automl_service.auto_select_and_optimize(
                    dataset_id="automl-dataset-1",
                    objective=OptimizationObjective.AUC,
                    max_algorithms=3,
                    enable_ensemble=True
                )
                
                optimization_time = time.time() - start_time
                
                # Display results
                print(f"\n🎉 AutoML Optimization Complete!")
                print(f"   Total time: {optimization_time:.1f}s")
                print(f"   Trials completed: {automl_result.trials_completed}")
                
                print(f"\n🏆 Best Algorithm: {automl_result.best_algorithm}")
                print(f"   Score: {automl_result.best_score:.4f}")
                print(f"   Parameters:")
                for param, value in automl_result.best_params.items():
                    if isinstance(value, float):
                        print(f"     {param}: {value:.4f}")
                    else:
                        print(f"     {param}: {value}")
                
                print(f"\n📈 Algorithm Rankings:")
                for i, (algorithm, score) in enumerate(automl_result.algorithm_rankings, 1):
                    print(f"   {i}. {algorithm}: {score:.4f}")
                
                if automl_result.ensemble_config:
                    print(f"\n🎭 Ensemble Configuration:")
                    print(f"   Method: {automl_result.ensemble_config['method']}")
                    print(f"   Voting: {automl_result.ensemble_config['voting_strategy']}")
                    print(f"   Algorithms in ensemble:")
                    for algo_config in automl_result.ensemble_config['algorithms']:
                        print(f"     - {algo_config['name']} (weight: {algo_config['weight']:.3f})")
                
                # Demonstrate the best algorithm
                print(f"\n🔬 Testing Best Algorithm on Your Data:")
                print("-" * 50)
                
                from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
                from pynomaly.domain.value_objects import ContaminationRate
                
                best_contamination = automl_result.best_params.get('contamination', 0.1)
                adapter = SklearnAdapter(
                    automl_result.best_algorithm, 
                    contamination_rate=ContaminationRate(best_contamination)
                )
                
                # Apply optimized parameters
                for param, value in automl_result.best_params.items():
                    if param != 'contamination' and hasattr(adapter, param):
                        setattr(adapter, param, value)
                
                # Train and detect
                adapter.fit(dataset)
                result = adapter.detect(dataset)
                
                anomaly_count = len(result.anomalies)
                print(f"   Total samples: {len(result.labels)}")
                print(f"   Anomalies detected: {anomaly_count}")
                print(f"   Anomaly rate: {anomaly_count/len(result.labels)*100:.1f}%")
                print(f"   Threshold: {result.threshold:.6f}")
                print(f"   Execution time: {result.execution_time_ms:.1f}ms")
                
                if anomaly_count > 0 and anomaly_count <= 20:
                    anomaly_indices = [i for i, label in enumerate(result.labels) if label == 1]
                    print(f"   Anomaly sample indices: {anomaly_indices}")
                elif anomaly_count > 20:
                    anomaly_indices = [i for i, label in enumerate(result.labels) if label == 1]
                    print(f"   First 20 anomaly indices: {anomaly_indices[:20]}")
                
                # Summary and recommendations
                print(f"\n💡 AutoML Recommendations:")
                summary = automl_service.get_optimization_summary(automl_result)
                for rec in summary['recommendations']:
                    print(f"   • {rec}")
                
                if automl_result.best_score >= 0.8:
                    print(f"   • Excellent performance achieved - ready for production")
                elif automl_result.best_score >= 0.7:
                    print(f"   • Good performance - consider fine-tuning for production")
                else:
                    print(f"   • Moderate performance - consider collecting more data or feature engineering")
                
                print(f"\n📝 Command to Reproduce Best Results:")
                cmd = f"python pynomaly_cli.py detect {file_path} {automl_result.best_algorithm} {best_contamination:.3f}"
                print(f"   {cmd}")
                
                return True
                
            except Exception as e:
                print(f"❌ AutoML optimization failed: {str(e)}")
                print(f"   Falling back to basic algorithm recommendation...")
                
                # Fallback: just show recommendations
                print(f"\n🎯 Recommended algorithms for your data:")
                for i, algorithm in enumerate(recommended_algorithms[:3], 1):
                    print(f"   {i}. {algorithm}")
                    print(f"      Try: python pynomaly_cli.py detect {file_path} {algorithm}")
                
                return False
                
        except Exception as e:
            print(f"❌ AutoML service initialization failed: {str(e)}")
            print(f"   Please check your data format and try again")
            return False
    
    # Run the async AutoML process
    print("🚀 Starting AutoML process...")
    success = asyncio.run(run_automl())
    
    if success:
        print(f"\n✅ AutoML completed successfully!")
    else:
        print(f"\n⚠️ AutoML completed with issues - see recommendations above")
    
    return success

def test_imports():
    """Test core system imports"""
    tests = [
        ("Domain entities", "from pynomaly.domain.entities import Anomaly, Dataset, Detector, DetectionResult"),
        ("Value objects", "from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate"),
        ("Configuration", "from pynomaly.infrastructure.config import Container, Settings"),
        ("Sklearn adapter", "from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter"),
        ("Detection service", "from pynomaly.application.services import DetectionService"),
        ("API app", "from pynomaly.presentation.api import create_app")
    ]
    
    print("Testing core system imports...")
    success_count = 0
    
    for test_name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✅ {test_name}")
            success_count += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
    
    print(f"\nImport test results: {success_count}/{len(tests)} successful")
    
    if success_count == len(tests):
        print("🎉 All core imports working!")
    elif success_count >= len(tests) * 0.8:
        print("⚠️ Most imports working, minor issues remain")
    else:
        print("❌ Significant import issues detected")

def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command in ['help', '-h', '--help']:
        show_help()
    elif command in ['version', '-v', '--version']:
        show_version()
    elif command == 'detector-list':
        list_detectors()
    elif command == 'dataset-info':
        if len(sys.argv) < 3:
            print("Error: Please provide a file path")
            print("Usage: python pynomaly_cli.py dataset-info <file>")
        else:
            show_dataset_info(sys.argv[2])
    elif command == 'validate':
        if len(sys.argv) < 3:
            print("Error: Please provide a file path")
            print("Usage: python pynomaly_cli.py validate <file>")
        else:
            validate_dataset(sys.argv[2])
    elif command == 'detect':
        if len(sys.argv) < 3:
            print("Error: Please provide a file path")
            print("Usage: python pynomaly_cli.py detect <file> [algorithm] [contamination]")
        else:
            file_path = sys.argv[2]
            algorithm = sys.argv[3] if len(sys.argv) > 3 else None
            contamination = sys.argv[4] if len(sys.argv) > 4 else None
            run_detection(file_path, algorithm, contamination)
    elif command == 'benchmark':
        if len(sys.argv) < 3:
            print("Error: Please provide a file path")
            print("Usage: python pynomaly_cli.py benchmark <file>")
        else:
            run_benchmark(sys.argv[2])
    elif command == 'server-start':
        start_server()
    elif command == 'test-imports':
        test_imports()
    elif command == 'perf-stats':
        show_performance_stats()
    elif command == 'auto-select':
        if len(sys.argv) < 3:
            print("Error: Please provide a file path")
            print("Usage: python pynomaly_cli.py auto-select <file>")
        else:
            auto_select_algorithm(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        print("Use 'python pynomaly_cli.py help' for available commands")

if __name__ == "__main__":
    main()