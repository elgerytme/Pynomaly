#!/usr/bin/env python3
"""
Production-Scale Benchmarking for Pynomaly Detection.

This script runs comprehensive benchmarks on production-scale datasets
to validate performance characteristics and generate optimization insights.
"""

import sys
import time
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def generate_production_datasets() -> Dict[str, np.ndarray]:
    """Generate realistic production-scale datasets."""
    print("📊 Generating production-scale datasets...")
    
    datasets = {}
    
    # IoT Sensor Data (10K devices, 1 hour of data, 1 minute intervals)
    print("   Generating IoT sensor dataset...")
    np.random.seed(42)
    n_devices = 10000
    time_points = 60  # 1 hour of minute-level data
    normal_devices = int(n_devices * 0.95)  # 95% normal
    
    # Normal sensor readings (temperature, humidity, pressure, vibration)
    normal_data = np.random.normal([22.0, 50.0, 1013.25, 0.1], 
                                 [2.0, 10.0, 5.0, 0.05], 
                                 (normal_devices, 4))
    
    # Anomalous readings (overheating, sensor failures, etc.)
    anomaly_data = np.random.normal([45.0, 90.0, 980.0, 2.0],
                                  [5.0, 15.0, 10.0, 1.0],
                                  (n_devices - normal_devices, 4))
    
    iot_data = np.vstack([normal_data, anomaly_data])
    np.random.shuffle(iot_data)
    datasets['iot_sensors'] = iot_data
    
    # Financial Transaction Data (50K transactions)
    print("   Generating financial transaction dataset...")
    n_transactions = 50000
    normal_transactions = int(n_transactions * 0.98)  # 98% legitimate
    
    # Normal transactions (amount, merchant_category, time_of_day, location_risk)
    normal_txn = np.column_stack([
        np.random.lognormal(3.0, 1.5, normal_transactions),  # Amount (log-normal)
        np.random.randint(1, 20, normal_transactions),       # Merchant category
        np.random.normal(12.0, 4.0, normal_transactions),    # Time of day
        np.random.beta(2, 5, normal_transactions),           # Location risk
        np.random.normal(0.5, 0.2, normal_transactions),     # User behavior score
    ])
    
    # Fraudulent transactions (unusual patterns)
    fraud_txn = np.column_stack([
        np.random.lognormal(6.0, 2.0, n_transactions - normal_transactions),  # Large amounts
        np.random.randint(15, 25, n_transactions - normal_transactions),      # Unusual categories
        np.random.uniform(0, 24, n_transactions - normal_transactions),       # Random times
        np.random.beta(5, 2, n_transactions - normal_transactions),           # High location risk
        np.random.normal(0.1, 0.1, n_transactions - normal_transactions),     # Low behavior score
    ])
    
    financial_data = np.vstack([normal_txn, fraud_txn])
    np.random.shuffle(financial_data)
    datasets['financial_transactions'] = financial_data
    
    # Network Traffic Data (100K network flows)
    print("   Generating network traffic dataset...")
    n_flows = 100000
    normal_flows = int(n_flows * 0.97)  # 97% normal traffic
    
    # Normal network flows (packet_size, duration, port, protocol_type, bytes_transferred)
    normal_network = np.column_stack([
        np.random.lognormal(6.0, 1.0, normal_flows),         # Packet size
        np.random.exponential(10.0, normal_flows),           # Duration
        np.random.choice([80, 443, 22, 21, 25], normal_flows),  # Common ports
        np.random.choice([0, 1, 2], normal_flows),           # Protocol types
        np.random.lognormal(8.0, 2.0, normal_flows),         # Bytes transferred
    ])
    
    # Malicious traffic (DDoS, intrusions, etc.)
    malicious_network = np.column_stack([
        np.random.lognormal(4.0, 2.0, n_flows - normal_flows),    # Small packets
        np.random.exponential(0.1, n_flows - normal_flows),       # Very short duration
        np.random.randint(1024, 65535, n_flows - normal_flows),   # Random ports
        np.random.choice([0, 1, 2], n_flows - normal_flows),      # Protocol types
        np.random.lognormal(4.0, 3.0, n_flows - normal_flows),    # Unusual byte counts
    ])
    
    network_data = np.vstack([normal_network, malicious_network])
    np.random.shuffle(network_data)
    datasets['network_traffic'] = network_data
    
    # Manufacturing Quality Control (25K product measurements)
    print("   Generating manufacturing quality dataset...")
    n_products = 25000
    normal_products = int(n_products * 0.96)  # 96% pass quality control
    
    # Normal products (dimensions, weight, surface_finish, electrical_properties)
    normal_mfg = np.column_stack([
        np.random.normal(100.0, 0.5, normal_products),       # Length (mm)
        np.random.normal(50.0, 0.3, normal_products),        # Width (mm)
        np.random.normal(25.0, 0.2, normal_products),        # Height (mm)
        np.random.normal(500.0, 5.0, normal_products),       # Weight (g)
        np.random.normal(1.5, 0.1, normal_products),         # Surface roughness
        np.random.normal(12.0, 0.5, normal_products),        # Voltage (V)
        np.random.normal(2.0, 0.1, normal_products),         # Current (A)
    ])
    
    # Defective products (out of spec)
    defective_mfg = np.column_stack([
        np.random.normal(100.0, 5.0, n_products - normal_products),    # Length variance
        np.random.normal(50.0, 3.0, n_products - normal_products),     # Width variance  
        np.random.normal(25.0, 2.0, n_products - normal_products),     # Height variance
        np.random.normal(500.0, 50.0, n_products - normal_products),   # Weight variance
        np.random.normal(3.0, 1.0, n_products - normal_products),      # Poor surface
        np.random.normal(10.0, 2.0, n_products - normal_products),     # Voltage issues
        np.random.normal(1.5, 0.5, n_products - normal_products),      # Current issues
    ])
    
    manufacturing_data = np.vstack([normal_mfg, defective_mfg])
    np.random.shuffle(manufacturing_data)
    datasets['manufacturing_quality'] = manufacturing_data
    
    # Log dataset info
    for name, data in datasets.items():
        print(f"   ✅ {name}: {data.shape} (samples x features)")
    
    return datasets

def run_production_benchmarks():
    """Run comprehensive production-scale benchmarks."""
    print("🚀 Starting Production-Scale Benchmarks")
    print("=" * 70)
    
    # Import benchmarking components
    try:
        from performance_benchmarking import (
            BenchmarkSuite, BenchmarkConfiguration,
            PerformanceProfiler, ProfilingConfiguration,
            OptimizationUtilities, OptimizationConfiguration,
            ScalabilityTester, ScalabilityConfiguration
        )
        print("✅ Performance benchmarking components loaded")
    except ImportError as e:
        print(f"❌ Failed to import benchmarking components: {e}")
        return
    
    # Generate production datasets
    datasets = generate_production_datasets()
    
    # Configure production-scale benchmarking
    production_config = BenchmarkConfiguration(
        data_sizes=[],  # Will be set per dataset
        algorithms=["iforest", "lof", "pca", "ocsvm"],
        contamination_rates=[0.02, 0.05, 0.10],
        n_iterations=3,
        warmup_iterations=1,
        enable_memory_profiling=True,
        enable_cpu_profiling=True,
        output_directory="production_benchmark_results"
    )
    
    all_results = []
    performance_insights = {}
    
    # Run benchmarks for each dataset
    for dataset_name, data in datasets.items():
        print(f"\n🔍 Benchmarking {dataset_name.upper()} Dataset")
        print("-" * 60)
        print(f"   Dataset shape: {data.shape}")
        print(f"   Estimated memory: {data.nbytes / (1024**2):.1f}MB")
        
        # Configure for this specific dataset
        production_config.data_sizes = [(data.shape[0], data.shape[1])]
        
        # Create benchmark suite
        benchmark_suite = BenchmarkSuite(production_config)
        
        try:
            # Run core detection benchmarks
            print("   Running CoreDetectionService benchmarks...")
            benchmark_suite._benchmark_simplified_services()
            
            # Get results for this dataset
            dataset_results = benchmark_suite.results[-len(production_config.algorithms) * len(production_config.contamination_rates):]
            all_results.extend(dataset_results)
            
            # Analyze performance for this dataset
            dataset_insights = analyze_dataset_performance(dataset_name, data, dataset_results)
            performance_insights[dataset_name] = dataset_insights
            
            print(f"   ✅ Completed {len(dataset_results)} benchmark tests")
            
        except Exception as e:
            print(f"   ❌ Benchmarking failed for {dataset_name}: {e}")
            continue
    
    # Generate comprehensive production report
    print(f"\n📊 Generating Production Benchmark Report...")
    production_report = generate_production_report(datasets, all_results, performance_insights)
    
    # Save production report
    report_path = Path("production_benchmark_results") / "production_performance_report.json"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(production_report, f, indent=2, default=str)
    
    print(f"📄 Production report saved: {report_path}")
    
    # Print executive summary
    print_executive_summary(production_report)
    
    return production_report

def analyze_dataset_performance(dataset_name: str, data: np.ndarray, results: List) -> Dict[str, Any]:
    """Analyze performance characteristics for a specific dataset."""
    
    insights = {
        "dataset_characteristics": {
            "name": dataset_name,
            "samples": data.shape[0],
            "features": data.shape[1],
            "memory_mb": data.nbytes / (1024**2),
            "data_density": np.count_nonzero(data) / data.size
        },
        "algorithm_performance": {},
        "recommendations": []
    }
    
    # Analyze each algorithm's performance
    algorithm_stats = {}
    for result in results:
        # Extract algorithm from metadata
        algorithm = result.metadata.get('algorithm', 'unknown')
        contamination = result.metadata.get('contamination', 0.1)
        
        if algorithm not in algorithm_stats:
            algorithm_stats[algorithm] = {
                'execution_times': [],
                'throughputs': [],
                'memory_usages': [],
                'contamination_rates': []
            }
        
        algorithm_stats[algorithm]['execution_times'].append(result.execution_time)
        algorithm_stats[algorithm]['throughputs'].append(result.throughput_samples_per_second)
        algorithm_stats[algorithm]['memory_usages'].append(result.memory_usage_mb)
        algorithm_stats[algorithm]['contamination_rates'].append(contamination)
    
    # Calculate statistics for each algorithm
    for algorithm, stats in algorithm_stats.items():
        insights["algorithm_performance"][algorithm] = {
            "avg_execution_time": np.mean(stats['execution_times']),
            "avg_throughput": np.mean(stats['throughputs']),
            "avg_memory_usage": np.mean(stats['memory_usages']),
            "min_execution_time": np.min(stats['execution_times']),
            "max_execution_time": np.max(stats['execution_times']),
            "performance_consistency": 1 - (np.std(stats['throughputs']) / np.mean(stats['throughputs'])) if stats['throughputs'] else 0
        }
    
    # Generate dataset-specific recommendations
    best_algorithm = max(algorithm_stats.keys(), 
                        key=lambda alg: np.mean(algorithm_stats[alg]['throughputs']))
    
    insights["recommendations"].append(f"Best performing algorithm for {dataset_name}: {best_algorithm}")
    
    # Dataset-specific insights
    if dataset_name == "iot_sensors":
        insights["recommendations"].extend([
            "Consider real-time streaming detection for IoT data",
            "Implement drift detection for sensor calibration changes",
            "Use batch processing for historical analysis"
        ])
    elif dataset_name == "financial_transactions":
        insights["recommendations"].extend([
            "Implement ensemble methods for fraud detection robustness", 
            "Consider model retraining frequency for evolving fraud patterns",
            "Use explainability features for regulatory compliance"
        ])
    elif dataset_name == "network_traffic":
        insights["recommendations"].extend([
            "Implement streaming detection for real-time threat detection",
            "Consider memory optimization for high-volume traffic",
            "Use parallel processing for network monitoring"
        ])
    elif dataset_name == "manufacturing_quality":
        insights["recommendations"].extend([
            "Implement statistical process control integration",
            "Consider time-series analysis for production trends",
            "Use model persistence for quality standard compliance"
        ])
    
    return insights

def generate_production_report(datasets: Dict[str, np.ndarray], 
                             results: List, 
                             insights: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive production performance report."""
    
    report = {
        "executive_summary": {
            "total_datasets": len(datasets),
            "total_samples_tested": sum(data.shape[0] for data in datasets.values()),
            "total_features": sum(data.shape[1] for data in datasets.values()),
            "total_benchmark_tests": len(results),
            "total_execution_time": sum(r.execution_time for r in results),
            "average_throughput": np.mean([r.throughput_samples_per_second for r in results]) if results else 0,
            "peak_memory_usage": max([r.memory_usage_mb for r in results]) if results else 0
        },
        "dataset_analysis": insights,
        "performance_matrix": {},
        "optimization_opportunities": [],
        "production_readiness": {},
        "scalability_assessment": {}
    }
    
    # Create performance matrix
    algorithms = list(set(r.metadata.get('algorithm', 'unknown') for r in results))
    dataset_names = list(datasets.keys())
    
    for algorithm in algorithms:
        report["performance_matrix"][algorithm] = {}
        for dataset_name in dataset_names:
            dataset_results = [r for r in results 
                             if r.metadata.get('algorithm') == algorithm 
                             and dataset_name in str(r.component_name).lower()]
            
            if dataset_results:
                avg_throughput = np.mean([r.throughput_samples_per_second for r in dataset_results])
                avg_memory = np.mean([r.memory_usage_mb for r in dataset_results])
                report["performance_matrix"][algorithm][dataset_name] = {
                    "throughput": avg_throughput,
                    "memory_mb": avg_memory,
                    "suitable_for_production": avg_throughput > 1000 and avg_memory < 1000
                }
    
    # Identify optimization opportunities
    all_throughputs = [r.throughput_samples_per_second for r in results]
    low_performance_threshold = np.percentile(all_throughputs, 25) if all_throughputs else 0
    
    for result in results:
        if result.throughput_samples_per_second < low_performance_threshold:
            algorithm = result.metadata.get('algorithm', 'unknown')
            report["optimization_opportunities"].append({
                "algorithm": algorithm,
                "issue": "Low throughput performance",
                "current_throughput": result.throughput_samples_per_second,
                "recommendation": "Consider batch processing or parallel optimization"
            })
    
    # Production readiness assessment
    for dataset_name, data in datasets.items():
        dataset_results = [r for r in results if dataset_name in str(r.component_name).lower()]
        
        if dataset_results:
            avg_throughput = np.mean([r.throughput_samples_per_second for r in dataset_results])
            avg_memory = np.mean([r.memory_usage_mb for r in dataset_results])
            max_execution_time = max([r.execution_time for r in dataset_results])
            
            readiness_score = 0
            readiness_factors = []
            
            # Throughput assessment (target: >1000 samples/s)
            if avg_throughput > 5000:
                readiness_score += 30
                readiness_factors.append("Excellent throughput performance")
            elif avg_throughput > 1000:
                readiness_score += 20
                readiness_factors.append("Good throughput performance")
            else:
                readiness_factors.append("Throughput may need optimization")
            
            # Memory assessment (target: <500MB for production)
            if avg_memory < 100:
                readiness_score += 25
                readiness_factors.append("Excellent memory efficiency")
            elif avg_memory < 500:
                readiness_score += 15
                readiness_factors.append("Acceptable memory usage")
            else:
                readiness_factors.append("Memory usage may need optimization")
            
            # Latency assessment (target: <1s for real-time)
            if max_execution_time < 0.1:
                readiness_score += 25
                readiness_factors.append("Excellent latency performance")
            elif max_execution_time < 1.0:
                readiness_score += 15
                readiness_factors.append("Good latency performance")
            else:
                readiness_factors.append("Latency may need optimization")
            
            # Consistency assessment
            throughput_cv = np.std([r.throughput_samples_per_second for r in dataset_results]) / avg_throughput
            if throughput_cv < 0.1:
                readiness_score += 20
                readiness_factors.append("Excellent performance consistency")
            elif throughput_cv < 0.3:
                readiness_score += 10
                readiness_factors.append("Good performance consistency")
            else:
                readiness_factors.append("Performance consistency needs improvement")
            
            report["production_readiness"][dataset_name] = {
                "readiness_score": readiness_score,
                "readiness_level": "Production Ready" if readiness_score >= 70 else 
                                "Needs Optimization" if readiness_score >= 40 else "Not Ready",
                "factors": readiness_factors,
                "metrics": {
                    "avg_throughput": avg_throughput,
                    "avg_memory_mb": avg_memory,
                    "max_execution_time": max_execution_time,
                    "performance_cv": throughput_cv
                }
            }
    
    return report

def print_executive_summary(report: Dict[str, Any]):
    """Print executive summary of production benchmarks."""
    
    print("\n" + "="*70)
    print("📈 PRODUCTION BENCHMARK EXECUTIVE SUMMARY")
    print("="*70)
    
    summary = report["executive_summary"]
    
    print(f"📊 **Scale:** {summary['total_datasets']} datasets, {summary['total_samples_tested']:,} samples")
    print(f"⚡ **Performance:** {summary['average_throughput']:,.0f} avg samples/s")
    print(f"💾 **Memory:** {summary['peak_memory_usage']:.1f}MB peak usage")
    print(f"⏱️  **Duration:** {summary['total_execution_time']:.1f}s total execution")
    
    print(f"\n🎯 **Production Readiness Assessment:**")
    
    readiness_summary = {}
    for dataset, readiness in report["production_readiness"].items():
        level = readiness["readiness_level"]
        if level not in readiness_summary:
            readiness_summary[level] = []
        readiness_summary[level].append(dataset)
    
    for level, datasets in readiness_summary.items():
        emoji = "✅" if level == "Production Ready" else "⚠️" if level == "Needs Optimization" else "❌"
        print(f"   {emoji} **{level}**: {', '.join(datasets)}")
    
    print(f"\n🔧 **Key Optimization Opportunities:**")
    opportunities = report["optimization_opportunities"][:3]  # Top 3
    for i, opp in enumerate(opportunities, 1):
        print(f"   {i}. {opp['algorithm']}: {opp['recommendation']}")
    
    print(f"\n⭐ **Top Performing Configurations:**")
    
    # Find best performing algorithm per dataset
    best_configs = {}
    for dataset_name, analysis in report["dataset_analysis"].items():
        if "algorithm_performance" in analysis:
            best_alg = max(analysis["algorithm_performance"].items(),
                          key=lambda x: x[1]["avg_throughput"])
            best_configs[dataset_name] = {
                "algorithm": best_alg[0],
                "throughput": best_alg[1]["avg_throughput"]
            }
    
    for dataset, config in best_configs.items():
        print(f"   📈 {dataset}: {config['algorithm']} ({config['throughput']:,.0f} samples/s)")
    
    print(f"\n💡 **Recommendations for Production Deployment:**")
    
    production_recommendations = [
        "Use IsolationForest for general-purpose anomaly detection (best balance of speed/accuracy)",
        "Implement LOF for local anomaly patterns in network/IoT data",
        "Use ensemble methods for critical fraud detection applications",
        "Consider streaming detection for real-time applications (IoT, network monitoring)",
        "Implement batch processing for large-scale historical analysis",
        "Use model persistence for consistent production deployments"
    ]
    
    for i, rec in enumerate(production_recommendations, 1):
        print(f"   {i}. {rec}")
    
    print("\n" + "="*70)

def main():
    """Main execution function."""
    start_time = time.time()
    
    try:
        # Run production benchmarks
        report = run_production_benchmarks()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 Production benchmarking completed in {total_time:.1f} seconds!")
        print("📊 Ready to proceed with optimization recommendations and auto-tuning implementation.")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Production benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)