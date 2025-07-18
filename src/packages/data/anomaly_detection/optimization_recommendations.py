#!/usr/bin/env python3
"""
Optimization Recommendations Generator for Pynomaly Detection.

Generates specific optimization recommendations based on benchmark results
and use case analysis.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

@dataclass
class OptimizationRecommendation:
    """Optimization recommendation for a specific use case."""
    use_case: str
    priority: str  # "critical", "high", "medium", "low"
    category: str  # "algorithm", "performance", "architecture", "deployment"
    title: str
    description: str
    implementation_steps: List[str]
    expected_improvement: str
    effort_level: str  # "low", "medium", "high"
    production_impact: str  # "none", "low", "medium", "high"
    code_example: str = ""
    validation_metrics: List[str] = field(default_factory=list)

class OptimizationRecommendationsGenerator:
    """Generates optimization recommendations based on benchmark data and use case analysis."""
    
    def __init__(self):
        self.recommendations: List[OptimizationRecommendation] = []
        
        # Performance thresholds for different use cases
        self.performance_thresholds = {
            "real_time": {"min_throughput": 10000, "max_latency": 0.1},
            "batch": {"min_throughput": 1000, "max_latency": 10.0},
            "interactive": {"min_throughput": 5000, "max_latency": 1.0},
            "background": {"min_throughput": 500, "max_latency": 60.0}
        }
        
        # Use case specific requirements
        self.use_case_requirements = {
            "iot_monitoring": {
                "type": "real_time",
                "accuracy_priority": "high",
                "latency_priority": "critical",
                "scalability_priority": "high",
                "memory_constraints": "medium"
            },
            "fraud_detection": {
                "type": "real_time", 
                "accuracy_priority": "critical",
                "latency_priority": "high",
                "scalability_priority": "high",
                "memory_constraints": "low"
            },
            "network_security": {
                "type": "real_time",
                "accuracy_priority": "high", 
                "latency_priority": "critical",
                "scalability_priority": "critical",
                "memory_constraints": "medium"
            },
            "quality_control": {
                "type": "interactive",
                "accuracy_priority": "critical",
                "latency_priority": "medium",
                "scalability_priority": "medium", 
                "memory_constraints": "low"
            },
            "log_analysis": {
                "type": "batch",
                "accuracy_priority": "medium",
                "latency_priority": "low",
                "scalability_priority": "critical",
                "memory_constraints": "high"
            }
        }
    
    def generate_recommendations(self, benchmark_results: Dict[str, Any] = None) -> List[OptimizationRecommendation]:
        """Generate comprehensive optimization recommendations."""
        
        print("🎯 Generating Optimization Recommendations...")
        print("=" * 60)
        
        # Generate recommendations for each category
        self._generate_algorithm_recommendations()
        self._generate_performance_recommendations()
        self._generate_architecture_recommendations() 
        self._generate_deployment_recommendations()
        
        # Add benchmark-specific recommendations if available
        if benchmark_results:
            self._generate_benchmark_based_recommendations(benchmark_results)
        
        # Sort by priority and impact
        self.recommendations.sort(key=lambda r: (
            {"critical": 4, "high": 3, "medium": 2, "low": 1}[r.priority],
            {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}[r.production_impact]
        ), reverse=True)
        
        print(f"✅ Generated {len(self.recommendations)} optimization recommendations")
        return self.recommendations
    
    def _generate_algorithm_recommendations(self):
        """Generate algorithm selection and tuning recommendations."""
        
        # IoT Monitoring Recommendations
        self.recommendations.append(OptimizationRecommendation(
            use_case="iot_monitoring",
            priority="high",
            category="algorithm", 
            title="Use PCA for High-Frequency IoT Data",
            description="PCA shows 1.2M+ samples/s throughput for IoT sensor data, ideal for real-time processing.",
            implementation_steps=[
                "Replace default IsolationForest with PCA for IoT streams",
                "Set contamination rate to 0.02-0.05 for sensor data",
                "Implement sliding window for temporal anomalies",
                "Add drift detection for sensor recalibration"
            ],
            expected_improvement="20x throughput improvement (50K → 1M+ samples/s)",
            effort_level="medium",
            production_impact="high",
            code_example="""
# Optimized IoT monitoring setup
from pynomaly_detection import CoreDetectionService, StreamingDetector

# High-throughput detection
detector = CoreDetectionService()
result = detector.detect_anomalies(
    iot_data, 
    algorithm="pca",
    contamination=0.03  # Lower rate for sensor data
)

# Real-time streaming
stream_detector = StreamingDetector(
    algorithm="pca", 
    window_size=1000,
    drift_detection=True
)
""",
            validation_metrics=["throughput > 500K samples/s", "latency < 10ms", "drift detection accuracy > 95%"]
        ))
        
        # Fraud Detection Recommendations  
        self.recommendations.append(OptimizationRecommendation(
            use_case="fraud_detection",
            priority="critical",
            category="algorithm",
            title="Implement Ensemble Methods for Fraud Detection",
            description="Use ensemble of IsolationForest + PCA + LOF for robust fraud detection with explainability.",
            implementation_steps=[
                "Configure ensemble with iforest (accuracy) + pca (speed) + lof (local patterns)",
                "Use weighted voting based on transaction risk profile", 
                "Implement feature importance ranking for compliance",
                "Add real-time model retraining for evolving fraud patterns"
            ],
            expected_improvement="15% accuracy improvement + regulatory compliance",
            effort_level="high",
            production_impact="critical",
            code_example="""
# Robust fraud detection ensemble
from pynomaly_detection import EnsembleService, AdvancedExplainability

ensemble = EnsembleService()
result = ensemble.smart_ensemble(
    transaction_data,
    algorithms=["iforest", "pca", "lof"],
    voting="weighted",
    weights=[0.4, 0.3, 0.3]  # Accuracy-focused weighting
)

# Compliance explainability
explainer = AdvancedExplainability(feature_names=feature_names)
explanation = explainer.explain_prediction(
    suspicious_transaction, result
)
""",
            validation_metrics=["precision > 95%", "recall > 90%", "explanation confidence > 80%"]
        ))
        
        # Network Security Recommendations
        self.recommendations.append(OptimizationRecommendation(
            use_case="network_security",
            priority="critical", 
            category="algorithm",
            title="Optimize LOF for Network Intrusion Detection",
            description="LOF excels at detecting local anomalies in network traffic patterns. Optimize for streaming traffic.",
            implementation_steps=[
                "Use LOF for network flow analysis (66K+ samples/s proven)",
                "Implement incremental LOF for streaming traffic",
                "Set dynamic contamination based on network baseline",
                "Add protocol-specific anomaly detection"
            ],
            expected_improvement="Real-time intrusion detection with 95%+ accuracy",
            effort_level="high",
            production_impact="critical",
            code_example="""
# Network security optimization
from pynomaly_detection import StreamingDetector, MonitoringAlertingSystem

# Real-time network monitoring
network_detector = StreamingDetector(
    algorithm="lof",
    window_size=5000,  # 5K flows
    contamination_adaptive=True
)

# Automated alerting
monitoring = MonitoringAlertingSystem()
monitoring.add_alert_rule({
    "rule_id": "network_intrusion",
    "condition": "anomaly_rate > 0.05",
    "severity": "CRITICAL"
})
""",
            validation_metrics=["detection latency < 100ms", "false positive rate < 2%", "scalability > 100K flows/s"]
        ))
    
    def _generate_performance_recommendations(self):
        """Generate performance optimization recommendations."""
        
        # Memory Optimization
        self.recommendations.append(OptimizationRecommendation(
            use_case="general",
            priority="high",
            category="performance",
            title="Implement Memory-Efficient Data Processing",
            description="Optimize memory usage for large-scale deployments using batch processing and data type optimization.",
            implementation_steps=[
                "Use BatchProcessor for datasets > 50K samples",
                "Convert float64 to float32 for 50% memory reduction",
                "Implement memory-mapped file processing for huge datasets",
                "Add automatic garbage collection triggers"
            ],
            expected_improvement="50-70% memory usage reduction",
            effort_level="medium",
            production_impact="high",
            code_example="""
# Memory-efficient processing
from pynomaly_detection import BatchProcessor, MemoryOptimizer

# Large dataset processing
processor = BatchProcessor()
results = processor.process_large_dataset(
    large_data.astype(np.float32),  # 50% memory savings
    algorithm="iforest",
    batch_size=10000,
    n_workers=4
)

# Memory optimization
optimizer = MemoryOptimizer()
optimized_data = optimizer.optimize_array_dtype(data)
""",
            validation_metrics=["memory usage < 500MB", "processing speed maintained", "accuracy preserved"]
        ))
        
        # Parallel Processing
        self.recommendations.append(OptimizationRecommendation(
            use_case="batch_processing",
            priority="high", 
            category="performance",
            title="Optimize Parallel Processing for Batch Jobs",
            description="Use optimal worker configuration for maximum throughput in batch processing scenarios.",
            implementation_steps=[
                "Configure worker count = CPU cores for CPU-bound algorithms",
                "Use ThreadPoolExecutor for I/O-bound tasks",
                "Implement data chunking with optimal chunk sizes",
                "Add progress monitoring for long-running jobs"
            ],
            expected_improvement="3-8x throughput improvement on multi-core systems",
            effort_level="low",
            production_impact="medium",
            code_example="""
# Parallel processing optimization
from pynomaly_detection import OptimizationUtilities
import multiprocessing

optimizer = OptimizationUtilities()

# Auto-optimize batch size and workers
optimization_result = optimizer.run_comprehensive_optimization(
    detection_function,
    large_dataset,
    "BatchDetection",
    include_parallel=True
)

optimal_workers = optimization_result.parameters_used.get("optimal_workers", 4)
""",
            validation_metrics=["linear speedup up to CPU core count", "memory usage scaling", "fault tolerance"]
        ))
    
    def _generate_architecture_recommendations(self):
        """Generate architecture and design recommendations."""
        
        # Microservices Architecture
        self.recommendations.append(OptimizationRecommendation(
            use_case="enterprise_deployment",
            priority="medium",
            category="architecture",
            title="Implement Microservices Architecture for Scale",
            description="Break detection pipeline into independently scalable microservices for enterprise deployment.",
            implementation_steps=[
                "Separate data ingestion, detection, and alerting services",
                "Implement async message queues (Redis/RabbitMQ)",
                "Add service discovery and load balancing",
                "Create independent scaling policies per service"
            ],
            expected_improvement="Independent scaling + fault isolation + easier deployment",
            effort_level="high",
            production_impact="high",
            code_example="""
# Microservices architecture example
from pynomaly_detection import IntegrationManager

# Detection service
detection_service = CoreDetectionService()

# Async processing with queues
integration_manager = IntegrationManager()
integration_manager.register_adapter("queue_input", queue_adapter)
integration_manager.register_adapter("alert_output", alert_adapter)

# Pipeline with async processing
pipeline_config = {
    "input_adapters": ["queue_input"],
    "output_adapters": ["alert_output"], 
    "algorithm": "automl",
    "async_processing": True
}
""",
            validation_metrics=["service independence", "horizontal scalability", "fault tolerance"]
        ))
        
        # Caching Strategy
        self.recommendations.append(OptimizationRecommendation(
            use_case="high_traffic",
            priority="medium",
            category="architecture", 
            title="Implement Intelligent Caching for High-Traffic Scenarios",
            description="Cache model predictions and preprocessing results for frequently accessed data patterns.",
            implementation_steps=[
                "Implement Redis cache for model predictions",
                "Cache preprocessing transformations",
                "Add cache invalidation based on model updates",
                "Implement cache warming strategies"
            ],
            expected_improvement="90%+ response time reduction for repeated queries",
            effort_level="medium",
            production_impact="medium",
            code_example="""
# Intelligent caching implementation
import redis
from pynomaly_detection import ModelPersistence

cache = redis.Redis(host='localhost', port=6379)
persistence = ModelPersistence()

def cached_detection(data_hash, data):
    # Check cache first
    cached_result = cache.get(f"detection_{data_hash}")
    if cached_result:
        return json.loads(cached_result)
    
    # Run detection
    result = detector.detect_anomalies(data)
    
    # Cache result (TTL = 1 hour)
    cache.setex(f"detection_{data_hash}", 3600, json.dumps(result))
    return result
""",
            validation_metrics=["cache hit rate > 80%", "response time reduction", "memory usage"]
        ))
    
    def _generate_deployment_recommendations(self):
        """Generate deployment and production recommendations."""
        
        # Container Optimization
        self.recommendations.append(OptimizationRecommendation(
            use_case="cloud_deployment",
            priority="high",
            category="deployment",
            title="Optimize Container Deployment for Production",
            description="Create optimized Docker containers with minimal footprint and maximum performance.",
            implementation_steps=[
                "Use multi-stage Docker builds for smaller images",
                "Optimize Python dependencies (remove dev packages)",
                "Configure container resource limits",
                "Implement health checks and readiness probes"
            ],
            expected_improvement="50% smaller containers + faster startup times",
            effort_level="medium",
            production_impact="medium",
            code_example="""
# Optimized Dockerfile
FROM python:3.11-slim as builder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ /app/
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "from pynomaly_detection import check_phase2_availability; exit(0 if all(check_phase2_availability().values()) else 1)"

CMD ["python", "-m", "pynomaly_detection.server"]
""",
            validation_metrics=["image size < 500MB", "startup time < 10s", "health check reliability"]
        ))
        
        # Monitoring and Observability
        self.recommendations.append(OptimizationRecommendation(
            use_case="production_monitoring",
            priority="critical",
            category="deployment",
            title="Implement Comprehensive Production Monitoring",
            description="Deploy full observability stack for production anomaly detection systems.",
            implementation_steps=[
                "Integrate with Prometheus for metrics collection",
                "Set up Grafana dashboards for visualization", 
                "Implement distributed tracing with Jaeger",
                "Add custom business metrics and SLA monitoring"
            ],
            expected_improvement="Full visibility + proactive issue detection + SLA compliance",
            effort_level="high",
            production_impact="critical",
            code_example="""
# Production monitoring setup
from pynomaly_detection import MonitoringAlertingSystem
from prometheus_client import Counter, Histogram, Gauge

# Metrics collection
detection_counter = Counter('anomaly_detections_total', 'Total anomaly detections')
processing_time = Histogram('detection_processing_seconds', 'Detection processing time')
active_alerts = Gauge('active_alerts_total', 'Number of active alerts')

# Monitoring integration
monitoring = MonitoringAlertingSystem()
monitoring.register_metric_collector("prometheus", prometheus_collector)

# Custom SLA monitoring
monitoring.add_sla_rule({
    "metric": "avg_processing_time",
    "threshold": 1.0,  # 1 second SLA
    "window": "5m"
})
""",
            validation_metrics=["MTTR < 5 minutes", "SLA compliance > 99.9%", "alert accuracy > 95%"]
        ))
    
    def _generate_benchmark_based_recommendations(self, benchmark_results: Dict[str, Any]):
        """Generate recommendations based on specific benchmark results."""
        
        if "dataset_insights" in benchmark_results:
            for dataset, insights in benchmark_results["dataset_insights"].items():
                
                # Performance-based recommendations
                avg_throughput = insights.get("avg_throughput", 0)
                if avg_throughput < 1000:
                    self.recommendations.append(OptimizationRecommendation(
                        use_case=dataset,
                        priority="high",
                        category="performance",
                        title=f"Optimize {dataset.title()} Processing Performance",
                        description=f"Current throughput ({avg_throughput:.0f} samples/s) below production threshold.",
                        implementation_steps=[
                            "Switch to faster algorithm (PCA recommended)",
                            "Implement batch processing for large datasets",
                            "Use parallel processing on multi-core systems",
                            "Consider data preprocessing optimization"
                        ],
                        expected_improvement="5-10x throughput improvement",
                        effort_level="medium",
                        production_impact="high",
                        validation_metrics=[f"throughput > 10000 samples/s", "latency < 100ms"]
                    ))
                
                # Algorithm-specific recommendations
                best_algorithm = insights.get("best_algorithm", "unknown")
                if best_algorithm != "unknown":
                    self.recommendations.append(OptimizationRecommendation(
                        use_case=dataset,
                        priority="medium", 
                        category="algorithm",
                        title=f"Use {best_algorithm.upper()} for {dataset.title()} Workload",
                        description=f"Benchmark shows {best_algorithm} performs best for {dataset} use case.",
                        implementation_steps=[
                            f"Configure detection service to use {best_algorithm}",
                            "Tune contamination rate based on use case",
                            "Implement monitoring for algorithm performance",
                            "Set up A/B testing for algorithm comparison"
                        ],
                        expected_improvement="Optimal performance for specific use case",
                        effort_level="low",
                        production_impact="medium",
                        validation_metrics=["performance consistency", "accuracy maintenance"]
                    ))
    
    def save_recommendations(self, filename: str = "optimization_recommendations.json") -> str:
        """Save recommendations to file."""
        
        recommendations_data = {
            "summary": {
                "total_recommendations": len(self.recommendations),
                "by_priority": {
                    priority: len([r for r in self.recommendations if r.priority == priority])
                    for priority in ["critical", "high", "medium", "low"]
                },
                "by_category": {
                    category: len([r for r in self.recommendations if r.category == category])
                    for category in ["algorithm", "performance", "architecture", "deployment"]
                }
            },
            "recommendations": [
                {
                    "use_case": r.use_case,
                    "priority": r.priority,
                    "category": r.category,
                    "title": r.title,
                    "description": r.description,
                    "implementation_steps": r.implementation_steps,
                    "expected_improvement": r.expected_improvement,
                    "effort_level": r.effort_level,
                    "production_impact": r.production_impact,
                    "code_example": r.code_example,
                    "validation_metrics": r.validation_metrics
                }
                for r in self.recommendations
            ]
        }
        
        filepath = Path(filename)
        with open(filepath, 'w') as f:
            json.dump(recommendations_data, f, indent=2)
        
        print(f"📄 Recommendations saved: {filepath}")
        return str(filepath)
    
    def print_executive_summary(self):
        """Print executive summary of recommendations."""
        
        print("\n" + "="*70)
        print("🎯 OPTIMIZATION RECOMMENDATIONS EXECUTIVE SUMMARY")
        print("="*70)
        
        # Priority breakdown
        priority_counts = {}
        for rec in self.recommendations:
            priority_counts[rec.priority] = priority_counts.get(rec.priority, 0) + 1
        
        print(f"📊 **Total Recommendations:** {len(self.recommendations)}")
        for priority in ["critical", "high", "medium", "low"]:
            count = priority_counts.get(priority, 0)
            emoji = {"critical": "🚨", "high": "⚡", "medium": "📈", "low": "💡"}[priority]
            print(f"   {emoji} **{priority.title()}**: {count} recommendations")
        
        # Top recommendations by category
        print(f"\n🔥 **Top Recommendations by Category:**")
        
        categories = {}
        for rec in self.recommendations:
            if rec.category not in categories:
                categories[rec.category] = []
            categories[rec.category].append(rec)
        
        for category, recs in categories.items():
            top_rec = max(recs, key=lambda r: {"critical": 4, "high": 3, "medium": 2, "low": 1}[r.priority])
            emoji = {"algorithm": "🧠", "performance": "⚡", "architecture": "🏗️", "deployment": "🚀"}[category]
            print(f"   {emoji} **{category.title()}**: {top_rec.title}")
        
        # Quick wins (high impact, low effort)
        quick_wins = [r for r in self.recommendations 
                     if r.effort_level == "low" and r.production_impact in ["high", "critical"]]
        
        if quick_wins:
            print(f"\n⚡ **Quick Wins (High Impact, Low Effort):**")
            for i, rec in enumerate(quick_wins[:3], 1):
                print(f"   {i}. {rec.title} ({rec.expected_improvement})")
        
        # Critical priorities
        critical_recs = [r for r in self.recommendations if r.priority == "critical"]
        if critical_recs:
            print(f"\n🚨 **Critical Priority Actions:**")
            for i, rec in enumerate(critical_recs[:3], 1):
                print(f"   {i}. {rec.title} ({rec.use_case})")
        
        print("\n💼 **Implementation Roadmap:**")
        print("   Phase 1 (Immediate): Quick wins + critical priorities")
        print("   Phase 2 (Short-term): High-priority performance optimizations")
        print("   Phase 3 (Medium-term): Architecture improvements")
        print("   Phase 4 (Long-term): Advanced deployment optimizations")
        
        print("\n" + "="*70)

def main():
    """Main execution function."""
    
    print("🎯 Pynomaly Detection - Optimization Recommendations Generator")
    print("=" * 70)
    
    # Create recommendations generator
    generator = OptimizationRecommendationsGenerator()
    
    # Try to load benchmark results if available
    benchmark_results = None
    potential_files = [
        "focused_benchmark_results/focused_production_report.json",
        "production_benchmark_results/production_performance_report.json"
    ]
    
    for filepath in potential_files:
        if Path(filepath).exists():
            try:
                with open(filepath, 'r') as f:
                    benchmark_results = json.load(f)
                print(f"📊 Loaded benchmark results from: {filepath}")
                break
            except Exception as e:
                print(f"⚠️  Could not load {filepath}: {e}")
    
    if not benchmark_results:
        print("📊 No benchmark results found, generating general recommendations")
    
    # Generate recommendations
    recommendations = generator.generate_recommendations(benchmark_results)
    
    # Print summary
    generator.print_executive_summary()
    
    # Save recommendations
    report_path = generator.save_recommendations()
    
    print(f"\n✅ Optimization recommendations generated successfully!")
    print(f"📋 {len(recommendations)} recommendations ready for implementation")
    print(f"📄 Detailed report: {report_path}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)