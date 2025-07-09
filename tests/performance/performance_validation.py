#!/usr/bin/env python3
"""
Performance Validation Suite for Pynomaly
This module validates performance optimizations and measures improvements
"""

import json
import time
import statistics
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import concurrent.futures
import psutil
import requests
from dataclasses import dataclass
import yaml
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    response_time_p50: float
    response_time_p95: float
    response_time_p99: float
    throughput_rps: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    network_latency: float
    concurrent_connections: int
    timestamp: str


class PerformanceValidator:
    """Performance validation and testing framework"""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.baseline_metrics = None
        self.current_metrics = None
        self.test_results = []
        self.performance_history = []
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file"""
        default_config = {
            'target_url': 'http://localhost:8000',
            'test_duration': 300,  # 5 minutes
            'concurrent_users': 100,
            'ramp_up_time': 60,
            'thresholds': {
                'response_time_p95': 500,
                'response_time_p99': 1000,
                'throughput_min': 100,
                'error_rate_max': 0.01,
                'cpu_usage_max': 80,
                'memory_usage_max': 80
            },
            'test_scenarios': [
                {
                    'name': 'health_check',
                    'endpoint': '/health',
                    'method': 'GET',
                    'weight': 20
                },
                {
                    'name': 'anomaly_detection',
                    'endpoint': '/api/v1/detect',
                    'method': 'POST',
                    'payload': {
                        'data': [1.0, 2.0, 3.0, 4.0, 5.0],
                        'threshold': 0.5
                    },
                    'weight': 60
                },
                {
                    'name': 'model_status',
                    'endpoint': '/api/v1/models/status',
                    'method': 'GET',
                    'weight': 20
                }
            ]
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    default_config.update(user_config)
        
        return default_config
    
    def collect_system_metrics(self) -> Dict:
        """Collect system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            memory_total = memory.total
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network metrics
            network = psutil.net_io_counters()
            
            return {
                'cpu_usage': cpu_percent,
                'cpu_count': cpu_count,
                'memory_usage': memory_percent,
                'memory_available': memory_available,
                'memory_total': memory_total,
                'disk_usage': disk_percent,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'network_packets_sent': network.packets_sent,
                'network_packets_recv': network.packets_recv,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def collect_application_metrics(self) -> Dict:
        """Collect application-specific metrics"""
        try:
            # Health check
            health_start = time.time()
            health_response = requests.get(
                f"{self.config['target_url']}/health",
                timeout=10
            )
            health_time = (time.time() - health_start) * 1000
            
            # API response time test
            api_response_times = []
            for _ in range(5):
                start_time = time.time()
                response = requests.get(
                    f"{self.config['target_url']}/api/v1/health",
                    timeout=10
                )
                response_time = (time.time() - start_time) * 1000
                api_response_times.append(response_time)
            
            # Metrics endpoint
            metrics_data = {}
            try:
                metrics_response = requests.get(
                    f"{self.config['target_url']}/metrics",
                    timeout=10
                )
                if metrics_response.status_code == 200:
                    metrics_data = self._parse_prometheus_metrics(metrics_response.text)
            except:
                pass
            
            return {
                'health_status': health_response.status_code == 200,
                'health_response_time': health_time,
                'api_response_times': api_response_times,
                'api_response_time_avg': statistics.mean(api_response_times),
                'api_response_time_p95': statistics.quantiles(api_response_times, n=20)[18],
                'metrics': metrics_data,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return {}
    
    def _parse_prometheus_metrics(self, metrics_text: str) -> Dict:
        """Parse Prometheus metrics format"""
        metrics = {}
        for line in metrics_text.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            
            try:
                parts = line.split(' ')
                if len(parts) >= 2:
                    metric_name = parts[0]
                    metric_value = float(parts[1])
                    metrics[metric_name] = metric_value
            except (ValueError, IndexError):
                continue
        
        return metrics
    
    def run_load_test(self, duration: int = None, concurrent_users: int = None) -> Dict:
        """Run load test using concurrent requests"""
        duration = duration or self.config['test_duration']
        concurrent_users = concurrent_users or self.config['concurrent_users']
        
        logger.info(f"Starting load test: {concurrent_users} users for {duration} seconds")
        
        # Initialize metrics
        response_times = []
        error_count = 0
        success_count = 0
        start_time = time.time()
        end_time = start_time + duration
        
        # Create thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            
            # Submit initial requests
            for _ in range(concurrent_users):
                future = executor.submit(self._load_test_worker, end_time)
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    worker_results = future.result()
                    response_times.extend(worker_results['response_times'])
                    error_count += worker_results['error_count']
                    success_count += worker_results['success_count']
                except Exception as e:
                    logger.error(f"Load test worker failed: {e}")
                    error_count += 1
        
        # Calculate metrics
        total_requests = success_count + error_count
        actual_duration = time.time() - start_time
        
        if response_times:
            p50 = statistics.median(response_times)
            p95 = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times)
            p99 = statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
        else:
            p50 = p95 = p99 = 0
        
        return {
            'duration': actual_duration,
            'total_requests': total_requests,
            'success_count': success_count,
            'error_count': error_count,
            'error_rate': error_count / total_requests if total_requests > 0 else 0,
            'throughput_rps': total_requests / actual_duration if actual_duration > 0 else 0,
            'response_time_p50': p50,
            'response_time_p95': p95,
            'response_time_p99': p99,
            'response_time_avg': statistics.mean(response_times) if response_times else 0,
            'response_time_min': min(response_times) if response_times else 0,
            'response_time_max': max(response_times) if response_times else 0,
            'concurrent_users': concurrent_users,
            'timestamp': datetime.now().isoformat()
        }
    
    def _load_test_worker(self, end_time: float) -> Dict:
        """Worker function for load testing"""
        response_times = []
        error_count = 0
        success_count = 0
        
        session = requests.Session()
        
        while time.time() < end_time:
            try:
                # Select random scenario
                scenario = self._select_test_scenario()
                
                # Make request
                start_time = time.time()
                response = self._make_scenario_request(session, scenario)
                response_time = (time.time() - start_time) * 1000
                
                if response and response.status_code == 200:
                    success_count += 1
                    response_times.append(response_time)
                else:
                    error_count += 1
                
                # Small delay to prevent overwhelming
                time.sleep(0.01)
                
            except Exception as e:
                error_count += 1
                logger.debug(f"Request failed: {e}")
        
        return {
            'response_times': response_times,
            'error_count': error_count,
            'success_count': success_count
        }
    
    def _select_test_scenario(self) -> Dict:
        """Select test scenario based on weights"""
        import random
        
        scenarios = self.config['test_scenarios']
        weights = [scenario['weight'] for scenario in scenarios]
        
        return random.choices(scenarios, weights=weights)[0]
    
    def _make_scenario_request(self, session: requests.Session, scenario: Dict) -> requests.Response:
        """Make request based on scenario configuration"""
        url = f"{self.config['target_url']}{scenario['endpoint']}"
        method = scenario['method'].upper()
        
        kwargs = {'timeout': 10}
        
        if 'payload' in scenario:
            kwargs['json'] = scenario['payload']
        
        return session.request(method, url, **kwargs)
    
    def run_stress_test(self, max_users: int = 1000, increment: int = 50) -> Dict:
        """Run stress test with increasing load"""
        logger.info(f"Starting stress test: 0 to {max_users} users")
        
        stress_results = []
        current_users = increment
        
        while current_users <= max_users:
            logger.info(f"Testing with {current_users} concurrent users")
            
            # Run load test
            load_result = self.run_load_test(duration=60, concurrent_users=current_users)
            
            # Collect system metrics
            system_metrics = self.collect_system_metrics()
            
            # Combine results
            result = {
                'concurrent_users': current_users,
                'load_test': load_result,
                'system_metrics': system_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            stress_results.append(result)
            
            # Check if system is failing
            if (load_result['error_rate'] > 0.1 or 
                system_metrics.get('cpu_usage', 0) > 95 or
                system_metrics.get('memory_usage', 0) > 95):
                logger.warning(f"System stress threshold reached at {current_users} users")
                break
            
            current_users += increment
            time.sleep(5)  # Brief pause between tests
        
        return {
            'max_users_tested': current_users - increment,
            'results': stress_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_endurance_test(self, duration: int = 3600) -> Dict:
        """Run endurance test for extended period"""
        logger.info(f"Starting endurance test: {duration} seconds")
        
        endurance_results = []
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time:
            # Run short load test
            load_result = self.run_load_test(duration=60)
            
            # Collect system metrics
            system_metrics = self.collect_system_metrics()
            app_metrics = self.collect_application_metrics()
            
            # Combine results
            result = {
                'elapsed_time': time.time() - start_time,
                'load_test': load_result,
                'system_metrics': system_metrics,
                'application_metrics': app_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            endurance_results.append(result)
            
            # Check for degradation
            if (load_result['error_rate'] > 0.05 or 
                load_result['response_time_p95'] > self.config['thresholds']['response_time_p95'] * 2):
                logger.warning("Performance degradation detected in endurance test")
            
            time.sleep(60)  # Test every minute
        
        return {
            'duration': duration,
            'total_samples': len(endurance_results),
            'results': endurance_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_performance_thresholds(self, metrics: Dict) -> Dict:
        """Validate performance against thresholds"""
        thresholds = self.config['thresholds']
        validation_results = {
            'passed': True,
            'violations': [],
            'scores': {}
        }
        
        # Check response time thresholds
        if 'response_time_p95' in metrics:
            if metrics['response_time_p95'] > thresholds['response_time_p95']:
                validation_results['passed'] = False
                validation_results['violations'].append({
                    'metric': 'response_time_p95',
                    'actual': metrics['response_time_p95'],
                    'threshold': thresholds['response_time_p95']
                })
        
        # Check throughput threshold
        if 'throughput_rps' in metrics:
            if metrics['throughput_rps'] < thresholds['throughput_min']:
                validation_results['passed'] = False
                validation_results['violations'].append({
                    'metric': 'throughput_rps',
                    'actual': metrics['throughput_rps'],
                    'threshold': thresholds['throughput_min']
                })
        
        # Check error rate threshold
        if 'error_rate' in metrics:
            if metrics['error_rate'] > thresholds['error_rate_max']:
                validation_results['passed'] = False
                validation_results['violations'].append({
                    'metric': 'error_rate',
                    'actual': metrics['error_rate'],
                    'threshold': thresholds['error_rate_max']
                })
        
        # Calculate performance scores
        validation_results['scores'] = self._calculate_performance_scores(metrics)
        
        return validation_results
    
    def _calculate_performance_scores(self, metrics: Dict) -> Dict:
        """Calculate performance scores"""
        scores = {}
        
        # Response time score (lower is better)
        if 'response_time_p95' in metrics:
            threshold = self.config['thresholds']['response_time_p95']
            score = max(0, 100 - (metrics['response_time_p95'] / threshold * 100))
            scores['response_time'] = min(100, score)
        
        # Throughput score (higher is better)
        if 'throughput_rps' in metrics:
            threshold = self.config['thresholds']['throughput_min']
            score = (metrics['throughput_rps'] / threshold * 100)
            scores['throughput'] = min(100, score)
        
        # Error rate score (lower is better)
        if 'error_rate' in metrics:
            threshold = self.config['thresholds']['error_rate_max']
            score = max(0, 100 - (metrics['error_rate'] / threshold * 100))
            scores['error_rate'] = min(100, score)
        
        # Overall score
        if scores:
            scores['overall'] = sum(scores.values()) / len(scores)
        
        return scores
    
    def compare_performance(self, baseline: Dict, current: Dict) -> Dict:
        """Compare performance between baseline and current metrics"""
        comparison = {
            'improvements': {},
            'degradations': {},
            'summary': {
                'total_improvements': 0,
                'total_degradations': 0,
                'overall_improvement': 0
            }
        }
        
        # Compare key metrics
        key_metrics = [
            'response_time_p50', 'response_time_p95', 'response_time_p99',
            'throughput_rps', 'error_rate'
        ]
        
        for metric in key_metrics:
            if metric in baseline and metric in current:
                baseline_val = baseline[metric]
                current_val = current[metric]
                
                if baseline_val > 0:
                    # For response time and error rate, lower is better
                    if metric.startswith('response_time') or metric == 'error_rate':
                        improvement = ((baseline_val - current_val) / baseline_val) * 100
                    else:
                        # For throughput, higher is better
                        improvement = ((current_val - baseline_val) / baseline_val) * 100
                    
                    if improvement > 0:
                        comparison['improvements'][metric] = improvement
                        comparison['summary']['total_improvements'] += 1
                    elif improvement < 0:
                        comparison['degradations'][metric] = abs(improvement)
                        comparison['summary']['total_degradations'] += 1
        
        # Calculate overall improvement
        if comparison['improvements']:
            comparison['summary']['overall_improvement'] = (
                sum(comparison['improvements'].values()) / len(comparison['improvements'])
            )
        
        return comparison
    
    def generate_performance_report(self, test_results: Dict) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            'summary': {
                'test_date': datetime.now().isoformat(),
                'test_duration': test_results.get('duration', 0),
                'total_requests': test_results.get('total_requests', 0),
                'success_rate': 1 - test_results.get('error_rate', 0),
                'throughput_rps': test_results.get('throughput_rps', 0),
                'response_time_p95': test_results.get('response_time_p95', 0)
            },
            'detailed_metrics': test_results,
            'threshold_validation': self.validate_performance_thresholds(test_results),
            'recommendations': self._generate_performance_recommendations(test_results),
            'config': self.config
        }
        
        return report
    
    def _generate_performance_recommendations(self, metrics: Dict) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        thresholds = self.config['thresholds']
        
        # Response time recommendations
        if metrics.get('response_time_p95', 0) > thresholds['response_time_p95']:
            recommendations.append("Consider implementing response caching")
            recommendations.append("Optimize database queries and indexes")
            recommendations.append("Review and optimize slow endpoints")
        
        # Throughput recommendations
        if metrics.get('throughput_rps', 0) < thresholds['throughput_min']:
            recommendations.append("Consider horizontal scaling")
            recommendations.append("Optimize application resource usage")
            recommendations.append("Implement connection pooling")
        
        # Error rate recommendations
        if metrics.get('error_rate', 0) > thresholds['error_rate_max']:
            recommendations.append("Review error handling and logging")
            recommendations.append("Implement circuit breakers")
            recommendations.append("Add health checks and monitoring")
        
        # General recommendations
        recommendations.extend([
            "Monitor key performance metrics continuously",
            "Implement auto-scaling based on demand",
            "Regular performance testing and optimization",
            "Consider CDN for static content delivery"
        ])
        
        return recommendations
    
    def run_comprehensive_validation(self) -> Dict:
        """Run comprehensive performance validation"""
        logger.info("Starting comprehensive performance validation")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'baseline_metrics': self.collect_application_metrics(),
            'load_test': self.run_load_test(),
            'system_metrics': self.collect_system_metrics(),
            'validation_results': None,
            'recommendations': []
        }
        
        # Validate against thresholds
        results['validation_results'] = self.validate_performance_thresholds(results['load_test'])
        
        # Generate report
        report = self.generate_performance_report(results['load_test'])
        results['report'] = report
        
        # Generate recommendations
        results['recommendations'] = self._generate_performance_recommendations(results['load_test'])
        
        logger.info("Performance validation completed")
        return results
    
    def save_results(self, results: Dict, filename: str = None):
        """Save results to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_validation_{timestamp}.json"
        
        reports_dir = Path("./reports/performance")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = reports_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
        return filepath


def main():
    """Main function for performance validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance Validation Suite')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--test-type', choices=['load', 'stress', 'endurance', 'comprehensive'],
                       default='comprehensive', help='Type of test to run')
    parser.add_argument('--duration', type=int, default=300, help='Test duration in seconds')
    parser.add_argument('--users', type=int, default=100, help='Number of concurrent users')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = PerformanceValidator(args.config)
    
    # Run selected test
    if args.test_type == 'load':
        results = validator.run_load_test(args.duration, args.users)
    elif args.test_type == 'stress':
        results = validator.run_stress_test(args.users)
    elif args.test_type == 'endurance':
        results = validator.run_endurance_test(args.duration)
    else:
        results = validator.run_comprehensive_validation()
    
    # Save results
    validator.save_results(results, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE VALIDATION RESULTS")
    print("="*60)
    
    if 'load_test' in results:
        load_test = results['load_test']
        print(f"Throughput: {load_test['throughput_rps']:.1f} RPS")
        print(f"Response Time P95: {load_test['response_time_p95']:.1f}ms")
        print(f"Error Rate: {load_test['error_rate']:.2%}")
    
    if 'validation_results' in results and results['validation_results']:
        validation = results['validation_results']
        print(f"Validation: {'PASSED' if validation['passed'] else 'FAILED'}")
        if validation['violations']:
            print("\nThreshold Violations:")
            for violation in validation['violations']:
                print(f"  - {violation['metric']}: {violation['actual']} (threshold: {violation['threshold']})")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()