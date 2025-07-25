"""
Comprehensive Load Testing Suite for Platform Performance

This module provides comprehensive load testing scenarios to validate
platform performance under various load conditions.
"""

import asyncio
import time
import statistics
import concurrent.futures
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, AsyncMock
import aiohttp
import json

# Performance testing utilities
class LoadTestMetrics:
    """Collect and analyze load test metrics"""
    
    def __init__(self):
        self.response_times = []
        self.success_count = 0
        self.error_count = 0
        self.throughput = 0
        self.errors = []
        self.start_time = None
        self.end_time = None
    
    def record_response(self, response_time: float, success: bool, error: str = None):
        """Record a response metric"""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
            if error:
                self.errors.append(error)
    
    def calculate_stats(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        if not self.response_times:
            return {}
        
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        total_requests = self.success_count + self.error_count
        
        return {
            'total_requests': total_requests,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / total_requests if total_requests > 0 else 0,
            'throughput': total_requests / duration if duration > 0 else 0,
            'response_times': {
                'min': min(self.response_times),
                'max': max(self.response_times),
                'mean': statistics.mean(self.response_times),
                'median': statistics.median(self.response_times),
                'p95': np.percentile(self.response_times, 95),
                'p99': np.percentile(self.response_times, 99)
            },
            'duration': duration
        }


class PlatformLoadTester:
    """Comprehensive platform load testing framework"""
    
    def __init__(self, base_url: str = 'http://localhost:8000'):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_request(self, endpoint: str, method: str = 'GET', data: Dict = None) -> Tuple[bool, float, str]:
        """Make a single HTTP request and return success, response time, and error"""
        start_time = time.time()
        try:
            if method == 'GET':
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    await response.text()
                    success = response.status < 400
            elif method == 'POST':
                async with self.session.post(f"{self.base_url}{endpoint}", json=data) as response:
                    await response.text()
                    success = response.status < 400
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response_time = time.time() - start_time
            return success, response_time, None
            
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time, str(e)
    
    async def run_load_test(self, endpoint: str, concurrent_users: int, duration: int, 
                           method: str = 'GET', data_generator=None) -> LoadTestMetrics:
        """Run a load test with specified parameters"""
        metrics = LoadTestMetrics()
        metrics.start_time = time.time()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def worker():
            """Worker coroutine for making requests"""
            while time.time() - metrics.start_time < duration:
                async with semaphore:
                    data = data_generator() if data_generator else None
                    success, response_time, error = await self.make_request(endpoint, method, data)
                    metrics.record_response(response_time, success, error)
                
                # Small delay to prevent overwhelming the server
                await asyncio.sleep(0.01)
        
        # Start worker tasks
        tasks = [asyncio.create_task(worker()) for _ in range(concurrent_users)]
        
        # Wait for test duration
        await asyncio.sleep(duration)
        
        # Cancel remaining tasks
        for task in tasks:
            task.cancel()
        
        # Wait for tasks to finish
        await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.end_time = time.time()
        return metrics


class TestPlatformLoadPerformance:
    """Comprehensive load testing suite"""
    
    @pytest.fixture
    def load_tester(self):
        """Create load tester instance"""
        return PlatformLoadTester()
    
    @pytest.fixture
    def sample_detection_data(self):
        """Generate sample data for anomaly detection requests"""
        def generator():
            return {
                'data': np.random.normal(0, 1, (100, 4)).tolist(),
                'algorithm': 'isolation_forest',
                'threshold': 0.8
            }
        return generator
    
    @pytest.mark.asyncio
    @pytest.mark.load_test
    async def test_anomaly_detection_endpoint_load(self, load_tester, sample_detection_data):
        """Test anomaly detection endpoint under load"""
        
        # Mock the actual service to avoid dependencies
        with patch('src.packages.data.anomaly_detection.src.anomaly_detection.application.services.anomaly_detection_service.AnomalyDetectionService') as mock_service:
            mock_service.return_value.detect_anomalies = AsyncMock(return_value={
                'anomalies_detected': 5,
                'confidence': 0.85,
                'processing_time': 0.1
            })
            
            async with load_tester:
                metrics = await load_tester.run_load_test(
                    endpoint='/api/v1/detect',
                    concurrent_users=10,
                    duration=30,  # 30 seconds
                    method='POST',
                    data_generator=sample_detection_data
                )
            
            stats = metrics.calculate_stats()
            
            # Performance assertions
            assert stats['error_rate'] < 0.05, f"Error rate too high: {stats['error_rate']:.2%}"
            assert stats['response_times']['p95'] < 2.0, f"P95 response time too high: {stats['response_times']['p95']:.3f}s"
            assert stats['throughput'] > 5, f"Throughput too low: {stats['throughput']:.2f} req/s"
            
            print(f"Anomaly Detection Load Test Results:")
            print(f"  Total Requests: {stats['total_requests']}")
            print(f"  Success Rate: {(1 - stats['error_rate']):.2%}")
            print(f"  Throughput: {stats['throughput']:.2f} req/s")
            print(f"  P95 Response Time: {stats['response_times']['p95']:.3f}s")
    
    @pytest.mark.asyncio
    @pytest.mark.load_test
    async def test_security_scanning_load(self, load_tester):
        """Test security scanning endpoint under load"""
        
        with patch('src.packages.data.anomaly_detection.src.anomaly_detection.application.services.security.vulnerability_scanner.VulnerabilityScanner') as mock_scanner:
            mock_scanner.return_value.scan_system = AsyncMock(return_value={
                'vulnerabilities_found': 2,
                'risk_score': 3.5,
                'scan_duration': 0.2
            })
            
            async with load_tester:
                metrics = await load_tester.run_load_test(
                    endpoint='/api/v1/security/scan',
                    concurrent_users=5,  # Lower concurrency for security scans
                    duration=20,
                    method='POST'
                )
            
            stats = metrics.calculate_stats()
            
            # Security scanning should be more resource-intensive
            assert stats['error_rate'] < 0.1, f"Error rate too high: {stats['error_rate']:.2%}"
            assert stats['response_times']['p95'] < 5.0, f"P95 response time too high: {stats['response_times']['p95']:.3f}s"
            assert stats['throughput'] > 1, f"Throughput too low: {stats['throughput']:.2f} req/s"
            
            print(f"Security Scanning Load Test Results:")
            print(f"  Total Requests: {stats['total_requests']}")
            print(f"  Success Rate: {(1 - stats['error_rate']):.2%}")
            print(f"  Throughput: {stats['throughput']:.2f} req/s")
            print(f"  P95 Response Time: {stats['response_times']['p95']:.3f}s")
    
    @pytest.mark.asyncio
    @pytest.mark.load_test
    async def test_analytics_dashboard_load(self, load_tester):
        """Test analytics dashboard endpoint under load"""
        
        def query_generator():
            return {
                'metric_type': np.random.choice(['cpu', 'memory', 'network', 'disk']),
                'time_range': '1h',
                'aggregation': 'avg'
            }
        
        with patch('src.packages.data.anomaly_detection.src.anomaly_detection.application.services.intelligence.analytics_engine.AnalyticsEngine') as mock_analytics:
            mock_analytics.return_value.query_metrics = AsyncMock(return_value={
                'data_points': 100,
                'insights': ['Trend analysis completed'],
                'processing_time': 0.05
            })
            
            async with load_tester:
                metrics = await load_tester.run_load_test(
                    endpoint='/api/v1/analytics/query',
                    concurrent_users=15,
                    duration=25,
                    method='POST',
                    data_generator=query_generator
                )
            
            stats = metrics.calculate_stats()
            
            # Analytics should be fast for dashboard queries
            assert stats['error_rate'] < 0.03, f"Error rate too high: {stats['error_rate']:.2%}"
            assert stats['response_times']['p95'] < 1.5, f"P95 response time too high: {stats['response_times']['p95']:.3f}s"
            assert stats['throughput'] > 10, f"Throughput too low: {stats['throughput']:.2f} req/s"
            
            print(f"Analytics Dashboard Load Test Results:")
            print(f"  Total Requests: {stats['total_requests']}")
            print(f"  Success Rate: {(1 - stats['error_rate']):.2%}")
            print(f"  Throughput: {stats['throughput']:.2f} req/s")
            print(f"  P95 Response Time: {stats['response_times']['p95']:.3f}s")
    
    @pytest.mark.asyncio
    @pytest.mark.load_test
    async def test_mixed_workload_scenario(self, load_tester, sample_detection_data):
        """Test mixed workload scenario with multiple endpoints"""
        
        # Define workload distribution
        workloads = [
            {'endpoint': '/api/v1/detect', 'method': 'POST', 'weight': 0.4, 'data_gen': sample_detection_data},
            {'endpoint': '/api/v1/health', 'method': 'GET', 'weight': 0.3, 'data_gen': None},
            {'endpoint': '/api/v1/analytics/query', 'method': 'POST', 'weight': 0.2, 'data_gen': lambda: {'metric': 'cpu'}},
            {'endpoint': '/api/v1/security/scan', 'method': 'POST', 'weight': 0.1, 'data_gen': None}
        ]
        
        # Mock all services
        with patch('src.packages.data.anomaly_detection.src.anomaly_detection.application.services.anomaly_detection_service.AnomalyDetectionService') as mock_detection, \
             patch('src.packages.data.anomaly_detection.src.anomaly_detection.application.services.intelligence.analytics_engine.AnalyticsEngine') as mock_analytics, \
             patch('src.packages.data.anomaly_detection.src.anomaly_detection.application.services.security.vulnerability_scanner.VulnerabilityScanner') as mock_security:
            
            # Configure mocks
            mock_detection.return_value.detect_anomalies = AsyncMock(return_value={'status': 'ok'})
            mock_analytics.return_value.query_metrics = AsyncMock(return_value={'status': 'ok'})
            mock_security.return_value.scan_system = AsyncMock(return_value={'status': 'ok'})
            
            async with load_tester:
                # Run concurrent tests for different workloads
                tasks = []
                for workload in workloads:
                    concurrent_users = max(1, int(20 * workload['weight']))  # Scale users by weight
                    task = asyncio.create_task(
                        load_tester.run_load_test(
                            endpoint=workload['endpoint'],
                            concurrent_users=concurrent_users,
                            duration=30,
                            method=workload['method'],
                            data_generator=workload['data_gen']
                        )
                    )
                    tasks.append((workload['endpoint'], task))
                
                # Wait for all tests to complete
                results = {}
                for endpoint, task in tasks:
                    results[endpoint] = await task
            
            # Analyze mixed workload results
            total_requests = sum(metrics.success_count + metrics.error_count for metrics in results.values())
            overall_error_rate = sum(metrics.error_count for metrics in results.values()) / total_requests
            
            assert overall_error_rate < 0.05, f"Overall error rate too high: {overall_error_rate:.2%}"
            assert total_requests > 100, f"Total requests too low: {total_requests}"
            
            print(f"Mixed Workload Test Results:")
            print(f"  Total Requests: {total_requests}")
            print(f"  Overall Error Rate: {overall_error_rate:.2%}")
            
            for endpoint, metrics in results.items():
                stats = metrics.calculate_stats()
                print(f"  {endpoint}: {stats['total_requests']} requests, "
                      f"{stats['throughput']:.2f} req/s, "
                      f"{stats['response_times']['p95']:.3f}s P95")
    
    @pytest.mark.asyncio
    @pytest.mark.load_test
    async def test_stress_testing_limits(self, load_tester):
        """Test system limits under extreme load"""
        
        # Gradually increase load to find breaking point
        load_levels = [10, 25, 50, 100, 200]
        results = {}
        
        with patch('src.packages.data.anomaly_detection.src.anomaly_detection.application.services.anomaly_detection_service.AnomalyDetectionService') as mock_service:
            mock_service.return_value.detect_anomalies = AsyncMock(return_value={'status': 'ok'})
            
            async with load_tester:
                for concurrent_users in load_levels:
                    print(f"Testing with {concurrent_users} concurrent users...")
                    
                    metrics = await load_tester.run_load_test(
                        endpoint='/api/v1/health',
                        concurrent_users=concurrent_users,
                        duration=15,  # Shorter duration for stress test
                        method='GET'
                    )
                    
                    stats = metrics.calculate_stats()
                    results[concurrent_users] = stats
                    
                    # Stop if error rate becomes too high
                    if stats['error_rate'] > 0.2:
                        print(f"Breaking point reached at {concurrent_users} users")
                        break
        
        # Analyze stress test results
        print(f"Stress Test Results:")
        for users, stats in results.items():
            print(f"  {users} users: {stats['throughput']:.2f} req/s, "
                  f"{stats['error_rate']:.2%} error rate, "
                  f"{stats['response_times']['p95']:.3f}s P95")
        
        # Validate that system can handle reasonable load
        assert any(stats['error_rate'] < 0.05 for stats in results.values()), "System cannot handle basic load"
    
    @pytest.mark.asyncio
    @pytest.mark.load_test
    async def test_sustained_load_endurance(self, load_tester):
        """Test system performance under sustained load"""
        
        with patch('src.packages.data.anomaly_detection.src.anomaly_detection.application.services.anomaly_detection_service.AnomalyDetectionService') as mock_service:
            mock_service.return_value.detect_anomalies = AsyncMock(return_value={'status': 'ok'})
            
            async with load_tester:
                # Run sustained load for longer duration
                metrics = await load_tester.run_load_test(
                    endpoint='/api/v1/health',
                    concurrent_users=20,
                    duration=120,  # 2 minutes sustained load
                    method='GET'
                )
            
            stats = metrics.calculate_stats()
            
            # Endurance test should maintain consistent performance
            assert stats['error_rate'] < 0.05, f"Error rate too high during endurance test: {stats['error_rate']:.2%}"
            assert stats['response_times']['p95'] < 2.0, f"P95 response time degraded: {stats['response_times']['p95']:.3f}s"
            assert stats['throughput'] > 5, f"Throughput degraded: {stats['throughput']:.2f} req/s"
            
            print(f"Sustained Load Endurance Test Results:")
            print(f"  Duration: {stats['duration']:.1f}s")
            print(f"  Total Requests: {stats['total_requests']}")
            print(f"  Average Throughput: {stats['throughput']:.2f} req/s")
            print(f"  Final Error Rate: {stats['error_rate']:.2%}")
            print(f"  P95 Response Time: {stats['response_times']['p95']:.3f}s")
    
    @pytest.mark.asyncio
    @pytest.mark.load_test
    async def test_memory_usage_under_load(self, load_tester):
        """Test memory usage patterns under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('src.packages.data.anomaly_detection.src.anomaly_detection.application.services.anomaly_detection_service.AnomalyDetectionService') as mock_service:
            mock_service.return_value.detect_anomalies = AsyncMock(return_value={'status': 'ok'})
            
            async with load_tester:
                metrics = await load_tester.run_load_test(
                    endpoint='/api/v1/health',
                    concurrent_users=50,
                    duration=30,
                    method='GET'
                )
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            stats = metrics.calculate_stats()
            
            # Memory usage should not grow excessively
            assert memory_increase < 100, f"Memory usage increased too much: {memory_increase:.2f}MB"
            
            print(f"Memory Usage Test Results:")
            print(f"  Initial Memory: {initial_memory:.2f}MB")
            print(f"  Final Memory: {final_memory:.2f}MB")
            print(f"  Memory Increase: {memory_increase:.2f}MB")
            print(f"  Requests Processed: {stats['total_requests']}")
            print(f"  Memory per Request: {memory_increase / stats['total_requests']:.4f}MB/req")
    
    def test_load_test_configuration_validation(self):
        """Validate load test configuration and setup"""
        
        # Test essential configuration parameters
        config = {
            'concurrent_users': [1, 10, 50, 100],
            'test_duration': [10, 30, 60, 120],
            'endpoints': ['/api/v1/health', '/api/v1/detect', '/api/v1/analytics/query'],
            'performance_thresholds': {
                'error_rate': 0.05,
                'p95_response_time': 2.0,
                'min_throughput': 5.0
            }
        }
        
        # Validate configuration completeness
        assert len(config['concurrent_users']) >= 3, "Need multiple concurrency levels"
        assert len(config['test_duration']) >= 3, "Need multiple test durations"
        assert len(config['endpoints']) >= 3, "Need multiple endpoints to test"
        assert all(threshold > 0 for threshold in config['performance_thresholds'].values()), "Invalid thresholds"
        
        print("Load test configuration validated successfully")


if __name__ == "__main__":
    # Run load tests with specific markers
    pytest.main([__file__, "-v", "-m", "load_test", "--tb=short"])