"""Integration tests for staging environment deployment."""

import asyncio
import aiohttp
import pytest
import json
import logging
import time
from typing import Dict, Any, List
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ServiceEndpoint:
    name: str
    base_url: str
    health_path: str = "/health"
    ready_path: str = "/ready"

class StagingIntegrationTest:
    """Integration test suite for staging environment."""
    
    def __init__(self):
        self.services = [
            ServiceEndpoint("data-quality", "http://localhost:8000"),
            ServiceEndpoint("machine-learning", "http://localhost:8001"),
            ServiceEndpoint("mlops", "http://localhost:8002"),
            ServiceEndpoint("anomaly-detection", "http://localhost:8003")
        ]
        self.test_results = {}
        self.session = None
    
    async def setup(self):
        """Setup test environment."""
        logger.info("Setting up staging integration tests...")
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def teardown(self):
        """Cleanup test environment."""
        if self.session:
            await self.session.close()
        logger.info("Test teardown completed")
    
    async def test_service_health_checks(self):
        """Test health endpoints for all services."""
        logger.info("Testing service health checks...")
        results = {}
        
        for service in self.services:
            try:
                async with self.session.get(f"{service.base_url}{service.health_path}") as response:
                    results[service.name] = {
                        "status_code": response.status,
                        "healthy": response.status == 200,
                        "response_time": time.time()
                    }
                    if response.status == 200:
                        try:
                            body = await response.json()
                            results[service.name]["response_body"] = body
                        except:
                            results[service.name]["response_body"] = await response.text()
                    
                    logger.info(f"{service.name} health check: {response.status}")
            except Exception as e:
                results[service.name] = {
                    "status_code": None,
                    "healthy": False,
                    "error": str(e)
                }
                logger.error(f"{service.name} health check failed: {e}")
        
        self.test_results["health_checks"] = results
        return results
    
    async def test_service_readiness(self):
        """Test readiness endpoints for all services."""
        logger.info("Testing service readiness...")
        results = {}
        
        for service in self.services:
            try:
                async with self.session.get(f"{service.base_url}{service.ready_path}") as response:
                    results[service.name] = {
                        "status_code": response.status,
                        "ready": response.status == 200,
                        "response_time": time.time()
                    }
                    logger.info(f"{service.name} readiness check: {response.status}")
            except Exception as e:
                results[service.name] = {
                    "status_code": None,
                    "ready": False,
                    "error": str(e)
                }
                logger.error(f"{service.name} readiness check failed: {e}")
        
        self.test_results["readiness_checks"] = results
        return results
    
    async def test_data_quality_api(self):
        """Test Data Quality service API endpoints."""
        logger.info("Testing Data Quality API...")
        service_url = "http://localhost:8000"
        results = {}
        
        # Test profile creation
        try:
            profile_data = {
                "data_source": "staging_test_data",
                "timestamp": "2024-01-01T00:00:00Z"
            }
            
            async with self.session.post(
                f"{service_url}/api/v1/profiles", 
                json=profile_data
            ) as response:
                results["create_profile"] = {
                    "status_code": response.status,
                    "success": response.status in [200, 201]
                }
                if response.status in [200, 201]:
                    results["create_profile"]["response"] = await response.json()
        except Exception as e:
            results["create_profile"] = {"error": str(e), "success": False}
        
        # Test validation
        try:
            validation_data = {
                "data_source": "staging_test_data",
                "rules": [
                    {"rule_name": "not_null_check", "description": "Check for null values"}
                ]
            }
            
            async with self.session.post(
                f"{service_url}/api/v1/validate",
                json=validation_data
            ) as response:
                results["validate_data"] = {
                    "status_code": response.status,
                    "success": response.status == 200
                }
                if response.status == 200:
                    results["validate_data"]["response"] = await response.json()
        except Exception as e:
            results["validate_data"] = {"error": str(e), "success": False}
        
        self.test_results["data_quality_api"] = results
        return results
    
    async def test_machine_learning_api(self):
        """Test Machine Learning service API endpoints."""
        logger.info("Testing Machine Learning API...")
        service_url = "http://localhost:8001"
        results = {}
        
        # Test data ingestion
        try:
            ingestion_data = {
                "source": "staging_ml_test",
                "features": [[1, 2, 3], [4, 5, 6]],
                "labels": [0, 1]
            }
            
            async with self.session.post(
                f"{service_url}/api/v1/ingest",
                json=ingestion_data
            ) as response:
                results["ingest_data"] = {
                    "status_code": response.status,
                    "success": response.status in [200, 201]
                }
                if response.status in [200, 201]:
                    results["ingest_data"]["response"] = await response.json()
        except Exception as e:
            results["ingest_data"] = {"error": str(e), "success": False}
        
        # Test prediction
        try:
            prediction_data = {
                "model_id": "test_model",
                "features": [1, 2, 3]
            }
            
            async with self.session.post(
                f"{service_url}/api/v1/predict",
                json=prediction_data
            ) as response:
                results["predict"] = {
                    "status_code": response.status,
                    "success": response.status == 200
                }
                if response.status == 200:
                    results["predict"]["response"] = await response.json()
        except Exception as e:
            results["predict"] = {"error": str(e), "success": False}
        
        self.test_results["machine_learning_api"] = results
        return results
    
    async def test_mlops_api(self):
        """Test MLOps service API endpoints."""
        logger.info("Testing MLOps API...")
        service_url = "http://localhost:8002"
        results = {}
        
        # Test pipeline creation
        try:
            pipeline_data = {
                "pipeline_name": "staging_test_pipeline",
                "steps": ["ingest", "process", "train", "deploy"]
            }
            
            async with self.session.post(
                f"{service_url}/api/v1/pipelines",
                json=pipeline_data
            ) as response:
                results["create_pipeline"] = {
                    "status_code": response.status,
                    "success": response.status in [200, 201]
                }
                if response.status in [200, 201]:
                    results["create_pipeline"]["response"] = await response.json()
        except Exception as e:
            results["create_pipeline"] = {"error": str(e), "success": False}
        
        # Test configuration
        try:
            async with self.session.get(f"{service_url}/api/v1/config") as response:
                results["get_config"] = {
                    "status_code": response.status,
                    "success": response.status == 200
                }
                if response.status == 200:
                    results["get_config"]["response"] = await response.json()
        except Exception as e:
            results["get_config"] = {"error": str(e), "success": False}
        
        self.test_results["mlops_api"] = results
        return results
    
    async def test_anomaly_detection_api(self):
        """Test Anomaly Detection service API endpoints."""
        logger.info("Testing Anomaly Detection API...")
        service_url = "http://localhost:8003"
        results = {}
        
        # Test anomaly detection
        try:
            detection_data = {
                "data_points": [1.0, 2.0, 1.5, 100.0, 1.8],  # 100.0 should be anomalous
                "threshold": 0.95
            }
            
            async with self.session.post(
                f"{service_url}/api/v1/detect",
                json=detection_data
            ) as response:
                results["detect_anomalies"] = {
                    "status_code": response.status,
                    "success": response.status == 200
                }
                if response.status == 200:
                    results["detect_anomalies"]["response"] = await response.json()
        except Exception as e:
            results["detect_anomalies"] = {"error": str(e), "success": False}
        
        self.test_results["anomaly_detection_api"] = results
        return results
    
    async def test_cross_service_integration(self):
        """Test integration between services."""
        logger.info("Testing cross-service integration...")
        results = {}
        
        try:
            # Simulate workflow: Ingest data -> Profile -> Detect anomalies
            
            # 1. Ingest data via ML service
            ml_data = {
                "source": "integration_test",
                "features": [[i, i*2, i*3] for i in range(100)],
                "labels": [i % 2 for i in range(100)]
            }
            
            async with self.session.post(
                "http://localhost:8001/api/v1/ingest",
                json=ml_data
            ) as response:
                if response.status in [200, 201]:
                    # 2. Profile the data via Data Quality service
                    profile_data = {
                        "data_source": "integration_test",
                        "timestamp": "2024-01-01T00:00:00Z"
                    }
                    
                    async with self.session.post(
                        "http://localhost:8000/api/v1/profiles",
                        json=profile_data
                    ) as profile_response:
                        if profile_response.status in [200, 201]:
                            # 3. Run anomaly detection
                            anomaly_data = {
                                "data_points": [i for i in range(100)],
                                "threshold": 0.95
                            }
                            
                            async with self.session.post(
                                "http://localhost:8003/api/v1/detect",
                                json=anomaly_data
                            ) as anomaly_response:
                                results["cross_service_workflow"] = {
                                    "ml_ingestion": response.status in [200, 201],
                                    "dq_profiling": profile_response.status in [200, 201],
                                    "anomaly_detection": anomaly_response.status == 200,
                                    "overall_success": all([
                                        response.status in [200, 201],
                                        profile_response.status in [200, 201],
                                        anomaly_response.status == 200
                                    ])
                                }
        except Exception as e:
            results["cross_service_workflow"] = {
                "error": str(e),
                "overall_success": False
            }
        
        self.test_results["cross_service_integration"] = results
        return results
    
    async def test_performance_under_load(self):
        """Test performance under concurrent load."""
        logger.info("Testing performance under load...")
        results = {}
        
        async def make_concurrent_requests(service_url: str, endpoint: str, data: dict, count: int):
            tasks = []
            start_time = time.time()
            
            for _ in range(count):
                tasks.append(self.session.post(f"{service_url}{endpoint}", json=data))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status == 200)
            return {
                "total_requests": count,
                "successful_requests": successful,
                "success_rate": successful / count,
                "total_time": end_time - start_time,
                "requests_per_second": count / (end_time - start_time)
            }
        
        # Test each service under load
        load_tests = [
            ("data-quality", "http://localhost:8000", "/api/v1/profiles", {
                "data_source": "load_test", "timestamp": "2024-01-01T00:00:00Z"
            }),
            ("machine-learning", "http://localhost:8001", "/api/v1/predict", {
                "model_id": "test", "features": [1, 2, 3]
            }),
        ]
        
        for service_name, base_url, endpoint, test_data in load_tests:
            try:
                result = await make_concurrent_requests(base_url, endpoint, test_data, 20)
                results[f"{service_name}_load_test"] = result
                logger.info(f"{service_name} load test: {result['success_rate']:.1%} success rate")
            except Exception as e:
                results[f"{service_name}_load_test"] = {"error": str(e)}
        
        self.test_results["performance_load"] = results
        return results
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = 0
        passed_tests = 0
        
        for test_category, test_results in self.test_results.items():
            if isinstance(test_results, dict):
                for test_name, test_result in test_results.items():
                    total_tests += 1
                    if isinstance(test_result, dict):
                        if test_result.get("success", test_result.get("healthy", test_result.get("ready", False))):
                            passed_tests += 1
        
        return {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "detailed_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check health status
        health_results = self.test_results.get("health_checks", {})
        unhealthy_services = [name for name, result in health_results.items() 
                            if not result.get("healthy", False)]
        if unhealthy_services:
            recommendations.append(f"Fix health checks for services: {', '.join(unhealthy_services)}")
        
        # Check API functionality
        for api_test in ["data_quality_api", "machine_learning_api", "mlops_api", "anomaly_detection_api"]:
            api_results = self.test_results.get(api_test, {})
            failed_endpoints = [name for name, result in api_results.items() 
                              if not result.get("success", False)]
            if failed_endpoints:
                recommendations.append(f"Fix {api_test} endpoints: {', '.join(failed_endpoints)}")
        
        # Check performance
        load_results = self.test_results.get("performance_load", {})
        for test_name, result in load_results.items():
            if isinstance(result, dict) and result.get("success_rate", 0) < 0.9:
                recommendations.append(f"Improve performance for {test_name} (success rate: {result.get('success_rate', 0):.1%})")
        
        if not recommendations:
            recommendations.append("All tests passed successfully! Staging environment is ready for production.")
        
        return recommendations

async def run_integration_tests():
    """Run all staging integration tests."""
    test_suite = StagingIntegrationTest()
    
    try:
        await test_suite.setup()
        
        # Run all test categories
        await test_suite.test_service_health_checks()
        await test_suite.test_service_readiness()
        await test_suite.test_data_quality_api()
        await test_suite.test_machine_learning_api()
        await test_suite.test_mlops_api()
        await test_suite.test_anomaly_detection_api()
        await test_suite.test_cross_service_integration()
        await test_suite.test_performance_under_load()
        
        # Generate and display report
        report = test_suite.generate_test_report()
        
        print("\n" + "="*80)
        print("STAGING ENVIRONMENT INTEGRATION TEST REPORT")
        print("="*80)
        
        summary = report["test_summary"]
        print(f"\n📊 TEST SUMMARY:")
        print(f"  • Total tests: {summary['total_tests']}")
        print(f"  • Passed: {summary['passed_tests']}")
        print(f"  • Failed: {summary['failed_tests']}")
        print(f"  • Success rate: {summary['success_rate']:.1%}")
        
        print(f"\n🔍 DETAILED RESULTS:")
        for category, results in report["detailed_results"].items():
            print(f"  • {category.replace('_', ' ').title()}:")
            if isinstance(results, dict):
                for test_name, result in results.items():
                    status = "✅" if result.get("success", result.get("healthy", result.get("ready", False))) else "❌"
                    print(f"    {status} {test_name}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for recommendation in report["recommendations"]:
            print(f"  • {recommendation}")
        
        print("\n" + "="*80)
        
        return summary['success_rate'] > 0.8
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False
    finally:
        await test_suite.teardown()

if __name__ == "__main__":
    asyncio.run(run_integration_tests())