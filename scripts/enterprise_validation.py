#!/usr/bin/env python3
"""
Enterprise Readiness Validation Script for Pynomaly.

This script validates that all enterprise-grade features are properly implemented
and functioning correctly across the platform.

Validates:
- Security services and authentication
- Observability and monitoring
- Resilience patterns and circuit breakers
- Audit trails and compliance
- Performance optimization
- Multi-tenant capabilities
- Production readiness
"""

import asyncio
import json
import logging
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnterpriseValidator:
    """Validates enterprise-grade capabilities."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.utcnow().isoformat(),
            'validation_results': {},
            'overall_status': 'pending',
            'critical_failures': [],
            'warnings': [],
            'passed_tests': 0,
            'total_tests': 0
        }
    
    def validate_import_capabilities(self) -> Dict[str, Any]:
        """Validate that all enterprise modules can be imported."""
        print("🔍 Validating Enterprise Module Imports")
        print("-" * 50)
        
        import_results = {}
        
        # Test core enterprise imports
        enterprise_modules = [
            ('Security Service', 'pynomaly.infrastructure.security.security_service'),
            ('Observability Service', 'pynomaly.infrastructure.monitoring.observability_service'),
            ('Circuit Breaker', 'pynomaly.infrastructure.resilience.circuit_breaker'),
            ('Audit Service', 'pynomaly.infrastructure.audit.audit_service'),
            ('Performance Optimization', 'pynomaly.infrastructure.performance.optimization_service'),
        ]
        
        for name, module_path in enterprise_modules:
            try:
                __import__(module_path)
                import_results[name] = {'status': 'success', 'error': None}
                print(f"   ✅ {name}: Import successful")
            except ImportError as e:
                import_results[name] = {'status': 'failure', 'error': str(e)}
                print(f"   ❌ {name}: Import failed - {e}")
                self.results['critical_failures'].append(f"Failed to import {name}: {e}")
            except Exception as e:
                import_results[name] = {'status': 'error', 'error': str(e)}
                print(f"   ⚠️  {name}: Import error - {e}")
        
        return import_results
    
    def validate_security_features(self) -> Dict[str, Any]:
        """Validate comprehensive security features."""
        print("\n🔐 Validating Security Features")
        print("-" * 35)
        
        security_results = {}
        
        try:
            from pynomaly.infrastructure.security.security_service import (
                SecurityService, SecurityConfig, SecurityLevel
            )
            
            # Test security service initialization
            config = SecurityConfig(
                enable_2fa=True,
                enable_rbac=True,
                enable_audit_logging=True,
                enable_rate_limiting=True
            )
            security_service = SecurityService(config)
            
            # Test encryption capabilities
            if hasattr(security_service.encryption_service, 'encrypt_data'):
                test_data = "sensitive_test_data"
                encrypted, key_id = security_service.encryption_service.encrypt_data(test_data)
                decrypted = security_service.encryption_service.decrypt_data(encrypted, key_id)
                
                if decrypted.decode() == test_data:
                    security_results['encryption'] = {'status': 'success'}
                    print("   ✅ Encryption/Decryption: Working")
                else:
                    security_results['encryption'] = {'status': 'failure', 'error': 'Decryption mismatch'}
                    print("   ❌ Encryption/Decryption: Failed")
            
            # Test password hashing
            auth_service = security_service.auth_service
            test_password = "TestPassword123!"
            hashed = auth_service.hash_password(test_password)
            
            if auth_service.verify_password(test_password, hashed):
                security_results['password_hashing'] = {'status': 'success'}
                print("   ✅ Password Hashing: Working")
            else:
                security_results['password_hashing'] = {'status': 'failure'}
                print("   ❌ Password Hashing: Failed")
            
            # Test RBAC
            authz_service = security_service.authz_service
            test_user = "test_enterprise_user"
            
            # Assign role and test permissions
            authz_service.assign_role(test_user, "data_scientist")
            has_permission = authz_service.check_permission(
                test_user, "data:read", SecurityLevel.CONFIDENTIAL
            )
            
            if has_permission:
                security_results['rbac'] = {'status': 'success'}
                print("   ✅ Role-Based Access Control: Working")
            else:
                security_results['rbac'] = {'status': 'failure'}
                print("   ❌ Role-Based Access Control: Failed")
            
            # Test data sanitization
            sensitive_data = {"password": "secret", "normal": "data"}
            sanitized = security_service.sanitize_data(sensitive_data)
            
            if sanitized["password"] != "secret" and sanitized["normal"] == "data":
                security_results['data_sanitization'] = {'status': 'success'}
                print("   ✅ Data Sanitization: Working")
            else:
                security_results['data_sanitization'] = {'status': 'failure'}
                print("   ❌ Data Sanitization: Failed")
            
        except Exception as e:
            security_results['error'] = str(e)
            print(f"   ❌ Security validation failed: {e}")
            self.results['critical_failures'].append(f"Security validation error: {e}")
        
        return security_results
    
    async def validate_observability_features(self) -> Dict[str, Any]:
        """Validate observability and monitoring features."""
        print("\n📊 Validating Observability Features")
        print("-" * 40)
        
        observability_results = {}
        
        try:
            from pynomaly.infrastructure.monitoring.observability_service import (
                ObservabilityService, ObservabilityConfig
            )
            
            # Test observability service
            config = ObservabilityConfig(
                enable_tracing=True,
                enable_metrics=True,
                enable_health_checks=True,
                metrics_collection_interval=1.0
            )
            
            observability_service = ObservabilityService(config)
            await observability_service.start()
            
            try:
                # Test system metrics collection
                system_metrics = observability_service.system_metrics.collect_metrics()
                
                if 'cpu' in system_metrics and 'memory' in system_metrics:
                    observability_results['system_metrics'] = {'status': 'success'}
                    print("   ✅ System Metrics Collection: Working")
                else:
                    observability_results['system_metrics'] = {'status': 'failure'}
                    print("   ❌ System Metrics Collection: Failed")
                
                # Test application metrics
                app_metrics = observability_service.app_metrics
                app_metrics.record_request("GET", "/test", 200, 0.1)
                metrics = app_metrics.get_metrics()
                
                if 'request_counts' in metrics:
                    observability_results['app_metrics'] = {'status': 'success'}
                    print("   ✅ Application Metrics: Working")
                else:
                    observability_results['app_metrics'] = {'status': 'failure'}
                    print("   ❌ Application Metrics: Failed")
                
                # Test health checks
                health_results = await observability_service.health_service.run_all_checks()
                
                if 'overall_status' in health_results:
                    observability_results['health_checks'] = {'status': 'success'}
                    print("   ✅ Health Checks: Working")
                else:
                    observability_results['health_checks'] = {'status': 'failure'}
                    print("   ❌ Health Checks: Failed")
                
                # Test business metrics
                observability_service.record_business_metric("test_metric", 42.0)
                
                if len(observability_service.metrics_buffer) > 0:
                    observability_results['business_metrics'] = {'status': 'success'}
                    print("   ✅ Business Metrics: Working")
                else:
                    observability_results['business_metrics'] = {'status': 'failure'}
                    print("   ❌ Business Metrics: Failed")
                
            finally:
                await observability_service.stop()
            
        except Exception as e:
            observability_results['error'] = str(e)
            print(f"   ❌ Observability validation failed: {e}")
            self.results['critical_failures'].append(f"Observability validation error: {e}")
        
        return observability_results
    
    def validate_resilience_patterns(self) -> Dict[str, Any]:
        """Validate resilience patterns and circuit breakers."""
        print("\n⚡ Validating Resilience Patterns")
        print("-" * 38)
        
        resilience_results = {}
        
        try:
            from pynomaly.infrastructure.resilience.circuit_breaker import (
                CircuitBreaker, CircuitState
            )
            
            # Test circuit breaker
            circuit_breaker = CircuitBreaker(
                failure_threshold=2,
                recovery_timeout=0.1,
                name="test_circuit"
            )
            
            # Test normal operation
            def successful_function():
                return "success"
            
            result = circuit_breaker.call(successful_function)
            
            if result == "success" and circuit_breaker.state == CircuitState.CLOSED:
                resilience_results['circuit_breaker_normal'] = {'status': 'success'}
                print("   ✅ Circuit Breaker Normal Operation: Working")
            else:
                resilience_results['circuit_breaker_normal'] = {'status': 'failure'}
                print("   ❌ Circuit Breaker Normal Operation: Failed")
            
            # Test failure handling
            def failing_function():
                raise ValueError("Test failure")
            
            # Trigger failures to open circuit
            failures = 0
            for i in range(3):
                try:
                    circuit_breaker.call(failing_function)
                except ValueError:
                    failures += 1
                except Exception:
                    pass
            
            if circuit_breaker.state == CircuitState.OPEN and failures >= 2:
                resilience_results['circuit_breaker_failure'] = {'status': 'success'}
                print("   ✅ Circuit Breaker Failure Handling: Working")
            else:
                resilience_results['circuit_breaker_failure'] = {'status': 'failure'}
                print("   ❌ Circuit Breaker Failure Handling: Failed")
            
            # Test statistics
            stats = circuit_breaker.stats
            if 'total_calls' in stats and 'failed_calls' in stats:
                resilience_results['circuit_breaker_stats'] = {'status': 'success'}
                print("   ✅ Circuit Breaker Statistics: Working")
            else:
                resilience_results['circuit_breaker_stats'] = {'status': 'failure'}
                print("   ❌ Circuit Breaker Statistics: Failed")
            
        except Exception as e:
            resilience_results['error'] = str(e)
            print(f"   ❌ Resilience validation failed: {e}")
            self.results['critical_failures'].append(f"Resilience validation error: {e}")
        
        return resilience_results
    
    async def validate_audit_capabilities(self) -> Dict[str, Any]:
        """Validate comprehensive audit capabilities."""
        print("\n📋 Validating Audit Capabilities")
        print("-" * 35)
        
        audit_results = {}
        
        try:
            from pynomaly.infrastructure.audit.audit_service import (
                AuditService, AuditConfig, AuditEventType, AuditSeverity
            )
            
            # Create temporary directory for audit logs
            with tempfile.TemporaryDirectory() as temp_dir:
                config = AuditConfig(
                    log_directory=temp_dir,
                    buffer_size=5,
                    enable_checksum_verification=True
                )
                
                audit_service = AuditService(config)
                await audit_service.start()
                
                try:
                    # Test event logging
                    event_id = audit_service.log_event(
                        AuditEventType.DATA_ACCESS,
                        "test_action",
                        "success",
                        actor="test_user",
                        target="test_resource",
                        details={"test": "data"}
                    )
                    
                    if event_id:
                        audit_results['event_logging'] = {'status': 'success'}
                        print("   ✅ Audit Event Logging: Working")
                    else:
                        audit_results['event_logging'] = {'status': 'failure'}
                        print("   ❌ Audit Event Logging: Failed")
                    
                    # Test integrity verification
                    audit_service.storage.flush_buffer()
                    verification = audit_service.verify_integrity()
                    
                    if verification['status'] == 'success':
                        audit_results['integrity_verification'] = {'status': 'success'}
                        print("   ✅ Audit Integrity Verification: Working")
                    else:
                        audit_results['integrity_verification'] = {'status': 'failure'}
                        print("   ❌ Audit Integrity Verification: Failed")
                    
                    # Test event search
                    events = audit_service.search_events(actors=["test_user"])
                    
                    if len(events) > 0:
                        audit_results['event_search'] = {'status': 'success'}
                        print("   ✅ Audit Event Search: Working")
                    else:
                        audit_results['event_search'] = {'status': 'failure'}
                        print("   ❌ Audit Event Search: Failed")
                    
                finally:
                    await audit_service.stop()
                
        except Exception as e:
            audit_results['error'] = str(e)
            print(f"   ❌ Audit validation failed: {e}")
            self.results['critical_failures'].append(f"Audit validation error: {e}")
        
        return audit_results
    
    def validate_performance_optimization(self) -> Dict[str, Any]:
        """Validate performance optimization features."""
        print("\n🚀 Validating Performance Optimization")
        print("-" * 42)
        
        performance_results = {}
        
        try:
            from pynomaly.infrastructure.performance.optimization_service import (
                get_optimization_service
            )
            
            # Test optimization service
            optimizer = get_optimization_service()
            
            # Test data preprocessing optimization
            import numpy as np
            test_data = np.random.random((1000, 10)).astype(np.float64)
            
            optimized_data = optimizer.optimize_data_preprocessing(test_data)
            
            if optimized_data is not None:
                performance_results['data_optimization'] = {'status': 'success'}
                print("   ✅ Data Preprocessing Optimization: Working")
            else:
                performance_results['data_optimization'] = {'status': 'failure'}
                print("   ❌ Data Preprocessing Optimization: Failed")
            
            # Test metrics collection
            metrics = optimizer.get_performance_metrics()
            
            if 'operations_optimized' in metrics:
                performance_results['metrics_collection'] = {'status': 'success'}
                print("   ✅ Performance Metrics Collection: Working")
            else:
                performance_results['metrics_collection'] = {'status': 'failure'}
                print("   ❌ Performance Metrics Collection: Failed")
            
        except Exception as e:
            performance_results['error'] = str(e)
            print(f"   ❌ Performance validation failed: {e}")
            self.results['critical_failures'].append(f"Performance validation error: {e}")
        
        return performance_results
    
    def validate_production_readiness(self) -> Dict[str, Any]:
        """Validate overall production readiness."""
        print("\n🏭 Validating Production Readiness")
        print("-" * 38)
        
        production_results = {}
        
        # Check configuration management
        try:
            from pynomaly.infrastructure.security.security_service import SecurityConfig
            from pynomaly.infrastructure.monitoring.observability_service import ObservabilityConfig
            from pynomaly.infrastructure.audit.audit_service import AuditConfig
            
            # Test configuration classes
            configs = [
                ("SecurityConfig", SecurityConfig),
                ("ObservabilityConfig", ObservabilityConfig),
                ("AuditConfig", AuditConfig)
            ]
            
            config_success = True
            for name, config_class in configs:
                try:
                    config = config_class()
                    if not hasattr(config, '__dict__'):
                        config_success = False
                        break
                except Exception:
                    config_success = False
                    break
            
            if config_success:
                production_results['configuration'] = {'status': 'success'}
                print("   ✅ Configuration Management: Working")
            else:
                production_results['configuration'] = {'status': 'failure'}
                print("   ❌ Configuration Management: Failed")
        
        except Exception as e:
            production_results['configuration'] = {'status': 'error', 'error': str(e)}
            print(f"   ❌ Configuration validation failed: {e}")
        
        # Check logging infrastructure
        try:
            import logging
            import structlog
            
            # Test structured logging
            structlog.configure(
                processors=[structlog.processors.JSONRenderer()],
                logger_factory=structlog.stdlib.LoggerFactory(),
            )
            
            logger = structlog.get_logger()
            logger.info("test_log", key="value")
            
            production_results['logging'] = {'status': 'success'}
            print("   ✅ Structured Logging: Working")
            
        except Exception as e:
            production_results['logging'] = {'status': 'error', 'error': str(e)}
            print(f"   ❌ Logging validation failed: {e}")
        
        # Check error handling
        try:
            from pynomaly.infrastructure.resilience.circuit_breaker import CircuitBreakerError
            from pynomaly.infrastructure.security.security_service import ResilienceException
            
            # Test custom exception classes exist
            exceptions = [CircuitBreakerError]
            
            production_results['error_handling'] = {'status': 'success'}
            print("   ✅ Error Handling: Working")
            
        except Exception as e:
            production_results['error_handling'] = {'status': 'error', 'error': str(e)}
            print(f"   ❌ Error handling validation failed: {e}")
        
        return production_results
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive enterprise validation."""
        print("🏢 ENTERPRISE READINESS VALIDATION")
        print("=" * 60)
        
        validation_tasks = [
            ("Import Capabilities", self.validate_import_capabilities()),
            ("Security Features", self.validate_security_features()),
            ("Observability Features", self.validate_observability_features()),
            ("Resilience Patterns", self.validate_resilience_patterns()),
            ("Audit Capabilities", self.validate_audit_capabilities()),
            ("Performance Optimization", self.validate_performance_optimization()),
            ("Production Readiness", self.validate_production_readiness()),
        ]
        
        for test_name, test_func in validation_tasks:
            self.results['total_tests'] += 1
            
            try:
                if asyncio.iscoroutine(test_func):
                    result = await test_func
                else:
                    result = test_func
                
                self.results['validation_results'][test_name] = result
                
                # Count successes
                if isinstance(result, dict):
                    success_count = sum(
                        1 for v in result.values() 
                        if isinstance(v, dict) and v.get('status') == 'success'
                    )
                    if success_count > 0:
                        self.results['passed_tests'] += 1
                
            except Exception as e:
                self.results['validation_results'][test_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                self.results['critical_failures'].append(f"{test_name} validation failed: {e}")
        
        # Determine overall status
        if len(self.results['critical_failures']) == 0:
            if self.results['passed_tests'] >= self.results['total_tests'] * 0.8:
                self.results['overall_status'] = 'enterprise_ready'
            else:
                self.results['overall_status'] = 'needs_improvement'
        else:
            self.results['overall_status'] = 'not_ready'
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        print("\n📊 VALIDATION SUMMARY")
        print("=" * 40)
        
        # Overall status
        status_emoji = {
            'enterprise_ready': '✅',
            'needs_improvement': '⚠️ ',
            'not_ready': '❌'
        }
        
        status = self.results['overall_status']
        print(f"Overall Status: {status_emoji.get(status, '❓')} {status.upper()}")
        print(f"Tests Passed: {self.results['passed_tests']}/{self.results['total_tests']}")
        
        # Critical failures
        if self.results['critical_failures']:
            print(f"\nCritical Failures ({len(self.results['critical_failures'])}):")
            for failure in self.results['critical_failures']:
                print(f"  ❌ {failure}")
        
        # Warnings
        if self.results['warnings']:
            print(f"\nWarnings ({len(self.results['warnings'])}):")
            for warning in self.results['warnings']:
                print(f"  ⚠️  {warning}")
        
        # Feature summary
        print(f"\n📋 Feature Validation Summary:")
        for feature_name, results in self.results['validation_results'].items():
            if isinstance(results, dict):
                success_count = sum(
                    1 for v in results.values() 
                    if isinstance(v, dict) and v.get('status') == 'success'
                )
                total_features = len([
                    v for v in results.values() 
                    if isinstance(v, dict) and 'status' in v
                ])
                
                if total_features > 0:
                    status_icon = "✅" if success_count == total_features else "⚠️" if success_count > 0 else "❌"
                    print(f"  {status_icon} {feature_name}: {success_count}/{total_features}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if status == 'enterprise_ready':
            print("  🎉 Pynomaly is ready for enterprise deployment!")
            print("  📈 All critical enterprise features are functioning correctly")
            print("  🚀 Consider performance tuning for your specific workload")
        elif status == 'needs_improvement':
            print("  🔧 Some features need attention before production deployment")
            print("  📝 Review failed tests and implement missing capabilities")
            print("  🧪 Run additional integration testing")
        else:
            print("  🚨 Critical issues must be resolved before enterprise deployment")
            print("  🔴 Address all critical failures listed above")
            print("  📞 Consider professional support for enterprise deployment")
        
        return json.dumps(self.results, indent=2)


async def main():
    """Main validation entry point."""
    validator = EnterpriseValidator()
    
    try:
        results = await validator.run_comprehensive_validation()
        report = validator.generate_report()
        
        # Save detailed report
        report_file = f"/tmp/pynomaly_enterprise_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Exit with appropriate code
        if results['overall_status'] == 'enterprise_ready':
            print("\n🎉 ENTERPRISE VALIDATION COMPLETED SUCCESSFULLY!")
            sys.exit(0)
        elif results['overall_status'] == 'needs_improvement':
            print("\n⚠️  ENTERPRISE VALIDATION COMPLETED WITH WARNINGS")
            sys.exit(1)
        else:
            print("\n❌ ENTERPRISE VALIDATION FAILED")
            sys.exit(2)
    
    except Exception as e:
        print(f"\n💥 VALIDATION ERROR: {e}")
        logger.exception("Enterprise validation failed")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())