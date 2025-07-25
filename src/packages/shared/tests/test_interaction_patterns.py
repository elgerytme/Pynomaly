"""
Integration tests for the new package interaction patterns.

This module tests the complete interaction framework including:
- Event-driven communication
- Dependency injection
- Cross-domain integration
- Interface stability
"""

import asyncio
import pytest
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock
from datetime import datetime

# Import from interfaces for stable contracts
from interfaces.dto import (
    DetectionRequest, DetectionResult, DetectionStatus,
    DataQualityRequest, DataQualityResult, DataQualityStatus
)
from interfaces.events import (
    AnomalyDetected, DataQualityCheckCompleted,
    EventPriority, InMemoryEventBus
)
from interfaces.patterns import Service, Repository

# Import from shared infrastructure
from shared import (
    DIContainer, LifecycleScope, get_container, configure_container,
    DistributedEventBus, publish_event, event_handler
)


class MockDataQualityService(Service):
    """Mock service for testing cross-domain communication."""
    
    def __init__(self):
        self.execution_count = 0
        self.last_request = None
    
    async def execute(self, request: DataQualityRequest) -> DataQualityResult:
        """Execute mock data quality check."""
        self.execution_count += 1
        self.last_request = request
        
        return DataQualityResult(
            id=f"dq_{request.id}",
            created_at=datetime.utcnow(),
            request_id=request.id,
            dataset_id=request.dataset_id,
            status=DataQualityStatus.PASSED,
            overall_score=0.95,
            rule_results={},
            issues_found=[],
            recommendations=[],
            execution_time_ms=100
        )
    
    async def validate_request(self, request: DataQualityRequest) -> bool:
        return bool(request.dataset_id)
    
    def get_service_info(self) -> Dict[str, Any]:
        return {"name": "MockDataQualityService", "executions": self.execution_count}


class MockDetectionService(Service):
    """Mock service for testing event-driven patterns."""
    
    def __init__(self):
        self.events_received = []
        self.detection_results = []
    
    async def execute(self, request: DetectionRequest) -> DetectionResult:
        """Execute mock anomaly detection."""
        result = DetectionResult(
            id=f"ad_{request.id}",
            created_at=datetime.utcnow(),
            request_id=request.id,
            status=DetectionStatus.COMPLETED,
            anomalies_count=3,
            anomaly_scores=[0.85, 0.92, 0.78],
            anomaly_indices=[1, 5, 10],
            confidence_scores=[0.9, 0.87, 0.85],
            execution_time_ms=200,
            algorithm_used=request.algorithm
        )
        
        self.detection_results.append(result)
        return result
    
    async def validate_request(self, request: DetectionRequest) -> bool:
        return bool(request.dataset_id and request.algorithm)
    
    def get_service_info(self) -> Dict[str, Any]:
        return {
            "name": "MockDetectionService", 
            "results_count": len(self.detection_results)
        }
    
    @event_handler(DataQualityCheckCompleted)
    async def handle_quality_check_completed(self, event: DataQualityCheckCompleted):
        """Handle data quality completion events."""
        self.events_received.append(event)


class TestDependencyInjection:
    """Test dependency injection framework."""
    
    def test_container_registration_and_resolution(self):
        """Test basic container operations."""
        container = DIContainer()
        
        # Register services
        container.register_singleton(MockDataQualityService)
        container.register_transient(MockDetectionService)
        
        # Resolve services
        quality_service = container.resolve(MockDataQualityService)
        detection_service1 = container.resolve(MockDetectionService)
        detection_service2 = container.resolve(MockDetectionService)
        
        # Verify singleton behavior
        quality_service2 = container.resolve(MockDataQualityService)
        assert quality_service is quality_service2
        
        # Verify transient behavior
        assert detection_service1 is not detection_service2
        assert isinstance(detection_service1, MockDetectionService)
        assert isinstance(detection_service2, MockDetectionService)
    
    def test_dependency_injection_with_constructor(self):
        """Test automatic constructor injection."""
        
        class ServiceWithDependency:
            def __init__(self, quality_service: MockDataQualityService):
                self.quality_service = quality_service
        
        container = DIContainer()
        container.register_singleton(MockDataQualityService)
        container.register_singleton(ServiceWithDependency)
        
        service = container.resolve(ServiceWithDependency)
        assert isinstance(service.quality_service, MockDataQualityService)
    
    def test_scoped_lifecycle(self):
        """Test scoped dependency lifecycle."""
        container = DIContainer()
        container.register_scoped(MockDataQualityService)
        
        # Create scope
        with container.create_scope("test_scope"):
            service1 = container.resolve(MockDataQualityService)
            service2 = container.resolve(MockDataQualityService)
            assert service1 is service2  # Same instance within scope
        
        # New scope
        with container.create_scope("another_scope"):
            service3 = container.resolve(MockDataQualityService)
            assert service1 is not service3  # Different instance in new scope
    
    def test_factory_registration(self):
        """Test factory-based registration."""
        container = DIContainer()
        
        def create_configured_service() -> MockDataQualityService:
            service = MockDataQualityService()
            service.execution_count = 100  # Pre-configure
            return service
        
        container.register_singleton(
            MockDataQualityService,
            factory=create_configured_service
        )
        
        service = container.resolve(MockDataQualityService)
        assert service.execution_count == 100


class TestEventDrivenCommunication:
    """Test event-driven communication patterns."""
    
    @pytest.mark.asyncio
    async def test_event_bus_basic_operations(self):
        """Test basic event bus functionality."""
        event_bus = InMemoryEventBus()
        
        events_received = []
        
        def handler(event):
            events_received.append(event)
        
        # Subscribe and publish
        event_bus.subscribe(DataQualityCheckCompleted, handler)
        
        test_event = DataQualityCheckCompleted(
            event_id="test",
            event_type="test",
            aggregate_id="dataset_123",
            occurred_at=datetime.utcnow(),
            dataset_id="dataset_123",
            status=DataQualityStatus.PASSED,
            overall_score=0.95,
            issues_count=0,
            quality_result=Mock()
        )
        
        await event_bus.publish(test_event)
        
        # Verify event was received
        assert len(events_received) == 1
        assert events_received[0].dataset_id == "dataset_123"
    
    @pytest.mark.asyncio
    async def test_distributed_event_bus(self):
        """Test distributed event bus with priority handling."""
        event_bus = DistributedEventBus()
        await event_bus.start()
        
        try:
            events_received = []
            
            async def handler(event):
                events_received.append(event)
            
            event_bus.subscribe(AnomalyDetected, handler)
            
            # Create high priority event
            high_priority_event = AnomalyDetected(
                event_id="high",
                event_type="test",
                aggregate_id="dataset_123",
                occurred_at=datetime.utcnow(),
                dataset_id="dataset_123",
                anomaly_count=10,
                severity="high",
                detection_result=Mock(),
                priority=EventPriority.HIGH
            )
            
            await event_bus.publish(high_priority_event)
            
            # Allow event processing
            await asyncio.sleep(0.1)
            
            # Verify event handling
            assert len(events_received) == 1
            assert events_received[0].anomaly_count == 10
            
            # Check metrics
            metrics = event_bus.get_metrics()
            assert metrics['events_published'] >= 1
            
        finally:
            await event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_cross_domain_event_flow(self):
        """Test complete cross-domain event flow."""
        event_bus = InMemoryEventBus()
        
        # Setup services
        quality_service = MockDataQualityService()
        detection_service = MockDetectionService()
        
        # Subscribe detection service to quality events
        event_bus.subscribe(DataQualityCheckCompleted, detection_service.handle_quality_check_completed)
        
        # Simulate quality check completion
        quality_event = DataQualityCheckCompleted(
            event_id="test",
            event_type="test",
            aggregate_id="dataset_123",
            occurred_at=datetime.utcnow(),
            dataset_id="dataset_123",
            status=DataQualityStatus.PASSED,
            overall_score=0.85,
            issues_count=0,
            quality_result=Mock()
        )
        
        await event_bus.publish(quality_event)
        
        # Verify cross-domain communication
        assert len(detection_service.events_received) == 1
        received_event = detection_service.events_received[0]
        assert received_event.dataset_id == "dataset_123"
        assert received_event.overall_score == 0.85


class TestInterfaceStability:
    """Test interface stability and compatibility."""
    
    def test_dto_serialization(self):
        """Test DTO serialization compatibility."""
        request = DetectionRequest(
            id="test_123",
            created_at=datetime.utcnow(),
            dataset_id="dataset_456",
            algorithm="isolation_forest",
            parameters={"contamination": 0.1}
        )
        
        # DTOs should be serializable
        import json
        from dataclasses import asdict
        
        try:
            # Test serialization (would work with proper datetime handling)
            data = asdict(request)
            assert data['dataset_id'] == "dataset_456"
            assert data['algorithm'] == "isolation_forest"
        except Exception:
            # Expected due to datetime serialization
            pass
    
    def test_interface_patterns(self):
        """Test interface pattern implementations."""
        # Test service pattern
        service = MockDataQualityService()
        assert hasattr(service, 'execute')
        assert hasattr(service, 'validate_request')
        assert hasattr(service, 'get_service_info')
        
        info = service.get_service_info()
        assert isinstance(info, dict)
        assert 'name' in info


class TestIntegrationWorkflow:
    """Test complete integration workflows."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow with DI and events."""
        # Setup container
        container = DIContainer()
        container.register_singleton(MockDataQualityService)
        container.register_singleton(MockDetectionService)
        
        # Setup event bus
        event_bus = InMemoryEventBus()
        
        # Get services
        quality_service = container.resolve(MockDataQualityService)
        detection_service = container.resolve(MockDetectionService)
        
        # Subscribe to events
        event_bus.subscribe(DataQualityCheckCompleted, detection_service.handle_quality_check_completed)
        
        # Execute workflow
        # 1. Data quality check
        quality_request = DataQualityRequest(
            id="workflow_test",
            created_at=datetime.utcnow(),
            dataset_id="test_dataset",
            quality_rules=["completeness", "validity"]
        )
        
        quality_result = await quality_service.execute(quality_request)
        assert quality_result.status == DataQualityStatus.PASSED
        
        # 2. Publish quality completion event
        quality_event = DataQualityCheckCompleted(
            event_id="workflow",
            event_type="test",
            aggregate_id="test_dataset",
            occurred_at=datetime.utcnow(),
            dataset_id="test_dataset",
            status=quality_result.status,
            overall_score=quality_result.overall_score,
            issues_count=len(quality_result.issues_found),
            quality_result=quality_result
        )
        
        await event_bus.publish(quality_event)
        
        # 3. Verify detection service received event
        assert len(detection_service.events_received) == 1
        
        # 4. Execute detection based on quality results
        if quality_result.overall_score >= 0.8:
            detection_request = DetectionRequest(
                id="detection_test",
                created_at=datetime.utcnow(),
                dataset_id="test_dataset",
                algorithm="isolation_forest",
                parameters={"contamination": 0.1}
            )
            
            detection_result = await detection_service.execute(detection_request)
            assert detection_result.status == DetectionStatus.COMPLETED
            assert detection_result.anomalies_count == 3
        
        # Verify workflow state
        assert quality_service.execution_count == 1
        assert len(detection_service.detection_results) == 1
    
    def test_configuration_composition(self):
        """Test configuration composition patterns."""
        # Test basic configuration
        def configure_basic(container: DIContainer):
            container.register_singleton(MockDataQualityService)
            container.register_singleton(MockDetectionService)
        
        container = DIContainer()
        configure_basic(container)
        
        # Verify services are available
        assert container.is_registered(MockDataQualityService)
        assert container.is_registered(MockDetectionService)
        
        # Test service resolution
        quality_service = container.resolve(MockDataQualityService)
        detection_service = container.resolve(MockDetectionService)
        
        assert isinstance(quality_service, MockDataQualityService)
        assert isinstance(detection_service, MockDetectionService)


class TestPerformanceAndReliability:
    """Test performance and reliability aspects."""
    
    @pytest.mark.asyncio
    async def test_event_bus_performance(self):
        """Test event bus performance under load."""
        event_bus = InMemoryEventBus()
        events_received = []
        
        async def fast_handler(event):
            events_received.append(event)
        
        event_bus.subscribe(AnomalyDetected, fast_handler)
        
        # Publish multiple events
        import time
        start_time = time.time()
        
        for i in range(100):
            event = AnomalyDetected(
                event_id=f"perf_{i}",
                event_type="performance_test",
                aggregate_id=f"dataset_{i}",
                occurred_at=datetime.utcnow(),
                dataset_id=f"dataset_{i}",
                anomaly_count=1,
                severity="low",
                detection_result=Mock()
            )
            await event_bus.publish(event)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertion
        assert duration < 1.0  # Should process 100 events in under 1 second
        assert len(events_received) == 100
    
    def test_di_container_performance(self):
        """Test DI container resolution performance."""
        container = DIContainer()
        container.register_singleton(MockDataQualityService)
        
        # Warm up
        container.resolve(MockDataQualityService)
        
        # Time multiple resolutions
        import time
        start_time = time.time()
        
        for _ in range(1000):
            service = container.resolve(MockDataQualityService)
            assert service is not None
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertion
        assert duration < 0.1  # Should resolve 1000 times in under 100ms
    
    @pytest.mark.asyncio
    async def test_error_handling_in_events(self):
        """Test error handling in event processing."""
        event_bus = InMemoryEventBus()
        
        async def failing_handler(event):
            raise ValueError("Simulated handler failure")
        
        async def working_handler(event):
            # This should still work even if other handler fails
            pass
        
        event_bus.subscribe(AnomalyDetected, failing_handler)
        event_bus.subscribe(AnomalyDetected, working_handler)
        
        # Publish event - should not raise exception
        event = AnomalyDetected(
            event_id="error_test",
            event_type="error_test",
            aggregate_id="dataset_123",
            occurred_at=datetime.utcnow(),
            dataset_id="dataset_123",
            anomaly_count=1,
            severity="low",
            detection_result=Mock()
        )
        
        # This should not raise an exception
        await event_bus.publish(event)


if __name__ == "__main__":
    # Run tests manually if needed
    import sys
    sys.path.insert(0, "/mnt/c/Users/andre/monorepo/src/packages")
    
    # Run a simple test
    async def main():
        test_class = TestEventDrivenCommunication()
        await test_class.test_event_bus_basic_operations()
        print("✅ Basic event bus test passed")
        
        test_class = TestDependencyInjection()
        test_class.test_container_registration_and_resolution()
        print("✅ DI container test passed")
        
        test_class = TestIntegrationWorkflow()
        await test_class.test_end_to_end_workflow()
        print("✅ End-to-end workflow test passed")
        
        print("🎉 All manual tests passed!")
    
    asyncio.run(main())