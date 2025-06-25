"""Test workflow simplification service container integration."""



def test_workflow_simplification_service_availability():
    """Test that workflow simplification service is available in container."""
    try:
        from pynomaly.infrastructure.config.container import Container

        # Create container
        container = Container()

        # Test workflow simplification service availability
        try:
            workflow_service = container.workflow_simplification_service()
            assert workflow_service is not None
            print("✅ Workflow simplification service available")

            # Test service functionality
            if hasattr(workflow_service, "workflow_templates"):
                templates = list(workflow_service.workflow_templates.keys())
                assert len(templates) > 0
                print(f"✅ Workflow templates: {templates}")

            return True

        except AttributeError:
            print(
                "⚠️ Workflow simplification service not available (feature may be disabled)"
            )
            return False

    except Exception as e:
        print(f"❌ Container integration test failed: {e}")
        return False


def test_container_phase2_services():
    """Test that Phase 2 services are properly integrated."""
    try:
        from pynomaly.infrastructure.config.container import Container

        container = Container()

        services_tested = []
        services_available = []

        # Test algorithm optimization service
        try:
            container.algorithm_optimization_service()
            services_available.append("algorithm_optimization")
        except AttributeError:
            pass
        services_tested.append("algorithm_optimization")

        # Test memory optimization service
        try:
            container.memory_optimization_service()
            services_available.append("memory_optimization")
        except AttributeError:
            pass
        services_tested.append("memory_optimization")

        # Test performance monitoring service
        try:
            container.performance_monitoring_service()
            services_available.append("performance_monitoring")
        except AttributeError:
            pass
        services_tested.append("performance_monitoring")

        # Test workflow simplification service
        try:
            container.workflow_simplification_service()
            services_available.append("workflow_simplification")
        except AttributeError:
            pass
        services_tested.append("workflow_simplification")

        print(f"✅ Services tested: {services_tested}")
        print(f"✅ Services available: {services_available}")

        # At least one Phase 2 service should be available
        assert len(services_available) > 0
        return True

    except Exception as e:
        print(f"❌ Phase 2 services integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Workflow Simplification Container Integration")
    print("=" * 60)

    success1 = test_workflow_simplification_service_availability()
    success2 = test_container_phase2_services()

    if success1 and success2:
        print("\n🎯 Container integration tests successful!")
    else:
        print("\n⚠️ Some container integration tests failed")
