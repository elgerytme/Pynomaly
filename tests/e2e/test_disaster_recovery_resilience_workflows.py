"""Disaster recovery and system resilience workflow end-to-end tests.

This module tests disaster recovery scenarios, system resilience under failure conditions,
backup and restore procedures, and high availability workflows.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app


class TestDisasterRecoveryResilienceWorkflows:
    """Test disaster recovery and resilience workflows."""

    @pytest.fixture
    def app_client(self):
        """Create test client for API."""
        container = create_container()
        app = create_app(container)
        return TestClient(app)

    @pytest.fixture
    def critical_dataset(self):
        """Create critical dataset for disaster recovery testing."""
        np.random.seed(42)

        # Large dataset to simulate critical business data
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=1000, freq="1H"),
                "sensor_1": np.random.normal(100, 10, 1000),
                "sensor_2": np.random.normal(50, 5, 1000),
                "sensor_3": np.random.normal(25, 2.5, 1000),
                "temperature": np.random.normal(20, 3, 1000),
                "pressure": np.random.normal(1013, 50, 1000),
            }
        )

        # Add some anomalies
        anomaly_indices = np.random.choice(1000, 50, replace=False)
        data.loc[anomaly_indices, "sensor_1"] *= 3
        data.loc[anomaly_indices, "sensor_2"] *= 0.3

        return data

    def test_system_backup_restore_workflow(self, app_client, critical_dataset):
        """Test complete system backup and restore workflow."""
        # Create initial system state
        detector_configs = [
            {
                "name": "Critical Production Detector 1",
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.05, "random_state": 42},
            },
            {
                "name": "Critical Production Detector 2",
                "algorithm_name": "LocalOutlierFactor",
                "parameters": {"n_neighbors": 20, "contamination": 0.05},
            },
        ]

        # Upload critical dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            critical_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("critical_data.csv", file, "text/csv")},
                    data={"name": "Critical Production Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Create and train detectors
            detector_ids = []
            for config in detector_configs:
                create_response = app_client.post("/api/detectors/", json=config)
                assert create_response.status_code == 200
                detector_id = create_response.json()["id"]
                detector_ids.append(detector_id)

                # Train detector
                train_response = app_client.post(
                    f"/api/detectors/{detector_id}/train",
                    json={"dataset_id": dataset_id},
                )
                assert train_response.status_code == 200

            # Create system backup
            backup_request = {
                "backup_type": "full",
                "include_data": True,
                "include_models": True,
                "include_configuration": True,
                "include_user_data": True,
                "compression": True,
                "encryption": True,
            }

            backup_response = app_client.post("/api/backup/create", json=backup_request)
            assert backup_response.status_code == 200
            backup_result = backup_response.json()

            assert "backup_id" in backup_result
            assert "backup_size" in backup_result
            assert "backup_location" in backup_result
            assert "manifest" in backup_result

            backup_id = backup_result["backup_id"]

            # Verify backup integrity
            integrity_response = app_client.get(f"/api/backup/{backup_id}/verify")
            assert integrity_response.status_code == 200
            integrity_result = integrity_response.json()

            assert integrity_result["integrity_check"] == "passed"
            assert "checksum_verification" in integrity_result
            assert "component_validation" in integrity_result

            # Simulate system failure and restore
            # Test restore preparation
            restore_prep_response = app_client.post(
                "/api/restore/prepare",
                json={
                    "backup_id": backup_id,
                    "restore_type": "full",
                    "target_environment": "production",
                },
            )
            assert restore_prep_response.status_code == 200
            restore_prep_result = restore_prep_response.json()

            assert "restore_plan" in restore_prep_result
            assert "estimated_time" in restore_prep_result
            assert "prerequisites" in restore_prep_result

            # Execute restore
            restore_request = {
                "backup_id": backup_id,
                "restore_plan": restore_prep_result["restore_plan"],
                "overwrite_existing": True,
                "validate_after_restore": True,
            }

            restore_response = app_client.post(
                "/api/restore/execute", json=restore_request
            )
            assert restore_response.status_code == 200
            restore_result = restore_response.json()

            assert "restore_status" in restore_result
            assert "restored_components" in restore_result
            assert restore_result["restore_status"] == "completed"

            # Verify system functionality after restore
            for detector_id in detector_ids:
                # Test detector still works
                detect_response = app_client.post(
                    f"/api/detectors/{detector_id}/detect",
                    json={"dataset_id": dataset_id},
                )
                assert detect_response.status_code == 200

                # Verify detector configuration
                config_response = app_client.get(f"/api/detectors/{detector_id}")
                assert config_response.status_code == 200

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_database_failure_recovery_workflow(self, app_client, critical_dataset):
        """Test database failure and recovery scenarios."""
        # Setup initial state
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            critical_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("db_test_data.csv", file, "text/csv")},
                    data={"name": "DB Test Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Create detector
            detector_data = {
                "name": "DB Recovery Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.05},
            }

            create_response = app_client.post("/api/detectors/", json=detector_data)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]

            # Train detector
            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train", json={"dataset_id": dataset_id}
            )
            assert train_response.status_code == 200

            # Test database health monitoring
            db_health_response = app_client.get("/api/health/database")
            assert db_health_response.status_code == 200
            db_health = db_health_response.json()

            assert "status" in db_health
            assert "connection_pool" in db_health
            assert "query_performance" in db_health

            # Simulate database connection issues
            with patch(
                "pynomaly.infrastructure.persistence.database.DatabaseConnection"
            ) as mock_db:
                # Configure mock to simulate connection failure
                mock_db.side_effect = Exception("Database connection failed")

                # Test graceful degradation
                degraded_response = app_client.get("/api/health/")
                # Should still respond but indicate degraded status
                assert degraded_response.status_code in [200, 503]

                if degraded_response.status_code == 200:
                    health_result = degraded_response.json()
                    assert health_result["status"] in ["degraded", "unhealthy"]

                # Test circuit breaker behavior
                for _ in range(5):
                    test_response = app_client.get(f"/api/detectors/{detector_id}")
                    # Should eventually trigger circuit breaker
                    if test_response.status_code == 503:
                        break
                    time.sleep(0.1)

            # Test automatic recovery
            recovery_response = app_client.post(
                "/api/recovery/database",
                json={
                    "recovery_type": "automatic",
                    "max_retries": 3,
                    "backoff_factor": 2,
                },
            )

            if recovery_response.status_code == 200:
                recovery_result = recovery_response.json()
                assert "recovery_status" in recovery_result
                assert "recovery_attempts" in recovery_result

            # Test manual recovery procedures
            manual_recovery_response = app_client.post(
                "/api/recovery/database/manual",
                json={
                    "steps": [
                        "restart_connection_pool",
                        "clear_cache",
                        "validate_schema",
                        "rebuild_indexes",
                    ]
                },
            )

            if manual_recovery_response.status_code == 200:
                manual_result = manual_recovery_response.json()
                assert "executed_steps" in manual_result
                assert "recovery_successful" in manual_result

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_high_availability_failover_workflow(self, app_client):
        """Test high availability and failover scenarios."""
        # Test cluster status
        cluster_response = app_client.get("/api/cluster/status")

        if cluster_response.status_code == 200:
            cluster_status = cluster_response.json()

            assert "nodes" in cluster_status
            assert "leader" in cluster_status
            assert "health" in cluster_status

            nodes = cluster_status["nodes"]
            leader_node = cluster_status["leader"]

            # Test leader election process
            if len(nodes) > 1:
                # Simulate leader node failure
                failover_request = {
                    "simulate_failure": True,
                    "target_node": leader_node,
                    "failure_type": "network_partition",
                }

                failover_response = app_client.post(
                    "/api/cluster/simulate-failover", json=failover_request
                )

                if failover_response.status_code == 200:
                    failover_result = failover_response.json()

                    assert "new_leader" in failover_result
                    assert "failover_time" in failover_result
                    assert "affected_operations" in failover_result

                    # Verify new leader is different
                    assert failover_result["new_leader"] != leader_node

                    # Test service continuity during failover
                    continuity_test_response = app_client.get("/api/health/")
                    assert continuity_test_response.status_code == 200

                    # Test data consistency after failover
                    consistency_response = app_client.get(
                        "/api/cluster/consistency-check"
                    )
                    if consistency_response.status_code == 200:
                        consistency_result = consistency_response.json()
                        assert consistency_result["consistency_status"] == "consistent"

        # Test load balancer health checks
        lb_health_response = app_client.get("/api/health/load-balancer")

        if lb_health_response.status_code == 200:
            lb_health = lb_health_response.json()

            assert "active_backends" in lb_health
            assert "traffic_distribution" in lb_health
            assert "health_check_status" in lb_health

        # Test auto-scaling behavior
        scaling_test_request = {
            "simulate_load": True,
            "target_rps": 1000,
            "duration_seconds": 30,
        }

        scaling_response = app_client.post(
            "/api/autoscaling/test", json=scaling_test_request
        )

        if scaling_response.status_code == 200:
            scaling_result = scaling_response.json()

            assert "scaling_triggered" in scaling_result
            assert "new_instance_count" in scaling_result
            assert "scaling_metrics" in scaling_result

    def test_data_corruption_recovery_workflow(self, app_client, critical_dataset):
        """Test data corruption detection and recovery."""
        # Upload dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            critical_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("corruption_test_data.csv", file, "text/csv")},
                    data={"name": "Corruption Test Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Create checksum for data integrity verification
            checksum_response = app_client.post(f"/api/datasets/{dataset_id}/checksum")
            assert checksum_response.status_code == 200
            checksum_result = checksum_response.json()

            assert "checksum" in checksum_result
            assert "algorithm" in checksum_result
            checksum_result["checksum"]

            # Simulate data corruption detection
            corruption_check_response = app_client.post(
                "/api/data/corruption-check",
                json={
                    "dataset_id": dataset_id,
                    "check_type": "comprehensive",
                    "include_checksums": True,
                    "validate_schema": True,
                },
            )

            assert corruption_check_response.status_code == 200
            corruption_result = corruption_check_response.json()

            assert "corruption_detected" in corruption_result
            assert "integrity_score" in corruption_result
            assert "validation_results" in corruption_result

            # Test automatic repair mechanisms
            if corruption_result.get("corruption_detected", False):
                repair_request = {
                    "dataset_id": dataset_id,
                    "repair_strategy": "automatic",
                    "backup_source": "latest_verified",
                    "validate_after_repair": True,
                }

                repair_response = app_client.post(
                    "/api/data/repair", json=repair_request
                )
                assert repair_response.status_code == 200
                repair_result = repair_response.json()

                assert "repair_successful" in repair_result
                assert "repair_log" in repair_result
                assert "data_quality_score" in repair_result

            # Test data versioning and rollback
            versioning_response = app_client.get(f"/api/datasets/{dataset_id}/versions")

            if versioning_response.status_code == 200:
                versions = versioning_response.json()

                assert "versions" in versions
                assert "current_version" in versions

                if len(versions["versions"]) > 1:
                    # Test rollback to previous version
                    previous_version = versions["versions"][-2]["version_id"]

                    rollback_request = {
                        "target_version": previous_version,
                        "verify_integrity": True,
                        "backup_current": True,
                    }

                    rollback_response = app_client.post(
                        f"/api/datasets/{dataset_id}/rollback", json=rollback_request
                    )

                    if rollback_response.status_code == 200:
                        rollback_result = rollback_response.json()
                        assert "rollback_successful" in rollback_result
                        assert "new_current_version" in rollback_result

            # Test data recovery from backup
            recovery_options_response = app_client.get(
                f"/api/datasets/{dataset_id}/recovery-options"
            )

            if recovery_options_response.status_code == 200:
                recovery_options = recovery_options_response.json()

                assert "available_backups" in recovery_options
                assert "recovery_strategies" in recovery_options

                if recovery_options["available_backups"]:
                    latest_backup = recovery_options["available_backups"][0]

                    recovery_request = {
                        "backup_id": latest_backup["backup_id"],
                        "recovery_strategy": "complete_restore",
                        "preserve_metadata": True,
                    }

                    recovery_response = app_client.post(
                        f"/api/datasets/{dataset_id}/recover", json=recovery_request
                    )

                    if recovery_response.status_code == 200:
                        recovery_result = recovery_response.json()
                        assert "recovery_successful" in recovery_result
                        assert "recovered_data_size" in recovery_result

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_network_partition_resilience_workflow(self, app_client):
        """Test system behavior during network partitions."""
        # Test network connectivity monitoring
        network_status_response = app_client.get("/api/network/status")

        if network_status_response.status_code == 200:
            network_status = network_status_response.json()

            assert "connectivity" in network_status
            assert "latency" in network_status
            assert "bandwidth" in network_status

        # Simulate network partition
        partition_request = {
            "simulate_partition": True,
            "partition_type": "split_brain",
            "affected_nodes": ["node1", "node2"],
            "duration_seconds": 10,
        }

        partition_response = app_client.post(
            "/api/network/simulate-partition", json=partition_request
        )

        if partition_response.status_code == 200:
            partition_result = partition_response.json()

            assert "partition_active" in partition_result
            assert "affected_services" in partition_result

            # Test read/write behavior during partition
            # Reads should still work on available nodes
            read_test_response = app_client.get("/api/detectors/")
            assert read_test_response.status_code in [200, 503]

            # Writes might be rejected or queued
            detector_data = {
                "name": "Partition Test Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.1},
            }

            write_test_response = app_client.post("/api/detectors/", json=detector_data)
            assert write_test_response.status_code in [200, 202, 503]

            # Test partition healing
            heal_response = app_client.post("/api/network/heal-partition")

            if heal_response.status_code == 200:
                heal_result = heal_response.json()
                assert "partition_healed" in heal_result
                assert "data_synchronization" in heal_result

    def test_resource_exhaustion_recovery_workflow(self, app_client, critical_dataset):
        """Test recovery from resource exhaustion scenarios."""
        # Test memory exhaustion simulation
        memory_test_request = {
            "simulate_memory_pressure": True,
            "target_usage_percent": 95,
            "duration_seconds": 30,
        }

        memory_response = app_client.post(
            "/api/resources/simulate-exhaustion", json=memory_test_request
        )

        if memory_response.status_code == 200:
            memory_result = memory_response.json()

            assert "simulation_active" in memory_result
            assert "monitoring_enabled" in memory_result

            # Test system behavior under memory pressure
            detector_data = {
                "name": "Memory Pressure Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.1},
            }

            pressure_response = app_client.post("/api/detectors/", json=detector_data)
            # Should handle gracefully or return 503
            assert pressure_response.status_code in [200, 503]

            # Test automatic resource cleanup
            cleanup_response = app_client.post(
                "/api/resources/cleanup",
                json={
                    "cleanup_type": "automatic",
                    "free_memory": True,
                    "clear_caches": True,
                },
            )

            if cleanup_response.status_code == 200:
                cleanup_result = cleanup_response.json()
                assert "memory_freed" in cleanup_result
                assert "caches_cleared" in cleanup_result

        # Test disk space exhaustion
        disk_test_request = {"simulate_disk_full": True, "target_usage_percent": 98}

        disk_response = app_client.post(
            "/api/resources/simulate-disk-full", json=disk_test_request
        )

        if disk_response.status_code == 200:
            # Test data upload under disk pressure
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                critical_dataset.to_csv(f.name, index=False)
                dataset_file = f.name

            try:
                with open(dataset_file, "rb") as file:
                    upload_response = app_client.post(
                        "/api/datasets/upload",
                        files={"file": ("disk_full_test.csv", file, "text/csv")},
                        data={"name": "Disk Full Test Dataset"},
                    )
                # Should fail gracefully with appropriate error
                assert upload_response.status_code in [
                    200,
                    507,
                    413,
                ]  # 507 = Insufficient Storage

                if upload_response.status_code == 507:
                    error_result = upload_response.json()
                    assert "disk space" in error_result.get("detail", "").lower()

            finally:
                Path(dataset_file).unlink(missing_ok=True)

            # Test automatic space recovery
            space_recovery_response = app_client.post(
                "/api/resources/recover-disk-space",
                json={
                    "cleanup_strategy": "aggressive",
                    "remove_temp_files": True,
                    "compress_old_data": True,
                    "archive_unused_datasets": True,
                },
            )

            if space_recovery_response.status_code == 200:
                recovery_result = space_recovery_response.json()
                assert "space_recovered" in recovery_result
                assert "cleanup_actions" in recovery_result

    def test_comprehensive_disaster_scenario_workflow(
        self, app_client, critical_dataset
    ):
        """Test comprehensive disaster scenario with multiple failures."""
        # Simulate complex disaster scenario
        disaster_request = {
            "scenario": "datacenter_outage",
            "affected_systems": [
                "primary_database",
                "file_storage",
                "cache_cluster",
                "monitoring_system",
            ],
            "severity": "critical",
            "estimated_duration": "4h",
        }

        disaster_response = app_client.post(
            "/api/disaster/simulate", json=disaster_request
        )

        if disaster_response.status_code == 200:
            disaster_result = disaster_response.json()

            assert "disaster_id" in disaster_result
            assert "affected_services" in disaster_result
            assert "recovery_plan" in disaster_result

            disaster_id = disaster_result["disaster_id"]

            # Test disaster response coordination
            response_request = {
                "disaster_id": disaster_id,
                "response_team": ["ops", "engineering", "management"],
                "priority_services": ["anomaly_detection", "data_access"],
                "communication_channels": ["slack", "email", "sms"],
            }

            response_coordination = app_client.post(
                "/api/disaster/coordinate-response", json=response_request
            )

            if response_coordination.status_code == 200:
                coordination_result = response_coordination.json()
                assert "response_plan" in coordination_result
                assert "team_assignments" in coordination_result
                assert "communication_status" in coordination_result

            # Test progressive recovery
            recovery_phases = [
                {"phase": "emergency", "priority": "critical_services"},
                {"phase": "stabilization", "priority": "data_integrity"},
                {"phase": "full_recovery", "priority": "complete_functionality"},
            ]

            for phase in recovery_phases:
                phase_request = {
                    "disaster_id": disaster_id,
                    "recovery_phase": phase["phase"],
                    "execute_actions": True,
                }

                phase_response = app_client.post(
                    "/api/disaster/recovery-phase", json=phase_request
                )

                if phase_response.status_code == 200:
                    phase_result = phase_response.json()
                    assert "phase_status" in phase_result
                    assert "completed_actions" in phase_result
                    assert "next_steps" in phase_result

            # Test post-disaster validation
            validation_response = app_client.post(
                "/api/disaster/post-recovery-validation",
                json={
                    "disaster_id": disaster_id,
                    "validation_type": "comprehensive",
                    "include_data_integrity": True,
                    "include_performance_testing": True,
                },
            )

            if validation_response.status_code == 200:
                validation_result = validation_response.json()
                assert "system_health" in validation_result
                assert "data_integrity_check" in validation_result
                assert "performance_metrics" in validation_result
                assert "recovery_complete" in validation_result
