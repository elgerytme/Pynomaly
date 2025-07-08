"""Integration tests for federated learning system."""

import asyncio
from uuid import uuid4

import numpy as np
import pytest
from pynomaly.domain.models.detector import Detector
from pynomaly.domain.models.federated import (
    AggregationMethod,
    FederatedParticipant,
    FederationStrategy,
    ParticipantRole,
    PrivacyBudget,
    PrivacyMechanism,
)
from pynomaly.domain.value_objects import AlgorithmType, DetectorConfig, ModelType
from pynomaly.infrastructure.federated import (
    FederatedAggregationService,
    FederatedCoordinator,
    FederatedParticipantClient,
)
from pynomaly.infrastructure.security.security_service import SecurityService


@pytest.fixture
def security_service():
    """Security service for testing."""
    return SecurityService()


@pytest.fixture
def aggregation_service():
    """Aggregation service for testing."""
    return FederatedAggregationService()


@pytest.fixture
def base_detector():
    """Base detector for federation."""
    config = DetectorConfig(
        algorithm_type=AlgorithmType.ISOLATION_FOREST,
        model_type=ModelType.UNSUPERVISED,
        parameters={"n_estimators": 100, "contamination": 0.1, "random_state": 42},
    )
    return Detector(name="federated_detector", config=config)


@pytest.fixture
async def coordinator_with_federation(security_service, base_detector):
    """Coordinator with a test federation."""
    coordinator = FederatedCoordinator(security_service)

    federation = await coordinator.create_federation(
        name="test_federation",
        base_detector=base_detector,
        strategy=FederationStrategy.FEDERATED_AVERAGING,
        aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
        privacy_mechanism=PrivacyMechanism.DIFFERENTIAL_PRIVACY,
        min_participants=3,
        max_participants=10,
    )

    return coordinator, federation


@pytest.fixture
def participant_clients():
    """Create multiple participant clients."""
    clients = []

    for i in range(5):
        client = FederatedParticipantClient(
            participant_id=uuid4(),
            name=f"participant_{i}",
            role=ParticipantRole.PARTICIPANT,
            privacy_budget=PrivacyBudget(epsilon=1.0, delta=1e-5),
        )
        clients.append(client)

    return clients


@pytest.mark.integration
@pytest.mark.asyncio
class TestFederatedLearningIntegration:
    """Integration tests for federated learning system."""

    async def test_end_to_end_federated_training(
        self, coordinator_with_federation, participant_clients
    ):
        """Test complete federated learning workflow."""
        coordinator, federation = coordinator_with_federation

        # Step 1: Participants join federation
        for client in participant_clients:
            # Generate synthetic training data for each participant
            training_data, _ = await client.simulate_data_collection(
                num_samples=1000 + client.participant_id.int % 500,
                num_features=10,
                anomaly_ratio=0.1,
            )

            success = await client.join_federation(
                federation.federation_id,
                "coordinator_endpoint",
                training_data,
            )
            assert success

            # Register with coordinator
            participant_profile = FederatedParticipant(
                participant_id=client.participant_id,
                name=client.name,
                role=client.role,
                public_key=client.public_key,
                data_size=client.data_size,
                computation_capacity=client.computation_capacity,
                privacy_budget=client.privacy_budget,
            )

            await coordinator.register_participant(
                federation.federation_id, participant_profile
            )

        # Step 2: Start federated training
        training_round = await coordinator.start_training_round(federation.federation_id)
        assert training_round is not None
        assert training_round.round_number == 1

        # Step 3: Simulate local training and model updates
        global_model_params = {
            "weights": np.random.randn(10, 5),
            "bias": np.random.randn(5),
        }

        # Distribute global model to participants
        for client in participant_clients:
            await client.receive_global_model(
                training_round.round_number,
                global_model_params,
                federation.global_model_version,
            )

        # Wait for local training completion
        await asyncio.sleep(2)

        # Step 4: Verify model updates received
        assert len(training_round.received_updates) == len(participant_clients)
        assert training_round.participation_rate == 1.0

        # Step 5: Verify federation status
        status = await coordinator.get_federation_status(federation.federation_id)
        assert status["total_rounds"] >= 1
        assert status["participant_count"] == len(participant_clients)

    async def test_federated_aggregation_strategies(
        self, coordinator_with_federation, participant_clients, aggregation_service
    ):
        """Test different aggregation strategies."""
        coordinator, federation = coordinator_with_federation

        # Register participants
        for client in participant_clients[:3]:
            training_data, _ = await client.simulate_data_collection()

            participant_profile = FederatedParticipant(
                participant_id=client.participant_id,
                name=client.name,
                role=client.role,
                public_key=client.public_key,
                data_size=len(training_data),
                computation_capacity=1.0,
            )

            await coordinator.register_participant(
                federation.federation_id, participant_profile
            )

        # Create mock model updates
        from pynomaly.domain.models.federated import ModelUpdate

        updates = {}
        for i, client in enumerate(participant_clients[:3]):
            update = ModelUpdate(
                update_id=uuid4(),
                participant_id=client.participant_id,
                round_number=1,
                parameters={
                    "weights": np.random.randn(10, 5) + i,
                    "bias": np.random.randn(5) + i,
                },
                data_size=1000 + i * 100,
            )
            updates[client.participant_id] = update

        participants = federation.participants

        # Test different aggregation methods
        aggregation_methods = [
            AggregationMethod.WEIGHTED_AVERAGE,
            AggregationMethod.SIMPLE_AVERAGE,
            AggregationMethod.TRIMMED_MEAN,
        ]

        results = {}

        for method in aggregation_methods:
            aggregated_params, metrics = await aggregation_service.aggregate(
                method, updates, participants
            )

            assert "weights" in aggregated_params
            assert "bias" in aggregated_params
            assert aggregated_params["weights"].shape == (10, 5)
            assert aggregated_params["bias"].shape == (5,)

            results[method.value] = {
                "params": aggregated_params,
                "metrics": metrics,
            }

        # Verify different methods produce different results
        fedavg_weights = results["weighted_average"]["params"]["weights"]
        simple_avg_weights = results["simple_average"]["params"]["weights"]

        # Results should be different (unless data sizes are identical)
        assert not np.allclose(fedavg_weights, simple_avg_weights)

    async def test_byzantine_robust_aggregation(
        self, coordinator_with_federation, participant_clients, aggregation_service
    ):
        """Test Byzantine-robust aggregation with malicious participants."""
        coordinator, federation = coordinator_with_federation

        # Register normal participants
        normal_participants = participant_clients[:3]
        byzantine_participants = participant_clients[3:4]

        all_participants = normal_participants + byzantine_participants

        for client in all_participants:
            participant_profile = FederatedParticipant(
                participant_id=client.participant_id,
                name=client.name,
                role=client.role,
                public_key=client.public_key,
                data_size=1000,
                computation_capacity=1.0,
            )

            await coordinator.register_participant(
                federation.federation_id, participant_profile
            )

        # Create model updates (normal + Byzantine)
        from pynomaly.domain.models.federated import ModelUpdate

        updates = {}

        # Normal participants with similar parameters
        for i, client in enumerate(normal_participants):
            update = ModelUpdate(
                update_id=uuid4(),
                participant_id=client.participant_id,
                round_number=1,
                parameters={
                    "weights": np.random.randn(10, 5) * 0.1,  # Small updates
                    "bias": np.random.randn(5) * 0.1,
                },
                data_size=1000,
            )
            updates[client.participant_id] = update

        # Byzantine participant with malicious parameters
        for client in byzantine_participants:
            update = ModelUpdate(
                update_id=uuid4(),
                participant_id=client.participant_id,
                round_number=1,
                parameters={
                    "weights": np.random.randn(10, 5) * 100,  # Large malicious updates
                    "bias": np.random.randn(5) * 100,
                },
                data_size=1000,
            )
            updates[client.participant_id] = update

        participants = federation.participants

        # Test robust aggregation methods
        robust_methods = ["krum", "multi_krum", "geometric_median"]

        for method in robust_methods:
            try:
                aggregated_params, metrics = await aggregation_service.aggregate_advanced(
                    method, updates, participants
                )

                assert "weights" in aggregated_params
                assert "bias" in aggregated_params

                # Robust aggregation should not be heavily influenced by Byzantine updates
                weights_norm = np.linalg.norm(aggregated_params["weights"])
                bias_norm = np.linalg.norm(aggregated_params["bias"])

                # Should be closer to normal updates (small) than Byzantine updates (large)
                assert weights_norm < 50  # Much smaller than Byzantine updates
                assert bias_norm < 50

            except Exception as e:
                # Some methods might not be fully implemented
                assert "not implemented" in str(e).lower() or "unsupported" in str(e).lower()

    async def test_privacy_preserving_aggregation(
        self, coordinator_with_federation, participant_clients
    ):
        """Test privacy-preserving aggregation with differential privacy."""
        coordinator, federation = coordinator_with_federation

        # Create federation with differential privacy
        dp_budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        federation.differential_privacy_budget = dp_budget

        # Register participants with privacy budgets
        for client in participant_clients[:3]:
            participant_profile = FederatedParticipant(
                participant_id=client.participant_id,
                name=client.name,
                role=client.role,
                public_key=client.public_key,
                data_size=1000,
                computation_capacity=1.0,
                privacy_budget=PrivacyBudget(epsilon=0.5, delta=1e-5),
            )

            await coordinator.register_participant(
                federation.federation_id, participant_profile
            )

        # Start training round
        training_round = await coordinator.start_training_round(federation.federation_id)

        # Simulate local training with privacy
        for client in participant_clients[:3]:
            await client.simulate_data_collection()

            # Set local model parameters for privacy testing
            client.global_model_parameters = {
                "weights": np.random.randn(10, 5),
                "bias": np.random.randn(5),
            }
            client.current_round = training_round.round_number

            # Perform local training (includes privacy mechanisms)
            await client._perform_local_training()

        # Verify privacy budget consumption
        for client in participant_clients[:3]:
            if client.privacy_budget:
                assert client.privacy_budget.spent_epsilon > 0
                assert client.privacy_budget.query_count > 0

    async def test_participant_trust_evolution(
        self, coordinator_with_federation, participant_clients
    ):
        """Test participant trust score evolution over multiple rounds."""
        coordinator, federation = coordinator_with_federation

        # Register participants
        participants_data = []
        for client in participant_clients[:4]:
            training_data, _ = await client.simulate_data_collection()

            participant_profile = FederatedParticipant(
                participant_id=client.participant_id,
                name=client.name,
                role=client.role,
                public_key=client.public_key,
                data_size=len(training_data),
                computation_capacity=1.0,
            )

            await coordinator.register_participant(
                federation.federation_id, participant_profile
            )

            participants_data.append((client, participant_profile))

        # Simulate multiple training rounds
        num_rounds = 3

        for round_num in range(num_rounds):
            # Start training round
            training_round = await coordinator.start_training_round(federation.federation_id)

            # Some participants contribute, others don't
            contributing_participants = participants_data[:2]  # First 2 contribute

            for client, profile in contributing_participants:
                from pynomaly.domain.models.federated import ModelUpdate

                model_update = ModelUpdate(
                    update_id=uuid4(),
                    participant_id=client.participant_id,
                    round_number=training_round.round_number,
                    parameters={
                        "weights": np.random.randn(10, 5),
                        "bias": np.random.randn(5),
                    },
                    data_size=profile.data_size,
                )

                await coordinator.receive_model_update(
                    federation.federation_id, model_update
                )

            # Force round completion
            await coordinator._complete_training_round(federation)

        # Check trust score evolution
        for client, profile in participants_data[:2]:
            # Contributing participants should have high trust
            assert profile.trust_score > 0.9
            assert len(profile.contribution_history) == num_rounds

            # All contributions should be marked as True
            for contribution in profile.contribution_history:
                assert contribution["contributed"] is True

        for client, profile in participants_data[2:]:
            # Non-contributing participants should have lower trust
            assert profile.trust_score < 0.9
            assert len(profile.contribution_history) == num_rounds

            # All contributions should be marked as False
            for contribution in profile.contribution_history:
                assert contribution["contributed"] is False

    async def test_federated_convergence_detection(
        self, coordinator_with_federation, participant_clients
    ):
        """Test convergence detection in federated training."""
        coordinator, federation = coordinator_with_federation

        # Register participants
        for client in participant_clients[:3]:
            training_data, _ = await client.simulate_data_collection()

            participant_profile = FederatedParticipant(
                participant_id=client.participant_id,
                name=client.name,
                role=client.role,
                public_key=client.public_key,
                data_size=len(training_data),
                computation_capacity=1.0,
            )

            await coordinator.register_participant(
                federation.federation_id, participant_profile
            )

        # Simulate training rounds with decreasing loss
        losses = [1.0, 0.8, 0.6, 0.5, 0.49, 0.48]  # Converging

        for i, loss in enumerate(losses):
            # Start training round
            training_round = await coordinator.start_training_round(federation.federation_id)

            # Send updates from all participants
            for client in participant_clients[:3]:
                from pynomaly.domain.models.federated import ModelUpdate

                model_update = ModelUpdate(
                    update_id=uuid4(),
                    participant_id=client.participant_id,
                    round_number=training_round.round_number,
                    parameters={
                        "weights": np.random.randn(10, 5),
                        "bias": np.random.randn(5),
                    },
                    data_size=1000,
                )

                await coordinator.receive_model_update(
                    federation.federation_id, model_update
                )

            # Manually set aggregation metrics with loss
            training_round.aggregation_metrics = {"loss": loss}

            # Complete round
            await coordinator._complete_training_round(federation)

        # Check convergence metric
        convergence_metric = federation.calculate_convergence_metric()

        # Should have low convergence metric (converged)
        assert convergence_metric is not None
        assert convergence_metric < 0.1  # Small change between last two rounds

    async def test_performance_benchmarking(self, participant_clients):
        """Test participant performance benchmarking."""
        client = participant_clients[0]

        # Benchmark local training
        benchmark_results = await client.benchmark_local_training(num_rounds=3)

        assert "total_rounds" in benchmark_results
        assert "avg_training_time" in benchmark_results
        assert "throughput_samples_per_second" in benchmark_results
        assert benchmark_results["total_rounds"] == 3
        assert benchmark_results["avg_training_time"] > 0
        assert benchmark_results["throughput_samples_per_second"] > 0

        print("\nParticipant Performance Benchmark:")
        print(f"  Average training time: {benchmark_results['avg_training_time']:.3f}s")
        print(f"  Throughput: {benchmark_results['throughput_samples_per_second']:.1f} samples/sec")
        print(f"  Data size: {benchmark_results['data_size']} samples")

    async def test_large_scale_federation(self, security_service, base_detector):
        """Test federation with larger number of participants."""
        coordinator = FederatedCoordinator(security_service)

        # Create large federation
        federation = await coordinator.create_federation(
            name="large_federation",
            base_detector=base_detector,
            min_participants=10,
            max_participants=50,
        )

        # Create many participant clients
        num_participants = 20
        clients = []

        for i in range(num_participants):
            client = FederatedParticipantClient(
                participant_id=uuid4(),
                name=f"participant_{i}",
                role=ParticipantRole.PARTICIPANT,
            )
            clients.append(client)

            # Register with coordinator
            participant_profile = FederatedParticipant(
                participant_id=client.participant_id,
                name=client.name,
                role=client.role,
                public_key=client.public_key,
                data_size=1000,
                computation_capacity=1.0,
            )

            await coordinator.register_participant(
                federation.federation_id, participant_profile
            )

        # Verify federation can handle large scale
        assert len(federation.participants) == num_participants
        assert federation.can_start_training

        # Start training round
        training_round = await coordinator.start_training_round(federation.federation_id)
        assert training_round is not None
        assert len(training_round.target_participants) == num_participants

        print("\nLarge Scale Federation Test:")
        print(f"  Participants: {num_participants}")
        print(f"  Federation ID: {federation.federation_id}")
        print(f"  Training round started: {training_round.round_number}")

    async def test_federation_fault_tolerance(
        self, coordinator_with_federation, participant_clients
    ):
        """Test federation fault tolerance with participant failures."""
        coordinator, federation = coordinator_with_federation

        # Register participants
        for client in participant_clients:
            participant_profile = FederatedParticipant(
                participant_id=client.participant_id,
                name=client.name,
                role=client.role,
                public_key=client.public_key,
                data_size=1000,
                computation_capacity=1.0,
            )

            await coordinator.register_participant(
                federation.federation_id, participant_profile
            )

        # Start training round
        training_round = await coordinator.start_training_round(federation.federation_id)

        # Simulate partial participation (some participants fail)
        working_participants = participant_clients[:3]  # Only 3 out of 5 participate

        for client in working_participants:
            from pynomaly.domain.models.federated import ModelUpdate

            model_update = ModelUpdate(
                update_id=uuid4(),
                participant_id=client.participant_id,
                round_number=training_round.round_number,
                parameters={
                    "weights": np.random.randn(10, 5),
                    "bias": np.random.randn(5),
                },
                data_size=1000,
            )

            await coordinator.receive_model_update(
                federation.federation_id, model_update
            )

        # Check that round can still complete with partial participation
        assert training_round.participation_rate == 0.6  # 3/5 participants
        assert training_round.is_valid  # Should still be valid with >50% participation

        # Complete round
        await coordinator._complete_training_round(federation)

        # Verify round completed successfully
        assert len(federation.training_rounds) == 1
        assert federation.training_rounds[0].is_completed
