"""Unit tests for federated learning coordinator."""

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from monorepo.domain.models.detector import Detector
from monorepo.domain.models.federated import (
    AggregationMethod,
    FederatedParticipant,
    FederationStrategy,
    ModelUpdate,
    ParticipantRole,
    PrivacyBudget,
    PrivacyMechanism,
)
from monorepo.domain.value_objects import AlgorithmType, DetectorConfig, ModelType
from monorepo.infrastructure.federated.coordinator import FederatedCoordinator
from monorepo.infrastructure.security.security_service import SecurityService


@pytest.fixture
def security_service():
    """Create security service for testing."""
    return SecurityService()


@pytest.fixture
def coordinator(security_service):
    """Create federated coordinator for testing."""
    return FederatedCoordinator(security_service)


@pytest.fixture
def base_detector():
    """Create base detector for federation."""
    config = DetectorConfig(
        algorithm_type=AlgorithmType.ISOLATION_FOREST,
        model_type=ModelType.UNSUPERVISED,
        parameters={"n_estimators": 100, "contamination": 0.1},
    )
    return Detector(name="test_detector", config=config)


@pytest.fixture
def test_participants():
    """Create test participants."""
    participants = []
    for i in range(5):
        participant = FederatedParticipant(
            participant_id=uuid4(),
            name=f"participant_{i}",
            role=ParticipantRole.PARTICIPANT,
            public_key=f"key_{i}",
            trust_score=0.9,
            data_size=1000 + i * 100,
            computation_capacity=1.0,
            privacy_budget=PrivacyBudget(epsilon=1.0, delta=1e-5),
        )
        participants.append(participant)
    return participants


class TestFederatedCoordinator:
    """Test federated learning coordinator functionality."""

    async def test_create_federation(self, coordinator, base_detector):
        """Test federation creation."""
        federation = await coordinator.create_federation(
            name="test_federation",
            base_detector=base_detector,
            strategy=FederationStrategy.FEDERATED_AVERAGING,
            min_participants=3,
            max_participants=10,
        )

        assert federation is not None
        assert federation.name == "test_federation"
        assert federation.strategy == FederationStrategy.FEDERATED_AVERAGING
        assert federation.min_participants == 3
        assert federation.max_participants == 10
        assert len(federation.participants) == 0
        assert not federation.is_active

    async def test_register_participants(
        self, coordinator, base_detector, test_participants
    ):
        """Test participant registration."""
        federation = await coordinator.create_federation(
            name="test_federation",
            base_detector=base_detector,
        )

        # Register participants
        for participant in test_participants:
            success = await coordinator.register_participant(
                federation.federation_id, participant
            )
            assert success

        assert len(federation.participants) == len(test_participants)
        assert len(federation.active_participants) == len(test_participants)

    async def test_start_training_round(
        self, coordinator, base_detector, test_participants
    ):
        """Test starting a training round."""
        federation = await coordinator.create_federation(
            name="test_federation",
            base_detector=base_detector,
            min_participants=3,
        )

        # Register enough participants
        for participant in test_participants[:4]:
            await coordinator.register_participant(
                federation.federation_id, participant
            )

        # Start training round
        training_round = await coordinator.start_training_round(
            federation.federation_id
        )

        assert training_round is not None
        assert training_round.round_number == 1
        assert len(training_round.target_participants) == 4
        assert federation.is_active

    async def test_cannot_start_training_insufficient_participants(
        self, coordinator, base_detector, test_participants
    ):
        """Test that training cannot start with insufficient participants."""
        federation = await coordinator.create_federation(
            name="test_federation",
            base_detector=base_detector,
            min_participants=5,
        )

        # Register only 2 participants (insufficient)
        for participant in test_participants[:2]:
            await coordinator.register_participant(
                federation.federation_id, participant
            )

        # Attempt to start training round
        training_round = await coordinator.start_training_round(
            federation.federation_id
        )

        assert training_round is None
        assert not federation.is_active

    async def test_receive_model_update(
        self, coordinator, base_detector, test_participants
    ):
        """Test receiving model updates from participants."""
        federation = await coordinator.create_federation(
            name="test_federation",
            base_detector=base_detector,
            min_participants=3,
        )

        # Register participants and start training
        for participant in test_participants[:4]:
            await coordinator.register_participant(
                federation.federation_id, participant
            )

        training_round = await coordinator.start_training_round(
            federation.federation_id
        )

        # Create model update
        model_update = ModelUpdate(
            update_id=uuid4(),
            participant_id=test_participants[0].participant_id,
            round_number=training_round.round_number,
            parameters={
                "weights": np.random.randn(10, 5),
                "bias": np.random.randn(5),
            },
            data_size=1000,
        )

        # Send model update
        success = await coordinator.receive_model_update(
            federation.federation_id, model_update
        )

        assert success
        assert len(training_round.received_updates) == 1
        assert test_participants[0].participant_id in training_round.received_updates

    async def test_complete_training_round(
        self, coordinator, base_detector, test_participants
    ):
        """Test completing a training round with aggregation."""
        federation = await coordinator.create_federation(
            name="test_federation",
            base_detector=base_detector,
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            min_participants=3,
        )

        # Register participants and start training
        for participant in test_participants[:3]:
            await coordinator.register_participant(
                federation.federation_id, participant
            )

        training_round = await coordinator.start_training_round(
            federation.federation_id
        )

        # Send model updates from all participants
        for i, participant in enumerate(test_participants[:3]):
            model_update = ModelUpdate(
                update_id=uuid4(),
                participant_id=participant.participant_id,
                round_number=training_round.round_number,
                parameters={
                    "weights": np.random.randn(10, 5) + i,  # Different values
                    "bias": np.random.randn(5) + i,
                },
                data_size=participant.data_size,
            )

            await coordinator.receive_model_update(
                federation.federation_id, model_update
            )

        # Wait for round completion
        await asyncio.sleep(0.1)

        # Check that round completed
        assert not federation.is_active
        assert len(federation.training_rounds) == 1
        assert federation.training_rounds[0].is_completed

    async def test_participant_trust_score_updates(
        self, coordinator, base_detector, test_participants
    ):
        """Test that participant trust scores are updated."""
        federation = await coordinator.create_federation(
            name="test_federation",
            base_detector=base_detector,
            min_participants=3,
        )

        # Register participants
        for participant in test_participants[:4]:
            await coordinator.register_participant(
                federation.federation_id, participant
            )

        # Store initial trust scores
        initial_trust_scores = {
            p.participant_id: p.trust_score for p in test_participants[:4]
        }

        # Start training round
        training_round = await coordinator.start_training_round(
            federation.federation_id
        )

        # Only some participants send updates
        for i, participant in enumerate(test_participants[:2]):
            model_update = ModelUpdate(
                update_id=uuid4(),
                participant_id=participant.participant_id,
                round_number=training_round.round_number,
                parameters={
                    "weights": np.random.randn(10, 5),
                    "bias": np.random.randn(5),
                },
                data_size=participant.data_size,
            )

            await coordinator.receive_model_update(
                federation.federation_id, model_update
            )

        # Force round completion by completing manually
        await coordinator._complete_training_round(federation)

        # Check trust score updates
        for participant in test_participants[:2]:
            # Contributing participants should have higher trust
            assert (
                participant.trust_score
                >= initial_trust_scores[participant.participant_id]
            )

        for participant in test_participants[2:4]:
            # Non-contributing participants should have lower trust
            assert (
                participant.trust_score
                < initial_trust_scores[participant.participant_id]
            )

    async def test_federation_status(
        self, coordinator, base_detector, test_participants
    ):
        """Test getting federation status."""
        federation = await coordinator.create_federation(
            name="test_federation",
            base_detector=base_detector,
        )

        # Register participants
        for participant in test_participants[:3]:
            await coordinator.register_participant(
                federation.federation_id, participant
            )

        # Get status
        status = await coordinator.get_federation_status(federation.federation_id)

        assert status["federation_id"] == str(federation.federation_id)
        assert status["name"] == "test_federation"
        assert status["participant_count"] == 3
        assert status["active_participant_count"] == 3
        assert status["total_rounds"] == 0
        assert not status["is_active"]

    async def test_privacy_budget_consumption(self, coordinator, base_detector):
        """Test privacy budget consumption during aggregation."""
        privacy_budget = PrivacyBudget(epsilon=1.0, delta=1e-5)

        federation = await coordinator.create_federation(
            name="test_federation",
            base_detector=base_detector,
            privacy_mechanism=PrivacyMechanism.DIFFERENTIAL_PRIVACY,
            differential_privacy_budget=privacy_budget,
        )

        # Create participant with privacy budget
        participant = FederatedParticipant(
            participant_id=uuid4(),
            name="test_participant",
            role=ParticipantRole.PARTICIPANT,
            public_key="test_key",
            data_size=1000,
            privacy_budget=PrivacyBudget(epsilon=0.5, delta=1e-5),
        )

        await coordinator.register_participant(federation.federation_id, participant)

        # Test privacy budget validation
        assert privacy_budget.can_answer_query(0.1)
        assert privacy_budget.spent_epsilon == 0.0

        # Consume some budget
        privacy_budget.consume_budget(0.2)
        assert privacy_budget.spent_epsilon == 0.2
        assert privacy_budget.can_answer_query(0.5)
        assert not privacy_budget.can_answer_query(1.0)

    async def test_byzantine_tolerance(
        self, coordinator, base_detector, test_participants
    ):
        """Test federation with Byzantine fault tolerance."""
        federation = await coordinator.create_federation(
            name="test_federation",
            base_detector=base_detector,
            aggregation_method=AggregationMethod.BYZANTINE_RESILIENT,
            byzantine_tolerance=0.2,  # 20% Byzantine nodes
        )

        # Register participants
        for participant in test_participants:
            await coordinator.register_participant(
                federation.federation_id, participant
            )

        assert federation.byzantine_tolerance == 0.2

        # Calculate maximum Byzantine participants
        max_byzantine = int(len(test_participants) * federation.byzantine_tolerance)
        assert max_byzantine >= 0

    async def test_multiple_federations(
        self, coordinator, base_detector, test_participants
    ):
        """Test managing multiple federations."""
        # Create first federation
        federation1 = await coordinator.create_federation(
            name="federation_1",
            base_detector=base_detector,
        )

        # Create second federation
        federation2 = await coordinator.create_federation(
            name="federation_2",
            base_detector=base_detector,
        )

        assert len(coordinator.federations) == 2
        assert federation1.federation_id != federation2.federation_id

        # Register participants in different federations
        await coordinator.register_participant(
            federation1.federation_id, test_participants[0]
        )
        await coordinator.register_participant(
            federation2.federation_id, test_participants[1]
        )

        assert len(federation1.participants) == 1
        assert len(federation2.participants) == 1

    async def test_coordinator_shutdown(
        self, coordinator, base_detector, test_participants
    ):
        """Test coordinator shutdown."""
        # Create federation and register participants
        federation = await coordinator.create_federation(
            name="test_federation",
            base_detector=base_detector,
        )

        for participant in test_participants[:2]:
            await coordinator.register_participant(
                federation.federation_id, participant
            )

        assert len(coordinator.federations) == 1
        assert len(coordinator.active_connections) > 0

        # Shutdown coordinator
        await coordinator.shutdown()

        assert len(coordinator.federations) == 0
        assert len(coordinator.active_connections) == 0

    async def test_invalid_federation_operations(self, coordinator):
        """Test error handling for invalid operations."""
        # Test operations on non-existent federation
        invalid_federation_id = uuid4()

        with pytest.raises(ValueError, match="Federation .* not found"):
            await coordinator.start_training_round(invalid_federation_id)

        with pytest.raises(ValueError, match="Federation .* not found"):
            await coordinator.get_federation_status(invalid_federation_id)

        # Test registering participant in non-existent federation
        participant = FederatedParticipant(
            participant_id=uuid4(),
            name="test",
            role=ParticipantRole.PARTICIPANT,
            public_key="key",
            data_size=100,
        )

        with pytest.raises(ValueError, match="Federation .* not found"):
            await coordinator.register_participant(invalid_federation_id, participant)
