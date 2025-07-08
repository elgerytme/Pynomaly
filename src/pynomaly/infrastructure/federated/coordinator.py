"""Federated learning coordinator for privacy-preserving distributed training."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.models.federated import (
    AggregationMethod,
    FederatedDetector,
    FederatedParticipant,
    FederatedRound,
    FederationStrategy,
    ModelUpdate,
    PrivacyBudget,
    PrivacyMechanism,
)
from pynomaly.infrastructure.security.security_service import SecurityService


class FederatedCoordinator:
    """Central coordinator for federated learning operations."""

    def __init__(
        self,
        security_service: SecurityService,
        coordinator_id: UUID | None = None,
    ):
        """Initialize federated coordinator."""
        self.coordinator_id = coordinator_id or uuid4()
        self.security_service = security_service
        self.logger = logging.getLogger(__name__)

        # Active federations managed by this coordinator
        self.federations: dict[UUID, FederatedDetector] = {}

        # Network state
        self.active_connections: dict[UUID, dict[str, Any]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()

        # Aggregation strategies
        self.aggregation_strategies = {
            AggregationMethod.WEIGHTED_AVERAGE: self._weighted_average_aggregation,
            AggregationMethod.SIMPLE_AVERAGE: self._simple_average_aggregation,
            AggregationMethod.MEDIAN: self._median_aggregation,
            AggregationMethod.TRIMMED_MEAN: self._trimmed_mean_aggregation,
            AggregationMethod.BYZANTINE_RESILIENT: self._byzantine_resilient_aggregation,
        }

        self.logger.info(f"Federated coordinator {self.coordinator_id} initialized")

    async def create_federation(
        self,
        name: str,
        base_detector: Any,  # Detector instance
        strategy: FederationStrategy = FederationStrategy.FEDERATED_AVERAGING,
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
        privacy_mechanism: PrivacyMechanism = PrivacyMechanism.DIFFERENTIAL_PRIVACY,
        min_participants: int = 3,
        max_participants: int = 100,
        differential_privacy_budget: PrivacyBudget | None = None,
    ) -> FederatedDetector:
        """Create new federated learning network."""

        federation_id = uuid4()

        federated_detector = FederatedDetector(
            federation_id=federation_id,
            name=name,
            base_detector=base_detector,
            strategy=strategy,
            aggregation_method=aggregation_method,
            privacy_mechanism=privacy_mechanism,
            min_participants=min_participants,
            max_participants=max_participants,
            differential_privacy_budget=differential_privacy_budget,
            coordinator_public_key=f"coordinator_{self.coordinator_id}",
        )

        self.federations[federation_id] = federated_detector

        self.logger.info(
            f"Created federation {federation_id} with strategy {strategy.value}"
        )

        return federated_detector

    async def register_participant(
        self,
        federation_id: UUID,
        participant: FederatedParticipant,
    ) -> bool:
        """Register participant in federation."""

        if federation_id not in self.federations:
            raise ValueError(f"Federation {federation_id} not found")

        federation = self.federations[federation_id]

        try:
            # Validate participant credentials and capacity
            if not await self._validate_participant(participant):
                self.logger.warning(
                    f"Participant validation failed: {participant.participant_id}"
                )
                return False

            # Add to federation
            federation.add_participant(participant)

            # Establish secure connection
            await self._establish_secure_connection(participant)

            self.logger.info(
                f"Registered participant {participant.participant_id} "
                f"in federation {federation_id}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to register participant: {e}")
            return False

    async def start_training_round(self, federation_id: UUID) -> FederatedRound | None:
        """Start new federated training round."""

        if federation_id not in self.federations:
            raise ValueError(f"Federation {federation_id} not found")

        federation = self.federations[federation_id]

        if not federation.can_start_training:
            self.logger.warning(
                f"Federation {federation_id} cannot start training: "
                f"insufficient participants or already active"
            )
            return None

        try:
            # Start new round
            training_round = federation.start_training_round()

            # Distribute global model to participants
            await self._distribute_global_model(federation, training_round)

            # Start round timeout monitoring
            asyncio.create_task(
                self._monitor_round_timeout(federation_id, training_round)
            )

            self.logger.info(
                f"Started training round {training_round.round_number} "
                f"for federation {federation_id}"
            )

            return training_round

        except Exception as e:
            self.logger.error(f"Failed to start training round: {e}")
            return None

    async def receive_model_update(
        self,
        federation_id: UUID,
        model_update: ModelUpdate,
    ) -> bool:
        """Receive and process model update from participant."""

        if federation_id not in self.federations:
            self.logger.error(f"Federation {federation_id} not found")
            return False

        federation = self.federations[federation_id]

        if not federation.current_round:
            self.logger.error("No active training round")
            return False

        try:
            # Validate update
            if not await self._validate_model_update(model_update, federation):
                return False

            # Apply privacy mechanisms
            processed_update = await self._apply_privacy_mechanisms(
                model_update, federation
            )

            # Add to current round
            federation.current_round.add_update(processed_update)

            self.logger.info(
                f"Received model update from participant "
                f"{model_update.participant_id} for round "
                f"{model_update.round_number}"
            )

            # Check if round is complete
            if self._is_round_complete(federation.current_round):
                await self._complete_training_round(federation)

            return True

        except Exception as e:
            self.logger.error(f"Failed to process model update: {e}")
            return False

    async def _complete_training_round(self, federation: FederatedDetector) -> None:
        """Complete current training round with aggregation."""

        if not federation.current_round:
            return

        current_round = federation.current_round

        try:
            # Aggregate model updates
            aggregated_params = await self._aggregate_model_updates(
                federation, current_round.received_updates
            )

            # Apply post-aggregation privacy mechanisms
            if federation.privacy_mechanism == PrivacyMechanism.DIFFERENTIAL_PRIVACY:
                aggregated_params = self._apply_differential_privacy(
                    aggregated_params, federation.differential_privacy_budget
                )

            # Complete the round
            federation.complete_current_round(aggregated_params)

            # Update participant trust scores
            await self._update_participant_trust_scores(federation, current_round)

            # Check for convergence
            convergence_metric = federation.calculate_convergence_metric()

            self.logger.info(
                f"Completed training round {current_round.round_number} "
                f"for federation {federation.federation_id}. "
                f"Convergence metric: {convergence_metric}"
            )

            # Auto-start next round if not converged
            if (
                convergence_metric is None
                or convergence_metric > federation.convergence_threshold
            ) and len(federation.training_rounds) < federation.max_rounds:
                # Wait a bit before starting next round
                await asyncio.sleep(5)
                await self.start_training_round(federation.federation_id)

        except Exception as e:
            self.logger.error(f"Failed to complete training round: {e}")

    async def _aggregate_model_updates(
        self,
        federation: FederatedDetector,
        updates: dict[UUID, ModelUpdate],
    ) -> dict[str, np.ndarray]:
        """Aggregate model updates using specified strategy."""

        if not updates:
            raise ValueError("No model updates to aggregate")

        aggregation_method = federation.aggregation_method
        aggregation_func = self.aggregation_strategies[aggregation_method]

        # Get participant weights
        participant_weights = federation.get_participant_weights()

        # Filter weights for participants with updates
        filtered_weights = {
            pid: weight for pid, weight in participant_weights.items() if pid in updates
        }

        # Normalize weights
        total_weight = sum(filtered_weights.values())
        if total_weight > 0:
            filtered_weights = {
                pid: weight / total_weight for pid, weight in filtered_weights.items()
            }

        return await aggregation_func(updates, filtered_weights)

    async def _weighted_average_aggregation(
        self,
        updates: dict[UUID, ModelUpdate],
        weights: dict[UUID, float],
    ) -> dict[str, np.ndarray]:
        """Weighted average aggregation of model parameters."""

        if not updates:
            return {}

        # Get parameter names from first update
        first_update = next(iter(updates.values()))
        param_names = list(first_update.parameters.keys())

        aggregated_params = {}

        for param_name in param_names:
            weighted_sum = None

            for participant_id, update in updates.items():
                if param_name in update.parameters:
                    param_value = update.parameters[param_name]
                    weight = weights.get(participant_id, 0.0)

                    weighted_param = param_value * weight

                    if weighted_sum is None:
                        weighted_sum = weighted_param
                    else:
                        weighted_sum += weighted_param

            if weighted_sum is not None:
                aggregated_params[param_name] = weighted_sum

        return aggregated_params

    async def _simple_average_aggregation(
        self,
        updates: dict[UUID, ModelUpdate],
        weights: dict[UUID, float],
    ) -> dict[str, np.ndarray]:
        """Simple average aggregation (ignoring weights)."""

        if not updates:
            return {}

        first_update = next(iter(updates.values()))
        param_names = list(first_update.parameters.keys())

        aggregated_params = {}

        for param_name in param_names:
            param_values = []

            for update in updates.values():
                if param_name in update.parameters:
                    param_values.append(update.parameters[param_name])

            if param_values:
                aggregated_params[param_name] = np.mean(param_values, axis=0)

        return aggregated_params

    async def _median_aggregation(
        self,
        updates: dict[UUID, ModelUpdate],
        weights: dict[UUID, float],
    ) -> dict[str, np.ndarray]:
        """Median aggregation for Byzantine resilience."""

        if not updates:
            return {}

        first_update = next(iter(updates.values()))
        param_names = list(first_update.parameters.keys())

        aggregated_params = {}

        for param_name in param_names:
            param_values = []

            for update in updates.values():
                if param_name in update.parameters:
                    param_values.append(update.parameters[param_name])

            if param_values:
                aggregated_params[param_name] = np.median(param_values, axis=0)

        return aggregated_params

    async def _trimmed_mean_aggregation(
        self,
        updates: dict[UUID, ModelUpdate],
        weights: dict[UUID, float],
        trim_ratio: float = 0.2,
    ) -> dict[str, np.ndarray]:
        """Trimmed mean aggregation for robustness."""

        if not updates:
            return {}

        first_update = next(iter(updates.values()))
        param_names = list(first_update.parameters.keys())

        aggregated_params = {}

        for param_name in param_names:
            param_values = []

            for update in updates.values():
                if param_name in update.parameters:
                    param_values.append(update.parameters[param_name])

            if param_values:
                # Sort values and trim extremes
                sorted_values = np.sort(param_values, axis=0)
                n = len(sorted_values)
                trim_count = int(n * trim_ratio)

                if trim_count > 0:
                    trimmed_values = sorted_values[trim_count:-trim_count]
                else:
                    trimmed_values = sorted_values

                aggregated_params[param_name] = np.mean(trimmed_values, axis=0)

        return aggregated_params

    async def _byzantine_resilient_aggregation(
        self,
        updates: dict[UUID, ModelUpdate],
        weights: dict[UUID, float],
    ) -> dict[str, np.ndarray]:
        """Byzantine-resilient aggregation using coordinate-wise median."""

        if not updates:
            return {}

        # For simplicity, use coordinate-wise median
        # In practice, more sophisticated methods like Krum could be used
        return await self._median_aggregation(updates, weights)

    def _apply_differential_privacy(
        self,
        parameters: dict[str, np.ndarray],
        privacy_budget: PrivacyBudget | None,
    ) -> dict[str, np.ndarray]:
        """Apply differential privacy to aggregated parameters."""

        if not privacy_budget:
            return parameters

        # Simple Gaussian noise mechanism
        epsilon_per_param = privacy_budget.epsilon / len(parameters)

        if not privacy_budget.can_answer_query(epsilon_per_param):
            self.logger.warning("Insufficient privacy budget for current query")
            return parameters

        noisy_parameters = {}

        for param_name, param_value in parameters.items():
            # Calculate sensitivity (simplified)
            sensitivity = np.std(param_value)

            # Calculate noise scale
            noise_scale = sensitivity / epsilon_per_param

            # Add Gaussian noise
            noise = np.random.normal(0, noise_scale, param_value.shape)
            noisy_parameters[param_name] = param_value + noise

        # Consume privacy budget
        privacy_budget.consume_budget(epsilon_per_param)

        return noisy_parameters

    async def _validate_participant(self, participant: FederatedParticipant) -> bool:
        """Validate participant credentials and capacity."""

        try:
            # Basic validation
            if not participant.public_key:
                return False

            if participant.trust_score < 0.5:
                return False

            if participant.computation_capacity < 0.1:
                return False

            # Additional security checks could be added here
            return True

        except Exception as e:
            self.logger.error(f"Participant validation error: {e}")
            return False

    async def _validate_model_update(
        self,
        model_update: ModelUpdate,
        federation: FederatedDetector,
    ) -> bool:
        """Validate model update from participant."""

        try:
            # Check integrity
            if not model_update.verify_integrity():
                self.logger.warning("Model update integrity check failed")
                return False

            # Check if participant is in federation
            if model_update.participant_id not in federation.participants:
                self.logger.warning("Update from unknown participant")
                return False

            # Check if participant is active
            participant = federation.participants[model_update.participant_id]
            if not participant.is_active:
                self.logger.warning("Update from inactive participant")
                return False

            # Check round number
            if (
                federation.current_round
                and model_update.round_number != federation.current_round.round_number
            ):
                self.logger.warning("Update for wrong round number")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Model update validation error: {e}")
            return False

    async def _apply_privacy_mechanisms(
        self,
        model_update: ModelUpdate,
        federation: FederatedDetector,
    ) -> ModelUpdate:
        """Apply privacy-preserving mechanisms to model update."""

        # For now, return update as-is
        # In practice, this would apply secure aggregation, etc.
        return model_update

    async def _establish_secure_connection(
        self,
        participant: FederatedParticipant,
    ) -> None:
        """Establish secure connection with participant."""

        # Placeholder for secure connection establishment
        self.active_connections[participant.participant_id] = {
            "established_at": datetime.utcnow(),
            "public_key": participant.public_key,
            "last_ping": datetime.utcnow(),
        }

    async def _distribute_global_model(
        self,
        federation: FederatedDetector,
        training_round: FederatedRound,
    ) -> None:
        """Distribute global model to participants for training."""

        # Placeholder for model distribution
        # In practice, this would send the current global model
        # parameters to all active participants

        for participant_id in training_round.target_participants:
            participant = federation.participants[participant_id]

            self.logger.debug(
                f"Distributing global model to participant {participant_id}"
            )

    async def _monitor_round_timeout(
        self,
        federation_id: UUID,
        training_round: FederatedRound,
    ) -> None:
        """Monitor training round timeout."""

        if not training_round.deadline:
            return

        # Wait until deadline
        timeout_seconds = (training_round.deadline - datetime.utcnow()).total_seconds()

        if timeout_seconds > 0:
            await asyncio.sleep(timeout_seconds)

        # Check if round is still active
        federation = self.federations.get(federation_id)
        if (
            federation
            and federation.current_round
            and federation.current_round.round_id == training_round.round_id
            and not federation.current_round.is_completed
        ):
            self.logger.warning(
                f"Training round {training_round.round_number} timed out "
                f"for federation {federation_id}"
            )

            # Complete round with available updates
            if federation.current_round.received_updates:
                await self._complete_training_round(federation)
            else:
                # No updates received, mark round as failed
                federation.current_round = None

    def _is_round_complete(self, training_round: FederatedRound) -> bool:
        """Check if training round has sufficient participation."""

        return (
            training_round.participation_rate >= 0.7  # 70% participation
            or len(training_round.received_updates)
            >= training_round.federation.min_participants
        )

    async def _update_participant_trust_scores(
        self,
        federation: FederatedDetector,
        completed_round: FederatedRound,
    ) -> None:
        """Update participant trust scores based on participation."""

        for participant_id, participant in federation.participants.items():
            if participant_id in completed_round.received_updates:
                # Increase trust for participants who contributed
                participant.trust_score = min(1.0, participant.trust_score + 0.01)

                # Update contribution history
                participant.contribution_history.append(
                    {
                        "round_number": completed_round.round_number,
                        "contributed": True,
                        "timestamp": datetime.utcnow(),
                    }
                )
            else:
                # Decrease trust for participants who didn't contribute
                participant.trust_score = max(0.0, participant.trust_score - 0.05)

                participant.contribution_history.append(
                    {
                        "round_number": completed_round.round_number,
                        "contributed": False,
                        "timestamp": datetime.utcnow(),
                    }
                )

    async def get_federation_status(self, federation_id: UUID) -> dict[str, Any]:
        """Get comprehensive federation status."""

        if federation_id not in self.federations:
            raise ValueError(f"Federation {federation_id} not found")

        federation = self.federations[federation_id]

        status = {
            "federation_id": str(federation.federation_id),
            "name": federation.name,
            "strategy": federation.strategy.value,
            "aggregation_method": federation.aggregation_method.value,
            "privacy_mechanism": federation.privacy_mechanism.value,
            "global_model_version": federation.global_model_version,
            "is_active": federation.is_active,
            "participant_count": len(federation.participants),
            "active_participant_count": len(federation.active_participants),
            "total_rounds": len(federation.training_rounds),
            "current_round": None,
            "convergence_metric": federation.calculate_convergence_metric(),
            "last_training": (
                federation.last_training.isoformat()
                if federation.last_training
                else None
            ),
        }

        if federation.current_round:
            status["current_round"] = {
                "round_number": federation.current_round.round_number,
                "started_at": federation.current_round.started_at.isoformat(),
                "deadline": (
                    federation.current_round.deadline.isoformat()
                    if federation.current_round.deadline
                    else None
                ),
                "participation_rate": federation.current_round.participation_rate,
                "received_updates": len(federation.current_round.received_updates),
                "target_participants": len(
                    federation.current_round.target_participants
                ),
            }

        return status

    async def shutdown(self) -> None:
        """Shutdown coordinator and cleanup resources."""

        self.logger.info(f"Shutting down federated coordinator {self.coordinator_id}")

        # Close active connections
        self.active_connections.clear()

        # Clear federations
        self.federations.clear()

        self.logger.info("Federated coordinator shutdown complete")
