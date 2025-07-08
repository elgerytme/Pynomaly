"""Federated learning participant client for distributed training."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.models.detector import Detector
from pynomaly.domain.models.federated import (
    FederatedParticipant,
    ModelUpdate,
    ParticipantRole,
    PrivacyBudget,
)
from pynomaly.infrastructure.security.security_service import SecurityService


class FederatedParticipantClient:
    """Client for participating in federated learning networks."""

    def __init__(
        self,
        participant_id: UUID,
        name: str,
        role: ParticipantRole = ParticipantRole.PARTICIPANT,
        security_service: SecurityService | None = None,
        privacy_budget: PrivacyBudget | None = None,
    ):
        """Initialize federated participant client."""
        self.participant_id = participant_id
        self.name = name
        self.role = role
        self.security_service = security_service
        self.privacy_budget = privacy_budget
        self.logger = logging.getLogger(__name__)

        # Participant state
        self.public_key = f"participant_{participant_id}"
        self.is_connected = False
        self.current_federation_id: UUID | None = None
        self.local_model: Detector | None = None
        self.training_data: np.ndarray | None = None

        # Training state
        self.current_round: int | None = None
        self.global_model_parameters: dict[str, np.ndarray] | None = None
        self.local_updates: dict[int, ModelUpdate] = {}

        # Performance metrics
        self.computation_capacity = 1.0
        self.network_latency_ms = 0.0
        self.data_size = 0

        # Communication
        self.coordinator_endpoint: str | None = None
        self.message_queue: asyncio.Queue = asyncio.Queue()

        self.logger.info(f"Federated participant {participant_id} initialized")

    async def join_federation(
        self,
        federation_id: UUID,
        coordinator_endpoint: str,
        training_data: np.ndarray,
        data_labels: np.ndarray | None = None,
    ) -> bool:
        """Join federated learning network."""

        try:
            self.current_federation_id = federation_id
            self.coordinator_endpoint = coordinator_endpoint
            self.training_data = training_data
            self.data_size = len(training_data)

            # Create participant profile
            participant_profile = FederatedParticipant(
                participant_id=self.participant_id,
                name=self.name,
                role=self.role,
                public_key=self.public_key,
                trust_score=1.0,
                data_size=self.data_size,
                computation_capacity=self.computation_capacity,
                network_latency_ms=self.network_latency_ms,
                privacy_budget=self.privacy_budget,
            )

            # Register with coordinator
            success = await self._register_with_coordinator(
                federation_id, participant_profile
            )

            if success:
                self.is_connected = True

                # Start listening for coordinator messages
                asyncio.create_task(self._listen_for_messages())

                self.logger.info(
                    f"Successfully joined federation {federation_id}"
                )
                return True
            else:
                self.logger.error(f"Failed to join federation {federation_id}")
                return False

        except Exception as e:
            self.logger.error(f"Error joining federation: {e}")
            return False

    async def receive_global_model(
        self,
        round_number: int,
        global_parameters: dict[str, np.ndarray],
        model_version: str,
    ) -> None:
        """Receive global model from coordinator."""

        self.current_round = round_number
        self.global_model_parameters = global_parameters

        self.logger.info(
            f"Received global model for round {round_number}, "
            f"version {model_version}"
        )

        # Start local training
        asyncio.create_task(self._perform_local_training())

    async def _perform_local_training(self) -> None:
        """Perform local training and generate model update."""

        if not self.global_model_parameters or not self.training_data:
            self.logger.error("Missing global model or training data")
            return

        try:
            start_time = datetime.utcnow()

            # Simulate local training
            local_parameters = await self._train_local_model()

            # Calculate gradients (difference from global model)
            gradients = self._calculate_gradients(local_parameters)

            # Apply privacy mechanisms
            if self.privacy_budget:
                local_parameters = self._apply_local_differential_privacy(
                    local_parameters
                )

            training_time = (datetime.utcnow() - start_time).total_seconds()

            # Create model update
            model_update = ModelUpdate(
                update_id=uuid4(),
                participant_id=self.participant_id,
                round_number=self.current_round,
                parameters=local_parameters,
                gradients=gradients,
                data_size=self.data_size,
                computation_time_seconds=training_time,
                privacy_spent=0.01 if self.privacy_budget else 0.0,
            )

            # Store locally
            self.local_updates[self.current_round] = model_update

            # Send to coordinator
            await self._send_model_update(model_update)

            self.logger.info(
                f"Completed local training for round {self.current_round} "
                f"in {training_time:.2f} seconds"
            )

        except Exception as e:
            self.logger.error(f"Error in local training: {e}")

    async def _train_local_model(self) -> dict[str, np.ndarray]:
        """Train local model on participant's data."""

        if not self.local_model or not self.training_data:
            raise ValueError("Missing local model or training data")

        # Simulate training process
        # In practice, this would perform actual model training

        # Start with global parameters
        local_params = {}
        for name, global_param in self.global_model_parameters.items():
            # Simulate parameter updates
            noise = np.random.normal(0, 0.01, global_param.shape)
            local_params[name] = global_param + noise

        # Simulate training iterations
        num_epochs = 5
        learning_rate = 0.01

        for epoch in range(num_epochs):
            # Simulate batch training
            batch_size = min(32, len(self.training_data))
            num_batches = len(self.training_data) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_data = self.training_data[start_idx:end_idx]

                # Simulate gradient computation and parameter update
                for name, param in local_params.items():
                    gradient_noise = np.random.normal(0, 0.001, param.shape)
                    local_params[name] = param - learning_rate * gradient_noise

                # Small delay to simulate computation
                await asyncio.sleep(0.01)

        return local_params

    def _calculate_gradients(
        self, local_parameters: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Calculate gradients as difference from global model."""

        gradients = {}

        for name, local_param in local_parameters.items():
            if name in self.global_model_parameters:
                global_param = self.global_model_parameters[name]
                gradients[name] = local_param - global_param

        return gradients

    def _apply_local_differential_privacy(
        self, parameters: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Apply local differential privacy to model parameters."""

        if not self.privacy_budget:
            return parameters

        epsilon_per_param = self.privacy_budget.epsilon / len(parameters)

        if not self.privacy_budget.can_answer_query(epsilon_per_param):
            self.logger.warning("Insufficient privacy budget for local DP")
            return parameters

        noisy_parameters = {}

        for param_name, param_value in parameters.items():
            # Calculate noise scale based on parameter sensitivity
            sensitivity = np.std(param_value)
            noise_scale = 2 * sensitivity / epsilon_per_param

            # Add Laplace noise for local DP
            noise = np.random.laplace(0, noise_scale, param_value.shape)
            noisy_parameters[param_name] = param_value + noise

        # Consume privacy budget
        self.privacy_budget.consume_budget(epsilon_per_param)

        return noisy_parameters

    async def _register_with_coordinator(
        self,
        federation_id: UUID,
        participant_profile: FederatedParticipant,
    ) -> bool:
        """Register with federation coordinator."""

        # Simulate network communication with coordinator
        # In practice, this would use HTTP/gRPC/WebSocket

        try:
            # Simulate network latency
            await asyncio.sleep(0.1)

            # Simulate successful registration
            self.logger.info(f"Registered with coordinator at {self.coordinator_endpoint}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register with coordinator: {e}")
            return False

    async def _send_model_update(self, model_update: ModelUpdate) -> None:
        """Send model update to coordinator."""

        try:
            # Simulate network communication
            await asyncio.sleep(0.05)  # Network latency

            self.logger.debug(
                f"Sent model update for round {model_update.round_number} "
                f"to coordinator"
            )

        except Exception as e:
            self.logger.error(f"Failed to send model update: {e}")

    async def _listen_for_messages(self) -> None:
        """Listen for messages from coordinator."""

        while self.is_connected:
            try:
                # Simulate receiving messages
                await asyncio.sleep(1)

                # Process any queued messages
                while not self.message_queue.empty():
                    message = await self.message_queue.get()
                    await self._process_coordinator_message(message)

            except Exception as e:
                self.logger.error(f"Error listening for messages: {e}")
                await asyncio.sleep(5)  # Retry after delay

    async def _process_coordinator_message(self, message: dict[str, Any]) -> None:
        """Process message from coordinator."""

        message_type = message.get("type")

        if message_type == "global_model":
            await self.receive_global_model(
                message["round_number"],
                message["parameters"],
                message["model_version"],
            )
        elif message_type == "round_complete":
            self._handle_round_completion(message)
        elif message_type == "federation_shutdown":
            await self._handle_federation_shutdown()
        else:
            self.logger.warning(f"Unknown message type: {message_type}")

    def _handle_round_completion(self, message: dict[str, Any]) -> None:
        """Handle round completion notification."""

        round_number = message.get("round_number")
        convergence_metric = message.get("convergence_metric")

        self.logger.info(
            f"Round {round_number} completed. "
            f"Convergence metric: {convergence_metric}"
        )

    async def _handle_federation_shutdown(self) -> None:
        """Handle federation shutdown notification."""

        self.logger.info("Federation is shutting down")
        await self.leave_federation()

    async def leave_federation(self) -> None:
        """Leave current federation."""

        if not self.is_connected:
            return

        try:
            # Notify coordinator
            await self._notify_coordinator_departure()

            # Cleanup local state
            self.is_connected = False
            self.current_federation_id = None
            self.local_model = None
            self.training_data = None
            self.current_round = None
            self.global_model_parameters = None

            self.logger.info("Successfully left federation")

        except Exception as e:
            self.logger.error(f"Error leaving federation: {e}")

    async def _notify_coordinator_departure(self) -> None:
        """Notify coordinator of departure."""

        try:
            # Simulate network communication
            await asyncio.sleep(0.05)

            self.logger.debug("Notified coordinator of departure")

        except Exception as e:
            self.logger.error(f"Failed to notify coordinator: {e}")

    def get_participant_status(self) -> dict[str, Any]:
        """Get current participant status."""

        return {
            "participant_id": str(self.participant_id),
            "name": self.name,
            "role": self.role.value,
            "is_connected": self.is_connected,
            "federation_id": str(self.current_federation_id) if self.current_federation_id else None,
            "current_round": self.current_round,
            "data_size": self.data_size,
            "computation_capacity": self.computation_capacity,
            "network_latency_ms": self.network_latency_ms,
            "total_updates_sent": len(self.local_updates),
            "privacy_budget_remaining": (
                self.privacy_budget.epsilon - self.privacy_budget.spent_epsilon
                if self.privacy_budget else None
            ),
        }

    async def simulate_data_collection(
        self,
        num_samples: int = 1000,
        num_features: int = 10,
        anomaly_ratio: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate local data collection for testing."""

        # Generate synthetic training data
        normal_samples = int(num_samples * (1 - anomaly_ratio))
        anomaly_samples = num_samples - normal_samples

        # Normal data (Gaussian)
        normal_data = np.random.randn(normal_samples, num_features)

        # Anomaly data (shifted Gaussian)
        anomaly_data = np.random.randn(anomaly_samples, num_features) * 2 + 3

        # Combine and shuffle
        data = np.vstack([normal_data, anomaly_data])
        labels = np.hstack([
            np.zeros(normal_samples, dtype=bool),
            np.ones(anomaly_samples, dtype=bool)
        ])

        # Shuffle
        indices = np.random.permutation(num_samples)
        data = data[indices]
        labels = labels[indices]

        self.training_data = data
        self.data_size = num_samples

        self.logger.info(
            f"Simulated {num_samples} samples "
            f"({anomaly_samples} anomalies, {normal_samples} normal)"
        )

        return data, labels

    async def benchmark_local_training(
        self,
        num_rounds: int = 5
    ) -> dict[str, Any]:
        """Benchmark local training performance."""

        if not self.training_data:
            await self.simulate_data_collection()

        # Initialize dummy global parameters
        self.global_model_parameters = {
            "weights": np.random.randn(10, 5),
            "bias": np.random.randn(5),
        }

        training_times = []

        for round_num in range(1, num_rounds + 1):
            self.current_round = round_num

            start_time = datetime.utcnow()
            await self._perform_local_training()
            training_time = (datetime.utcnow() - start_time).total_seconds()

            training_times.append(training_time)

            self.logger.info(
                f"Round {round_num} training completed in {training_time:.3f}s"
            )

        return {
            "total_rounds": num_rounds,
            "avg_training_time": np.mean(training_times),
            "max_training_time": np.max(training_times),
            "min_training_time": np.min(training_times),
            "total_time": sum(training_times),
            "data_size": self.data_size,
            "throughput_samples_per_second": self.data_size / np.mean(training_times),
        }
