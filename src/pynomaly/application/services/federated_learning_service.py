"""Federated learning service for privacy-preserving distributed anomaly detection training."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FederatedStrategy(str, Enum):
    """Federated learning strategies."""

    FEDERATED_AVERAGING = "federated_averaging"
    FEDERATED_SGD = "federated_sgd"
    FEDERATED_PROXIMAL = "federated_proximal"
    SECURE_AGGREGATION = "secure_aggregation"
    DIFFERENTIAL_PRIVATE = "differential_private"
    BYZANTINE_RESILIENT = "byzantine_resilient"


class PrivacyLevel(str, Enum):
    """Privacy protection levels."""

    BASIC = "basic"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SECURE_MULTIPARTY = "secure_multiparty"
    ZERO_KNOWLEDGE = "zero_knowledge"


class ClientStatus(str, Enum):
    """Client participation status."""

    ACTIVE = "active"
    IDLE = "idle"
    TRAINING = "training"
    UPLOADING = "uploading"
    FAILED = "failed"
    DISCONNECTED = "disconnected"


@dataclass
class PrivacyBudget:
    """Differential privacy budget tracking."""

    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5  # Failure probability
    spent_epsilon: float = 0.0
    remaining_epsilon: float = 1.0
    noise_multiplier: float = 1.1
    clipping_norm: float = 1.0

    def consume_budget(self, amount: float) -> bool:
        """Consume privacy budget."""
        if self.remaining_epsilon >= amount:
            self.spent_epsilon += amount
            self.remaining_epsilon -= amount
            return True
        return False

    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.remaining_epsilon <= 0.01


@dataclass
class ClientMetrics:
    """Metrics for federated learning client."""

    client_id: str
    data_size: int = 0
    training_time: float = 0.0
    communication_time: float = 0.0
    model_accuracy: float = 0.0
    contribution_score: float = 0.0
    privacy_budget_used: float = 0.0
    rounds_participated: int = 0
    last_seen: datetime = field(default_factory=datetime.now)
    reliability_score: float = 1.0


class FederatedClient(BaseModel):
    """Federated learning client representation."""

    client_id: str
    status: ClientStatus = ClientStatus.IDLE
    capabilities: dict[str, Any] = Field(default_factory=dict)
    privacy_budget: PrivacyBudget = Field(default_factory=PrivacyBudget)
    metrics: ClientMetrics = Field(default_factory=lambda: ClientMetrics(client_id=""))
    local_model_weights: dict[str, Any] | None = None
    data_characteristics: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        if not self.metrics.client_id:
            self.metrics.client_id = self.client_id


@dataclass
class FederatedRound:
    """Information about a federated learning round."""

    round_number: int
    start_time: datetime
    end_time: datetime | None = None
    participating_clients: list[str] = field(default_factory=list)
    aggregated_weights: dict[str, Any] | None = None
    global_performance: dict[str, float] = field(default_factory=dict)
    privacy_guarantees: dict[str, float] = field(default_factory=dict)
    convergence_metrics: dict[str, float] = field(default_factory=dict)


class SecureAggregator:
    """Secure aggregation with differential privacy and byzantine resilience."""

    def __init__(
        self,
        privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY,
        byzantine_tolerance: float = 0.3,
        noise_multiplier: float = 1.1,
        clipping_norm: float = 1.0,
    ):
        self.privacy_level = privacy_level
        self.byzantine_tolerance = byzantine_tolerance
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm

    async def aggregate_weights(
        self,
        client_weights: list[tuple[str, dict[str, Any], float]],
        privacy_budgets: dict[str, PrivacyBudget],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Securely aggregate model weights from clients.

        Args:
            client_weights: List of (client_id, weights, sample_size) tuples
            privacy_budgets: Privacy budgets for each client

        Returns:
            Tuple of (aggregated_weights, privacy_metrics)
        """
        if not client_weights:
            return {}, {}

        # Step 1: Byzantine filtering
        filtered_weights = await self._filter_byzantine_clients(client_weights)

        # Step 2: Apply differential privacy
        if self.privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY:
            noisy_weights = await self._apply_differential_privacy(
                filtered_weights, privacy_budgets
            )
        else:
            noisy_weights = filtered_weights

        # Step 3: Secure aggregation
        aggregated_weights = await self._federated_averaging(noisy_weights)

        # Step 4: Calculate privacy metrics
        privacy_metrics = await self._calculate_privacy_metrics(
            client_weights, privacy_budgets
        )

        return aggregated_weights, privacy_metrics

    async def _filter_byzantine_clients(
        self, client_weights: list[tuple[str, dict[str, Any], float]]
    ) -> list[tuple[str, dict[str, Any], float]]:
        """Filter out potentially malicious clients using statistical methods."""
        if len(client_weights) <= 2:
            return client_weights

        # Extract weight vectors for analysis
        weight_vectors = []
        for client_id, weights, sample_size in client_weights:
            # Flatten weights into a single vector
            flattened = self._flatten_weights(weights)
            weight_vectors.append((client_id, flattened, sample_size))

        # Calculate pairwise similarities
        similarities = {}
        for i, (client_i, weights_i, _) in enumerate(weight_vectors):
            for j, (client_j, weights_j, _) in enumerate(weight_vectors):
                if i != j:
                    similarity = self._calculate_cosine_similarity(weights_i, weights_j)
                    similarities[(client_i, client_j)] = similarity

        # Identify outliers (potential byzantine clients)
        client_scores = defaultdict(list)
        for (client_i, client_j), similarity in similarities.items():
            client_scores[client_i].append(similarity)

        # Calculate median similarity for each client
        client_median_similarities = {}
        for client_id, scores in client_scores.items():
            client_median_similarities[client_id] = np.median(scores) if scores else 0.0

        # Filter out clients with very low median similarity
        similarity_threshold = 0.3  # Configurable
        filtered_clients = []

        for client_id, weights, sample_size in client_weights:
            median_sim = client_median_similarities.get(client_id, 1.0)
            if median_sim >= similarity_threshold:
                filtered_clients.append((client_id, weights, sample_size))
            else:
                logger.warning(
                    f"Filtering potential byzantine client: {client_id} (similarity: {median_sim:.3f})"
                )

        # Ensure we don't filter too many clients
        max_filtered = int(len(client_weights) * self.byzantine_tolerance)
        if len(client_weights) - len(filtered_clients) > max_filtered:
            # Keep clients with highest similarities
            sorted_clients = sorted(
                client_weights,
                key=lambda x: client_median_similarities.get(x[0], 0.0),
                reverse=True,
            )
            filtered_clients = sorted_clients[: len(client_weights) - max_filtered]

        return filtered_clients

    def _flatten_weights(self, weights: dict[str, Any]) -> np.ndarray:
        """Flatten nested weight dictionary into a single vector."""
        flattened = []
        for key in sorted(weights.keys()):  # Ensure consistent ordering
            value = weights[key]
            if isinstance(value, (list, np.ndarray)):
                flattened.extend(np.array(value).flatten())
            elif isinstance(value, (int, float)):
                flattened.append(value)
            elif isinstance(value, dict):
                # Recursively flatten nested dictionaries
                nested_flattened = self._flatten_weights(value)
                flattened.extend(nested_flattened)

        return np.array(flattened)

    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Ensure vectors have same length
            min_len = min(len(vec1), len(vec2))
            vec1, vec2 = vec1[:min_len], vec2[:min_len]

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0

    async def _apply_differential_privacy(
        self,
        client_weights: list[tuple[str, dict[str, Any], float]],
        privacy_budgets: dict[str, PrivacyBudget],
    ) -> list[tuple[str, dict[str, Any], float]]:
        """Apply differential privacy to client weights."""
        noisy_weights = []

        for client_id, weights, sample_size in client_weights:
            budget = privacy_budgets.get(client_id)
            if not budget or budget.is_exhausted():
                logger.warning(f"Privacy budget exhausted for client {client_id}")
                continue

            # Apply gradient clipping
            clipped_weights = await self._clip_weights(weights, budget.clipping_norm)

            # Add calibrated noise
            noisy_weights_dict = await self._add_gaussian_noise(
                clipped_weights, budget.noise_multiplier, budget.clipping_norm
            )

            # Consume privacy budget
            epsilon_used = 1.0 / (budget.noise_multiplier**2)  # Simplified calculation
            budget.consume_budget(epsilon_used)

            noisy_weights.append((client_id, noisy_weights_dict, sample_size))

        return noisy_weights

    async def _clip_weights(
        self, weights: dict[str, Any], clip_norm: float
    ) -> dict[str, Any]:
        """Apply gradient clipping to weights."""
        clipped_weights = {}

        for key, value in weights.items():
            if isinstance(value, (list, np.ndarray)):
                value_array = np.array(value)
                norm = np.linalg.norm(value_array)

                if norm > clip_norm:
                    clipped_value = value_array * (clip_norm / norm)
                else:
                    clipped_value = value_array

                clipped_weights[key] = clipped_value.tolist()
            elif isinstance(value, (int, float)):
                # Clip scalar values
                clipped_weights[key] = max(-clip_norm, min(clip_norm, value))
            elif isinstance(value, dict):
                # Recursively clip nested dictionaries
                clipped_weights[key] = await self._clip_weights(value, clip_norm)
            else:
                clipped_weights[key] = value

        return clipped_weights

    async def _add_gaussian_noise(
        self, weights: dict[str, Any], noise_multiplier: float, sensitivity: float
    ) -> dict[str, Any]:
        """Add calibrated Gaussian noise for differential privacy."""
        noisy_weights = {}

        for key, value in weights.items():
            if isinstance(value, (list, np.ndarray)):
                value_array = np.array(value)
                noise_scale = noise_multiplier * sensitivity
                noise = np.random.normal(0, noise_scale, value_array.shape)
                noisy_weights[key] = (value_array + noise).tolist()
            elif isinstance(value, (int, float)):
                noise_scale = noise_multiplier * sensitivity
                noise = np.random.normal(0, noise_scale)
                noisy_weights[key] = value + noise
            elif isinstance(value, dict):
                # Recursively add noise to nested dictionaries
                noisy_weights[key] = await self._add_gaussian_noise(
                    value, noise_multiplier, sensitivity
                )
            else:
                noisy_weights[key] = value

        return noisy_weights

    async def _federated_averaging(
        self, client_weights: list[tuple[str, dict[str, Any], float]]
    ) -> dict[str, Any]:
        """Perform federated averaging aggregation."""
        if not client_weights:
            return {}

        # Calculate total sample size for weighted averaging
        total_samples = sum(sample_size for _, _, sample_size in client_weights)

        if total_samples == 0:
            # Equal weighting if no sample size information
            weights = [1.0 / len(client_weights)] * len(client_weights)
        else:
            weights = [
                sample_size / total_samples for _, _, sample_size in client_weights
            ]

        # Initialize aggregated weights with first client's structure
        _, first_weights, _ = client_weights[0]
        aggregated_weights = self._initialize_aggregated_weights(first_weights)

        # Aggregate weights using weighted average
        for i, (client_id, client_weights_dict, _) in enumerate(client_weights):
            weight = weights[i]
            aggregated_weights = self._weighted_add(
                aggregated_weights, client_weights_dict, weight
            )

        return aggregated_weights

    def _initialize_aggregated_weights(
        self, weights_structure: dict[str, Any]
    ) -> dict[str, Any]:
        """Initialize aggregated weights with zeros matching the structure."""
        initialized = {}

        for key, value in weights_structure.items():
            if isinstance(value, (list, np.ndarray)):
                initialized[key] = np.zeros_like(np.array(value)).tolist()
            elif isinstance(value, (int, float)):
                initialized[key] = 0.0
            elif isinstance(value, dict):
                initialized[key] = self._initialize_aggregated_weights(value)
            else:
                initialized[key] = value  # Keep non-numeric values as-is

        return initialized

    def _weighted_add(
        self, aggregated: dict[str, Any], client_weights: dict[str, Any], weight: float
    ) -> dict[str, Any]:
        """Add client weights to aggregated weights with given weight."""
        result = {}

        for key in aggregated.keys():
            if key in client_weights:
                agg_value = aggregated[key]
                client_value = client_weights[key]

                if isinstance(agg_value, list) and isinstance(
                    client_value, (list, np.ndarray)
                ):
                    agg_array = np.array(agg_value)
                    client_array = np.array(client_value)

                    # Ensure same shape
                    if agg_array.shape == client_array.shape:
                        result[key] = (agg_array + weight * client_array).tolist()
                    else:
                        result[key] = agg_value  # Keep original if shapes don't match

                elif isinstance(agg_value, (int, float)) and isinstance(
                    client_value, (int, float)
                ):
                    result[key] = agg_value + weight * client_value

                elif isinstance(agg_value, dict) and isinstance(client_value, dict):
                    result[key] = self._weighted_add(agg_value, client_value, weight)

                else:
                    result[key] = agg_value  # Keep original for incompatible types
            else:
                result[key] = aggregated[key]  # Keep aggregated value if key missing

        return result

    async def _calculate_privacy_metrics(
        self,
        client_weights: list[tuple[str, dict[str, Any], float]],
        privacy_budgets: dict[str, PrivacyBudget],
    ) -> dict[str, float]:
        """Calculate privacy metrics for the aggregation."""
        metrics = {}

        # Total privacy budget consumed
        total_epsilon_consumed = sum(
            budget.spent_epsilon for budget in privacy_budgets.values()
        )

        # Average noise level
        noise_levels = [budget.noise_multiplier for budget in privacy_budgets.values()]
        avg_noise_level = np.mean(noise_levels) if noise_levels else 0.0

        # Privacy efficiency (utility vs privacy trade-off)
        remaining_budgets = [
            budget.remaining_epsilon for budget in privacy_budgets.values()
        ]
        avg_remaining_budget = np.mean(remaining_budgets) if remaining_budgets else 0.0

        metrics = {
            "total_epsilon_consumed": total_epsilon_consumed,
            "average_noise_level": avg_noise_level,
            "average_remaining_budget": avg_remaining_budget,
            "participating_clients": len(client_weights),
            "privacy_efficiency": 1.0
            - (total_epsilon_consumed / (len(privacy_budgets) * 1.0)),
        }

        return metrics


class FederatedLearningService:
    """Federated learning service for privacy-preserving distributed training."""

    def __init__(
        self,
        strategy: FederatedStrategy = FederatedStrategy.FEDERATED_AVERAGING,
        privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY,
        min_clients_per_round: int = 3,
        max_clients_per_round: int = 100,
        rounds_per_epoch: int = 10,
        client_selection_strategy: str = "random",
        convergence_threshold: float = 0.001,
    ):
        """Initialize federated learning service.

        Args:
            strategy: Federated learning strategy
            privacy_level: Privacy protection level
            min_clients_per_round: Minimum clients required per round
            max_clients_per_round: Maximum clients per round
            rounds_per_epoch: Number of rounds per training epoch
            client_selection_strategy: Strategy for selecting clients
            convergence_threshold: Threshold for convergence detection
        """
        self.strategy = strategy
        self.privacy_level = privacy_level
        self.min_clients_per_round = min_clients_per_round
        self.max_clients_per_round = max_clients_per_round
        self.rounds_per_epoch = rounds_per_epoch
        self.client_selection_strategy = client_selection_strategy
        self.convergence_threshold = convergence_threshold

        # Client management
        self.clients: dict[str, FederatedClient] = {}
        self.client_queues: dict[str, asyncio.Queue] = {}

        # Training state
        self.current_round = 0
        self.global_model_weights: dict[str, Any] | None = None
        self.round_history: list[FederatedRound] = []

        # Secure aggregation
        self.aggregator = SecureAggregator(
            privacy_level=privacy_level,
            byzantine_tolerance=0.3,
            noise_multiplier=1.1,
            clipping_norm=1.0,
        )

        # Performance tracking
        self.convergence_history: deque = deque(maxlen=50)
        self.performance_metrics: dict[str, list[float]] = defaultdict(list)

        logger.info(f"Initialized federated learning service with strategy: {strategy}")

    async def register_client(
        self,
        client_id: str,
        capabilities: dict[str, Any],
        privacy_preferences: dict[str, Any],
    ) -> bool:
        """Register a new federated learning client.

        Args:
            client_id: Unique client identifier
            capabilities: Client computational capabilities
            privacy_preferences: Client privacy preferences

        Returns:
            Success status
        """
        try:
            # Create privacy budget based on preferences
            privacy_budget = PrivacyBudget(
                epsilon=privacy_preferences.get("epsilon", 1.0),
                delta=privacy_preferences.get("delta", 1e-5),
                noise_multiplier=privacy_preferences.get("noise_multiplier", 1.1),
                clipping_norm=privacy_preferences.get("clipping_norm", 1.0),
            )

            # Create client instance
            client = FederatedClient(
                client_id=client_id,
                capabilities=capabilities,
                privacy_budget=privacy_budget,
                status=ClientStatus.IDLE,
            )

            # Store client
            self.clients[client_id] = client
            self.client_queues[client_id] = asyncio.Queue()

            logger.info(f"Registered federated client: {client_id}")
            return True

        except Exception as e:
            logger.error(f"Error registering client {client_id}: {e}")
            return False

    async def unregister_client(self, client_id: str) -> bool:
        """Unregister a federated learning client."""
        try:
            if client_id in self.clients:
                del self.clients[client_id]
                del self.client_queues[client_id]
                logger.info(f"Unregistered federated client: {client_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error unregistering client {client_id}: {e}")
            return False

    async def start_training_round(self) -> dict[str, Any]:
        """Start a new federated training round.

        Returns:
            Round information and selected clients
        """
        self.current_round += 1
        logger.info(f"Starting federated training round {self.current_round}")

        # Select clients for this round
        selected_clients = await self._select_clients_for_round()

        if len(selected_clients) < self.min_clients_per_round:
            logger.warning(
                f"Insufficient clients for round {self.current_round}: {len(selected_clients)} < {self.min_clients_per_round}"
            )
            return {
                "round_number": self.current_round,
                "status": "insufficient_clients",
                "selected_clients": [],
                "message": "Not enough clients available for training",
            }

        # Create round information
        round_info = FederatedRound(
            round_number=self.current_round,
            start_time=datetime.now(),
            participating_clients=[client.client_id for client in selected_clients],
        )

        # Update client status
        for client in selected_clients:
            client.status = ClientStatus.TRAINING

        # Send training tasks to selected clients
        training_tasks = await self._create_training_tasks(selected_clients)

        return {
            "round_number": self.current_round,
            "status": "started",
            "selected_clients": [client.client_id for client in selected_clients],
            "training_tasks": training_tasks,
            "privacy_requirements": self._get_privacy_requirements(selected_clients),
        }

    async def _select_clients_for_round(self) -> list[FederatedClient]:
        """Select clients for the current training round."""
        # Get available clients
        available_clients = [
            client
            for client in self.clients.values()
            if client.status in [ClientStatus.IDLE, ClientStatus.ACTIVE]
            and not client.privacy_budget.is_exhausted()
        ]

        if not available_clients:
            return []

        # Apply client selection strategy
        if self.client_selection_strategy == "random":
            selected = await self._random_client_selection(available_clients)
        elif self.client_selection_strategy == "weighted":
            selected = await self._weighted_client_selection(available_clients)
        elif self.client_selection_strategy == "diverse":
            selected = await self._diverse_client_selection(available_clients)
        else:
            # Default to random
            selected = await self._random_client_selection(available_clients)

        return selected[: self.max_clients_per_round]

    async def _random_client_selection(
        self, available_clients: list[FederatedClient]
    ) -> list[FederatedClient]:
        """Random client selection."""
        num_clients = min(len(available_clients), self.max_clients_per_round)
        indices = np.random.choice(len(available_clients), num_clients, replace=False)
        return [available_clients[i] for i in indices]

    async def _weighted_client_selection(
        self, available_clients: list[FederatedClient]
    ) -> list[FederatedClient]:
        """Weighted client selection based on reliability and contribution."""
        # Calculate selection weights
        weights = []
        for client in available_clients:
            weight = (
                client.metrics.reliability_score * 0.4
                + client.metrics.contribution_score * 0.3
                + (
                    1.0
                    - client.privacy_budget.spent_epsilon
                    / client.privacy_budget.epsilon
                )
                * 0.3
            )
            weights.append(weight)

        # Normalize weights
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(available_clients)) / len(available_clients)

        # Select clients based on weights
        num_clients = min(len(available_clients), self.max_clients_per_round)
        indices = np.random.choice(
            len(available_clients), num_clients, replace=False, p=weights
        )

        return [available_clients[i] for i in indices]

    async def _diverse_client_selection(
        self, available_clients: list[FederatedClient]
    ) -> list[FederatedClient]:
        """Diverse client selection to maximize data heterogeneity."""
        # This is a simplified version - in practice, you'd use more sophisticated diversity metrics

        # Group clients by data characteristics similarity
        client_groups = defaultdict(list)
        for client in available_clients:
            # Create a simple hash of data characteristics
            chars_str = str(sorted(client.data_characteristics.items()))
            group_key = hashlib.md5(chars_str.encode()).hexdigest()[:8]
            client_groups[group_key].append(client)

        # Select clients from different groups
        selected_clients = []
        group_list = list(client_groups.values())

        while len(selected_clients) < self.max_clients_per_round and group_list:
            for group in group_list:
                if group and len(selected_clients) < self.max_clients_per_round:
                    # Select client with highest reliability from this group
                    best_client = max(group, key=lambda c: c.metrics.reliability_score)
                    selected_clients.append(best_client)
                    group.remove(best_client)

            # Remove empty groups
            group_list = [group for group in group_list if group]

        return selected_clients

    async def _create_training_tasks(
        self, selected_clients: list[FederatedClient]
    ) -> dict[str, Any]:
        """Create training tasks for selected clients."""
        tasks = {}

        for client in selected_clients:
            task = {
                "client_id": client.client_id,
                "round_number": self.current_round,
                "global_model_weights": self.global_model_weights,
                "training_config": {
                    "local_epochs": 5,
                    "learning_rate": 0.01,
                    "batch_size": 32,
                    "privacy_requirements": {
                        "noise_multiplier": client.privacy_budget.noise_multiplier,
                        "clipping_norm": client.privacy_budget.clipping_norm,
                        "epsilon_budget": client.privacy_budget.remaining_epsilon,
                    },
                },
                "deadline": (datetime.now() + timedelta(minutes=30)).isoformat(),
            }

            tasks[client.client_id] = task

        return tasks

    def _get_privacy_requirements(
        self, selected_clients: list[FederatedClient]
    ) -> dict[str, Any]:
        """Get privacy requirements for the round."""
        return {
            "privacy_level": self.privacy_level.value,
            "differential_privacy": self.privacy_level
            == PrivacyLevel.DIFFERENTIAL_PRIVACY,
            "min_noise_multiplier": min(
                client.privacy_budget.noise_multiplier for client in selected_clients
            ),
            "max_epsilon_per_client": max(
                client.privacy_budget.remaining_epsilon for client in selected_clients
            ),
            "byzantine_tolerance": 0.3,
        }

    async def receive_client_update(
        self,
        client_id: str,
        local_weights: dict[str, Any],
        training_metrics: dict[str, float],
        data_size: int,
    ) -> dict[str, Any]:
        """Receive and process client model update.

        Args:
            client_id: Client identifier
            local_weights: Local model weights from client
            training_metrics: Training metrics from client
            data_size: Size of client's training data

        Returns:
            Reception status and feedback
        """
        try:
            if client_id not in self.clients:
                return {"status": "error", "message": "Client not registered"}

            client = self.clients[client_id]

            # Validate update
            if not self._validate_client_update(
                client, local_weights, training_metrics
            ):
                return {"status": "error", "message": "Invalid client update"}

            # Store client update
            client.local_model_weights = local_weights
            client.status = ClientStatus.UPLOADING

            # Update client metrics
            client.metrics.data_size = data_size
            client.metrics.model_accuracy = training_metrics.get("accuracy", 0.0)
            client.metrics.training_time = training_metrics.get("training_time", 0.0)
            client.metrics.rounds_participated += 1
            client.metrics.last_seen = datetime.now()

            # Calculate contribution score
            client.metrics.contribution_score = self._calculate_contribution_score(
                client, training_metrics, data_size
            )

            logger.info(
                f"Received update from client {client_id} (data_size: {data_size})"
            )

            return {
                "status": "received",
                "message": "Update received successfully",
                "contribution_score": client.metrics.contribution_score,
            }

        except Exception as e:
            logger.error(f"Error receiving update from client {client_id}: {e}")
            return {"status": "error", "message": str(e)}

    def _validate_client_update(
        self,
        client: FederatedClient,
        local_weights: dict[str, Any],
        training_metrics: dict[str, float],
    ) -> bool:
        """Validate client model update."""
        # Check if client is expected to participate
        if client.status != ClientStatus.TRAINING:
            logger.warning(
                f"Unexpected update from client {client.client_id} (status: {client.status})"
            )
            return False

        # Check if weights have expected structure
        if not local_weights or not isinstance(local_weights, dict):
            logger.warning(f"Invalid weights structure from client {client.client_id}")
            return False

        # Check training metrics
        required_metrics = ["accuracy", "training_time"]
        for metric in required_metrics:
            if metric not in training_metrics:
                logger.warning(
                    f"Missing metric {metric} from client {client.client_id}"
                )
                return False

        # Check for suspicious metrics
        accuracy = training_metrics.get("accuracy", 0.0)
        if accuracy < 0.0 or accuracy > 1.0:
            logger.warning(
                f"Suspicious accuracy from client {client.client_id}: {accuracy}"
            )
            return False

        return True

    def _calculate_contribution_score(
        self,
        client: FederatedClient,
        training_metrics: dict[str, float],
        data_size: int,
    ) -> float:
        """Calculate client contribution score."""
        # Base score on data size (normalized)
        max_data_size = max(c.metrics.data_size for c in self.clients.values()) or 1
        data_score = min(1.0, data_size / max_data_size)

        # Score based on model accuracy
        accuracy_score = training_metrics.get("accuracy", 0.0)

        # Score based on reliability (inverse of training time variability)
        training_time = training_metrics.get("training_time", 1.0)
        if client.metrics.rounds_participated > 1:
            # Simplified reliability based on consistent training times
            reliability_score = min(
                1.0, 60.0 / max(training_time, 1.0)
            )  # Prefer faster training
        else:
            reliability_score = 0.5  # Default for new clients

        # Combined contribution score
        contribution_score = (
            0.4 * data_score + 0.4 * accuracy_score + 0.2 * reliability_score
        )

        return contribution_score

    async def aggregate_round(self) -> dict[str, Any]:
        """Aggregate client updates for the current round.

        Returns:
            Aggregation results and updated global model
        """
        try:
            # Get clients with updates for current round
            clients_with_updates = [
                client
                for client in self.clients.values()
                if (
                    client.status == ClientStatus.UPLOADING
                    and client.local_model_weights is not None
                )
            ]

            if len(clients_with_updates) < self.min_clients_per_round:
                return {
                    "status": "insufficient_updates",
                    "message": f"Only {len(clients_with_updates)} clients provided updates",
                    "required": self.min_clients_per_round,
                }

            # Prepare client weights for aggregation
            client_weights = []
            privacy_budgets = {}

            for client in clients_with_updates:
                client_weights.append(
                    (
                        client.client_id,
                        client.local_model_weights,
                        client.metrics.data_size,
                    )
                )
                privacy_budgets[client.client_id] = client.privacy_budget

            # Perform secure aggregation
            (
                aggregated_weights,
                privacy_metrics,
            ) = await self.aggregator.aggregate_weights(client_weights, privacy_budgets)

            # Update global model
            self.global_model_weights = aggregated_weights

            # Calculate round performance
            round_performance = self._calculate_round_performance(clients_with_updates)

            # Update convergence tracking
            convergence_metrics = self._check_convergence(round_performance)

            # Create round record
            current_round_info = FederatedRound(
                round_number=self.current_round,
                start_time=datetime.now() - timedelta(minutes=30),  # Approximate
                end_time=datetime.now(),
                participating_clients=[
                    client.client_id for client in clients_with_updates
                ],
                aggregated_weights=aggregated_weights,
                global_performance=round_performance,
                privacy_guarantees=privacy_metrics,
                convergence_metrics=convergence_metrics,
            )

            self.round_history.append(current_round_info)

            # Update client status
            for client in clients_with_updates:
                client.status = ClientStatus.IDLE
                client.local_model_weights = None

            # Update performance tracking
            self.performance_metrics["global_accuracy"].append(
                round_performance.get("weighted_accuracy", 0.0)
            )
            self.performance_metrics["privacy_efficiency"].append(
                privacy_metrics.get("privacy_efficiency", 0.0)
            )

            logger.info(f"Completed aggregation for round {self.current_round}")

            return {
                "status": "completed",
                "round_number": self.current_round,
                "participating_clients": len(clients_with_updates),
                "global_model_updated": True,
                "performance_metrics": round_performance,
                "privacy_metrics": privacy_metrics,
                "convergence_metrics": convergence_metrics,
                "next_round_ready": True,
            }

        except Exception as e:
            logger.error(f"Error in round aggregation: {e}")
            return {"status": "error", "message": str(e)}

    def _calculate_round_performance(
        self, participating_clients: list[FederatedClient]
    ) -> dict[str, float]:
        """Calculate performance metrics for the round."""
        if not participating_clients:
            return {}

        # Weighted average accuracy
        total_data_size = sum(
            client.metrics.data_size for client in participating_clients
        )
        if total_data_size > 0:
            weighted_accuracy = (
                sum(
                    client.metrics.model_accuracy * client.metrics.data_size
                    for client in participating_clients
                )
                / total_data_size
            )
        else:
            weighted_accuracy = np.mean(
                [client.metrics.model_accuracy for client in participating_clients]
            )

        # Average training time
        avg_training_time = np.mean(
            [client.metrics.training_time for client in participating_clients]
        )

        # Client diversity (simplified)
        contribution_scores = [
            client.metrics.contribution_score for client in participating_clients
        ]
        diversity_score = (
            np.std(contribution_scores) if len(contribution_scores) > 1 else 0.0
        )

        return {
            "weighted_accuracy": weighted_accuracy,
            "average_training_time": avg_training_time,
            "client_diversity": diversity_score,
            "participation_rate": len(participating_clients) / len(self.clients),
            "total_data_samples": total_data_size,
        }

    def _check_convergence(
        self, round_performance: dict[str, float]
    ) -> dict[str, float]:
        """Check convergence of federated learning."""
        current_accuracy = round_performance.get("weighted_accuracy", 0.0)
        self.convergence_history.append(current_accuracy)

        convergence_metrics = {
            "current_accuracy": current_accuracy,
            "is_converged": False,
            "improvement_rate": 0.0,
            "stability_score": 0.0,
        }

        if len(self.convergence_history) >= 3:
            # Calculate improvement rate
            recent_accuracies = list(self.convergence_history)[-3:]
            improvement_rate = (recent_accuracies[-1] - recent_accuracies[0]) / 2
            convergence_metrics["improvement_rate"] = improvement_rate

            # Check convergence
            if abs(improvement_rate) < self.convergence_threshold:
                convergence_metrics["is_converged"] = True

            # Calculate stability
            stability_score = 1.0 - np.std(recent_accuracies)
            convergence_metrics["stability_score"] = max(0.0, stability_score)

        return convergence_metrics

    async def get_federated_status(self) -> dict[str, Any]:
        """Get comprehensive federated learning system status."""
        # Client statistics
        client_stats = {
            "total_clients": len(self.clients),
            "active_clients": len(
                [c for c in self.clients.values() if c.status == ClientStatus.ACTIVE]
            ),
            "training_clients": len(
                [c for c in self.clients.values() if c.status == ClientStatus.TRAINING]
            ),
            "failed_clients": len(
                [c for c in self.clients.values() if c.status == ClientStatus.FAILED]
            ),
        }

        # Privacy statistics
        privacy_stats = {
            "clients_with_budget": len(
                [
                    c
                    for c in self.clients.values()
                    if not c.privacy_budget.is_exhausted()
                ]
            ),
            "average_privacy_budget": (
                np.mean(
                    [c.privacy_budget.remaining_epsilon for c in self.clients.values()]
                )
                if self.clients
                else 0.0
            ),
            "privacy_level": self.privacy_level.value,
        }

        # Training statistics
        training_stats = {
            "current_round": self.current_round,
            "total_rounds_completed": len(self.round_history),
            "average_participation_rate": (
                np.mean(
                    [
                        round_info.global_performance.get("participation_rate", 0.0)
                        for round_info in self.round_history
                    ]
                )
                if self.round_history
                else 0.0
            ),
            "convergence_status": (
                self.convergence_history[-1] if self.convergence_history else 0.0
            ),
        }

        # Performance trends
        performance_trends = {}
        for metric, values in self.performance_metrics.items():
            if values:
                performance_trends[metric] = {
                    "current": values[-1],
                    "average": np.mean(values),
                    "trend": (
                        "improving"
                        if len(values) > 1 and values[-1] > values[-2]
                        else "stable"
                    ),
                }

        return {
            "system_status": {
                "strategy": self.strategy.value,
                "privacy_level": self.privacy_level.value,
                "is_training": any(
                    c.status == ClientStatus.TRAINING for c in self.clients.values()
                ),
                "last_round_completed": (
                    self.round_history[-1].end_time.isoformat()
                    if self.round_history
                    else None
                ),
            },
            "client_statistics": client_stats,
            "privacy_statistics": privacy_stats,
            "training_statistics": training_stats,
            "performance_trends": performance_trends,
            "recommendations": self._generate_federated_recommendations(),
        }

    def _generate_federated_recommendations(self) -> list[str]:
        """Generate recommendations for federated learning optimization."""
        recommendations = []

        # Check client participation
        active_clients = len(
            [
                c
                for c in self.clients.values()
                if c.status in [ClientStatus.ACTIVE, ClientStatus.IDLE]
            ]
        )
        if active_clients < self.min_clients_per_round:
            recommendations.append(
                f"Recruit more clients - only {active_clients} active clients available"
            )

        # Check privacy budget utilization
        clients_with_budget = len(
            [c for c in self.clients.values() if not c.privacy_budget.is_exhausted()]
        )
        if clients_with_budget < len(self.clients) * 0.5:
            recommendations.append(
                "Many clients have exhausted privacy budgets - consider budget reallocation"
            )

        # Check convergence
        if self.convergence_history and len(self.convergence_history) > 5:
            recent_improvement = (
                self.convergence_history[-1] - self.convergence_history[-5]
            )
            if recent_improvement < 0.01:
                recommendations.append(
                    "Model convergence slowing - consider adjusting learning parameters"
                )

        # Check client diversity
        if self.round_history:
            latest_round = self.round_history[-1]
            diversity = latest_round.global_performance.get("client_diversity", 0.0)
            if diversity < 0.1:
                recommendations.append(
                    "Low client diversity detected - consider diverse client selection"
                )

        if not recommendations:
            recommendations.append("Federated learning system is operating optimally")

        return recommendations
