"""Configuration Discovery and Search Service.

This service provides advanced search, discovery, and organization capabilities
for configuration management with intelligent tagging, similarity analysis,
and recommendation systems.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# TODO: Create local configuration DTOs
# TODO: Create local configuration service
# TODO: Create local feature flags
from monorepo.infrastructure.persistence.configuration_repository import (
    ConfigurationRepository,
)

logger = logging.getLogger(__name__)


class ConfigurationDiscoveryService:
    """Service for advanced configuration discovery and organization."""

    def __init__(
        self,
        configuration_service: ConfigurationCaptureService,
        repository: ConfigurationRepository,
        enable_similarity_analysis: bool = True,
        enable_auto_tagging: bool = True,
        enable_clustering: bool = True,
    ):
        """Initialize configuration discovery service.

        Args:
            configuration_service: Configuration capture service
            repository: Configuration repository
            enable_similarity_analysis: Enable similarity-based recommendations
            enable_auto_tagging: Enable automatic tagging
            enable_clustering: Enable configuration clustering
        """
        self.configuration_service = configuration_service
        self.repository = repository
        self.enable_similarity_analysis = enable_similarity_analysis
        self.enable_auto_tagging = enable_auto_tagging
        self.enable_clustering = enable_clustering

        # Discovery statistics
        self.discovery_stats = {
            "total_searches": 0,
            "similarity_searches": 0,
            "tag_based_searches": 0,
            "cluster_based_searches": 0,
            "auto_tags_generated": 0,
            "collections_created": 0,
            "recommendations_generated": 0,
        }

        # Caching for performance
        self._configuration_cache: dict[UUID, ExperimentConfigurationDTO] = {}
        self._similarity_cache: dict[tuple[UUID, UUID], float] = {}
        self._tag_index: dict[str, set[UUID]] = defaultdict(set)
        self._clusters: dict[int, list[UUID]] | None = None
        self._last_cache_update = datetime.min

        # TF-IDF vectorizer for text similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )

    @require_feature("advanced_automl")
    async def advanced_search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        similarity_search: bool = True,
        include_recommendations: bool = True,
        max_results: int = 50,
    ) -> ConfigurationResponseDTO:
        """Perform advanced configuration search with multiple strategies.

        Args:
            query: Search query (text, algorithm name, or description)
            filters: Additional filters (source, tags, performance, etc.)
            similarity_search: Include similarity-based results
            include_recommendations: Include recommended configurations
            max_results: Maximum number of results

        Returns:
            Advanced search results with relevance scoring
        """
        self.discovery_stats["total_searches"] += 1

        logger.info(f"Performing advanced search for query: '{query}'")

        # Ensure cache is up to date
        await self._update_cache_if_needed()

        # Perform multiple search strategies
        search_results = []

        # 1. Traditional keyword search
        keyword_results = await self._keyword_search(query, filters, max_results // 2)
        search_results.extend(keyword_results)

        # 2. Similarity-based search
        if similarity_search and self.enable_similarity_analysis:
            self.discovery_stats["similarity_searches"] += 1
            similarity_results = await self._similarity_search(
                query, filters, max_results // 4
            )
            search_results.extend(similarity_results)

        # 3. Tag-based search
        if self.enable_auto_tagging:
            self.discovery_stats["tag_based_searches"] += 1
            tag_results = await self._tag_based_search(query, filters, max_results // 4)
            search_results.extend(tag_results)

        # Remove duplicates and score results
        unique_results = self._deduplicate_and_score_results(search_results, query)

        # Sort by relevance score
        unique_results.sort(key=lambda x: x[1], reverse=True)

        # Extract configurations
        configurations = [result[0] for result in unique_results[:max_results]]

        # Add recommendations if requested
        recommendations = []
        if include_recommendations and configurations:
            self.discovery_stats["recommendations_generated"] += 1
            recommendations = await self._generate_search_recommendations(
                configurations, query
            )

        logger.info(
            f"Advanced search completed: {len(configurations)} results, {len(recommendations)} recommendations"
        )

        return ConfigurationResponseDTO(
            success=True,
            message=f"Found {len(configurations)} configurations",
            configurations=configurations,
            recommendations=recommendations,
            total_count=len(configurations),
            search_metadata={
                "query": query,
                "strategies_used": ["keyword", "similarity", "tags"],
                "relevance_scored": True,
                "recommendations_included": include_recommendations,
            },
        )

    async def find_similar_configurations(
        self,
        reference_config_id: UUID,
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        include_performance_similarity: bool = True,
    ) -> list[tuple[ExperimentConfigurationDTO, float]]:
        """Find configurations similar to a reference configuration.

        Args:
            reference_config_id: Reference configuration ID
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of similar configurations
            include_performance_similarity: Include performance in similarity calculation

        Returns:
            List of (configuration, similarity_score) tuples
        """
        await self._update_cache_if_needed()

        reference_config = self._configuration_cache.get(reference_config_id)
        if not reference_config:
            reference_config = await self.repository.load_configuration(
                reference_config_id
            )
            if not reference_config:
                return []

        logger.info(f"Finding configurations similar to {reference_config.name}")

        similar_configs = []

        for config_id, config in self._configuration_cache.items():
            if config_id == reference_config_id:
                continue

            # Calculate similarity
            similarity_score = await self._calculate_configuration_similarity(
                reference_config, config, include_performance_similarity
            )

            if similarity_score >= similarity_threshold:
                similar_configs.append((config, similarity_score))

        # Sort by similarity score
        similar_configs.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Found {len(similar_configs)} similar configurations")
        return similar_configs[:max_results]

    async def auto_generate_tags(
        self,
        config_id: UUID | None = None,
        batch_process: bool = False,
        overwrite_existing: bool = False,
    ) -> dict[UUID, list[str]]:
        """Automatically generate tags for configurations.

        Args:
            config_id: Specific configuration ID (None for all)
            batch_process: Process all configurations
            overwrite_existing: Overwrite existing tags

        Returns:
            Dictionary mapping configuration IDs to generated tags
        """
        if not self.enable_auto_tagging:
            return {}

        await self._update_cache_if_needed()

        generated_tags = {}

        configs_to_process = []
        if config_id:
            config = self._configuration_cache.get(config_id)
            if config:
                configs_to_process = [config]
        elif batch_process:
            configs_to_process = list(self._configuration_cache.values())

        logger.info(
            f"Auto-generating tags for {len(configs_to_process)} configurations"
        )

        for config in configs_to_process:
            if not overwrite_existing and config.metadata.tags:
                continue  # Skip if tags already exist

            # Generate tags based on configuration characteristics
            tags = self._generate_tags_for_configuration(config)

            if tags:
                # Update configuration
                config.metadata.tags.extend(tags)
                config.metadata.tags = list(
                    set(config.metadata.tags)
                )  # Remove duplicates

                # Save updated configuration
                await self.repository.save_configuration(config)

                generated_tags[config.id] = tags
                self.discovery_stats["auto_tags_generated"] += len(tags)

        # Update tag index
        await self._rebuild_tag_index()

        logger.info(f"Generated tags for {len(generated_tags)} configurations")
        return generated_tags

    async def create_smart_collection(
        self,
        collection_name: str,
        description: str,
        criteria: dict[str, Any],
        auto_update: bool = True,
    ) -> ConfigurationCollectionDTO:
        """Create a smart collection based on criteria.

        Args:
            collection_name: Name for the collection
            description: Collection description
            criteria: Collection criteria (algorithm, performance, tags, etc.)
            auto_update: Automatically update collection with new configurations

        Returns:
            Created configuration collection
        """
        logger.info(f"Creating smart collection '{collection_name}'")

        # Find configurations matching criteria
        matching_configs = await self._find_configurations_by_criteria(criteria)

        # Select featured configuration (best performing)
        featured_config = None
        if matching_configs:
            featured_config = max(
                matching_configs,
                key=lambda c: (
                    c.performance_results.accuracy
                    if c.performance_results and c.performance_results.accuracy
                    else 0
                ),
            ).id

        # Create collection
        collection = ConfigurationCollectionDTO(
            id=UUID.hex,
            name=collection_name,
            description=description,
            configurations=[config.id for config in matching_configs],
            featured_configuration=featured_config,
            tags=self._extract_common_tags(matching_configs),
            category=self._determine_collection_category(criteria),
            created_by="auto_discovery",
            is_public=False,
            total_configurations=len(matching_configs),
            success_rate=self._calculate_collection_success_rate(matching_configs),
            average_performance=self._calculate_collection_average_performance(
                matching_configs
            ),
        )

        # Save collection
        await self.repository.save_collection(collection)

        self.discovery_stats["collections_created"] += 1

        logger.info(
            f"Smart collection '{collection_name}' created with {len(matching_configs)} configurations"
        )
        return collection

    async def cluster_configurations(
        self,
        n_clusters: int | None = None,
        feature_weights: dict[str, float] | None = None,
    ) -> dict[int, list[UUID]]:
        """Cluster configurations based on similarity.

        Args:
            n_clusters: Number of clusters (auto-determined if None)
            feature_weights: Weights for different features in clustering

        Returns:
            Dictionary mapping cluster IDs to configuration IDs
        """
        if not self.enable_clustering:
            return {}

        await self._update_cache_if_needed()

        configurations = list(self._configuration_cache.values())
        if len(configurations) < 3:
            return {}

        logger.info(f"Clustering {len(configurations)} configurations")

        # Extract features for clustering
        feature_matrix = self._extract_clustering_features(
            configurations, feature_weights
        )

        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters = min(max(2, len(configurations) // 5), 10)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)

        # Organize results
        clusters = defaultdict(list)
        for i, config in enumerate(configurations):
            cluster_id = int(cluster_labels[i])
            clusters[cluster_id].append(config.id)

        # Cache clusters
        self._clusters = dict(clusters)

        logger.info(f"Configurations clustered into {len(clusters)} groups")
        return self._clusters

    async def get_configuration_trends(
        self, time_period_days: int = 30, group_by: str = "algorithm"
    ) -> dict[str, Any]:
        """Analyze configuration trends over time.

        Args:
            time_period_days: Time period to analyze
            group_by: Grouping criteria (algorithm, source, performance_tier)

        Returns:
            Trend analysis results
        """
        await self._update_cache_if_needed()

        # Filter configurations by time period
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        recent_configs = [
            config
            for config in self._configuration_cache.values()
            if config.metadata.created_at >= cutoff_date
        ]

        logger.info(
            f"Analyzing trends for {len(recent_configs)} configurations over {time_period_days} days"
        )

        trends = {
            "time_period_days": time_period_days,
            "total_configurations": len(recent_configs),
            "group_by": group_by,
            "trends": {},
            "summary": {},
        }

        # Group configurations
        groups = defaultdict(list)
        for config in recent_configs:
            if group_by == "algorithm":
                key = config.algorithm_config.algorithm_name
            elif group_by == "source":
                key = config.metadata.source
            elif group_by == "performance_tier":
                if config.performance_results and config.performance_results.accuracy:
                    if config.performance_results.accuracy >= 0.9:
                        key = "high_performance"
                    elif config.performance_results.accuracy >= 0.7:
                        key = "medium_performance"
                    else:
                        key = "low_performance"
                else:
                    key = "unknown_performance"
            else:
                key = "other"

            groups[key].append(config)

        # Analyze trends for each group
        for group_name, group_configs in groups.items():
            # Time series analysis
            daily_counts = defaultdict(int)
            for config in group_configs:
                day = config.metadata.created_at.date()
                daily_counts[day] += 1

            # Performance trends
            performance_trend = []
            if group_configs and group_configs[0].performance_results:
                sorted_configs = sorted(
                    group_configs, key=lambda c: c.metadata.created_at
                )
                for config in sorted_configs:
                    if (
                        config.performance_results
                        and config.performance_results.accuracy
                    ):
                        performance_trend.append(config.performance_results.accuracy)

            trends["trends"][group_name] = {
                "count": len(group_configs),
                "daily_counts": dict(daily_counts),
                "performance_trend": performance_trend,
                "average_performance": (
                    sum(performance_trend) / len(performance_trend)
                    if performance_trend
                    else 0
                ),
                "latest_configs": [
                    c.id
                    for c in sorted(
                        group_configs, key=lambda c: c.metadata.created_at, reverse=True
                    )[:5]
                ],
            }

        # Summary statistics
        trends["summary"] = {
            "most_active_group": (
                max(groups.keys(), key=lambda k: len(groups[k])) if groups else None
            ),
            "total_groups": len(groups),
            "average_group_size": (
                sum(len(configs) for configs in groups.values()) / len(groups)
                if groups
                else 0
            ),
            "trend_direction": self._calculate_trend_direction(recent_configs),
        }

        return trends

    def get_discovery_statistics(self) -> dict[str, Any]:
        """Get discovery service statistics.

        Returns:
            Discovery statistics dictionary
        """
        return {
            "discovery_stats": self.discovery_stats,
            "cache_info": {
                "cached_configurations": len(self._configuration_cache),
                "cached_similarities": len(self._similarity_cache),
                "tag_index_size": len(self._tag_index),
                "clusters_available": self._clusters is not None,
                "last_cache_update": self._last_cache_update.isoformat(),
            },
            "feature_status": {
                "similarity_analysis": self.enable_similarity_analysis,
                "auto_tagging": self.enable_auto_tagging,
                "clustering": self.enable_clustering,
            },
        }

    # Private methods

    async def _update_cache_if_needed(self) -> None:
        """Update cache if it's stale."""
        now = datetime.now()
        if (now - self._last_cache_update).total_seconds() > 300:  # 5 minutes
            await self._rebuild_cache()

    async def _rebuild_cache(self) -> None:
        """Rebuild configuration cache."""
        logger.debug("Rebuilding configuration cache")

        # Get all configurations
        search_request = ConfigurationSearchRequestDTO(limit=10000)
        configurations = await self.repository.search_configurations(search_request)

        # Update cache
        self._configuration_cache = {config.id: config for config in configurations}

        # Rebuild tag index
        await self._rebuild_tag_index()

        self._last_cache_update = datetime.now()
        logger.debug(
            f"Cache rebuilt with {len(self._configuration_cache)} configurations"
        )

    async def _rebuild_tag_index(self) -> None:
        """Rebuild tag index for fast tag-based searches."""
        self._tag_index = defaultdict(set)

        for config_id, config in self._configuration_cache.items():
            for tag in config.metadata.tags:
                self._tag_index[tag.lower()].add(config_id)

    async def _keyword_search(
        self, query: str, filters: dict[str, Any] | None, max_results: int
    ) -> list[tuple[ExperimentConfigurationDTO, float]]:
        """Perform traditional keyword search."""
        # Create search request
        search_request = ConfigurationSearchRequestDTO(
            query=query, limit=max_results, sort_by="created_at", sort_order="desc"
        )

        # Apply filters
        if filters:
            if "source" in filters:
                search_request.source = filters["source"]
            if "algorithm" in filters:
                search_request.algorithm = filters["algorithm"]
            if "tags" in filters:
                search_request.tags = filters["tags"]
            if "min_accuracy" in filters:
                search_request.min_accuracy = filters["min_accuracy"]

        # Execute search
        results = await self.repository.search_configurations(search_request)

        # Score results based on query relevance
        scored_results = []
        query_lower = query.lower()

        for config in results:
            score = 0.0

            # Name match
            if query_lower in config.name.lower():
                score += 0.4

            # Algorithm match
            if query_lower in config.algorithm_config.algorithm_name.lower():
                score += 0.3

            # Description match
            if (
                config.metadata.description
                and query_lower in config.metadata.description.lower()
            ):
                score += 0.2

            # Tag match
            for tag in config.metadata.tags:
                if query_lower in tag.lower():
                    score += 0.1
                    break

            scored_results.append((config, score))

        return scored_results

    async def _similarity_search(
        self, query: str, filters: dict[str, Any] | None, max_results: int
    ) -> list[tuple[ExperimentConfigurationDTO, float]]:
        """Perform similarity-based search."""
        # Find configurations that match query keywords first
        keyword_matches = await self._keyword_search(query, filters, max_results * 2)

        if not keyword_matches:
            return []

        # Use best keyword match as reference for similarity
        reference_config = keyword_matches[0][0]

        # Find similar configurations
        similar_configs = await self.find_similar_configurations(
            reference_config.id, similarity_threshold=0.5, max_results=max_results
        )

        return similar_configs

    async def _tag_based_search(
        self, query: str, filters: dict[str, Any] | None, max_results: int
    ) -> list[tuple[ExperimentConfigurationDTO, float]]:
        """Perform tag-based search."""
        query_lower = query.lower()

        # Find tags that match the query
        matching_tags = [tag for tag in self._tag_index.keys() if query_lower in tag]

        # Get configurations with matching tags
        config_ids = set()
        for tag in matching_tags:
            config_ids.update(self._tag_index[tag])

        # Score based on tag relevance
        scored_results = []
        for config_id in list(config_ids)[:max_results]:
            config = self._configuration_cache.get(config_id)
            if config:
                # Calculate tag relevance score
                score = 0.0
                for tag in config.metadata.tags:
                    if query_lower in tag.lower():
                        score += 1.0 / len(
                            config.metadata.tags
                        )  # Normalize by number of tags

                scored_results.append((config, score))

        return scored_results

    def _deduplicate_and_score_results(
        self, results: list[tuple[ExperimentConfigurationDTO, float]], query: str
    ) -> list[tuple[ExperimentConfigurationDTO, float]]:
        """Remove duplicates and calculate final relevance scores."""
        # Group by configuration ID
        config_scores = defaultdict(list)
        for config, score in results:
            config_scores[config.id].append((config, score))

        # Calculate final scores (average of all search strategy scores)
        final_results = []
        for _config_id, config_score_list in config_scores.items():
            config = config_score_list[0][0]  # All configs are the same
            scores = [score for _, score in config_score_list]

            # Weighted average with boost for multiple strategy matches
            final_score = sum(scores) / len(scores)
            if len(scores) > 1:
                final_score *= 1.2  # Boost for multi-strategy matches

            final_results.append((config, final_score))

        return final_results

    async def _generate_search_recommendations(
        self, search_results: list[ExperimentConfigurationDTO], query: str
    ) -> list[dict[str, Any]]:
        """Generate recommendations based on search results."""
        recommendations = []

        if not search_results:
            return recommendations

        # Recommend similar algorithms
        algorithms = Counter(
            config.algorithm_config.algorithm_name for config in search_results
        )
        most_common_algorithm = algorithms.most_common(1)[0][0]

        recommendations.append(
            {
                "type": "algorithm_suggestion",
                "title": f"Consider {most_common_algorithm}",
                "description": f"{most_common_algorithm} appears in {algorithms[most_common_algorithm]} of your search results",
                "relevance": 0.8,
            }
        )

        # Recommend high-performing configurations
        high_perf_configs = [
            config
            for config in search_results
            if config.performance_results
            and config.performance_results.accuracy
            and config.performance_results.accuracy > 0.85
        ]

        if high_perf_configs:
            recommendations.append(
                {
                    "type": "performance_suggestion",
                    "title": "High-Performance Configurations Available",
                    "description": f"Found {len(high_perf_configs)} configurations with >85% accuracy",
                    "configurations": [config.id for config in high_perf_configs[:3]],
                    "relevance": 0.9,
                }
            )

        # Recommend collections
        common_tags = self._find_common_tags(search_results)
        if common_tags:
            recommendations.append(
                {
                    "type": "collection_suggestion",
                    "title": "Create Collection",
                    "description": f"Your search results share common themes: {', '.join(common_tags[:3])}",
                    "suggested_tags": common_tags,
                    "relevance": 0.6,
                }
            )

        return recommendations

    async def _calculate_configuration_similarity(
        self,
        config1: ExperimentConfigurationDTO,
        config2: ExperimentConfigurationDTO,
        include_performance: bool = True,
    ) -> float:
        """Calculate similarity between two configurations."""
        # Check cache first
        cache_key = (config1.id, config2.id)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        similarity_score = 0.0

        # Algorithm similarity
        if (
            config1.algorithm_config.algorithm_name
            == config2.algorithm_config.algorithm_name
        ):
            similarity_score += 0.3

        # Hyperparameter similarity
        if (
            config1.algorithm_config.hyperparameters
            and config2.algorithm_config.hyperparameters
        ):
            param_similarity = self._calculate_parameter_similarity(
                config1.algorithm_config.hyperparameters,
                config2.algorithm_config.hyperparameters,
            )
            similarity_score += param_similarity * 0.2

        # Tag similarity
        tags1 = set(config1.metadata.tags)
        tags2 = set(config2.metadata.tags)
        if tags1 and tags2:
            tag_similarity = len(tags1.intersection(tags2)) / len(tags1.union(tags2))
            similarity_score += tag_similarity * 0.2

        # Source similarity
        if config1.metadata.source == config2.metadata.source:
            similarity_score += 0.1

        # Performance similarity
        if (
            include_performance
            and config1.performance_results
            and config2.performance_results
        ):
            if (
                config1.performance_results.accuracy
                and config2.performance_results.accuracy
            ):
                perf_diff = abs(
                    config1.performance_results.accuracy
                    - config2.performance_results.accuracy
                )
                perf_similarity = 1.0 - perf_diff
                similarity_score += perf_similarity * 0.2

        # Cache result
        self._similarity_cache[cache_key] = similarity_score
        self._similarity_cache[(config2.id, config1.id)] = similarity_score  # Symmetric

        return similarity_score

    def _calculate_parameter_similarity(
        self, params1: dict[str, Any], params2: dict[str, Any]
    ) -> float:
        """Calculate similarity between parameter dictionaries."""
        all_keys = set(params1.keys()).union(set(params2.keys()))
        if not all_keys:
            return 1.0

        similarities = []
        for key in all_keys:
            if key in params1 and key in params2:
                val1, val2 = params1[key], params2[key]

                if isinstance(val1, int | float) and isinstance(val2, int | float):
                    # Numeric similarity
                    max_val = max(abs(val1), abs(val2), 1)  # Avoid division by zero
                    similarity = 1.0 - abs(val1 - val2) / max_val
                    similarities.append(similarity)
                elif val1 == val2:
                    # Exact match
                    similarities.append(1.0)
                else:
                    # Different values
                    similarities.append(0.0)
            else:
                # Missing parameter
                similarities.append(0.0)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _generate_tags_for_configuration(
        self, config: ExperimentConfigurationDTO
    ) -> list[str]:
        """Generate tags for a configuration based on its characteristics."""
        tags = []

        # Algorithm-based tags
        algorithm = config.algorithm_config.algorithm_name.lower()
        tags.append(algorithm.replace("_", "-"))

        if "isolation" in algorithm:
            tags.extend(["tree-based", "ensemble"])
        elif "lof" in algorithm or "outlier" in algorithm:
            tags.extend(["density-based", "neighbor-based"])
        elif "svm" in algorithm:
            tags.extend(["kernel-based", "support-vector"])
        elif "autoencoder" in algorithm or "neural" in algorithm:
            tags.extend(["deep-learning", "neural-network"])

        # Performance-based tags
        if config.performance_results and config.performance_results.accuracy:
            accuracy = config.performance_results.accuracy
            if accuracy >= 0.9:
                tags.append("high-accuracy")
            elif accuracy >= 0.8:
                tags.append("good-accuracy")
            elif accuracy >= 0.7:
                tags.append("moderate-accuracy")
            else:
                tags.append("low-accuracy")

            if config.performance_results.training_time_seconds:
                training_time = config.performance_results.training_time_seconds
                if training_time < 30:
                    tags.append("fast-training")
                elif training_time < 120:
                    tags.append("moderate-training")
                else:
                    tags.append("slow-training")

        # Source-based tags
        source = config.metadata.source
        if source == ConfigurationSource.AUTOML:
            tags.extend(["automl", "optimized"])
        elif source == ConfigurationSource.AUTONOMOUS:
            tags.extend(["autonomous", "auto-configured"])
        elif source == ConfigurationSource.CLI:
            tags.extend(["cli", "manual"])

        # Complexity-based tags
        if config.algorithm_config.hyperparameters:
            param_count = len(config.algorithm_config.hyperparameters)
            if param_count > 10:
                tags.append("complex-config")
            elif param_count > 5:
                tags.append("moderate-config")
            else:
                tags.append("simple-config")

        # Preprocessing tags
        if config.preprocessing_config:
            tags.append("preprocessed")
            if config.preprocessing_config.apply_pca:
                tags.append("pca")
            if config.preprocessing_config.scaling_method:
                tags.append(f"{config.preprocessing_config.scaling_method}-scaling")

        return tags

    async def _find_configurations_by_criteria(
        self, criteria: dict[str, Any]
    ) -> list[ExperimentConfigurationDTO]:
        """Find configurations matching specific criteria."""
        # Convert criteria to search request
        search_request = ConfigurationSearchRequestDTO(
            limit=1000,  # Large limit for collection creation
            sort_by="accuracy",
            sort_order="desc",
        )

        # Apply criteria
        if "algorithm" in criteria:
            search_request.algorithm = criteria["algorithm"]
        if "source" in criteria:
            search_request.source = criteria["source"]
        if "tags" in criteria:
            search_request.tags = criteria["tags"]
        if "min_accuracy" in criteria:
            search_request.min_accuracy = criteria["min_accuracy"]
        if "max_training_time" in criteria:
            # This would need custom filtering in the repository
            pass

        return await self.repository.search_configurations(search_request)

    def _extract_common_tags(
        self, configurations: list[ExperimentConfigurationDTO]
    ) -> list[str]:
        """Extract common tags from a list of configurations."""
        if not configurations:
            return []

        # Count tag occurrences
        tag_counts = Counter()
        for config in configurations:
            tag_counts.update(config.metadata.tags)

        # Return tags that appear in at least 25% of configurations
        min_count = max(1, len(configurations) // 4)
        common_tags = [tag for tag, count in tag_counts.items() if count >= min_count]

        return common_tags[:10]  # Limit to top 10

    def _find_common_tags(
        self, configurations: list[ExperimentConfigurationDTO]
    ) -> list[str]:
        """Find common tags across configurations (alias for _extract_common_tags)."""
        return self._extract_common_tags(configurations)

    def _determine_collection_category(self, criteria: dict[str, Any]) -> str | None:
        """Determine category for a collection based on criteria."""
        if "algorithm" in criteria:
            return "algorithm-specific"
        elif "source" in criteria:
            return "source-based"
        elif "min_accuracy" in criteria:
            return "performance-based"
        elif "tags" in criteria:
            return "tag-based"
        else:
            return "mixed-criteria"

    def _calculate_collection_success_rate(
        self, configurations: list[ExperimentConfigurationDTO]
    ) -> float | None:
        """Calculate success rate for a collection."""
        if not configurations:
            return None

        # Define "success" as accuracy > 0.7
        successful_configs = [
            config
            for config in configurations
            if config.performance_results
            and config.performance_results.accuracy
            and config.performance_results.accuracy > 0.7
        ]

        return len(successful_configs) / len(configurations)

    def _calculate_collection_average_performance(
        self, configurations: list[ExperimentConfigurationDTO]
    ) -> float | None:
        """Calculate average performance for a collection."""
        accuracies = [
            config.performance_results.accuracy
            for config in configurations
            if config.performance_results and config.performance_results.accuracy
        ]

        if not accuracies:
            return None

        return sum(accuracies) / len(accuracies)

    def _extract_clustering_features(
        self,
        configurations: list[ExperimentConfigurationDTO],
        feature_weights: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Extract features for clustering configurations."""
        feature_weights = feature_weights or {
            "algorithm": 1.0,
            "performance": 0.8,
            "hyperparameters": 0.6,
            "source": 0.4,
            "tags": 0.3,
        }

        features = []

        # Get all unique algorithms
        all_algorithms = list(
            {config.algorithm_config.algorithm_name for config in configurations}
        )
        algorithm_to_idx = {algo: i for i, algo in enumerate(all_algorithms)}

        # Get all unique sources
        all_sources = list({config.metadata.source for config in configurations})
        source_to_idx = {source: i for i, source in enumerate(all_sources)}

        # Get all unique tags
        all_tags = list(
            {tag for config in configurations for tag in config.metadata.tags}
        )
        tag_to_idx = {tag: i for i, tag in enumerate(all_tags)}

        for config in configurations:
            feature_vector = []

            # Algorithm features (one-hot encoding)
            algo_features = [0.0] * len(all_algorithms)
            algo_idx = algorithm_to_idx[config.algorithm_config.algorithm_name]
            algo_features[algo_idx] = feature_weights["algorithm"]
            feature_vector.extend(algo_features)

            # Performance features
            if config.performance_results and config.performance_results.accuracy:
                feature_vector.append(
                    config.performance_results.accuracy * feature_weights["performance"]
                )
            else:
                feature_vector.append(0.0)

            # Hyperparameter count (normalized)
            param_count = (
                len(config.algorithm_config.hyperparameters)
                if config.algorithm_config.hyperparameters
                else 0
            )
            feature_vector.append(
                (param_count / 20.0) * feature_weights["hyperparameters"]
            )  # Normalize by max expected params

            # Source features (one-hot encoding)
            source_features = [0.0] * len(all_sources)
            source_idx = source_to_idx[config.metadata.source]
            source_features[source_idx] = feature_weights["source"]
            feature_vector.extend(source_features)

            # Tag features (binary encoding for common tags)
            tag_features = [0.0] * min(len(all_tags), 20)  # Limit to top 20 tags
            for tag in config.metadata.tags[:20]:
                if tag in tag_to_idx and tag_to_idx[tag] < len(tag_features):
                    tag_features[tag_to_idx[tag]] = feature_weights["tags"]
            feature_vector.extend(tag_features)

            features.append(feature_vector)

        return np.array(features)

    def _calculate_trend_direction(
        self, configurations: list[ExperimentConfigurationDTO]
    ) -> str:
        """Calculate overall trend direction for configurations."""
        if len(configurations) < 2:
            return "stable"

        # Sort by creation date
        sorted_configs = sorted(configurations, key=lambda c: c.metadata.created_at)

        # Compare first half vs second half
        mid_point = len(sorted_configs) // 2
        first_half = sorted_configs[:mid_point]
        second_half = sorted_configs[mid_point:]

        first_half_count = len(first_half)
        second_half_count = len(second_half)

        if second_half_count > first_half_count * 1.2:
            return "increasing"
        elif second_half_count < first_half_count * 0.8:
            return "decreasing"
        else:
            return "stable"
