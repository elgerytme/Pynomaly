#!/usr/bin/env python3
"""
Comprehensive PyGOD Graph Anomaly Detection Example

This example demonstrates the complete PyGOD integration in Pynomaly,
including installation verification, data preparation, model training,
and result analysis.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_pygod_availability() -> bool:
    """Check if PyGOD dependencies are available."""
    try:
        import pygod
        import torch
        import torch_geometric

        logger.info("✅ All PyGOD dependencies are available")
        return True
    except ImportError as e:
        logger.warning(f"❌ PyGOD dependencies not available: {e}")
        logger.info("Install with: pip install 'pynomaly[graph]'")
        return False


def create_sample_social_network_data() -> pd.DataFrame:
    """Create a synthetic social network dataset for demonstration."""
    np.random.seed(42)

    # Create a social network with 100 users
    n_users = 100
    n_edges = 300

    # Generate user features
    user_data = []
    for user_id in range(n_users):
        # Normal users have certain patterns
        if user_id < 90:  # 90% normal users
            account_age = np.random.normal(365, 180)  # ~1 year old accounts
            post_frequency = np.random.exponential(2)  # Posts per day
            engagement_rate = np.random.beta(2, 5)  # Like rate per post
            follower_count = np.random.lognormal(3, 1)  # Log-normal distribution
        else:  # 10% suspicious users (bots/fake accounts)
            account_age = np.random.uniform(1, 30)  # Very new accounts
            post_frequency = np.random.uniform(10, 50)  # High posting frequency
            engagement_rate = np.random.uniform(0, 0.1)  # Low engagement
            follower_count = np.random.uniform(1000, 5000)  # Many followers

        user_data.append(
            {
                "user_id": user_id,
                "account_age_days": max(1, account_age),
                "posts_per_day": max(0, post_frequency),
                "engagement_rate": np.clip(engagement_rate, 0, 1),
                "follower_count": max(1, follower_count),
                "is_verified": np.random.choice([0, 1], p=[0.95, 0.05]),
            }
        )

    # Generate social connections (edges)
    edges = []
    for _ in range(n_edges):
        source = np.random.randint(0, n_users)
        target = np.random.randint(0, n_users)
        if source != target:  # No self-loops
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "interaction_weight": np.random.exponential(1),
                }
            )

    # Combine user features with edges
    user_df = pd.DataFrame(user_data)
    edges_df = pd.DataFrame(edges)

    # Merge to create final dataset
    # For each edge, we'll include the source user's features
    merged_data = edges_df.merge(
        user_df.add_prefix("source_"),
        left_on="source",
        right_on="source_user_id",
        how="left",
    )

    # Keep only necessary columns
    final_data = merged_data[
        [
            "source",
            "target",
            "interaction_weight",
            "source_account_age_days",
            "source_posts_per_day",
            "source_engagement_rate",
            "source_follower_count",
            "source_is_verified",
        ]
    ].rename(
        columns={
            "source_account_age_days": "account_age_days",
            "source_posts_per_day": "posts_per_day",
            "source_engagement_rate": "engagement_rate",
            "source_follower_count": "follower_count",
            "source_is_verified": "is_verified",
        }
    )

    logger.info(
        f"Created social network with {len(final_data)} interactions between {n_users} users"
    )
    return final_data


def create_sample_citation_network_data() -> pd.DataFrame:
    """Create a synthetic citation network dataset."""
    np.random.seed(123)

    n_papers = 200
    n_citations = 400

    # Generate paper features
    papers = []
    for paper_id in range(n_papers):
        # Most papers are normal
        if paper_id < 180:  # 90% normal papers
            author_count = np.random.poisson(3) + 1
            citation_velocity = np.random.exponential(0.5)
            journal_impact = np.random.normal(2, 0.5)
            self_citation_rate = np.random.beta(1, 9)  # Low self-citation
        else:  # 10% suspicious papers
            author_count = np.random.poisson(8) + 1  # Many authors
            citation_velocity = np.random.exponential(3)  # High citation rate
            journal_impact = np.random.uniform(0.5, 1.5)  # Lower impact journals
            self_citation_rate = np.random.beta(3, 2)  # High self-citation

        papers.append(
            {
                "paper_id": paper_id,
                "author_count": author_count,
                "citation_velocity": citation_velocity,
                "journal_impact_factor": max(0.1, journal_impact),
                "self_citation_rate": np.clip(self_citation_rate, 0, 1),
                "years_since_publication": np.random.exponential(3) + 0.1,
            }
        )

    # Generate citations (edges)
    citations = []
    for _ in range(n_citations):
        citing_paper = np.random.randint(0, n_papers)
        cited_paper = np.random.randint(0, n_papers)
        if citing_paper != cited_paper:
            citations.append(
                {
                    "source": citing_paper,  # citing paper
                    "target": cited_paper,  # cited paper
                    "citation_context_relevance": np.random.uniform(0.3, 1.0),
                }
            )

    # Merge paper features with citations
    papers_df = pd.DataFrame(papers)
    citations_df = pd.DataFrame(citations)

    merged_data = citations_df.merge(
        papers_df.add_prefix("citing_"),
        left_on="source",
        right_on="citing_paper_id",
        how="left",
    )

    final_data = merged_data[
        [
            "source",
            "target",
            "citation_context_relevance",
            "citing_author_count",
            "citing_citation_velocity",
            "citing_journal_impact_factor",
            "citing_self_citation_rate",
            "citing_years_since_publication",
        ]
    ].rename(
        columns={
            "citing_author_count": "author_count",
            "citing_citation_velocity": "citation_velocity",
            "citing_journal_impact_factor": "journal_impact_factor",
            "citing_self_citation_rate": "self_citation_rate",
            "citing_years_since_publication": "years_since_publication",
        }
    )

    logger.info(
        f"Created citation network with {len(final_data)} citations between {n_papers} papers"
    )
    return final_data


def run_pygod_example_with_mocks():
    """Run PyGOD example using mocks when dependencies are not available."""
    from unittest.mock import MagicMock, patch

    from pynomaly_detection.domain.entities import Dataset
    from pynomaly_detection.domain.value_objects import ContaminationRate
    from pynomaly_detection.infrastructure.adapters.pygod_adapter import PyGODAdapter

    logger.info("🔧 Running PyGOD example with mocked dependencies")

    # Create sample data
    data = create_sample_social_network_data()

    # Create dataset
    dataset = Dataset(
        id="mock_social_network",
        name="Mock Social Network Dataset",
        data=data,
        metadata={
            "is_graph": True,
            "edge_columns": ["source", "target"],
            "weight_column": "interaction_weight",
            "feature_columns": [
                "account_age_days",
                "posts_per_day",
                "engagement_rate",
                "follower_count",
                "is_verified",
            ],
        },
    )

    # Mock PyGOD dependencies
    with patch.object(PyGODAdapter, "_get_algorithm_map") as mock_map:
        mock_algorithm_class = MagicMock()
        mock_model = MagicMock()

        # Configure mock algorithm
        mock_algorithm_class.return_value = mock_model
        mock_map.return_value = {
            "DOMINANT": mock_algorithm_class,
            "SCAN": mock_algorithm_class,
        }

        # Mock model behavior
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.array([0] * 295 + [1] * 5)  # 5 anomalies
        mock_model.decision_function.return_value = np.random.uniform(0, 1, 300)

        # Test different algorithms
        algorithms_to_test = [
            ("DOMINANT", {"hidden_dim": 64, "num_layers": 2, "epoch": 100}),
            ("SCAN", {"eps": 0.5, "mu": 2}),
        ]

        results = {}

        for algo_name, params in algorithms_to_test:
            logger.info(f"Testing {algo_name} algorithm...")

            # Create adapter
            adapter = PyGODAdapter(
                algorithm_name=algo_name,
                contamination_rate=ContaminationRate(0.1),
                **params,
            )

            # Verify adapter properties
            logger.info(f"  - Name: {adapter.name}")
            logger.info(f"  - Algorithm: {adapter.algorithm_name}")
            logger.info(f"  - Contamination rate: {adapter.contamination_rate}")
            logger.info(f"  - Parameters: {adapter.parameters}")

            # Mock the model attribute
            adapter._model = mock_model

            try:
                # Test training
                logger.info(f"  - Training {algo_name}...")
                adapter.fit(dataset)
                logger.info(
                    f"  - ✅ Training completed, is_fitted: {adapter.is_fitted}"
                )

                # Test prediction
                logger.info(f"  - Predicting with {algo_name}...")
                result = adapter.predict(dataset)

                # Analyze results
                anomaly_count = sum(result.labels)
                avg_score = np.mean([score.value for score in result.scores])
                avg_confidence = np.mean([score.confidence for score in result.scores])

                logger.info("  - ✅ Prediction completed:")
                logger.info(
                    f"    • Detected anomalies: {anomaly_count}/{len(result.labels)}"
                )
                logger.info(f"    • Average anomaly score: {avg_score:.3f}")
                logger.info(f"    • Average confidence: {avg_confidence:.3f}")
                logger.info(
                    f"    • Graph metadata: {result.metadata.get('is_graph', False)}"
                )

                results[algo_name] = {
                    "anomaly_count": anomaly_count,
                    "avg_score": avg_score,
                    "avg_confidence": avg_confidence,
                    "total_samples": len(result.labels),
                }

            except Exception as e:
                logger.error(f"  - ❌ Error with {algo_name}: {e}")
                results[algo_name] = {"error": str(e)}

    return results


def run_pygod_example_real():
    """Run PyGOD example with real dependencies."""
    from pynomaly_detection.domain.entities import Dataset
    from pynomaly_detection.domain.value_objects import ContaminationRate
    from pynomaly_detection.infrastructure.adapters.pygod_adapter import PyGODAdapter

    logger.info("🚀 Running PyGOD example with real dependencies")

    # Create sample data
    data = create_sample_citation_network_data()

    # Create dataset
    dataset = Dataset(
        id="citation_network",
        name="Citation Network Dataset",
        data=data,
        metadata={
            "is_graph": True,
            "edge_columns": ["source", "target"],
            "weight_column": "citation_context_relevance",
            "feature_columns": [
                "author_count",
                "citation_velocity",
                "journal_impact_factor",
                "self_citation_rate",
                "years_since_publication",
            ],
        },
    )

    # Get available algorithms
    available_algorithms = PyGODAdapter.get_supported_algorithms()
    logger.info(f"Available algorithms: {available_algorithms}")

    if not available_algorithms:
        logger.error(
            "No algorithms available - this shouldn't happen if dependencies are installed"
        )
        return {}

    # Test with a statistical algorithm (faster, no GPU required)
    algo_name = "SCAN" if "SCAN" in available_algorithms else available_algorithms[0]

    logger.info(f"Testing with {algo_name} algorithm...")

    # Get algorithm information
    try:
        algo_info = PyGODAdapter.get_algorithm_info(algo_name)
        logger.info(f"Algorithm info: {algo_info['description'][:100]}...")
    except Exception as e:
        logger.warning(f"Could not get algorithm info: {e}")

    # Create and configure adapter
    if algo_name == "SCAN":
        adapter = PyGODAdapter(
            algorithm_name=algo_name,
            contamination_rate=ContaminationRate(0.1),
            eps=0.5,
            mu=2,
        )
    else:
        adapter = PyGODAdapter(
            algorithm_name=algo_name, contamination_rate=ContaminationRate(0.1)
        )

    try:
        # Train the model
        logger.info("Training the model...")
        adapter.fit(dataset)
        logger.info(f"✅ Training completed, is_fitted: {adapter.is_fitted}")

        # Predict anomalies
        logger.info("Detecting anomalies...")
        result = adapter.predict(dataset)

        # Analyze results
        anomaly_count = sum(result.labels)
        avg_score = np.mean([score.value for score in result.scores])
        avg_confidence = np.mean([score.confidence for score in result.scores])

        logger.info("✅ Detection completed:")
        logger.info(f"  • Detected anomalies: {anomaly_count}/{len(result.labels)}")
        logger.info(f"  • Average anomaly score: {avg_score:.3f}")
        logger.info(f"  • Average confidence: {avg_confidence:.3f}")
        logger.info(f"  • Contamination rate: {anomaly_count/len(result.labels):.3f}")

        # Print metadata
        logger.info(f"  • Result metadata: {result.metadata}")

        return {
            "algorithm": algo_name,
            "anomaly_count": anomaly_count,
            "total_samples": len(result.labels),
            "avg_score": avg_score,
            "avg_confidence": avg_confidence,
            "contamination_rate": anomaly_count / len(result.labels),
        }

    except Exception as e:
        logger.error(f"❌ Error during {algo_name} execution: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return {"error": str(e)}


def demonstrate_algorithm_selection():
    """Demonstrate how to select appropriate algorithms for different scenarios."""
    from pynomaly_detection.infrastructure.adapters.pygod_adapter import PyGODAdapter

    logger.info("📋 Algorithm Selection Guide")

    try:
        available_algorithms = PyGODAdapter.get_supported_algorithms()

        if not available_algorithms:
            logger.info("No algorithms available - install PyGOD dependencies first")
            return

        logger.info(f"Available algorithms: {available_algorithms}")

        # Categorize algorithms
        deep_learning_algorithms = [
            "DOMINANT",
            "GCNAE",
            "ANOMALOUS",
            "MLPAE",
            "ANOMALYDAE",
            "GAAN",
            "GUIDE",
            "CONAD",
            "GADNR",
        ]
        statistical_algorithms = ["SCAN", "RADAR"]

        available_dl = [
            algo for algo in deep_learning_algorithms if algo in available_algorithms
        ]
        available_stat = [
            algo for algo in statistical_algorithms if algo in available_algorithms
        ]

        logger.info(f"  Deep Learning methods: {available_dl}")
        logger.info(f"  Statistical methods: {available_stat}")

        # Show algorithm information for a few key algorithms
        for algo in ["DOMINANT", "SCAN"]:
            if algo in available_algorithms:
                try:
                    info = PyGODAdapter.get_algorithm_info(algo)
                    logger.info(f"\n{algo} Algorithm:")
                    logger.info(f"  Type: {info.get('type', 'Unknown')}")
                    logger.info(
                        f"  Description: {info.get('description', 'No description')[:100]}..."
                    )

                    if "parameters" in info:
                        key_params = list(info["parameters"].keys())[:3]
                        logger.info(f"  Key parameters: {key_params}")

                    if "suitable_for" in info:
                        logger.info(f"  Suitable for: {info['suitable_for']}")

                except Exception as e:
                    logger.warning(f"Could not get info for {algo}: {e}")

    except Exception as e:
        logger.error(f"Error in algorithm demonstration: {e}")


def main():
    """Main function to run all examples."""
    logger.info("🎯 PyGOD Graph Anomaly Detection Example")
    logger.info("=" * 50)

    # Check dependencies
    has_pygod = check_pygod_availability()

    # Show algorithm selection guide
    demonstrate_algorithm_selection()

    # Run appropriate example based on dependency availability
    if has_pygod:
        logger.info("\n🚀 Running example with real PyGOD dependencies...")
        results = run_pygod_example_real()
    else:
        logger.info("\n🔧 Running example with mocked dependencies...")
        results = run_pygod_example_with_mocks()

    logger.info("\n📊 Final Results:")
    logger.info("=" * 30)
    for algo, result in results.items():
        if "error" in result:
            logger.error(f"{algo}: ❌ {result['error']}")
        else:
            logger.info(
                f"{algo}: ✅ {result.get('anomaly_count', 0)} anomalies detected"
            )

    logger.info("\n✅ Example completed!")

    if not has_pygod:
        logger.info("\n💡 To run with real dependencies:")
        logger.info("   pip install 'pynomaly[graph]'")


if __name__ == "__main__":
    main()
