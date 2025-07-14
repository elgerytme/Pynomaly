#!/usr/bin/env python3
"""
Deposit Transaction Anomaly Detection
Identifies suspicious deposit patterns that may indicate money laundering, structuring, or fraud.
"""

import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Pynomaly imports
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class DepositAnomalyDetector:
    """Detects anomalies in deposit transactions."""

    def __init__(self):
        # Initialize adapters with specific algorithms
        pass

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load deposit transaction data."""
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for anomaly detection."""
        features_df = df.copy()

        # Time-based features
        features_df["hour"] = features_df["timestamp"].dt.hour
        features_df["day_of_week"] = features_df["timestamp"].dt.dayofweek
        features_df["is_weekend"] = features_df["day_of_week"].isin([5, 6]).astype(int)
        features_df["is_business_hours"] = (
            (features_df["hour"] >= 9) & (features_df["hour"] <= 17)
        ).astype(int)

        # Amount-based features
        features_df["amount_log"] = np.log1p(features_df["amount"])
        features_df["amount_zscore"] = (
            features_df["amount"] - features_df["amount"].mean()
        ) / features_df["amount"].std()

        # Customer behavior features
        customer_stats = (
            features_df.groupby("customer_id")["amount"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )
        customer_stats.columns = [
            "customer_id",
            "deposit_count",
            "avg_deposit",
            "std_deposit",
            "min_deposit",
            "max_deposit",
        ]
        customer_stats["std_deposit"] = customer_stats["std_deposit"].fillna(0)
        features_df = features_df.merge(customer_stats, on="customer_id")

        # Velocity features (deposits in short time windows)
        features_df = features_df.sort_values(["customer_id", "timestamp"])
        features_df["deposits_last_hour"] = 0
        features_df["deposits_last_day"] = 0

        for idx, row in features_df.iterrows():
            time_window_hour = row["timestamp"] - timedelta(hours=1)
            time_window_day = row["timestamp"] - timedelta(days=1)

            deposits_hour = features_df[
                (features_df["customer_id"] == row["customer_id"])
                & (features_df["timestamp"] >= time_window_hour)
                & (features_df["timestamp"] < row["timestamp"])
            ].shape[0]

            deposits_day = features_df[
                (features_df["customer_id"] == row["customer_id"])
                & (features_df["timestamp"] >= time_window_day)
                & (features_df["timestamp"] < row["timestamp"])
            ].shape[0]

            features_df.at[idx, "deposits_last_hour"] = deposits_hour
            features_df.at[idx, "deposits_last_day"] = deposits_day

        # Structuring indicators (amounts just under $10,000)
        features_df["near_threshold"] = (
            (features_df["amount"] >= 9000) & (features_df["amount"] < 10000)
        ).astype(int)

        # Source type encoding
        source_type_encoded = pd.get_dummies(
            features_df["source_type"], prefix="source"
        )
        features_df = pd.concat([features_df, source_type_encoded], axis=1)

        return features_df

    def select_features(self, df: pd.DataFrame) -> np.ndarray:
        """Select relevant features for anomaly detection."""
        feature_columns = [
            "amount",
            "amount_log",
            "amount_zscore",
            "hour",
            "day_of_week",
            "is_weekend",
            "is_business_hours",
            "deposit_count",
            "avg_deposit",
            "std_deposit",
            "deposits_last_hour",
            "deposits_last_day",
            "near_threshold",
        ]

        # Add source type columns
        source_columns = [col for col in df.columns if col.startswith("source_")]
        feature_columns.extend(source_columns)

        return df[feature_columns].fillna(0).values

    def detect_anomalies(self, df: pd.DataFrame, contamination: float = 0.05):
        """Detect anomalies using multiple algorithms."""
        features_df = self.engineer_features(df)
        X = self.select_features(features_df)

        # Create dataset
        dataset = Dataset(
            data=X,
            target=None,
            feature_names=[f"feature_{i}" for i in range(X.shape[1])],
        )

        contamination_rate = ContaminationRate(contamination)

        results = {}

        # Isolation Forest
        iso_adapter = SklearnAdapter(
            "IsolationForest", contamination_rate=contamination_rate
        )
        iso_result = iso_adapter.detect_anomalies(dataset)
        results["isolation_forest"] = iso_result.anomaly_scores

        # One-Class SVM
        svm_adapter = SklearnAdapter(
            "OneClassSVM", contamination_rate=contamination_rate
        )
        svm_result = svm_adapter.detect_anomalies(dataset)
        results["one_class_svm"] = svm_result.anomaly_scores

        # Local Outlier Factor
        lof_adapter = SklearnAdapter(
            "LocalOutlierFactor", contamination_rate=contamination_rate
        )
        lof_result = lof_adapter.detect_anomalies(dataset)
        results["lof"] = lof_result.anomaly_scores

        # Combine results
        features_df["iso_score"] = results["isolation_forest"]
        features_df["svm_score"] = results["one_class_svm"]
        features_df["lof_score"] = results["lof"]

        # Ensemble score (average)
        features_df["ensemble_score"] = (
            features_df["iso_score"]
            + features_df["svm_score"]
            + features_df["lof_score"]
        ) / 3

        # Flag top anomalies
        threshold = np.percentile(
            features_df["ensemble_score"], (1 - contamination) * 100
        )
        features_df["predicted_anomaly"] = (
            features_df["ensemble_score"] > threshold
        ).astype(int)

        return features_df, results

    def analyze_anomalies(self, df: pd.DataFrame):
        """Analyze detected anomalies and provide insights."""
        anomalies = df[df["predicted_anomaly"] == 1]
        normal = df[df["predicted_anomaly"] == 0]

        print("=== DEPOSIT ANOMALY ANALYSIS ===\n")
        print(f"Total transactions: {len(df)}")
        print(
            f"Detected anomalies: {len(anomalies)} ({len(anomalies) / len(df) * 100:.1f}%)"
        )
        print(
            f"Actual anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean() * 100:.1f}%)"
        )

        if len(anomalies) > 0:
            print("\nAnomaly Characteristics:")
            print(
                f"Average amount: ${anomalies['amount'].mean():,.2f} vs ${normal['amount'].mean():,.2f} (normal)"
            )
            print(f"Max amount: ${anomalies['amount'].max():,.2f}")
            print(f"Deposits near $10K threshold: {anomalies['near_threshold'].sum()}")
            print(
                f"Weekend deposits: {anomalies['is_weekend'].sum()} ({anomalies['is_weekend'].mean() * 100:.1f}%)"
            )
            print(f"After-hours deposits: {(1 - anomalies['is_business_hours']).sum()}")

            print("\nSource Type Distribution (Anomalies):")
            for source in ["cash", "check", "wire", "ach"]:
                if f"source_{source}" in anomalies.columns:
                    count = anomalies[f"source_{source}"].sum()
                    pct = count / len(anomalies) * 100 if len(anomalies) > 0 else 0
                    print(f"  {source}: {count} ({pct:.1f}%)")

            print("\nVelocity Patterns:")
            print(
                f"Multiple deposits in same hour: {(anomalies['deposits_last_hour'] > 0).sum()}"
            )
            print(
                f"Multiple deposits in same day: {(anomalies['deposits_last_day'] > 3).sum()}"
            )

        return anomalies

    def visualize_results(self, df: pd.DataFrame, output_dir: str = None):
        """Create visualizations of anomaly detection results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Deposit Transaction Anomaly Analysis", fontsize=16)

        # Amount distribution
        axes[0, 0].hist(
            df[df["predicted_anomaly"] == 0]["amount"],
            bins=50,
            alpha=0.7,
            label="Normal",
            density=True,
        )
        axes[0, 0].hist(
            df[df["predicted_anomaly"] == 1]["amount"],
            bins=50,
            alpha=0.7,
            label="Anomaly",
            density=True,
        )
        axes[0, 0].set_xlabel("Amount ($)")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].set_title("Amount Distribution")
        axes[0, 0].legend()
        axes[0, 0].set_yscale("log")

        # Time patterns
        hour_counts = (
            df.groupby(["hour", "predicted_anomaly"]).size().unstack(fill_value=0)
        )
        hour_counts.plot(kind="bar", ax=axes[0, 1], color=["blue", "red"])
        axes[0, 1].set_xlabel("Hour of Day")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Hourly Distribution")
        axes[0, 1].legend(["Normal", "Anomaly"])

        # Source type
        source_cols = [col for col in df.columns if col.startswith("source_")]
        if source_cols:
            source_data = (
                df[source_cols + ["predicted_anomaly"]]
                .groupby("predicted_anomaly")
                .sum()
            )
            source_data.T.plot(kind="bar", ax=axes[0, 2], color=["blue", "red"])
            axes[0, 2].set_xlabel("Source Type")
            axes[0, 2].set_ylabel("Count")
            axes[0, 2].set_title("Source Type Distribution")
            axes[0, 2].legend(["Normal", "Anomaly"])

        # Anomaly scores
        axes[1, 0].scatter(
            df["iso_score"],
            df["svm_score"],
            c=df["predicted_anomaly"],
            cmap="coolwarm",
            alpha=0.6,
        )
        axes[1, 0].set_xlabel("Isolation Forest Score")
        axes[1, 0].set_ylabel("One-Class SVM Score")
        axes[1, 0].set_title("Algorithm Score Comparison")

        # Near threshold analysis
        threshold_data = (
            df.groupby(["near_threshold", "predicted_anomaly"])
            .size()
            .unstack(fill_value=0)
        )
        threshold_data.plot(kind="bar", ax=axes[1, 1], color=["blue", "red"])
        axes[1, 1].set_xlabel("Near $10K Threshold")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Structuring Indicator")
        axes[1, 1].legend(["Normal", "Anomaly"])

        # Velocity analysis
        axes[1, 2].scatter(
            df["deposits_last_hour"],
            df["deposits_last_day"],
            c=df["predicted_anomaly"],
            cmap="coolwarm",
            alpha=0.6,
        )
        axes[1, 2].set_xlabel("Deposits Last Hour")
        axes[1, 2].set_ylabel("Deposits Last Day")
        axes[1, 2].set_title("Deposit Velocity Patterns")

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(
                os.path.join(output_dir, "deposit_anomaly_analysis.png"),
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

    def generate_report(
        self, df: pd.DataFrame, anomalies: pd.DataFrame, output_dir: str = None
    ):
        """Generate a detailed anomaly report."""
        report = []
        report.append("# Deposit Transaction Anomaly Detection Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        report.append("## Executive Summary")
        report.append(f"- Total transactions analyzed: {len(df):,}")
        report.append(
            f"- Anomalies detected: {len(anomalies):,} ({len(anomalies) / len(df) * 100:.1f}%)"
        )
        report.append(
            f"- Highest risk transaction: ${anomalies['amount'].max():,.2f}"
            if len(anomalies) > 0
            else "- No high-risk transactions detected"
        )

        if len(anomalies) > 0:
            report.append("\n## Key Risk Indicators")

            # Large amounts
            large_amounts = anomalies[anomalies["amount"] > 50000]
            if len(large_amounts) > 0:
                report.append(f"- {len(large_amounts)} transactions over $50,000")
                report.append(f"  - Largest: ${large_amounts['amount'].max():,.2f}")

            # Structuring
            structuring = anomalies[anomalies["near_threshold"] == 1]
            if len(structuring) > 0:
                report.append(
                    f"- {len(structuring)} potential structuring transactions (near $10K threshold)"
                )

            # Timing anomalies
            after_hours = anomalies[anomalies["is_business_hours"] == 0]
            if len(after_hours) > 0:
                report.append(f"- {len(after_hours)} after-hours deposits")

            # High velocity
            high_velocity = anomalies[anomalies["deposits_last_hour"] > 0]
            if len(high_velocity) > 0:
                report.append(
                    f"- {len(high_velocity)} accounts with multiple deposits in same hour"
                )

            report.append("\n## Top 10 Highest Risk Transactions")
            top_anomalies = anomalies.nlargest(10, "ensemble_score")[
                [
                    "transaction_id",
                    "customer_id",
                    "amount",
                    "source_type",
                    "timestamp",
                    "ensemble_score",
                ]
            ]
            report.append(top_anomalies.to_string(index=False))

        report_text = "\n".join(report)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "deposit_anomaly_report.md"), "w") as f:
                f.write(report_text)

        print(report_text)
        return report_text


def main():
    """Main execution function."""
    detector = DepositAnomalyDetector()

    # Load data
    data_path = "../datasets/deposits.csv"
    df = detector.load_data(data_path)

    # Detect anomalies
    print("Detecting anomalies in deposit transactions...")
    results_df, scores = detector.detect_anomalies(df)

    # Analyze results
    anomalies = detector.analyze_anomalies(results_df)

    # Create visualizations
    detector.visualize_results(results_df, output_dir="../outputs")

    # Generate report
    detector.generate_report(results_df, anomalies, output_dir="../outputs")


if __name__ == "__main__":
    main()
