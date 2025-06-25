#!/usr/bin/env python3
"""
Credit Card Transaction Anomaly Detection
Identifies fraudulent credit card transactions and unusual spending patterns.
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
from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class CreditCardAnomalyDetector:
    """Detects anomalies in credit card transactions."""

    def __init__(self):
        self.sklearn_adapter = SklearnAdapter()
        self.pyod_adapter = PyODAdapter()

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load credit card transaction data."""
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for fraud detection."""
        features_df = df.copy()

        # Time-based features
        features_df["hour"] = features_df["timestamp"].dt.hour
        features_df["day_of_week"] = features_df["timestamp"].dt.dayofweek
        features_df["is_weekend"] = features_df["day_of_week"].isin([5, 6]).astype(int)
        features_df["is_night"] = (
            (features_df["hour"] < 6) | (features_df["hour"] > 22)
        ).astype(int)

        # Amount-based features
        features_df["amount_log"] = np.log1p(features_df["amount"])
        features_df["amount_zscore"] = (
            features_df["amount"] - features_df["amount"].mean()
        ) / features_df["amount"].std()
        features_df["high_amount"] = (features_df["amount"] > 1000).astype(int)

        # Customer spending patterns
        customer_stats = (
            features_df.groupby("customer_id")
            .agg(
                {
                    "amount": ["count", "mean", "std", "min", "max"],
                    "timestamp": ["min", "max"],
                }
            )
            .reset_index()
        )

        customer_stats.columns = [
            "customer_id",
            "tx_count",
            "avg_amount",
            "std_amount",
            "min_amount",
            "max_amount",
            "first_tx",
            "last_tx",
        ]
        customer_stats["std_amount"] = customer_stats["std_amount"].fillna(0)
        customer_stats["spending_range"] = (
            customer_stats["max_amount"] - customer_stats["min_amount"]
        )
        customer_stats["account_age_days"] = (
            customer_stats["last_tx"] - customer_stats["first_tx"]
        ).dt.days

        features_df = features_df.merge(customer_stats, on="customer_id")

        # Transaction velocity features
        features_df = features_df.sort_values(["customer_id", "timestamp"])
        features_df["tx_last_hour"] = 0
        features_df["tx_last_day"] = 0
        features_df["amount_last_hour"] = 0.0
        features_df["amount_last_day"] = 0.0

        for idx, row in features_df.iterrows():
            time_window_hour = row["timestamp"] - timedelta(hours=1)
            time_window_day = row["timestamp"] - timedelta(days=1)

            recent_hour = features_df[
                (features_df["customer_id"] == row["customer_id"])
                & (features_df["timestamp"] >= time_window_hour)
                & (features_df["timestamp"] < row["timestamp"])
            ]

            recent_day = features_df[
                (features_df["customer_id"] == row["customer_id"])
                & (features_df["timestamp"] >= time_window_day)
                & (features_df["timestamp"] < row["timestamp"])
            ]

            features_df.at[idx, "tx_last_hour"] = len(recent_hour)
            features_df.at[idx, "tx_last_day"] = len(recent_day)
            features_df.at[idx, "amount_last_hour"] = recent_hour["amount"].sum()
            features_df.at[idx, "amount_last_day"] = recent_day["amount"].sum()

        # Merchant and category features
        merchant_encoded = pd.get_dummies(features_df["merchant"], prefix="merchant")
        category_encoded = pd.get_dummies(features_df["category"], prefix="category")
        location_encoded = pd.get_dummies(features_df["location"], prefix="location")

        features_df = pd.concat(
            [features_df, merchant_encoded, category_encoded, location_encoded], axis=1
        )

        # Deviation from normal patterns
        features_df["amount_dev_from_avg"] = abs(
            features_df["amount"] - features_df["avg_amount"]
        ) / (features_df["std_amount"] + 1)
        features_df["unusual_amount"] = (features_df["amount_dev_from_avg"] > 3).astype(
            int
        )

        # Card present indicator
        features_df["card_present_int"] = features_df["card_present"].astype(int)

        return features_df

    def select_features(self, df: pd.DataFrame) -> np.ndarray:
        """Select relevant features for fraud detection."""
        feature_columns = [
            "amount",
            "amount_log",
            "amount_zscore",
            "high_amount",
            "hour",
            "day_of_week",
            "is_weekend",
            "is_night",
            "tx_count",
            "avg_amount",
            "std_amount",
            "spending_range",
            "tx_last_hour",
            "tx_last_day",
            "amount_last_hour",
            "amount_last_day",
            "amount_dev_from_avg",
            "unusual_amount",
            "card_present_int",
            "mcc_code",
        ]

        # Add encoded categorical features (top ones to avoid too many features)
        merchant_cols = [col for col in df.columns if col.startswith("merchant_")][:10]
        category_cols = [col for col in df.columns if col.startswith("category_")]
        location_cols = [col for col in df.columns if col.startswith("location_")]

        feature_columns.extend(merchant_cols + category_cols + location_cols)

        # Only keep columns that exist in the dataframe
        feature_columns = [col for col in feature_columns if col in df.columns]

        return df[feature_columns].fillna(0).values

    def detect_anomalies(self, df: pd.DataFrame, contamination: float = 0.05):
        """Detect fraudulent transactions using multiple algorithms."""
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

        # Isolation Forest (good for fraud detection)
        iso_result = self.sklearn_adapter.detect_anomalies(
            dataset=dataset,
            algorithm_type="isolation_forest",
            contamination=contamination_rate,
        )
        results["isolation_forest"] = iso_result.anomaly_scores

        # Local Outlier Factor (detects density-based anomalies)
        lof_result = self.pyod_adapter.detect_anomalies(
            dataset=dataset, algorithm_type="lof", contamination=contamination_rate
        )
        results["lof"] = lof_result.anomaly_scores

        # CBLOF (Cluster-based outlier detection)
        try:
            cblof_result = self.pyod_adapter.detect_anomalies(
                dataset=dataset,
                algorithm_type="cblof",
                contamination=contamination_rate,
            )
            results["cblof"] = cblof_result.anomaly_scores
        except:
            # Fallback to One-Class SVM if CBLOF fails
            svm_result = self.sklearn_adapter.detect_anomalies(
                dataset=dataset,
                algorithm_type="one_class_svm",
                contamination=contamination_rate,
            )
            results["one_class_svm"] = svm_result.anomaly_scores

        # Combine results
        for algo, scores in results.items():
            features_df[f"{algo}_score"] = scores

        # Ensemble score
        score_columns = [col for col in features_df.columns if col.endswith("_score")]
        features_df["ensemble_score"] = features_df[score_columns].mean(axis=1)

        # Flag anomalies
        threshold = np.percentile(
            features_df["ensemble_score"], (1 - contamination) * 100
        )
        features_df["predicted_anomaly"] = (
            features_df["ensemble_score"] > threshold
        ).astype(int)

        return features_df, results

    def analyze_anomalies(self, df: pd.DataFrame):
        """Analyze detected fraud patterns."""
        anomalies = df[df["predicted_anomaly"] == 1]
        normal = df[df["predicted_anomaly"] == 0]

        print("=== CREDIT CARD FRAUD ANALYSIS ===\n")
        print(f"Total transactions: {len(df):,}")
        print(
            f"Detected anomalies: {len(anomalies):,} ({len(anomalies) / len(df) * 100:.1f}%)"
        )
        print(
            f"Actual anomalies: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean() * 100:.1f}%)"
        )

        if len(anomalies) > 0:
            print("\nFraud Characteristics:")
            print(
                f"Average amount: ${anomalies['amount'].mean():,.2f} vs ${normal['amount'].mean():,.2f} (normal)"
            )
            print(f"Max fraudulent amount: ${anomalies['amount'].max():,.2f}")
            print(f"High-value transactions (>$1000): {anomalies['high_amount'].sum()}")
            print(
                f"Card-not-present transactions: {(1 - anomalies['card_present_int']).sum()} ({(1 - anomalies['card_present_int']).mean() * 100:.1f}%)"
            )
            print(
                f"Night-time transactions: {anomalies['is_night'].sum()} ({anomalies['is_night'].mean() * 100:.1f}%)"
            )
            print(
                f"Weekend transactions: {anomalies['is_weekend'].sum()} ({anomalies['is_weekend'].mean() * 100:.1f}%)"
            )

            print("\nLocation Patterns:")
            if "location_international" in anomalies.columns:
                intl_count = anomalies["location_international"].sum()
                print(
                    f"International transactions: {intl_count} ({intl_count / len(anomalies) * 100:.1f}%)"
                )

            print("\nVelocity Patterns:")
            print(
                f"Multiple transactions in same hour: {(anomalies['tx_last_hour'] > 0).sum()}"
            )
            print(
                f"High daily transaction volume: {(anomalies['tx_last_day'] > 5).sum()}"
            )
            print(f"High daily spending: {(anomalies['amount_last_day'] > 2000).sum()}")

            print("\nMerchant Categories (Top 5 for anomalies):")
            category_cols = [
                col for col in anomalies.columns if col.startswith("category_")
            ]
            if category_cols:
                category_sums = (
                    anomalies[category_cols].sum().sort_values(ascending=False)
                )
                for _i, (cat, count) in enumerate(category_sums.head().items()):
                    if count > 0:
                        cat_name = cat.replace("category_", "")
                        print(f"  {cat_name}: {count} transactions")

        return anomalies

    def visualize_results(self, df: pd.DataFrame, output_dir: str = None):
        """Create fraud detection visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Credit Card Fraud Detection Analysis", fontsize=16)

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
            label="Fraud",
            density=True,
        )
        axes[0, 0].set_xlabel("Amount ($)")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].set_title("Transaction Amount Distribution")
        axes[0, 0].legend()
        axes[0, 0].set_xscale("log")

        # Time patterns
        hour_counts = (
            df.groupby(["hour", "predicted_anomaly"]).size().unstack(fill_value=0)
        )
        hour_counts.plot(kind="bar", ax=axes[0, 1], color=["blue", "red"])
        axes[0, 1].set_xlabel("Hour of Day")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Hourly Transaction Distribution")
        axes[0, 1].legend(["Normal", "Fraud"])

        # Card present vs not present
        card_present_data = (
            df.groupby(["card_present", "predicted_anomaly"])
            .size()
            .unstack(fill_value=0)
        )
        card_present_data.plot(kind="bar", ax=axes[0, 2], color=["blue", "red"])
        axes[0, 2].set_xlabel("Card Present")
        axes[0, 2].set_ylabel("Count")
        axes[0, 2].set_title("Card Present vs Not Present")
        axes[0, 2].legend(["Normal", "Fraud"])

        # Transaction velocity
        axes[1, 0].scatter(
            df["tx_last_hour"],
            df["amount"],
            c=df["predicted_anomaly"],
            cmap="coolwarm",
            alpha=0.6,
        )
        axes[1, 0].set_xlabel("Transactions in Last Hour")
        axes[1, 0].set_ylabel("Transaction Amount ($)")
        axes[1, 0].set_title("Transaction Velocity vs Amount")
        axes[1, 0].set_yscale("log")

        # Amount deviation
        axes[1, 1].scatter(
            df["amount_dev_from_avg"],
            df["amount"],
            c=df["predicted_anomaly"],
            cmap="coolwarm",
            alpha=0.6,
        )
        axes[1, 1].set_xlabel("Deviation from Average Amount")
        axes[1, 1].set_ylabel("Transaction Amount ($)")
        axes[1, 1].set_title("Amount Deviation Pattern")
        axes[1, 1].set_yscale("log")

        # Algorithm scores comparison
        score_cols = [
            col
            for col in df.columns
            if col.endswith("_score") and col != "ensemble_score"
        ]
        if len(score_cols) >= 2:
            axes[1, 2].scatter(
                df[score_cols[0]],
                df[score_cols[1]],
                c=df["predicted_anomaly"],
                cmap="coolwarm",
                alpha=0.6,
            )
            axes[1, 2].set_xlabel(score_cols[0].replace("_", " ").title())
            axes[1, 2].set_ylabel(score_cols[1].replace("_", " ").title())
            axes[1, 2].set_title("Algorithm Score Comparison")

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(
                os.path.join(output_dir, "credit_card_fraud_analysis.png"),
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

    def generate_fraud_report(
        self, df: pd.DataFrame, anomalies: pd.DataFrame, output_dir: str = None
    ):
        """Generate a detailed fraud detection report."""
        report = []
        report.append("# Credit Card Fraud Detection Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        report.append("## Executive Summary")
        report.append(f"- Total transactions analyzed: {len(df):,}")
        report.append(
            f"- Fraudulent transactions detected: {len(anomalies):,} ({len(anomalies) / len(df) * 100:.1f}%)"
        )
        report.append(
            f"- Total fraud amount: ${anomalies['amount'].sum():,.2f}"
            if len(anomalies) > 0
            else "- No fraudulent transactions detected"
        )
        report.append(
            f"- Average fraud amount: ${anomalies['amount'].mean():,.2f}"
            if len(anomalies) > 0
            else ""
        )

        if len(anomalies) > 0:
            report.append("\n## Fraud Patterns Identified")

            # High-value fraud
            high_value = anomalies[anomalies["amount"] > 1000]
            if len(high_value) > 0:
                report.append(
                    f"- {len(high_value)} high-value fraudulent transactions (>$1,000)"
                )
                report.append(f"  - Total value: ${high_value['amount'].sum():,.2f}")

            # Card-not-present fraud
            cnp_fraud = anomalies[anomalies["card_present_int"] == 0]
            if len(cnp_fraud) > 0:
                report.append(
                    f"- {len(cnp_fraud)} card-not-present fraudulent transactions"
                )

            # Velocity-based fraud
            velocity_fraud = anomalies[anomalies["tx_last_hour"] > 0]
            if len(velocity_fraud) > 0:
                report.append(
                    f"- {len(velocity_fraud)} accounts with multiple transactions in same hour"
                )

            # Timing patterns
            night_fraud = anomalies[anomalies["is_night"] == 1]
            if len(night_fraud) > 0:
                report.append(
                    f"- {len(night_fraud)} night-time fraudulent transactions"
                )

            # Location-based fraud
            if "location_international" in anomalies.columns:
                intl_fraud = anomalies[anomalies["location_international"] == 1]
                if len(intl_fraud) > 0:
                    report.append(
                        f"- {len(intl_fraud)} international fraudulent transactions"
                    )

            report.append("\n## Highest Risk Transactions")
            top_fraud = anomalies.nlargest(10, "ensemble_score")[
                [
                    "transaction_id",
                    "customer_id",
                    "amount",
                    "merchant",
                    "timestamp",
                    "ensemble_score",
                ]
            ]
            report.append(top_fraud.to_string(index=False))

            report.append("\n## Recommended Actions")
            report.append(
                "1. Immediately flag transactions with ensemble score > 0.8 for manual review"
            )
            report.append("2. Implement real-time monitoring for velocity patterns")
            report.append("3. Enhanced verification for card-not-present transactions")
            report.append("4. Review international transaction policies")
            report.append(
                "5. Set up alerts for transactions deviating significantly from customer patterns"
            )

        report_text = "\n".join(report)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(
                os.path.join(output_dir, "credit_card_fraud_report.md"), "w"
            ) as f:
                f.write(report_text)

        print(report_text)
        return report_text


def main():
    """Main execution function."""
    detector = CreditCardAnomalyDetector()

    # Load data
    data_path = "../datasets/credit_card_transactions.csv"
    df = detector.load_data(data_path)

    # Detect fraud
    print("Detecting fraud in credit card transactions...")
    results_df, scores = detector.detect_anomalies(df)

    # Analyze results
    anomalies = detector.analyze_anomalies(results_df)

    # Create visualizations
    detector.visualize_results(results_df, output_dir="../outputs")

    # Generate report
    detector.generate_fraud_report(results_df, anomalies, output_dir="../outputs")


if __name__ == "__main__":
    main()
