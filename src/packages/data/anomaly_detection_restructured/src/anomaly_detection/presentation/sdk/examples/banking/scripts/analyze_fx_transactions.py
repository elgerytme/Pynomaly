#!/usr/bin/env python3
"""
Foreign Exchange Transaction Anomaly Detection
Identifies money laundering, trade-based money laundering, and suspicious FX patterns.
"""

"""
TODO: This file needs dependency injection refactoring.
Replace direct monorepo imports with dependency injection.
Use interfaces/shared/base_entity.py for abstractions.
"""



import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Pynomaly imports
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from interfaces.domain.entities.dataset import Dataset
from interfaces.domain.value_objects.contamination_rate import ContaminationRate
# TODO: Create local pyod adapter
# TODO: Create local sklearn adapter


class FXAnomalyDetector:
    """Detects anomalies in foreign exchange transactions."""

    def __init__(self):
        self.sklearn_adapter = SklearnAdapter()
        self.pyod_adapter = PyODAdapter()

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load FX transaction data."""
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for FX anomaly detection."""
        features_df = df.copy()

        # Time-based features
        features_df["hour"] = features_df["timestamp"].dt.hour
        features_df["day_of_week"] = features_df["timestamp"].dt.dayofweek
        features_df["is_weekend"] = features_df["day_of_week"].isin([5, 6]).astype(int)
        features_df["is_market_hours"] = (
            (features_df["hour"] >= 8) & (features_df["hour"] <= 17)
        ).astype(int)

        # Amount-based features
        features_df["amount_log"] = np.log1p(features_df["amount_from"])
        features_df["large_amount"] = (features_df["amount_from"] > 50000).astype(int)
        features_df["near_threshold"] = (
            (features_df["amount_from"] >= 9500) & (features_df["amount_from"] < 10000)
        ).astype(int)

        # Exchange rate features
        # Calculate expected rates based on historical data (simplified)
        rate_stats = (
            features_df.groupby("to_currency")["exchange_rate"]
            .agg(["mean", "std"])
            .reset_index()
        )
        rate_stats.columns = ["to_currency", "rate_mean", "rate_std"]
        features_df = features_df.merge(rate_stats, on="to_currency")

        # Rate deviation
        features_df["rate_deviation"] = abs(
            features_df["exchange_rate"] - features_df["rate_mean"]
        ) / (features_df["rate_std"] + 0.01)
        features_df["suspicious_rate"] = (features_df["rate_deviation"] > 2).astype(int)

        # Customer behavior
        customer_stats = (
            features_df.groupby("customer_id")
            .agg(
                {
                    "amount_from": ["count", "mean", "std", "sum"],
                    "timestamp": ["min", "max"],
                }
            )
            .reset_index()
        )

        customer_stats.columns = [
            "customer_id",
            "fx_count",
            "avg_amount",
            "std_amount",
            "total_amount",
            "first_fx",
            "last_fx",
        ]
        customer_stats["std_amount"] = customer_stats["std_amount"].fillna(0)
        features_df = features_df.merge(customer_stats, on="customer_id")

        # Velocity features
        features_df = features_df.sort_values(["customer_id", "timestamp"])
        features_df["fx_last_day"] = 0
        features_df["amount_last_day"] = 0.0

        for idx, row in features_df.iterrows():
            time_window = row["timestamp"] - timedelta(days=1)
            recent_fx = features_df[
                (features_df["customer_id"] == row["customer_id"])
                & (features_df["timestamp"] >= time_window)
                & (features_df["timestamp"] < row["timestamp"])
            ]

            features_df.at[idx, "fx_last_day"] = len(recent_fx)
            features_df.at[idx, "amount_last_day"] = recent_fx["amount_from"].sum()

        # Currency and purpose encoding
        currency_encoded = pd.get_dummies(features_df["to_currency"], prefix="curr")
        purpose_encoded = pd.get_dummies(features_df["purpose"], prefix="purpose")
        method_encoded = pd.get_dummies(features_df["method"], prefix="method")

        features_df = pd.concat(
            [features_df, currency_encoded, purpose_encoded, method_encoded], axis=1
        )

        # Fee analysis
        features_df["fee_rate"] = features_df["fee_usd"] / features_df["amount_from"]
        features_df["low_fee"] = (features_df["fee_rate"] < 0.001).astype(
            int
        )  # Suspiciously low fee

        return features_df

    def select_features(self, df: pd.DataFrame) -> np.ndarray:
        """Select relevant features for FX anomaly detection."""
        feature_columns = [
            "amount_from",
            "amount_log",
            "large_amount",
            "near_threshold",
            "exchange_rate",
            "rate_deviation",
            "suspicious_rate",
            "hour",
            "day_of_week",
            "is_weekend",
            "is_market_hours",
            "fx_count",
            "avg_amount",
            "std_amount",
            "fx_last_day",
            "amount_last_day",
            "fee_usd",
            "fee_rate",
            "low_fee",
        ]

        # Add encoded features
        curr_cols = [col for col in df.columns if col.startswith("curr_")][
            :5
        ]  # Top 5 currencies
        purpose_cols = [col for col in df.columns if col.startswith("purpose_")]
        method_cols = [col for col in df.columns if col.startswith("method_")]

        feature_columns.extend(curr_cols + purpose_cols + method_cols)

        # Only keep columns that exist
        feature_columns = [col for col in feature_columns if col in df.columns]

        return df[feature_columns].fillna(0).values

    def detect_anomalies(self, df: pd.DataFrame, contamination: float = 0.1):
        """Detect FX anomalies using multiple algorithms."""
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
        iso_result = self.sklearn_adapter.detect_anomalies(
            dataset=dataset,
            algorithm_type="isolation_forest",
            contamination=contamination_rate,
        )
        results["isolation_forest"] = iso_result.anomaly_scores

        # One-Class SVM
        svm_result = self.sklearn_adapter.detect_anomalies(
            dataset=dataset,
            algorithm_type="one_class_svm",
            contamination=contamination_rate,
        )
        results["one_class_svm"] = svm_result.anomaly_scores

        # Local Outlier Factor
        lof_result = self.pyod_adapter.detect_anomalies(
            dataset=dataset, algorithm_type="lof", contamination=contamination_rate
        )
        results["lof"] = lof_result.anomaly_scores

        # Combine results
        features_df["iso_score"] = results["isolation_forest"]
        features_df["svm_score"] = results["one_class_svm"]
        features_df["lof_score"] = results["lof"]

        # Ensemble score
        features_df["ensemble_score"] = (
            features_df["iso_score"]
            + features_df["svm_score"]
            + features_df["lof_score"]
        ) / 3

        # Flag anomalies
        threshold = np.percentile(
            features_df["ensemble_score"], (1 - contamination) * 100
        )
        features_df["predicted_anomaly"] = (
            features_df["ensemble_score"] > threshold
        ).astype(int)

        return features_df, results

    def analyze_anomalies(self, df: pd.DataFrame):
        """Analyze detected FX anomalies."""
        anomalies = df[df["predicted_anomaly"] == 1]
        normal = df[df["predicted_anomaly"] == 0]

        print("=== FOREIGN EXCHANGE ANOMALY ANALYSIS ===\n")
        print(f"Total FX transactions: {len(df):,}")
        print(
            f"Detected anomalies: {len(anomalies):,} ({len(anomalies) / len(df) * 100:.1f}%)"
        )
        print(
            f"Actual anomalies: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean() * 100:.1f}%)"
        )

        if len(anomalies) > 0:
            print("\nAnomaly Characteristics:")
            print(
                f"Average amount: ${anomalies['amount_from'].mean():,.2f} vs ${normal['amount_from'].mean():,.2f} (normal)"
            )
            print(f"Total suspicious amount: ${anomalies['amount_from'].sum():,.2f}")
            print(f"Large transactions (>$50K): {anomalies['large_amount'].sum()}")
            print(f"Near-threshold transactions: {anomalies['near_threshold'].sum()}")
            print(f"Suspicious exchange rates: {anomalies['suspicious_rate'].sum()}")
            print(f"Low fee transactions: {anomalies['low_fee'].sum()}")

            print("\nTiming Patterns:")
            print(
                f"Off-market-hours transactions: {(1 - anomalies['is_market_hours']).sum()}"
            )
            print(f"Weekend transactions: {anomalies['is_weekend'].sum()}")

            print("\nVelocity Patterns:")
            print(f"High daily FX volume: {(anomalies['fx_last_day'] > 3).sum()}")
            print(f"High daily amount: {(anomalies['amount_last_day'] > 100000).sum()}")

            print("\nCurrency Distribution (Top 5):")
            curr_cols = [col for col in anomalies.columns if col.startswith("curr_")]
            if curr_cols:
                curr_sums = anomalies[curr_cols].sum().sort_values(ascending=False)
                for _i, (curr, count) in enumerate(curr_sums.head().items()):
                    if count > 0:
                        curr_name = curr.replace("curr_", "")
                        print(f"  {curr_name}: {count} transactions")

            print("\nPurpose Analysis:")
            purpose_cols = [
                col for col in anomalies.columns if col.startswith("purpose_")
            ]
            if purpose_cols:
                purpose_sums = (
                    anomalies[purpose_cols].sum().sort_values(ascending=False)
                )
                for purpose, count in purpose_sums.items():
                    if count > 0:
                        purpose_name = purpose.replace("purpose_", "")
                        print(f"  {purpose_name}: {count} transactions")

        return anomalies

    def generate_report(
        self, df: pd.DataFrame, anomalies: pd.DataFrame, output_dir: str = None
    ):
        """Generate FX anomaly report."""
        report = []
        report.append("# Foreign Exchange Transaction Anomaly Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        report.append("## Executive Summary")
        report.append(f"- Total FX transactions analyzed: {len(df):,}")
        report.append(
            f"- Suspicious transactions detected: {len(anomalies):,} ({len(anomalies) / len(df) * 100:.1f}%)"
        )
        report.append(
            f"- Total suspicious amount: ${anomalies['amount_from'].sum():,.2f}"
            if len(anomalies) > 0
            else "- No suspicious transactions detected"
        )

        if len(anomalies) > 0:
            report.append("\n## Money Laundering Indicators")

            # Large amounts
            large_amounts = anomalies[anomalies["large_amount"] == 1]
            if len(large_amounts) > 0:
                report.append(
                    f"- {len(large_amounts)} large FX transactions (>$50,000)"
                )
                report.append(
                    f"  - Total value: ${large_amounts['amount_from'].sum():,.2f}"
                )

            # Structuring
            structuring = anomalies[anomalies["near_threshold"] == 1]
            if len(structuring) > 0:
                report.append(
                    f"- {len(structuring)} potential structuring transactions (near $10K threshold)"
                )

            # Rate manipulation
            rate_manip = anomalies[anomalies["suspicious_rate"] == 1]
            if len(rate_manip) > 0:
                report.append(
                    f"- {len(rate_manip)} transactions with suspicious exchange rates"
                )

            # Low fees (possible insider deals)
            low_fees = anomalies[anomalies["low_fee"] == 1]
            if len(low_fees) > 0:
                report.append(f"- {len(low_fees)} transactions with unusually low fees")

            # High velocity
            high_velocity = anomalies[anomalies["fx_last_day"] > 3]
            if len(high_velocity) > 0:
                report.append(
                    f"- {len(high_velocity)} customers with high daily FX activity"
                )

            report.append("\n## Highest Risk Transactions")
            top_anomalies = anomalies.nlargest(10, "ensemble_score")[
                [
                    "transaction_id",
                    "customer_id",
                    "amount_from",
                    "to_currency",
                    "purpose",
                    "ensemble_score",
                ]
            ]
            report.append(top_anomalies.to_string(index=False))

            report.append("\n## Regulatory Recommendations")
            report.append(
                "1. File Suspicious Activity Reports (SARs) for transactions with score > 0.8"
            )
            report.append(
                "2. Enhanced due diligence for customers with multiple high-risk FX transactions"
            )
            report.append("3. Review exchange rate policies to prevent manipulation")
            report.append("4. Implement real-time monitoring for structuring patterns")
            report.append(
                "5. Verify business purpose for large commercial FX transactions"
            )

        report_text = "\n".join(report)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "fx_anomaly_report.md"), "w") as f:
                f.write(report_text)

        print(report_text)
        return report_text


def main():
    """Main execution function."""
    detector = FXAnomalyDetector()

    # Load data
    data_path = "../datasets/fx_transactions.csv"
    df = detector.load_data(data_path)

    # Detect anomalies
    print("Detecting anomalies in FX transactions...")
    results_df, scores = detector.detect_anomalies(df)

    # Analyze results
    anomalies = detector.analyze_anomalies(results_df)

    # Generate report
    detector.generate_report(results_df, anomalies, output_dir="../outputs")


if __name__ == "__main__":
    main()
