#!/usr/bin/env python3
"""
Banking Sample Data Generator
Generates realistic banking datasets with embedded anomalies for testing anomaly detection algorithms.
"""

import os
import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


class BankingDataGenerator:
    """Generates realistic banking data with embedded anomalies."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        self.customers = self._generate_customer_base()

    def _generate_customer_base(self) -> list[dict[str, Any]]:
        """Generate a base of customers for transactions."""
        customers = []
        for i in range(1000):
            customers.append(
                {
                    "customer_id": f"CUST_{i:06d}",
                    "account_number": f"ACC_{i:08d}",
                    "customer_type": np.random.choice(
                        ["individual", "business"], p=[0.8, 0.2]
                    ),
                    "risk_profile": np.random.choice(
                        ["low", "medium", "high"], p=[0.6, 0.3, 0.1]
                    ),
                    "location": np.random.choice(
                        ["domestic", "international"], p=[0.85, 0.15]
                    ),
                    "account_balance": np.random.lognormal(8, 2),
                }
            )
        return customers

    def generate_deposits(self, n_records: int = 10000) -> pd.DataFrame:
        """Generate deposit transactions with anomalies."""
        data = []
        start_date = datetime.now() - timedelta(days=365)

        for i in range(n_records):
            customer = random.choice(self.customers)

            # Normal deposits
            if i < n_records * 0.95:  # 95% normal
                amount = np.random.lognormal(6, 1.5)  # Normal deposit amounts
                timestamp = start_date + timedelta(
                    days=random.randint(0, 365),
                    hours=random.randint(8, 18),  # Business hours
                    minutes=random.randint(0, 59),
                )
                source_type = np.random.choice(
                    ["cash", "check", "wire", "ach"], p=[0.3, 0.4, 0.2, 0.1]
                )

            else:  # 5% anomalies
                # Anomaly types: unusually large amounts, odd timing, suspicious sources
                anomaly_type = random.choice(
                    ["large_amount", "odd_timing", "rapid_sequence"]
                )

                if anomaly_type == "large_amount":
                    amount = np.random.uniform(50000, 500000)  # Unusually large
                    timestamp = start_date + timedelta(
                        days=random.randint(0, 365),
                        hours=random.randint(8, 18),
                        minutes=random.randint(0, 59),
                    )
                elif anomaly_type == "odd_timing":
                    amount = np.random.lognormal(6, 1)
                    timestamp = start_date + timedelta(
                        days=random.randint(0, 365),
                        hours=random.choice([2, 3, 23]),  # Odd hours
                        minutes=random.randint(0, 59),
                    )
                else:  # rapid_sequence
                    amount = np.random.uniform(
                        9000, 9999
                    )  # Just under reporting threshold
                    timestamp = start_date + timedelta(
                        days=random.randint(0, 365),
                        hours=random.randint(8, 18),
                        minutes=random.randint(0, 59),
                    )

                source_type = np.random.choice(["cash", "wire"], p=[0.7, 0.3])

            data.append(
                {
                    "transaction_id": f"DEP_{i:08d}",
                    "customer_id": customer["customer_id"],
                    "account_number": customer["account_number"],
                    "timestamp": timestamp,
                    "amount": round(amount, 2),
                    "source_type": source_type,
                    "branch_id": f"BR_{random.randint(1, 50):03d}",
                    "teller_id": (
                        f"T_{random.randint(1, 200):04d}"
                        if source_type != "ach"
                        else None
                    ),
                    "description": f"Deposit via {source_type}",
                    "is_anomaly": i >= n_records * 0.95,
                }
            )

        return pd.DataFrame(data)

    def generate_loans(self, n_records: int = 5000) -> pd.DataFrame:
        """Generate loan applications and transactions with anomalies."""
        data = []
        start_date = datetime.now() - timedelta(days=365)

        loan_types = ["mortgage", "auto", "personal", "business", "credit_line"]

        for i in range(n_records):
            customer = random.choice(self.customers)
            loan_type = np.random.choice(loan_types, p=[0.4, 0.25, 0.2, 0.1, 0.05])

            # Normal loans
            if i < n_records * 0.92:  # 92% normal
                if loan_type == "mortgage":
                    amount = np.random.normal(300000, 100000)
                    term_months = random.choice([180, 240, 300, 360])
                elif loan_type == "auto":
                    amount = np.random.normal(25000, 10000)
                    term_months = random.choice([36, 48, 60, 72])
                elif loan_type == "personal":
                    amount = np.random.normal(15000, 7500)
                    term_months = random.choice([12, 24, 36, 48])
                elif loan_type == "business":
                    amount = np.random.lognormal(11, 1)
                    term_months = random.choice([60, 84, 120])
                else:  # credit_line
                    amount = np.random.normal(50000, 25000)
                    term_months = 0  # Revolving

                interest_rate = np.random.normal(5.5, 2.0)
                credit_score = np.random.normal(720, 80)

            else:  # 8% anomalies
                # Anomalies: suspicious loan patterns
                if loan_type == "mortgage":
                    amount = np.random.uniform(800000, 2000000)  # Unusually high
                    term_months = random.choice([180, 240, 300, 360])
                    interest_rate = np.random.uniform(1.0, 3.0)  # Suspiciously low rate
                    credit_score = np.random.uniform(
                        500, 600
                    )  # Low score for high amount
                else:
                    amount = np.random.uniform(100000, 500000)  # High for non-mortgage
                    term_months = random.choice([12, 24])  # Short term for high amount
                    interest_rate = np.random.uniform(15, 25)  # Very high rate
                    credit_score = np.random.uniform(450, 550)  # Poor credit

            data.append(
                {
                    "loan_id": f"LOAN_{i:07d}",
                    "customer_id": customer["customer_id"],
                    "loan_type": loan_type,
                    "application_date": start_date
                    + timedelta(days=random.randint(0, 365)),
                    "amount": max(1000, round(amount, 2)),
                    "term_months": term_months,
                    "interest_rate": max(0.5, round(interest_rate, 2)),
                    "credit_score": max(300, min(850, int(credit_score))),
                    "loan_to_value": (
                        round(np.random.uniform(0.6, 0.95), 3)
                        if loan_type in ["mortgage", "auto"]
                        else None
                    ),
                    "debt_to_income": round(np.random.uniform(0.1, 0.5), 3),
                    "employment_years": random.randint(0, 20),
                    "status": np.random.choice(
                        ["approved", "denied", "pending"], p=[0.7, 0.2, 0.1]
                    ),
                    "is_anomaly": i >= n_records * 0.92,
                }
            )

        return pd.DataFrame(data)

    def generate_investments(self, n_records: int = 8000) -> pd.DataFrame:
        """Generate investment transactions with anomalies."""
        data = []
        start_date = datetime.now() - timedelta(days=365)

        securities = [
            "AAPL",
            "GOOGL",
            "MSFT",
            "AMZN",
            "TSLA",
            "BRK.A",
            "JPM",
            "JNJ",
            "V",
            "PG",
        ]
        transaction_types = ["buy", "sell"]

        for i in range(n_records):
            customer = random.choice(self.customers)
            security = random.choice(securities)

            # Normal investments
            if i < n_records * 0.94:  # 94% normal
                transaction_type = random.choice(transaction_types)
                quantity = random.randint(1, 1000)
                price = np.random.uniform(50, 500)
                timestamp = start_date + timedelta(
                    days=random.randint(0, 365),
                    hours=random.randint(9, 16),  # Market hours
                    minutes=random.randint(0, 59),
                )

            else:  # 6% anomalies
                # Anomalies: insider trading patterns, pump and dump, unusual volumes
                anomaly_type = random.choice(
                    ["large_volume", "timing_pattern", "price_manipulation"]
                )

                if anomaly_type == "large_volume":
                    transaction_type = random.choice(transaction_types)
                    quantity = random.randint(10000, 100000)  # Unusually large
                    price = np.random.uniform(50, 500)
                elif anomaly_type == "timing_pattern":
                    transaction_type = "buy"
                    quantity = random.randint(1000, 5000)
                    price = np.random.uniform(50, 500)
                    # Just before market close
                    timestamp = start_date + timedelta(
                        days=random.randint(0, 365),
                        hours=15,
                        minutes=random.randint(50, 59),
                    )
                else:  # price_manipulation
                    transaction_type = random.choice(transaction_types)
                    quantity = random.randint(100, 1000)
                    price = np.random.uniform(1, 10)  # Penny stock

                if "timestamp" not in locals():
                    timestamp = start_date + timedelta(
                        days=random.randint(0, 365),
                        hours=random.randint(9, 16),
                        minutes=random.randint(0, 59),
                    )

            total_value = quantity * price

            data.append(
                {
                    "transaction_id": f"INV_{i:08d}",
                    "customer_id": customer["customer_id"],
                    "security_symbol": security,
                    "transaction_type": transaction_type,
                    "quantity": quantity,
                    "price_per_share": round(price, 2),
                    "total_value": round(total_value, 2),
                    "timestamp": timestamp,
                    "commission": round(max(4.95, total_value * 0.001), 2),
                    "order_type": np.random.choice(
                        ["market", "limit", "stop"], p=[0.6, 0.3, 0.1]
                    ),
                    "account_type": np.random.choice(
                        ["taxable", "ira", "401k"], p=[0.5, 0.3, 0.2]
                    ),
                    "is_anomaly": i >= n_records * 0.94,
                }
            )

        return pd.DataFrame(data)

    def generate_fx_transactions(self, n_records: int = 3000) -> pd.DataFrame:
        """Generate foreign exchange transactions with anomalies."""
        data = []
        start_date = datetime.now() - timedelta(days=365)

        currencies = [
            "EUR",
            "GBP",
            "JPY",
            "CHF",
            "CAD",
            "AUD",
            "CNY",
            "MXN",
            "BRL",
            "INR",
        ]

        for i in range(n_records):
            customer = random.choice(self.customers)
            from_currency = "USD"
            to_currency = random.choice(currencies)

            # Normal FX transactions
            if i < n_records * 0.90:  # 90% normal
                amount_usd = np.random.lognormal(8, 1.5)  # Normal amounts
                # Simulate realistic exchange rates with some volatility
                base_rates = {
                    "EUR": 0.85,
                    "GBP": 0.75,
                    "JPY": 110,
                    "CHF": 0.92,
                    "CAD": 1.25,
                    "AUD": 1.35,
                    "CNY": 6.5,
                    "MXN": 20,
                    "BRL": 5.2,
                    "INR": 75,
                }
                volatility = np.random.normal(1, 0.02)
                exchange_rate = base_rates[to_currency] * volatility

            else:  # 10% anomalies
                # Anomalies: money laundering patterns, unusual rates, large amounts
                anomaly_type = random.choice(
                    ["large_amount", "rate_anomaly", "structuring"]
                )

                if anomaly_type == "large_amount":
                    amount_usd = np.random.uniform(100000, 1000000)
                    base_rates = {
                        "EUR": 0.85,
                        "GBP": 0.75,
                        "JPY": 110,
                        "CHF": 0.92,
                        "CAD": 1.25,
                        "AUD": 1.35,
                        "CNY": 6.5,
                        "MXN": 20,
                        "BRL": 5.2,
                        "INR": 75,
                    }
                    exchange_rate = base_rates[to_currency] * np.random.normal(1, 0.02)
                elif anomaly_type == "rate_anomaly":
                    amount_usd = np.random.lognormal(8, 1)
                    base_rates = {
                        "EUR": 0.85,
                        "GBP": 0.75,
                        "JPY": 110,
                        "CHF": 0.92,
                        "CAD": 1.25,
                        "AUD": 1.35,
                        "CNY": 6.5,
                        "MXN": 20,
                        "BRL": 5.2,
                        "INR": 75,
                    }
                    # Suspicious rate - too good to be true
                    exchange_rate = base_rates[to_currency] * np.random.uniform(
                        1.1, 1.3
                    )
                else:  # structuring
                    amount_usd = np.random.uniform(
                        9500, 9999
                    )  # Just under reporting threshold
                    base_rates = {
                        "EUR": 0.85,
                        "GBP": 0.75,
                        "JPY": 110,
                        "CHF": 0.92,
                        "CAD": 1.25,
                        "AUD": 1.35,
                        "CNY": 6.5,
                        "MXN": 20,
                        "BRL": 5.2,
                        "INR": 75,
                    }
                    exchange_rate = base_rates[to_currency] * np.random.normal(1, 0.02)

            converted_amount = amount_usd * exchange_rate

            data.append(
                {
                    "transaction_id": f"FX_{i:07d}",
                    "customer_id": customer["customer_id"],
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "amount_from": round(amount_usd, 2),
                    "amount_to": round(converted_amount, 2),
                    "exchange_rate": round(exchange_rate, 4),
                    "timestamp": start_date
                    + timedelta(
                        days=random.randint(0, 365),
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59),
                    ),
                    "purpose": np.random.choice(
                        ["travel", "business", "investment", "family"],
                        p=[0.4, 0.3, 0.2, 0.1],
                    ),
                    "method": np.random.choice(
                        ["wire", "cash", "card"], p=[0.5, 0.3, 0.2]
                    ),
                    "fee_usd": round(max(15, amount_usd * 0.005), 2),
                    "is_anomaly": i >= n_records * 0.90,
                }
            )

        return pd.DataFrame(data)

    def generate_atm_transactions(self, n_records: int = 15000) -> pd.DataFrame:
        """Generate ATM transactions with anomalies."""
        data = []
        start_date = datetime.now() - timedelta(days=365)

        transaction_types = ["withdrawal", "deposit", "balance_inquiry", "transfer"]

        for i in range(n_records):
            customer = random.choice(self.customers)

            # Normal ATM transactions
            if i < n_records * 0.97:  # 97% normal
                transaction_type = np.random.choice(
                    transaction_types, p=[0.6, 0.2, 0.15, 0.05]
                )

                if transaction_type == "withdrawal":
                    amount = random.choice(
                        [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
                    )
                elif transaction_type == "deposit":
                    amount = np.random.lognormal(4, 1)
                else:
                    amount = 0

                # Normal timing - business hours are more likely
                # Late night (0-5): 0.5%, Early morning (6-7): 2%, Business hours (8-17): 6%, Evening (18-21): 3%, Late (22-23): 1%
                probs = [0.005] * 6 + [0.02] * 2 + [0.06] * 10 + [0.03] * 4 + [0.01] * 2
                probs = np.array(probs)
                probs = probs / probs.sum()  # Normalize to sum to 1
                hour = np.random.choice(range(24), p=probs)

            else:  # 3% anomalies
                # Anomalies: card skimming, unusual patterns, location anomalies
                anomaly_type = random.choice(
                    ["multiple_attempts", "unusual_amount", "location_jump"]
                )

                transaction_type = "withdrawal"
                if anomaly_type == "multiple_attempts":
                    amount = random.choice([20, 40, 60])
                elif anomaly_type == "unusual_amount":
                    amount = random.choice(
                        [500, 600, 700, 800]
                    )  # Max daily limit attempts
                else:  # location_jump
                    amount = random.choice([100, 200])

                hour = random.randint(0, 23)

            # Generate location
            if customer["location"] == "domestic":
                atm_location = random.choice(
                    ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
                )
            else:
                atm_location = random.choice(
                    ["London", "Tokyo", "Singapore", "Dubai", "Sydney"]
                )

            data.append(
                {
                    "transaction_id": f"ATM_{i:08d}",
                    "customer_id": customer["customer_id"],
                    "card_number": f"****{random.randint(1000, 9999)}",
                    "atm_id": f"ATM_{random.randint(1, 1000):04d}",
                    "transaction_type": transaction_type,
                    "amount": round(amount, 2) if amount > 0 else None,
                    "timestamp": start_date
                    + timedelta(
                        days=random.randint(0, 365),
                        hours=int(hour),
                        minutes=random.randint(0, 59),
                    ),
                    "location": atm_location,
                    "fee": (
                        2.50
                        if customer["location"] != "domestic" or random.random() < 0.3
                        else 0
                    ),
                    "status": np.random.choice(
                        ["success", "declined", "error"], p=[0.9, 0.08, 0.02]
                    ),
                    "is_anomaly": i >= n_records * 0.97,
                }
            )

        return pd.DataFrame(data)

    def generate_card_transactions(
        self, n_records: int = 25000, card_type: str = "debit"
    ) -> pd.DataFrame:
        """Generate debit/credit card transactions with anomalies."""
        data = []
        start_date = datetime.now() - timedelta(days=365)

        merchants = [
            "Amazon",
            "Walmart",
            "Target",
            "Starbucks",
            "McDonalds",
            "Shell",
            "Exxon",
            "Home Depot",
            "CVS",
            "Walgreens",
            "Best Buy",
            "Costco",
            "Whole Foods",
            "Uber",
            "Lyft",
            "Netflix",
            "Spotify",
            "Apple",
            "Google",
            "Microsoft",
        ]

        categories = [
            "grocery",
            "gas",
            "restaurant",
            "retail",
            "entertainment",
            "travel",
            "healthcare",
            "utilities",
            "subscription",
            "other",
        ]

        for i in range(n_records):
            customer = random.choice(self.customers)
            merchant = random.choice(merchants)
            category = random.choice(categories)

            # Normal card transactions
            if i < n_records * 0.95:  # 95% normal
                # Amount based on category
                if category == "grocery":
                    amount = np.random.lognormal(4, 0.8)
                elif category == "gas":
                    amount = np.random.normal(45, 15)
                elif category == "restaurant":
                    amount = np.random.lognormal(3, 0.7)
                elif category == "retail":
                    amount = np.random.lognormal(4.5, 1)
                elif category == "travel":
                    amount = np.random.lognormal(6, 1.5)
                else:
                    amount = np.random.lognormal(3.5, 1)

                # Normal timing and location
                timestamp = start_date + timedelta(
                    days=random.randint(0, 365),
                    hours=random.randint(6, 22),
                    minutes=random.randint(0, 59),
                )
                location = customer["location"]

            else:  # 5% anomalies
                # Anomalies: fraud patterns, unusual spending
                anomaly_type = random.choice(
                    ["large_amount", "velocity", "location_anomaly", "unusual_merchant"]
                )

                if anomaly_type == "large_amount":
                    amount = np.random.uniform(2000, 10000)
                    timestamp = start_date + timedelta(
                        days=random.randint(0, 365),
                        hours=random.randint(6, 22),
                        minutes=random.randint(0, 59),
                    )
                elif anomaly_type == "velocity":
                    amount = np.random.uniform(100, 500)
                    # Multiple transactions in short time
                    base_time = start_date + timedelta(days=random.randint(0, 365))
                    timestamp = base_time + timedelta(minutes=random.randint(0, 30))
                elif anomaly_type == "location_anomaly":
                    amount = np.random.lognormal(4, 1)
                    timestamp = start_date + timedelta(
                        days=random.randint(0, 365),
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59),
                    )
                    location = (
                        "international"
                        if customer["location"] == "domestic"
                        else "domestic"
                    )
                else:  # unusual_merchant
                    amount = np.random.uniform(500, 2000)
                    merchant = random.choice(
                        ["Unknown Merchant", "Cash Advance", "Money Transfer"]
                    )
                    timestamp = start_date + timedelta(
                        days=random.randint(0, 365),
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59),
                    )

                if "location" not in locals():
                    location = customer["location"]

            data.append(
                {
                    "transaction_id": f"{card_type.upper()}_{i:08d}",
                    "customer_id": customer["customer_id"],
                    "card_number": f"****{random.randint(1000, 9999)}",
                    "merchant": merchant,
                    "category": category,
                    "amount": round(max(0.01, amount), 2),
                    "timestamp": timestamp,
                    "location": location,
                    "mcc_code": random.randint(1000, 9999),
                    "status": np.random.choice(
                        ["approved", "declined"], p=[0.95, 0.05]
                    ),
                    "card_present": random.random() < 0.7,
                    "is_anomaly": i >= n_records * 0.95,
                }
            )

        return pd.DataFrame(data)

    def generate_expense_transactions(self, n_records: int = 8000) -> pd.DataFrame:
        """Generate corporate expense transactions with anomalies."""
        data = []
        start_date = datetime.now() - timedelta(days=365)

        expense_types = [
            "travel",
            "food",
            "learning",
            "certification",
            "business_event",
            "equipment",
            "software",
        ]

        for i in range(n_records):
            customer = random.choice(
                [c for c in self.customers if c["customer_type"] == "business"]
            )
            expense_type = random.choice(expense_types)

            # Normal expenses
            if i < n_records * 0.93:  # 93% normal
                if expense_type == "travel":
                    amount = np.random.lognormal(7, 1)  # $1000 average
                    description = random.choice(
                        ["Flight to NYC", "Hotel - Chicago", "Rental Car", "Taxi"]
                    )
                elif expense_type == "food":
                    amount = np.random.lognormal(4, 0.5)  # $50 average
                    description = random.choice(
                        ["Client Dinner", "Team Lunch", "Conference Meal"]
                    )
                elif expense_type == "learning":
                    amount = np.random.lognormal(8, 0.5)  # $3000 average
                    description = random.choice(
                        ["AWS Training", "Data Science Course", "Leadership Program"]
                    )
                elif expense_type == "certification":
                    amount = np.random.normal(500, 200)
                    description = random.choice(
                        ["PMP Certification", "AWS Certification", "CPA Exam"]
                    )
                elif expense_type == "business_event":
                    amount = np.random.lognormal(9, 1)  # $8000 average
                    description = random.choice(
                        ["Trade Show", "Conference", "Client Event"]
                    )
                elif expense_type == "equipment":
                    amount = np.random.lognormal(7.5, 0.8)  # $1800 average
                    description = random.choice(
                        ["Laptop", "Monitor", "Phone", "Desk Setup"]
                    )
                else:  # software
                    amount = np.random.lognormal(6, 0.5)  # $400 average
                    description = random.choice(
                        ["Adobe License", "Microsoft Office", "Salesforce"]
                    )

            else:  # 7% anomalies
                # Anomalies: expense fraud, duplicate submissions, inflated amounts
                anomaly_type = random.choice(
                    ["inflated", "duplicate", "personal_expense"]
                )

                if anomaly_type == "inflated":
                    if expense_type == "food":
                        amount = np.random.uniform(200, 500)  # Expensive meal
                        description = "Client Dinner - Fine Dining"
                    else:
                        amount = np.random.uniform(5000, 20000)  # Inflated amount
                        description = f"{expense_type.title()} - Premium Service"
                elif anomaly_type == "duplicate":
                    amount = np.random.lognormal(6, 1)
                    description = f"{expense_type.title()} - Duplicate Submission"
                else:  # personal_expense
                    amount = np.random.uniform(100, 1000)
                    description = random.choice(
                        ["Personal Shopping", "Family Dinner", "Personal Travel"]
                    )

            data.append(
                {
                    "expense_id": f"EXP_{i:07d}",
                    "employee_id": customer["customer_id"],
                    "expense_type": expense_type,
                    "amount": round(max(1, amount), 2),
                    "description": description,
                    "date": start_date + timedelta(days=random.randint(0, 365)),
                    "submitted_date": start_date
                    + timedelta(days=random.randint(0, 365)),
                    "status": np.random.choice(
                        ["submitted", "approved", "reimbursed", "rejected"],
                        p=[0.2, 0.3, 0.4, 0.1],
                    ),
                    "receipt_provided": random.random() < 0.8,
                    "manager_id": f"MGR_{random.randint(1, 50):03d}",
                    "project_code": f"PRJ_{random.randint(1, 100):03d}",
                    "is_anomaly": i >= n_records * 0.93,
                }
            )

        return pd.DataFrame(data)

    def generate_gl_transactions(self, n_records: int = 12000) -> pd.DataFrame:
        """Generate general ledger transactions with anomalies."""
        data = []
        start_date = datetime.now() - timedelta(days=365)

        account_types = ["asset", "liability", "equity", "revenue", "expense"]
        gl_accounts = {
            "asset": [
                "1001-Cash",
                "1100-Accounts Receivable",
                "1200-Inventory",
                "1500-Equipment",
            ],
            "liability": [
                "2001-Accounts Payable",
                "2100-Accrued Expenses",
                "2200-Notes Payable",
            ],
            "equity": ["3001-Common Stock", "3100-Retained Earnings"],
            "revenue": [
                "4001-Sales Revenue",
                "4100-Interest Income",
                "4200-Service Revenue",
            ],
            "expense": [
                "5001-Salaries",
                "5100-Rent",
                "5200-Utilities",
                "5300-Depreciation",
            ],
        }

        for i in range(n_records):
            account_type = random.choice(account_types)
            account = random.choice(gl_accounts[account_type])

            # Normal GL transactions
            if i < n_records * 0.96:  # 96% normal
                if account_type in ["asset", "expense"]:
                    debit = np.random.lognormal(8, 2)
                    credit = 0
                else:
                    debit = 0
                    credit = np.random.lognormal(8, 2)

                # Business hours posting
                timestamp = start_date + timedelta(
                    days=random.randint(0, 365),
                    hours=random.randint(8, 17),
                    minutes=random.randint(0, 59),
                )

            else:  # 4% anomalies
                # Anomalies: unusual amounts, off-hours posting, imbalanced entries
                anomaly_type = random.choice(
                    ["large_amount", "off_hours", "manual_override"]
                )

                if anomaly_type == "large_amount":
                    amount = np.random.uniform(100000, 1000000)
                    if random.random() < 0.5:
                        debit, credit = amount, 0
                    else:
                        debit, credit = 0, amount
                    timestamp = start_date + timedelta(
                        days=random.randint(0, 365),
                        hours=random.randint(8, 17),
                        minutes=random.randint(0, 59),
                    )
                elif anomaly_type == "off_hours":
                    if account_type in ["asset", "expense"]:
                        debit = np.random.lognormal(8, 1)
                        credit = 0
                    else:
                        debit = 0
                        credit = np.random.lognormal(8, 1)
                    # Off-hours posting
                    timestamp = start_date + timedelta(
                        days=random.randint(0, 365),
                        hours=random.choice([22, 23, 0, 1, 2]),
                        minutes=random.randint(0, 59),
                    )
                else:  # manual_override
                    # Imbalanced entry
                    debit = np.random.lognormal(7, 1)
                    credit = debit * np.random.uniform(0.9, 1.1)  # Slight imbalance
                    timestamp = start_date + timedelta(
                        days=random.randint(0, 365),
                        hours=random.randint(8, 17),
                        minutes=random.randint(0, 59),
                    )

            data.append(
                {
                    "journal_entry_id": f"JE_{i:08d}",
                    "gl_account": account,
                    "account_type": account_type,
                    "debit_amount": round(debit, 2) if debit > 0 else None,
                    "credit_amount": round(credit, 2) if credit > 0 else None,
                    "timestamp": timestamp,
                    "description": f"{account_type.title()} transaction",
                    "reference_number": f"REF_{random.randint(100000, 999999)}",
                    "posted_by": f"USER_{random.randint(1, 50):03d}",
                    "source_system": random.choice(
                        ["ERP", "Manual", "Interface", "Automated"]
                    ),
                    "is_anomaly": i >= n_records * 0.96,
                }
            )

        return pd.DataFrame(data)

    def save_all_datasets(self, output_dir: str = "examples/banking/datasets"):
        """Generate and save all banking datasets."""
        os.makedirs(output_dir, exist_ok=True)

        datasets = {
            "deposits": self.generate_deposits(),
            "loans": self.generate_loans(),
            "investments": self.generate_investments(),
            "fx_transactions": self.generate_fx_transactions(),
            "atm_transactions": self.generate_atm_transactions(),
            "debit_card_transactions": self.generate_card_transactions(
                card_type="debit"
            ),
            "credit_card_transactions": self.generate_card_transactions(
                card_type="credit"
            ),
            "expense_transactions": self.generate_expense_transactions(),
            "gl_transactions": self.generate_gl_transactions(),
        }

        for name, df in datasets.items():
            filepath = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(filepath, index=False)
            print(
                f"Generated {name}: {len(df)} records with {df['is_anomaly'].sum()} anomalies"
            )

        print(f"\nAll datasets saved to {output_dir}")
        return datasets


if __name__ == "__main__":
    generator = BankingDataGenerator()
    datasets = generator.save_all_datasets()
