"""
Test Data Generator for Advanced UI Testing

Provides comprehensive test data generation including:
- Dynamic form data creation
- User personas and scenarios
- Mock API response data
- Performance test datasets
- Accessibility test scenarios
"""

import random
import string
from datetime import datetime, timedelta
from typing import Any

try:
    from faker import Faker

    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False


class TestDataGenerator:
    """
    Comprehensive test data generator for UI testing scenarios
    """

    def __init__(self, seed: int | None = None):
        if FAKER_AVAILABLE:
            self.faker = Faker()
            if seed:
                Faker.seed(seed)
                random.seed(seed)
        else:
            self.faker = None
            if seed:
                random.seed(seed)

        self.user_personas = {
            "data_scientist": {
                "role": "Data Scientist",
                "experience": "Expert",
                "primary_tasks": [
                    "model_training",
                    "data_analysis",
                    "experiment_tracking",
                ],
                "tools": ["jupyter", "python", "sql", "ml_libraries"],
            },
            "business_analyst": {
                "role": "Business Analyst",
                "experience": "Intermediate",
                "primary_tasks": [
                    "dashboard_viewing",
                    "report_generation",
                    "trend_analysis",
                ],
                "tools": ["excel", "tableau", "power_bi"],
            },
            "ml_engineer": {
                "role": "ML Engineer",
                "experience": "Expert",
                "primary_tasks": [
                    "model_deployment",
                    "pipeline_management",
                    "monitoring",
                ],
                "tools": ["docker", "kubernetes", "mlops_tools"],
            },
            "admin": {
                "role": "System Administrator",
                "experience": "Expert",
                "primary_tasks": [
                    "user_management",
                    "system_configuration",
                    "monitoring",
                ],
                "tools": ["admin_panels", "monitoring_tools", "user_management"],
            },
            "beginner": {
                "role": "New User",
                "experience": "Beginner",
                "primary_tasks": ["exploration", "learning", "basic_tasks"],
                "tools": ["guided_tours", "help_documentation"],
            },
        }

    def generate_user_data(self, persona: str = "data_scientist") -> dict[str, Any]:
        """Generate realistic user data for testing"""
        persona_info = self.user_personas.get(
            persona, self.user_personas["data_scientist"]
        )

        if FAKER_AVAILABLE and self.faker:
            return {
                "id": self.faker.uuid4(),
                "username": self.faker.user_name(),
                "email": self.faker.email(),
                "first_name": self.faker.first_name(),
                "last_name": self.faker.last_name(),
                "full_name": self.faker.name(),
                "role": persona_info["role"],
                "experience_level": persona_info["experience"],
                "department": self.faker.random_element(
                    ["Engineering", "Data Science", "Analytics", "Operations"]
                ),
                "created_at": self.faker.date_time_between(
                    start_date="-2y", end_date="now"
                ),
                "last_login": self.faker.date_time_between(
                    start_date="-30d", end_date="now"
                ),
                "preferences": {
                    "theme": self.faker.random_element(["light", "dark", "auto"]),
                    "language": self.faker.random_element(["en", "es", "fr", "de"]),
                    "timezone": self.faker.timezone(),
                    "notifications": self.faker.boolean(),
                },
                "persona": persona,
                "primary_tasks": persona_info["primary_tasks"],
                "preferred_tools": persona_info["tools"],
            }
        else:
            # Fallback when Faker is not available
            return {
                "id": f"user_{random.randint(1000, 9999)}",
                "username": f"testuser_{random.randint(100, 999)}",
                "email": f"user{random.randint(100, 999)}@test.com",
                "first_name": random.choice(["John", "Jane", "Alex", "Sam", "Taylor"]),
                "last_name": random.choice(
                    ["Smith", "Johnson", "Brown", "Davis", "Wilson"]
                ),
                "full_name": "Test User",
                "role": persona_info["role"],
                "experience_level": persona_info["experience"],
                "department": random.choice(
                    ["Engineering", "Data Science", "Analytics", "Operations"]
                ),
                "created_at": datetime.now() - timedelta(days=random.randint(30, 730)),
                "last_login": datetime.now() - timedelta(days=random.randint(1, 30)),
                "preferences": {
                    "theme": random.choice(["light", "dark", "auto"]),
                    "language": random.choice(["en", "es", "fr", "de"]),
                    "timezone": "UTC",
                    "notifications": random.choice([True, False]),
                },
                "persona": persona,
                "primary_tasks": persona_info["primary_tasks"],
                "preferred_tools": persona_info["tools"],
            }

    def generate_dataset_metadata(self, complexity: str = "medium") -> dict[str, Any]:
        """Generate realistic dataset metadata for testing"""
        complexity_config = {
            "simple": {
                "rows": (100, 1000),
                "cols": (5, 15),
                "features": ["numeric", "categorical"],
            },
            "medium": {
                "rows": (1000, 50000),
                "cols": (15, 50),
                "features": ["numeric", "categorical", "text", "datetime"],
            },
            "complex": {
                "rows": (50000, 1000000),
                "cols": (50, 200),
                "features": [
                    "numeric",
                    "categorical",
                    "text",
                    "datetime",
                    "json",
                    "arrays",
                ],
            },
        }

        config = complexity_config.get(complexity, complexity_config["medium"])
        n_rows = self.faker.random_int(min=config["rows"][0], max=config["rows"][1])
        n_cols = self.faker.random_int(min=config["cols"][0], max=config["cols"][1])

        return {
            "id": self.faker.uuid4(),
            "name": self.faker.random_element(
                [
                    "customer_transactions",
                    "sensor_data",
                    "user_behavior",
                    "financial_records",
                    "iot_measurements",
                    "web_analytics",
                    "sales_data",
                    "inventory_records",
                    "system_logs",
                ]
            )
            + "_"
            + self.faker.random_element(["2023", "2024", "q1", "q2", "production"]),
            "description": self.faker.text(max_nb_chars=200),
            "shape": [n_rows, n_cols],
            "n_samples": n_rows,
            "n_features": n_cols,
            "feature_names": [f"feature_{i}" for i in range(n_cols)],
            "feature_types": self.faker.random_choices(
                config["features"], length=n_cols
            ),
            "size_mb": round(
                (n_rows * n_cols * 8) / (1024 * 1024), 2
            ),  # Rough estimate
            "created_at": self.faker.date_time_between(
                start_date="-1y", end_date="now"
            ),
            "updated_at": self.faker.date_time_between(
                start_date="-30d", end_date="now"
            ),
            "source": self.faker.random_element(
                ["upload", "api", "database", "stream"]
            ),
            "format": self.faker.random_element(["csv", "parquet", "json", "avro"]),
            "quality_score": round(self.faker.random.uniform(0.7, 1.0), 2),
            "has_target": self.faker.boolean(chance_of_getting_true=70),
            "target_column": "target"
            if self.faker.boolean(chance_of_getting_true=70)
            else None,
            "missing_values": self.faker.random_int(min=0, max=int(n_rows * 0.1)),
            "duplicate_rows": self.faker.random_int(min=0, max=int(n_rows * 0.05)),
            "tags": self.faker.random_choices(
                ["production", "testing", "experiment", "clean", "raw", "processed"],
                length=self.faker.random_int(min=1, max=3),
            ),
        }

    def generate_anomaly_detection_config(self) -> dict[str, Any]:
        """Generate anomaly detection configuration for testing"""
        algorithms = [
            "IsolationForest",
            "LocalOutlierFactor",
            "OneClassSVM",
            "EllipticEnvelope",
            "AutoEncoder",
            "LSTM",
            "DeepSVDD",
            "KMeans",
            "DBSCAN",
            "PCA",
        ]

        return {
            "id": self.faker.uuid4(),
            "name": f"detector_{self.faker.word()}_{self.faker.random_int(min=1, max=999)}",
            "algorithm": self.faker.random_element(algorithms),
            "parameters": {
                "contamination": round(self.faker.random.uniform(0.01, 0.1), 3),
                "random_state": self.faker.random_int(min=1, max=10000),
                "n_estimators": self.faker.random_element([50, 100, 200, 500])
                if "Forest" in self.faker.random_element(algorithms)
                else None,
                "learning_rate": round(self.faker.random.uniform(0.001, 0.1), 4)
                if "LSTM" in self.faker.random_element(algorithms)
                else None,
            },
            "training_config": {
                "validation_split": round(self.faker.random.uniform(0.1, 0.3), 2),
                "epochs": self.faker.random_int(min=10, max=200),
                "batch_size": self.faker.random_element([32, 64, 128, 256]),
                "early_stopping": self.faker.boolean(chance_of_getting_true=80),
            },
            "created_at": self.faker.date_time_between(
                start_date="-6m", end_date="now"
            ),
            "status": self.faker.random_element(
                ["draft", "training", "trained", "deployed", "failed"]
            ),
            "accuracy_metrics": {
                "precision": round(self.faker.random.uniform(0.6, 0.95), 3),
                "recall": round(self.faker.random.uniform(0.6, 0.95), 3),
                "f1_score": round(self.faker.random.uniform(0.6, 0.95), 3),
                "auc_roc": round(self.faker.random.uniform(0.7, 0.98), 3),
            }
            if self.faker.boolean(chance_of_getting_true=60)
            else None,
        }

    def generate_form_test_data(self, form_type: str) -> dict[str, Any]:
        """Generate test data for different form types"""
        form_generators = {
            "login": self._generate_login_data,
            "registration": self._generate_registration_data,
            "dataset_upload": self._generate_dataset_upload_data,
            "detector_config": self._generate_detector_config_data,
            "user_profile": self._generate_user_profile_data,
            "search": self._generate_search_data,
            "feedback": self._generate_feedback_data,
        }

        generator = form_generators.get(form_type, self._generate_generic_form_data)
        return generator()

    def _generate_login_data(self) -> dict[str, Any]:
        """Generate login form test data"""
        return {
            "valid": {
                "username": "testuser@example.com",
                "password": "TestPassword123!",
            },
            "invalid_email": {
                "username": "invalid-email",
                "password": "TestPassword123!",
            },
            "invalid_password": {
                "username": "testuser@example.com",
                "password": "weak",
            },
            "empty_fields": {"username": "", "password": ""},
            "sql_injection": {
                "username": "admin'; DROP TABLE users; --",
                "password": "password",
            },
            "xss_attempt": {
                "username": "<script>alert('xss')</script>",
                "password": "password",
            },
        }

    def _generate_registration_data(self) -> dict[str, Any]:
        """Generate registration form test data"""
        return {
            "valid": {
                "first_name": self.faker.first_name(),
                "last_name": self.faker.last_name(),
                "email": self.faker.email(),
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!",
                "terms_accepted": True,
            },
            "password_mismatch": {
                "first_name": self.faker.first_name(),
                "last_name": self.faker.last_name(),
                "email": self.faker.email(),
                "password": "SecurePass123!",
                "confirm_password": "DifferentPass123!",
                "terms_accepted": True,
            },
            "weak_password": {
                "first_name": self.faker.first_name(),
                "last_name": self.faker.last_name(),
                "email": self.faker.email(),
                "password": "123",
                "confirm_password": "123",
                "terms_accepted": True,
            },
            "terms_not_accepted": {
                "first_name": self.faker.first_name(),
                "last_name": self.faker.last_name(),
                "email": self.faker.email(),
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!",
                "terms_accepted": False,
            },
        }

    def _generate_dataset_upload_data(self) -> dict[str, Any]:
        """Generate dataset upload form test data"""
        return {
            "valid": {
                "name": f"test_dataset_{self.faker.random_int(min=1, max=999)}",
                "description": self.faker.text(max_nb_chars=200),
                "file_format": "csv",
                "has_header": True,
                "delimiter": ",",
                "target_column": "target",
            },
            "no_name": {
                "name": "",
                "description": self.faker.text(max_nb_chars=200),
                "file_format": "csv",
                "has_header": True,
                "delimiter": ",",
                "target_column": "target",
            },
            "invalid_format": {
                "name": f"test_dataset_{self.faker.random_int(min=1, max=999)}",
                "description": self.faker.text(max_nb_chars=200),
                "file_format": "invalid",
                "has_header": True,
                "delimiter": ",",
                "target_column": "target",
            },
            "special_characters": {
                "name": "test_dataset_<script>alert('xss')</script>",
                "description": "Description with special chars: !@#$%^&*()",
                "file_format": "csv",
                "has_header": True,
                "delimiter": ",",
                "target_column": "target",
            },
        }

    def _generate_detector_config_data(self) -> dict[str, Any]:
        """Generate detector configuration form test data"""
        return {
            "valid": {
                "name": f"detector_{self.faker.word()}",
                "algorithm": "IsolationForest",
                "contamination": 0.1,
                "random_state": 42,
                "dataset_id": self.faker.uuid4(),
            },
            "invalid_contamination": {
                "name": f"detector_{self.faker.word()}",
                "algorithm": "IsolationForest",
                "contamination": 1.5,  # Invalid: > 1
                "random_state": 42,
                "dataset_id": self.faker.uuid4(),
            },
            "negative_values": {
                "name": f"detector_{self.faker.word()}",
                "algorithm": "IsolationForest",
                "contamination": -0.1,  # Invalid: negative
                "random_state": -1,  # Invalid: negative
                "dataset_id": self.faker.uuid4(),
            },
            "missing_required": {
                "name": "",  # Missing required field
                "algorithm": "",  # Missing required field
                "contamination": 0.1,
                "random_state": 42,
                "dataset_id": "",  # Missing required field
            },
        }

    def _generate_user_profile_data(self) -> dict[str, Any]:
        """Generate user profile form test data"""
        return {
            "valid": {
                "first_name": self.faker.first_name(),
                "last_name": self.faker.last_name(),
                "email": self.faker.email(),
                "phone": self.faker.phone_number(),
                "company": self.faker.company(),
                "job_title": self.faker.job(),
                "timezone": "UTC",
                "language": "en",
                "notifications_email": True,
                "notifications_push": False,
            },
            "invalid_email": {
                "first_name": self.faker.first_name(),
                "last_name": self.faker.last_name(),
                "email": "invalid-email-format",
                "phone": self.faker.phone_number(),
                "company": self.faker.company(),
                "job_title": self.faker.job(),
                "timezone": "UTC",
                "language": "en",
            },
            "xss_injection": {
                "first_name": "<script>alert('xss')</script>",
                "last_name": self.faker.last_name(),
                "email": self.faker.email(),
                "company": "<img src='x' onerror='alert(1)'>",
                "job_title": self.faker.job(),
            },
        }

    def _generate_search_data(self) -> dict[str, Any]:
        """Generate search form test data"""
        return {
            "valid": {
                "query": "anomaly detection",
                "filters": ["algorithm:IsolationForest", "status:trained"],
                "sort_by": "created_at",
                "sort_order": "desc",
            },
            "special_characters": {
                "query": "search with !@#$%^&*()_+ special chars",
                "filters": [],
                "sort_by": "name",
                "sort_order": "asc",
            },
            "sql_injection": {
                "query": "'; DROP TABLE datasets; --",
                "filters": [],
                "sort_by": "name",
                "sort_order": "asc",
            },
            "very_long_query": {
                "query": "a" * 1000,  # Very long search query
                "filters": [],
                "sort_by": "name",
                "sort_order": "asc",
            },
            "empty_query": {
                "query": "",
                "filters": [],
                "sort_by": "name",
                "sort_order": "asc",
            },
        }

    def _generate_feedback_data(self) -> dict[str, Any]:
        """Generate feedback form test data"""
        return {
            "valid": {
                "rating": 4,
                "title": "Great platform!",
                "message": self.faker.text(max_nb_chars=500),
                "category": "feature_request",
                "email": self.faker.email(),
            },
            "negative_feedback": {
                "rating": 1,
                "title": "Needs improvement",
                "message": "The interface is confusing and slow. Please improve performance.",
                "category": "bug_report",
                "email": self.faker.email(),
            },
            "html_injection": {
                "rating": 3,
                "title": "<h1>HTML Injection Test</h1>",
                "message": "<script>alert('xss')</script>This is a test of HTML injection",
                "category": "security_test",
                "email": self.faker.email(),
            },
            "very_long_message": {
                "rating": 3,
                "title": "Long feedback",
                "message": self.faker.text(max_nb_chars=5000),  # Very long message
                "category": "other",
                "email": self.faker.email(),
            },
        }

    def _generate_generic_form_data(self) -> dict[str, Any]:
        """Generate generic form test data"""
        return {
            "valid": {
                "text_field": self.faker.text(max_nb_chars=100),
                "email_field": self.faker.email(),
                "number_field": self.faker.random_int(min=1, max=100),
                "date_field": self.faker.date(),
                "boolean_field": self.faker.boolean(),
            },
            "boundary_values": {
                "text_field": "a" * 255,  # Max length test
                "number_field": 999999999,  # Large number
                "date_field": "2030-12-31",  # Future date
            },
            "edge_cases": {
                "text_field": "",  # Empty string
                "number_field": 0,  # Zero value
                "date_field": "1900-01-01",  # Very old date
            },
        }

    def generate_performance_test_scenarios(self) -> list[dict[str, Any]]:
        """Generate performance testing scenarios"""
        return [
            {
                "name": "Light Load",
                "description": "Minimal concurrent users for baseline performance",
                "concurrent_users": 1,
                "duration_seconds": 30,
                "actions_per_user": 10,
                "think_time_seconds": 1,
            },
            {
                "name": "Normal Load",
                "description": "Expected typical usage patterns",
                "concurrent_users": 10,
                "duration_seconds": 60,
                "actions_per_user": 20,
                "think_time_seconds": 2,
            },
            {
                "name": "Peak Load",
                "description": "High but realistic peak usage",
                "concurrent_users": 50,
                "duration_seconds": 120,
                "actions_per_user": 30,
                "think_time_seconds": 1,
            },
            {
                "name": "Stress Test",
                "description": "Beyond normal capacity to test limits",
                "concurrent_users": 100,
                "duration_seconds": 180,
                "actions_per_user": 50,
                "think_time_seconds": 0.5,
            },
            {
                "name": "Spike Test",
                "description": "Sudden increase in load",
                "concurrent_users": 200,
                "duration_seconds": 60,
                "actions_per_user": 15,
                "think_time_seconds": 0.1,
            },
        ]

    def generate_accessibility_test_scenarios(self) -> list[dict[str, Any]]:
        """Generate accessibility testing scenarios"""
        return [
            {
                "name": "Keyboard Only Navigation",
                "description": "Navigate entire application using only keyboard",
                "assistive_technology": "keyboard",
                "user_profile": "motor_impairment",
                "test_actions": [
                    "tab_through_all_elements",
                    "access_all_interactive_elements",
                    "use_skip_links",
                    "operate_all_controls",
                ],
            },
            {
                "name": "Screen Reader Navigation",
                "description": "Navigate using screen reader simulation",
                "assistive_technology": "screen_reader",
                "user_profile": "visual_impairment",
                "test_actions": [
                    "read_page_structure",
                    "navigate_by_headings",
                    "read_form_labels",
                    "understand_page_content",
                ],
            },
            {
                "name": "Voice Control",
                "description": "Navigate using voice commands",
                "assistive_technology": "voice_control",
                "user_profile": "motor_impairment",
                "test_actions": [
                    "voice_activate_buttons",
                    "voice_fill_forms",
                    "voice_navigate_menus",
                ],
            },
            {
                "name": "High Contrast Mode",
                "description": "Test with high contrast settings",
                "assistive_technology": "high_contrast",
                "user_profile": "low_vision",
                "test_actions": [
                    "verify_contrast_ratios",
                    "check_color_independence",
                    "validate_text_readability",
                ],
            },
            {
                "name": "Magnification",
                "description": "Test with screen magnification",
                "assistive_technology": "magnification",
                "user_profile": "low_vision",
                "test_actions": [
                    "zoom_to_200_percent",
                    "zoom_to_400_percent",
                    "verify_no_horizontal_scroll",
                    "check_content_reflow",
                ],
            },
        ]

    def generate_error_scenarios(self) -> list[dict[str, Any]]:
        """Generate error handling test scenarios"""
        return [
            {
                "name": "Network Error",
                "description": "Simulate network connectivity issues",
                "error_type": "network",
                "trigger": "disconnect_network",
                "expected_behavior": "show_offline_message",
            },
            {
                "name": "Server Error 500",
                "description": "Simulate internal server error",
                "error_type": "server",
                "trigger": "force_500_response",
                "expected_behavior": "show_error_page",
            },
            {
                "name": "Rate Limit Exceeded",
                "description": "Simulate API rate limiting",
                "error_type": "rate_limit",
                "trigger": "exceed_request_limit",
                "expected_behavior": "show_rate_limit_message",
            },
            {
                "name": "Authentication Expired",
                "description": "Simulate expired session",
                "error_type": "auth",
                "trigger": "expire_session_token",
                "expected_behavior": "redirect_to_login",
            },
            {
                "name": "Validation Error",
                "description": "Submit invalid form data",
                "error_type": "validation",
                "trigger": "submit_invalid_data",
                "expected_behavior": "show_validation_errors",
            },
            {
                "name": "File Upload Error",
                "description": "Upload invalid or oversized file",
                "error_type": "upload",
                "trigger": "upload_invalid_file",
                "expected_behavior": "show_upload_error",
            },
        ]

    def generate_browser_compatibility_matrix(self) -> list[dict[str, Any]]:
        """Generate browser compatibility test matrix"""
        return [
            {
                "browser": "chromium",
                "version": "latest",
                "platform": "desktop",
                "viewport": {"width": 1920, "height": 1080},
                "features_to_test": ["all"],
            },
            {
                "browser": "firefox",
                "version": "latest",
                "platform": "desktop",
                "viewport": {"width": 1920, "height": 1080},
                "features_to_test": ["all"],
            },
            {
                "browser": "webkit",
                "version": "latest",
                "platform": "desktop",
                "viewport": {"width": 1920, "height": 1080},
                "features_to_test": ["all"],
            },
            {
                "browser": "chromium",
                "version": "latest",
                "platform": "mobile",
                "viewport": {"width": 375, "height": 667},
                "features_to_test": ["responsive", "touch", "mobile_navigation"],
            },
            {
                "browser": "webkit",
                "version": "latest",
                "platform": "mobile",
                "viewport": {"width": 414, "height": 896},
                "features_to_test": ["responsive", "touch", "mobile_navigation"],
            },
        ]

    def generate_random_string(
        self, length: int = 10, include_special: bool = False
    ) -> str:
        """Generate random string for testing"""
        chars = string.ascii_letters + string.digits
        if include_special:
            chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"

        return "".join(random.choice(chars) for _ in range(length))

    def generate_large_dataset_simulation(self, size: str = "medium") -> dict[str, Any]:
        """Generate large dataset parameters for performance testing"""
        size_configs = {
            "small": {"rows": 1000, "cols": 10, "size_mb": 0.1},
            "medium": {"rows": 100000, "cols": 50, "size_mb": 50},
            "large": {"rows": 1000000, "cols": 100, "size_mb": 500},
            "xlarge": {"rows": 10000000, "cols": 200, "size_mb": 5000},
        }

        config = size_configs.get(size, size_configs["medium"])

        return {
            "name": f"performance_test_dataset_{size}",
            "estimated_rows": config["rows"],
            "estimated_columns": config["cols"],
            "estimated_size_mb": config["size_mb"],
            "data_types": ["float64", "int64", "object", "datetime64"],
            "has_missing_values": True,
            "complexity": size,
            "expected_load_time_seconds": config["size_mb"] / 10,  # Rough estimate
        }
