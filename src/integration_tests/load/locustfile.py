"""
Comprehensive Load Testing Suite for Pynomaly
Using Locust for distributed load testing
"""

import random
import time
from datetime import datetime, timedelta

from locust import HttpUser, between, events, task


class PynomaliUser(HttpUser):
    """Base user class for Pynomaly load testing"""

    wait_time = between(1, 3)
    host = "http://localhost:8000"

    def on_start(self):
        """Setup user session"""
        self.api_key = None
        self.user_id = f"loadtest_user_{random.randint(1000, 9999)}"
        self.dataset_id = None
        self.model_id = None

        # Authenticate user
        self.authenticate()

    def authenticate(self):
        """Authenticate user and get API key"""
        auth_payload = {"username": self.user_id, "password": "loadtest_password"}

        with self.client.post(
            "/auth/login", json=auth_payload, catch_response=True
        ) as response:
            if response.status_code == 200:
                self.api_key = response.json().get("api_key")
                self.client.headers.update({"Authorization": f"Bearer {self.api_key}"})
            else:
                # Create user if doesn't exist
                self.create_user()

    def create_user(self):
        """Create a new user for testing"""
        user_payload = {
            "username": self.user_id,
            "password": "loadtest_password",
            "email": f"{self.user_id}@loadtest.com",
        }

        with self.client.post(
            "/auth/register", json=user_payload, catch_response=True
        ) as response:
            if response.status_code == 201:
                self.authenticate()
            else:
                response.failure("Failed to create user")

    def generate_time_series_data(self, size: int = 1000) -> list[dict]:
        """Generate sample time series data for testing"""
        base_time = datetime.now() - timedelta(hours=size)
        data = []

        for i in range(size):
            timestamp = base_time + timedelta(minutes=i)

            # Generate normal data with some anomalies
            if random.random() < 0.05:  # 5% anomalies
                value = random.gauss(10, 2)  # Anomalous values
            else:
                value = random.gauss(0, 1)  # Normal values

            data.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "value": value,
                    "features": {
                        "feature_1": random.random(),
                        "feature_2": random.random(),
                        "feature_3": random.random(),
                    },
                }
            )

        return data

    def generate_batch_data(self, batch_size: int = 100) -> list[dict]:
        """Generate batch data for processing"""
        return [
            {
                "id": f"batch_item_{i}",
                "value": random.gauss(0, 1),
                "timestamp": datetime.now().isoformat(),
            }
            for i in range(batch_size)
        ]


class AnomalyDetectionUser(PynomaliUser):
    """User focused on anomaly detection operations"""

    weight = 60  # 60% of users

    @task(3)
    def detect_anomalies_single(self):
        """Detect anomalies in single data point"""
        data_point = {
            "timestamp": datetime.now().isoformat(),
            "value": random.gauss(0, 1),
            "features": {
                "feature_1": random.random(),
                "feature_2": random.random(),
                "feature_3": random.random(),
            },
        }

        with self.client.post(
            "/api/v1/detect", json=data_point, catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "is_anomaly" in result and "score" in result:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Detection failed: {response.status_code}")

    @task(2)
    def detect_anomalies_batch(self):
        """Detect anomalies in batch of data points"""
        batch_data = self.generate_batch_data(50)

        with self.client.post(
            "/api/v1/detect/batch", json={"data": batch_data}, catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "results" in result and len(result["results"]) == len(batch_data):
                    response.success()
                else:
                    response.failure("Invalid batch response format")
            else:
                response.failure(f"Batch detection failed: {response.status_code}")

    @task(1)
    def detect_anomalies_streaming(self):
        """Simulate streaming anomaly detection"""
        stream_data = {
            "stream_id": f"stream_{random.randint(1, 10)}",
            "data_points": self.generate_batch_data(10),
        }

        with self.client.post(
            "/api/v1/detect/stream", json=stream_data, catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Streaming detection failed: {response.status_code}")


class ModelTrainingUser(PynomaliUser):
    """User focused on model training operations"""

    weight = 20  # 20% of users

    @task(2)
    def upload_dataset(self):
        """Upload training dataset"""
        if not self.dataset_id:
            dataset = {
                "name": f"loadtest_dataset_{random.randint(1000, 9999)}",
                "description": "Load testing dataset",
                "data": self.generate_time_series_data(500),
            }

            with self.client.post(
                "/api/v1/datasets", json=dataset, catch_response=True
            ) as response:
                if response.status_code == 201:
                    self.dataset_id = response.json().get("dataset_id")
                    response.success()
                else:
                    response.failure(f"Dataset upload failed: {response.status_code}")

    @task(1)
    def train_model(self):
        """Train anomaly detection model"""
        if not self.dataset_id:
            self.upload_dataset()
            return

        training_config = {
            "dataset_id": self.dataset_id,
            "algorithm": random.choice(
                ["isolation_forest", "one_class_svm", "local_outlier_factor"]
            ),
            "parameters": {
                "contamination": 0.1,
                "n_estimators": 100,
                "max_samples": "auto",
            },
        }

        with self.client.post(
            "/api/v1/train", json=training_config, catch_response=True
        ) as response:
            if response.status_code == 202:  # Accepted for async processing
                self.model_id = response.json().get("model_id")
                response.success()
            else:
                response.failure(f"Model training failed: {response.status_code}")

    @task(1)
    def check_training_status(self):
        """Check model training status"""
        if not self.model_id:
            return

        with self.client.get(
            f"/api/v1/models/{self.model_id}/status", catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status check failed: {response.status_code}")


class DataManagementUser(PynomaliUser):
    """User focused on data management operations"""

    weight = 15  # 15% of users

    @task(2)
    def list_datasets(self):
        """List available datasets"""
        with self.client.get("/api/v1/datasets", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Dataset listing failed: {response.status_code}")

    @task(2)
    def get_dataset_info(self):
        """Get dataset information"""
        dataset_id = f"dataset_{random.randint(1, 100)}"

        with self.client.get(
            f"/api/v1/datasets/{dataset_id}", catch_response=True
        ) as response:
            if response.status_code in [
                200,
                404,
            ]:  # 404 is acceptable for non-existent datasets
                response.success()
            else:
                response.failure(
                    f"Dataset info retrieval failed: {response.status_code}"
                )

    @task(1)
    def delete_dataset(self):
        """Delete dataset"""
        if self.dataset_id:
            with self.client.delete(
                f"/api/v1/datasets/{self.dataset_id}", catch_response=True
            ) as response:
                if response.status_code == 204:
                    self.dataset_id = None
                    response.success()
                else:
                    response.failure(f"Dataset deletion failed: {response.status_code}")


class MonitoringUser(PynomaliUser):
    """User focused on monitoring and health checks"""

    weight = 5  # 5% of users

    @task(3)
    def health_check(self):
        """Check application health"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(2)
    def metrics_check(self):
        """Check metrics endpoint"""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics check failed: {response.status_code}")

    @task(1)
    def system_info(self):
        """Get system information"""
        with self.client.get("/api/v1/system/info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"System info failed: {response.status_code}")


# Event handlers for custom metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Custom request handler for additional metrics"""
    if exception:
        print(f"Request failed: {request_type} {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Test start handler"""
    print("Load test starting...")
    print(f"Host: {environment.host}")
    print(f"Users: {environment.runner.target_user_count}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Test stop handler"""
    print("Load test completed!")

    # Print summary statistics
    stats = environment.runner.stats
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Failed requests: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"99th percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    print(f"Requests per second: {stats.total.total_rps:.2f}")


# Custom user classes for specific scenarios
class HighVolumeUser(AnomalyDetectionUser):
    """User that generates high volume of requests"""

    wait_time = between(0.1, 0.5)  # Very short wait time

    @task(5)
    def rapid_fire_detection(self):
        """Rapid fire anomaly detection"""
        for _ in range(5):
            self.detect_anomalies_single()
            time.sleep(0.1)


class StressTestUser(PynomaliUser):
    """User for stress testing with heavy operations"""

    wait_time = between(0.5, 1.0)

    @task(1)
    def heavy_computation(self):
        """Trigger heavy computational task"""
        large_dataset = self.generate_time_series_data(5000)

        with self.client.post(
            "/api/v1/analyze/comprehensive",
            json={"data": large_dataset},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Heavy computation failed: {response.status_code}")


class LongRunningUser(PynomaliUser):
    """User that performs long-running operations"""

    wait_time = between(5, 10)

    @task(1)
    def long_running_task(self):
        """Simulate long-running task"""
        config = {
            "operation": "deep_analysis",
            "data_size": 10000,
            "complexity": "high",
        }

        with self.client.post(
            "/api/v1/tasks/long-running", json=config, catch_response=True, timeout=30
        ) as response:
            if response.status_code in [200, 202]:
                response.success()
            else:
                response.failure(f"Long-running task failed: {response.status_code}")


# Load test scenarios
class LoadTestScenario1(AnomalyDetectionUser):
    """Scenario 1: Basic detection workload"""

    pass


class LoadTestScenario2(ModelTrainingUser):
    """Scenario 2: Model training workload"""

    pass


class LoadTestScenario3(DataManagementUser):
    """Scenario 3: Data management workload"""

    pass


class LoadTestScenario4(MonitoringUser):
    """Scenario 4: Monitoring workload"""

    pass


# Performance test user for specific metrics
class PerformanceTestUser(PynomaliUser):
    """User for performance testing with specific metrics"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_times = []
        self.error_count = 0
        self.success_count = 0

    @task(1)
    def performance_test_endpoint(self):
        """Test specific endpoint for performance metrics"""
        start_time = time.time()

        with self.client.get(
            "/api/v1/performance-test", catch_response=True
        ) as response:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds

            self.response_times.append(response_time)

            if response.status_code == 200:
                self.success_count += 1
                response.success()
            else:
                self.error_count += 1
                response.failure(f"Performance test failed: {response.status_code}")

    def on_stop(self):
        """Called when user stops"""
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            print(f"User {self.user_id} - Avg response time: {avg_response_time:.2f}ms")
            print(
                f"User {self.user_id} - Success rate: {self.success_count}/{self.success_count + self.error_count}"
            )


# For distributed testing
class DistributedTestUser(PynomaliUser):
    """User for distributed load testing"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker_id = f"worker_{random.randint(1, 1000)}"

    @task(1)
    def distributed_task(self):
        """Task that simulates distributed workload"""
        payload = {
            "worker_id": self.worker_id,
            "task_type": "distributed_processing",
            "data": self.generate_batch_data(100),
        }

        with self.client.post(
            "/api/v1/distributed/process", json=payload, catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Distributed task failed: {response.status_code}")


# User for testing specific API endpoints
class APITestUser(PynomaliUser):
    """User for testing specific API endpoints"""

    @task(1)
    def test_swagger_docs(self):
        """Test Swagger documentation endpoint"""
        with self.client.get("/docs", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Swagger docs failed: {response.status_code}")

    @task(1)
    def test_openapi_spec(self):
        """Test OpenAPI specification endpoint"""
        with self.client.get("/openapi.json", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"OpenAPI spec failed: {response.status_code}")


# Main user classes for different load patterns
class NormalLoadUser(AnomalyDetectionUser):
    """Normal load user (default)"""

    pass


class PeakLoadUser(HighVolumeUser):
    """Peak load user"""

    pass


class SustainedLoadUser(LongRunningUser):
    """Sustained load user"""

    pass


# User class selection based on test configuration
def get_user_class(test_type: str = "normal"):
    """Get user class based on test type"""
    user_classes = {
        "normal": NormalLoadUser,
        "peak": PeakLoadUser,
        "sustained": SustainedLoadUser,
        "stress": StressTestUser,
        "performance": PerformanceTestUser,
        "distributed": DistributedTestUser,
        "api": APITestUser,
    }

    return user_classes.get(test_type, NormalLoadUser)


# Configuration for different test scenarios
TEST_SCENARIOS = {
    "basic": {
        "user_classes": [AnomalyDetectionUser, MonitoringUser],
        "spawn_rate": 10,
        "users": 50,
        "duration": "5m",
    },
    "comprehensive": {
        "user_classes": [
            AnomalyDetectionUser,
            ModelTrainingUser,
            DataManagementUser,
            MonitoringUser,
        ],
        "spawn_rate": 20,
        "users": 100,
        "duration": "30m",
    },
    "stress": {
        "user_classes": [StressTestUser, HighVolumeUser],
        "spawn_rate": 50,
        "users": 200,
        "duration": "10m",
    },
    "endurance": {
        "user_classes": [LongRunningUser, SustainedLoadUser],
        "spawn_rate": 5,
        "users": 25,
        "duration": "60m",
    },
}


if __name__ == "__main__":
    # This allows the script to be run directly for testing
    print("Pynomaly Load Testing Suite")
    print("Available test scenarios:", list(TEST_SCENARIOS.keys()))
    print("Use with Locust: locust -f locustfile.py --host=http://localhost:8000")
