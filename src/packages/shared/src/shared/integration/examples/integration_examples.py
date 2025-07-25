"""Examples demonstrating cross-domain integration patterns.

This module provides practical examples of how to implement and use
the cross-domain integration patterns in real scenarios.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum

from ..cross_domain_patterns import (
    CrossDomainIntegrationManager, DomainService, CrossDomainMessage,
    IntegrationContext, IntegrationResult, IntegrationStatus, MessageType,
    SagaStep, SagaOrchestrator, get_integration_manager
)
from ..domain_adapters import (
    StandardDomainAdapter, FieldMapping, register_standard_adapter,
    EnrichmentAdapter, register_enrichment_adapter
)


# Example domain entities
@dataclass
class UserProfile:
    """User profile entity from user management domain."""
    user_id: str
    username: str
    email: str
    full_name: str
    created_at: datetime
    preferences: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "created_at": self.created_at.isoformat(),
            "preferences": self.preferences
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            email=data["email"],
            full_name=data["full_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            preferences=data.get("preferences", {})
        )


@dataclass
class DataQualityReport:
    """Data quality report from data quality domain."""
    report_id: str
    dataset_id: str
    quality_score: float
    issues_found: List[Dict[str, Any]]
    generated_at: datetime
    generated_by: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "dataset_id": self.dataset_id,
            "quality_score": self.quality_score,
            "issues_found": self.issues_found,
            "generated_at": self.generated_at.isoformat(),
            "generated_by": self.generated_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataQualityReport':
        return cls(
            report_id=data["report_id"],
            dataset_id=data["dataset_id"],
            quality_score=data["quality_score"],
            issues_found=data.get("issues_found", []),
            generated_at=datetime.fromisoformat(data["generated_at"]),
            generated_by=data["generated_by"]
        )


@dataclass
class MLModelMetrics:
    """ML model metrics from machine learning domain."""
    model_id: str
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    evaluated_at: datetime
    dataset_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "evaluated_at": self.evaluated_at.isoformat(),
            "dataset_size": self.dataset_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MLModelMetrics':
        return cls(
            model_id=data["model_id"],
            model_name=data["model_name"],
            accuracy=data["accuracy"],
            precision=data["precision"],
            recall=data["recall"],
            f1_score=data["f1_score"],
            evaluated_at=datetime.fromisoformat(data["evaluated_at"]),
            dataset_size=data["dataset_size"]
        )


# Example domain services
class UserManagementService(DomainService):
    """Example user management domain service."""
    
    def __init__(self):
        self.domain_name = "user_management"
        self._users: Dict[str, UserProfile] = {}
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample users."""
        sample_users = [
            UserProfile(
                user_id="user_1",
                username="alice_smith",
                email="alice@company.com",
                full_name="Alice Smith",
                created_at=datetime.now(timezone.utc) - timedelta(days=30),
                preferences={"theme": "dark", "notifications": True}
            ),
            UserProfile(
                user_id="user_2",
                username="bob_jones",
                email="bob@company.com",
                full_name="Bob Jones",
                created_at=datetime.now(timezone.utc) - timedelta(days=15),
                preferences={"theme": "light", "notifications": False}
            )
        ]
        
        for user in sample_users:
            self._users[user.user_id] = user
    
    async def handle_message(self, message: CrossDomainMessage) -> IntegrationResult[Any]:
        """Handle incoming cross-domain messages."""
        try:
            if message.operation == "get_user_profile":
                user_id = message.payload.get("user_id")
                user = self._users.get(user_id)
                
                if user:
                    return IntegrationResult(
                        status=IntegrationStatus.COMPLETED,
                        data=user.to_dict(),
                        message="User profile retrieved successfully"
                    )
                else:
                    return IntegrationResult(
                        status=IntegrationStatus.FAILED,
                        error="User not found",
                        message=f"User {user_id} not found"
                    )
            
            elif message.operation == "update_user_preferences":
                user_id = message.payload.get("user_id")
                new_preferences = message.payload.get("preferences", {})
                
                user = self._users.get(user_id)
                if user:
                    user.preferences.update(new_preferences)
                    return IntegrationResult(
                        status=IntegrationStatus.COMPLETED,
                        data=user.to_dict(),
                        message="User preferences updated successfully"
                    )
                else:
                    return IntegrationResult(
                        status=IntegrationStatus.FAILED,
                        error="User not found"
                    )
            
            elif message.operation == "list_users":
                users_data = [user.to_dict() for user in self._users.values()]
                return IntegrationResult(
                    status=IntegrationStatus.COMPLETED,
                    data={"users": users_data, "count": len(users_data)}
                )
            
            else:
                return IntegrationResult(
                    status=IntegrationStatus.FAILED,
                    error=f"Unsupported operation: {message.operation}"
                )
                
        except Exception as e:
            return IntegrationResult(
                status=IntegrationStatus.FAILED,
                error=str(e)
            )
    
    def get_supported_operations(self) -> List[str]:
        """Get supported operations."""
        return ["get_user_profile", "update_user_preferences", "list_users"]


class DataQualityService(DomainService):
    """Example data quality domain service."""
    
    def __init__(self):
        self.domain_name = "data_quality"
        self._reports: Dict[str, DataQualityReport] = {}
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample reports."""
        sample_reports = [
            DataQualityReport(
                report_id="report_1",
                dataset_id="dataset_users",
                quality_score=0.92,
                issues_found=[
                    {"type": "missing_values", "field": "phone", "count": 5},
                    {"type": "format_error", "field": "email", "count": 2}
                ],
                generated_at=datetime.now(timezone.utc) - timedelta(hours=2),
                generated_by="user_1"
            ),
            DataQualityReport(
                report_id="report_2", 
                dataset_id="dataset_transactions",
                quality_score=0.88,
                issues_found=[
                    {"type": "outliers", "field": "amount", "count": 12},
                    {"type": "duplicates", "field": "transaction_id", "count": 3}
                ],
                generated_at=datetime.now(timezone.utc) - timedelta(hours=1),
                generated_by="user_2"
            )
        ]
        
        for report in sample_reports:
            self._reports[report.report_id] = report
    
    async def handle_message(self, message: CrossDomainMessage) -> IntegrationResult[Any]:
        """Handle incoming cross-domain messages."""
        try:
            if message.operation == "get_quality_report":
                report_id = message.payload.get("report_id")
                report = self._reports.get(report_id)
                
                if report:
                    return IntegrationResult(
                        status=IntegrationStatus.COMPLETED,
                        data=report.to_dict()
                    )
                else:
                    return IntegrationResult(
                        status=IntegrationStatus.FAILED,
                        error="Report not found"
                    )
            
            elif message.operation == "generate_quality_report":
                dataset_id = message.payload.get("dataset_id")
                user_id = message.payload.get("user_id", "system")
                
                # Simulate report generation
                import random
                new_report = DataQualityReport(
                    report_id=f"report_{len(self._reports) + 1}",
                    dataset_id=dataset_id,
                    quality_score=round(random.uniform(0.7, 0.98), 2),
                    issues_found=[
                        {"type": "missing_values", "field": "field1", "count": random.randint(0, 10)},
                        {"type": "format_errors", "field": "field2", "count": random.randint(0, 5)}
                    ],
                    generated_at=datetime.now(timezone.utc),
                    generated_by=user_id
                )
                
                self._reports[new_report.report_id] = new_report
                
                return IntegrationResult(
                    status=IntegrationStatus.COMPLETED,
                    data=new_report.to_dict(),
                    message="Quality report generated successfully"
                )
            
            elif message.operation == "list_reports":
                reports_data = [report.to_dict() for report in self._reports.values()]
                return IntegrationResult(
                    status=IntegrationStatus.COMPLETED,
                    data={"reports": reports_data, "count": len(reports_data)}
                )
            
            else:
                return IntegrationResult(
                    status=IntegrationStatus.FAILED,
                    error=f"Unsupported operation: {message.operation}"
                )
                
        except Exception as e:
            return IntegrationResult(
                status=IntegrationStatus.FAILED,
                error=str(e)
            )
    
    def get_supported_operations(self) -> List[str]:
        """Get supported operations."""
        return ["get_quality_report", "generate_quality_report", "list_reports"]


class MachineLearningService(DomainService):
    """Example machine learning domain service."""
    
    def __init__(self):
        self.domain_name = "machine_learning"
        self._models: Dict[str, MLModelMetrics] = {}
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample model metrics."""
        sample_models = [
            MLModelMetrics(
                model_id="model_1",
                model_name="fraud_detection_v2",
                accuracy=0.94,
                precision=0.92,
                recall=0.89,
                f1_score=0.90,
                evaluated_at=datetime.now(timezone.utc) - timedelta(hours=6),
                dataset_size=10000
            ),
            MLModelMetrics(
                model_id="model_2",
                model_name="customer_churn_predictor",
                accuracy=0.87,
                precision=0.85,
                recall=0.88,
                f1_score=0.86,
                evaluated_at=datetime.now(timezone.utc) - timedelta(hours=3),
                dataset_size=15000
            )
        ]
        
        for model in sample_models:
            self._models[model.model_id] = model
    
    async def handle_message(self, message: CrossDomainMessage) -> IntegrationResult[Any]:
        """Handle incoming cross-domain messages."""
        try:
            if message.operation == "get_model_metrics":
                model_id = message.payload.get("model_id")
                model = self._models.get(model_id)
                
                if model:
                    return IntegrationResult(
                        status=IntegrationStatus.COMPLETED,
                        data=model.to_dict()
                    )
                else:
                    return IntegrationResult(
                        status=IntegrationStatus.FAILED,
                        error="Model not found"
                    )
            
            elif message.operation == "train_model":
                model_name = message.payload.get("model_name")
                dataset_id = message.payload.get("dataset_id")
                
                # Simulate model training
                import random
                new_model = MLModelMetrics(
                    model_id=f"model_{len(self._models) + 1}",
                    model_name=model_name,
                    accuracy=round(random.uniform(0.8, 0.95), 2),
                    precision=round(random.uniform(0.8, 0.95), 2),
                    recall=round(random.uniform(0.8, 0.95), 2),
                    f1_score=round(random.uniform(0.8, 0.95), 2),
                    evaluated_at=datetime.now(timezone.utc),
                    dataset_size=random.randint(5000, 20000)
                )
                
                self._models[new_model.model_id] = new_model
                
                return IntegrationResult(
                    status=IntegrationStatus.COMPLETED,
                    data=new_model.to_dict(),
                    message="Model trained successfully"
                )
            
            elif message.operation == "list_models":
                models_data = [model.to_dict() for model in self._models.values()]
                return IntegrationResult(
                    status=IntegrationStatus.COMPLETED,
                    data={"models": models_data, "count": len(models_data)}
                )
            
            else:
                return IntegrationResult(
                    status=IntegrationStatus.FAILED,
                    error=f"Unsupported operation: {message.operation}"
                )
                
        except Exception as e:
            return IntegrationResult(
                status=IntegrationStatus.FAILED,
                error=str(e)
            )
    
    def get_supported_operations(self) -> List[str]:
        """Get supported operations."""
        return ["get_model_metrics", "train_model", "list_models"]


# Example integration setup and usage
async def setup_example_integration():
    """Set up example integration between domains."""
    
    # Get integration manager
    manager = get_integration_manager()
    
    # Register domain services
    user_service = UserManagementService()
    data_quality_service = DataQualityService()
    ml_service = MachineLearningService()
    
    manager.register_service(user_service)
    manager.register_service(data_quality_service)
    manager.register_service(ml_service)
    
    # Set up domain adapters
    setup_domain_adapters()
    
    # Set up event subscriptions
    await setup_event_subscriptions(manager)
    
    return manager


def setup_domain_adapters():
    """Set up domain adapters for data transformation."""
    
    # Adapter from user management to data quality
    user_to_dq_mappings = {
        "generate_report": [
            FieldMapping("user_id", "generated_by"),
            FieldMapping("dataset", "dataset_id"),
        ]
    }
    
    register_standard_adapter(
        source_domain="user_management",
        target_domain="data_quality", 
        operation_mappings={"create_quality_report": "generate_quality_report"},
        field_mappings=user_to_dq_mappings
    )
    
    # Enrichment adapter from data quality to machine learning
    def enrich_training_request(payload: Dict[str, Any], context: IntegrationContext) -> Dict[str, Any]:
        """Enrich model training request with data quality info."""
        enriched = payload.copy()
        
        # Add data quality context
        enriched["data_validation"] = {
            "quality_check_required": True,
            "min_quality_score": 0.85,
            "requested_by": context.user_id
        }
        
        return enriched
    
    register_enrichment_adapter(
        source_domain="data_quality",
        target_domain="machine_learning",
        enrichment_functions={
            "train_model": enrich_training_request
        }
    )


async def setup_event_subscriptions(manager: CrossDomainIntegrationManager):
    """Set up cross-domain event subscriptions."""
    
    event_bus = manager.get_event_bus()
    
    # Subscribe to user events in data quality service
    async def handle_user_created(event: CrossDomainMessage):
        """Handle user created event."""
        user_data = event.payload
        print(f"Data Quality: New user created - {user_data.get('username')}")
        # Could trigger initial data quality checks for user's data
    
    # Subscribe to quality report events in ML service
    async def handle_quality_report_generated(event: CrossDomainMessage):
        """Handle quality report generated event."""
        report_data = event.payload
        quality_score = report_data.get('quality_score', 0)
        
        if quality_score < 0.8:
            print(f"ML Service: Low quality data detected (score: {quality_score})")
            # Could trigger data cleaning or model retraining
    
    event_bus.subscribe("user.*", handle_user_created)
    event_bus.subscribe("quality_report.*", handle_quality_report_generated)


# Example usage scenarios
async def example_simple_query():
    """Example of simple cross-domain query."""
    print("\n=== Simple Cross-Domain Query Example ===")
    
    manager = await setup_example_integration()
    
    # Query user profile from another domain
    context = IntegrationContext(
        source_domain="analytics",
        target_domain="user_management",
        user_id="system"
    )
    
    result = await manager.send_query(
        target_domain="user_management",
        operation="get_user_profile",
        payload={"user_id": "user_1"},
        context=context
    )
    
    if result.status == IntegrationStatus.COMPLETED:
        user_data = result.data
        print(f"Retrieved user: {user_data['username']} ({user_data['email']})")
    else:
        print(f"Query failed: {result.error}")


async def example_command_with_adaptation():
    """Example of cross-domain command with data adaptation."""
    print("\n=== Command with Data Adaptation Example ===")
    
    manager = await setup_example_integration()
    
    # Send command that will be adapted by registered adapter
    context = IntegrationContext(
        source_domain="user_management",
        target_domain="data_quality",
        user_id="user_1"
    )
    
    result = await manager.send_command(
        target_domain="data_quality",
        operation="create_quality_report",  # Will be mapped to generate_quality_report
        payload={
            "user_id": "user_1",  # Will be mapped to generated_by
            "dataset": "user_activity_logs"  # Will be mapped to dataset_id
        },
        context=context
    )
    
    if result.status == IntegrationStatus.COMPLETED:
        report_data = result.data
        print(f"Generated quality report: {report_data['report_id']} "
              f"(score: {report_data['quality_score']})")
    else:
        print(f"Command failed: {result.error}")


async def example_event_publishing():
    """Example of cross-domain event publishing."""
    print("\n=== Event Publishing Example ===")
    
    manager = await setup_example_integration()
    
    # Publish domain event
    context = IntegrationContext(
        source_domain="user_management",
        user_id="user_1"
    )
    
    await manager.publish_event(
        operation="user.profile.updated",
        payload={
            "user_id": "user_1",
            "username": "alice_smith",
            "changes": ["preferences"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        context=context
    )
    
    print("Published user profile updated event")
    
    # Small delay to allow event processing
    await asyncio.sleep(0.1)


async def example_saga_transaction():
    """Example of distributed saga transaction."""
    print("\n=== Saga Transaction Example ===")
    
    manager = await setup_example_integration()
    saga_orchestrator = SagaOrchestrator(manager)
    
    # Define saga steps for a complex workflow
    saga_steps = [
        SagaStep(
            step_id="validate_user",
            operation="get_user_profile",
            target_domain="user_management",
            payload={"user_id": "user_1"}
        ),
        SagaStep(
            step_id="generate_quality_report",
            operation="generate_quality_report",
            target_domain="data_quality",
            payload={
                "dataset_id": "user_1_data",
                "user_id": "user_1"
            },
            compensation_operation="delete_quality_report"
        ),
        SagaStep(
            step_id="train_model",
            operation="train_model",
            target_domain="machine_learning",
            payload={
                "model_name": "user_1_personalized_model",
                "dataset_id": "user_1_data"
            },
            compensation_operation="delete_model"
        )
    ]
    
    # Execute saga
    context = IntegrationContext(
        source_domain="orchestration",
        user_id="user_1"
    )
    
    result = await saga_orchestrator.execute_saga(
        saga_id="user_model_pipeline_1",
        steps=saga_steps,
        context=context
    )
    
    if result.status == IntegrationStatus.COMPLETED:
        print(f"Saga completed successfully: {result.data}")
    else:
        print(f"Saga failed: {result.error}")


async def run_all_examples():
    """Run all integration examples."""
    print("Running Cross-Domain Integration Examples")
    print("=" * 50)
    
    await example_simple_query()
    await example_command_with_adaptation()
    await example_event_publishing()
    await example_saga_transaction()
    
    print("\n=== Integration Examples Complete ===")


# Main execution
if __name__ == "__main__":
    asyncio.run(run_all_examples())