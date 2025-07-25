"""
Real-world integration examples demonstrating practical usage of the interaction framework.

This module provides complete examples for common business scenarios:
1. Financial fraud detection pipeline
2. Healthcare data processing workflow
3. E-commerce recommendation system
4. IoT sensor data processing
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import random

# Import from interfaces for stable contracts
from interfaces.dto import (
    BaseDTO, DetectionRequest, DetectionResult,
    DataQualityRequest, DataQualityResult,
    AnalyticsRequest, AnalyticsResult
)
from interfaces.events import (
    DomainEvent, AnomalyDetected, DataQualityCheckCompleted,
    SystemHealthChanged, EventPriority
)
from interfaces.patterns import Service, Repository

# Import from shared infrastructure
from shared import (
    get_event_bus, get_container, configure_container,
    publish_event, event_handler, DistributedEventBus
)

logger = logging.getLogger(__name__)


# =============================================================================
# Financial Fraud Detection Pipeline
# =============================================================================

class TransactionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"


@dataclass
class Transaction(BaseDTO):
    """Financial transaction DTO."""
    account_id: str
    amount: float
    merchant: str
    location: str
    timestamp: datetime
    status: TransactionStatus = TransactionStatus.PENDING
    risk_score: Optional[float] = None


@dataclass
class FraudAlert(DomainEvent):
    """Event fired when fraud is detected."""
    transaction_id: str
    risk_score: float
    fraud_indicators: List[str]
    severity: str


class TransactionService(Service):
    """Service for processing financial transactions."""
    
    def __init__(self):
        self.processed_transactions = []
        self.blocked_accounts = set()
    
    async def execute(self, transaction: Transaction) -> Transaction:
        """Process a financial transaction."""
        logger.info(f"Processing transaction {transaction.id} for ${transaction.amount}")
        
        # Simulate transaction processing
        await asyncio.sleep(0.05)
        
        # Basic fraud indicators
        fraud_indicators = []
        risk_score = 0.0
        
        # High amount transactions
        if transaction.amount > 10000:
            fraud_indicators.append("high_amount")
            risk_score += 0.3
        
        # Unusual time (late night)
        if transaction.timestamp.hour < 6 or transaction.timestamp.hour > 22:
            fraud_indicators.append("unusual_time")
            risk_score += 0.2
        
        # Known high-risk locations
        if "offshore" in transaction.location.lower():
            fraud_indicators.append("high_risk_location")
            risk_score += 0.4
        
        # Blocked account
        if transaction.account_id in self.blocked_accounts:
            fraud_indicators.append("blocked_account")
            risk_score = 1.0
        
        transaction.risk_score = min(risk_score, 1.0)
        
        # Determine status
        if risk_score >= 0.8:
            transaction.status = TransactionStatus.REJECTED
            severity = "high"
        elif risk_score >= 0.5:
            transaction.status = TransactionStatus.FLAGGED
            severity = "medium"
        else:
            transaction.status = TransactionStatus.APPROVED
            severity = "low"
        
        # Publish fraud alert if risky
        if risk_score >= 0.5:
            fraud_alert = FraudAlert(
                event_id="",
                event_type="",
                aggregate_id=transaction.id,
                occurred_at=datetime.utcnow(),
                transaction_id=transaction.id,
                risk_score=risk_score,
                fraud_indicators=fraud_indicators,
                severity=severity,
                priority=EventPriority.HIGH if risk_score >= 0.8 else EventPriority.NORMAL
            )
            await publish_event(fraud_alert)
        
        self.processed_transactions.append(transaction)
        return transaction
    
    async def validate_request(self, transaction: Transaction) -> bool:
        return bool(transaction.account_id and transaction.amount > 0)
    
    def get_service_info(self) -> Dict[str, Any]:
        return {
            "name": "TransactionService",
            "processed_count": len(self.processed_transactions),
            "blocked_accounts": len(self.blocked_accounts)
        }
    
    def block_account(self, account_id: str):
        """Block an account due to fraud."""
        self.blocked_accounts.add(account_id)
        logger.warning(f"Account {account_id} has been blocked")


class FraudMonitoringService:
    """Service for monitoring and responding to fraud alerts."""
    
    def __init__(self, transaction_service: TransactionService):
        self.transaction_service = transaction_service
        self.fraud_alerts = []
        self.auto_block_threshold = 0.9
        
        # Subscribe to fraud events
        event_bus = get_event_bus()
        event_bus.subscribe(FraudAlert, self.handle_fraud_alert)
    
    @event_handler(FraudAlert)
    async def handle_fraud_alert(self, alert: FraudAlert):
        """Handle fraud alert events."""
        self.fraud_alerts.append(alert)
        logger.warning(f"Fraud alert: Transaction {alert.transaction_id} - Risk: {alert.risk_score}")
        
        # Auto-block high-risk accounts
        if alert.risk_score >= self.auto_block_threshold:
            # Extract account from transaction (simulated)
            account_id = f"acc_{alert.transaction_id[:8]}"
            self.transaction_service.block_account(account_id)
            
            # Notify security team
            await self._notify_security_team(alert)
    
    async def _notify_security_team(self, alert: FraudAlert):
        """Notify security team about high-risk transactions."""
        logger.critical(f"SECURITY ALERT: High-risk transaction {alert.transaction_id}")
        # In real implementation, would send email/SMS/Slack notification


# =============================================================================
# Healthcare Data Processing Workflow
# =============================================================================

@dataclass
class PatientData(BaseDTO):
    """Patient medical data DTO."""
    patient_id: str
    medical_record_number: str
    vital_signs: Dict[str, float]
    lab_results: Dict[str, float]
    medications: List[str]
    diagnoses: List[str]
    
    def is_phi_compliant(self) -> bool:
        """Check if data is PHI compliant (simplified)."""
        # No direct identifiers allowed
        return not any(field in str(self.__dict__) for field in ["ssn", "phone", "email"])


class HealthDataQualityService(Service):
    """Service for healthcare data quality assessment."""
    
    async def execute(self, request: DataQualityRequest) -> DataQualityResult:
        """Assess healthcare data quality."""
        logger.info(f"Assessing healthcare data quality for {request.dataset_id}")
        
        # Healthcare-specific quality checks
        quality_score = 0.0
        issues = []
        
        # PHI compliance check
        if "phi_compliant" in request.quality_rules:
            quality_score += 0.3
        
        # Required fields check
        if "required_fields" in request.quality_rules:
            quality_score += 0.25
        
        # Value ranges check (e.g., vital signs)
        if "value_ranges" in request.quality_rules:
            quality_score += 0.25
        
        # Consistency check
        if "consistency" in request.quality_rules:
            quality_score += 0.2
        
        # Simulate some issues in healthcare data
        if random.random() < 0.3:
            issues.append("Missing vital signs data")
            quality_score -= 0.1
        
        if random.random() < 0.2:
            issues.append("Inconsistent medication dosage")
            quality_score -= 0.15
        
        status = "passed" if quality_score >= 0.8 and len(issues) == 0 else "failed"
        
        return DataQualityResult(
            id=f"hdq_{request.id}",
            created_at=datetime.utcnow(),
            request_id=request.id,
            dataset_id=request.dataset_id,
            status=status,
            overall_score=max(0, quality_score),
            rule_results={
                rule: {"score": quality_score, "passed": quality_score >= 0.8}
                for rule in request.quality_rules
            },
            issues_found=issues,
            recommendations=["Implement stricter data validation", "Add PHI compliance checks"],
            execution_time_ms=150
        )
    
    async def validate_request(self, request: DataQualityRequest) -> bool:
        return bool(request.dataset_id and "healthcare" in request.dataset_id.lower())
    
    def get_service_info(self) -> Dict[str, Any]:
        return {"name": "HealthDataQualityService", "domain": "healthcare"}


class ClinicalAnomalyDetectionService(Service):
    """Service for detecting clinical anomalies."""
    
    async def execute(self, request: DetectionRequest) -> DetectionResult:
        """Detect clinical anomalies in patient data."""
        logger.info(f"Detecting clinical anomalies for {request.dataset_id}")
        
        # Simulate clinical anomaly detection
        await asyncio.sleep(0.1)
        
        # Healthcare-specific anomaly patterns
        anomaly_patterns = []
        confidence_scores = []
        
        # Critical vital signs
        if "vital_signs" in request.parameters:
            anomaly_patterns.extend([
                "heart_rate_critical", "blood_pressure_critical", 
                "temperature_fever"
            ])
            confidence_scores.extend([0.95, 0.88, 0.82])
        
        # Drug interactions
        if "medications" in request.parameters:
            anomaly_patterns.extend(["drug_interaction_warning"])
            confidence_scores.extend([0.91])
        
        # Lab value anomalies
        if "lab_results" in request.parameters:
            anomaly_patterns.extend(["glucose_critical", "kidney_function_low"])
            confidence_scores.extend([0.89, 0.75])
        
        anomaly_count = len(anomaly_patterns)
        
        result = DetectionResult(
            id=f"cad_{request.id}",
            created_at=datetime.utcnow(),
            request_id=request.id,
            status="completed",
            anomalies_count=anomaly_count,
            anomaly_scores=confidence_scores[:anomaly_count],
            anomaly_indices=list(range(anomaly_count)),
            confidence_scores=confidence_scores[:anomaly_count],
            execution_time_ms=100,
            algorithm_used=request.algorithm,
            metadata={"anomaly_patterns": anomaly_patterns}
        )
        
        # Publish critical health alerts
        if anomaly_count > 0 and max(confidence_scores[:anomaly_count]) > 0.9:
            health_alert = SystemHealthChanged(
                event_id="",
                event_type="",
                aggregate_id=request.dataset_id,
                occurred_at=datetime.utcnow(),
                component="clinical_monitoring",
                status="critical",
                previous_status="normal",
                metrics={"anomaly_count": anomaly_count, "max_confidence": max(confidence_scores[:anomaly_count])},
                priority=EventPriority.CRITICAL
            )
            await publish_event(health_alert)
        
        return result
    
    async def validate_request(self, request: DetectionRequest) -> bool:
        return bool(request.dataset_id and request.algorithm in ["clinical_rules", "vital_signs_monitor"])
    
    def get_service_info(self) -> Dict[str, Any]:
        return {"name": "ClinicalAnomalyDetectionService", "domain": "healthcare"}


# =============================================================================
# E-commerce Recommendation System
# =============================================================================

@dataclass
class UserBehavior(BaseDTO):
    """User behavior data for recommendations."""
    user_id: str
    session_id: str
    page_views: List[str]
    purchases: List[str]
    search_queries: List[str]
    time_spent_minutes: float
    device_type: str


@dataclass
class RecommendationRequest(BaseDTO):
    """Request for product recommendations."""
    user_id: str
    session_id: str
    current_product_id: Optional[str] = None
    recommendation_type: str = "collaborative"  # collaborative, content_based, hybrid
    max_recommendations: int = 10


@dataclass
class RecommendationResult(BaseDTO):
    """Product recommendation results."""
    user_id: str
    recommended_products: List[Dict[str, Any]]
    confidence_scores: List[float]
    recommendation_type: str
    explanation: str


class RecommendationService(Service):
    """E-commerce recommendation service."""
    
    def __init__(self):
        self.user_interactions = {}
        self.product_catalog = self._load_mock_catalog()
    
    def _load_mock_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Load mock product catalog."""
        return {
            "prod_001": {"name": "Wireless Headphones", "category": "electronics", "price": 99.99},
            "prod_002": {"name": "Running Shoes", "category": "sports", "price": 129.99},
            "prod_003": {"name": "Coffee Maker", "category": "home", "price": 79.99},
            "prod_004": {"name": "Smartphone", "category": "electronics", "price": 699.99},
            "prod_005": {"name": "Yoga Mat", "category": "sports", "price": 29.99},
        }
    
    async def execute(self, request: RecommendationRequest) -> RecommendationResult:
        """Generate product recommendations."""
        logger.info(f"Generating recommendations for user {request.user_id}")
        
        # Simulate recommendation algorithm
        await asyncio.sleep(0.05)
        
        recommendations = []
        confidence_scores = []
        
        if request.recommendation_type == "collaborative":
            # Collaborative filtering simulation
            recommendations = [
                {"product_id": "prod_001", "name": "Wireless Headphones", "reason": "Users like you also bought"},
                {"product_id": "prod_004", "name": "Smartphone", "reason": "Frequently bought together"},
                {"product_id": "prod_003", "name": "Coffee Maker", "reason": "Popular in your area"}
            ]
            confidence_scores = [0.85, 0.78, 0.72]
            explanation = "Based on similar users' preferences"
            
        elif request.recommendation_type == "content_based":
            # Content-based filtering simulation
            recommendations = [
                {"product_id": "prod_002", "name": "Running Shoes", "reason": "Similar to your recent views"},
                {"product_id": "prod_005", "name": "Yoga Mat", "reason": "Matches your sports interests"}
            ]
            confidence_scores = [0.82, 0.69]
            explanation = "Based on your browsing history and preferences"
            
        else:  # hybrid
            # Hybrid approach
            recommendations = [
                {"product_id": "prod_001", "name": "Wireless Headphones", "reason": "Popular and matches your profile"},
                {"product_id": "prod_002", "name": "Running Shoes", "reason": "Trending in your category"},
                {"product_id": "prod_004", "name": "Smartphone", "reason": "Upgrade recommendation"}
            ]
            confidence_scores = [0.88, 0.81, 0.75]
            explanation = "Hybrid algorithm combining multiple factors"
        
        # Limit to requested count
        recommendations = recommendations[:request.max_recommendations]
        confidence_scores = confidence_scores[:len(recommendations)]
        
        return RecommendationResult(
            id=f"rec_{request.id}",
            created_at=datetime.utcnow(),
            user_id=request.user_id,
            recommended_products=recommendations,
            confidence_scores=confidence_scores,
            recommendation_type=request.recommendation_type,
            explanation=explanation
        )
    
    async def validate_request(self, request: RecommendationRequest) -> bool:
        return bool(request.user_id and request.recommendation_type in ["collaborative", "content_based", "hybrid"])
    
    def get_service_info(self) -> Dict[str, Any]:
        return {
            "name": "RecommendationService",
            "domain": "ecommerce",
            "catalog_size": len(self.product_catalog)
        }


# =============================================================================
# IoT Sensor Data Processing
# =============================================================================

@dataclass
class SensorReading(BaseDTO):
    """IoT sensor reading data."""
    sensor_id: str
    sensor_type: str  # temperature, humidity, pressure, motion
    value: float
    unit: str
    location: str
    battery_level: float
    signal_strength: int


@dataclass
class SensorAlert(DomainEvent):
    """Alert for sensor anomalies."""
    sensor_id: str
    sensor_type: str
    current_value: float
    threshold_exceeded: str
    severity: str


class IoTDataProcessingService(Service):
    """Service for processing IoT sensor data."""
    
    def __init__(self):
        self.sensor_thresholds = {
            "temperature": {"min": -10, "max": 45, "critical": 50},
            "humidity": {"min": 0, "max": 100, "critical": 95},
            "pressure": {"min": 950, "max": 1050, "critical": 1060},
            "motion": {"min": 0, "max": 1, "critical": 1}
        }
        self.processed_readings = []
    
    async def execute(self, reading: SensorReading) -> SensorReading:
        """Process IoT sensor reading."""
        logger.info(f"Processing reading from sensor {reading.sensor_id}")
        
        # Check for anomalies
        if reading.sensor_type in self.sensor_thresholds:
            thresholds = self.sensor_thresholds[reading.sensor_type]
            
            # Check for threshold violations
            if reading.value > thresholds["critical"]:
                severity = "critical"
                threshold_type = f"critical_high_{thresholds['critical']}"
            elif reading.value < thresholds["min"]:
                severity = "warning"
                threshold_type = f"below_minimum_{thresholds['min']}"
            elif reading.value > thresholds["max"]:
                severity = "warning"
                threshold_type = f"above_maximum_{thresholds['max']}"
            else:
                severity = "normal"
                threshold_type = None
            
            # Publish alert if needed
            if threshold_type:
                alert = SensorAlert(
                    event_id="",
                    event_type="",
                    aggregate_id=reading.sensor_id,
                    occurred_at=datetime.utcnow(),
                    sensor_id=reading.sensor_id,
                    sensor_type=reading.sensor_type,
                    current_value=reading.value,
                    threshold_exceeded=threshold_type,
                    severity=severity,
                    priority=EventPriority.CRITICAL if severity == "critical" else EventPriority.NORMAL
                )
                await publish_event(alert)
        
        # Check battery level
        if reading.battery_level < 0.2:
            battery_alert = SystemHealthChanged(
                event_id="",
                event_type="",
                aggregate_id=reading.sensor_id,
                occurred_at=datetime.utcnow(),
                component=f"sensor_{reading.sensor_id}",
                status="low_battery",
                previous_status="normal",
                metrics={"battery_level": reading.battery_level}
            )
            await publish_event(battery_alert)
        
        self.processed_readings.append(reading)
        return reading
    
    async def validate_request(self, reading: SensorReading) -> bool:
        return bool(reading.sensor_id and reading.sensor_type and reading.value is not None)
    
    def get_service_info(self) -> Dict[str, Any]:
        return {
            "name": "IoTDataProcessingService",
            "domain": "iot",
            "processed_count": len(self.processed_readings)
        }


# =============================================================================
# Integration Examples Runner
# =============================================================================

class RealWorldIntegrationDemo:
    """Demonstrates real-world integration scenarios."""
    
    def __init__(self):
        self.scenarios = {
            "fraud_detection": self._run_fraud_detection_demo,
            "healthcare": self._run_healthcare_demo,
            "ecommerce": self._run_ecommerce_demo,
            "iot": self._run_iot_demo
        }
    
    async def run_all_scenarios(self):
        """Run all real-world integration scenarios."""
        print("🚀 Running Real-World Integration Scenarios")
        print("=" * 60)
        
        for name, scenario in self.scenarios.items():
            print(f"\n📊 Running {name.replace('_', ' ').title()} Scenario")
            print("-" * 40)
            
            try:
                await scenario()
                print(f"✅ {name} scenario completed successfully")
            except Exception as e:
                print(f"❌ {name} scenario failed: {e}")
        
        print(f"\n🎉 All scenarios completed!")
    
    async def _run_fraud_detection_demo(self):
        """Demo: Financial fraud detection pipeline."""
        # Setup services
        transaction_service = TransactionService()
        fraud_monitor = FraudMonitoringService(transaction_service)
        
        # Simulate transactions
        transactions = [
            Transaction(
                id="txn_001",
                created_at=datetime.utcnow(),
                account_id="acc_12345",
                amount=150.00,
                merchant="Coffee Shop",
                location="Downtown",
                timestamp=datetime.utcnow()
            ),
            Transaction(
                id="txn_002", 
                created_at=datetime.utcnow(),
                account_id="acc_67890",
                amount=15000.00,  # High amount - will trigger alert
                merchant="Electronics Store",
                location="Offshore Mall",  # High risk location
                timestamp=datetime.utcnow().replace(hour=2)  # Late night
            ),
            Transaction(
                id="txn_003",
                created_at=datetime.utcnow(),
                account_id="acc_11111",
                amount=50.00,
                merchant="Grocery Store",
                location="Local",
                timestamp=datetime.utcnow()
            )
        ]
        
        # Process transactions
        for txn in transactions:
            result = await transaction_service.execute(txn)
            print(f"  Transaction {txn.id}: {result.status.value} (Risk: {result.risk_score:.2f})")
        
        # Wait for event processing
        await asyncio.sleep(0.1)
        
        print(f"  📈 Fraud alerts generated: {len(fraud_monitor.fraud_alerts)}")
        print(f"  🚫 Accounts blocked: {len(transaction_service.blocked_accounts)}")
    
    async def _run_healthcare_demo(self):
        """Demo: Healthcare data processing workflow."""
        # Setup services
        health_quality_service = HealthDataQualityService()
        clinical_detection_service = ClinicalAnomalyDetectionService()
        
        # Assess data quality
        quality_request = DataQualityRequest(
            id="hq_001",
            created_at=datetime.utcnow(),
            dataset_id="healthcare_patient_data_2024",
            quality_rules=["phi_compliant", "required_fields", "value_ranges", "consistency"]
        )
        
        quality_result = await health_quality_service.execute(quality_request)
        print(f"  📊 Data quality score: {quality_result.overall_score:.2f}")
        print(f"  ⚠️  Issues found: {len(quality_result.issues_found)}")
        
        # Run clinical anomaly detection if quality is good
        if quality_result.overall_score >= 0.7:
            detection_request = DetectionRequest(
                id="cad_001",
                created_at=datetime.utcnow(),
                dataset_id="healthcare_patient_data_2024",
                algorithm="clinical_rules",
                parameters={"vital_signs": True, "medications": True, "lab_results": True}
            )
            
            detection_result = await clinical_detection_service.execute(detection_request)
            print(f"  🔍 Clinical anomalies detected: {detection_result.anomalies_count}")
            
            if detection_result.metadata and "anomaly_patterns" in detection_result.metadata:
                patterns = detection_result.metadata["anomaly_patterns"]
                print(f"  🚨 Alert patterns: {', '.join(patterns[:3])}")
        
        await asyncio.sleep(0.1)  # Allow event processing
    
    async def _run_ecommerce_demo(self):
        """Demo: E-commerce recommendation system."""
        recommendation_service = RecommendationService()
        
        # Generate different types of recommendations
        recommendation_types = ["collaborative", "content_based", "hybrid"]
        
        for rec_type in recommendation_types:
            request = RecommendationRequest(
                id=f"rec_{rec_type}",
                created_at=datetime.utcnow(),
                user_id="user_12345",
                session_id="sess_67890",
                recommendation_type=rec_type,
                max_recommendations=3
            )
            
            result = await recommendation_service.execute(request)
            print(f"  🛒 {rec_type.title()} recommendations: {len(result.recommended_products)}")
            print(f"      Top recommendation: {result.recommended_products[0]['name']} (confidence: {result.confidence_scores[0]:.2f})")
    
    async def _run_iot_demo(self):
        """Demo: IoT sensor data processing."""
        iot_service = IoTDataProcessingService()
        
        # Simulate various sensor readings
        sensor_readings = [
            SensorReading(
                id="reading_001",
                created_at=datetime.utcnow(),
                sensor_id="temp_001",
                sensor_type="temperature",
                value=22.5,  # Normal
                unit="celsius",
                location="Office Room 1",
                battery_level=0.85,
                signal_strength=95
            ),
            SensorReading(
                id="reading_002",
                created_at=datetime.utcnow(),
                sensor_id="temp_002",
                sensor_type="temperature",
                value=55.0,  # Critical high
                unit="celsius",
                location="Server Room",
                battery_level=0.15,  # Low battery
                signal_strength=78
            ),
            SensorReading(
                id="reading_003",
                created_at=datetime.utcnow(),
                sensor_id="humid_001",
                sensor_type="humidity",
                value=98.0,  # High humidity
                unit="percent",
                location="Warehouse",
                battery_level=0.65,
                signal_strength=82
            )
        ]
        
        # Process readings
        alerts_generated = 0
        for reading in sensor_readings:
            await iot_service.execute(reading)
            print(f"  📡 Processed {reading.sensor_type} reading: {reading.value}{reading.unit}")
            
            # Count potential alerts
            if reading.sensor_type == "temperature" and reading.value > 50:
                alerts_generated += 1
            if reading.battery_level < 0.2:
                alerts_generated += 1
        
        await asyncio.sleep(0.1)  # Allow event processing
        print(f"  🚨 Alerts generated: {alerts_generated}")


async def main():
    """Main function to run real-world integration examples."""
    # Configure services (simplified for demo)
    def configure_demo_services(container):
        # Financial services
        container.register_singleton(TransactionService)
        
        # Healthcare services
        container.register_singleton(HealthDataQualityService)
        container.register_singleton(ClinicalAnomalyDetectionService)
        
        # E-commerce services
        container.register_singleton(RecommendationService)
        
        # IoT services
        container.register_singleton(IoTDataProcessingService)
    
    configure_container(configure_demo_services)
    
    # Start event bus
    event_bus = get_event_bus()
    await event_bus.start()
    
    try:
        # Run integration demos
        demo = RealWorldIntegrationDemo()
        await demo.run_all_scenarios()
        
        # Show event bus metrics
        print(f"\n📊 Event Bus Metrics:")
        metrics = event_bus.get_metrics()
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
    finally:
        await event_bus.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
    asyncio.run(main())