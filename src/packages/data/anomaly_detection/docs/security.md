# Security and Privacy Implementation Guide

This guide covers comprehensive security and privacy considerations for the Anomaly Detection package, including data protection, access control, secure deployment, and compliance requirements.

## Table of Contents

1. [Overview](#overview)
2. [Data Security](#data-security)
3. [Authentication and Authorization](#authentication-and-authorization)
4. [Encryption](#encryption)
5. [Privacy Protection](#privacy-protection)
6. [Secure API Design](#secure-api-design)
7. [Infrastructure Security](#infrastructure-security)
8. [Compliance and Governance](#compliance-and-governance)
9. [Audit and Monitoring](#audit-and-monitoring)
10. [Incident Response](#incident-response)
11. [Security Testing](#security-testing)
12. [Best Practices](#best-practices)

## Overview

Security and privacy are critical considerations for anomaly detection systems, especially when processing sensitive data or operating in regulated environments. This guide provides comprehensive security measures and implementation patterns.

### Security Principles

- **Defense in Depth**: Multiple layers of security controls
- **Least Privilege**: Minimal necessary access rights
- **Zero Trust**: Never trust, always verify
- **Privacy by Design**: Privacy considerations from the start
- **Data Minimization**: Collect and process only necessary data
- **Transparency**: Clear data handling practices

### Threat Model

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class ThreatCategory(Enum):
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MODEL_POISONING = "model_poisoning"
    ADVERSARIAL_ATTACKS = "adversarial_attacks"
    INFERENCE_ATTACKS = "inference_attacks"
    DENIAL_OF_SERVICE = "denial_of_service"
    INSIDER_THREATS = "insider_threats"
    SUPPLY_CHAIN = "supply_chain"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityThreat:
    """Security threat definition."""
    id: str
    name: str
    category: ThreatCategory
    risk_level: RiskLevel
    description: str
    potential_impact: List[str]
    attack_vectors: List[str]
    mitigations: List[str]
    detection_methods: List[str]

# Common threats in anomaly detection systems
SECURITY_THREATS = [
    SecurityThreat(
        id="T001",
        name="Training Data Poisoning",
        category=ThreatCategory.MODEL_POISONING,
        risk_level=RiskLevel.HIGH,
        description="Malicious injection of crafted data to corrupt model training",
        potential_impact=[
            "Model performance degradation",
            "False negative/positive rates increase",
            "System reliability compromise"
        ],
        attack_vectors=[
            "Compromised data sources",
            "Insider access to training pipeline",
            "External data feed manipulation"
        ],
        mitigations=[
            "Data validation and sanitization",
            "Statistical outlier detection in training data",
            "Multiple data source validation",
            "Continuous model performance monitoring"
        ],
        detection_methods=[
            "Training data anomaly detection",
            "Model performance drift monitoring",
            "Data provenance tracking"
        ]
    ),
    
    SecurityThreat(
        id="T002",
        name="Model Inversion Attack",
        category=ThreatCategory.INFERENCE_ATTACKS,
        risk_level=RiskLevel.MEDIUM,
        description="Attempt to reconstruct training data from model predictions",
        potential_impact=[
            "Training data privacy breach",
            "Sensitive information exposure",
            "Regulatory compliance violations"
        ],
        attack_vectors=[
            "API query patterns",
            "Model output analysis",
            "Prediction confidence exploitation"
        ],
        mitigations=[
            "Differential privacy implementation",
            "Query rate limiting",
            "Output perturbation",
            "Minimal information disclosure"
        ],
        detection_methods=[
            "Unusual query pattern monitoring",
            "API access pattern analysis",
            "Response time analysis"
        ]
    ),
    
    SecurityThreat(
        id="T003",
        name="Adversarial Input Attacks",
        category=ThreatCategory.ADVERSARIAL_ATTACKS,
        risk_level=RiskLevel.MEDIUM,
        description="Crafted inputs designed to evade anomaly detection",
        potential_impact=[
            "False negative results",
            "Security monitoring bypass",
            "System integrity compromise"
        ],
        attack_vectors=[
            "Input feature manipulation",
            "Gradient-based attacks",
            "Evolutionary attack strategies"
        ],
        mitigations=[
            "Input validation and sanitization",
            "Adversarial training",
            "Ensemble methods",
            "Input transformation defenses"
        ],
        detection_methods=[
            "Input anomaly detection",
            "Prediction confidence analysis",
            "Multiple model consensus"
        ]
    )
]

@dataclass
class SecurityControl:
    """Security control implementation."""
    id: str
    name: str
    category: str
    description: str
    implementation_level: str  # "required", "recommended", "optional"
    applicable_threats: List[str]
    implementation_guidance: str
    verification_method: str
```

## Data Security

### Data Classification and Handling

```python
# security/data_classification.py
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import hashlib
import logging
from datetime import datetime, timedelta

class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class DataCategory(Enum):
    PERSONAL_DATA = "personal_data"
    FINANCIAL_DATA = "financial_data"
    HEALTH_DATA = "health_data"
    TECHNICAL_DATA = "technical_data"
    BUSINESS_DATA = "business_data"

@dataclass
class DataHandlingPolicy:
    classification: DataClassification
    retention_period_days: int
    encryption_required: bool
    access_logging_required: bool
    geographic_restrictions: List[str]
    anonymization_required: bool
    pseudonymization_allowed: bool
    deletion_method: str  # "secure_wipe", "cryptographic_erasure", "standard"

class SecureDataHandler:
    """Secure data handling with classification-based policies."""
    
    def __init__(self):
        self.policies = self._initialize_policies()
        self.access_logger = logging.getLogger("data_access")
        self.data_registry = {}  # Track data lineage
        
    def _initialize_policies(self) -> Dict[DataClassification, DataHandlingPolicy]:
        """Initialize data handling policies."""
        return {
            DataClassification.PUBLIC: DataHandlingPolicy(
                classification=DataClassification.PUBLIC,
                retention_period_days=365,
                encryption_required=False,
                access_logging_required=False,
                geographic_restrictions=[],
                anonymization_required=False,
                pseudonymization_allowed=True,
                deletion_method="standard"
            ),
            DataClassification.INTERNAL: DataHandlingPolicy(
                classification=DataClassification.INTERNAL,
                retention_period_days=1095,  # 3 years
                encryption_required=True,
                access_logging_required=True,
                geographic_restrictions=[],
                anonymization_required=False,
                pseudonymization_allowed=True,
                deletion_method="secure_wipe"
            ),
            DataClassification.CONFIDENTIAL: DataHandlingPolicy(
                classification=DataClassification.CONFIDENTIAL,
                retention_period_days=730,  # 2 years
                encryption_required=True,
                access_logging_required=True,
                geographic_restrictions=["EU", "US"],
                anonymization_required=True,
                pseudonymization_allowed=True,
                deletion_method="cryptographic_erasure"
            ),
            DataClassification.RESTRICTED: DataHandlingPolicy(
                classification=DataClassification.RESTRICTED,
                retention_period_days=365,  # 1 year
                encryption_required=True,
                access_logging_required=True,
                geographic_restrictions=["local_only"],
                anonymization_required=True,
                pseudonymization_allowed=False,
                deletion_method="cryptographic_erasure"
            )
        }
    
    def classify_data(self, data_sample: Dict[str, Any], 
                     data_categories: List[DataCategory]) -> DataClassification:
        """Automatically classify data based on content and categories."""
        
        # Check for restricted data categories
        if DataCategory.HEALTH_DATA in data_categories:
            return DataClassification.RESTRICTED
        
        if DataCategory.FINANCIAL_DATA in data_categories:
            return DataClassification.CONFIDENTIAL
        
        if DataCategory.PERSONAL_DATA in data_categories:
            return DataClassification.CONFIDENTIAL
        
        # Check data content for sensitive patterns
        data_str = str(data_sample).lower()
        
        # Look for PII patterns
        sensitive_patterns = [
            "ssn", "social security", "passport", "credit card",
            "account number", "routing number", "medical record",
            "patient id", "diagnosis", "prescription"
        ]
        
        if any(pattern in data_str for pattern in sensitive_patterns):
            return DataClassification.RESTRICTED
        
        # Check for business-sensitive terms
        business_patterns = [
            "proprietary", "confidential", "trade secret",
            "financial results", "strategy", "merger"
        ]
        
        if any(pattern in data_str for pattern in business_patterns):
            return DataClassification.CONFIDENTIAL
        
        # Default to internal
        return DataClassification.INTERNAL
    
    def register_data(self, data_id: str, data_sample: Dict[str, Any],
                     categories: List[DataCategory], user_id: str,
                     purpose: str) -> Dict[str, Any]:
        """Register data with classification and tracking."""
        
        classification = self.classify_data(data_sample, categories)
        policy = self.policies[classification]
        
        # Create data registry entry
        registry_entry = {
            'data_id': data_id,
            'classification': classification.value,
            'categories': [cat.value for cat in categories],
            'registered_by': user_id,
            'registered_at': datetime.now(),
            'purpose': purpose,
            'retention_until': datetime.now() + timedelta(days=policy.retention_period_days),
            'access_count': 0,
            'last_accessed': None,
            'encryption_key_id': None,
            'anonymized': False,
            'pseudonymized': False
        }
        
        # Apply encryption if required
        if policy.encryption_required:
            encrypted_data, key_id = self._encrypt_data(data_sample)
            registry_entry['encryption_key_id'] = key_id
            data_sample = encrypted_data
        
        # Apply anonymization if required
        if policy.anonymization_required:
            data_sample = self._anonymize_data(data_sample, categories)
            registry_entry['anonymized'] = True
        
        self.data_registry[data_id] = registry_entry
        
        # Log data registration
        self.access_logger.info(f"Data registered: {data_id}, "
                               f"classification: {classification.value}, "
                               f"user: {user_id}, purpose: {purpose}")
        
        return {
            'data_id': data_id,
            'classification': classification.value,
            'policy': policy,
            'processed_data': data_sample
        }
    
    def access_data(self, data_id: str, user_id: str, purpose: str) -> Optional[Dict[str, Any]]:
        """Secure data access with logging and policy enforcement."""
        
        if data_id not in self.data_registry:
            self.access_logger.warning(f"Access denied: Data not found - {data_id}")
            return None
        
        registry_entry = self.data_registry[data_id]
        classification = DataClassification(registry_entry['classification'])
        policy = self.policies[classification]
        
        # Check retention period
        if datetime.now() > registry_entry['retention_until']:
            self.access_logger.warning(f"Access denied: Data expired - {data_id}")
            return None
        
        # Update access tracking
        registry_entry['access_count'] += 1
        registry_entry['last_accessed'] = datetime.now()
        
        # Log access
        if policy.access_logging_required:
            self.access_logger.info(f"Data accessed: {data_id}, "
                                   f"user: {user_id}, purpose: {purpose}, "
                                   f"access_count: {registry_entry['access_count']}")
        
        return registry_entry
    
    def _encrypt_data(self, data: Dict[str, Any]) -> tuple:
        """Encrypt sensitive data (simplified implementation)."""
        # In production, use proper encryption with key management
        import json
        from cryptography.fernet import Fernet
        
        key = Fernet.generate_key()
        fernet = Fernet(key)
        
        data_str = json.dumps(data, default=str)
        encrypted_data = fernet.encrypt(data_str.encode())
        
        # Generate key ID for key management
        key_id = hashlib.sha256(key).hexdigest()[:16]
        
        # Store key securely (simplified - use proper key management in production)
        return {'encrypted_data': encrypted_data.decode()}, key_id
    
    def _anonymize_data(self, data: Dict[str, Any], 
                       categories: List[DataCategory]) -> Dict[str, Any]:
        """Apply anonymization techniques."""
        anonymized_data = data.copy()
        
        # Remove direct identifiers
        identifier_fields = ['id', 'user_id', 'customer_id', 'patient_id', 
                           'name', 'email', 'phone', 'address', 'ssn']
        
        for field in identifier_fields:
            if field in anonymized_data:
                # Replace with hash or remove
                if field.endswith('_id'):
                    anonymized_data[field] = hashlib.sha256(
                        str(anonymized_data[field]).encode()
                    ).hexdigest()[:16]
                else:
                    del anonymized_data[field]
        
        # Apply k-anonymity or differential privacy for specific categories
        if DataCategory.HEALTH_DATA in categories:
            anonymized_data = self._apply_k_anonymity(anonymized_data, k=5)
        
        if DataCategory.FINANCIAL_DATA in categories:
            anonymized_data = self._apply_differential_privacy(anonymized_data)
        
        return anonymized_data
    
    def _apply_k_anonymity(self, data: Dict[str, Any], k: int = 5) -> Dict[str, Any]:
        """Apply k-anonymity (simplified implementation)."""
        # In production, implement proper k-anonymity algorithms
        # This is a simplified version for demonstration
        
        quasi_identifiers = ['age', 'zip_code', 'birth_date']
        
        for field in quasi_identifiers:
            if field in data:
                if field == 'age' and isinstance(data[field], (int, float)):
                    # Generalize age to age ranges
                    age = int(data[field])
                    age_range = f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
                    data[field] = age_range
                elif field == 'zip_code':
                    # Generalize zip code
                    zip_code = str(data[field])
                    data[field] = zip_code[:3] + "**"
        
        return data
    
    def _apply_differential_privacy(self, data: Dict[str, Any], 
                                  epsilon: float = 1.0) -> Dict[str, Any]:
        """Apply differential privacy (simplified implementation)."""
        import numpy as np
        
        # Add Laplace noise to numerical fields
        for field, value in data.items():
            if isinstance(value, (int, float)) and field not in ['id', 'timestamp']:
                # Calculate sensitivity (simplified)
                sensitivity = abs(value) * 0.1  # Simplified sensitivity calculation
                
                # Add Laplace noise
                noise = np.random.laplace(0, sensitivity / epsilon)
                data[field] = value + noise
        
        return data
    
    def cleanup_expired_data(self):
        """Clean up expired data according to policies."""
        expired_data = []
        current_time = datetime.now()
        
        for data_id, registry_entry in self.data_registry.items():
            if current_time > registry_entry['retention_until']:
                expired_data.append(data_id)
        
        for data_id in expired_data:
            registry_entry = self.data_registry[data_id]
            classification = DataClassification(registry_entry['classification'])
            policy = self.policies[classification]
            
            # Apply appropriate deletion method
            self._secure_delete_data(data_id, policy.deletion_method)
            
            # Remove from registry
            del self.data_registry[data_id]
            
            self.access_logger.info(f"Data deleted (expired): {data_id}, "
                                   f"method: {policy.deletion_method}")
    
    def _secure_delete_data(self, data_id: str, deletion_method: str):
        """Securely delete data according to policy."""
        if deletion_method == "cryptographic_erasure":
            # Delete encryption keys to make data unrecoverable
            # In production, this would integrate with key management system
            pass
        elif deletion_method == "secure_wipe":
            # Perform secure overwrite of data
            # In production, this would securely overwrite storage
            pass
        # Standard deletion is handled by removing from registry
    
    def get_data_inventory(self) -> List[Dict[str, Any]]:
        """Get inventory of all registered data."""
        inventory = []
        
        for data_id, registry_entry in self.data_registry.items():
            inventory.append({
                'data_id': data_id,
                'classification': registry_entry['classification'],
                'categories': registry_entry['categories'],
                'registered_at': registry_entry['registered_at'],
                'retention_until': registry_entry['retention_until'],
                'access_count': registry_entry['access_count'],
                'last_accessed': registry_entry['last_accessed'],
                'anonymized': registry_entry['anonymized']
            })
        
        return inventory

# Usage example
def demonstrate_secure_data_handling():
    """Demonstrate secure data handling."""
    handler = SecureDataHandler()
    
    # Sample data with different sensitivity levels
    samples = [
        {
            'data': {'temperature': 25.5, 'humidity': 60.2, 'location': 'factory_floor'},
            'categories': [DataCategory.TECHNICAL_DATA],
            'purpose': 'anomaly_detection'
        },
        {
            'data': {'patient_id': 'P12345', 'blood_pressure': 120, 'heart_rate': 75},
            'categories': [DataCategory.HEALTH_DATA, DataCategory.PERSONAL_DATA],
            'purpose': 'health_monitoring'
        },
        {
            'data': {'account_balance': 50000, 'transaction_amount': 1000, 'user_id': 'U789'},
            'categories': [DataCategory.FINANCIAL_DATA, DataCategory.PERSONAL_DATA],
            'purpose': 'fraud_detection'
        }
    ]
    
    # Register and process data
    for i, sample in enumerate(samples):
        data_id = f"data_{i+1}"
        result = handler.register_data(
            data_id=data_id,
            data_sample=sample['data'],
            categories=sample['categories'],
            user_id="analyst_001",
            purpose=sample['purpose']
        )
        
        print(f"Data {data_id}:")
        print(f"  Classification: {result['classification']}")
        print(f"  Encryption required: {result['policy'].encryption_required}")
        print(f"  Anonymization required: {result['policy'].anonymization_required}")
        print()
    
    # Demonstrate data access
    print("Accessing data...")
    registry_entry = handler.access_data("data_2", "analyst_001", "model_training")
    if registry_entry:
        print(f"Access granted, access count: {registry_entry['access_count']}")
    
    # Show data inventory
    print("\nData Inventory:")
    inventory = handler.get_data_inventory()
    for item in inventory:
        print(f"  {item['data_id']}: {item['classification']} "
              f"(accessed {item['access_count']} times)")

if __name__ == "__main__":
    demonstrate_secure_data_handling()
```

### Data Loss Prevention

```python
# security/data_loss_prevention.py
import re
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class DLPAction(Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    REDACT = "redact"
    ENCRYPT = "encrypt"

class DLPRuleType(Enum):
    REGEX_PATTERN = "regex_pattern"
    KEYWORD_MATCH = "keyword_match"
    DATA_TYPE_CHECK = "data_type_check"
    STATISTICAL_ANALYSIS = "statistical_analysis"

@dataclass
class DLPRule:
    """Data Loss Prevention rule definition."""
    id: str
    name: str
    description: str
    rule_type: DLPRuleType
    pattern: str
    action: DLPAction
    severity: str  # "low", "medium", "high", "critical"
    enabled: bool = True

class DataLossPreventionSystem:
    """Data Loss Prevention system for anomaly detection pipeline."""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.logger = logging.getLogger("dlp_system")
        self.violation_count = 0
        
    def _initialize_rules(self) -> List[DLPRule]:
        """Initialize DLP rules."""
        return [
            # Personal Identifiable Information (PII)
            DLPRule(
                id="PII_SSN",
                name="Social Security Number",
                description="Detect US Social Security Numbers",
                rule_type=DLPRuleType.REGEX_PATTERN,
                pattern=r'\b\d{3}-?\d{2}-?\d{4}\b',
                action=DLPAction.REDACT,
                severity="critical"
            ),
            
            DLPRule(
                id="PII_EMAIL",
                name="Email Address",
                description="Detect email addresses",
                rule_type=DLPRuleType.REGEX_PATTERN,
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                action=DLPAction.REDACT,
                severity="medium"
            ),
            
            DLPRule(
                id="PII_PHONE",
                name="Phone Number",
                description="Detect phone numbers",
                rule_type=DLPRuleType.REGEX_PATTERN,
                pattern=r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
                action=DLPAction.REDACT,
                severity="medium"
            ),
            
            # Financial Information
            DLPRule(
                id="FIN_CREDIT_CARD",
                name="Credit Card Number",
                description="Detect credit card numbers",
                rule_type=DLPRuleType.REGEX_PATTERN,
                pattern=r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
                action=DLPAction.BLOCK,
                severity="critical"
            ),
            
            DLPRule(
                id="FIN_BANK_ACCOUNT",
                name="Bank Account Number",
                description="Detect bank account numbers",
                rule_type=DLPRuleType.REGEX_PATTERN,
                pattern=r'\b[0-9]{8,17}\b',
                action=DLPAction.WARN,
                severity="high"
            ),
            
            # Health Information
            DLPRule(
                id="HEALTH_MEDICAL_RECORD",
                name="Medical Record Number",
                description="Detect medical record numbers",
                rule_type=DLPRuleType.REGEX_PATTERN,
                pattern=r'\b(MRN|MR|MEDICAL RECORD)\s*:?\s*[A-Z0-9]{6,12}\b',
                action=DLPAction.BLOCK,
                severity="critical"
            ),
            
            # Keywords and sensitive terms
            DLPRule(
                id="KEYWORD_CONFIDENTIAL",
                name="Confidential Keywords",
                description="Detect confidential data markers",
                rule_type=DLPRuleType.KEYWORD_MATCH,
                pattern="confidential|proprietary|trade secret|classified|restricted",
                action=DLPAction.WARN,
                severity="high"
            ),
            
            # Statistical anomalies
            DLPRule(
                id="STAT_OUTLIER_DETECTION",
                name="Statistical Outliers",
                description="Detect unusual data distributions",
                rule_type=DLPRuleType.STATISTICAL_ANALYSIS,
                pattern="z_score>3",
                action=DLPAction.WARN,
                severity="medium"
            )
        ]
    
    def scan_data(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Scan data for DLP violations."""
        violations = []
        actions_taken = []
        processed_data = data
        
        # Convert data to string for pattern matching
        if isinstance(data, dict):
            data_str = str(data)
            data_dict = data
        elif isinstance(data, (list, tuple)):
            data_str = str(data)
            data_dict = {"data": data}
        else:
            data_str = str(data)
            data_dict = {"value": data}
        
        # Apply each enabled rule
        for rule in self.rules:
            if not rule.enabled:
                continue
                
            violation = self._apply_rule(rule, data_str, data_dict, context)
            
            if violation:
                violations.append(violation)
                
                # Take action based on rule
                action_result = self._take_action(rule, processed_data, violation)
                actions_taken.append(action_result)
                
                if action_result.get('data_modified'):
                    processed_data = action_result['modified_data']
                
                # Log violation
                self._log_violation(rule, violation, context)
        
        # Update violation count
        self.violation_count += len(violations)
        
        return {
            'violations_found': len(violations),
            'violations': violations,
            'actions_taken': actions_taken,
            'processed_data': processed_data,
            'scan_passed': len([v for v in violations if v['action'] == DLPAction.BLOCK]) == 0
        }
    
    def _apply_rule(self, rule: DLPRule, data_str: str, data_dict: Dict[str, Any], 
                   context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Apply a single DLP rule to data."""
        
        if rule.rule_type == DLPRuleType.REGEX_PATTERN:
            matches = re.finditer(rule.pattern, data_str, re.IGNORECASE)
            matches_list = list(matches)
            
            if matches_list:
                return {
                    'rule_id': rule.id,
                    'rule_name': rule.name,
                    'rule_type': rule.rule_type.value,
                    'action': rule.action,
                    'severity': rule.severity,
                    'matches': [
                        {
                            'text': match.group(),
                            'start': match.start(),
                            'end': match.end()
                        }
                        for match in matches_list
                    ],
                    'match_count': len(matches_list)
                }
        
        elif rule.rule_type == DLPRuleType.KEYWORD_MATCH:
            keywords = [kw.strip().lower() for kw in rule.pattern.split('|')]
            found_keywords = [kw for kw in keywords if kw in data_str.lower()]
            
            if found_keywords:
                return {
                    'rule_id': rule.id,
                    'rule_name': rule.name,
                    'rule_type': rule.rule_type.value,
                    'action': rule.action,
                    'severity': rule.severity,
                    'keywords_found': found_keywords,
                    'match_count': len(found_keywords)
                }
        
        elif rule.rule_type == DLPRuleType.STATISTICAL_ANALYSIS:
            # Simplified statistical analysis
            numerical_values = []
            
            def extract_numbers(obj):
                if isinstance(obj, (int, float)):
                    numerical_values.append(float(obj))
                elif isinstance(obj, dict):
                    for value in obj.values():
                        extract_numbers(value)
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        extract_numbers(item)
            
            extract_numbers(data_dict)
            
            if numerical_values and len(numerical_values) >= 3:
                import numpy as np
                
                z_scores = np.abs((numerical_values - np.mean(numerical_values)) / np.std(numerical_values))
                outliers = [val for val, z in zip(numerical_values, z_scores) if z > 3]
                
                if outliers:
                    return {
                        'rule_id': rule.id,
                        'rule_name': rule.name,
                        'rule_type': rule.rule_type.value,
                        'action': rule.action,
                        'severity': rule.severity,
                        'outliers': outliers,
                        'z_scores': z_scores.tolist(),
                        'match_count': len(outliers)
                    }
        
        return None
    
    def _take_action(self, rule: DLPRule, data: Any, violation: Dict[str, Any]) -> Dict[str, Any]:
        """Take action based on DLP rule violation."""
        action_result = {
            'rule_id': rule.id,
            'action_taken': rule.action.value,
            'data_modified': False,
            'modified_data': data
        }
        
        if rule.action == DLPAction.REDACT:
            # Redact sensitive information
            if isinstance(data, str):
                modified_data = data
                if 'matches' in violation:
                    for match in violation['matches']:
                        # Replace with asterisks
                        replacement = '*' * len(match['text'])
                        modified_data = modified_data.replace(match['text'], replacement)
                
                action_result['data_modified'] = True
                action_result['modified_data'] = modified_data
            
            elif isinstance(data, dict):
                modified_data = data.copy()
                # Redact dictionary values that contain sensitive information
                for key, value in modified_data.items():
                    if isinstance(value, str) and 'matches' in violation:
                        for match in violation['matches']:
                            if match['text'] in value:
                                replacement = '*' * len(match['text'])
                                modified_data[key] = value.replace(match['text'], replacement)
                
                action_result['data_modified'] = True
                action_result['modified_data'] = modified_data
        
        elif rule.action == DLPAction.ENCRYPT:
            # Encrypt sensitive data
            if isinstance(data, (str, dict)):
                encrypted_data = self._encrypt_sensitive_data(data, violation)
                action_result['data_modified'] = True
                action_result['modified_data'] = encrypted_data
        
        elif rule.action == DLPAction.BLOCK:
            # Block the data entirely
            action_result['data_modified'] = True
            action_result['modified_data'] = None
            action_result['blocked'] = True
        
        return action_result
    
    def _encrypt_sensitive_data(self, data: Any, violation: Dict[str, Any]) -> Any:
        """Encrypt sensitive portions of data."""
        # Simplified encryption - in production use proper encryption
        if isinstance(data, str) and 'matches' in violation:
            modified_data = data
            for match in violation['matches']:
                # Replace with hash
                hash_value = hashlib.sha256(match['text'].encode()).hexdigest()[:16]
                encrypted_replacement = f"[ENCRYPTED:{hash_value}]"
                modified_data = modified_data.replace(match['text'], encrypted_replacement)
            return modified_data
        
        return data
    
    def _log_violation(self, rule: DLPRule, violation: Dict[str, Any], 
                      context: Dict[str, Any] = None):
        """Log DLP violation."""
        log_entry = {
            'timestamp': self._get_timestamp(),
            'rule_id': rule.id,
            'rule_name': rule.name,
            'severity': rule.severity,
            'action': rule.action.value,
            'match_count': violation.get('match_count', 0),
            'context': context or {}
        }
        
        if rule.severity in ['critical', 'high']:
            self.logger.warning(f"DLP Violation: {log_entry}")
        else:
            self.logger.info(f"DLP Detection: {log_entry}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def add_custom_rule(self, rule: DLPRule):
        """Add custom DLP rule."""
        # Check if rule already exists
        existing_rule = next((r for r in self.rules if r.id == rule.id), None)
        
        if existing_rule:
            # Update existing rule
            existing_rule.pattern = rule.pattern
            existing_rule.action = rule.action
            existing_rule.severity = rule.severity
            existing_rule.enabled = rule.enabled
        else:
            # Add new rule
            self.rules.append(rule)
        
        self.logger.info(f"DLP rule {'updated' if existing_rule else 'added'}: {rule.id}")
    
    def disable_rule(self, rule_id: str):
        """Disable a DLP rule."""
        rule = next((r for r in self.rules if r.id == rule_id), None)
        if rule:
            rule.enabled = False
            self.logger.info(f"DLP rule disabled: {rule_id}")
    
    def enable_rule(self, rule_id: str):
        """Enable a DLP rule."""
        rule = next((r for r in self.rules if r.id == rule_id), None)
        if rule:
            rule.enabled = True
            self.logger.info(f"DLP rule enabled: {rule_id}")
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of DLP violations."""
        return {
            'total_violations': self.violation_count,
            'active_rules': len([r for r in self.rules if r.enabled]),
            'total_rules': len(self.rules),
            'rules_by_severity': {
                'critical': len([r for r in self.rules if r.severity == 'critical']),
                'high': len([r for r in self.rules if r.severity == 'high']),
                'medium': len([r for r in self.rules if r.severity == 'medium']),
                'low': len([r for r in self.rules if r.severity == 'low'])
            }
        }

# Usage example
def demonstrate_dlp_system():
    """Demonstrate Data Loss Prevention system."""
    dlp = DataLossPreventionSystem()
    
    # Test data with various sensitive information
    test_data = [
        "John Doe, SSN: 123-45-6789, Email: john.doe@company.com",
        {
            'user_id': 'U123',
            'credit_card': '4532-1234-5678-9012',
            'phone': '(555) 123-4567',
            'balance': 50000
        },
        "This document contains CONFIDENTIAL information about our trade secrets.",
        [25.5, 26.1, 25.8, 150.0, 26.2]  # Contains statistical outlier
    ]
    
    print("DLP System Demonstration")
    print("=" * 40)
    
    for i, data in enumerate(test_data, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {data}")
        
        # Scan data
        result = dlp.scan_data(data, context={'source': f'test_case_{i}'})
        
        print(f"Violations found: {result['violations_found']}")
        print(f"Scan passed: {result['scan_passed']}")
        
        if result['violations_found'] > 0:
            for violation in result['violations']:
                print(f"  - {violation['rule_name']} ({violation['severity']}): {violation['action'].value}")
        
        if result['processed_data'] != data:
            print(f"Modified data: {result['processed_data']}")
    
    # Show summary
    print(f"\nViolation Summary:")
    summary = dlp.get_violation_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    demonstrate_dlp_system()
```

## Authentication and Authorization

### Multi-factor Authentication System

```python
# security/authentication.py
import hashlib
import secrets
import time
import pyotp
import qrcode
import io
import base64
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import jwt
import bcrypt

class AuthenticationMethod(Enum):
    PASSWORD = "password"
    TOTP = "totp"  # Time-based One-Time Password
    SMS = "sms"
    EMAIL = "email"
    BIOMETRIC = "biometric"
    HARDWARE_TOKEN = "hardware_token"

class UserRole(Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"

class Permission(Enum):
    # Data permissions
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    EXPORT_DATA = "export_data"
    
    # Model permissions
    TRAIN_MODEL = "train_model"
    DEPLOY_MODEL = "deploy_model"
    DELETE_MODEL = "delete_model"
    VIEW_MODEL = "view_model"
    
    # System permissions
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    CONFIGURE_SYSTEM = "configure_system"
    MANAGE_API_KEYS = "manage_api_keys"

@dataclass
class User:
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: List[UserRole]
    mfa_enabled: bool
    mfa_secret: Optional[str]
    created_at: datetime
    last_login: Optional[datetime]
    failed_login_attempts: int
    account_locked: bool
    password_expires_at: Optional[datetime]

@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    mfa_verified: bool
    permissions: List[Permission]

class SecureAuthenticationSystem:
    """Secure authentication system with MFA support."""
    
    def __init__(self, jwt_secret: str, session_timeout_minutes: int = 60):
        self.jwt_secret = jwt_secret
        self.session_timeout_minutes = session_timeout_minutes
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.role_permissions = self._initialize_role_permissions()
        
        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 30
        self.password_min_length = 12
        self.password_complexity_required = True
        
    def _initialize_role_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Initialize role-based permissions."""
        return {
            UserRole.ADMIN: [
                Permission.READ_DATA, Permission.WRITE_DATA, Permission.DELETE_DATA, 
                Permission.EXPORT_DATA, Permission.TRAIN_MODEL, Permission.DEPLOY_MODEL,
                Permission.DELETE_MODEL, Permission.VIEW_MODEL, Permission.MANAGE_USERS,
                Permission.VIEW_AUDIT_LOGS, Permission.CONFIGURE_SYSTEM, Permission.MANAGE_API_KEYS
            ],
            UserRole.ANALYST: [
                Permission.READ_DATA, Permission.WRITE_DATA, Permission.EXPORT_DATA,
                Permission.TRAIN_MODEL, Permission.DEPLOY_MODEL, Permission.VIEW_MODEL
            ],
            UserRole.VIEWER: [
                Permission.READ_DATA, Permission.VIEW_MODEL
            ],
            UserRole.API_USER: [
                Permission.READ_DATA, Permission.VIEW_MODEL
            ]
        }
    
    def create_user(self, username: str, email: str, password: str, 
                   roles: List[UserRole]) -> Dict[str, Any]:
        """Create a new user with secure password handling."""
        
        # Validate password strength
        password_validation = self._validate_password_strength(password)
        if not password_validation['valid']:
            raise ValueError(f"Password validation failed: {password_validation['message']}")
        
        # Check if user already exists
        if any(user.username == username or user.email == email for user in self.users.values()):
            raise ValueError("User with this username or email already exists")
        
        # Generate user ID and hash password
        user_id = self._generate_secure_id()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Create user
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles,
            mfa_enabled=False,
            mfa_secret=None,
            created_at=datetime.now(),
            last_login=None,
            failed_login_attempts=0,
            account_locked=False,
            password_expires_at=datetime.now() + timedelta(days=90)
        )
        
        self.users[user_id] = user
        
        return {
            'user_id': user_id,
            'username': username,
            'roles': [role.value for role in roles],
            'mfa_enabled': False
        }
    
    def _validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength."""
        if len(password) < self.password_min_length:
            return {
                'valid': False,
                'message': f'Password must be at least {self.password_min_length} characters long'
            }
        
        if self.password_complexity_required:
            has_upper = any(c.isupper() for c in password)
            has_lower = any(c.islower() for c in password)
            has_digit = any(c.isdigit() for c in password)
            has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password)
            
            if not all([has_upper, has_lower, has_digit, has_special]):
                return {
                    'valid': False,
                    'message': 'Password must contain uppercase, lowercase, digit, and special characters'
                }
        
        # Check against common passwords (simplified)
        common_passwords = ['password', '123456', 'qwerty', 'admin', 'letmein']
        if password.lower() in common_passwords:
            return {
                'valid': False,
                'message': 'Password is too common'
            }
        
        return {'valid': True, 'message': 'Password is strong'}
    
    def enable_mfa(self, user_id: str) -> Dict[str, str]:
        """Enable multi-factor authentication for user."""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Generate TOTP secret
        secret = pyotp.random_base32()
        user.mfa_secret = secret
        user.mfa_enabled = True
        
        # Generate QR code
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user.email,
            issuer_name="Anomaly Detection System"
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        # Convert QR code to base64 string
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        qr_code_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            'secret': secret,
            'qr_code': qr_code_base64,
            'backup_codes': self._generate_backup_codes()
        }
    
    def _generate_backup_codes(self) -> List[str]:
        """Generate backup codes for MFA."""
        return [secrets.token_hex(4).upper() for _ in range(10)]
    
    def authenticate(self, username: str, password: str, 
                    totp_code: Optional[str] = None,
                    ip_address: str = "unknown",
                    user_agent: str = "unknown") -> Dict[str, Any]:
        """Authenticate user with optional MFA."""
        
        # Find user
        user = next((u for u in self.users.values() 
                    if u.username == username), None)
        
        if not user:
            # Prevent username enumeration
            bcrypt.checkpw(b"dummy", bcrypt.gensalt())
            raise ValueError("Invalid credentials")
        
        # Check if account is locked
        if user.account_locked:
            raise ValueError("Account is locked due to too many failed attempts")
        
        # Check password expiration
        if user.password_expires_at and datetime.now() > user.password_expires_at:
            raise ValueError("Password has expired")
        
        # Verify password
        if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            user.failed_login_attempts += 1
            
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.account_locked = True
                # In production, schedule unlock after lockout duration
            
            raise ValueError("Invalid credentials")
        
        # Check MFA if enabled
        if user.mfa_enabled:
            if not totp_code:
                return {
                    'authentication_status': 'mfa_required',
                    'user_id': user.user_id,
                    'message': 'Multi-factor authentication required'
                }
            
            # Verify TOTP code
            totp = pyotp.TOTP(user.mfa_secret)
            if not totp.verify(totp_code, valid_window=1):
                user.failed_login_attempts += 1
                raise ValueError("Invalid MFA code")
        
        # Authentication successful
        user.failed_login_attempts = 0
        user.last_login = datetime.now()
        
        # Create session
        session = self._create_session(user, ip_address, user_agent)
        
        # Generate JWT token
        jwt_token = self._generate_jwt_token(user, session)
        
        return {
            'authentication_status': 'success',
            'user_id': user.user_id,
            'username': user.username,
            'session_id': session.session_id,
            'jwt_token': jwt_token,
            'expires_at': session.expires_at.isoformat(),
            'permissions': [perm.value for perm in session.permissions]
        }
    
    def _create_session(self, user: User, ip_address: str, user_agent: str) -> Session:
        """Create user session."""
        session_id = self._generate_secure_id()
        
        # Get user permissions based on roles
        permissions = set()
        for role in user.roles:
            permissions.update(self.role_permissions.get(role, []))
        
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=self.session_timeout_minutes),
            ip_address=ip_address,
            user_agent=user_agent,
            mfa_verified=user.mfa_enabled,
            permissions=list(permissions)
        )
        
        self.sessions[session_id] = session
        return session
    
    def _generate_jwt_token(self, user: User, session: Session) -> str:
        """Generate JWT token."""
        payload = {
            'user_id': user.user_id,
            'session_id': session.session_id,
            'username': user.username,
            'roles': [role.value for role in user.roles],
            'permissions': [perm.value for perm in session.permissions],
            'iat': int(time.time()),
            'exp': int(session.expires_at.timestamp())
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return user information."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Check if session still exists and is valid
            session = self.sessions.get(payload['session_id'])
            if not session or datetime.now() > session.expires_at:
                return None
            
            return payload
            
        except jwt.InvalidTokenError:
            return None
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission."""
        user = self.users.get(user_id)
        if not user:
            return False
        
        # Get user permissions
        permissions = set()
        for role in user.roles:
            permissions.update(self.role_permissions.get(role, []))
        
        return permission in permissions
    
    def logout(self, session_id: str):
        """Logout user by invalidating session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def _generate_secure_id(self) -> str:
        """Generate cryptographically secure ID."""
        return secrets.token_urlsafe(32)
    
    def change_password(self, user_id: str, old_password: str, new_password: str):
        """Change user password with validation."""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Verify old password
        if not bcrypt.checkpw(old_password.encode('utf-8'), user.password_hash.encode('utf-8')):
            raise ValueError("Invalid current password")
        
        # Validate new password
        password_validation = self._validate_password_strength(new_password)
        if not password_validation['valid']:
            raise ValueError(f"New password validation failed: {password_validation['message']}")
        
        # Update password
        user.password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        user.password_expires_at = datetime.now() + timedelta(days=90)
        
        # Invalidate all existing sessions for security
        user_sessions = [sid for sid, session in self.sessions.items() 
                        if session.user_id == user_id]
        for session_id in user_sessions:
            del self.sessions[session_id]
    
    def unlock_account(self, user_id: str):
        """Unlock locked user account (admin function)."""
        user = self.users.get(user_id)
        if user:
            user.account_locked = False
            user.failed_login_attempts = 0
    
    def get_user_audit_info(self, user_id: str) -> Dict[str, Any]:
        """Get user audit information."""
        user = self.users.get(user_id)
        if not user:
            return {}
        
        active_sessions = [
            {
                'session_id': session.session_id,
                'created_at': session.created_at.isoformat(),
                'expires_at': session.expires_at.isoformat(),
                'ip_address': session.ip_address,
                'user_agent': session.user_agent
            }
            for session in self.sessions.values()
            if session.user_id == user_id
        ]
        
        return {
            'user_id': user.user_id,
            'username': user.username,
            'email': user.email,
            'roles': [role.value for role in user.roles],
            'mfa_enabled': user.mfa_enabled,
            'created_at': user.created_at.isoformat(),
            'last_login': user.last_login.isoformat() if user.last_login else None,
            'failed_login_attempts': user.failed_login_attempts,
            'account_locked': user.account_locked,
            'password_expires_at': user.password_expires_at.isoformat() if user.password_expires_at else None,
            'active_sessions': active_sessions
        }

# Usage example
def demonstrate_authentication_system():
    """Demonstrate secure authentication system."""
    auth_system = SecureAuthenticationSystem(jwt_secret="your-secret-key")
    
    # Create users
    admin_user = auth_system.create_user(
        username="admin",
        email="admin@company.com",
        password="SecureAdminPass123!",
        roles=[UserRole.ADMIN]
    )
    
    analyst_user = auth_system.create_user(
        username="analyst",
        email="analyst@company.com", 
        password="AnalystSecure456!",
        roles=[UserRole.ANALYST]
    )
    
    print("Users created successfully")
    
    # Enable MFA for admin
    mfa_setup = auth_system.enable_mfa(admin_user['user_id'])
    print(f"MFA enabled for admin. Secret: {mfa_setup['secret']}")
    
    # Authenticate analyst (no MFA)
    auth_result = auth_system.authenticate(
        username="analyst",
        password="AnalystSecure456!",
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0..."
    )
    
    print(f"Analyst authenticated: {auth_result['authentication_status']}")
    print(f"Permissions: {auth_result['permissions']}")
    
    # Check permissions
    can_train_model = auth_system.check_permission(
        analyst_user['user_id'], 
        Permission.TRAIN_MODEL
    )
    print(f"Analyst can train models: {can_train_model}")
    
    can_manage_users = auth_system.check_permission(
        analyst_user['user_id'],
        Permission.MANAGE_USERS
    )
    print(f"Analyst can manage users: {can_manage_users}")

if __name__ == "__main__":
    demonstrate_authentication_system()
```

This comprehensive security guide provides robust security measures for anomaly detection systems, covering data protection, authentication, authorization, and privacy protection. The implementations include production-ready patterns for secure deployment and compliance with security best practices.