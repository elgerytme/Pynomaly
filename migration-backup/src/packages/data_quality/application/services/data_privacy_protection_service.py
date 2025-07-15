"""
Data Privacy Protection Service - Comprehensive PII detection and masking for enterprise compliance.

This service provides advanced privacy protection features including PII detection,
data masking, anonymization, and privacy compliance management for GDPR, HIPAA,
and other regulatory frameworks.
"""

import asyncio
import logging
import re
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import pandas as pd
import numpy as np

from core.shared.error_handling import handle_exceptions

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of Personally Identifiable Information."""
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    EMAIL = "email"
    PHONE = "phone"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    DRIVER_LICENSE = "driver_license"
    PASSPORT = "passport"
    MEDICAL_ID = "medical_id"
    BANK_ACCOUNT = "bank_account"
    CUSTOM = "custom"


class MaskingStrategy(Enum):
    """Data masking strategies."""
    REDACTION = "redaction"           # Replace with [REDACTED]
    HASH = "hash"                     # One-way hash
    TOKENIZATION = "tokenization"     # Replace with tokens
    PARTIAL_MASK = "partial_mask"     # Show first/last few characters
    PSEUDONYMIZATION = "pseudonymization"  # Replace with fake but consistent data
    ENCRYPTION = "encryption"         # Reversible encryption
    RANDOMIZATION = "randomization"   # Random replacement
    NULL_OUT = "null_out"            # Replace with NULL/None


class ComplianceStandard(Enum):
    """Data privacy compliance standards."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    CCPA = "ccpa"
    FERPA = "ferpa"
    CUSTOM = "custom"


@dataclass
class PIIPattern:
    """Pattern definition for PII detection."""
    pii_type: PIIType
    pattern: str
    regex_flags: int = re.IGNORECASE
    confidence_threshold: float = 0.8
    validation_function: Optional[str] = None
    description: str = ""


@dataclass
class MaskingRule:
    """Rule for data masking."""
    pii_type: PIIType
    strategy: MaskingStrategy
    preserve_length: bool = True
    preserve_format: bool = False
    replacement_char: str = "*"
    token_prefix: str = "TOK_"
    encryption_key: Optional[str] = None


@dataclass
class PrivacyPolicy:
    """Privacy protection policy."""
    name: str
    compliance_standards: List[ComplianceStandard]
    pii_patterns: List[PIIPattern] = field(default_factory=list)
    masking_rules: List[MaskingRule] = field(default_factory=list)
    retention_period_days: Optional[int] = None
    require_explicit_consent: bool = True
    allow_data_export: bool = False
    auto_delete_expired: bool = True
    audit_all_access: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PIIDetectionResult:
    """Result of PII detection."""
    column_name: str
    pii_type: PIIType
    confidence_score: float
    pattern_matched: str
    sample_values: List[str] = field(default_factory=list)
    total_matches: int = 0
    percentage_matches: float = 0.0


@dataclass
class PrivacyReport:
    """Comprehensive privacy protection report."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    dataset_id: str = ""
    policy_applied: str = ""
    
    # Detection results
    pii_detections: List[PIIDetectionResult] = field(default_factory=list)
    sensitive_columns: List[str] = field(default_factory=list)
    
    # Processing results
    rows_processed: int = 0
    columns_masked: int = 0
    privacy_violations: List[str] = field(default_factory=list)
    
    # Compliance status
    compliance_standards_met: List[ComplianceStandard] = field(default_factory=list)
    compliance_issues: List[str] = field(default_factory=list)
    
    # Performance metrics
    processing_time_seconds: float = 0.0
    detection_accuracy: float = 0.0


class DataPrivacyProtectionService:
    """Comprehensive data privacy protection service."""
    
    def __init__(self):
        """Initialize the data privacy protection service."""
        self.privacy_policies: Dict[str, PrivacyPolicy] = {}
        self.pii_patterns: Dict[PIIType, List[PIIPattern]] = {}
        self.masking_rules: Dict[PIIType, MaskingRule] = {}
        self.token_mappings: Dict[str, str] = {}
        self.encryption_keys: Dict[str, str] = {}
        
        # Initialize default patterns
        self._initialize_default_patterns()
        self._initialize_default_policies()
        
        logger.info("Data Privacy Protection Service initialized")
    
    def _initialize_default_patterns(self):
        """Initialize default PII detection patterns."""
        default_patterns = [
            PIIPattern(
                pii_type=PIIType.SSN,
                pattern=r'\b\d{3}-?\d{2}-?\d{4}\b',
                description="US Social Security Number"
            ),
            PIIPattern(
                pii_type=PIIType.CREDIT_CARD,
                pattern=r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                description="Credit card number"
            ),
            PIIPattern(
                pii_type=PIIType.EMAIL,
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                description="Email address"
            ),
            PIIPattern(
                pii_type=PIIType.PHONE,
                pattern=r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                description="Phone number"
            ),
            PIIPattern(
                pii_type=PIIType.IP_ADDRESS,
                pattern=r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                description="IP address"
            ),
            PIIPattern(
                pii_type=PIIType.DATE_OF_BIRTH,
                pattern=r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b',
                description="Date of birth (MM/DD/YYYY)"
            ),
            PIIPattern(
                pii_type=PIIType.DRIVER_LICENSE,
                pattern=r'\b[A-Z]{1,2}\d{6,8}\b',
                description="Driver license number"
            )
        ]
        
        for pattern in default_patterns:
            if pattern.pii_type not in self.pii_patterns:
                self.pii_patterns[pattern.pii_type] = []
            self.pii_patterns[pattern.pii_type].append(pattern)
    
    def _initialize_default_policies(self):
        """Initialize default privacy policies for common compliance standards."""
        # GDPR Policy
        gdpr_policy = PrivacyPolicy(
            name="GDPR_Compliance",
            compliance_standards=[ComplianceStandard.GDPR],
            retention_period_days=2555,  # 7 years
            require_explicit_consent=True,
            allow_data_export=True,
            auto_delete_expired=True,
            audit_all_access=True
        )
        
        # HIPAA Policy
        hipaa_policy = PrivacyPolicy(
            name="HIPAA_Compliance",
            compliance_standards=[ComplianceStandard.HIPAA],
            retention_period_days=2190,  # 6 years
            require_explicit_consent=True,
            allow_data_export=False,
            auto_delete_expired=True,
            audit_all_access=True
        )
        
        # PCI DSS Policy
        pci_policy = PrivacyPolicy(
            name="PCI_DSS_Compliance",
            compliance_standards=[ComplianceStandard.PCI_DSS],
            retention_period_days=365,  # 1 year for most data
            require_explicit_consent=False,
            allow_data_export=False,
            auto_delete_expired=True,
            audit_all_access=True
        )
        
        self.privacy_policies["gdpr"] = gdpr_policy
        self.privacy_policies["hipaa"] = hipaa_policy
        self.privacy_policies["pci_dss"] = pci_policy
        
        # Initialize default masking rules
        self._initialize_default_masking_rules()
    
    def _initialize_default_masking_rules(self):
        """Initialize default masking rules for different PII types."""
        default_rules = [
            MaskingRule(
                pii_type=PIIType.SSN,
                strategy=MaskingStrategy.PARTIAL_MASK,
                preserve_length=True,
                replacement_char="*"
            ),
            MaskingRule(
                pii_type=PIIType.CREDIT_CARD,
                strategy=MaskingStrategy.PARTIAL_MASK,
                preserve_length=True,
                replacement_char="*"
            ),
            MaskingRule(
                pii_type=PIIType.EMAIL,
                strategy=MaskingStrategy.PARTIAL_MASK,
                preserve_length=False,
                replacement_char="*"
            ),
            MaskingRule(
                pii_type=PIIType.PHONE,
                strategy=MaskingStrategy.PARTIAL_MASK,
                preserve_length=True,
                replacement_char="*"
            ),
            MaskingRule(
                pii_type=PIIType.NAME,
                strategy=MaskingStrategy.PSEUDONYMIZATION,
                preserve_length=False
            )
        ]
        
        for rule in default_rules:
            self.masking_rules[rule.pii_type] = rule
    
    @handle_exceptions
    async def detect_pii(
        self,
        data: Union[pd.DataFrame, Dict[str, List], List[Dict]],
        policy_name: Optional[str] = None,
        custom_patterns: Optional[List[PIIPattern]] = None
    ) -> List[PIIDetectionResult]:
        """
        Detect PII in the provided data.
        
        Args:
            data: Data to scan for PII
            policy_name: Privacy policy to use
            custom_patterns: Additional custom patterns
            
        Returns:
            List of PII detection results
        """
        logger.info("Starting PII detection")
        
        # Convert data to DataFrame for consistent processing
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        detection_results = []
        patterns_to_use = self._get_patterns_for_policy(policy_name, custom_patterns)
        
        # Scan each column
        for column_name in df.columns:
            column_data = df[column_name].astype(str)
            
            for pii_type, patterns in patterns_to_use.items():
                for pattern in patterns:
                    matches = self._find_pattern_matches(column_data, pattern)
                    
                    if matches['total_matches'] > 0:
                        confidence = self._calculate_confidence(
                            matches['total_matches'],
                            len(column_data),
                            pattern.confidence_threshold
                        )
                        
                        if confidence >= pattern.confidence_threshold:
                            detection_results.append(PIIDetectionResult(
                                column_name=column_name,
                                pii_type=pii_type,
                                confidence_score=confidence,
                                pattern_matched=pattern.pattern,
                                sample_values=matches['sample_values'],
                                total_matches=matches['total_matches'],
                                percentage_matches=matches['percentage_matches']
                            ))
        
        logger.info(f"PII detection completed. Found {len(detection_results)} potential PII fields")
        return detection_results
    
    def _get_patterns_for_policy(
        self,
        policy_name: Optional[str],
        custom_patterns: Optional[List[PIIPattern]]
    ) -> Dict[PIIType, List[PIIPattern]]:
        """Get PII patterns for the specified policy."""
        patterns = self.pii_patterns.copy()
        
        if policy_name and policy_name in self.privacy_policies:
            policy = self.privacy_policies[policy_name]
            if policy.pii_patterns:
                # Use policy-specific patterns
                policy_patterns = {}
                for pattern in policy.pii_patterns:
                    if pattern.pii_type not in policy_patterns:
                        policy_patterns[pattern.pii_type] = []
                    policy_patterns[pattern.pii_type].append(pattern)
                patterns = policy_patterns
        
        # Add custom patterns
        if custom_patterns:
            for pattern in custom_patterns:
                if pattern.pii_type not in patterns:
                    patterns[pattern.pii_type] = []
                patterns[pattern.pii_type].append(pattern)
        
        return patterns
    
    def _find_pattern_matches(self, column_data: pd.Series, pattern: PIIPattern) -> Dict[str, Any]:
        """Find matches for a specific pattern in column data."""
        regex = re.compile(pattern.pattern, pattern.regex_flags)
        matches = []
        sample_values = []
        
        for value in column_data.dropna():
            if regex.search(str(value)):
                matches.append(value)
                if len(sample_values) < 5:  # Keep up to 5 samples
                    sample_values.append(str(value))
        
        total_matches = len(matches)
        percentage_matches = (total_matches / len(column_data)) * 100 if len(column_data) > 0 else 0
        
        return {
            'total_matches': total_matches,
            'percentage_matches': percentage_matches,
            'sample_values': sample_values
        }
    
    def _calculate_confidence(self, matches: int, total_values: int, base_threshold: float) -> float:
        """Calculate confidence score for PII detection."""
        if total_values == 0:
            return 0.0
        
        match_percentage = matches / total_values
        
        # Adjust confidence based on match percentage
        if match_percentage >= 0.9:
            return min(1.0, base_threshold + 0.2)
        elif match_percentage >= 0.5:
            return base_threshold
        elif match_percentage >= 0.1:
            return max(0.0, base_threshold - 0.2)
        else:
            return max(0.0, base_threshold - 0.4)
    
    @handle_exceptions
    async def apply_privacy_protection(
        self,
        data: Union[pd.DataFrame, Dict[str, List], List[Dict]],
        detection_results: List[PIIDetectionResult],
        policy_name: Optional[str] = None,
        custom_rules: Optional[Dict[PIIType, MaskingRule]] = None
    ) -> Tuple[Union[pd.DataFrame, Dict[str, List], List[Dict]], PrivacyReport]:
        """
        Apply privacy protection (masking) to detected PII.
        
        Args:
            data: Original data
            detection_results: PII detection results
            policy_name: Privacy policy to use
            custom_rules: Custom masking rules
            
        Returns:
            Tuple of (protected_data, privacy_report)
        """
        logger.info("Applying privacy protection")
        start_time = datetime.utcnow()
        
        # Convert data to DataFrame for consistent processing
        original_type = type(data)
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        protected_df = df.copy()
        columns_masked = 0
        privacy_violations = []
        
        # Get masking rules
        rules_to_use = self._get_masking_rules(policy_name, custom_rules)
        
        # Apply masking to detected PII columns
        for detection in detection_results:
            if detection.pii_type in rules_to_use:
                rule = rules_to_use[detection.pii_type]
                
                try:
                    protected_df[detection.column_name] = await self._apply_masking_rule(
                        protected_df[detection.column_name],
                        rule,
                        detection.pii_type
                    )
                    columns_masked += 1
                    logger.debug(f"Applied masking to column: {detection.column_name}")
                    
                except Exception as e:
                    privacy_violations.append(
                        f"Failed to mask column {detection.column_name}: {str(e)}"
                    )
                    logger.error(f"Masking failed for {detection.column_name}: {e}")
        
        # Convert back to original type
        if original_type == dict:
            protected_data = protected_df.to_dict('list')
        elif original_type == list:
            protected_data = protected_df.to_dict('records')
        else:
            protected_data = protected_df
        
        # Generate privacy report
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        report = PrivacyReport(
            dataset_id=str(hash(str(data))),
            policy_applied=policy_name or "default",
            pii_detections=detection_results,
            sensitive_columns=[d.column_name for d in detection_results],
            rows_processed=len(df),
            columns_masked=columns_masked,
            privacy_violations=privacy_violations,
            processing_time_seconds=processing_time,
            detection_accuracy=self._calculate_detection_accuracy(detection_results)
        )
        
        # Check compliance
        await self._check_compliance(report, policy_name)
        
        logger.info(f"Privacy protection completed. Masked {columns_masked} columns")
        return protected_data, report
    
    def _get_masking_rules(
        self,
        policy_name: Optional[str],
        custom_rules: Optional[Dict[PIIType, MaskingRule]]
    ) -> Dict[PIIType, MaskingRule]:
        """Get masking rules for the specified policy."""
        rules = self.masking_rules.copy()
        
        if policy_name and policy_name in self.privacy_policies:
            policy = self.privacy_policies[policy_name]
            if policy.masking_rules:
                # Use policy-specific rules
                policy_rules = {rule.pii_type: rule for rule in policy.masking_rules}
                rules.update(policy_rules)
        
        # Add custom rules
        if custom_rules:
            rules.update(custom_rules)
        
        return rules
    
    async def _apply_masking_rule(
        self,
        column_data: pd.Series,
        rule: MaskingRule,
        pii_type: PIIType
    ) -> pd.Series:
        """Apply a specific masking rule to column data."""
        masked_data = column_data.copy()
        
        for idx, value in enumerate(column_data):
            if pd.isna(value):
                continue
                
            str_value = str(value)
            
            if rule.strategy == MaskingStrategy.REDACTION:
                masked_data.iloc[idx] = "[REDACTED]"
                
            elif rule.strategy == MaskingStrategy.HASH:
                hash_value = hashlib.sha256(str_value.encode()).hexdigest()[:16]
                masked_data.iloc[idx] = f"HASH_{hash_value}"
                
            elif rule.strategy == MaskingStrategy.TOKENIZATION:
                token = self._get_or_create_token(str_value, rule.token_prefix)
                masked_data.iloc[idx] = token
                
            elif rule.strategy == MaskingStrategy.PARTIAL_MASK:
                masked_value = self._apply_partial_mask(str_value, rule)
                masked_data.iloc[idx] = masked_value
                
            elif rule.strategy == MaskingStrategy.PSEUDONYMIZATION:
                pseudo_value = await self._generate_pseudonym(str_value, pii_type)
                masked_data.iloc[idx] = pseudo_value
                
            elif rule.strategy == MaskingStrategy.ENCRYPTION:
                encrypted_value = await self._encrypt_value(str_value, rule)
                masked_data.iloc[idx] = encrypted_value
                
            elif rule.strategy == MaskingStrategy.RANDOMIZATION:
                random_value = self._generate_random_value(str_value, pii_type)
                masked_data.iloc[idx] = random_value
                
            elif rule.strategy == MaskingStrategy.NULL_OUT:
                masked_data.iloc[idx] = None
        
        return masked_data
    
    def _apply_partial_mask(self, value: str, rule: MaskingRule) -> str:
        """Apply partial masking to a value."""
        if len(value) <= 4:
            return rule.replacement_char * len(value)
        
        # Show first 2 and last 2 characters for most PII types
        if rule.pii_type == PIIType.EMAIL:
            at_pos = value.find('@')
            if at_pos > 0:
                username = value[:at_pos]
                domain = value[at_pos:]
                if len(username) > 2:
                    masked_username = username[0] + rule.replacement_char * (len(username) - 2) + username[-1]
                else:
                    masked_username = rule.replacement_char * len(username)
                return masked_username + domain
        
        # Default partial masking
        visible_chars = 2
        masked_middle = rule.replacement_char * (len(value) - 2 * visible_chars)
        return value[:visible_chars] + masked_middle + value[-visible_chars:]
    
    def _get_or_create_token(self, value: str, prefix: str) -> str:
        """Get existing token or create new one for a value."""
        if value in self.token_mappings:
            return self.token_mappings[value]
        
        token = f"{prefix}{secrets.token_hex(8).upper()}"
        self.token_mappings[value] = token
        return token
    
    async def _generate_pseudonym(self, value: str, pii_type: PIIType) -> str:
        """Generate a pseudonym for a value."""
        # Simple pseudonym generation - in production, use more sophisticated methods
        hash_value = hashlib.md5(value.encode()).hexdigest()[:8]
        
        if pii_type == PIIType.NAME:
            return f"Person_{hash_value}"
        elif pii_type == PIIType.EMAIL:
            return f"user_{hash_value}@example.com"
        elif pii_type == PIIType.PHONE:
            return f"555-{hash_value[:3]}-{hash_value[3:7]}"
        else:
            return f"PSEUDO_{hash_value}"
    
    async def _encrypt_value(self, value: str, rule: MaskingRule) -> str:
        """Encrypt a value (simplified implementation)."""
        # In production, use proper encryption libraries
        key = rule.encryption_key or "default_key"
        encoded = value.encode()
        # Simplified encryption for demo
        encrypted = bytes([b ^ ord(key[i % len(key)]) for i, b in enumerate(encoded)])
        return f"ENC_{encrypted.hex()}"
    
    def _generate_random_value(self, value: str, pii_type: PIIType) -> str:
        """Generate a random value of the same type."""
        import random
        
        if pii_type == PIIType.SSN:
            return f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
        elif pii_type == PIIType.PHONE:
            return f"{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}"
        elif pii_type == PIIType.EMAIL:
            domains = ["example.com", "test.org", "demo.net"]
            return f"user{random.randint(1000, 9999)}@{random.choice(domains)}"
        else:
            return f"RANDOM_{secrets.token_hex(4).upper()}"
    
    def _calculate_detection_accuracy(self, detection_results: List[PIIDetectionResult]) -> float:
        """Calculate overall detection accuracy."""
        if not detection_results:
            return 0.0
        
        total_confidence = sum(result.confidence_score for result in detection_results)
        return total_confidence / len(detection_results)
    
    async def _check_compliance(self, report: PrivacyReport, policy_name: Optional[str]) -> None:
        """Check compliance with privacy standards."""
        if not policy_name or policy_name not in self.privacy_policies:
            return
        
        policy = self.privacy_policies[policy_name]
        
        # Check if all required PII is properly protected
        for standard in policy.compliance_standards:
            if self._check_standard_compliance(report, standard, policy):
                report.compliance_standards_met.append(standard)
            else:
                report.compliance_issues.append(f"Non-compliance with {standard.value}")
    
    def _check_standard_compliance(
        self,
        report: PrivacyReport,
        standard: ComplianceStandard,
        policy: PrivacyPolicy
    ) -> bool:
        """Check compliance with a specific standard."""
        if standard == ComplianceStandard.GDPR:
            # GDPR requires protection of all personal data
            return len(report.privacy_violations) == 0 and report.columns_masked > 0
        
        elif standard == ComplianceStandard.HIPAA:
            # HIPAA requires protection of health information
            health_pii_types = {PIIType.SSN, PIIType.NAME, PIIType.DATE_OF_BIRTH, PIIType.MEDICAL_ID}
            detected_health_pii = {d.pii_type for d in report.pii_detections}
            return len(detected_health_pii.intersection(health_pii_types)) == 0 or report.columns_masked > 0
        
        elif standard == ComplianceStandard.PCI_DSS:
            # PCI DSS requires protection of payment card information
            payment_pii_types = {PIIType.CREDIT_CARD}
            detected_payment_pii = {d.pii_type for d in report.pii_detections}
            return len(detected_payment_pii.intersection(payment_pii_types)) == 0 or report.columns_masked > 0
        
        return True  # Default to compliant for unknown standards
    
    @handle_exceptions
    async def create_privacy_policy(
        self,
        name: str,
        compliance_standards: List[ComplianceStandard],
        **kwargs
    ) -> PrivacyPolicy:
        """Create a new privacy policy."""
        policy = PrivacyPolicy(
            name=name,
            compliance_standards=compliance_standards,
            **kwargs
        )
        
        self.privacy_policies[name] = policy
        logger.info(f"Created privacy policy: {name}")
        return policy
    
    @handle_exceptions
    async def add_custom_pii_pattern(
        self,
        pii_type: PIIType,
        pattern: str,
        description: str = "",
        confidence_threshold: float = 0.8
    ) -> PIIPattern:
        """Add a custom PII pattern."""
        custom_pattern = PIIPattern(
            pii_type=pii_type,
            pattern=pattern,
            description=description,
            confidence_threshold=confidence_threshold
        )
        
        if pii_type not in self.pii_patterns:
            self.pii_patterns[pii_type] = []
        
        self.pii_patterns[pii_type].append(custom_pattern)
        logger.info(f"Added custom PII pattern for {pii_type.value}")
        return custom_pattern
    
    @handle_exceptions
    async def get_privacy_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive privacy protection dashboard."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "privacy_policies": {
                name: {
                    "compliance_standards": [s.value for s in policy.compliance_standards],
                    "pii_patterns_count": len(policy.pii_patterns),
                    "masking_rules_count": len(policy.masking_rules),
                    "retention_period_days": policy.retention_period_days,
                    "created_at": policy.created_at.isoformat()
                }
                for name, policy in self.privacy_policies.items()
            },
            "pii_detection_capabilities": {
                pii_type.value: len(patterns)
                for pii_type, patterns in self.pii_patterns.items()
            },
            "masking_strategies": {
                pii_type.value: rule.strategy.value
                for pii_type, rule in self.masking_rules.items()
            },
            "token_mappings_count": len(self.token_mappings),
            "supported_compliance_standards": [s.value for s in ComplianceStandard],
            "supported_pii_types": [t.value for t in PIIType],
            "supported_masking_strategies": [s.value for s in MaskingStrategy]
        }