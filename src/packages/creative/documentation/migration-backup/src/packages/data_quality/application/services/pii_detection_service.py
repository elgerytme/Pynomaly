"""PII Detection and Masking Service for data quality operations."""

import hashlib
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from ...domain.entities.security_entity import PIIDetectionResult, PIIType
from ...domain.entities.quality_profile import QualityProfile


class PIIDetectionService:
    """Service for detecting and masking personally identifiable information."""
    
    def __init__(self):
        """Initialize PII detection service."""
        self._detection_patterns = self._initialize_patterns()
        self._masking_strategies = self._initialize_masking_strategies()
    
    def detect_pii(self, data: Dict[str, Any], profile: Optional[QualityProfile] = None) -> List[PIIDetectionResult]:
        """
        Detect PII in data using pattern matching and ML techniques.
        
        Args:
            data: Data to analyze for PII
            profile: Optional quality profile with PII configuration
            
        Returns:
            List of PII detection results
        """
        results = []
        
        # Analyze each field in the data
        for field_name, field_value in data.items():
            if field_value is None:
                continue
                
            # Convert to string for pattern matching
            str_value = str(field_value)
            
            # Check each PII type
            for pii_type, pattern_info in self._detection_patterns.items():
                matches = self._check_pattern(str_value, field_name, pattern_info)
                for match in matches:
                    result = PIIDetectionResult(
                        pii_type=pii_type,
                        confidence=match['confidence'],
                        location={'field': field_name, 'value': str_value},
                        value_sample=match['sample'],
                        detection_method=match['method'],
                        metadata={'pattern': match.get('pattern', '')}
                    )
                    results.append(result)
        
        # Apply ML-based detection if profile is provided
        if profile:
            ml_results = self._ml_detection(data, profile)
            results.extend(ml_results)
        
        return results
    
    def mask_pii(self, data: Dict[str, Any], pii_results: List[PIIDetectionResult], 
                 masking_strategy: str = "redaction") -> Dict[str, Any]:
        """
        Mask PII in data based on detection results.
        
        Args:
            data: Original data
            pii_results: PII detection results
            masking_strategy: Masking strategy to use
            
        Returns:
            Data with PII masked
        """
        masked_data = data.copy()
        
        for result in pii_results:
            field_name = result.location['field']
            if field_name in masked_data:
                masked_value = self._apply_masking(
                    masked_data[field_name], 
                    result.pii_type, 
                    masking_strategy
                )
                masked_data[field_name] = masked_value
                result.masked_value = str(masked_value)
        
        return masked_data
    
    def anonymize_data(self, data: Dict[str, Any], anonymization_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize data for quality testing purposes.
        
        Args:
            data: Original data
            anonymization_config: Configuration for anonymization
            
        Returns:
            Anonymized data
        """
        anonymized_data = data.copy()
        
        # Apply k-anonymity
        if anonymization_config.get('k_anonymity'):
            anonymized_data = self._apply_k_anonymity(anonymized_data, anonymization_config['k_anonymity'])
        
        # Apply l-diversity
        if anonymization_config.get('l_diversity'):
            anonymized_data = self._apply_l_diversity(anonymized_data, anonymization_config['l_diversity'])
        
        # Apply t-closeness
        if anonymization_config.get('t_closeness'):
            anonymized_data = self._apply_t_closeness(anonymized_data, anonymization_config['t_closeness'])
        
        return anonymized_data
    
    def pseudonymize_data(self, data: Dict[str, Any], pseudonymization_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pseudonymize data while maintaining referential integrity.
        
        Args:
            data: Original data
            pseudonymization_config: Configuration for pseudonymization
            
        Returns:
            Pseudonymized data
        """
        pseudonymized_data = data.copy()
        
        # Apply consistent pseudonymization
        for field_name, config in pseudonymization_config.items():
            if field_name in pseudonymized_data:
                pseudonymized_data[field_name] = self._generate_pseudonym(
                    pseudonymized_data[field_name], 
                    config
                )
        
        return pseudonymized_data
    
    def _initialize_patterns(self) -> Dict[PIIType, Dict[str, Any]]:
        """Initialize PII detection patterns."""
        return {
            PIIType.EMAIL: {
                'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'field_keywords': ['email', 'mail', 'e-mail'],
                'confidence_base': 0.9
            },
            PIIType.PHONE: {
                'pattern': r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                'field_keywords': ['phone', 'telephone', 'mobile', 'tel'],
                'confidence_base': 0.8
            },
            PIIType.SSN: {
                'pattern': r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
                'field_keywords': ['ssn', 'social', 'security'],
                'confidence_base': 0.95
            },
            PIIType.CREDIT_CARD: {
                'pattern': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                'field_keywords': ['card', 'credit', 'cc', 'payment'],
                'confidence_base': 0.9
            },
            PIIType.NAME: {
                'pattern': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                'field_keywords': ['name', 'firstname', 'lastname', 'fullname'],
                'confidence_base': 0.7
            },
            PIIType.ADDRESS: {
                'pattern': r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)',
                'field_keywords': ['address', 'street', 'location'],
                'confidence_base': 0.8
            },
            PIIType.DATE_OF_BIRTH: {
                'pattern': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b',
                'field_keywords': ['birth', 'dob', 'birthday'],
                'confidence_base': 0.8
            }
        }
    
    def _initialize_masking_strategies(self) -> Dict[str, callable]:
        """Initialize masking strategies."""
        return {
            'redaction': self._redact_value,
            'hashing': self._hash_value,
            'partial_masking': self._partial_mask_value,
            'synthetic': self._synthetic_value,
            'tokenization': self._tokenize_value
        }
    
    def _check_pattern(self, value: str, field_name: str, pattern_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if value matches PII pattern."""
        matches = []
        
        # Pattern matching
        pattern_matches = re.findall(pattern_info['pattern'], value)
        if pattern_matches:
            confidence = pattern_info['confidence_base']
            
            # Boost confidence if field name contains keywords
            for keyword in pattern_info['field_keywords']:
                if keyword.lower() in field_name.lower():
                    confidence = min(1.0, confidence + 0.1)
                    break
            
            matches.append({
                'confidence': confidence,
                'sample': pattern_matches[0] if pattern_matches else value[:20],
                'method': 'pattern_matching',
                'pattern': pattern_info['pattern']
            })
        
        return matches
    
    def _ml_detection(self, data: Dict[str, Any], profile: QualityProfile) -> List[PIIDetectionResult]:
        """ML-based PII detection using profile context."""
        results = []
        
        # Implement ML-based detection logic here
        # This would use trained models specific to the data domain
        
        return results
    
    def _apply_masking(self, value: Any, pii_type: PIIType, strategy: str) -> Any:
        """Apply masking strategy to value."""
        if strategy in self._masking_strategies:
            return self._masking_strategies[strategy](value, pii_type)
        return value
    
    def _redact_value(self, value: Any, pii_type: PIIType) -> str:
        """Redact value completely."""
        type_labels = {
            PIIType.EMAIL: "[EMAIL]",
            PIIType.PHONE: "[PHONE]",
            PIIType.SSN: "[SSN]",
            PIIType.CREDIT_CARD: "[CREDIT_CARD]",
            PIIType.NAME: "[NAME]",
            PIIType.ADDRESS: "[ADDRESS]",
            PIIType.DATE_OF_BIRTH: "[DOB]"
        }
        return type_labels.get(pii_type, "[REDACTED]")
    
    def _hash_value(self, value: Any, pii_type: PIIType) -> str:
        """Hash value for consistency."""
        return hashlib.sha256(str(value).encode()).hexdigest()[:16]
    
    def _partial_mask_value(self, value: Any, pii_type: PIIType) -> str:
        """Partially mask value."""
        str_value = str(value)
        
        if pii_type == PIIType.EMAIL:
            parts = str_value.split('@')
            if len(parts) == 2:
                return f"{parts[0][:2]}***@{parts[1]}"
        elif pii_type == PIIType.PHONE:
            return f"***-***-{str_value[-4:]}"
        elif pii_type == PIIType.CREDIT_CARD:
            return f"****-****-****-{str_value[-4:]}"
        elif pii_type == PIIType.NAME:
            parts = str_value.split()
            if len(parts) >= 2:
                return f"{parts[0][0]}*** {parts[-1][0]}***"
        
        # Default partial masking
        if len(str_value) > 4:
            return f"{str_value[:2]}***{str_value[-2:]}"
        return "***"
    
    def _synthetic_value(self, value: Any, pii_type: PIIType) -> str:
        """Generate synthetic value."""
        synthetic_values = {
            PIIType.EMAIL: "user@example.com",
            PIIType.PHONE: "555-0123",
            PIIType.SSN: "123-45-6789",
            PIIType.CREDIT_CARD: "4111-1111-1111-1111",
            PIIType.NAME: "John Doe",
            PIIType.ADDRESS: "123 Main St",
            PIIType.DATE_OF_BIRTH: "01/01/1990"
        }
        return synthetic_values.get(pii_type, "SYNTHETIC_VALUE")
    
    def _tokenize_value(self, value: Any, pii_type: PIIType) -> str:
        """Generate reversible token."""
        # This would integrate with a tokenization service
        token_base = hashlib.md5(str(value).encode()).hexdigest()[:8]
        return f"TOKEN_{token_base.upper()}"
    
    def _apply_k_anonymity(self, data: Dict[str, Any], k: int) -> Dict[str, Any]:
        """Apply k-anonymity to data."""
        # Implement k-anonymity algorithm
        # This would group records and generalize quasi-identifiers
        return data
    
    def _apply_l_diversity(self, data: Dict[str, Any], l: int) -> Dict[str, Any]:
        """Apply l-diversity to data."""
        # Implement l-diversity algorithm
        # This would ensure each group has at least l diverse sensitive values
        return data
    
    def _apply_t_closeness(self, data: Dict[str, Any], t: float) -> Dict[str, Any]:
        """Apply t-closeness to data."""
        # Implement t-closeness algorithm
        # This would ensure sensitive attribute distribution is close to overall distribution
        return data
    
    def _generate_pseudonym(self, value: Any, config: Dict[str, Any]) -> str:
        """Generate consistent pseudonym."""
        # Use deterministic hashing with salt for consistency
        salt = config.get('salt', 'default_salt')
        hash_input = f"{value}{salt}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:config.get('length', 16)]


class PIIAuditService:
    """Service for auditing PII detection and masking operations."""
    
    def __init__(self):
        """Initialize PII audit service."""
        self._audit_log = []
    
    def log_pii_detection(self, operation_id: UUID, results: List[PIIDetectionResult], 
                         metadata: Dict[str, Any]) -> None:
        """Log PII detection operation."""
        audit_entry = {
            'operation_id': operation_id,
            'operation_type': 'pii_detection',
            'pii_types_found': [result.pii_type.value for result in results],
            'detection_count': len(results),
            'confidence_scores': [result.confidence for result in results],
            'metadata': metadata
        }
        self._audit_log.append(audit_entry)
    
    def log_pii_masking(self, operation_id: UUID, masking_strategy: str, 
                       fields_masked: List[str], metadata: Dict[str, Any]) -> None:
        """Log PII masking operation."""
        audit_entry = {
            'operation_id': operation_id,
            'operation_type': 'pii_masking',
            'masking_strategy': masking_strategy,
            'fields_masked': fields_masked,
            'metadata': metadata
        }
        self._audit_log.append(audit_entry)
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get PII audit log."""
        return self._audit_log.copy()