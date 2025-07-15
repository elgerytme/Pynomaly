"""Consent Management Service for data quality operations."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from ...domain.entities.security_entity import ConsentRecord, ComplianceFramework


class ConsentManagementService:
    """Service for managing data processing consent."""
    
    def __init__(self):
        """Initialize consent management service."""
        self.consent_records: Dict[UUID, ConsentRecord] = {}
        self.consent_policies: Dict[str, Dict[str, Any]] = {}
        self.consent_audit_log: List[Dict[str, Any]] = []
    
    def record_consent(self, subject_id: str, purpose: str, legal_basis: str, 
                      consent_given: bool, metadata: Optional[Dict[str, Any]] = None) -> ConsentRecord:
        """
        Record consent for data processing.
        
        Args:
            subject_id: Data subject identifier
            purpose: Purpose of data processing
            legal_basis: Legal basis for processing
            consent_given: Whether consent was given
            metadata: Additional metadata
            
        Returns:
            Consent record
        """
        consent_record = ConsentRecord(
            subject_id=subject_id,
            purpose=purpose,
            legal_basis=legal_basis,
            consent_given=consent_given,
            consent_date=datetime.now(),
            metadata=metadata or {}
        )
        
        self.consent_records[consent_record.consent_id] = consent_record
        
        # Log consent action
        self._log_consent_action(
            action="consent_recorded",
            subject_id=subject_id,
            consent_id=consent_record.consent_id,
            details={
                'purpose': purpose,
                'legal_basis': legal_basis,
                'consent_given': consent_given
            }
        )
        
        return consent_record
    
    def withdraw_consent(self, consent_id: UUID, withdrawal_reason: Optional[str] = None) -> bool:
        """
        Withdraw consent for data processing.
        
        Args:
            consent_id: Consent record ID
            withdrawal_reason: Reason for withdrawal
            
        Returns:
            Whether withdrawal was successful
        """
        if consent_id not in self.consent_records:
            return False
        
        consent_record = self.consent_records[consent_id]
        consent_record.withdrawal_date = datetime.now()
        consent_record.consent_given = False
        
        if withdrawal_reason:
            consent_record.metadata['withdrawal_reason'] = withdrawal_reason
        
        # Log withdrawal action
        self._log_consent_action(
            action="consent_withdrawn",
            subject_id=consent_record.subject_id,
            consent_id=consent_id,
            details={
                'withdrawal_reason': withdrawal_reason,
                'withdrawal_date': consent_record.withdrawal_date.isoformat()
            }
        )
        
        return True
    
    def check_consent(self, subject_id: str, purpose: str) -> bool:
        """
        Check if valid consent exists for processing.
        
        Args:
            subject_id: Data subject identifier
            purpose: Purpose of processing
            
        Returns:
            Whether valid consent exists
        """
        for consent_record in self.consent_records.values():
            if (consent_record.subject_id == subject_id and 
                consent_record.purpose == purpose and 
                consent_record.is_valid()):
                return True
        
        return False
    
    def get_consent_status(self, subject_id: str) -> Dict[str, Any]:
        """
        Get consent status for a data subject.
        
        Args:
            subject_id: Data subject identifier
            
        Returns:
            Consent status summary
        """
        subject_consents = [
            consent for consent in self.consent_records.values()
            if consent.subject_id == subject_id
        ]
        
        active_consents = [
            consent for consent in subject_consents
            if consent.is_valid()
        ]
        
        return {
            'subject_id': subject_id,
            'total_consents': len(subject_consents),
            'active_consents': len(active_consents),
            'withdrawn_consents': len([c for c in subject_consents if c.withdrawal_date]),
            'expired_consents': len([c for c in subject_consents if c.expiry_date and datetime.now() > c.expiry_date]),
            'consent_purposes': list(set(c.purpose for c in active_consents)),
            'last_consent_date': max((c.consent_date for c in subject_consents), default=None)
        }
    
    def update_consent_expiry(self, consent_id: UUID, new_expiry: datetime) -> bool:
        """
        Update consent expiry date.
        
        Args:
            consent_id: Consent record ID
            new_expiry: New expiry date
            
        Returns:
            Whether update was successful
        """
        if consent_id not in self.consent_records:
            return False
        
        old_expiry = self.consent_records[consent_id].expiry_date
        self.consent_records[consent_id].expiry_date = new_expiry
        
        # Log expiry update
        self._log_consent_action(
            action="consent_expiry_updated",
            subject_id=self.consent_records[consent_id].subject_id,
            consent_id=consent_id,
            details={
                'old_expiry': old_expiry.isoformat() if old_expiry else None,
                'new_expiry': new_expiry.isoformat()
            }
        )
        
        return True
    
    def create_consent_policy(self, policy_name: str, policy_config: Dict[str, Any]) -> None:
        """
        Create consent policy.
        
        Args:
            policy_name: Policy name
            policy_config: Policy configuration
        """
        self.consent_policies[policy_name] = {
            'config': policy_config,
            'created_at': datetime.now(),
            'active': True
        }
        
        # Log policy creation
        self._log_consent_action(
            action="policy_created",
            subject_id=None,
            consent_id=None,
            details={
                'policy_name': policy_name,
                'config': policy_config
            }
        )
    
    def validate_consent_requirements(self, processing_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate consent requirements for processing request.
        
        Args:
            processing_request: Processing request details
            
        Returns:
            Validation result
        """
        subject_id = processing_request.get('subject_id')
        purpose = processing_request.get('purpose')
        legal_basis = processing_request.get('legal_basis')
        
        validation_result = {
            'valid': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check if consent exists
        if not self.check_consent(subject_id, purpose):
            validation_result['valid'] = False
            validation_result['issues'].append('No valid consent found')
            validation_result['recommendations'].append('Obtain consent before processing')
        
        # Check legal basis
        if legal_basis not in ['consent', 'contract', 'legal_obligation', 'vital_interests', 'public_task', 'legitimate_interests']:
            validation_result['valid'] = False
            validation_result['issues'].append('Invalid legal basis')
        
        # Check compliance framework requirements
        frameworks = processing_request.get('compliance_frameworks', [])
        for framework in frameworks:
            framework_validation = self._validate_framework_requirements(framework, processing_request)
            if not framework_validation['valid']:
                validation_result['valid'] = False
                validation_result['issues'].extend(framework_validation['issues'])
                validation_result['recommendations'].extend(framework_validation['recommendations'])
        
        return validation_result
    
    def generate_consent_report(self, report_type: str = 'summary') -> Dict[str, Any]:
        """
        Generate consent management report.
        
        Args:
            report_type: Type of report to generate
            
        Returns:
            Consent report
        """
        if report_type == 'summary':
            return self._generate_summary_report()
        elif report_type == 'detailed':
            return self._generate_detailed_report()
        elif report_type == 'compliance':
            return self._generate_compliance_report()
        else:
            return {'error': 'Unknown report type'}
    
    def cleanup_expired_consents(self) -> Dict[str, Any]:
        """
        Clean up expired consent records.
        
        Returns:
            Cleanup summary
        """
        expired_count = 0
        current_time = datetime.now()
        
        for consent_id, consent_record in list(self.consent_records.items()):
            if consent_record.expiry_date and current_time > consent_record.expiry_date:
                # Mark as expired but don't delete (for audit purposes)
                consent_record.metadata['expired'] = True
                expired_count += 1
        
        return {
            'expired_consents': expired_count,
            'cleanup_date': current_time.isoformat()
        }
    
    def _log_consent_action(self, action: str, subject_id: Optional[str], 
                           consent_id: Optional[UUID], details: Dict[str, Any]) -> None:
        """Log consent management action."""
        log_entry = {
            'action': action,
            'subject_id': subject_id,
            'consent_id': str(consent_id) if consent_id else None,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.consent_audit_log.append(log_entry)
    
    def _validate_framework_requirements(self, framework: ComplianceFramework, 
                                       request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compliance framework requirements."""
        validation_result = {
            'valid': True,
            'issues': [],
            'recommendations': []
        }
        
        if framework == ComplianceFramework.GDPR:
            # GDPR-specific validation
            if not request.get('data_subject_rights_notice'):
                validation_result['issues'].append('Missing data subject rights notice')
                validation_result['recommendations'].append('Provide data subject rights information')
            
            if not request.get('data_retention_period'):
                validation_result['issues'].append('Missing data retention period')
                validation_result['recommendations'].append('Specify data retention period')
        
        elif framework == ComplianceFramework.CCPA:
            # CCPA-specific validation
            if not request.get('right_to_know_notice'):
                validation_result['issues'].append('Missing right to know notice')
                validation_result['recommendations'].append('Provide right to know information')
        
        elif framework == ComplianceFramework.HIPAA:
            # HIPAA-specific validation
            if not request.get('minimum_necessary_standard'):
                validation_result['issues'].append('Missing minimum necessary standard compliance')
                validation_result['recommendations'].append('Ensure minimum necessary standard compliance')
        
        if validation_result['issues']:
            validation_result['valid'] = False
        
        return validation_result
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary consent report."""
        total_consents = len(self.consent_records)
        active_consents = sum(1 for c in self.consent_records.values() if c.is_valid())
        withdrawn_consents = sum(1 for c in self.consent_records.values() if c.withdrawal_date)
        
        return {
            'report_type': 'summary',
            'generated_at': datetime.now().isoformat(),
            'total_consents': total_consents,
            'active_consents': active_consents,
            'withdrawn_consents': withdrawn_consents,
            'consent_rate': active_consents / max(1, total_consents),
            'unique_subjects': len(set(c.subject_id for c in self.consent_records.values())),
            'unique_purposes': len(set(c.purpose for c in self.consent_records.values()))
        }
    
    def _generate_detailed_report(self) -> Dict[str, Any]:
        """Generate detailed consent report."""
        consent_by_purpose = {}
        consent_by_legal_basis = {}
        
        for consent_record in self.consent_records.values():
            # Group by purpose
            purpose = consent_record.purpose
            if purpose not in consent_by_purpose:
                consent_by_purpose[purpose] = {'total': 0, 'active': 0, 'withdrawn': 0}
            
            consent_by_purpose[purpose]['total'] += 1
            if consent_record.is_valid():
                consent_by_purpose[purpose]['active'] += 1
            if consent_record.withdrawal_date:
                consent_by_purpose[purpose]['withdrawn'] += 1
            
            # Group by legal basis
            legal_basis = consent_record.legal_basis
            if legal_basis not in consent_by_legal_basis:
                consent_by_legal_basis[legal_basis] = {'total': 0, 'active': 0}
            
            consent_by_legal_basis[legal_basis]['total'] += 1
            if consent_record.is_valid():
                consent_by_legal_basis[legal_basis]['active'] += 1
        
        return {
            'report_type': 'detailed',
            'generated_at': datetime.now().isoformat(),
            'consent_by_purpose': consent_by_purpose,
            'consent_by_legal_basis': consent_by_legal_basis,
            'recent_actions': self.consent_audit_log[-10:]  # Last 10 actions
        }
    
    def _generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance-focused consent report."""
        gdpr_consents = []
        ccpa_consents = []
        hipaa_consents = []
        
        for consent_record in self.consent_records.values():
            frameworks = consent_record.metadata.get('compliance_frameworks', [])
            
            if 'GDPR' in frameworks:
                gdpr_consents.append(consent_record)
            if 'CCPA' in frameworks:
                ccpa_consents.append(consent_record)
            if 'HIPAA' in frameworks:
                hipaa_consents.append(consent_record)
        
        return {
            'report_type': 'compliance',
            'generated_at': datetime.now().isoformat(),
            'gdpr_compliance': {
                'total_consents': len(gdpr_consents),
                'active_consents': sum(1 for c in gdpr_consents if c.is_valid()),
                'withdrawal_rate': sum(1 for c in gdpr_consents if c.withdrawal_date) / max(1, len(gdpr_consents))
            },
            'ccpa_compliance': {
                'total_consents': len(ccpa_consents),
                'active_consents': sum(1 for c in ccpa_consents if c.is_valid()),
                'withdrawal_rate': sum(1 for c in ccpa_consents if c.withdrawal_date) / max(1, len(ccpa_consents))
            },
            'hipaa_compliance': {
                'total_consents': len(hipaa_consents),
                'active_consents': sum(1 for c in hipaa_consents if c.is_valid()),
                'withdrawal_rate': sum(1 for c in hipaa_consents if c.withdrawal_date) / max(1, len(hipaa_consents))
            }
        }


class ConsentOrchestrationService:
    """Service for orchestrating consent across quality operations."""
    
    def __init__(self, consent_service: ConsentManagementService):
        """
        Initialize consent orchestration service.
        
        Args:
            consent_service: Consent management service
        """
        self.consent_service = consent_service
        self.operation_consents: Dict[UUID, Set[UUID]] = {}
    
    def validate_operation_consent(self, operation_id: UUID, 
                                 processing_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate consent for quality operation.
        
        Args:
            operation_id: Operation identifier
            processing_details: Details of data processing
            
        Returns:
            Validation result
        """
        validation_result = {
            'operation_id': str(operation_id),
            'valid': True,
            'consent_checks': [],
            'issues': [],
            'recommendations': []
        }
        
        # Check consent for each data subject
        subjects = processing_details.get('data_subjects', [])
        purpose = processing_details.get('purpose', '')
        
        for subject_id in subjects:
            consent_valid = self.consent_service.check_consent(subject_id, purpose)
            
            validation_result['consent_checks'].append({
                'subject_id': subject_id,
                'purpose': purpose,
                'consent_valid': consent_valid
            })
            
            if not consent_valid:
                validation_result['valid'] = False
                validation_result['issues'].append(f'No valid consent for subject {subject_id}')
                validation_result['recommendations'].append(f'Obtain consent for subject {subject_id}')
        
        return validation_result
    
    def register_operation_consent(self, operation_id: UUID, consent_ids: List[UUID]) -> None:
        """
        Register consent records for operation.
        
        Args:
            operation_id: Operation identifier
            consent_ids: List of consent record IDs
        """
        self.operation_consents[operation_id] = set(consent_ids)
    
    def get_operation_consent_status(self, operation_id: UUID) -> Dict[str, Any]:
        """
        Get consent status for operation.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            Consent status
        """
        if operation_id not in self.operation_consents:
            return {'error': 'Operation not found'}
        
        consent_ids = self.operation_consents[operation_id]
        valid_consents = 0
        expired_consents = 0
        withdrawn_consents = 0
        
        for consent_id in consent_ids:
            consent_record = self.consent_service.consent_records.get(consent_id)
            if consent_record:
                if consent_record.is_valid():
                    valid_consents += 1
                elif consent_record.withdrawal_date:
                    withdrawn_consents += 1
                elif consent_record.expiry_date and datetime.now() > consent_record.expiry_date:
                    expired_consents += 1
        
        return {
            'operation_id': str(operation_id),
            'total_consents': len(consent_ids),
            'valid_consents': valid_consents,
            'expired_consents': expired_consents,
            'withdrawn_consents': withdrawn_consents,
            'consent_valid': valid_consents == len(consent_ids)
        }