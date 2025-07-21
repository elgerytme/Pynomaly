"""Comprehensive security integration tests for data quality package."""

import unittest
from datetime import datetime, timedelta
from uuid import uuid4

from ..application.services.security_orchestration_service import SecurityOrchestrationService
from ..application.services.pii_detection_service import PIIDetectionService
from ..application.services.privacy_analytics_service import PrivacyPreservingAnalyticsService
from ..application.services.consent_management_service import ConsentManagementService
from ..application.services.privacy_impact_assessment_service import PrivacyImpactAssessmentService
from ..application.services.compliance_framework_service import ComplianceFrameworkService
from ..application.services.access_control_service import RoleBasedAccessControlService
from ..domain.entities.security_entity import PIIType, ComplianceFramework, PrivacyLevel


class SecurityIntegrationTests(unittest.TestCase):
    """Integration tests for security services."""
    
    def setUp(self):
        """Set up test environment."""
        self.security_orchestrator = SecurityOrchestrationService()
        self.pii_service = PIIDetectionService()
        self.privacy_analytics = PrivacyPreservingAnalyticsService()
        self.consent_service = ConsentManagementService()
        self.pia_service = PrivacyImpactAssessmentService()
        self.compliance_service = ComplianceFrameworkService()
        self.rbac_service = RoleBasedAccessControlService()
    
    def test_pii_detection_and_masking(self):
        """Test PII detection and masking functionality."""
        # Test data with PII
        test_data = {
            'name': 'John Doe',
            'email': 'john.doe@example.com',
            'phone': '555-123-4567',
            'ssn': '123-45-6789',
            'age': 30
        }
        
        # Detect PII
        pii_results = self.pii_service.detect_pii(test_data)
        
        # Verify PII detection
        self.assertGreater(len(pii_results), 0)
        detected_types = [result.pii_type for result in pii_results]
        self.assertIn(PIIType.EMAIL, detected_types)
        self.assertIn(PIIType.PHONE, detected_types)
        self.assertIn(PIIType.SSN, detected_types)
        
        # Test masking
        masked_data = self.pii_service.mask_pii(test_data, pii_results)
        
        # Verify masking
        self.assertNotEqual(masked_data['email'], test_data['email'])
        self.assertNotEqual(masked_data['phone'], test_data['phone'])
        self.assertNotEqual(masked_data['ssn'], test_data['ssn'])
        self.assertEqual(masked_data['age'], test_data['age'])  # Non-PII unchanged
    
    def test_privacy_preserving_analytics(self):
        """Test privacy-preserving analytics with differential privacy."""
        # Test data
        test_data = [
            {'id': 1, 'age': 25, 'salary': 50000},
            {'id': 2, 'age': 30, 'salary': 60000},
            {'id': 3, 'age': 35, 'salary': 70000}
        ]
        
        # Create mock quality profile
        class MockProfile:
            def __init__(self):
                self.metadata = {
                    'fields': [
                        {'name': 'age', 'type': 'integer'},
                        {'name': 'salary', 'type': 'integer'}
                    ]
                }
        
        profile = MockProfile()
        
        # Analyze with privacy preservation
        analytics_result = self.privacy_analytics.analyze_quality_metrics(test_data, profile)
        
        # Verify analytics
        self.assertIn('completeness', analytics_result)
        self.assertIn('accuracy', analytics_result)
        self.assertIn('consistency', analytics_result)
        
        # Generate privacy report
        privacy_report = self.privacy_analytics.generate_privacy_report(analytics_result)
        
        # Verify privacy report
        self.assertEqual(privacy_report['privacy_level'], PrivacyLevel.CONFIDENTIAL.value)
        self.assertIn('privacy_budget_used', privacy_report)
        self.assertIn('privacy_guarantees', privacy_report)
    
    def test_consent_management_workflow(self):
        """Test complete consent management workflow."""
        # Record consent
        consent_record = self.consent_service.record_consent(
            subject_id='user123',
            purpose='data_quality_analysis',
            legal_basis='consent',
            consent_given=True,
            metadata={'source': 'web_form'}
        )
        
        # Verify consent recording
        self.assertIsNotNone(consent_record.consent_id)
        self.assertTrue(consent_record.consent_given)
        self.assertEqual(consent_record.subject_id, 'user123')
        
        # Check consent
        consent_valid = self.consent_service.check_consent('user123', 'data_quality_analysis')
        self.assertTrue(consent_valid)
        
        # Withdraw consent
        withdrawal_success = self.consent_service.withdraw_consent(
            consent_record.consent_id,
            'user_request'
        )
        self.assertTrue(withdrawal_success)
        
        # Verify consent withdrawn
        consent_valid_after = self.consent_service.check_consent('user123', 'data_quality_analysis')
        self.assertFalse(consent_valid_after)
    
    def test_privacy_impact_assessment(self):
        """Test privacy impact assessment functionality."""
        # Create assessment
        assessment_data = {
            'data_categories': ['personal', 'financial'],
            'processing_purposes': ['quality_analysis', 'reporting'],
            'legal_basis': 'consent',
            'data_subjects': ['customers'],
            'compliance_frameworks': [ComplianceFramework.GDPR]
        }
        
        assessment = self.pia_service.create_assessment(
            operation_name='Data Quality Analysis',
            description='Automated data quality assessment',
            assessor='security_team',
            assessment_data=assessment_data
        )
        
        # Verify assessment creation
        self.assertIsNotNone(assessment.assessment_id)
        self.assertEqual(assessment.operation_name, 'Data Quality Analysis')
        self.assertIn('quality_analysis', assessment.processing_purposes)
        
        # Conduct detailed assessment
        assessment_criteria = {
            'data_collection_limited': True,
            'purposes_defined': True,
            'retention_periods_defined': True,
            'accuracy_measures': True,
            'encryption_implemented': True,
            'privacy_notices': True,
            'rights_mechanisms': True,
            'security_monitoring': True
        }
        
        assessment_results = self.pia_service.conduct_assessment(
            assessment.assessment_id,
            assessment_criteria
        )
        
        # Verify assessment results
        self.assertIn('overall_score', assessment_results)
        self.assertIn('recommendations', assessment_results)
        self.assertGreater(assessment_results['overall_score'], 0.0)
    
    def test_compliance_framework_integration(self):
        """Test compliance framework integration."""
        # Test GDPR compliance
        gdpr_data = {
            'legal_basis': True,
            'privacy_notice': True,
            'privacy_by_design': True,
            'security_measures': True,
            'high_risk_processing': False,
            'dpia_conducted': True
        }
        
        gdpr_result = self.compliance_service.check_gdpr_compliance(gdpr_data)
        
        # Verify GDPR compliance
        self.assertEqual(gdpr_result['framework'], 'GDPR')
        self.assertTrue(gdpr_result['compliant'])
        self.assertIn('requirements_met', gdpr_result)
        
        # Test HIPAA compliance
        hipaa_data = {
            'minimum_necessary': True,
            'access_controls': True,
            'audit_controls': True,
            'integrity_controls': True,
            'transmission_security': True,
            'breach_notification_procedures': True
        }
        
        hipaa_result = self.compliance_service.check_hipaa_compliance(hipaa_data)
        
        # Verify HIPAA compliance
        self.assertEqual(hipaa_result['framework'], 'HIPAA')
        self.assertTrue(hipaa_result['compliant'])
        self.assertIn('requirements_met', hipaa_result)
    
    def test_rbac_functionality(self):
        """Test role-based access control."""
        # Create custom role
        permissions = {'data.read', 'data.write', 'quality.analyze'}
        role = self.rbac_service.create_role(
            role_name='quality_analyst',
            description='Quality analyst role',
            permissions=permissions,
            created_by='admin'
        )
        
        # Verify role creation
        self.assertEqual(role.name, 'quality_analyst')
        self.assertEqual(role.permissions, permissions)
        
        # Assign role to user
        assignment_success = self.rbac_service.assign_role(
            user_id='analyst123',
            role_name='quality_analyst',
            assigned_by='admin'
        )
        self.assertTrue(assignment_success)
        
        # Check permissions
        has_read_permission = self.rbac_service.check_permission('analyst123', 'data.read')
        has_write_permission = self.rbac_service.check_permission('analyst123', 'data.write')
        has_admin_permission = self.rbac_service.check_permission('analyst123', 'system.admin')
        
        # Verify permissions
        self.assertTrue(has_read_permission)
        self.assertTrue(has_write_permission)
        self.assertFalse(has_admin_permission)
    
    def test_security_orchestration_workflow(self):
        """Test complete security orchestration workflow."""
        # Setup consent
        consent_record = self.security_orchestrator.consent_service.record_consent(
            subject_id='user123',
            purpose='data_processing',
            legal_basis='consent',
            consent_given=True
        )
        
        # Test data with PII
        test_data = {
            'id': 'record123',
            'name': 'John Doe',
            'email': 'john@example.com',
            'age': 30,
            'salary': 50000
        }
        
        # Process data securely
        operation_id = uuid4()
        result = self.security_orchestrator.secure_data_processing(
            operation_id=operation_id,
            data=test_data,
            user_id='user123',
            operation_type='data_processing'
        )
        
        # Verify secure processing
        self.assertTrue(result['success'])
        self.assertIn('data', result)
        self.assertIn('audit_id', result)
        self.assertIn('compliance_score', result)
        
        # Verify data was processed securely
        processed_data = result['data']
        self.assertNotEqual(processed_data['email'], test_data['email'])  # Should be encrypted
    
    def test_comprehensive_security_assessment(self):
        """Test comprehensive security assessment."""
        # Perform security assessment
        assessment = self.security_orchestrator.comprehensive_security_assessment('organization')
        
        # Verify assessment structure
        self.assertIn('overall_score', assessment)
        self.assertIn('assessments', assessment)
        self.assertIn('recommendations', assessment)
        
        # Verify framework assessments
        assessments = assessment['assessments']
        self.assertIn('GDPR', assessments)
        self.assertIn('HIPAA', assessments)
        self.assertIn('SOX', assessments)
        self.assertIn('CCPA', assessments)
        self.assertIn('security_controls', assessments)
        
        # Verify scores
        self.assertGreater(assessment['overall_score'], 0.0)
        self.assertLessEqual(assessment['overall_score'], 1.0)
    
    def test_threat_detection(self):
        """Test threat detection functionality."""
        threat_service = self.security_orchestrator.threat_detection
        
        # Test suspicious activity
        suspicious_activity = {
            'user_id': 'user123',
            'indicators': ['multiple_failed_logins', 'unusual_access_pattern'],
            'timestamp': datetime.now()
        }
        
        # Analyze activity
        threat_result = threat_service.analyze_activity('user123', suspicious_activity)
        
        # Verify threat detection
        if threat_result:
            self.assertIn('brute_force', threat_result.threat_type)
            self.assertIn('high', threat_result.severity)
            self.assertGreater(threat_result.confidence, 0.0)
    
    def test_audit_logging(self):
        """Test comprehensive audit logging."""
        audit_service = self.security_orchestrator.audit_service
        
        # Start audit
        operation_id = uuid4()
        audit_record = audit_service.start_audit(
            operation_id=operation_id,
            user_id='user123',
            action='data_access',
            resource_type='dataset',
            resource_id='dataset123'
        )
        
        # Verify audit start
        self.assertIsNotNone(audit_record.audit_id)
        self.assertEqual(audit_record.user_id, 'user123')
        self.assertEqual(audit_record.action, 'data_access')
        
        # Complete audit
        audit_service.complete_audit(
            audit_record.audit_id,
            'success',
            {'records_processed': 100}
        )
        
        # Query audit records
        audit_records = audit_service.query_audit_records({'user_id': 'user123'})
        
        # Verify audit records
        self.assertGreater(len(audit_records), 0)
        found_record = next((r for r in audit_records if r.audit_id == audit_record.audit_id), None)
        self.assertIsNotNone(found_record)
    
    def test_incident_response(self):
        """Test incident response functionality."""
        incident_service = self.security_orchestrator.incident_response
        
        # Create security incident
        incident_id = incident_service.handle_security_incident(
            incident_type='data_breach',
            severity='high',
            details={
                'affected_users': 100,
                'data_types': ['email', 'phone'],
                'discovery_time': datetime.now().isoformat()
            }
        )
        
        # Verify incident handling
        self.assertIsNotNone(incident_id)
        
        # Check incident was recorded
        incidents = incident_service.incidents
        incident = next((i for i in incidents if i['incident_id'] == incident_id), None)
        self.assertIsNotNone(incident)
        self.assertEqual(incident['type'], 'data_breach')
        self.assertEqual(incident['severity'], 'high')
        self.assertIn('response_actions', incident)
    
    def test_encryption_functionality(self):
        """Test data encryption functionality."""
        encryption_service = self.security_orchestrator.encryption_service
        
        # Test data encryption
        sensitive_data = {
            'name': 'John Doe',
            'email': 'john@example.com',
            'phone': '555-123-4567',
            'age': 30
        }
        
        encrypted_data = encryption_service.encrypt_data(sensitive_data)
        
        # Verify encryption
        self.assertNotEqual(encrypted_data['email'], sensitive_data['email'])
        self.assertNotEqual(encrypted_data['phone'], sensitive_data['phone'])
        self.assertEqual(encrypted_data['age'], sensitive_data['age'])  # Non-sensitive unchanged
        
        # Verify encrypted format
        self.assertTrue(encrypted_data['email'].startswith('encrypted_'))
        self.assertTrue(encrypted_data['phone'].startswith('encrypted_'))
    
    def test_session_management(self):
        """Test session management functionality."""
        session_service = self.security_orchestrator.session_service
        
        # Create session
        session_id = session_service.create_session('user123', '192.168.1.100')
        
        # Verify session creation
        self.assertIsNotNone(session_id)
        
        # Validate session
        validation_result = session_service.validate_session(session_id)
        
        # Verify session validation
        self.assertTrue(validation_result['valid'])
        self.assertEqual(validation_result['user_id'], 'user123')
        
        # Terminate session
        session_service.terminate_session(session_id)
        
        # Verify session termination
        validation_result_after = session_service.validate_session(session_id)
        self.assertFalse(validation_result_after['valid'])
    
    def test_authentication_with_mfa(self):
        """Test authentication with multi-factor authentication."""
        auth_service = self.security_orchestrator.authentication_service
        
        # Setup MFA
        mfa_secret = auth_service.setup_mfa('user123')
        self.assertIsNotNone(mfa_secret)
        
        # Test authentication without MFA
        auth_result = auth_service.authenticate_user('user123', 'password123')
        self.assertTrue(auth_result['success'])
        
        # Test authentication with MFA
        auth_result_mfa = auth_service.authenticate_user('user123', 'password123', '123456')
        self.assertTrue(auth_result_mfa['success'])
        
        # Test authentication with invalid MFA
        auth_result_invalid = auth_service.authenticate_user('user123', 'password123', 'invalid')
        self.assertFalse(auth_result_invalid['success'])


class SecurityPerformanceTests(unittest.TestCase):
    """Performance tests for security services."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.security_orchestrator = SecurityOrchestrationService()
    
    def test_pii_detection_performance(self):
        """Test PII detection performance with large dataset."""
        import time
        
        # Create large test dataset
        test_data = []
        for i in range(1000):
            test_data.append({
                'id': i,
                'name': f'User {i}',
                'email': f'user{i}@example.com',
                'phone': f'555-{i:04d}',
                'age': 25 + (i % 40)
            })
        
        # Measure PII detection time
        start_time = time.time()
        
        for record in test_data:
            pii_results = self.security_orchestrator.pii_service.detect_pii(record)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify performance (should process 1000 records in reasonable time)
        self.assertLess(processing_time, 10.0)  # Less than 10 seconds
        
        # Calculate throughput
        throughput = len(test_data) / processing_time
        self.assertGreater(throughput, 100)  # At least 100 records per second
    
    def test_encryption_performance(self):
        """Test encryption performance."""
        import time
        
        encryption_service = self.security_orchestrator.encryption_service
        
        # Test data
        test_data = {
            'email': 'test@example.com',
            'phone': '555-1234',
            'ssn': '123-45-6789'
        }
        
        # Measure encryption time
        start_time = time.time()
        
        for _ in range(1000):
            encrypted_data = encryption_service.encrypt_data(test_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify performance
        self.assertLess(processing_time, 5.0)  # Less than 5 seconds for 1000 encryptions
        
        # Calculate throughput
        throughput = 1000 / processing_time
        self.assertGreater(throughput, 200)  # At least 200 operations per second


if __name__ == '__main__':
    unittest.main()