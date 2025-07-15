"""Comprehensive compliance framework service for regulatory requirements."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from ...domain.entities.security_entity import (
    ComplianceFramework, ComplianceAssessment, AuditRecord
)


class ComplianceFrameworkService:
    """Service for managing compliance across multiple regulatory frameworks."""
    
    def __init__(self):
        """Initialize compliance framework service."""
        self.frameworks: Dict[ComplianceFramework, Dict[str, Any]] = self._initialize_frameworks()
        self.assessments: Dict[UUID, ComplianceAssessment] = {}
        self.compliance_policies: Dict[str, Dict[str, Any]] = {}
        self.audit_records: List[AuditRecord] = []
    
    def assess_compliance(self, framework: ComplianceFramework, scope: str, 
                         assessment_data: Dict[str, Any], assessor: str) -> ComplianceAssessment:
        """
        Assess compliance against a specific framework.
        
        Args:
            framework: Compliance framework to assess against
            scope: Scope of assessment
            assessment_data: Assessment data
            assessor: Person conducting assessment
            
        Returns:
            Compliance assessment result
        """
        framework_config = self.frameworks.get(framework, {})
        requirements = framework_config.get('requirements', {})
        
        # Assess each requirement
        requirement_scores = {}
        violations = []
        recommendations = []
        evidence = []
        
        for req_id, req_config in requirements.items():
            score, req_violations, req_recommendations, req_evidence = self._assess_requirement(
                req_id, req_config, assessment_data
            )
            
            requirement_scores[req_id] = score
            violations.extend(req_violations)
            recommendations.extend(req_recommendations)
            evidence.extend(req_evidence)
        
        # Calculate overall score
        overall_score = sum(requirement_scores.values()) / len(requirement_scores) if requirement_scores else 0.0
        
        # Create assessment
        assessment = ComplianceAssessment(
            framework=framework,
            scope=scope,
            assessor=assessor,
            overall_score=overall_score,
            requirement_scores=requirement_scores,
            violations=violations,
            recommendations=recommendations,
            evidence=evidence,
            next_assessment_date=datetime.now() + timedelta(days=365)
        )
        
        self.assessments[assessment.assessment_id] = assessment
        
        # Log assessment
        self._log_compliance_action(
            action="compliance_assessment",
            framework=framework,
            details={
                'assessment_id': str(assessment.assessment_id),
                'scope': scope,
                'overall_score': overall_score,
                'assessor': assessor
            }
        )
        
        return assessment
    
    def check_gdpr_compliance(self, data_processing_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check GDPR compliance for data processing activities.
        
        Args:
            data_processing_details: Details of data processing
            
        Returns:
            GDPR compliance assessment
        """
        compliance_result = {
            'framework': 'GDPR',
            'compliant': True,
            'issues': [],
            'recommendations': [],
            'requirements_met': [],
            'requirements_failed': []
        }
        
        # Article 6 - Lawfulness of processing
        if not data_processing_details.get('legal_basis'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('No legal basis specified')
            compliance_result['recommendations'].append('Specify legal basis for processing')
            compliance_result['requirements_failed'].append('Article 6 - Lawfulness')
        else:
            compliance_result['requirements_met'].append('Article 6 - Lawfulness')
        
        # Article 13/14 - Information to be provided
        if not data_processing_details.get('privacy_notice'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Privacy notice not provided')
            compliance_result['recommendations'].append('Provide comprehensive privacy notice')
            compliance_result['requirements_failed'].append('Article 13/14 - Information')
        else:
            compliance_result['requirements_met'].append('Article 13/14 - Information')
        
        # Article 25 - Data protection by design and by default
        if not data_processing_details.get('privacy_by_design'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Privacy by design not implemented')
            compliance_result['recommendations'].append('Implement privacy by design principles')
            compliance_result['requirements_failed'].append('Article 25 - Privacy by Design')
        else:
            compliance_result['requirements_met'].append('Article 25 - Privacy by Design')
        
        # Article 32 - Security of processing
        if not data_processing_details.get('security_measures'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Appropriate security measures not implemented')
            compliance_result['recommendations'].append('Implement appropriate security measures')
            compliance_result['requirements_failed'].append('Article 32 - Security')
        else:
            compliance_result['requirements_met'].append('Article 32 - Security')
        
        # Article 35 - Data protection impact assessment
        if data_processing_details.get('high_risk_processing') and not data_processing_details.get('dpia_conducted'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('DPIA required for high-risk processing')
            compliance_result['recommendations'].append('Conduct Data Protection Impact Assessment')
            compliance_result['requirements_failed'].append('Article 35 - DPIA')
        else:
            compliance_result['requirements_met'].append('Article 35 - DPIA')
        
        return compliance_result
    
    def check_hipaa_compliance(self, healthcare_data_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check HIPAA compliance for healthcare data processing.
        
        Args:
            healthcare_data_details: Details of healthcare data processing
            
        Returns:
            HIPAA compliance assessment
        """
        compliance_result = {
            'framework': 'HIPAA',
            'compliant': True,
            'issues': [],
            'recommendations': [],
            'requirements_met': [],
            'requirements_failed': []
        }
        
        # Privacy Rule - Minimum necessary standard
        if not healthcare_data_details.get('minimum_necessary'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Minimum necessary standard not met')
            compliance_result['recommendations'].append('Implement minimum necessary standard')
            compliance_result['requirements_failed'].append('Privacy Rule - Minimum Necessary')
        else:
            compliance_result['requirements_met'].append('Privacy Rule - Minimum Necessary')
        
        # Security Rule - Access control
        if not healthcare_data_details.get('access_controls'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Access controls not implemented')
            compliance_result['recommendations'].append('Implement user access controls')
            compliance_result['requirements_failed'].append('Security Rule - Access Control')
        else:
            compliance_result['requirements_met'].append('Security Rule - Access Control')
        
        # Security Rule - Audit controls
        if not healthcare_data_details.get('audit_controls'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Audit controls not implemented')
            compliance_result['recommendations'].append('Implement audit controls')
            compliance_result['requirements_failed'].append('Security Rule - Audit Controls')
        else:
            compliance_result['requirements_met'].append('Security Rule - Audit Controls')
        
        # Security Rule - Integrity controls
        if not healthcare_data_details.get('integrity_controls'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Integrity controls not implemented')
            compliance_result['recommendations'].append('Implement integrity controls')
            compliance_result['requirements_failed'].append('Security Rule - Integrity')
        else:
            compliance_result['requirements_met'].append('Security Rule - Integrity')
        
        # Security Rule - Transmission security
        if not healthcare_data_details.get('transmission_security'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Transmission security not implemented')
            compliance_result['recommendations'].append('Implement transmission security')
            compliance_result['requirements_failed'].append('Security Rule - Transmission')
        else:
            compliance_result['requirements_met'].append('Security Rule - Transmission')
        
        # Breach Notification Rule
        if not healthcare_data_details.get('breach_notification_procedures'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Breach notification procedures not established')
            compliance_result['recommendations'].append('Establish breach notification procedures')
            compliance_result['requirements_failed'].append('Breach Notification Rule')
        else:
            compliance_result['requirements_met'].append('Breach Notification Rule')
        
        return compliance_result
    
    def check_sox_compliance(self, financial_data_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check SOX compliance for financial data processing.
        
        Args:
            financial_data_details: Details of financial data processing
            
        Returns:
            SOX compliance assessment
        """
        compliance_result = {
            'framework': 'SOX',
            'compliant': True,
            'issues': [],
            'recommendations': [],
            'requirements_met': [],
            'requirements_failed': []
        }
        
        # Section 302 - Corporate responsibility
        if not financial_data_details.get('executive_certification'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Executive certification not provided')
            compliance_result['recommendations'].append('Implement executive certification process')
            compliance_result['requirements_failed'].append('Section 302 - Corporate Responsibility')
        else:
            compliance_result['requirements_met'].append('Section 302 - Corporate Responsibility')
        
        # Section 404 - Management assessment of internal controls
        if not financial_data_details.get('internal_controls_assessment'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Internal controls assessment not conducted')
            compliance_result['recommendations'].append('Conduct internal controls assessment')
            compliance_result['requirements_failed'].append('Section 404 - Internal Controls')
        else:
            compliance_result['requirements_met'].append('Section 404 - Internal Controls')
        
        # Section 409 - Real-time disclosure
        if not financial_data_details.get('real_time_disclosure'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Real-time disclosure not implemented')
            compliance_result['recommendations'].append('Implement real-time disclosure')
            compliance_result['requirements_failed'].append('Section 409 - Real-time Disclosure')
        else:
            compliance_result['requirements_met'].append('Section 409 - Real-time Disclosure')
        
        # Data integrity and retention
        if not financial_data_details.get('data_integrity_controls'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Data integrity controls not implemented')
            compliance_result['recommendations'].append('Implement data integrity controls')
            compliance_result['requirements_failed'].append('Data Integrity Controls')
        else:
            compliance_result['requirements_met'].append('Data Integrity Controls')
        
        # Audit trail requirements
        if not financial_data_details.get('audit_trail'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Audit trail not maintained')
            compliance_result['recommendations'].append('Implement comprehensive audit trail')
            compliance_result['requirements_failed'].append('Audit Trail Requirements')
        else:
            compliance_result['requirements_met'].append('Audit Trail Requirements')
        
        return compliance_result
    
    def check_ccpa_compliance(self, california_data_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check CCPA compliance for California consumer data processing.
        
        Args:
            california_data_details: Details of California consumer data processing
            
        Returns:
            CCPA compliance assessment
        """
        compliance_result = {
            'framework': 'CCPA',
            'compliant': True,
            'issues': [],
            'recommendations': [],
            'requirements_met': [],
            'requirements_failed': []
        }
        
        # Right to know
        if not california_data_details.get('right_to_know_notice'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Right to know notice not provided')
            compliance_result['recommendations'].append('Provide right to know notice')
            compliance_result['requirements_failed'].append('Right to Know')
        else:
            compliance_result['requirements_met'].append('Right to Know')
        
        # Right to delete
        if not california_data_details.get('deletion_mechanisms'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Deletion mechanisms not implemented')
            compliance_result['recommendations'].append('Implement deletion mechanisms')
            compliance_result['requirements_failed'].append('Right to Delete')
        else:
            compliance_result['requirements_met'].append('Right to Delete')
        
        # Right to opt-out
        if not california_data_details.get('opt_out_mechanisms'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Opt-out mechanisms not implemented')
            compliance_result['recommendations'].append('Implement opt-out mechanisms')
            compliance_result['requirements_failed'].append('Right to Opt-Out')
        else:
            compliance_result['requirements_met'].append('Right to Opt-Out')
        
        # Non-discrimination
        if not california_data_details.get('non_discrimination_policy'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Non-discrimination policy not implemented')
            compliance_result['recommendations'].append('Implement non-discrimination policy')
            compliance_result['requirements_failed'].append('Non-discrimination')
        else:
            compliance_result['requirements_met'].append('Non-discrimination')
        
        return compliance_result
    
    def check_pci_dss_compliance(self, payment_data_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check PCI DSS compliance for payment data processing.
        
        Args:
            payment_data_details: Details of payment data processing
            
        Returns:
            PCI DSS compliance assessment
        """
        compliance_result = {
            'framework': 'PCI DSS',
            'compliant': True,
            'issues': [],
            'recommendations': [],
            'requirements_met': [],
            'requirements_failed': []
        }
        
        # Requirement 1: Install and maintain firewall
        if not payment_data_details.get('firewall_configured'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Firewall not properly configured')
            compliance_result['recommendations'].append('Install and configure firewall')
            compliance_result['requirements_failed'].append('Requirement 1 - Firewall')
        else:
            compliance_result['requirements_met'].append('Requirement 1 - Firewall')
        
        # Requirement 2: Do not use vendor-supplied defaults
        if not payment_data_details.get('default_passwords_changed'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Default passwords not changed')
            compliance_result['recommendations'].append('Change all default passwords')
            compliance_result['requirements_failed'].append('Requirement 2 - Default Passwords')
        else:
            compliance_result['requirements_met'].append('Requirement 2 - Default Passwords')
        
        # Requirement 3: Protect stored cardholder data
        if not payment_data_details.get('cardholder_data_protected'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Cardholder data not properly protected')
            compliance_result['recommendations'].append('Implement cardholder data protection')
            compliance_result['requirements_failed'].append('Requirement 3 - Data Protection')
        else:
            compliance_result['requirements_met'].append('Requirement 3 - Data Protection')
        
        # Requirement 4: Encrypt transmission of cardholder data
        if not payment_data_details.get('transmission_encrypted'):
            compliance_result['compliant'] = False
            compliance_result['issues'].append('Cardholder data transmission not encrypted')
            compliance_result['recommendations'].append('Encrypt cardholder data transmission')
            compliance_result['requirements_failed'].append('Requirement 4 - Encryption')
        else:
            compliance_result['requirements_met'].append('Requirement 4 - Encryption')
        
        return compliance_result
    
    def generate_compliance_report(self, assessment_id: UUID) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report.
        
        Args:
            assessment_id: Assessment identifier
            
        Returns:
            Compliance report
        """
        if assessment_id not in self.assessments:
            return {'error': 'Assessment not found'}
        
        assessment = self.assessments[assessment_id]
        
        return {
            'assessment_summary': {
                'assessment_id': str(assessment_id),
                'framework': assessment.framework.value,
                'scope': assessment.scope,
                'overall_score': assessment.overall_score,
                'assessment_date': assessment.assessment_date.isoformat(),
                'assessor': assessment.assessor,
                'approved': assessment.approved
            },
            'requirement_analysis': {
                'total_requirements': len(assessment.requirement_scores),
                'requirements_met': sum(1 for score in assessment.requirement_scores.values() if score >= 0.8),
                'requirements_failed': sum(1 for score in assessment.requirement_scores.values() if score < 0.5),
                'requirement_scores': assessment.requirement_scores
            },
            'violations': assessment.violations,
            'recommendations': assessment.recommendations,
            'evidence': assessment.evidence,
            'remediation_plan': self._generate_remediation_plan(assessment),
            'next_assessment_date': assessment.next_assessment_date.isoformat()
        }
    
    def _initialize_frameworks(self) -> Dict[ComplianceFramework, Dict[str, Any]]:
        """Initialize compliance frameworks with their requirements."""
        return {
            ComplianceFramework.GDPR: {
                'name': 'General Data Protection Regulation',
                'requirements': {
                    'lawfulness': {'weight': 0.2, 'description': 'Article 6 - Lawfulness of processing'},
                    'consent': {'weight': 0.15, 'description': 'Article 7 - Conditions for consent'},
                    'information': {'weight': 0.15, 'description': 'Article 13/14 - Information to be provided'},
                    'rights': {'weight': 0.15, 'description': 'Chapter III - Rights of data subjects'},
                    'privacy_by_design': {'weight': 0.1, 'description': 'Article 25 - Privacy by design'},
                    'security': {'weight': 0.15, 'description': 'Article 32 - Security of processing'},
                    'breach_notification': {'weight': 0.1, 'description': 'Article 33/34 - Breach notification'}
                }
            },
            ComplianceFramework.HIPAA: {
                'name': 'Health Insurance Portability and Accountability Act',
                'requirements': {
                    'minimum_necessary': {'weight': 0.2, 'description': 'Privacy Rule - Minimum necessary'},
                    'access_controls': {'weight': 0.15, 'description': 'Security Rule - Access control'},
                    'audit_controls': {'weight': 0.15, 'description': 'Security Rule - Audit controls'},
                    'integrity': {'weight': 0.15, 'description': 'Security Rule - Integrity'},
                    'transmission_security': {'weight': 0.15, 'description': 'Security Rule - Transmission security'},
                    'breach_notification': {'weight': 0.2, 'description': 'Breach Notification Rule'}
                }
            },
            ComplianceFramework.SOX: {
                'name': 'Sarbanes-Oxley Act',
                'requirements': {
                    'corporate_responsibility': {'weight': 0.25, 'description': 'Section 302 - Corporate responsibility'},
                    'internal_controls': {'weight': 0.25, 'description': 'Section 404 - Internal controls'},
                    'real_time_disclosure': {'weight': 0.2, 'description': 'Section 409 - Real-time disclosure'},
                    'data_integrity': {'weight': 0.15, 'description': 'Data integrity controls'},
                    'audit_trail': {'weight': 0.15, 'description': 'Audit trail requirements'}
                }
            },
            ComplianceFramework.CCPA: {
                'name': 'California Consumer Privacy Act',
                'requirements': {
                    'right_to_know': {'weight': 0.25, 'description': 'Right to know'},
                    'right_to_delete': {'weight': 0.25, 'description': 'Right to delete'},
                    'right_to_opt_out': {'weight': 0.25, 'description': 'Right to opt-out'},
                    'non_discrimination': {'weight': 0.25, 'description': 'Non-discrimination'}
                }
            },
            ComplianceFramework.PCI_DSS: {
                'name': 'Payment Card Industry Data Security Standard',
                'requirements': {
                    'firewall': {'weight': 0.15, 'description': 'Requirement 1 - Firewall'},
                    'default_passwords': {'weight': 0.1, 'description': 'Requirement 2 - Default passwords'},
                    'data_protection': {'weight': 0.25, 'description': 'Requirement 3 - Data protection'},
                    'encryption': {'weight': 0.2, 'description': 'Requirement 4 - Encryption'},
                    'access_control': {'weight': 0.15, 'description': 'Requirement 8 - Access control'},
                    'monitoring': {'weight': 0.15, 'description': 'Requirement 10 - Monitoring'}
                }
            }
        }
    
    def _assess_requirement(self, req_id: str, req_config: Dict[str, Any], 
                           assessment_data: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Assess a specific requirement."""
        score = 0.0
        violations = []
        recommendations = []
        evidence = []
        
        # Check if requirement is met based on assessment data
        requirement_met = assessment_data.get(req_id, False)
        
        if requirement_met:
            score = 1.0
            evidence.append({
                'requirement': req_id,
                'evidence_type': 'compliance_check',
                'description': f'Requirement {req_id} is met',
                'timestamp': datetime.now().isoformat()
            })
        else:
            score = 0.0
            violations.append({
                'requirement': req_id,
                'severity': 'high',
                'description': f'Requirement {req_id} is not met',
                'remediation': f'Implement controls for {req_id}'
            })
            recommendations.append({
                'requirement': req_id,
                'recommendation': f'Implement {req_config["description"]}',
                'priority': 'high'
            })
        
        return score, violations, recommendations, evidence
    
    def _generate_remediation_plan(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate remediation plan based on assessment."""
        plan = {
            'priority_actions': [],
            'medium_term_actions': [],
            'long_term_actions': [],
            'estimated_timeline': '6 months',
            'estimated_cost': 'Medium'
        }
        
        # Prioritize actions based on violation severity
        for violation in assessment.violations:
            if violation.get('severity') == 'high':
                plan['priority_actions'].append({
                    'action': violation.get('remediation', ''),
                    'requirement': violation.get('requirement', ''),
                    'timeline': '30 days'
                })
            elif violation.get('severity') == 'medium':
                plan['medium_term_actions'].append({
                    'action': violation.get('remediation', ''),
                    'requirement': violation.get('requirement', ''),
                    'timeline': '90 days'
                })
            else:
                plan['long_term_actions'].append({
                    'action': violation.get('remediation', ''),
                    'requirement': violation.get('requirement', ''),
                    'timeline': '180 days'
                })
        
        return plan
    
    def _log_compliance_action(self, action: str, framework: ComplianceFramework, 
                              details: Dict[str, Any]) -> None:
        """Log compliance-related action."""
        audit_record = AuditRecord(
            user_id='system',
            action=action,
            resource_type='compliance_assessment',
            resource_id=str(framework.value),
            changes=details,
            compliance_context=[framework],
            timestamp=datetime.now(),
            metadata={'framework': framework.value}
        )
        
        self.audit_records.append(audit_record)


class ComplianceOrchestrationService:
    """Service for orchestrating compliance across multiple frameworks."""
    
    def __init__(self, framework_service: ComplianceFrameworkService):
        """
        Initialize compliance orchestration service.
        
        Args:
            framework_service: Compliance framework service
        """
        self.framework_service = framework_service
        self.compliance_matrix: Dict[str, List[ComplianceFramework]] = {}
    
    def register_data_processing(self, processing_id: str, 
                                frameworks: List[ComplianceFramework]) -> None:
        """
        Register data processing activity with applicable frameworks.
        
        Args:
            processing_id: Processing activity identifier
            frameworks: Applicable compliance frameworks
        """
        self.compliance_matrix[processing_id] = frameworks
    
    def assess_multi_framework_compliance(self, processing_id: str, 
                                        assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess compliance across multiple frameworks.
        
        Args:
            processing_id: Processing activity identifier
            assessment_data: Assessment data
            
        Returns:
            Multi-framework compliance assessment
        """
        if processing_id not in self.compliance_matrix:
            return {'error': 'Processing activity not registered'}
        
        frameworks = self.compliance_matrix[processing_id]
        results = {}
        
        for framework in frameworks:
            if framework == ComplianceFramework.GDPR:
                results['GDPR'] = self.framework_service.check_gdpr_compliance(assessment_data)
            elif framework == ComplianceFramework.HIPAA:
                results['HIPAA'] = self.framework_service.check_hipaa_compliance(assessment_data)
            elif framework == ComplianceFramework.SOX:
                results['SOX'] = self.framework_service.check_sox_compliance(assessment_data)
            elif framework == ComplianceFramework.CCPA:
                results['CCPA'] = self.framework_service.check_ccpa_compliance(assessment_data)
            elif framework == ComplianceFramework.PCI_DSS:
                results['PCI_DSS'] = self.framework_service.check_pci_dss_compliance(assessment_data)
        
        # Generate combined compliance status
        overall_compliant = all(result.get('compliant', False) for result in results.values())
        
        return {
            'processing_id': processing_id,
            'overall_compliant': overall_compliant,
            'framework_results': results,
            'combined_recommendations': self._combine_recommendations(results)
        }
    
    def _combine_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Combine recommendations from multiple frameworks."""
        all_recommendations = []
        
        for framework_result in results.values():
            all_recommendations.extend(framework_result.get('recommendations', []))
        
        # Remove duplicates and return
        return list(set(all_recommendations))