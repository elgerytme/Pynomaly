"""Privacy Impact Assessment Service for data quality operations."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from ...domain.entities.security_entity import (
    PrivacyImpactAssessment, ComplianceFramework, PrivacyLevel
)


class PrivacyImpactAssessmentService:
    """Service for conducting privacy impact assessments."""
    
    def __init__(self):
        """Initialize privacy impact assessment service."""
        self.assessments: Dict[UUID, PrivacyImpactAssessment] = {}
        self.assessment_templates: Dict[str, Dict[str, Any]] = {}
        self.risk_matrix: Dict[str, Dict[str, str]] = self._initialize_risk_matrix()
    
    def create_assessment(self, operation_name: str, description: str, 
                         assessor: str, assessment_data: Dict[str, Any]) -> PrivacyImpactAssessment:
        """
        Create a new privacy impact assessment.
        
        Args:
            operation_name: Name of the operation
            description: Description of the operation
            assessor: Person conducting the assessment
            assessment_data: Assessment data
            
        Returns:
            Privacy impact assessment
        """
        assessment = PrivacyImpactAssessment(
            operation_name=operation_name,
            description=description,
            assessor=assessor,
            data_categories=assessment_data.get('data_categories', []),
            processing_purposes=assessment_data.get('processing_purposes', []),
            legal_basis=assessment_data.get('legal_basis', ''),
            data_subjects=assessment_data.get('data_subjects', []),
            compliance_frameworks=assessment_data.get('compliance_frameworks', []),
            metadata=assessment_data.get('metadata', {})
        )
        
        # Assess risk level
        assessment.risk_level = self._assess_risk_level(assessment_data)
        
        # Generate mitigation measures
        assessment.mitigation_measures = self._generate_mitigation_measures(assessment_data)
        
        # Set review date
        assessment.review_date = datetime.now() + timedelta(days=365)  # Annual review
        
        self.assessments[assessment.assessment_id] = assessment
        
        return assessment
    
    def conduct_assessment(self, assessment_id: UUID, 
                          assessment_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conduct detailed privacy impact assessment.
        
        Args:
            assessment_id: Assessment identifier
            assessment_criteria: Assessment criteria
            
        Returns:
            Assessment results
        """
        if assessment_id not in self.assessments:
            raise ValueError(f"Assessment {assessment_id} not found")
        
        assessment = self.assessments[assessment_id]
        
        # Conduct assessment across different dimensions
        results = {
            'assessment_id': str(assessment_id),
            'data_minimization': self._assess_data_minimization(assessment, assessment_criteria),
            'purpose_limitation': self._assess_purpose_limitation(assessment, assessment_criteria),
            'storage_limitation': self._assess_storage_limitation(assessment, assessment_criteria),
            'accuracy_assessment': self._assess_accuracy(assessment, assessment_criteria),
            'security_assessment': self._assess_security(assessment, assessment_criteria),
            'transparency_assessment': self._assess_transparency(assessment, assessment_criteria),
            'rights_assessment': self._assess_data_subject_rights(assessment, assessment_criteria),
            'international_transfer': self._assess_international_transfer(assessment, assessment_criteria),
            'automated_processing': self._assess_automated_processing(assessment, assessment_criteria),
            'overall_score': 0.0,
            'risk_level': assessment.risk_level,
            'recommendations': [],
            'compliance_gaps': []
        }
        
        # Calculate overall score
        scores = []
        for key, value in results.items():
            if isinstance(value, dict) and 'score' in value:
                scores.append(value['score'])
        
        if scores:
            results['overall_score'] = sum(scores) / len(scores)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        # Identify compliance gaps
        results['compliance_gaps'] = self._identify_compliance_gaps(assessment, results)
        
        return results
    
    def approve_assessment(self, assessment_id: UUID, approver: str, 
                          approval_conditions: Optional[List[str]] = None) -> bool:
        """
        Approve privacy impact assessment.
        
        Args:
            assessment_id: Assessment identifier
            approver: Person approving the assessment
            approval_conditions: Conditions for approval
            
        Returns:
            Whether approval was successful
        """
        if assessment_id not in self.assessments:
            return False
        
        assessment = self.assessments[assessment_id]
        assessment.approved = True
        assessment.approval_date = datetime.now()
        assessment.metadata['approver'] = approver
        
        if approval_conditions:
            assessment.metadata['approval_conditions'] = approval_conditions
        
        return True
    
    def get_assessment_status(self, assessment_id: UUID) -> Dict[str, Any]:
        """
        Get assessment status.
        
        Args:
            assessment_id: Assessment identifier
            
        Returns:
            Assessment status
        """
        if assessment_id not in self.assessments:
            return {'error': 'Assessment not found'}
        
        assessment = self.assessments[assessment_id]
        
        return {
            'assessment_id': str(assessment_id),
            'operation_name': assessment.operation_name,
            'assessor': assessment.assessor,
            'assessment_date': assessment.assessment_date.isoformat(),
            'approved': assessment.approved,
            'approval_date': assessment.approval_date.isoformat() if assessment.approval_date else None,
            'review_date': assessment.review_date.isoformat() if assessment.review_date else None,
            'risk_level': assessment.risk_level,
            'compliance_frameworks': [f.value for f in assessment.compliance_frameworks],
            'requires_review': assessment.review_date and datetime.now() > assessment.review_date
        }
    
    def generate_assessment_report(self, assessment_id: UUID) -> Dict[str, Any]:
        """
        Generate comprehensive assessment report.
        
        Args:
            assessment_id: Assessment identifier
            
        Returns:
            Assessment report
        """
        if assessment_id not in self.assessments:
            return {'error': 'Assessment not found'}
        
        assessment = self.assessments[assessment_id]
        
        return {
            'assessment_summary': {
                'operation_name': assessment.operation_name,
                'description': assessment.description,
                'assessor': assessment.assessor,
                'assessment_date': assessment.assessment_date.isoformat(),
                'risk_level': assessment.risk_level,
                'approved': assessment.approved
            },
            'data_processing_details': {
                'data_categories': assessment.data_categories,
                'processing_purposes': assessment.processing_purposes,
                'legal_basis': assessment.legal_basis,
                'data_subjects': assessment.data_subjects
            },
            'compliance_framework_analysis': self._analyze_compliance_frameworks(assessment),
            'risk_assessment': {
                'risk_level': assessment.risk_level,
                'risk_factors': self._identify_risk_factors(assessment),
                'mitigation_measures': assessment.mitigation_measures
            },
            'recommendations': self._generate_detailed_recommendations(assessment),
            'next_steps': self._generate_next_steps(assessment)
        }
    
    def _initialize_risk_matrix(self) -> Dict[str, Dict[str, str]]:
        """Initialize risk assessment matrix."""
        return {
            'low': {
                'impact': 'minimal',
                'likelihood': 'unlikely',
                'mitigation': 'standard'
            },
            'medium': {
                'impact': 'moderate',
                'likelihood': 'possible',
                'mitigation': 'enhanced'
            },
            'high': {
                'impact': 'significant',
                'likelihood': 'likely',
                'mitigation': 'comprehensive'
            },
            'critical': {
                'impact': 'severe',
                'likelihood': 'very_likely',
                'mitigation': 'immediate'
            }
        }
    
    def _assess_risk_level(self, assessment_data: Dict[str, Any]) -> str:
        """Assess overall risk level."""
        risk_factors = 0
        
        # High-risk data categories
        high_risk_categories = ['biometric', 'health', 'genetic', 'financial', 'children']
        for category in assessment_data.get('data_categories', []):
            if category.lower() in high_risk_categories:
                risk_factors += 2
        
        # High-risk processing purposes
        high_risk_purposes = ['profiling', 'automated_decision_making', 'surveillance']
        for purpose in assessment_data.get('processing_purposes', []):
            if purpose.lower() in high_risk_purposes:
                risk_factors += 2
        
        # Vulnerable data subjects
        vulnerable_subjects = ['children', 'elderly', 'disabled', 'employees']
        for subject in assessment_data.get('data_subjects', []):
            if subject.lower() in vulnerable_subjects:
                risk_factors += 1
        
        # International transfers
        if assessment_data.get('international_transfer', False):
            risk_factors += 1
        
        # Large scale processing
        if assessment_data.get('large_scale_processing', False):
            risk_factors += 1
        
        # Determine risk level
        if risk_factors >= 6:
            return 'critical'
        elif risk_factors >= 4:
            return 'high'
        elif risk_factors >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_mitigation_measures(self, assessment_data: Dict[str, Any]) -> List[str]:
        """Generate mitigation measures based on assessment data."""
        measures = []
        
        # Data minimization
        measures.append('Implement data minimization principles')
        
        # Purpose limitation
        measures.append('Ensure processing is limited to specified purposes')
        
        # Security measures
        measures.append('Implement appropriate technical and organizational measures')
        
        # Transparency
        measures.append('Provide clear privacy notices to data subjects')
        
        # Rights facilitation
        measures.append('Implement mechanisms to facilitate data subject rights')
        
        # Data retention
        measures.append('Establish clear data retention and deletion policies')
        
        # Staff training
        measures.append('Provide privacy training to staff')
        
        # Regular reviews
        measures.append('Conduct regular privacy compliance reviews')
        
        return measures
    
    def _assess_data_minimization(self, assessment: PrivacyImpactAssessment, 
                                 criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data minimization compliance."""
        score = 0.8  # Base score
        issues = []
        
        # Check if data collection is limited to what's necessary
        if not criteria.get('data_collection_limited', False):
            score -= 0.2
            issues.append('Data collection not limited to necessary data')
        
        # Check if data retention is limited
        if not criteria.get('retention_limited', False):
            score -= 0.2
            issues.append('Data retention period not limited')
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'recommendations': ['Implement data minimization controls']
        }
    
    def _assess_purpose_limitation(self, assessment: PrivacyImpactAssessment, 
                                  criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Assess purpose limitation compliance."""
        score = 0.8  # Base score
        issues = []
        
        # Check if purposes are clearly defined
        if not criteria.get('purposes_defined', False):
            score -= 0.3
            issues.append('Processing purposes not clearly defined')
        
        # Check if processing is limited to defined purposes
        if not criteria.get('processing_limited', False):
            score -= 0.3
            issues.append('Processing not limited to defined purposes')
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'recommendations': ['Clearly define and limit processing purposes']
        }
    
    def _assess_storage_limitation(self, assessment: PrivacyImpactAssessment, 
                                  criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Assess storage limitation compliance."""
        score = 0.8  # Base score
        issues = []
        
        # Check if retention periods are defined
        if not criteria.get('retention_periods_defined', False):
            score -= 0.3
            issues.append('Retention periods not defined')
        
        # Check if automatic deletion is implemented
        if not criteria.get('automatic_deletion', False):
            score -= 0.2
            issues.append('Automatic deletion not implemented')
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'recommendations': ['Implement data retention and deletion policies']
        }
    
    def _assess_accuracy(self, assessment: PrivacyImpactAssessment, 
                        criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data accuracy compliance."""
        score = 0.8  # Base score
        issues = []
        
        # Check if data accuracy measures are in place
        if not criteria.get('accuracy_measures', False):
            score -= 0.3
            issues.append('Data accuracy measures not implemented')
        
        # Check if data can be corrected
        if not criteria.get('correction_mechanisms', False):
            score -= 0.2
            issues.append('Data correction mechanisms not available')
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'recommendations': ['Implement data accuracy and correction measures']
        }
    
    def _assess_security(self, assessment: PrivacyImpactAssessment, 
                        criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Assess security compliance."""
        score = 0.8  # Base score
        issues = []
        
        # Check if encryption is implemented
        if not criteria.get('encryption_implemented', False):
            score -= 0.3
            issues.append('Encryption not implemented')
        
        # Check if access controls are in place
        if not criteria.get('access_controls', False):
            score -= 0.2
            issues.append('Access controls not implemented')
        
        # Check if security monitoring is in place
        if not criteria.get('security_monitoring', False):
            score -= 0.2
            issues.append('Security monitoring not implemented')
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'recommendations': ['Implement comprehensive security measures']
        }
    
    def _assess_transparency(self, assessment: PrivacyImpactAssessment, 
                           criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Assess transparency compliance."""
        score = 0.8  # Base score
        issues = []
        
        # Check if privacy notices are provided
        if not criteria.get('privacy_notices', False):
            score -= 0.3
            issues.append('Privacy notices not provided')
        
        # Check if processing activities are documented
        if not criteria.get('processing_documented', False):
            score -= 0.2
            issues.append('Processing activities not documented')
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'recommendations': ['Provide clear privacy notices and documentation']
        }
    
    def _assess_data_subject_rights(self, assessment: PrivacyImpactAssessment, 
                                   criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data subject rights compliance."""
        score = 0.8  # Base score
        issues = []
        
        # Check if rights mechanisms are implemented
        if not criteria.get('rights_mechanisms', False):
            score -= 0.3
            issues.append('Data subject rights mechanisms not implemented')
        
        # Check if rights requests are handled timely
        if not criteria.get('timely_responses', False):
            score -= 0.2
            issues.append('Rights requests not handled timely')
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'recommendations': ['Implement data subject rights mechanisms']
        }
    
    def _assess_international_transfer(self, assessment: PrivacyImpactAssessment, 
                                      criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Assess international transfer compliance."""
        score = 0.8  # Base score
        issues = []
        
        # Check if transfers are to adequate countries
        if criteria.get('international_transfers', False):
            if not criteria.get('adequate_countries', False):
                score -= 0.3
                issues.append('International transfers not to adequate countries')
            
            # Check if appropriate safeguards are in place
            if not criteria.get('transfer_safeguards', False):
                score -= 0.2
                issues.append('Transfer safeguards not implemented')
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'recommendations': ['Implement transfer safeguards for international transfers']
        }
    
    def _assess_automated_processing(self, assessment: PrivacyImpactAssessment, 
                                    criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Assess automated processing compliance."""
        score = 0.8  # Base score
        issues = []
        
        # Check if automated processing is disclosed
        if criteria.get('automated_processing', False):
            if not criteria.get('automated_processing_disclosed', False):
                score -= 0.3
                issues.append('Automated processing not disclosed')
            
            # Check if human review is available
            if not criteria.get('human_review_available', False):
                score -= 0.2
                issues.append('Human review not available for automated decisions')
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'recommendations': ['Implement transparency for automated processing']
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on assessment results."""
        recommendations = []
        
        # Low score recommendations
        for key, value in results.items():
            if isinstance(value, dict) and 'score' in value:
                if value['score'] < 0.6:
                    recommendations.extend(value.get('recommendations', []))
        
        # Risk-based recommendations
        if results['risk_level'] in ['high', 'critical']:
            recommendations.append('Conduct regular privacy audits')
            recommendations.append('Implement privacy by design principles')
            recommendations.append('Consider appointing a Data Protection Officer')
        
        return list(set(recommendations))  # Remove duplicates
    
    def _identify_compliance_gaps(self, assessment: PrivacyImpactAssessment, 
                                 results: Dict[str, Any]) -> List[str]:
        """Identify compliance gaps."""
        gaps = []
        
        # Framework-specific gaps
        for framework in assessment.compliance_frameworks:
            if framework == ComplianceFramework.GDPR:
                if results['overall_score'] < 0.7:
                    gaps.append('GDPR compliance score below acceptable threshold')
            elif framework == ComplianceFramework.CCPA:
                if results['transparency_assessment']['score'] < 0.7:
                    gaps.append('CCPA transparency requirements not met')
        
        return gaps
    
    def _analyze_compliance_frameworks(self, assessment: PrivacyImpactAssessment) -> Dict[str, Any]:
        """Analyze compliance frameworks."""
        framework_analysis = {}
        
        for framework in assessment.compliance_frameworks:
            if framework == ComplianceFramework.GDPR:
                framework_analysis['GDPR'] = {
                    'applicable': True,
                    'key_requirements': ['consent', 'data_minimization', 'transparency', 'rights'],
                    'risk_level': 'high' if 'EU' in assessment.metadata.get('geographic_scope', []) else 'medium'
                }
            elif framework == ComplianceFramework.CCPA:
                framework_analysis['CCPA'] = {
                    'applicable': True,
                    'key_requirements': ['notice', 'opt_out', 'access', 'deletion'],
                    'risk_level': 'high' if 'California' in assessment.metadata.get('geographic_scope', []) else 'medium'
                }
        
        return framework_analysis
    
    def _identify_risk_factors(self, assessment: PrivacyImpactAssessment) -> List[str]:
        """Identify risk factors."""
        risk_factors = []
        
        # High-risk data categories
        high_risk_categories = ['biometric', 'health', 'genetic', 'financial']
        for category in assessment.data_categories:
            if category.lower() in high_risk_categories:
                risk_factors.append(f'Processing of {category} data')
        
        # High-risk processing purposes
        high_risk_purposes = ['profiling', 'automated_decision_making']
        for purpose in assessment.processing_purposes:
            if purpose.lower() in high_risk_purposes:
                risk_factors.append(f'Processing for {purpose}')
        
        return risk_factors
    
    def _generate_detailed_recommendations(self, assessment: PrivacyImpactAssessment) -> List[str]:
        """Generate detailed recommendations."""
        recommendations = []
        
        # Risk-based recommendations
        if assessment.risk_level == 'high':
            recommendations.append('Conduct regular privacy audits')
            recommendations.append('Implement privacy by design')
            recommendations.append('Consider appointing a DPO')
        
        # Framework-specific recommendations
        for framework in assessment.compliance_frameworks:
            if framework == ComplianceFramework.GDPR:
                recommendations.append('Implement GDPR Article 25 requirements')
                recommendations.append('Establish data subject rights procedures')
            elif framework == ComplianceFramework.CCPA:
                recommendations.append('Implement CCPA consumer rights')
                recommendations.append('Provide clear privacy notices')
        
        return recommendations
    
    def _generate_next_steps(self, assessment: PrivacyImpactAssessment) -> List[str]:
        """Generate next steps."""
        next_steps = []
        
        if not assessment.approved:
            next_steps.append('Obtain approval for privacy impact assessment')
        
        next_steps.append('Implement recommended mitigation measures')
        next_steps.append('Establish regular review schedule')
        next_steps.append('Train staff on privacy requirements')
        next_steps.append('Monitor compliance on ongoing basis')
        
        return next_steps