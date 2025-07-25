# Patient Records Management System Example

A HIPAA-compliant electronic health records (EHR) system built with security-first principles, enabling healthcare providers to manage patient data securely and efficiently.

## Features

- **Patient Demographics**: Comprehensive patient information management
- **Medical History**: Detailed medical records and visit notes
- **Prescription Management**: Medication tracking and e-prescribing
- **Appointment Scheduling**: Calendar integration and reminders
- **Laboratory Results**: Test results and imaging integration
- **Clinical Decision Support**: AI-powered diagnostic assistance
- **Billing Integration**: Insurance and payment processing
- **Telemedicine**: Remote consultation capabilities
- **Mobile Access**: Healthcare provider mobile app

## Architecture

Clean architecture with strict data protection:

```
src/patient_records/
├── domain/                     # Core medical entities
│   ├── entities/              # Patient, Provider, Visit
│   ├── value_objects/         # MedicalRecord, Prescription
│   ├── aggregates/            # Patient aggregate root
│   └── services/              # Medical calculation services
├── application/               # Healthcare use cases
│   ├── use_cases/            # Create patient, schedule visit
│   ├── security/             # Access control and audit
│   └── integrations/         # HL7, FHIR interfaces
├── infrastructure/           # External systems
│   ├── databases/            # Encrypted PostgreSQL
│   ├── hl7/                  # HL7 message processing
│   ├── imaging/              # DICOM integration
│   └── compliance/           # HIPAA audit logging
└── presentation/             # Secure API layer
    ├── api/                  # REST API with OAuth2
    ├── webhooks/             # Lab results webhooks
    └── reports/              # Clinical reporting
```

## Quick Start

```bash
# Start the EHR system
cd examples/healthcare-patient-records
docker-compose up -d

# Access the API documentation
open http://localhost:8000/docs

# Create a patient record
curl -X POST http://localhost:8000/api/patients \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "John",
    "last_name": "Doe",
    "date_of_birth": "1985-06-15",
    "gender": "M",
    "ssn_last_four": "1234",
    "contact": {
      "phone": "+1-555-0123",
      "email": "john.doe@email.com",
      "address": {
        "street": "123 Main St",
        "city": "Anytown",
        "state": "CA",
        "zip_code": "12345"
      }
    }
  }'
```

## API Endpoints

### Patient Management
- `POST /api/patients` - Create patient record
- `GET /api/patients/{patient_id}` - Get patient details
- `PUT /api/patients/{patient_id}` - Update patient information
- `GET /api/patients/search` - Search patients

### Medical Records
- `POST /api/patients/{patient_id}/visits` - Create visit record
- `GET /api/patients/{patient_id}/visits` - Get visit history
- `POST /api/patients/{patient_id}/allergies` - Add allergy
- `GET /api/patients/{patient_id}/medical-history` - Get medical history

### Prescriptions
- `POST /api/prescriptions` - Create prescription
- `GET /api/prescriptions/{prescription_id}` - Get prescription
- `PUT /api/prescriptions/{prescription_id}/status` - Update status

### Laboratory
- `POST /api/lab-orders` - Create lab order
- `GET /api/lab-results/{patient_id}` - Get lab results
- `POST /api/lab-results` - Upload lab results (webhook)

### Appointments
- `POST /api/appointments` - Schedule appointment
- `GET /api/appointments/provider/{provider_id}` - Provider schedule
- `PUT /api/appointments/{appointment_id}` - Update appointment

## Security & Compliance

### HIPAA Compliance
- **Access Controls**: Role-based access with minimum necessary principle
- **Audit Logging**: Complete audit trail of all data access
- **Data Encryption**: AES-256 encryption at rest and in transit
- **Business Associate Agreements**: Vendor compliance tracking

### Authentication & Authorization
```python
# JWT token with role-based access
@require_role("physician", "nurse", "admin")
async def get_patient_record(patient_id: str, current_user: User):
    # Verify user has access to this patient
    access_granted = await verify_patient_access(
        user_id=current_user.id,
        patient_id=patient_id,
        access_type="read"
    )
    
    if not access_granted:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Log access for audit
    await audit_logger.log_access(
        user_id=current_user.id,
        resource_type="patient_record",
        resource_id=patient_id,
        action="read"
    )
    
    return await patient_service.get_patient(patient_id)
```

### Data Protection
- **PHI Encryption**: All protected health information encrypted
- **Data Masking**: Sensitive data masked in non-production environments
- **Access Logging**: Every data access logged with user context
- **Automatic Logout**: Session timeout for security

## HL7 & FHIR Integration

### HL7 Message Processing
```python
# HL7 v2.x message handling
class HL7MessageProcessor:
    async def process_adt_message(self, hl7_message: str):
        """Process Admission/Discharge/Transfer message"""
        message = hl7.parse(hl7_message)
        
        patient_data = self.extract_patient_info(message)
        visit_data = self.extract_visit_info(message)
        
        # Update patient record
        await self.patient_service.update_patient(patient_data)
        
        # Create or update visit
        await self.visit_service.create_visit(visit_data)
```

### FHIR API Compliance
```python
# FHIR R4 resource endpoints
@router.get("/fhir/Patient/{patient_id}")
async def get_patient_fhir(patient_id: str):
    """Get patient in FHIR format"""
    patient = await patient_service.get_patient(patient_id)
    return convert_to_fhir_patient(patient)

@router.post("/fhir/Observation")
async def create_observation_fhir(observation: FHIRObservation):
    """Create lab result observation"""
    return await lab_service.create_observation(observation)
```

## Clinical Decision Support

### Drug Interaction Checking
```python
class DrugInteractionChecker:
    async def check_prescription(
        self, 
        patient_id: str, 
        new_medication: Medication
    ) -> List[DrugInteraction]:
        """Check for drug interactions"""
        current_meds = await medication_service.get_active_medications(patient_id)
        
        interactions = []
        for med in current_meds:
            interaction = await self.drug_database.check_interaction(
                med.ndc_code, 
                new_medication.ndc_code
            )
            if interaction.severity in ["major", "contraindicated"]:
                interactions.append(interaction)
        
        return interactions
```

### Clinical Alerts
```python
class ClinicalAlertSystem:
    async def evaluate_patient_alerts(self, patient_id: str):
        """Evaluate patient for clinical alerts"""
        patient = await patient_service.get_patient(patient_id)
        vitals = await vitals_service.get_latest_vitals(patient_id)
        labs = await lab_service.get_recent_labs(patient_id)
        
        alerts = []
        
        # Critical value alerts
        if vitals.blood_pressure_systolic > 180:
            alerts.append(ClinicalAlert(
                type="critical_vital",
                message="Severe hypertension detected",
                priority="high"
            ))
        
        # Lab value alerts
        if labs.glucose > 400:
            alerts.append(ClinicalAlert(
                type="critical_lab",
                message="Severe hyperglycemia",
                priority="critical"
            ))
        
        return alerts
```

## Medical Billing Integration

### Insurance Verification
```python
class InsuranceVerification:
    async def verify_coverage(
        self, 
        patient_id: str, 
        procedure_codes: List[str]
    ) -> CoverageResult:
        """Verify insurance coverage for procedures"""
        patient = await patient_service.get_patient(patient_id)
        insurance = patient.primary_insurance
        
        # Real-time eligibility check
        eligibility = await self.payer_api.check_eligibility(
            member_id=insurance.member_id,
            payer_id=insurance.payer_id,
            procedure_codes=procedure_codes
        )
        
        return CoverageResult(
            covered_procedures=eligibility.covered,
            copay_amount=eligibility.copay,
            deductible_remaining=eligibility.deductible
        )
```

## Telemedicine Integration

### Video Consultation
```python
class TelemedicineService:
    async def start_consultation(
        self, 
        provider_id: str, 
        patient_id: str
    ) -> ConsultationSession:
        """Start telemedicine session"""
        # Verify appointment exists
        appointment = await appointment_service.get_active_appointment(
            provider_id, patient_id
        )
        
        # Create secure video session
        session = await video_service.create_session(
            participants=[provider_id, patient_id],
            encryption=True,
            recording=True  # For medical records
        )
        
        # Log consultation start
        await audit_logger.log_telemedicine_start(
            provider_id=provider_id,
            patient_id=patient_id,
            session_id=session.id
        )
        
        return session
```

## Configuration

```yaml
# Database encryption
database:
  encryption_key: "your-32-byte-encryption-key"
  connection_encryption: true
  backup_encryption: true

# HIPAA compliance
hipaa:
  audit_logging: true
  access_controls: strict
  session_timeout_minutes: 30
  password_complexity: high

# HL7 integration
hl7:
  listener_port: 2575
  message_types:
    - ADT  # Admission/Discharge/Transfer
    - ORM  # Order messages
    - ORU  # Observation results

# FHIR settings
fhir:
  version: "R4"
  base_url: "/fhir"
  supported_resources:
    - Patient
    - Observation
    - Medication
    - Appointment

# Clinical decision support
cds:
  drug_interaction_checking: true
  allergy_alerts: true
  critical_value_alerts: true
  clinical_reminders: true
```

## Monitoring & Compliance

### Audit Reporting
- User access patterns
- Data modification logs
- Security incident tracking
- Compliance violation alerts

### Performance Metrics
- API response times
- Database query performance
- HL7 message processing rates
- User session analytics

### Security Monitoring
- Failed authentication attempts
- Unusual access patterns
- Data export activities
- Privilege escalation attempts

## Testing

### HIPAA Compliance Testing
```bash
# Test access controls
pytest tests/security/test_access_controls.py -v

# Test audit logging
pytest tests/compliance/test_audit_logging.py -v

# Test data encryption
pytest tests/security/test_encryption.py -v
```

### HL7 Integration Testing
```bash
# Test HL7 message processing
pytest tests/integration/test_hl7_processing.py -v
```

## Deployment

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/ehr_db
ENCRYPTION_KEY=your-32-byte-encryption-key

# Authentication
JWT_SECRET_KEY=your-jwt-secret
OAUTH2_CLIENT_ID=your-oauth2-client-id

# HL7
HL7_LISTENER_PORT=2575
HL7_PROCESSING_ENABLED=true

# FHIR
FHIR_BASE_URL=https://your-domain.com/fhir

# Compliance
HIPAA_AUDIT_ENABLED=true
SESSION_TIMEOUT_MINUTES=30
```

## Extensions

This EHR system can be extended with:
- **AI Diagnostics**: Machine learning for diagnostic assistance
- **Population Health**: Analytics across patient populations
- **Research Integration**: De-identified data for clinical research
- **IoT Device Integration**: Wearable device data collection
- **Genomics**: Genetic data integration and analysis
- **Quality Measures**: CMS quality reporting automation