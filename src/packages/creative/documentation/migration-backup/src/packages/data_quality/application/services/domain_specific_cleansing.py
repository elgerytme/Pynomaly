"""
Domain-Specific Data Cleansing Modules
Specialized cleansing for financial, healthcare, geographic, e-commerce, and customer data domains.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class DomainCleansingResult:
    """Result of domain-specific cleansing operation."""
    
    domain: str
    rules_applied: List[str]
    records_processed: int
    records_modified: int
    quality_improvement: float
    domain_specific_metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


class FinancialDataCleansing:
    """Financial data cleansing with currency normalization, account numbers, and regulatory compliance."""
    
    def __init__(self):
        self.currency_symbols = {
            '$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY', '₹': 'INR',
            'C$': 'CAD', 'A$': 'AUD', 'CHF': 'CHF', '₽': 'RUB'
        }
        
        self.financial_patterns = {
            'iban': re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}$'),
            'swift': re.compile(r'^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$'),
            'credit_card': re.compile(r'^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})$'),
            'account_number': re.compile(r'^[0-9]{8,17}$')
        }
    
    def cleanse_financial_dataset(self, df: pd.DataFrame) -> DomainCleansingResult:
        """Cleanse financial dataset with industry-specific rules."""
        
        logger.info("Starting financial data cleansing")
        
        cleansed_df = df.copy()
        rules_applied = []
        errors = []
        warnings = []
        initial_quality = self._calculate_financial_quality(df)
        
        # Currency normalization
        currency_columns = self._detect_currency_columns(df)
        for col in currency_columns:
            try:
                cleansed_df[col] = self._normalize_currency_values(df[col])
                rules_applied.append(f'currency_normalization_{col}')
            except Exception as e:
                errors.append(f"Currency normalization failed for {col}: {str(e)}")
        
        # Account number standardization
        account_columns = self._detect_account_columns(df)
        for col in account_columns:
            try:
                cleansed_df[col] = self._standardize_account_numbers(df[col])
                rules_applied.append(f'account_standardization_{col}')
            except Exception as e:
                errors.append(f"Account standardization failed for {col}: {str(e)}")
        
        # Financial identifier validation
        identifier_columns = self._detect_financial_identifiers(df)
        for col, identifier_type in identifier_columns.items():
            try:
                valid_mask, invalid_count = self._validate_financial_identifiers(df[col], identifier_type)
                if invalid_count > 0:
                    warnings.append(f"Found {invalid_count} invalid {identifier_type} in column {col}")
                rules_applied.append(f'identifier_validation_{col}')
            except Exception as e:
                errors.append(f"Identifier validation failed for {col}: {str(e)}")
        
        # Regulatory compliance checks
        compliance_issues = self._check_regulatory_compliance(cleansed_df)
        if compliance_issues:
            warnings.extend(compliance_issues)
            rules_applied.append('regulatory_compliance_check')
        
        # Calculate final quality and metrics
        final_quality = self._calculate_financial_quality(cleansed_df)
        quality_improvement = final_quality - initial_quality
        
        records_modified = (df != cleansed_df).any(axis=1).sum()
        
        domain_metrics = {
            'currency_columns_processed': len(currency_columns),
            'account_columns_processed': len(account_columns),
            'financial_identifiers_validated': len(identifier_columns),
            'compliance_issues_found': len(compliance_issues),
            'quality_score': final_quality
        }
        
        logger.info(f"Financial cleansing completed. Quality improvement: {quality_improvement:.2%}")
        
        return DomainCleansingResult(
            domain='financial',
            rules_applied=rules_applied,
            records_processed=len(df),
            records_modified=records_modified,
            quality_improvement=quality_improvement,
            domain_specific_metrics=domain_metrics,
            errors=errors,
            warnings=warnings
        )
    
    def _detect_currency_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns containing currency data."""
        currency_columns = []
        
        for col in df.columns:
            col_name = col.lower()
            if any(indicator in col_name for indicator in ['amount', 'price', 'cost', 'salary', 'revenue', 'profit', 'value']):
                currency_columns.append(col)
            elif df[col].dtype == 'object':
                # Check if values contain currency symbols
                sample_values = df[col].dropna().head(100).astype(str)
                if sample_values.str.contains(r'[$€£¥₹]', regex=True).any():
                    currency_columns.append(col)
        
        return currency_columns
    
    def _normalize_currency_values(self, series: pd.Series) -> pd.Series:
        """Normalize currency values to standard format."""
        
        def normalize_currency(value):
            if pd.isna(value):
                return value
            
            value_str = str(value).strip()
            
            # Extract currency symbol
            currency_code = 'USD'  # Default
            for symbol, code in self.currency_symbols.items():
                if symbol in value_str:
                    currency_code = code
                    break
            
            # Extract numeric value
            numeric_part = re.sub(r'[^\d.,\-]', '', value_str)
            numeric_part = numeric_part.replace(',', '')
            
            try:
                amount = float(numeric_part)
                return f"{currency_code} {amount:.2f}"
            except ValueError:
                return value_str
        
        return series.apply(normalize_currency)
    
    def _detect_account_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns containing account numbers."""
        account_columns = []
        
        for col in df.columns:
            col_name = col.lower()
            if any(indicator in col_name for indicator in ['account', 'acct', 'number', 'id']):
                # Verify it contains account-like patterns
                sample_values = df[col].dropna().head(100).astype(str)
                if sample_values.str.match(r'^[0-9\-\s]+$').mean() > 0.7:
                    account_columns.append(col)
        
        return account_columns
    
    def _standardize_account_numbers(self, series: pd.Series) -> pd.Series:
        """Standardize account numbers by removing spaces and hyphens."""
        
        def standardize_account(value):
            if pd.isna(value):
                return value
            
            account_str = str(value).strip()
            # Remove spaces and hyphens
            standardized = re.sub(r'[\s\-]', '', account_str)
            
            # Pad with zeros if too short for account number
            if standardized.isdigit() and len(standardized) < 8:
                standardized = standardized.zfill(10)
            
            return standardized
        
        return series.apply(standardize_account)
    
    def _detect_financial_identifiers(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect financial identifiers (IBAN, SWIFT, credit cards)."""
        identifier_columns = {}
        
        for col in df.columns:
            col_name = col.lower()
            sample_values = df[col].dropna().head(50).astype(str)
            
            if 'iban' in col_name or sample_values.str.match(self.financial_patterns['iban']).any():
                identifier_columns[col] = 'iban'
            elif 'swift' in col_name or sample_values.str.match(self.financial_patterns['swift']).any():
                identifier_columns[col] = 'swift'
            elif any(indicator in col_name for indicator in ['card', 'credit']) or \
                 sample_values.str.match(self.financial_patterns['credit_card']).any():
                identifier_columns[col] = 'credit_card'
        
        return identifier_columns
    
    def _validate_financial_identifiers(self, series: pd.Series, identifier_type: str) -> Tuple[pd.Series, int]:
        """Validate financial identifiers and return validity mask."""
        
        pattern = self.financial_patterns.get(identifier_type)
        if not pattern:
            return pd.Series([True] * len(series), index=series.index), 0
        
        valid_mask = series.dropna().astype(str).str.match(pattern)
        invalid_count = (~valid_mask).sum()
        
        return valid_mask, invalid_count
    
    def _check_regulatory_compliance(self, df: pd.DataFrame) -> List[str]:
        """Check for potential regulatory compliance issues."""
        issues = []
        
        # Check for potential PII in financial data
        for col in df.columns:
            col_name = col.lower()
            if any(pii_indicator in col_name for pii_indicator in ['ssn', 'social', 'personal', 'private']):
                issues.append(f"Potential PII detected in column {col} - review for compliance")
        
        # Check for unmasked sensitive data
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(100).astype(str)
                if sample_values.str.match(r'^\d{3}-\d{2}-\d{4}$').any():  # SSN pattern
                    issues.append(f"Unmasked SSN pattern detected in column {col}")
        
        return issues
    
    def _calculate_financial_quality(self, df: pd.DataFrame) -> float:
        """Calculate financial data quality score."""
        
        quality_factors = []
        
        # Completeness
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        quality_factors.append(completeness)
        
        # Format consistency for currency columns
        currency_cols = self._detect_currency_columns(df)
        if currency_cols:
            format_consistency = 0
            for col in currency_cols:
                if df[col].dtype == 'object':
                    # Check format consistency
                    formats = df[col].dropna().astype(str).apply(lambda x: re.sub(r'[\d.]', 'N', x)).value_counts()
                    consistency = formats.iloc[0] / formats.sum() if len(formats) > 0 else 1
                    format_consistency += consistency
            quality_factors.append(format_consistency / len(currency_cols))
        
        return np.mean(quality_factors) if quality_factors else 0.5


class HealthcareDataCleansing:
    """Healthcare data cleansing with medical codes, patient identifiers, and HIPAA compliance."""
    
    def __init__(self):
        self.medical_code_patterns = {
            'icd10': re.compile(r'^[A-TV-Z][0-9][A-Z0-9](\.[A-Z0-9]{1,4})?$'),
            'cpt': re.compile(r'^[0-9]{5}$'),
            'ndc': re.compile(r'^[0-9]{4,5}-[0-9]{4}-[0-9]{2}$'),
            'mrn': re.compile(r'^[0-9]{6,10}$')
        }
        
        self.phi_patterns = {
            'ssn': re.compile(r'^\d{3}-\d{2}-\d{4}$'),
            'phone': re.compile(r'^\d{3}-\d{3}-\d{4}$'),
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        }
    
    def cleanse_healthcare_dataset(self, df: pd.DataFrame) -> DomainCleansingResult:
        """Cleanse healthcare dataset with HIPAA compliance and medical code validation."""
        
        logger.info("Starting healthcare data cleansing")
        
        cleansed_df = df.copy()
        rules_applied = []
        errors = []
        warnings = []
        initial_quality = self._calculate_healthcare_quality(df)
        
        # Medical code validation and standardization
        medical_code_columns = self._detect_medical_code_columns(df)
        for col, code_type in medical_code_columns.items():
            try:
                valid_codes, invalid_count = self._validate_medical_codes(df[col], code_type)
                if invalid_count > 0:
                    warnings.append(f"Found {invalid_count} invalid {code_type} codes in column {col}")
                rules_applied.append(f'medical_code_validation_{col}')
            except Exception as e:
                errors.append(f"Medical code validation failed for {col}: {str(e)}")
        
        # PHI detection and masking recommendations
        phi_columns = self._detect_phi_columns(df)
        for col, phi_type in phi_columns.items():
            warnings.append(f"Potential PHI ({phi_type}) detected in column {col} - consider masking")
            rules_applied.append(f'phi_detection_{col}')
        
        # Patient identifier standardization
        patient_id_columns = self._detect_patient_id_columns(df)
        for col in patient_id_columns:
            try:
                cleansed_df[col] = self._standardize_patient_ids(df[col])
                rules_applied.append(f'patient_id_standardization_{col}')
            except Exception as e:
                errors.append(f"Patient ID standardization failed for {col}: {str(e)}")
        
        # HIPAA compliance check
        hipaa_issues = self._check_hipaa_compliance(df)
        if hipaa_issues:
            warnings.extend(hipaa_issues)
            rules_applied.append('hipaa_compliance_check')
        
        # Calculate final quality
        final_quality = self._calculate_healthcare_quality(cleansed_df)
        quality_improvement = final_quality - initial_quality
        
        records_modified = (df != cleansed_df).any(axis=1).sum()
        
        domain_metrics = {
            'medical_code_columns_validated': len(medical_code_columns),
            'phi_columns_detected': len(phi_columns),
            'patient_id_columns_processed': len(patient_id_columns),
            'hipaa_issues_found': len(hipaa_issues),
            'quality_score': final_quality
        }
        
        logger.info(f"Healthcare cleansing completed. Quality improvement: {quality_improvement:.2%}")
        
        return DomainCleansingResult(
            domain='healthcare',
            rules_applied=rules_applied,
            records_processed=len(df),
            records_modified=records_modified,
            quality_improvement=quality_improvement,
            domain_specific_metrics=domain_metrics,
            errors=errors,
            warnings=warnings
        )
    
    def _detect_medical_code_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect columns containing medical codes."""
        code_columns = {}
        
        for col in df.columns:
            col_name = col.lower()
            sample_values = df[col].dropna().head(50).astype(str)
            
            if 'icd' in col_name or sample_values.str.match(self.medical_code_patterns['icd10']).any():
                code_columns[col] = 'icd10'
            elif 'cpt' in col_name or sample_values.str.match(self.medical_code_patterns['cpt']).any():
                code_columns[col] = 'cpt'
            elif 'ndc' in col_name or sample_values.str.match(self.medical_code_patterns['ndc']).any():
                code_columns[col] = 'ndc'
        
        return code_columns
    
    def _validate_medical_codes(self, series: pd.Series, code_type: str) -> Tuple[pd.Series, int]:
        """Validate medical codes against patterns."""
        
        pattern = self.medical_code_patterns.get(code_type)
        if not pattern:
            return pd.Series([True] * len(series), index=series.index), 0
        
        valid_mask = series.dropna().astype(str).str.match(pattern)
        invalid_count = (~valid_mask).sum()
        
        return valid_mask, invalid_count
    
    def _detect_phi_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect columns potentially containing PHI."""
        phi_columns = {}
        
        for col in df.columns:
            col_name = col.lower()
            
            # Check column names for PHI indicators
            if any(indicator in col_name for indicator in ['name', 'address', 'phone', 'email', 'ssn', 'dob']):
                if 'name' in col_name:
                    phi_columns[col] = 'name'
                elif 'address' in col_name:
                    phi_columns[col] = 'address'
                elif 'phone' in col_name:
                    phi_columns[col] = 'phone'
                elif 'email' in col_name:
                    phi_columns[col] = 'email'
                elif 'ssn' in col_name:
                    phi_columns[col] = 'ssn'
                elif 'dob' in col_name:
                    phi_columns[col] = 'date_of_birth'
            
            # Check data patterns
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(50).astype(str)
                
                if sample_values.str.match(self.phi_patterns['ssn']).any():
                    phi_columns[col] = 'ssn'
                elif sample_values.str.match(self.phi_patterns['phone']).any():
                    phi_columns[col] = 'phone'
                elif sample_values.str.match(self.phi_patterns['email']).any():
                    phi_columns[col] = 'email'
        
        return phi_columns
    
    def _detect_patient_id_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect patient identifier columns."""
        patient_id_columns = []
        
        for col in df.columns:
            col_name = col.lower()
            if any(indicator in col_name for indicator in ['patient_id', 'mrn', 'medical_record']):
                patient_id_columns.append(col)
        
        return patient_id_columns
    
    def _standardize_patient_ids(self, series: pd.Series) -> pd.Series:
        """Standardize patient identifiers."""
        
        def standardize_id(value):
            if pd.isna(value):
                return value
            
            id_str = str(value).strip()
            # Remove non-alphanumeric characters
            standardized = re.sub(r'[^A-Za-z0-9]', '', id_str)
            
            # Pad with zeros if numeric and too short
            if standardized.isdigit() and len(standardized) < 8:
                standardized = standardized.zfill(8)
            
            return standardized.upper()
        
        return series.apply(standardize_id)
    
    def _check_hipaa_compliance(self, df: pd.DataFrame) -> List[str]:
        """Check for HIPAA compliance issues."""
        issues = []
        
        # Check for direct identifiers
        phi_columns = self._detect_phi_columns(df)
        if phi_columns:
            issues.append(f"Direct identifiers detected in {len(phi_columns)} columns - ensure proper authorization")
        
        # Check for quasi-identifiers combination
        quasi_identifiers = []
        for col in df.columns:
            col_name = col.lower()
            if any(indicator in col_name for indicator in ['age', 'zip', 'date', 'gender', 'race']):
                quasi_identifiers.append(col)
        
        if len(quasi_identifiers) >= 3:
            issues.append(f"Multiple quasi-identifiers present ({len(quasi_identifiers)}) - risk of re-identification")
        
        return issues
    
    def _calculate_healthcare_quality(self, df: pd.DataFrame) -> float:
        """Calculate healthcare data quality score."""
        
        quality_factors = []
        
        # Basic completeness
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        quality_factors.append(completeness)
        
        # Medical code validity
        medical_codes = self._detect_medical_code_columns(df)
        if medical_codes:
            code_validity = 0
            for col, code_type in medical_codes.items():
                valid_mask, invalid_count = self._validate_medical_codes(df[col], code_type)
                validity_rate = 1 - (invalid_count / len(df[col].dropna()))
                code_validity += validity_rate
            quality_factors.append(code_validity / len(medical_codes))
        
        return np.mean(quality_factors) if quality_factors else 0.5


class GeographicDataCleansing:
    """Geographic data cleansing with coordinates, postal codes, and address standardization."""
    
    def __init__(self):
        self.postal_patterns = {
            'us_zip': re.compile(r'^\d{5}(-\d{4})?$'),
            'ca_postal': re.compile(r'^[A-Z]\d[A-Z] \d[A-Z]\d$'),
            'uk_postal': re.compile(r'^[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}$'),
            'de_postal': re.compile(r'^\d{5}$')
        }
        
        self.coordinate_ranges = {
            'latitude': (-90, 90),
            'longitude': (-180, 180)
        }
    
    def cleanse_geographic_dataset(self, df: pd.DataFrame) -> DomainCleansingResult:
        """Cleanse geographic dataset with coordinate validation and address standardization."""
        
        logger.info("Starting geographic data cleansing")
        
        cleansed_df = df.copy()
        rules_applied = []
        errors = []
        warnings = []
        initial_quality = self._calculate_geographic_quality(df)
        
        # Coordinate validation
        coordinate_columns = self._detect_coordinate_columns(df)
        for col, coord_type in coordinate_columns.items():
            try:
                valid_coords, invalid_count = self._validate_coordinates(df[col], coord_type)
                if invalid_count > 0:
                    warnings.append(f"Found {invalid_count} invalid {coord_type} values in column {col}")
                rules_applied.append(f'coordinate_validation_{col}')
            except Exception as e:
                errors.append(f"Coordinate validation failed for {col}: {str(e)}")
        
        # Postal code standardization
        postal_columns = self._detect_postal_columns(df)
        for col in postal_columns:
            try:
                cleansed_df[col] = self._standardize_postal_codes(df[col])
                rules_applied.append(f'postal_code_standardization_{col}')
            except Exception as e:
                errors.append(f"Postal code standardization failed for {col}: {str(e)}")
        
        # Address standardization
        address_columns = self._detect_address_columns(df)
        for col in address_columns:
            try:
                cleansed_df[col] = self._standardize_addresses(df[col])
                rules_applied.append(f'address_standardization_{col}')
            except Exception as e:
                errors.append(f"Address standardization failed for {col}: {str(e)}")
        
        # Calculate final quality
        final_quality = self._calculate_geographic_quality(cleansed_df)
        quality_improvement = final_quality - initial_quality
        
        records_modified = (df != cleansed_df).any(axis=1).sum()
        
        domain_metrics = {
            'coordinate_columns_validated': len(coordinate_columns),
            'postal_columns_processed': len(postal_columns),
            'address_columns_processed': len(address_columns),
            'quality_score': final_quality
        }
        
        logger.info(f"Geographic cleansing completed. Quality improvement: {quality_improvement:.2%}")
        
        return DomainCleansingResult(
            domain='geographic',
            rules_applied=rules_applied,
            records_processed=len(df),
            records_modified=records_modified,
            quality_improvement=quality_improvement,
            domain_specific_metrics=domain_metrics,
            errors=errors,
            warnings=warnings
        )
    
    def _detect_coordinate_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect coordinate columns (latitude/longitude)."""
        coordinate_columns = {}
        
        for col in df.columns:
            col_name = col.lower()
            
            if any(indicator in col_name for indicator in ['lat', 'latitude']):
                coordinate_columns[col] = 'latitude'
            elif any(indicator in col_name for indicator in ['lng', 'lon', 'longitude']):
                coordinate_columns[col] = 'longitude'
            elif df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Check if values are in coordinate ranges
                values = df[col].dropna()
                if len(values) > 0:
                    min_val, max_val = values.min(), values.max()
                    if -90 <= min_val and max_val <= 90:
                        coordinate_columns[col] = 'latitude'
                    elif -180 <= min_val and max_val <= 180:
                        coordinate_columns[col] = 'longitude'
        
        return coordinate_columns
    
    def _validate_coordinates(self, series: pd.Series, coord_type: str) -> Tuple[pd.Series, int]:
        """Validate coordinate values are within valid ranges."""
        
        valid_range = self.coordinate_ranges.get(coord_type, (-180, 180))
        
        numeric_values = pd.to_numeric(series, errors='coerce')
        valid_mask = (numeric_values >= valid_range[0]) & (numeric_values <= valid_range[1])
        invalid_count = (~valid_mask).sum()
        
        return valid_mask, invalid_count
    
    def _detect_postal_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect postal code columns."""
        postal_columns = []
        
        for col in df.columns:
            col_name = col.lower()
            if any(indicator in col_name for indicator in ['zip', 'postal', 'postcode']):
                postal_columns.append(col)
        
        return postal_columns
    
    def _standardize_postal_codes(self, series: pd.Series) -> pd.Series:
        """Standardize postal codes based on detected country format."""
        
        def standardize_postal(value):
            if pd.isna(value):
                return value
            
            postal_str = str(value).strip().upper()
            
            # Try different postal code formats
            if self.postal_patterns['us_zip'].match(postal_str):
                return postal_str
            elif self.postal_patterns['ca_postal'].match(postal_str):
                # Ensure proper spacing for Canadian postal codes
                if len(postal_str) == 6:
                    return f"{postal_str[:3]} {postal_str[3:]}"
                return postal_str
            elif self.postal_patterns['uk_postal'].match(postal_str):
                return postal_str
            elif self.postal_patterns['de_postal'].match(postal_str):
                return postal_str
            else:
                # Generic cleaning - remove special characters
                cleaned = re.sub(r'[^A-Z0-9\s]', '', postal_str)
                return cleaned
        
        return series.apply(standardize_postal)
    
    def _detect_address_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect address columns."""
        address_columns = []
        
        for col in df.columns:
            col_name = col.lower()
            if any(indicator in col_name for indicator in ['address', 'street', 'location']):
                address_columns.append(col)
        
        return address_columns
    
    def _standardize_addresses(self, series: pd.Series) -> pd.Series:
        """Standardize address formats."""
        
        def standardize_address(value):
            if pd.isna(value):
                return value
            
            address = str(value).strip()
            
            # Standardize common abbreviations
            abbreviations = {
                r'\bStreet\b': 'St',
                r'\bAvenue\b': 'Ave',
                r'\bBoulevard\b': 'Blvd',
                r'\bRoad\b': 'Rd',
                r'\bDrive\b': 'Dr',
                r'\bLane\b': 'Ln',
                r'\bCourt\b': 'Ct',
                r'\bApartment\b': 'Apt',
                r'\bSuite\b': 'Ste',
                r'\bNorth\b': 'N',
                r'\bSouth\b': 'S',
                r'\bEast\b': 'E',
                r'\bWest\b': 'W'
            }
            
            for pattern, replacement in abbreviations.items():
                address = re.sub(pattern, replacement, address, flags=re.IGNORECASE)
            
            # Title case
            address = address.title()
            
            # Clean extra spaces
            address = re.sub(r'\s+', ' ', address)
            
            return address
        
        return series.apply(standardize_address)
    
    def _calculate_geographic_quality(self, df: pd.DataFrame) -> float:
        """Calculate geographic data quality score."""
        
        quality_factors = []
        
        # Basic completeness
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        quality_factors.append(completeness)
        
        # Coordinate validity
        coordinate_columns = self._detect_coordinate_columns(df)
        if coordinate_columns:
            coord_validity = 0
            for col, coord_type in coordinate_columns.items():
                valid_mask, invalid_count = self._validate_coordinates(df[col], coord_type)
                validity_rate = 1 - (invalid_count / len(df[col].dropna()))
                coord_validity += validity_rate
            quality_factors.append(coord_validity / len(coordinate_columns))
        
        return np.mean(quality_factors) if quality_factors else 0.5


class DomainSpecificCleansingOrchestrator:
    """Orchestrator for domain-specific data cleansing operations."""
    
    def __init__(self):
        self.domain_cleansers = {
            'financial': FinancialDataCleansing(),
            'healthcare': HealthcareDataCleansing(),
            'geographic': GeographicDataCleansing()
        }
        
        logger.info("Initialized DomainSpecificCleansingOrchestrator")
    
    def cleanse_by_domain(self, df: pd.DataFrame, domain: str) -> DomainCleansingResult:
        """Perform domain-specific cleansing."""
        
        if domain not in self.domain_cleansers:
            raise ValueError(f"Unsupported domain: {domain}. Available domains: {list(self.domain_cleansers.keys())}")
        
        cleanser = self.domain_cleansers[domain]
        
        if domain == 'financial':
            return cleanser.cleanse_financial_dataset(df)
        elif domain == 'healthcare':
            return cleanser.cleanse_healthcare_dataset(df)
        elif domain == 'geographic':
            return cleanser.cleanse_geographic_dataset(df)
        else:
            raise ValueError(f"Domain {domain} not implemented")
    
    def detect_domain(self, df: pd.DataFrame) -> str:
        """Auto-detect the most likely domain for the dataset."""
        
        domain_scores = {}
        
        # Financial domain indicators
        financial_indicators = ['amount', 'price', 'cost', 'salary', 'revenue', 'account', 'transaction']
        financial_score = sum(1 for col in df.columns if any(ind in col.lower() for ind in financial_indicators))
        domain_scores['financial'] = financial_score
        
        # Healthcare domain indicators
        healthcare_indicators = ['patient', 'medical', 'diagnosis', 'treatment', 'icd', 'cpt', 'mrn']
        healthcare_score = sum(1 for col in df.columns if any(ind in col.lower() for ind in healthcare_indicators))
        domain_scores['healthcare'] = healthcare_score
        
        # Geographic domain indicators
        geographic_indicators = ['address', 'location', 'coordinates', 'lat', 'lng', 'zip', 'postal']
        geographic_score = sum(1 for col in df.columns if any(ind in col.lower() for ind in geographic_indicators))
        domain_scores['geographic'] = geographic_score
        
        # Return domain with highest score
        if max(domain_scores.values()) == 0:
            return 'general'
        
        return max(domain_scores, key=domain_scores.get)
    
    def get_supported_domains(self) -> List[str]:
        """Get list of supported domains."""
        return list(self.domain_cleansers.keys())
    
    def add_domain_cleanser(self, domain: str, cleanser):
        """Add custom domain cleanser."""
        self.domain_cleansers[domain] = cleanser
        logger.info(f"Added domain cleanser for: {domain}")