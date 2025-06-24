#!/usr/bin/env python3
"""
Healthcare Data Preprocessing Pipeline Template

This template provides a comprehensive preprocessing pipeline specifically designed
for healthcare data including medical records, patient data, and clinical measurements.
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Data processing imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# Medical data specific imports
from scipy import stats
from scipy.stats import zscore, chi2_contingency
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthcareDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for healthcare datasets.
    
    Features:
    - Medical data validation and quality assessment
    - HIPAA-compliant data handling
    - Clinical measurement normalization
    - Medical code standardization
    - Age and demographic processing
    - Temporal health event processing
    - Risk factor calculation
    - Clinical decision support preparation
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 preserve_original: bool = True,
                 verbose: bool = True,
                 anonymize: bool = True):
        """
        Initialize the healthcare data preprocessor.
        
        Args:
            config: Configuration dictionary for preprocessing steps
            preserve_original: Whether to preserve original column values
            verbose: Enable detailed logging
            anonymize: Enable HIPAA-compliant anonymization
        """
        self.config = config or self._get_default_config()
        self.preserve_original = preserve_original
        self.verbose = verbose
        self.anonymize = anonymize
        
        # Initialize preprocessing components
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}
        
        # Healthcare-specific components
        self.medical_code_mappings = {}
        self.vital_sign_ranges = {}
        self.lab_value_ranges = {}
        self.anonymization_mappings = {}
        
        # Metadata tracking
        self.preprocessing_steps = []
        self.data_profile = {}
        self.clinical_metadata = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for healthcare data preprocessing."""
        return {
            'anonymization': {
                'remove_identifiers': True,
                'hash_ids': True,
                'age_binning': True,
                'date_shifting': True,
                'location_generalization': True
            },
            'medical_validation': {
                'vital_signs_ranges': True,
                'lab_values_ranges': True,
                'medication_validation': True,
                'icd_code_validation': True,
                'age_validation': True
            },
            'missing_values': {
                'strategy': 'iterative',  # 'iterative', 'knn', 'medical_mean'
                'threshold': 0.4,  # Drop features with >40% missing
                'medical_imputation': True,  # Use medical knowledge for imputation
                'preserve_na_indicators': True
            },
            'outliers': {
                'method': 'medical_ranges',  # 'medical_ranges', 'iqr', 'zscore'
                'vital_signs_handling': 'cap',  # 'cap', 'flag', 'remove'
                'lab_values_handling': 'flag',  # Flag unusual lab values
                'use_reference_ranges': True
            },
            'feature_engineering': {
                'age_groups': True,
                'bmi_calculation': True,
                'vital_signs_ratios': True,
                'lab_value_ratios': True,
                'temporal_features': True,
                'comorbidity_scores': True,
                'medication_interactions': True
            },
            'encoding': {
                'icd_codes': 'hierarchical',  # 'hierarchical', 'one_hot', 'embedding'
                'medications': 'drug_class',  # 'drug_class', 'one_hot', 'similarity'
                'categorical_threshold': 20,
                'rare_category_threshold': 0.01
            },
            'scaling': {
                'method': 'robust',  # Robust to outliers in medical data
                'per_measurement_type': True,  # Scale vital signs, labs separately
                'preserve_interpretability': True
            },
            'quality_assessment': {
                'data_completeness': True,
                'temporal_consistency': True,
                'clinical_plausibility': True,
                'duplicate_detection': True
            }
        }
    
    def preprocess(self, 
                  data: pd.DataFrame,
                  patient_id_col: str = 'patient_id',
                  medical_columns: Dict[str, List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply comprehensive preprocessing pipeline to healthcare data.
        
        Args:
            data: Input DataFrame
            patient_id_col: Name of patient ID column
            medical_columns: Dict mapping column types to column names
                            {'vital_signs': [...], 'lab_values': [...], 'medications': [...], etc.}
            
        Returns:
            Tuple of (processed_data, preprocessing_metadata)
        """
        logger.info("Starting healthcare data preprocessing pipeline")
        
        # Create copy to avoid modifying original
        df = data.copy()
        original_shape = df.shape
        
        # Auto-detect medical column types if not provided
        if medical_columns is None:
            medical_columns = self._auto_detect_medical_columns(df)
        
        # 1. HIPAA Compliance and Anonymization
        if self.anonymize:
            self._log_step("HIPAA Compliance and Data Anonymization")
            df = self._anonymize_data(df, patient_id_col)
        
        # 2. Medical Data Validation
        self._log_step("Medical Data Validation and Quality Assessment")
        validation_results = self._validate_medical_data(df, medical_columns)
        
        # 3. Handle Missing Values with Medical Context
        self._log_step("Medical-Aware Missing Value Treatment")
        df = self._handle_missing_values(df, medical_columns)
        
        # 4. Outlier Detection and Treatment
        self._log_step("Medical Outlier Detection and Treatment")
        df = self._handle_medical_outliers(df, medical_columns)
        
        # 5. Medical Feature Engineering
        self._log_step("Medical Feature Engineering")
        df = self._engineer_medical_features(df, medical_columns)
        
        # 6. Medical Code Processing
        self._log_step("Medical Code Processing and Standardization")
        df = self._process_medical_codes(df, medical_columns)
        
        # 7. Categorical Encoding
        self._log_step("Medical Categorical Encoding")
        df = self._encode_medical_categories(df, medical_columns)
        
        # 8. Feature Scaling
        self._log_step("Medical Feature Scaling")
        df = self._scale_medical_features(df, medical_columns)
        
        # 9. Feature Selection
        self._log_step("Clinical Feature Selection")
        df = self._select_clinical_features(df)
        
        # 10. Final Clinical Validation
        self._log_step("Final Clinical Validation")
        final_validation = self._final_clinical_validation(df, original_shape)
        
        # Prepare metadata
        metadata = {
            'preprocessing_steps': self.preprocessing_steps,
            'data_profile': self.data_profile,
            'clinical_metadata': self.clinical_metadata,
            'validation_results': validation_results,
            'final_validation': final_validation,
            'original_shape': original_shape,
            'final_shape': df.shape,
            'config': self.config,
            'medical_columns': medical_columns,
            'anonymization_applied': self.anonymize
        }
        
        logger.info(f"Preprocessing complete: {original_shape} -> {df.shape}")
        return df, metadata
    
    def _auto_detect_medical_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Auto-detect medical column types based on column names and patterns."""
        medical_columns = {
            'vital_signs': [],
            'lab_values': [],
            'medications': [],
            'diagnoses': [],
            'demographics': [],
            'temporal': [],
            'clinical_notes': []
        }
        
        # Define patterns for different medical data types
        patterns = {
            'vital_signs': ['blood_pressure', 'heart_rate', 'temperature', 'weight', 'height', 
                           'bp_', 'hr_', 'temp_', 'pulse', 'respiration', 'oxygen', 'spo2'],
            'lab_values': ['glucose', 'cholesterol', 'hemoglobin', 'creatinine', 'sodium', 'potassium',
                          'lab_', 'test_', 'result_', 'value_', 'level_', 'count_'],
            'medications': ['medication', 'drug', 'prescription', 'dosage', 'med_', 'rx_'],
            'diagnoses': ['diagnosis', 'icd', 'condition', 'disease', 'disorder', 'dx_'],
            'demographics': ['age', 'gender', 'sex', 'race', 'ethnicity', 'insurance'],
            'temporal': ['date', 'time', 'admission', 'discharge', 'visit'],
            'clinical_notes': ['note', 'comment', 'observation', 'description', 'narrative']
        }
        
        for col in df.columns:
            col_lower = col.lower()
            for category, keywords in patterns.items():
                if any(keyword in col_lower for keyword in keywords):
                    medical_columns[category].append(col)
                    break
        
        self.clinical_metadata['detected_columns'] = medical_columns
        return medical_columns
    
    def _anonymize_data(self, df: pd.DataFrame, patient_id_col: str) -> pd.DataFrame:
        """Apply HIPAA-compliant anonymization to healthcare data."""
        anonymization_config = self.config['anonymization']
        anonymization_log = {}
        
        # Hash patient IDs
        if anonymization_config['hash_ids'] and patient_id_col in df.columns:
            df[patient_id_col] = df[patient_id_col].apply(
                lambda x: abs(hash(str(x))) % (10**8)
            )
            anonymization_log['patient_ids'] = 'hashed'
        
        # Remove or generalize identifiers
        if anonymization_config['remove_identifiers']:
            identifier_patterns = ['ssn', 'social', 'phone', 'email', 'address', 'name']
            identifier_cols = []
            
            for col in df.columns:
                if any(pattern in col.lower() for pattern in identifier_patterns):
                    identifier_cols.append(col)
            
            if identifier_cols:
                df = df.drop(columns=identifier_cols)
                anonymization_log['removed_identifiers'] = identifier_cols
        
        # Age binning
        if anonymization_config['age_binning']:
            age_columns = [col for col in df.columns if 'age' in col.lower()]
            for col in age_columns:
                if col in df.columns:
                    df[f"{col}_group"] = pd.cut(
                        df[col], 
                        bins=[0, 18, 30, 50, 65, 80, 100], 
                        labels=['<18', '18-29', '30-49', '50-64', '65-79', '80+']
                    )
                    anonymization_log[f'{col}_binning'] = 'applied'
        
        # Date shifting (shift dates by random offset while preserving intervals)
        if anonymization_config['date_shifting']:
            date_columns = df.select_dtypes(include=['datetime64']).columns
            if len(date_columns) > 0:
                # Generate random shift (same for all dates to preserve intervals)
                shift_days = np.random.randint(-365, 365)
                for col in date_columns:
                    df[col] = df[col] + pd.Timedelta(days=shift_days)
                anonymization_log['date_shifting'] = f'{shift_days}_days'
        
        # Location generalization
        if anonymization_config['location_generalization']:
            location_columns = [col for col in df.columns if any(
                loc in col.lower() for loc in ['zip', 'postal', 'city', 'state', 'county']
            )]
            for col in location_columns:
                if col in df.columns:
                    # Generalize to state/region level only
                    if 'zip' in col.lower() or 'postal' in col.lower():
                        # Keep only first 3 digits of ZIP
                        df[col] = df[col].astype(str).str[:3] + 'XX'
                    anonymization_log[f'{col}_generalization'] = 'applied'
        
        self.preprocessing_steps.append({
            'step': 'hipaa_anonymization',
            'anonymization_applied': anonymization_log
        })
        
        return df
    
    def _validate_medical_data(self, df: pd.DataFrame, medical_columns: Dict[str, List[str]]) -> Dict[str, Any]:
        """Validate medical data for clinical plausibility and quality."""
        validation_config = self.config['medical_validation']
        validation_results = {
            'total_patients': len(df),
            'data_quality_score': 0.0,
            'clinical_issues': [],
            'validation_details': {}
        }
        
        # Vital signs validation
        if validation_config['vital_signs_ranges']:
            vital_ranges = {
                'heart_rate': (40, 200),
                'systolic_bp': (70, 250),
                'diastolic_bp': (40, 150),
                'temperature': (95, 110),  # Fahrenheit
                'weight': (50, 500),  # pounds
                'height': (48, 84)  # inches
            }
            
            for col in medical_columns['vital_signs']:
                if col in df.columns:
                    col_lower = col.lower()
                    for vital_type, (min_val, max_val) in vital_ranges.items():
                        if vital_type in col_lower:
                            invalid_values = ((df[col] < min_val) | (df[col] > max_val)).sum()
                            if invalid_values > 0:
                                validation_results['clinical_issues'].append({
                                    'type': 'invalid_vital_signs',
                                    'column': col,
                                    'count': invalid_values,
                                    'severity': 'high'
                                })
                            break
        
        # Lab values validation
        if validation_config['lab_values_ranges']:
            lab_ranges = {
                'glucose': (70, 400),  # mg/dL
                'cholesterol': (100, 400),  # mg/dL
                'hemoglobin': (8, 20),  # g/dL
                'creatinine': (0.5, 10),  # mg/dL
                'sodium': (130, 150),  # mEq/L
                'potassium': (3.0, 6.0)  # mEq/L
            }
            
            for col in medical_columns['lab_values']:
                if col in df.columns:
                    col_lower = col.lower()
                    for lab_type, (min_val, max_val) in lab_ranges.items():
                        if lab_type in col_lower:
                            invalid_values = ((df[col] < min_val) | (df[col] > max_val)).sum()
                            if invalid_values > 0:
                                validation_results['clinical_issues'].append({
                                    'type': 'invalid_lab_values',
                                    'column': col,
                                    'count': invalid_values,
                                    'severity': 'medium'
                                })
                            break
        
        # Age validation
        if validation_config['age_validation']:
            age_columns = [col for col in df.columns if 'age' in col.lower()]
            for col in age_columns:
                if col in df.columns:
                    invalid_ages = ((df[col] < 0) | (df[col] > 120)).sum()
                    if invalid_ages > 0:
                        validation_results['clinical_issues'].append({
                            'type': 'invalid_ages',
                            'column': col,
                            'count': invalid_ages,
                            'severity': 'high'
                        })
        
        # Calculate overall data quality score
        total_issues = sum(issue['count'] for issue in validation_results['clinical_issues'])
        total_data_points = len(df) * len(df.columns)
        validation_results['data_quality_score'] = max(0, 1 - (total_issues / total_data_points))
        
        self.data_profile['validation'] = validation_results
        return validation_results
    
    def _handle_missing_values(self, df: pd.DataFrame, medical_columns: Dict[str, List[str]]) -> pd.DataFrame:
        """Handle missing values with medical domain knowledge."""
        strategy = self.config['missing_values']['strategy']
        threshold = self.config['missing_values']['threshold']
        
        # Track missing value patterns
        missing_patterns = {}
        
        # Drop columns with too many missing values
        missing_ratios = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratios[missing_ratios > threshold].index.tolist()
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            missing_patterns['dropped_columns'] = cols_to_drop
        
        # Create missing value indicators for important medical measurements
        if self.config['missing_values']['preserve_na_indicators']:
            important_medical_cols = (medical_columns['vital_signs'] + 
                                    medical_columns['lab_values'])
            
            for col in important_medical_cols:
                if col in df.columns:
                    df[f"{col}_missing"] = df[col].isnull().astype(int)
        
        # Apply domain-specific imputation
        if strategy == 'iterative':
            # Use iterative imputation for medical data
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) > 1:
                imputer = IterativeImputer(
                    max_iter=10, 
                    random_state=42,
                    skip_complete=True
                )
                df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
                self.imputers['iterative'] = imputer
                missing_patterns['iterative_imputation'] = len(numeric_columns)
        
        elif strategy == 'medical_mean':
            # Use medical reference values for imputation
            medical_reference_values = {
                'heart_rate': 72,
                'systolic_bp': 120,
                'diastolic_bp': 80,
                'temperature': 98.6,
                'glucose': 100,
                'cholesterol': 200
            }
            
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64'] and df[col].isnull().any():
                    col_lower = col.lower()
                    
                    # Check if this is a known medical measurement
                    reference_value = None
                    for med_type, ref_val in medical_reference_values.items():
                        if med_type in col_lower:
                            reference_value = ref_val
                            break
                    
                    if reference_value:
                        df[col] = df[col].fillna(reference_value)
                        missing_patterns[col] = f'medical_reference_{reference_value}'
                    else:
                        # Use median for unknown medical measurements
                        df[col] = df[col].fillna(df[col].median())
                        missing_patterns[col] = 'median'
        
        # Handle categorical missing values
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna('Unknown')
                missing_patterns[col] = 'unknown_category'
        
        self.preprocessing_steps.append({
            'step': 'medical_missing_value_treatment',
            'strategy': strategy,
            'missing_patterns': missing_patterns
        })
        
        return df
    
    def _handle_medical_outliers(self, df: pd.DataFrame, medical_columns: Dict[str, List[str]]) -> pd.DataFrame:
        """Handle outliers in medical data using clinical knowledge."""
        method = self.config['outliers']['method']
        outlier_info = {}
        
        if method == 'medical_ranges':
            # Use established medical reference ranges
            medical_ranges = {
                'heart_rate': (50, 180),
                'systolic_bp': (80, 200),
                'diastolic_bp': (50, 120),
                'temperature': (96, 104),
                'glucose': (70, 300),
                'cholesterol': (120, 350),
                'hemoglobin': (10, 18)
            }
            
            # Process vital signs
            for col in medical_columns['vital_signs']:
                if col in df.columns:
                    col_lower = col.lower()
                    
                    for measure, (min_val, max_val) in medical_ranges.items():
                        if measure in col_lower:
                            outliers = (df[col] < min_val) | (df[col] > max_val)
                            outlier_count = outliers.sum()
                            
                            if outlier_count > 0:
                                handling = self.config['outliers']['vital_signs_handling']
                                
                                if handling == 'cap':
                                    df.loc[df[col] < min_val, col] = min_val
                                    df.loc[df[col] > max_val, col] = max_val
                                elif handling == 'flag':
                                    df[f"{col}_outlier"] = outliers.astype(int)
                                elif handling == 'remove':
                                    df = df[~outliers]
                                
                                outlier_info[col] = {
                                    'count': outlier_count,
                                    'handling': handling,
                                    'reference_range': (min_val, max_val)
                                }
                            break
            
            # Process lab values
            for col in medical_columns['lab_values']:
                if col in df.columns:
                    col_lower = col.lower()
                    
                    for measure, (min_val, max_val) in medical_ranges.items():
                        if measure in col_lower:
                            outliers = (df[col] < min_val) | (df[col] > max_val)
                            outlier_count = outliers.sum()
                            
                            if outlier_count > 0:
                                handling = self.config['outliers']['lab_values_handling']
                                
                                if handling == 'flag':
                                    df[f"{col}_abnormal"] = outliers.astype(int)
                                elif handling == 'cap':
                                    df.loc[df[col] < min_val, col] = min_val
                                    df.loc[df[col] > max_val, col] = max_val
                                
                                outlier_info[col] = {
                                    'count': outlier_count,
                                    'handling': handling,
                                    'reference_range': (min_val, max_val)
                                }
                            break
        
        self.preprocessing_steps.append({
            'step': 'medical_outlier_treatment',
            'method': method,
            'outliers_handled': outlier_info
        })
        
        return df
    
    def _engineer_medical_features(self, df: pd.DataFrame, medical_columns: Dict[str, List[str]]) -> pd.DataFrame:
        """Engineer healthcare-specific features."""
        feature_config = self.config['feature_engineering']
        new_features = []
        
        # Age groups
        if feature_config['age_groups']:
            age_columns = [col for col in df.columns if 'age' in col.lower()]
            for col in age_columns:
                if col in df.columns:
                    df[f"{col}_pediatric"] = (df[col] < 18).astype(int)
                    df[f"{col}_geriatric"] = (df[col] >= 65).astype(int)
                    df[f"{col}_adult"] = ((df[col] >= 18) & (df[col] < 65)).astype(int)
                    new_features.extend([f"{col}_pediatric", f"{col}_geriatric", f"{col}_adult"])
        
        # BMI calculation
        if feature_config['bmi_calculation']:
            weight_cols = [col for col in df.columns if 'weight' in col.lower()]
            height_cols = [col for col in df.columns if 'height' in col.lower()]
            
            if weight_cols and height_cols:
                weight_col = weight_cols[0]
                height_col = height_cols[0]
                
                # Convert to metric if needed and calculate BMI
                # Assuming weight in pounds and height in inches
                df['bmi'] = (df[weight_col] * 0.453592) / ((df[height_col] * 0.0254) ** 2)
                
                # BMI categories
                df['bmi_underweight'] = (df['bmi'] < 18.5).astype(int)
                df['bmi_normal'] = ((df['bmi'] >= 18.5) & (df['bmi'] < 25)).astype(int)
                df['bmi_overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
                df['bmi_obese'] = (df['bmi'] >= 30).astype(int)
                
                new_features.extend(['bmi', 'bmi_underweight', 'bmi_normal', 'bmi_overweight', 'bmi_obese'])
        
        # Vital signs ratios and derived features
        if feature_config['vital_signs_ratios']:
            # Blood pressure ratios
            systolic_cols = [col for col in df.columns if 'systolic' in col.lower()]
            diastolic_cols = [col for col in df.columns if 'diastolic' in col.lower()]
            
            if systolic_cols and diastolic_cols:
                systolic_col = systolic_cols[0]
                diastolic_col = diastolic_cols[0]
                
                df['pulse_pressure'] = df[systolic_col] - df[diastolic_col]
                df['mean_arterial_pressure'] = df[diastolic_col] + (df['pulse_pressure'] / 3)
                
                new_features.extend(['pulse_pressure', 'mean_arterial_pressure'])
        
        # Lab value ratios
        if feature_config['lab_value_ratios']:
            # Common clinical ratios
            if 'sodium' in df.columns and 'potassium' in df.columns:
                df['sodium_potassium_ratio'] = df['sodium'] / df['potassium']
                new_features.append('sodium_potassium_ratio')
            
            if 'glucose' in df.columns and 'hemoglobin' in df.columns:
                df['glucose_hemoglobin_ratio'] = df['glucose'] / df['hemoglobin']
                new_features.append('glucose_hemoglobin_ratio')
        
        # Temporal features
        if feature_config['temporal_features']:
            date_columns = df.select_dtypes(include=['datetime64']).columns
            
            for col in date_columns:
                if 'admission' in col.lower() or 'visit' in col.lower():
                    df[f"{col}_hour"] = df[col].dt.hour
                    df[f"{col}_day_of_week"] = df[col].dt.dayofweek
                    df[f"{col}_is_weekend"] = (df[col].dt.dayofweek >= 5).astype(int)
                    df[f"{col}_is_night"] = ((df[col].dt.hour < 6) | (df[col].dt.hour > 22)).astype(int)
                    
                    new_features.extend([
                        f"{col}_hour", f"{col}_day_of_week", 
                        f"{col}_is_weekend", f"{col}_is_night"
                    ])
        
        # Comorbidity scores (simplified)
        if feature_config['comorbidity_scores']:
            # Create simple comorbidity count from diagnosis columns
            diagnosis_cols = medical_columns['diagnoses']
            if diagnosis_cols:
                # Count non-null diagnoses as comorbidity indicator
                comorbidity_count = 0
                for col in diagnosis_cols:
                    if col in df.columns:
                        comorbidity_count += (~df[col].isnull()).astype(int)
                
                if isinstance(comorbidity_count, int) and comorbidity_count == 0:
                    df['comorbidity_count'] = 0
                else:
                    df['comorbidity_count'] = comorbidity_count
                
                df['high_comorbidity'] = (df['comorbidity_count'] >= 3).astype(int)
                new_features.extend(['comorbidity_count', 'high_comorbidity'])
        
        self.preprocessing_steps.append({
            'step': 'medical_feature_engineering',
            'new_features': new_features,
            'feature_count': len(new_features)
        })
        
        return df
    
    def _process_medical_codes(self, df: pd.DataFrame, medical_columns: Dict[str, List[str]]) -> pd.DataFrame:
        """Process and standardize medical codes."""
        encoding_config = self.config['encoding']
        
        # Process ICD codes
        icd_columns = medical_columns['diagnoses']
        for col in icd_columns:
            if col in df.columns:
                # Extract ICD code categories (first 3 characters)
                df[f"{col}_category"] = df[col].astype(str).str[:3]
                
                # Create binary indicators for common conditions
                common_icd_patterns = {
                    'diabetes': ['250', 'E10', 'E11'],
                    'hypertension': ['401', 'I10', 'I11'],
                    'heart_disease': ['410', 'I20', 'I21', 'I25'],
                    'respiratory': ['490', 'J44', 'J45'],
                    'mental_health': ['296', 'F32', 'F33']
                }
                
                for condition, codes in common_icd_patterns.items():
                    df[f"{col}_{condition}"] = df[f"{col}_category"].isin(codes).astype(int)
        
        # Process medication codes/names
        medication_columns = medical_columns['medications']
        for col in medication_columns:
            if col in df.columns:
                # Create drug class categories (simplified)
                drug_classes = {
                    'antihypertensive': ['lisinopril', 'amlodipine', 'metoprolol', 'losartan'],
                    'diabetes': ['metformin', 'insulin', 'glipizide', 'glyburide'],
                    'cardiac': ['aspirin', 'warfarin', 'digoxin', 'furosemide'],
                    'antibiotic': ['amoxicillin', 'azithromycin', 'ciprofloxacin'],
                    'analgesic': ['acetaminophen', 'ibuprofen', 'morphine']
                }
                
                for drug_class, medications in drug_classes.items():
                    df[f"{col}_{drug_class}"] = df[col].astype(str).str.lower().str.contains(
                        '|'.join(medications), na=False
                    ).astype(int)
        
        self.preprocessing_steps.append({
            'step': 'medical_code_processing',
            'icd_columns_processed': len(icd_columns),
            'medication_columns_processed': len(medication_columns)
        })
        
        return df
    
    def _encode_medical_categories(self, df: pd.DataFrame, medical_columns: Dict[str, List[str]]) -> pd.DataFrame:
        """Encode categorical variables with medical context."""
        threshold = self.config['encoding']['categorical_threshold']
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        encoding_info = {}
        
        for col in categorical_columns:
            unique_count = df[col].nunique()
            
            if unique_count <= threshold:
                # One-hot encode low cardinality categories
                encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
                encoded_data = encoder.fit_transform(df[[col]])
                
                feature_names = [f'{col}_{cat}' for cat in encoder.categories_[0][1:]]
                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
                
                df = pd.concat([df, encoded_df], axis=1)
                df = df.drop(columns=[col])
                
                self.encoders[col] = encoder
                encoding_info[col] = f'one_hot_{len(feature_names)}_features'
            else:
                # Label encode high cardinality categories
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                
                self.encoders[col] = encoder
                encoding_info[col] = 'label_encoding'
        
        self.preprocessing_steps.append({
            'step': 'medical_categorical_encoding',
            'encoding_methods': encoding_info
        })
        
        return df
    
    def _scale_medical_features(self, df: pd.DataFrame, medical_columns: Dict[str, List[str]]) -> pd.DataFrame:
        """Scale features with medical measurement considerations."""
        method = self.config['scaling']['method']
        per_measurement_type = self.config['scaling']['per_measurement_type']
        
        if per_measurement_type:
            # Scale different types of medical measurements separately
            measurement_groups = {
                'vital_signs': medical_columns['vital_signs'],
                'lab_values': medical_columns['lab_values'],
                'derived_features': [col for col in df.columns if any(
                    keyword in col.lower() for keyword in ['bmi', 'ratio', 'pressure', 'score']
                )]
            }
            
            scaling_info = {}
            
            for group_name, feature_list in measurement_groups.items():
                # Find actual columns that exist and are numeric
                group_columns = [col for col in feature_list 
                               if col in df.columns and df[col].dtype in ['int64', 'float64']]
                
                if len(group_columns) > 0:
                    if method == 'standard':
                        scaler = StandardScaler()
                    elif method == 'minmax':
                        scaler = MinMaxScaler()
                    elif method == 'robust':
                        scaler = RobustScaler()
                    
                    df[group_columns] = scaler.fit_transform(df[group_columns])
                    self.scalers[group_name] = scaler
                    scaling_info[group_name] = len(group_columns)
            
            self.preprocessing_steps.append({
                'step': 'medical_feature_scaling_by_type',
                'method': method,
                'measurement_groups': scaling_info
            })
        else:
            # Scale all numeric features together
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) > 0:
                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                elif method == 'robust':
                    scaler = RobustScaler()
                
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                self.scalers['all'] = scaler
                
                self.preprocessing_steps.append({
                    'step': 'medical_feature_scaling_global',
                    'method': method,
                    'features_scaled': len(numeric_columns)
                })
        
        return df
    
    def _select_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select clinically relevant features for anomaly detection."""
        original_features = len(df.columns)
        
        # Remove low variance features
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            selector = VarianceThreshold(threshold=0.01)
            selected_features = selector.fit_transform(numeric_df)
            
            selected_columns = numeric_df.columns[selector.get_support()]
            non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
            
            df = df[list(non_numeric_columns) + list(selected_columns)]
            self.feature_selectors['variance'] = selector
        
        # Remove highly correlated features (but keep clinically important ones)
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            
            # Preserve important clinical features
            important_features = [col for col in numeric_df.columns if any(
                keyword in col.lower() for keyword in [
                    'age', 'bmi', 'blood_pressure', 'heart_rate', 'glucose', 
                    'comorbidity', 'abnormal', 'missing'
                ]
            )]
            
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = []
            for column in upper_triangle.columns:
                if any(upper_triangle[column] > 0.9):
                    if column not in important_features:
                        to_drop.append(column)
            
            df = df.drop(columns=to_drop)
            
            if to_drop:
                self.preprocessing_steps.append({
                    'step': 'clinical_feature_selection',
                    'removed_features': len(to_drop),
                    'preserved_important_features': len(important_features)
                })
        
        self.preprocessing_steps.append({
            'step': 'feature_selection_summary',
            'original_features': original_features,
            'final_features': len(df.columns),
            'features_removed': original_features - len(df.columns)
        })
        
        return df
    
    def _final_clinical_validation(self, df: pd.DataFrame, original_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Perform final validation of processed healthcare data."""
        validation_results = {
            'shape_change': f'{original_shape} -> {df.shape}',
            'missing_values': df.isnull().sum().sum(),
            'infinite_values': np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
            'clinical_features_retained': len([col for col in df.columns if any(
                keyword in col.lower() for keyword in [
                    'age', 'bmi', 'blood', 'heart', 'glucose', 'pressure', 
                    'comorbidity', 'diagnosis', 'medication'
                ]
            )]),
            'data_types': dict(df.dtypes.astype(str)),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'hipaa_compliance': self.anonymize,
            'processing_success': True
        }
        
        # Check for remaining issues
        issues = []
        if validation_results['missing_values'] > 0:
            issues.append('Missing values still present')
        if validation_results['infinite_values'] > 0:
            issues.append('Infinite values detected')
        if validation_results['clinical_features_retained'] == 0:
            issues.append('No clinical features retained')
        
        validation_results['issues'] = issues
        validation_results['processing_success'] = len(issues) == 0
        
        return validation_results
    
    def _log_step(self, step_name: str):
        """Log preprocessing step."""
        if self.verbose:
            logger.info(f"Executing: {step_name}")
    
    def save_pipeline(self, filepath: str):
        """Save the preprocessing pipeline configuration and fitted components."""
        pipeline_data = {
            'config': self.config,
            'preprocessing_steps': self.preprocessing_steps,
            'clinical_metadata': self.clinical_metadata,
            'anonymization_applied': self.anonymize,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(pipeline_data, f, indent=2, default=str)
        
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """Load a saved preprocessing pipeline configuration."""
        with open(filepath, 'r') as f:
            pipeline_data = json.load(f)
        
        self.config = pipeline_data['config']
        self.preprocessing_steps = pipeline_data.get('preprocessing_steps', [])
        self.clinical_metadata = pipeline_data.get('clinical_metadata', {})
        self.anonymize = pipeline_data.get('anonymization_applied', True)
        
        logger.info(f"Pipeline loaded from {filepath}")


def main():
    """Example usage of the Healthcare Data Preprocessor."""
    # Create sample healthcare data
    np.random.seed(42)
    n_patients = 5000
    
    # Generate synthetic patient data
    data = {
        'patient_id': [f'PAT_{i:06d}' for i in range(n_patients)],
        'age': np.random.normal(55, 20, n_patients).clip(18, 95),
        'gender': np.random.choice(['M', 'F'], n_patients),
        'height': np.random.normal(68, 4, n_patients).clip(60, 80),  # inches
        'weight': np.random.normal(170, 40, n_patients).clip(100, 350),  # pounds
        'systolic_bp': np.random.normal(130, 20, n_patients).clip(90, 200),
        'diastolic_bp': np.random.normal(80, 15, n_patients).clip(60, 120),
        'heart_rate': np.random.normal(72, 12, n_patients).clip(50, 120),
        'temperature': np.random.normal(98.6, 1, n_patients).clip(96, 102),
        'glucose': np.random.lognormal(4.6, 0.3, n_patients).clip(70, 300),
        'cholesterol': np.random.normal(200, 40, n_patients).clip(120, 400),
        'hemoglobin': np.random.normal(14, 2, n_patients).clip(10, 18),
        'admission_date': pd.date_range('2023-01-01', periods=n_patients, freq='2H'),
        'primary_diagnosis': np.random.choice([
            'I10', 'E11.9', 'J44.1', 'N18.6', 'F32.9', 'M79.3'
        ], n_patients),
        'medication': np.random.choice([
            'lisinopril', 'metformin', 'aspirin', 'amlodipine', 'metoprolol'
        ], n_patients),
        'insurance': np.random.choice(['Medicare', 'Medicaid', 'Private', 'Uninsured'], n_patients)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values and anomalies
    df.loc[np.random.choice(df.index, 500), 'glucose'] = np.nan
    df.loc[np.random.choice(df.index, 100), 'systolic_bp'] = np.random.uniform(200, 250, 100)  # Hypertensive crisis
    df.loc[np.random.choice(df.index, 50), 'heart_rate'] = np.random.uniform(150, 200, 50)  # Tachycardia
    
    print("Original Data Shape:", df.shape)
    print("\nOriginal Data Info:")
    print(df.info())
    
    # Initialize preprocessor with custom config
    config = {
        'anonymization': {
            'hash_ids': True,
            'age_binning': True,
            'date_shifting': True
        },
        'missing_values': {
            'strategy': 'iterative',
            'medical_imputation': True,
            'preserve_na_indicators': True
        },
        'outliers': {
            'method': 'medical_ranges',
            'vital_signs_handling': 'flag',
            'lab_values_handling': 'flag'
        },
        'feature_engineering': {
            'age_groups': True,
            'bmi_calculation': True,
            'vital_signs_ratios': True,
            'comorbidity_scores': True
        },
        'scaling': {
            'method': 'robust',
            'per_measurement_type': True
        }
    }
    
    # Define medical column types
    medical_columns = {
        'vital_signs': ['systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature', 'weight', 'height'],
        'lab_values': ['glucose', 'cholesterol', 'hemoglobin'],
        'medications': ['medication'],
        'diagnoses': ['primary_diagnosis'],
        'demographics': ['age', 'gender', 'insurance'],
        'temporal': ['admission_date'],
        'clinical_notes': []
    }
    
    preprocessor = HealthcareDataPreprocessor(config=config, verbose=True, anonymize=True)
    
    # Apply preprocessing
    processed_df, metadata = preprocessor.preprocess(df, 'patient_id', medical_columns)
    
    print(f"\nProcessed Data Shape: {processed_df.shape}")
    print("\nPreprocessing Steps Applied:")
    for i, step in enumerate(metadata['preprocessing_steps'], 1):
        print(f"{i}. {step['step']}")
    
    print(f"\nClinical Validation Results:")
    print(f"- Processing Success: {metadata['final_validation']['processing_success']}")
    print(f"- Clinical Features Retained: {metadata['final_validation']['clinical_features_retained']}")
    print(f"- HIPAA Compliance: {metadata['final_validation']['hipaa_compliance']}")
    print(f"- Data Quality Score: {metadata['validation_results']['data_quality_score']:.3f}")
    print(f"- Memory Usage: {metadata['final_validation']['memory_usage_mb']:.2f} MB")
    
    # Save pipeline for reuse
    preprocessor.save_pipeline('healthcare_preprocessing_pipeline.json')
    
    print("\nHealthcare preprocessing pipeline completed successfully!")
    
    # Show some feature examples
    print("\nSample Features (first 3 rows):")
    feature_cols = [col for col in processed_df.columns if any(
        feat_type in col.lower() for feat_type in ['bmi', 'pressure', '_flag', '_missing', '_abnormal']
    )][:10]  # Show first 10 feature columns
    
    if feature_cols:
        print(processed_df[feature_cols].head(3))


if __name__ == "__main__":
    main()