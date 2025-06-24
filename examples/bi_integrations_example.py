"""
Business Intelligence Integrations Example

This example demonstrates how to use Pynomaly's business intelligence integrations
to export anomaly detection results to Excel, Power BI, Google Sheets, and Smartsheet.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from uuid import uuid4

# Pynomaly imports
from pynomaly.application.services.export_service import ExportService
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.application.dto.export_options import ExportOptions, ExportFormat


def create_sample_detection_result():
    """Create a sample detection result for demonstration."""
    # Generate sample data
    n_samples = 100
    np.random.seed(42)
    
    # Create mostly normal scores with some anomalies
    scores = []
    labels = []
    anomalies = []
    
    for i in range(n_samples):
        if i in [15, 23, 45, 67, 89]:  # Predefined anomaly indices
            score_value = np.random.uniform(0.7, 0.95)
            scores.append(AnomalyScore(score_value))
            labels.append(1)
            
            # Create anomaly entity
            anomaly = Anomaly(
                id=uuid4(),
                index=i,
                score=scores[-1],
                feature_values={
                    "feature_1": np.random.uniform(10, 20),
                    "feature_2": np.random.uniform(-5, 5),
                    "feature_3": np.random.uniform(100, 200)
                }
            )
            anomalies.append(anomaly)
        else:
            score_value = np.random.uniform(0.0, 0.4)
            scores.append(AnomalyScore(score_value))
            labels.append(0)
    
    return DetectionResult(
        detector_id=uuid4(),
        dataset_id=uuid4(),
        anomalies=anomalies,
        scores=scores,
        labels=np.array(labels),
        threshold=0.5,
        execution_time_ms=250.0,
        metadata={
            "detector_name": "IsolationForest",
            "algorithm": "Isolation Forest",
            "contamination": 0.05,
            "n_estimators": 100
        }
    )


def excel_export_example():
    """Demonstrate Excel export functionality."""
    print("=== Excel Export Example ===")
    
    # Create sample results
    results = create_sample_detection_result()
    
    # Initialize export service
    export_service = ExportService()
    
    # Configure Excel export options
    excel_options = ExportOptions().for_excel()
    excel_options.include_charts = True
    excel_options.highlight_anomalies = True
    excel_options.add_conditional_formatting = True
    excel_options.use_advanced_formatting = True
    
    try:
        # Export to Excel
        output_path = "anomaly_results.xlsx"
        export_result = export_service.export_results(
            results=results,
            file_path=output_path,
            options=excel_options
        )
        
        print(f"✅ Excel export successful!")
        print(f"📁 File: {export_result['file_path']}")
        print(f"📊 Total samples: {export_result['total_samples']}")
        print(f"🚨 Anomalies detected: {export_result['anomalies_count']}")
        print(f"📋 Worksheets: {', '.join(export_result['worksheets'])}")
        
    except Exception as e:
        print(f"❌ Excel export failed: {e}")
        print("💡 Make sure to install Excel dependencies: pip install openpyxl xlsxwriter")


def powerbi_export_example():
    """Demonstrate Power BI export functionality."""
    print("\n=== Power BI Export Example ===")
    
    # Create sample results
    results = create_sample_detection_result()
    
    # Initialize export service
    export_service = ExportService()
    
    # Configure Power BI export options
    powerbi_options = ExportOptions().for_powerbi(
        workspace_id="your-workspace-id",
        dataset_name="Anomaly Detection Results"
    )
    powerbi_options.streaming_dataset = True
    powerbi_options.table_name = "AnomalyResults"
    
    try:
        # Note: This requires proper Azure AD authentication
        # You would need to set up credentials first
        
        print("🔐 Power BI export requires authentication setup:")
        print("1. Create Azure AD application")
        print("2. Grant Power BI API permissions")
        print("3. Configure authentication credentials")
        print("4. Set workspace_id and dataset_name")
        
        # Uncomment below for actual export (requires credentials)
        # export_result = export_service.export_results(
        #     results=results,
        #     file_path="",  # Not used for Power BI
        #     options=powerbi_options
        # )
        
        print("📊 Power BI export would create:")
        print(f"   - Dataset: {powerbi_options.dataset_name}")
        print(f"   - Table: {powerbi_options.table_name}")
        print(f"   - Streaming: {powerbi_options.streaming_dataset}")
        
    except Exception as e:
        print(f"❌ Power BI export setup needed: {e}")
        print("💡 Install Power BI dependencies: pip install msal azure-identity")


def google_sheets_export_example():
    """Demonstrate Google Sheets export functionality."""
    print("\n=== Google Sheets Export Example ===")
    
    # Create sample results
    results = create_sample_detection_result()
    
    # Initialize export service
    export_service = ExportService()
    
    # Configure Google Sheets export options
    gsheets_options = ExportOptions().for_gsheets()
    gsheets_options.share_with_emails = ["user@example.com"]
    gsheets_options.permissions = "edit"
    gsheets_options.include_charts = True
    
    try:
        print("🔐 Google Sheets export requires authentication setup:")
        print("1. Create Google Cloud project")
        print("2. Enable Google Sheets and Drive APIs")
        print("3. Create service account or OAuth credentials")
        print("4. Download credentials JSON file")
        
        # Uncomment below for actual export (requires credentials)
        # adapter = GoogleSheetsAdapter(
        #     service_account_path="path/to/credentials.json"
        # )
        # export_result = adapter.export_results(
        #     results=results,
        #     file_path="",  # Not used for Google Sheets
        #     options=gsheets_options
        # )
        
        print("📊 Google Sheets export would create:")
        print("   - New spreadsheet with multiple worksheets")
        print("   - Conditional formatting for anomalies")
        print("   - Charts and visualizations")
        print("   - Shared with specified users")
        
    except Exception as e:
        print(f"❌ Google Sheets export setup needed: {e}")
        print("💡 Install Google Sheets dependencies: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")


def smartsheet_export_example():
    """Demonstrate Smartsheet export functionality."""
    print("\n=== Smartsheet Export Example ===")
    
    # Create sample results
    results = create_sample_detection_result()
    
    # Initialize export service
    export_service = ExportService()
    
    # Configure Smartsheet export options
    smartsheet_options = ExportOptions().for_smartsheet()
    smartsheet_options.workspace_name = "Anomaly Detection Workspace"
    smartsheet_options.notify_on_completion = True
    smartsheet_options.notification_emails = ["admin@example.com"]
    
    try:
        print("🔐 Smartsheet export requires authentication setup:")
        print("1. Create Smartsheet account")
        print("2. Generate API access token")
        print("3. Configure workspace permissions")
        
        # Uncomment below for actual export (requires API token)
        # adapter = SmartsheetAdapter(access_token="your-api-token")
        # export_result = adapter.export_results(
        #     results=results,
        #     file_path="",  # Not used for Smartsheet
        #     options=smartsheet_options
        # )
        
        print("📊 Smartsheet export would create:")
        print("   - New sheet with project-oriented columns")
        print("   - Status tracking for anomaly investigation")
        print("   - Automated workflow rules")
        print("   - Team collaboration features")
        
    except Exception as e:
        print(f"❌ Smartsheet export setup needed: {e}")
        print("💡 Install Smartsheet dependencies: pip install smartsheet-python-sdk")


def unified_export_example():
    """Demonstrate unified export service with multiple formats."""
    print("\n=== Unified Export Service Example ===")
    
    # Create sample results
    results = create_sample_detection_result()
    
    # Initialize export service
    export_service = ExportService()
    
    # Get supported formats
    supported_formats = export_service.get_supported_formats()
    print(f"📋 Supported formats: {[f.value for f in supported_formats]}")
    
    # Get service statistics
    stats = export_service.get_export_statistics()
    print(f"📊 Export statistics:")
    print(f"   Total formats: {stats['total_formats']}")
    print(f"   Available adapters: {list(stats['adapters'].keys())}")
    
    # Validate export requests
    print("\n🔍 Validating export requests:")
    
    for format in supported_formats:
        if format == ExportFormat.EXCEL:
            validation = export_service.validate_export_request(
                format=format,
                file_path="test.xlsx"
            )
            print(f"   {format.value}: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")
        else:
            validation = export_service.validate_export_request(
                format=format,
                file_path=""
            )
            print(f"   {format.value}: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")
    
    # Export to multiple formats (Excel only for demo)
    if ExportFormat.EXCEL in supported_formats:
        print("\n📤 Exporting to available formats...")
        
        multi_export_results = export_service.export_multiple_formats(
            results=results,
            base_path="multi_export",
            formats=[ExportFormat.EXCEL]
        )
        
        for format, result in multi_export_results.items():
            if result.get('success', False):
                print(f"   ✅ {format.value}: Success")
            else:
                print(f"   ❌ {format.value}: {result.get('error', 'Failed')}")


def main():
    """Run all integration examples."""
    print("🚀 Pynomaly Business Intelligence Integrations Examples")
    print("=" * 60)
    
    # Run individual examples
    excel_export_example()
    powerbi_export_example()
    google_sheets_export_example()
    smartsheet_export_example()
    unified_export_example()
    
    print("\n" + "=" * 60)
    print("✨ Examples completed!")
    print("\n📚 Next steps:")
    print("1. Install required dependencies for your target platforms")
    print("2. Set up authentication credentials")
    print("3. Configure platform-specific settings")
    print("4. Test with your own anomaly detection results")
    print("\n📖 For detailed setup instructions, see the documentation:")
    print("   - docs/integrations/excel.md")
    print("   - docs/integrations/powerbi.md")
    print("   - docs/integrations/google_sheets.md")
    print("   - docs/integrations/smartsheet.md")


if __name__ == "__main__":
    main()