#!/bin/bash
# Batch Detection CLI Example
# This script shows how to process multiple datasets with multiple algorithms

echo "🔍 Pynomaly CLI Batch Detection Example"
echo "======================================="
echo ""

# Configuration
ALGORITHMS=("IsolationForest" "LOF" "OCSVM" "COPOD")
CONTAMINATION=0.1
DATA_DIR="examples/sample_data"

# Step 1: Load all datasets
echo "1. Loading datasets from $DATA_DIR..."
for dataset in "$DATA_DIR"/*.csv; do
    filename=$(basename "$dataset")
    name="${filename%.*}"

    echo "   Loading $filename as '$name'..."
    python cli.py dataset load "$dataset" --name "$name"
done
echo ""

# Step 2: Create detectors for each algorithm
echo "2. Creating detectors for each algorithm..."
for algo in "${ALGORITHMS[@]}"; do
    echo "   Creating $algo detector..."
    python cli.py detector create \
        --name "$algo Batch Detector" \
        --algorithm "$algo" \
        --contamination $CONTAMINATION
done
echo ""

# Step 3: Show what we have
echo "3. Current setup:"
echo "   Datasets:"
python cli.py dataset list
echo ""
echo "   Detectors:"
python cli.py detector list
echo ""

# Step 4: Batch detection script
echo "4. Batch detection commands:"
echo "   To run batch detection, use these commands with actual IDs:"
echo ""

# Generate example commands
for algo in "${ALGORITHMS[@]}"; do
    echo "   # $algo on all datasets:"
    echo "   python cli.py detect train <${algo}_detector_id> <dataset_id>"
    echo "   python cli.py detect run <${algo}_detector_id> <dataset_id>"
    echo ""
done

# Step 5: Results aggregation
echo "5. Viewing batch results:"
echo "   # Latest results across all detectors:"
echo "   python cli.py detect results --latest --limit 10"
echo ""
echo "   # Results for specific detector:"
echo "   python cli.py detect results --detector <detector_id>"
echo ""
echo "   # Export results to CSV:"
echo "   python cli.py detect results --export results.csv"
echo ""

# Step 6: Comparison script
cat << 'EOF' > compare_results.py
#!/usr/bin/env python3
"""Compare detection results across algorithms."""
import pandas as pd
import sys
sys.path.append('.')

from pynomaly.infrastructure.config import create_container

container = create_container()
result_repo = container.detection_result_repository()

# Get all results
results = result_repo.get_all()

# Create comparison summary
summary = []
for result in results:
    detector = container.detector_repository().get(result.detector_id)
    dataset = container.dataset_repository().get(result.dataset_id)

    summary.append({
        'Algorithm': detector.algorithm,
        'Dataset': dataset.name,
        'Anomalies': result.n_anomalies,
        'Rate': f"{result.anomaly_rate:.1%}",
        'Runtime': f"{result.metadata.get('runtime_ms', 0)/1000:.2f}s"
    })

df = pd.DataFrame(summary)
print("\nDetection Results Comparison:")
print(df.to_string(index=False))
EOF

echo "6. Running comparison script..."
echo "   python compare_results.py"
echo ""

echo "✅ Batch detection example completed!"
echo ""
echo "Tips for production batch processing:"
echo "- Use JSON output format for parsing: --format json"
echo "- Implement parallel processing for large batches"
echo "- Set up automated alerts for high anomaly rates"
echo "- Schedule regular batch runs with cron"
echo "- Archive results for trend analysis"
