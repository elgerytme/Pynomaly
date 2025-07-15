#!/bin/bash
# Basic CLI Workflow Example
# This script demonstrates a typical anomaly detection workflow using the Pynomaly CLI

echo "🔍 Pynomaly CLI Basic Workflow Example"
echo "====================================="
echo ""

# Note: This assumes you have activated the virtual environment
# and are running from the project root directory

# Step 1: Check CLI is working
echo "1. Checking Pynomaly CLI..."
python cli.py --version
echo ""

# Step 2: List available algorithms
echo "2. Available anomaly detection algorithms:"
python cli.py detector algorithms
echo ""

# Step 3: Load a dataset
echo "3. Loading sample dataset..."
python cli.py dataset load examples/sample_data/credit_transactions.csv \
    --name "Credit Card Transactions" \
    --description "Sample credit card transaction data for fraud detection"
echo ""

# Save the dataset ID (in practice, you'd capture this from the output)
# DATASET_ID=$(python cli.py dataset list --format json | jq -r '.[0].id')

# Step 4: Create a detector
echo "4. Creating an Isolation Forest detector..."
python cli.py detector create \
    --name "Fraud Detector" \
    --algorithm IsolationForest \
    --contamination 0.1 \
    --description "Isolation Forest for credit card fraud detection"
echo ""

# Step 5: List datasets and detectors
echo "5. Current datasets:"
python cli.py dataset list
echo ""

echo "6. Current detectors:"
python cli.py detector list
echo ""

# Step 6: Train the detector (you'll need to replace with actual IDs)
echo "7. Training detector..."
echo "Run: python cli.py detect train <detector_id> <dataset_id>"
echo "Example: python cli.py detect train abc123 def456"
echo ""

# Step 7: Run detection
echo "8. Running anomaly detection..."
echo "Run: python cli.py detect run <detector_id> <dataset_id>"
echo ""

# Step 8: View results
echo "9. Viewing detection results..."
echo "Run: python cli.py detect results --latest"
echo ""

# Step 9: Start the web UI
echo "10. Starting web interface..."
echo "Run: python cli.py server start"
echo "Then open: http://localhost:8000"
echo ""

echo "✅ Basic workflow example completed!"
echo ""
echo "Next steps:"
echo "- Try different algorithms (LOF, OCSVM, etc.)"
echo "- Adjust contamination parameter"
echo "- Load your own datasets"
echo "- Use the web UI for visualization"
