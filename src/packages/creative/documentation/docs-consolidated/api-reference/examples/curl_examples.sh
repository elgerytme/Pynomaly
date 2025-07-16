#!/bin/bash
# Pynomaly API Client Example - cURL

# Base URL
BASE_URL="https://api.pynomaly.com"

# Login and get JWT token
echo "Logging in..."
LOGIN_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}')

# Extract access token
ACCESS_TOKEN=$(echo $LOGIN_RESPONSE | jq -r '.access_token')

if [ "$ACCESS_TOKEN" = "null" ]; then
    echo "Login failed"
    exit 1
fi

echo "Login successful"

# Detect anomalies
echo "Detecting anomalies..."
DETECTION_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/detection/detect" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
    "algorithm": "isolation_forest",
    "parameters": {"contamination": 0.1}
  }')

echo "Detection result:"
echo $DETECTION_RESPONSE | jq '.'

# Train model
echo "Training model..."
TRAINING_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/detection/train" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "training_data": "s3://pynomaly-data/training/sample.csv",
    "algorithm": "lstm_autoencoder",
    "parameters": {"epochs": 50, "batch_size": 32},
    "model_name": "test_model"
  }')

echo "Training result:"
echo $TRAINING_RESPONSE | jq '.'

# Health check
echo "Checking health..."
HEALTH_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/health")

echo "Health status:"
echo $HEALTH_RESPONSE | jq '.'
