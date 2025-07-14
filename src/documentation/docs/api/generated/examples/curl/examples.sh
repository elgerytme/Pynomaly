# Health check
curl -X GET "https://api.pynomaly.com/health" \
  -H "X-API-Key: your-api-key-here"

# List detectors
curl -X GET "https://api.pynomaly.com/api/v1/detectors" \
  -H "X-API-Key: your-api-key-here"
