# ONNX Model Support

This document describes the ONNX (Open Neural Network Exchange) model support in Pynomaly.

## Overview

Pynomaly now supports saving and loading anomaly detection models in ONNX format, enabling cross-platform model deployment and interoperability with other machine learning frameworks.

## Current Implementation Status

✅ **WORKING**: ONNX model persistence and loading  
✅ **WORKING**: Stub model creation for non-PyTorch models  
✅ **WORKING**: Error handling and fallback mechanisms  
✅ **WORKING**: Export functionality for deployment  
✅ **WORKING**: Comprehensive test coverage  

## Usage

### Saving Models in ONNX Format

```python
from pynomaly.application.services.model_persistence_service import ModelPersistenceService

# Save a fitted detector in ONNX format
model_path = await service.save_model(detector.id, format='onnx')
```

### Loading ONNX Models

```python
# Load an ONNX model
loaded_detector = await service.load_model(detector.id, format='onnx')
```

### Exporting for Deployment

```python
# Export model for production deployment
export_dir = Path('/path/to/export')
exported_files = await service.export_model(detector.id, export_dir)
```

## Implementation Details

### PyTorch Model Export

When a PyTorch-based anomaly detection model is available:
- Real ONNX export using `torch.onnx.export()`
- Supports dynamic batch sizes
- Includes proper error handling for export failures

### Fallback Stub Models

For non-PyTorch models or when PyTorch is unavailable:
- Creates JSON-based stub models
- Maintains compatibility with the ONNX loading interface
- Falls back to SklearnAdapter for actual inference

### Error Handling

Comprehensive error handling for:
- Missing PyTorch dependencies
- ONNX export failures
- Malformed model files
- Feature flag restrictions

## Feature Flags

ONNX support respects the deep learning feature flag:
```bash
export PYNOMALY_DEEP_LEARNING=true
```

When disabled, ONNX export will raise a `RuntimeError`.

## File Formats

### Real ONNX Models
- Extension: `.onnx`
- Format: Standard ONNX binary format
- Compatible with ONNX Runtime

### Stub Models
- Extension: `.onnx` (for consistency)
- Format: JSON with model metadata
- Contains fallback information for SklearnAdapter

## Testing

Comprehensive test suite covers:
- Save/load round trips
- Error conditions
- Feature flag interactions
- Export functionality
- Stub model creation

Run ONNX-specific tests:
```bash
python -m pytest tests/application/services/test_onnx_model_persistence.py -v
```

## Limitations

1. **Real ONNX export requires PyTorch**: Full ONNX functionality needs PyTorch installed
2. **Inference limitations**: Stub models use SklearnAdapter fallback
3. **Model complexity**: Complex models may not export successfully to ONNX

## Future Improvements

- Support for TensorFlow model export
- Enhanced ONNX Runtime integration
- Better handling of complex model architectures
- Optimized inference performance