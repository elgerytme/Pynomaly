"""
Edge Computing Deployment Manager for Pynomaly Detection
========================================================

Lightweight model optimization and deployment for edge devices including
Raspberry Pi, NVIDIA Jetson, mobile devices, and IoT gateways.
"""

import logging
import os
import json
import pickle
import tempfile
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import shutil

try:
    import tensorflow as tf
    from tensorflow import lite as tflite
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import openvino.runtime as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Import our services
from ...simplified_services.core_detection_service import CoreDetectionService

logger = logging.getLogger(__name__)

@dataclass
class EdgeConfig:
    """Configuration for edge deployment."""
    # Target device specifications
    target_device: str = "raspberry_pi"  # raspberry_pi, jetson_nano, mobile, generic_arm
    cpu_cores: int = 4
    memory_mb: int = 1024
    storage_mb: int = 512
    has_gpu: bool = False
    
    # Model optimization
    optimization_level: str = "balanced"  # aggressive, balanced, conservative
    target_latency_ms: float = 100.0
    max_model_size_mb: float = 50.0
    precision: str = "float32"  # float32, float16, int8
    
    # Performance requirements
    min_accuracy: float = 0.85
    max_memory_usage_mb: float = 256.0
    max_inference_time_ms: float = 50.0
    
    # Deployment options
    deployment_format: str = "tflite"  # tflite, onnx, openvino, sklearn, pytorch_mobile
    include_preprocessing: bool = True
    enable_quantization: bool = True
    enable_pruning: bool = False
    
    # Edge-specific features
    enable_offline_mode: bool = True
    enable_model_caching: bool = True
    enable_batch_processing: bool = False
    max_batch_size: int = 1

@dataclass
class EdgeModel:
    """Edge-optimized model container."""
    model_path: str
    model_format: str
    model_size_mb: float
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    preprocessing_config: Dict[str, Any]
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]

class LiteModelOptimizer:
    """Model optimization for edge deployment."""
    
    def __init__(self, config: EdgeConfig = None):
        """Initialize lite model optimizer.
        
        Args:
            config: Edge deployment configuration
        """
        self.config = config or EdgeConfig()
        self.original_model = None
        self.optimized_model = None
        self.optimization_metrics = {}
        
        logger.info(f"Lite Model Optimizer initialized for {self.config.target_device}")
    
    def optimize_sklearn_model(self, model: BaseEstimator, X_sample: np.ndarray) -> EdgeModel:
        """Optimize scikit-learn model for edge deployment.
        
        Args:
            model: Trained scikit-learn model
            X_sample: Sample input data for shape inference
            
        Returns:
            Optimized edge model
        """
        logger.info("Optimizing scikit-learn model for edge deployment")
        
        # Create optimized model based on target device
        if self.config.optimization_level == "aggressive":
            optimized_model = self._create_simplified_tree_model(model, X_sample)
        elif self.config.optimization_level == "balanced":
            optimized_model = self._optimize_sklearn_parameters(model)
        else:
            optimized_model = model
        
        # Save optimized model
        model_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_dir, "edge_model.joblib")
        joblib.dump(optimized_model, model_path)
        
        # Calculate model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # Create preprocessing config
        preprocessing_config = {
            "scaler_type": "standard",
            "input_shape": X_sample.shape[1:],
            "feature_names": [f"feature_{i}" for i in range(X_sample.shape[1])]
        }
        
        # Performance metrics
        performance_metrics = {
            "model_size_mb": model_size_mb,
            "estimated_inference_time_ms": self._estimate_inference_time(optimized_model, X_sample),
            "compression_ratio": self._calculate_compression_ratio(model, optimized_model)
        }
        
        return EdgeModel(
            model_path=model_path,
            model_format="sklearn",
            model_size_mb=model_size_mb,
            input_shape=X_sample.shape[1:],
            output_shape=(1,),  # Binary classification
            preprocessing_config=preprocessing_config,
            metadata={
                "original_model_type": type(model).__name__,
                "optimization_level": self.config.optimization_level,
                "target_device": self.config.target_device
            },
            performance_metrics=performance_metrics
        )
    
    def optimize_tensorflow_model(self, model_path: str, X_sample: np.ndarray) -> EdgeModel:
        """Optimize TensorFlow model for edge deployment.
        
        Args:
            model_path: Path to TensorFlow model
            X_sample: Sample input data
            
        Returns:
            Optimized edge model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for TensorFlow model optimization")
        
        logger.info("Optimizing TensorFlow model for edge deployment")
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Convert to TensorFlow Lite
        converter = tflite.TFLiteConverter.from_keras_model(model)
        
        # Apply optimizations based on config
        if self.config.enable_quantization:
            if self.config.precision == "int8":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = lambda: self._representative_dataset(X_sample)
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            elif self.config.precision == "float16":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save optimized model
        model_dir = tempfile.mkdtemp()
        tflite_path = os.path.join(model_dir, "edge_model.tflite")
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Calculate model size
        model_size_mb = len(tflite_model) / (1024 * 1024)
        
        # Get input/output shapes
        interpreter = tflite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = tuple(input_details[0]['shape'][1:])  # Remove batch dimension
        output_shape = tuple(output_details[0]['shape'][1:])
        
        # Performance metrics
        performance_metrics = {
            "model_size_mb": model_size_mb,
            "estimated_inference_time_ms": self._estimate_tflite_inference_time(tflite_model, X_sample),
            "quantization": self.config.precision if self.config.enable_quantization else "none"
        }
        
        return EdgeModel(
            model_path=tflite_path,
            model_format="tflite",
            model_size_mb=model_size_mb,
            input_shape=input_shape,
            output_shape=output_shape,
            preprocessing_config={"normalize": True, "input_dtype": "float32"},
            metadata={
                "quantization": self.config.precision,
                "optimization_level": self.config.optimization_level
            },
            performance_metrics=performance_metrics
        )
    
    def optimize_pytorch_model(self, model: nn.Module, X_sample: np.ndarray) -> EdgeModel:
        """Optimize PyTorch model for edge deployment.
        
        Args:
            model: PyTorch model
            X_sample: Sample input data
            
        Returns:
            Optimized edge model
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorch model optimization")
        
        logger.info("Optimizing PyTorch model for edge deployment")
        
        model.eval()
        
        # Convert to TorchScript
        example_input = torch.FloatTensor(X_sample[:1])  # Single sample
        traced_model = torch.jit.trace(model, example_input)
        
        # Optimize for mobile if specified
        if self.config.deployment_format == "pytorch_mobile":
            optimized_model = torch.jit.optimize_for_inference(traced_model)
        else:
            optimized_model = traced_model
        
        # Save model
        model_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_dir, "edge_model.pt")
        optimized_model.save(model_path)
        
        # Calculate model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # Get shapes
        input_shape = tuple(X_sample.shape[1:])
        
        with torch.no_grad():
            output = optimized_model(example_input)
            output_shape = tuple(output.shape[1:])
        
        # Performance metrics
        performance_metrics = {
            "model_size_mb": model_size_mb,
            "estimated_inference_time_ms": self._estimate_pytorch_inference_time(optimized_model, X_sample),
            "optimization": "torchscript"
        }
        
        return EdgeModel(
            model_path=model_path,
            model_format="pytorch",
            model_size_mb=model_size_mb,
            input_shape=input_shape,
            output_shape=output_shape,
            preprocessing_config={"normalize": True, "input_dtype": "float32"},
            metadata={
                "framework": "pytorch",
                "optimization": "torchscript"
            },
            performance_metrics=performance_metrics
        )
    
    def convert_to_onnx(self, model_path: str, X_sample: np.ndarray, 
                       model_type: str = "tensorflow") -> EdgeModel:
        """Convert model to ONNX format for edge deployment.
        
        Args:
            model_path: Path to original model
            X_sample: Sample input data
            model_type: Type of original model (tensorflow, pytorch)
            
        Returns:
            ONNX edge model
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX is required for ONNX conversion")
        
        logger.info(f"Converting {model_type} model to ONNX")
        
        model_dir = tempfile.mkdtemp()
        onnx_path = os.path.join(model_dir, "edge_model.onnx")
        
        if model_type == "tensorflow" and TF_AVAILABLE:
            # TensorFlow to ONNX conversion
            import tf2onnx
            
            model = tf.keras.models.load_model(model_path)
            spec = (tf.TensorSpec(X_sample.shape, tf.float32, name="input"),)
            
            output_path = onnx_path
            model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
            
            with open(output_path, "wb") as f:
                f.write(model_proto.SerializeToString())
        
        elif model_type == "pytorch" and TORCH_AVAILABLE:
            # PyTorch to ONNX conversion
            model = torch.jit.load(model_path)
            example_input = torch.FloatTensor(X_sample[:1])
            
            torch.onnx.export(
                model,
                example_input,
                onnx_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
        else:
            raise ValueError(f"Unsupported model type for ONNX conversion: {model_type}")
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Calculate model size
        model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        
        # Performance metrics
        performance_metrics = {
            "model_size_mb": model_size_mb,
            "estimated_inference_time_ms": self._estimate_onnx_inference_time(onnx_path, X_sample),
            "format": "onnx"
        }
        
        return EdgeModel(
            model_path=onnx_path,
            model_format="onnx",
            model_size_mb=model_size_mb,
            input_shape=tuple(X_sample.shape[1:]),
            output_shape=(1,),  # Assuming binary classification
            preprocessing_config={"normalize": True},
            metadata={"original_format": model_type, "onnx_opset": 13},
            performance_metrics=performance_metrics
        )
    
    def _create_simplified_tree_model(self, model: BaseEstimator, X_sample: np.ndarray) -> BaseEstimator:
        """Create simplified tree model for aggressive optimization."""
        if hasattr(model, 'tree_'):
            # Single decision tree - reduce depth
            simplified_model = DecisionTreeClassifier(
                max_depth=min(5, getattr(model, 'max_depth', 10)),
                min_samples_split=max(10, getattr(model, 'min_samples_split', 2)),
                min_samples_leaf=max(5, getattr(model, 'min_samples_leaf', 1))
            )
        elif hasattr(model, 'n_estimators'):
            # Random Forest - reduce number of trees
            simplified_model = RandomForestClassifier(
                n_estimators=min(10, getattr(model, 'n_estimators', 100)),
                max_depth=min(5, getattr(model, 'max_depth', 10)),
                min_samples_split=max(10, getattr(model, 'min_samples_split', 2))
            )
        else:
            # Return original model if can't simplify
            return model
        
        # Copy fitted parameters if possible
        if hasattr(model, 'classes_'):
            # Fit simplified model on sample data (pseudo-training)
            # In practice, you'd retrain on the actual dataset
            y_sample = model.predict(X_sample)
            simplified_model.fit(X_sample, y_sample)
        
        return simplified_model
    
    def _optimize_sklearn_parameters(self, model: BaseEstimator) -> BaseEstimator:
        """Optimize sklearn model parameters for edge deployment."""
        # Create a copy with optimized parameters
        optimized_params = {}
        
        if hasattr(model, 'n_estimators'):
            # Reduce ensemble size
            optimized_params['n_estimators'] = min(50, getattr(model, 'n_estimators', 100))
        
        if hasattr(model, 'max_depth'):
            # Limit tree depth
            optimized_params['max_depth'] = min(8, getattr(model, 'max_depth', None) or 10)
        
        if hasattr(model, 'n_neighbors'):
            # Reduce neighbors for KNN-based models
            optimized_params['n_neighbors'] = min(10, getattr(model, 'n_neighbors', 5))
        
        # Create new model with optimized parameters
        model_class = type(model)
        try:
            optimized_model = model_class(**optimized_params)
            
            # Copy fitted state if possible
            if hasattr(model, '__dict__'):
                for attr, value in model.__dict__.items():
                    if not attr.startswith('_') and attr not in optimized_params:
                        setattr(optimized_model, attr, value)
            
            return optimized_model
            
        except Exception as e:
            logger.warning(f"Failed to optimize model parameters: {e}")
            return model
    
    def _representative_dataset(self, X_sample: np.ndarray):
        """Generate representative dataset for quantization."""
        for i in range(min(100, len(X_sample))):
            yield [X_sample[i:i+1].astype(np.float32)]
    
    def _estimate_inference_time(self, model: BaseEstimator, X_sample: np.ndarray) -> float:
        """Estimate inference time for sklearn model."""
        import time
        
        # Warm up
        for _ in range(5):
            model.predict(X_sample[:1])
        
        # Measure inference time
        start_time = time.time()
        for _ in range(100):
            model.predict(X_sample[:1])
        
        avg_time_s = (time.time() - start_time) / 100
        return avg_time_s * 1000  # Convert to milliseconds
    
    def _estimate_tflite_inference_time(self, tflite_model: bytes, X_sample: np.ndarray) -> float:
        """Estimate inference time for TensorFlow Lite model."""
        import time
        
        interpreter = tflite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        sample_input = X_sample[:1].astype(np.float32)
        
        # Warm up
        for _ in range(5):
            interpreter.set_tensor(input_details[0]['index'], sample_input)
            interpreter.invoke()
        
        # Measure inference time
        start_time = time.time()
        for _ in range(100):
            interpreter.set_tensor(input_details[0]['index'], sample_input)
            interpreter.invoke()
        
        avg_time_s = (time.time() - start_time) / 100
        return avg_time_s * 1000
    
    def _estimate_pytorch_inference_time(self, model: torch.jit.ScriptModule, X_sample: np.ndarray) -> float:
        """Estimate inference time for PyTorch model."""
        import time
        
        example_input = torch.FloatTensor(X_sample[:1])
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                model(example_input)
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                model(example_input)
        
        avg_time_s = (time.time() - start_time) / 100
        return avg_time_s * 1000
    
    def _estimate_onnx_inference_time(self, onnx_path: str, X_sample: np.ndarray) -> float:
        """Estimate inference time for ONNX model."""
        import time
        
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        sample_input = X_sample[:1].astype(np.float32)
        
        # Warm up
        for _ in range(5):
            session.run(None, {input_name: sample_input})
        
        # Measure inference time
        start_time = time.time()
        for _ in range(100):
            session.run(None, {input_name: sample_input})
        
        avg_time_s = (time.time() - start_time) / 100
        return avg_time_s * 1000
    
    def _calculate_compression_ratio(self, original_model: Any, optimized_model: Any) -> float:
        """Calculate compression ratio between original and optimized models."""
        try:
            # Save both models temporarily to compare sizes
            with tempfile.NamedTemporaryFile() as orig_file, tempfile.NamedTemporaryFile() as opt_file:
                joblib.dump(original_model, orig_file.name)
                joblib.dump(optimized_model, opt_file.name)
                
                orig_size = os.path.getsize(orig_file.name)
                opt_size = os.path.getsize(opt_file.name)
                
                return orig_size / max(opt_size, 1)
        except Exception:
            return 1.0

class EdgeDeploymentManager:
    """Comprehensive edge deployment management."""
    
    def __init__(self, config: EdgeConfig = None):
        """Initialize edge deployment manager.
        
        Args:
            config: Edge deployment configuration
        """
        self.config = config or EdgeConfig()
        self.core_service = CoreDetectionService()
        self.optimizer = LiteModelOptimizer(config)
        self.deployed_models = {}
        
        logger.info(f"Edge Deployment Manager initialized for {self.config.target_device}")
    
    def create_edge_package(self, model: Any, X_sample: np.ndarray, 
                           model_type: str = "sklearn", 
                           package_name: str = "pynomaly_edge") -> str:
        """Create complete edge deployment package.
        
        Args:
            model: Trained model to deploy
            X_sample: Sample input data
            model_type: Type of model (sklearn, tensorflow, pytorch)
            package_name: Name for deployment package
            
        Returns:
            Path to deployment package
        """
        logger.info(f"Creating edge package for {model_type} model")
        
        # Create package directory
        package_dir = tempfile.mkdtemp()
        package_path = os.path.join(package_dir, package_name)
        os.makedirs(package_path, exist_ok=True)
        
        # Optimize model
        if model_type == "sklearn":
            edge_model = self.optimizer.optimize_sklearn_model(model, X_sample)
        elif model_type == "tensorflow":
            edge_model = self.optimizer.optimize_tensorflow_model(model, X_sample)
        elif model_type == "pytorch":
            edge_model = self.optimizer.optimize_pytorch_model(model, X_sample)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Copy optimized model to package
        model_filename = f"model.{edge_model.model_format}"
        target_model_path = os.path.join(package_path, model_filename)
        shutil.copy2(edge_model.model_path, target_model_path)
        
        # Create inference script
        self._create_inference_script(package_path, edge_model, model_filename)
        
        # Create requirements file
        self._create_requirements_file(package_path, edge_model.model_format)
        
        # Create deployment configuration
        self._create_deployment_config(package_path, edge_model)
        
        # Create setup script
        self._create_setup_script(package_path)
        
        # Create README
        self._create_readme(package_path, edge_model)
        
        # Create package archive
        archive_path = shutil.make_archive(package_path, 'zip', package_dir, package_name)
        
        logger.info(f"Edge package created: {archive_path}")
        return archive_path
    
    def validate_edge_deployment(self, edge_model: EdgeModel, X_test: np.ndarray) -> Dict[str, Any]:
        """Validate edge model deployment.
        
        Args:
            edge_model: Edge model to validate
            X_test: Test data for validation
            
        Returns:
            Validation results
        """
        logger.info("Validating edge model deployment")
        
        validation_results = {
            "model_size_check": edge_model.model_size_mb <= self.config.max_model_size_mb,
            "inference_time_check": edge_model.performance_metrics.get("estimated_inference_time_ms", 0) <= self.config.max_inference_time_ms,
            "format_compatibility": self._check_format_compatibility(edge_model.model_format),
            "performance_metrics": edge_model.performance_metrics,
            "deployment_ready": False
        }
        
        # Test actual inference
        try:
            inference_test = self._test_inference(edge_model, X_test[:10])
            validation_results.update(inference_test)
        except Exception as e:
            validation_results["inference_error"] = str(e)
        
        # Overall deployment readiness
        validation_results["deployment_ready"] = all([
            validation_results["model_size_check"],
            validation_results["inference_time_check"],
            validation_results["format_compatibility"],
            validation_results.get("inference_success", False)
        ])
        
        return validation_results
    
    def _create_inference_script(self, package_path: str, edge_model: EdgeModel, model_filename: str):
        """Create inference script for edge deployment."""
        if edge_model.model_format == "sklearn":
            script_content = f"""#!/usr/bin/env python3
\"\"\"
Pynomaly Edge Inference Script
Generated for {self.config.target_device}
\"\"\"

import joblib
import numpy as np
import json
from typing import Union, List, Dict, Any

class PynomAlyEdgeDetector:
    def __init__(self, model_path: str = "{model_filename}"):
        self.model = joblib.load(model_path)
        self.preprocessing_config = {json.dumps(edge_model.preprocessing_config, indent=8)}
        
    def predict(self, X: Union[np.ndarray, List]) -> Dict[str, Any]:
        \"\"\"Predict anomalies on input data.\"\"\"
        if isinstance(X, list):
            X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Make predictions
        predictions = self.model.predict(X)
        scores = self.model.decision_function(X) if hasattr(self.model, 'decision_function') else predictions
        
        return {{
            "predictions": predictions.tolist(),
            "scores": scores.tolist(),
            "anomalies": (predictions == -1).tolist()
        }}
    
    def predict_single(self, sample: Union[np.ndarray, List]) -> bool:
        \"\"\"Predict single sample (returns True if anomaly).\"\"\"
        result = self.predict([sample])
        return result["anomalies"][0]

if __name__ == "__main__":
    import sys
    
    detector = PynomAlyEdgeDetector()
    
    # Example usage
    if len(sys.argv) > 1:
        # Read input from command line
        input_data = json.loads(sys.argv[1])
        result = detector.predict(input_data)
        print(json.dumps(result))
    else:
        print("Pynomaly Edge Detector ready. Use predict(data) or predict_single(sample)")
"""
        
        elif edge_model.model_format == "tflite":
            script_content = f"""#!/usr/bin/env python3
\"\"\"
Pynomaly Edge Inference Script (TensorFlow Lite)
Generated for {self.config.target_device}
\"\"\"

import tensorflow as tf
import numpy as np
import json
from typing import Union, List, Dict, Any

class PynomAlyEdgeDetector:
    def __init__(self, model_path: str = "{model_filename}"):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.preprocessing_config = {json.dumps(edge_model.preprocessing_config, indent=8)}
        
    def predict(self, X: Union[np.ndarray, List]) -> Dict[str, Any]:
        \"\"\"Predict anomalies on input data.\"\"\"
        if isinstance(X, list):
            X = np.array(X, dtype=np.float32)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        results = []
        for sample in X:
            sample = sample.reshape(1, -1).astype(np.float32)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], sample)
            self.interpreter.invoke()
            
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            results.append(output[0])
        
        scores = np.array(results)
        predictions = (scores > 0.5).astype(int)
        
        return {{
            "predictions": predictions.tolist(),
            "scores": scores.tolist(),
            "anomalies": (predictions == 1).tolist()
        }}
    
    def predict_single(self, sample: Union[np.ndarray, List]) -> bool:
        \"\"\"Predict single sample (returns True if anomaly).\"\"\"
        result = self.predict([sample])
        return result["anomalies"][0]

if __name__ == "__main__":
    import sys
    
    detector = PynomAlyEdgeDetector()
    
    if len(sys.argv) > 1:
        input_data = json.loads(sys.argv[1])
        result = detector.predict(input_data)
        print(json.dumps(result))
    else:
        print("Pynomaly Edge Detector (TensorFlow Lite) ready")
"""
        
        script_path = os.path.join(package_path, "inference.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
    
    def _create_requirements_file(self, package_path: str, model_format: str):
        """Create requirements file for deployment."""
        base_requirements = [
            "numpy>=1.19.0",
            "scipy>=1.7.0"
        ]
        
        if model_format == "sklearn":
            base_requirements.extend([
                "scikit-learn>=1.0.0",
                "joblib>=1.0.0"
            ])
        elif model_format == "tflite":
            base_requirements.append("tensorflow>=2.8.0")
        elif model_format == "pytorch":
            base_requirements.append("torch>=1.10.0")
        elif model_format == "onnx":
            base_requirements.extend([
                "onnx>=1.12.0",
                "onnxruntime>=1.12.0"
            ])
        
        requirements_path = os.path.join(package_path, "requirements.txt")
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(base_requirements))
    
    def _create_deployment_config(self, package_path: str, edge_model: EdgeModel):
        """Create deployment configuration file."""
        config = {
            "model_info": {
                "format": edge_model.model_format,
                "size_mb": edge_model.model_size_mb,
                "input_shape": edge_model.input_shape,
                "output_shape": edge_model.output_shape
            },
            "target_device": {
                "device_type": self.config.target_device,
                "cpu_cores": self.config.cpu_cores,
                "memory_mb": self.config.memory_mb,
                "has_gpu": self.config.has_gpu
            },
            "performance": edge_model.performance_metrics,
            "preprocessing": edge_model.preprocessing_config,
            "metadata": edge_model.metadata
        }
        
        config_path = os.path.join(package_path, "deployment_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _create_setup_script(self, package_path: str):
        """Create setup script for edge deployment."""
        setup_script = """#!/bin/bash
# Pynomaly Edge Deployment Setup Script

echo "Setting up Pynomaly Edge Detection..."

# Install Python dependencies
pip3 install -r requirements.txt

# Make inference script executable
chmod +x inference.py

# Test deployment
echo "Testing deployment..."
python3 -c "from inference import PynomAlyEdgeDetector; print('Deployment test successful')"

echo "Pynomaly Edge Detection setup complete!"
echo "Usage: python3 inference.py '[data_array]'"
"""
        
        setup_path = os.path.join(package_path, "setup.sh")
        with open(setup_path, 'w') as f:
            f.write(setup_script)
        
        os.chmod(setup_path, 0o755)
    
    def _create_readme(self, package_path: str, edge_model: EdgeModel):
        """Create README file for deployment package."""
        readme_content = f"""# Pynomaly Edge Detection Deployment

## Overview
This package contains an optimized anomaly detection model for edge deployment on {self.config.target_device}.

## Model Information
- **Format**: {edge_model.model_format}
- **Size**: {edge_model.model_size_mb:.2f} MB
- **Input Shape**: {edge_model.input_shape}
- **Estimated Inference Time**: {edge_model.performance_metrics.get('estimated_inference_time_ms', 'N/A')} ms

## Quick Start

### 1. Setup
```bash
./setup.sh
```

### 2. Basic Usage
```python
from inference import PynomAlyEdgeDetector

# Initialize detector
detector = PynomAlyEdgeDetector()

# Predict single sample
is_anomaly = detector.predict_single([1.0, 2.0, 3.0])

# Predict batch
results = detector.predict([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
```

### 3. Command Line Usage
```bash
python3 inference.py '[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]'
```

## Performance Characteristics
- **Target Device**: {self.config.target_device}
- **Memory Usage**: < {self.config.max_memory_usage_mb} MB
- **Inference Time**: < {self.config.max_inference_time_ms} ms
- **Model Size**: {edge_model.model_size_mb:.2f} MB

## Deployment Notes
- Optimized for {self.config.optimization_level} optimization level
- Supports offline operation: {self.config.enable_offline_mode}
- Batch processing: {self.config.enable_batch_processing}

## Troubleshooting
1. Ensure all requirements are installed: `pip3 install -r requirements.txt`
2. Check Python version compatibility (3.7+)
3. Verify input data format matches expected shape: {edge_model.input_shape}

For support, visit: https://github.com/pynomaly/pynomaly-detection
"""
        
        readme_path = os.path.join(package_path, "README.md")
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    def _check_format_compatibility(self, model_format: str) -> bool:
        """Check if model format is compatible with target device."""
        compatibility_matrix = {
            "raspberry_pi": ["sklearn", "tflite", "onnx"],
            "jetson_nano": ["sklearn", "tflite", "pytorch", "onnx"],
            "mobile": ["tflite", "pytorch_mobile"],
            "generic_arm": ["sklearn", "tflite", "onnx"]
        }
        
        supported_formats = compatibility_matrix.get(self.config.target_device, ["sklearn"])
        return model_format in supported_formats
    
    def _test_inference(self, edge_model: EdgeModel, X_test: np.ndarray) -> Dict[str, Any]:
        """Test model inference on sample data."""
        try:
            if edge_model.model_format == "sklearn":
                model = joblib.load(edge_model.model_path)
                predictions = model.predict(X_test)
                success = True
            elif edge_model.model_format == "tflite":
                interpreter = tflite.Interpreter(model_path=edge_model.model_path)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                sample = X_test[:1].astype(np.float32)
                
                interpreter.set_tensor(input_details[0]['index'], sample)
                interpreter.invoke()
                success = True
            else:
                success = False
            
            return {
                "inference_success": success,
                "test_samples": len(X_test)
            }
            
        except Exception as e:
            return {
                "inference_success": False,
                "inference_error": str(e)
            }
    
    def get_deployment_recommendations(self) -> Dict[str, Any]:
        """Get deployment recommendations for target device."""
        recommendations = {
            "target_device": self.config.target_device,
            "recommended_format": self._get_recommended_format(),
            "optimization_settings": self._get_optimization_recommendations(),
            "performance_expectations": self._get_performance_expectations(),
            "deployment_checklist": self._get_deployment_checklist()
        }
        
        return recommendations
    
    def _get_recommended_format(self) -> str:
        """Get recommended model format for target device."""
        format_recommendations = {
            "raspberry_pi": "tflite",
            "jetson_nano": "tflite",
            "mobile": "tflite",
            "generic_arm": "sklearn"
        }
        
        return format_recommendations.get(self.config.target_device, "sklearn")
    
    def _get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations."""
        device_optimizations = {
            "raspberry_pi": {
                "precision": "int8",
                "optimization_level": "balanced",
                "enable_quantization": True,
                "max_model_size_mb": 25.0
            },
            "jetson_nano": {
                "precision": "float16",
                "optimization_level": "balanced",
                "enable_quantization": True,
                "max_model_size_mb": 100.0
            },
            "mobile": {
                "precision": "int8",
                "optimization_level": "aggressive",
                "enable_quantization": True,
                "max_model_size_mb": 10.0
            }
        }
        
        return device_optimizations.get(self.config.target_device, {
            "precision": "float32",
            "optimization_level": "conservative",
            "enable_quantization": False,
            "max_model_size_mb": 50.0
        })
    
    def _get_performance_expectations(self) -> Dict[str, Any]:
        """Get performance expectations for target device."""
        performance_profiles = {
            "raspberry_pi": {
                "inference_time_ms": "50-200",
                "memory_usage_mb": "100-300",
                "throughput_samples_sec": "5-20"
            },
            "jetson_nano": {
                "inference_time_ms": "10-50",
                "memory_usage_mb": "200-500",
                "throughput_samples_sec": "20-100"
            },
            "mobile": {
                "inference_time_ms": "20-100",
                "memory_usage_mb": "50-150",
                "throughput_samples_sec": "10-50"
            }
        }
        
        return performance_profiles.get(self.config.target_device, {
            "inference_time_ms": "variable",
            "memory_usage_mb": "variable",
            "throughput_samples_sec": "variable"
        })
    
    def _get_deployment_checklist(self) -> List[str]:
        """Get deployment checklist."""
        return [
            "Verify target device compatibility",
            "Test model inference on sample data",
            "Validate performance requirements",
            "Check memory and storage constraints",
            "Test offline operation capabilities",
            "Verify dependency installation",
            "Test error handling and recovery",
            "Validate monitoring and logging"
        ]