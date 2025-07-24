"""Computer Vision service for advanced image and video processing."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from datetime import datetime

from computer_vision.domain.entities.vision_model import (
    VisionTaskType, ModelArchitecture, VisionPrediction, VisionModelMetadata,
    VisionModelConfig, BoundingBox, Polygon, Keypoint, ImageFormat, VideoFormat
)


class VisionService:
    """Advanced computer vision service for multiple AI tasks."""
    
    def __init__(self, config: VisionModelConfig):
        """Initialize the vision service.
        
        Args:
            config: Configuration for the vision service
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._models: Dict[str, Any] = {}
        self._model_metadata: Dict[str, VisionModelMetadata] = {}
        self._performance_metrics: Dict[str, List[float]] = {}
        
        # Initialize device
        self._setup_device()
        
    def _setup_device(self):
        """Setup compute device for inference."""
        try:
            import torch
            
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = self.config.device
                
            self.logger.info(f"Using device: {self.device}")
            
        except ImportError:
            self.device = "cpu"
            self.logger.warning("PyTorch not available, using CPU")
    
    async def load_model(
        self,
        model_id: str,
        model_path: Union[str, Path],
        task_type: VisionTaskType,
        architecture: ModelArchitecture,
        metadata: Optional[VisionModelMetadata] = None
    ) -> bool:
        """Load a computer vision model.
        
        Args:
            model_id: Unique identifier for the model
            model_path: Path to the model file
            task_type: Type of vision task
            architecture: Model architecture
            metadata: Optional model metadata
            
        Returns:
            True if model loaded successfully
        """
        try:
            self.logger.info(f"Loading model {model_id} from {model_path}")
            
            # Load model based on architecture and task type
            model = await self._load_model_implementation(
                model_path, task_type, architecture
            )
            
            if model is None:
                self.logger.error(f"Failed to load model {model_id}")
                return False
            
            # Store model and metadata
            self._models[model_id] = model
            if metadata:
                self._model_metadata[model_id] = metadata
            
            # Initialize performance tracking
            self._performance_metrics[model_id] = []
            
            self.logger.info(f"Successfully loaded model {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_id}: {str(e)}")
            return False
    
    async def _load_model_implementation(
        self,
        model_path: Union[str, Path],
        task_type: VisionTaskType,
        architecture: ModelArchitecture
    ) -> Optional[Any]:
        """Load model implementation based on architecture.
        
        Args:
            model_path: Path to the model file
            task_type: Type of vision task
            architecture: Model architecture
            
        Returns:
            Loaded model or None if failed
        """
        try:
            import torch
            import torchvision.transforms as transforms
            
            # Load model based on architecture
            if architecture == ModelArchitecture.RESNET:
                model = await self._load_resnet_model(model_path, task_type)
            elif architecture == ModelArchitecture.EFFICIENTNET:
                model = await self._load_efficientnet_model(model_path, task_type)
            elif architecture == ModelArchitecture.VIT:
                model = await self._load_vision_transformer(model_path, task_type)
            elif architecture == ModelArchitecture.YOLO:
                model = await self._load_yolo_model(model_path, task_type)
            elif architecture == ModelArchitecture.UNET:
                model = await self._load_unet_model(model_path, task_type)
            elif architecture == ModelArchitecture.CLIP:
                model = await self._load_clip_model(model_path, task_type)
            else:
                # Generic PyTorch model loading
                model = torch.load(model_path, map_location=self.device)
            
            if model is not None:
                model.eval()
                model = model.to(self.device)
                
                # Apply optimizations
                if self.config.use_torch_script:
                    model = torch.jit.script(model)
                
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model implementation: {str(e)}")
            return None
    
    async def _load_resnet_model(self, model_path: Union[str, Path], task_type: VisionTaskType) -> Optional[Any]:
        """Load ResNet model."""
        try:
            import torch
            import torchvision.models as models
            
            # Load pretrained ResNet or custom checkpoint
            if str(model_path).endswith('.pth') or str(model_path).endswith('.pt'):
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model = models.resnet50()  # Default architecture
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model = checkpoint
            else:
                # Load from torchvision
                model = models.resnet50(pretrained=True)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading ResNet model: {str(e)}")
            return None
    
    async def _load_efficientnet_model(self, model_path: Union[str, Path], task_type: VisionTaskType) -> Optional[Any]:
        """Load EfficientNet model."""
        try:
            import torch
            import torchvision.models as models
            
            if str(model_path).endswith('.pth') or str(model_path).endswith('.pt'):
                model = torch.load(model_path, map_location=self.device)
            else:
                model = models.efficientnet_b0(pretrained=True)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading EfficientNet model: {str(e)}")
            return None
    
    async def _load_vision_transformer(self, model_path: Union[str, Path], task_type: VisionTaskType) -> Optional[Any]:
        """Load Vision Transformer model."""
        try:
            import torch
            
            # Load ViT model (implementation would depend on specific library)
            model = torch.load(model_path, map_location=self.device)
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading Vision Transformer: {str(e)}")
            return None
    
    async def _load_yolo_model(self, model_path: Union[str, Path], task_type: VisionTaskType) -> Optional[Any]:
        """Load YOLO model for object detection."""
        try:
            # YOLOv5/YOLOv8 loading (would use ultralytics library)
            # This is a placeholder implementation
            import torch
            model = torch.load(model_path, map_location=self.device)
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {str(e)}")
            return None
    
    async def _load_unet_model(self, model_path: Union[str, Path], task_type: VisionTaskType) -> Optional[Any]:
        """Load U-Net model for segmentation."""
        try:
            import torch
            model = torch.load(model_path, map_location=self.device)
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading U-Net model: {str(e)}")
            return None
    
    async def _load_clip_model(self, model_path: Union[str, Path], task_type: VisionTaskType) -> Optional[Any]:
        """Load CLIP model for vision-language tasks."""
        try:
            # CLIP model loading (would use OpenAI CLIP library)
            import torch
            model = torch.load(model_path, map_location=self.device)
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading CLIP model: {str(e)}")
            return None
    
    async def predict_image(
        self,
        model_id: str,
        image_data: Union[np.ndarray, str, Path],
        task_type: VisionTaskType,
        **kwargs
    ) -> VisionPrediction:
        """Perform prediction on a single image.
        
        Args:
            model_id: ID of the model to use
            image_data: Image data as numpy array or path to image file
            task_type: Type of vision task to perform
            **kwargs: Additional task-specific parameters
            
        Returns:
            Vision prediction results
        """
        start_time = time.time()
        
        try:
            # Validate model exists
            if model_id not in self._models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model = self._models[model_id]
            
            # Preprocess image
            preprocessed_image, image_dims = await self._preprocess_image(
                image_data, task_type
            )
            preprocess_time = (time.time() - start_time) * 1000
            
            # Run inference
            inference_start = time.time()
            with_grad = kwargs.get('requires_grad', False)
            
            if not with_grad:
                import torch
                with torch.no_grad():
                    raw_output = await self._run_inference(
                        model, preprocessed_image, task_type, **kwargs
                    )
            else:
                raw_output = await self._run_inference(
                    model, preprocessed_image, task_type, **kwargs
                )
            
            inference_time = (time.time() - inference_start) * 1000
            
            # Post-process results
            postprocess_start = time.time()
            prediction = await self._postprocess_results(
                raw_output, task_type, image_dims, **kwargs
            )
            postprocess_time = (time.time() - postprocess_start) * 1000
            
            # Update prediction with timing information
            total_time = (time.time() - start_time) * 1000
            prediction.processing_time_ms = inference_time
            prediction.preprocessing_time_ms = preprocess_time
            prediction.postprocessing_time_ms = postprocess_time
            prediction.model_version = self._model_metadata.get(model_id, {}).version if model_id in self._model_metadata else None
            
            # Track performance
            self._performance_metrics[model_id].append(total_time)
            if len(self._performance_metrics[model_id]) > 1000:
                self._performance_metrics[model_id] = self._performance_metrics[model_id][-1000:]
            
            self.logger.info(f"Image prediction completed in {total_time:.2f}ms")
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error during image prediction: {str(e)}")
            # Return error prediction
            total_time = (time.time() - start_time) * 1000
            return VisionPrediction(
                task_type=task_type,
                confidence=0.0,
                processing_time_ms=total_time,
                image_dimensions=(0, 0)
            )
    
    async def predict_batch(
        self,
        model_id: str,
        image_batch: List[Union[np.ndarray, str, Path]],
        task_type: VisionTaskType,
        **kwargs
    ) -> List[VisionPrediction]:
        """Perform batch prediction on multiple images.
        
        Args:
            model_id: ID of the model to use
            image_batch: List of images to process
            task_type: Type of vision task to perform
            **kwargs: Additional task-specific parameters
            
        Returns:
            List of prediction results
        """
        start_time = time.time()
        
        try:
            if model_id not in self._models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model = self._models[model_id]
            
            # Process images in batches
            batch_size = min(self.config.batch_size, len(image_batch))
            predictions = []
            
            for i in range(0, len(image_batch), batch_size):
                batch = image_batch[i:i + batch_size]
                
                # Preprocess batch
                preprocessed_batch = []
                image_dims_batch = []
                
                for image_data in batch:
                    proc_img, dims = await self._preprocess_image(image_data, task_type)
                    preprocessed_batch.append(proc_img)
                    image_dims_batch.append(dims)
                
                # Stack into batch tensor
                import torch
                if preprocessed_batch:
                    batch_tensor = torch.stack(preprocessed_batch)
                    
                    # Run batch inference
                    with torch.no_grad():
                        batch_output = await self._run_batch_inference(
                            model, batch_tensor, task_type, **kwargs
                        )
                    
                    # Post-process each result
                    for j, (output, dims) in enumerate(zip(batch_output, image_dims_batch)):
                        prediction = await self._postprocess_results(
                            output, task_type, dims, **kwargs
                        )
                        prediction.model_version = self._model_metadata.get(model_id, {}).version if model_id in self._model_metadata else None
                        predictions.append(prediction)
            
            total_time = (time.time() - start_time) * 1000
            avg_time_per_image = total_time / len(image_batch)
            
            # Update timing for all predictions
            for pred in predictions:
                pred.processing_time_ms = avg_time_per_image
            
            self.logger.info(
                f"Batch prediction completed: {len(image_batch)} images in {total_time:.2f}ms "
                f"({avg_time_per_image:.2f}ms per image)"
            )
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error during batch prediction: {str(e)}")
            return [
                VisionPrediction(
                    task_type=task_type,
                    confidence=0.0,
                    processing_time_ms=0.0,
                    image_dimensions=(0, 0)
                )
                for _ in image_batch
            ]
    
    async def _preprocess_image(
        self,
        image_data: Union[np.ndarray, str, Path],
        task_type: VisionTaskType
    ) -> Tuple[Any, Tuple[int, int]]:
        """Preprocess image for model input.
        
        Args:
            image_data: Image data as numpy array or file path
            task_type: Type of vision task
            
        Returns:
            Preprocessed image tensor and original dimensions
        """
        try:
            import torch
            import torchvision.transforms as transforms
            from PIL import Image
            
            # Load image if path provided
            if isinstance(image_data, (str, Path)):
                image = Image.open(image_data).convert('RGB')
                image_array = np.array(image)
            elif isinstance(image_data, np.ndarray):
                if image_data.shape[-1] == 3:  # RGB
                    image = Image.fromarray(image_data.astype(np.uint8))
                else:  # Assume grayscale
                    image = Image.fromarray(image_data.astype(np.uint8)).convert('RGB')
                image_array = image_data
            else:
                raise ValueError("Unsupported image data type")
            
            original_dims = image.size  # (width, height)
            
            # Define preprocessing transforms based on task
            if task_type in [VisionTaskType.CLASSIFICATION, VisionTaskType.FACE_RECOGNITION]:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            elif task_type == VisionTaskType.OBJECT_DETECTION:
                transform = transforms.Compose([
                    transforms.Resize((640, 640)),
                    transforms.ToTensor(),
                ])
            elif task_type in [VisionTaskType.SEMANTIC_SEGMENTATION, VisionTaskType.INSTANCE_SEGMENTATION]:
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                # Default transform
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
            
            # Apply transforms
            tensor_image = transform(image)
            tensor_image = tensor_image.to(self.device)
            
            return tensor_image, original_dims
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    async def _run_inference(
        self,
        model: Any,
        image_tensor: Any,
        task_type: VisionTaskType,
        **kwargs
    ) -> Any:
        """Run model inference on preprocessed image.
        
        Args:
            model: Loaded model
            image_tensor: Preprocessed image tensor
            task_type: Type of vision task
            **kwargs: Additional parameters
            
        Returns:
            Raw model output
        """
        try:
            # Add batch dimension if needed
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Run inference
            output = model(image_tensor)
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error during inference: {str(e)}")
            raise
    
    async def _run_batch_inference(
        self,
        model: Any,
        batch_tensor: Any,
        task_type: VisionTaskType,
        **kwargs
    ) -> List[Any]:
        """Run model inference on batch of images.
        
        Args:
            model: Loaded model
            batch_tensor: Batch of preprocessed image tensors
            task_type: Type of vision task
            **kwargs: Additional parameters
            
        Returns:
            List of raw model outputs
        """
        try:
            # Run batch inference
            batch_output = model(batch_tensor)
            
            # Split batch output into individual results
            if isinstance(batch_output, tuple):
                # Handle models that return multiple outputs
                individual_outputs = []
                for i in range(batch_tensor.shape[0]):
                    output_tuple = tuple(out[i] for out in batch_output)
                    individual_outputs.append(output_tuple)
                return individual_outputs
            else:
                # Single output tensor
                return [batch_output[i] for i in range(batch_tensor.shape[0])]
            
        except Exception as e:
            self.logger.error(f"Error during batch inference: {str(e)}")
            raise
    
    async def _postprocess_results(
        self,
        raw_output: Any,
        task_type: VisionTaskType,
        image_dims: Tuple[int, int],
        **kwargs
    ) -> VisionPrediction:
        """Post-process raw model output into structured prediction.
        
        Args:
            raw_output: Raw model output
            task_type: Type of vision task
            image_dims: Original image dimensions (width, height)
            **kwargs: Additional parameters
            
        Returns:
            Structured prediction result
        """
        try:
            import torch
            import torch.nn.functional as F
            
            prediction = VisionPrediction(
                task_type=task_type,
                confidence=0.0,
                processing_time_ms=0.0,
                image_dimensions=image_dims
            )
            
            if task_type == VisionTaskType.CLASSIFICATION:
                # Classification post-processing
                if isinstance(raw_output, torch.Tensor):
                    probabilities = F.softmax(raw_output, dim=1)
                    probs_np = probabilities.cpu().numpy()[0]
                    
                    # Get class names if available
                    metadata = self._model_metadata.get(kwargs.get('model_id', ''), None)
                    class_names = metadata.output_classes if metadata else [f"class_{i}" for i in range(len(probs_np))]
                    
                    # Create classification results
                    classification_results = {
                        class_names[i]: float(probs_np[i])
                        for i in range(len(probs_np))
                    }
                    
                    prediction.classification_results = classification_results
                    prediction.confidence = float(np.max(probs_np))
            
            elif task_type == VisionTaskType.OBJECT_DETECTION:
                # Object detection post-processing
                bounding_boxes = self._postprocess_object_detection(
                    raw_output, image_dims, **kwargs
                )
                prediction.bounding_boxes = bounding_boxes
                prediction.confidence = max([bbox.confidence for bbox in bounding_boxes], default=0.0)
            
            elif task_type in [VisionTaskType.SEMANTIC_SEGMENTATION, VisionTaskType.INSTANCE_SEGMENTATION]:
                # Segmentation post-processing
                mask, polygons = self._postprocess_segmentation(
                    raw_output, image_dims, **kwargs
                )
                prediction.segmentation_mask = mask
                prediction.polygons = polygons
                prediction.confidence = 1.0  # Placeholder
            
            elif task_type == VisionTaskType.POSE_ESTIMATION:
                # Pose estimation post-processing
                keypoints = self._postprocess_pose_estimation(
                    raw_output, image_dims, **kwargs
                )
                prediction.keypoints = keypoints
                prediction.confidence = np.mean([kp.confidence for kp in keypoints]) if keypoints else 0.0
            
            elif task_type == VisionTaskType.OCR:
                # OCR post-processing
                text_results = self._postprocess_ocr(
                    raw_output, image_dims, **kwargs
                )
                prediction.text_results = text_results
                prediction.confidence = np.mean([result.get('confidence', 0.0) for result in text_results]) if text_results else 0.0
            
            else:
                # Generic post-processing
                if isinstance(raw_output, torch.Tensor):
                    prediction.features = raw_output.cpu().numpy()
                    prediction.confidence = 1.0
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error post-processing results: {str(e)}")
            return VisionPrediction(
                task_type=task_type,
                confidence=0.0,
                processing_time_ms=0.0,
                image_dimensions=image_dims
            )
    
    def _postprocess_object_detection(
        self,
        raw_output: Any,
        image_dims: Tuple[int, int],
        **kwargs
    ) -> List[BoundingBox]:
        """Post-process object detection output."""
        bounding_boxes = []
        
        try:
            import torch
            
            # This is a simplified implementation
            # Real implementation would depend on specific model output format
            if isinstance(raw_output, torch.Tensor):
                # Assume output format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, conf, class]
                detections = raw_output.cpu().numpy()
                
                if len(detections.shape) == 3:
                    detections = detections[0]  # Remove batch dimension
                
                confidence_threshold = kwargs.get('confidence_threshold', self.config.confidence_threshold)
                
                for detection in detections:
                    if len(detection) >= 6:
                        x1, y1, x2, y2, conf, class_id = detection[:6]
                        
                        if conf >= confidence_threshold:
                            # Normalize coordinates
                            x = x1 / image_dims[0]
                            y = y1 / image_dims[1]
                            width = (x2 - x1) / image_dims[0]
                            height = (y2 - y1) / image_dims[1]
                            
                            bbox = BoundingBox(
                                x=float(x),
                                y=float(y),
                                width=float(width),
                                height=float(height),
                                confidence=float(conf),
                                label=f"class_{int(class_id)}",
                                label_id=int(class_id)
                            )
                            bounding_boxes.append(bbox)
            
        except Exception as e:
            self.logger.error(f"Error post-processing object detection: {str(e)}")
        
        return bounding_boxes
    
    def _postprocess_segmentation(
        self,
        raw_output: Any,
        image_dims: Tuple[int, int],
        **kwargs
    ) -> Tuple[Optional[np.ndarray], List[Polygon]]:
        """Post-process segmentation output."""
        try:
            import torch
            
            mask = None
            polygons = []
            
            if isinstance(raw_output, torch.Tensor):
                # Convert to numpy and resize to original image dimensions
                mask_tensor = raw_output.cpu()
                if len(mask_tensor.shape) == 4:  # [batch, channels, height, width]
                    mask_tensor = mask_tensor[0]  # Remove batch dimension
                
                if len(mask_tensor.shape) == 3:  # [channels, height, width]
                    # Take argmax over channels for multi-class segmentation
                    mask_tensor = torch.argmax(mask_tensor, dim=0)
                
                mask = mask_tensor.numpy().astype(np.uint8)
                
                # Resize mask to original image dimensions
                from PIL import Image
                mask_pil = Image.fromarray(mask)
                mask_pil = mask_pil.resize(image_dims, Image.NEAREST)
                mask = np.array(mask_pil)
            
            return mask, polygons
            
        except Exception as e:
            self.logger.error(f"Error post-processing segmentation: {str(e)}")
            return None, []
    
    def _postprocess_pose_estimation(
        self,
        raw_output: Any,
        image_dims: Tuple[int, int],
        **kwargs
    ) -> List[Keypoint]:
        """Post-process pose estimation output."""
        keypoints = []
        
        try:
            import torch
            
            if isinstance(raw_output, torch.Tensor):
                # Simplified pose estimation post-processing
                kp_data = raw_output.cpu().numpy()
                
                if len(kp_data.shape) == 3:  # [batch, num_keypoints, 3] (x, y, confidence)
                    kp_data = kp_data[0]  # Remove batch dimension
                
                keypoint_names = [
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle"
                ]
                
                for i, (x, y, conf) in enumerate(kp_data):
                    if i < len(keypoint_names):
                        keypoint = Keypoint(
                            x=float(x / image_dims[0]),  # Normalize
                            y=float(y / image_dims[1]),  # Normalize
                            confidence=float(conf),
                            label=keypoint_names[i],
                            visible=conf > 0.5
                        )
                        keypoints.append(keypoint)
            
        except Exception as e:
            self.logger.error(f"Error post-processing pose estimation: {str(e)}")
        
        return keypoints
    
    def _postprocess_ocr(
        self,
        raw_output: Any,
        image_dims: Tuple[int, int],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Post-process OCR output."""
        text_results = []
        
        try:
            # Simplified OCR post-processing
            # Real implementation would use libraries like EasyOCR or PaddleOCR
            
            if isinstance(raw_output, (list, tuple)):
                for result in raw_output:
                    if isinstance(result, dict):
                        text_results.append(result)
            elif isinstance(raw_output, dict):
                text_results.append(raw_output)
            
        except Exception as e:
            self.logger.error(f"Error post-processing OCR: {str(e)}")
        
        return text_results
    
    def get_model_info(self, model_id: str) -> Optional[VisionModelMetadata]:
        """Get metadata for a loaded model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model metadata or None if not found
        """
        return self._model_metadata.get(model_id)
    
    def get_performance_stats(self, model_id: str) -> Dict[str, float]:
        """Get performance statistics for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Performance statistics
        """
        if model_id not in self._performance_metrics:
            return {}
        
        times = self._performance_metrics[model_id]
        if not times:
            return {}
        
        return {
            "mean_time_ms": np.mean(times),
            "median_time_ms": np.median(times),
            "p95_time_ms": np.percentile(times, 95),
            "p99_time_ms": np.percentile(times, 99),
            "min_time_ms": np.min(times),
            "max_time_ms": np.max(times),
            "total_predictions": len(times)
        }
    
    def list_loaded_models(self) -> List[str]:
        """List all loaded model IDs.
        
        Returns:
            List of loaded model IDs
        """
        return list(self._models.keys())
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory.
        
        Args:
            model_id: ID of the model to unload
            
        Returns:
            True if model was unloaded successfully
        """
        try:
            if model_id in self._models:
                del self._models[model_id]
                
            if model_id in self._model_metadata:
                del self._model_metadata[model_id]
                
            if model_id in self._performance_metrics:
                del self._performance_metrics[model_id]
            
            # Clear GPU cache if using CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            self.logger.info(f"Model {model_id} unloaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unloading model {model_id}: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the vision service.
        
        Returns:
            Health check results
        """
        try:
            import torch
            
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "device": self.device,
                "models_loaded": len(self._models),
                "total_predictions": sum(len(times) for times in self._performance_metrics.values())
            }
            
            # Check GPU status if using CUDA
            if self.device == "cuda":
                try:
                    health_status["gpu_available"] = torch.cuda.is_available()
                    health_status["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
                    health_status["gpu_memory_cached"] = torch.cuda.memory_reserved() / 1024**3  # GB
                except Exception:
                    health_status["gpu_status"] = "error"
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }