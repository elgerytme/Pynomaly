"""Computer Vision model entities for advanced image and video processing."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from pathlib import Path


class VisionTaskType(Enum):
    """Types of computer vision tasks."""
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    FACE_RECOGNITION = "face_recognition"
    OCR = "ocr"
    STYLE_TRANSFER = "style_transfer"
    SUPER_RESOLUTION = "super_resolution"
    ANOMALY_DETECTION = "anomaly_detection"
    POSE_ESTIMATION = "pose_estimation"
    VIDEO_ANALYSIS = "video_analysis"
    DEEPFAKE_DETECTION = "deepfake_detection"


class ModelArchitecture(Enum):
    """Computer vision model architectures."""
    RESNET = "resnet"
    EFFICIENTNET = "efficientnet"
    VIT = "vision_transformer"
    YOLO = "yolo"
    RCNN = "rcnn"
    UNET = "unet"
    DETR = "detr"
    SWIN = "swin_transformer"
    CONVNEXT = "convnext"
    CLIP = "clip"
    DINO = "dino"
    CUSTOM = "custom"


class ImageFormat(Enum):
    """Supported image formats."""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    TIFF = "tiff"
    BMP = "bmp"
    SVG = "svg"
    HEIC = "heic"


class VideoFormat(Enum):
    """Supported video formats."""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"
    WEBM = "webm"
    FLV = "flv"


@dataclass
class BoundingBox:
    """Represents a bounding box for object detection."""
    x: float
    y: float
    width: float
    height: float
    confidence: float
    label: str
    label_id: int
    
    def __post_init__(self):
        """Validate bounding box coordinates."""
        if not (0 <= self.x <= 1 and 0 <= self.y <= 1):
            raise ValueError("Coordinates must be normalized between 0 and 1")
        if not (0 <= self.width <= 1 and 0 <= self.height <= 1):
            raise ValueError("Width and height must be normalized between 0 and 1")
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")
    
    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center coordinates of bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bounding box."""
        # Calculate intersection coordinates
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        # Check if there's intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Calculate intersection area
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class Keypoint:
    """Represents a keypoint for pose estimation."""
    x: float
    y: float
    confidence: float
    label: str
    visible: bool = True
    
    def __post_init__(self):
        """Validate keypoint coordinates."""
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class Polygon:
    """Represents a polygon for segmentation."""
    points: List[Tuple[float, float]]
    label: str
    label_id: int
    confidence: float
    
    def __post_init__(self):
        """Validate polygon."""
        if len(self.points) < 3:
            raise ValueError("Polygon must have at least 3 points")
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")
    
    @property
    def area(self) -> float:
        """Calculate polygon area using shoelace formula."""
        n = len(self.points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i][0] * self.points[j][1]
            area -= self.points[j][0] * self.points[i][1]
        return abs(area) / 2.0


@dataclass
class VisionPrediction:
    """Represents a computer vision model prediction."""
    task_type: VisionTaskType
    confidence: float
    processing_time_ms: float
    image_dimensions: Tuple[int, int]
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Task-specific results
    classification_results: Optional[Dict[str, float]] = None
    bounding_boxes: Optional[List[BoundingBox]] = None
    segmentation_mask: Optional[np.ndarray] = None
    polygons: Optional[List[Polygon]] = None
    keypoints: Optional[List[Keypoint]] = None
    text_results: Optional[List[Dict[str, Any]]] = None
    features: Optional[np.ndarray] = None
    
    # Metadata
    model_version: Optional[str] = None
    preprocessing_time_ms: Optional[float] = None
    postprocessing_time_ms: Optional[float] = None
    gpu_utilization: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    def __post_init__(self):
        """Validate prediction."""
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")
        if self.processing_time_ms < 0:
            raise ValueError("Processing time must be non-negative")
    
    @property
    def total_processing_time_ms(self) -> float:
        """Calculate total processing time including pre/post-processing."""
        total = self.processing_time_ms
        if self.preprocessing_time_ms:
            total += self.preprocessing_time_ms
        if self.postprocessing_time_ms:
            total += self.postprocessing_time_ms
        return total
    
    def get_top_k_classes(self, k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k classification results."""
        if not self.classification_results:
            return []
        
        sorted_results = sorted(
            self.classification_results.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_results[:k]
    
    def filter_detections_by_confidence(self, min_confidence: float) -> List[BoundingBox]:
        """Filter object detections by minimum confidence."""
        if not self.bounding_boxes:
            return []
        
        return [
            bbox for bbox in self.bounding_boxes
            if bbox.confidence >= min_confidence
        ]


@dataclass
class VisionModelMetadata:
    """Metadata for computer vision models."""
    model_id: str
    name: str
    version: str
    task_type: VisionTaskType
    architecture: ModelArchitecture
    created_at: datetime
    
    # Model specifications
    input_size: Tuple[int, int, int]  # height, width, channels
    output_classes: List[str]
    model_size_mb: float
    parameters_count: int
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    map_score: Optional[float] = None  # Mean Average Precision for detection
    inference_time_ms: Optional[float] = None
    
    # Training information
    training_dataset: Optional[str] = None
    training_samples: Optional[int] = None
    validation_samples: Optional[int] = None
    training_epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    
    # Hardware requirements
    min_gpu_memory_gb: Optional[float] = None
    supports_cpu: bool = True
    supports_gpu: bool = True
    supports_tpu: bool = False
    
    # Deployment information
    framework: str = "pytorch"
    quantization: Optional[str] = None
    optimization: Optional[str] = None
    deployment_target: Optional[str] = None
    
    # Licensing and compliance
    license: Optional[str] = None
    ethical_considerations: List[str] = field(default_factory=list)
    bias_metrics: Optional[Dict[str, float]] = None
    fairness_constraints: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate metadata."""
        if self.model_size_mb <= 0:
            raise ValueError("Model size must be positive")
        if self.parameters_count <= 0:
            raise ValueError("Parameters count must be positive")
        if len(self.input_size) != 3:
            raise ValueError("Input size must specify height, width, channels")
        if not self.output_classes:
            raise ValueError("Output classes cannot be empty")


@dataclass
class VisionModelConfig:
    """Configuration for computer vision models."""
    model_path: Union[str, Path]
    device: str = "auto"  # auto, cpu, cuda, mps
    batch_size: int = 1
    precision: str = "fp32"  # fp32, fp16, int8
    
    # Preprocessing configuration
    normalize: bool = True
    resize_method: str = "bilinear"
    center_crop: bool = False
    random_augmentation: bool = False
    
    # Postprocessing configuration
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4  # Non-Maximum Suppression
    max_detections: int = 100
    
    # Performance optimization
    use_torch_script: bool = False
    use_tensorrt: bool = False
    use_onnx: bool = False
    enable_profiling: bool = False
    
    # Memory management
    max_memory_gb: Optional[float] = None
    use_memory_pool: bool = True
    clear_cache: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if not (0 <= self.confidence_threshold <= 1):
            raise ValueError("Confidence threshold must be between 0 and 1")
        if not (0 <= self.nms_threshold <= 1):
            raise ValueError("NMS threshold must be between 0 and 1")
        if self.max_detections <= 0:
            raise ValueError("Max detections must be positive")


@dataclass
class VisionDataset:
    """Represents a computer vision dataset."""
    dataset_id: str
    name: str
    task_type: VisionTaskType
    version: str
    created_at: datetime
    
    # Dataset statistics
    total_samples: int
    training_samples: int
    validation_samples: int
    test_samples: int
    
    # Data characteristics
    image_formats: List[ImageFormat]
    video_formats: List[VideoFormat] = field(default_factory=list)
    resolution_range: Tuple[Tuple[int, int], Tuple[int, int]] = ((224, 224), (1024, 1024))
    channels: int = 3
    
    # Annotations
    classes: List[str] = field(default_factory=list)
    annotation_format: str = "coco"  # coco, yolo, pascal_voc, custom
    has_bounding_boxes: bool = False
    has_masks: bool = False
    has_keypoints: bool = False
    
    # Storage information
    storage_path: Union[str, Path] = ""
    compressed_size_gb: float = 0.0
    uncompressed_size_gb: float = 0.0
    
    # Quality metrics
    label_quality_score: Optional[float] = None
    data_balance_score: Optional[float] = None
    diversity_score: Optional[float] = None
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    source: Optional[str] = None
    license: Optional[str] = None
    citation: Optional[str] = None
    
    def __post_init__(self):
        """Validate dataset."""
        if self.total_samples != (self.training_samples + self.validation_samples + self.test_samples):
            raise ValueError("Total samples must equal sum of training, validation, and test samples")
        if self.total_samples <= 0:
            raise ValueError("Total samples must be positive")
        if not self.classes and self.task_type in [VisionTaskType.CLASSIFICATION, VisionTaskType.OBJECT_DETECTION]:
            raise ValueError("Classes must be specified for classification and detection tasks")
    
    @property
    def class_count(self) -> int:
        """Get number of classes."""
        return len(self.classes)
    
    @property
    def samples_per_class(self) -> float:
        """Average samples per class."""
        return self.total_samples / max(self.class_count, 1)
    
    def get_data_split_ratio(self) -> Tuple[float, float, float]:
        """Get train/validation/test split ratios."""
        total = self.total_samples
        return (
            self.training_samples / total,
            self.validation_samples / total,
            self.test_samples / total
        )


@dataclass
class VisionExperiment:
    """Represents a computer vision experiment."""
    experiment_id: str
    name: str
    task_type: VisionTaskType
    created_at: datetime
    updated_at: datetime
    
    # Experiment configuration
    model_architecture: ModelArchitecture
    dataset_id: str
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    
    # Results
    best_accuracy: Optional[float] = None
    best_loss: Optional[float] = None
    training_history: List[Dict[str, float]] = field(default_factory=list)
    validation_history: List[Dict[str, float]] = field(default_factory=list)
    
    # Model artifacts
    model_checkpoints: List[str] = field(default_factory=list)
    final_model_path: Optional[str] = None
    tensorboard_logs: Optional[str] = None
    
    # Experiment metadata
    status: str = "running"  # running, completed, failed, cancelled
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    researcher: Optional[str] = None
    
    # Resource usage
    training_time_hours: Optional[float] = None
    gpu_hours: Optional[float] = None
    memory_peak_gb: Optional[float] = None
    cost_usd: Optional[float] = None
    
    def add_training_metric(self, epoch: int, metrics: Dict[str, float]):
        """Add training metrics for an epoch."""
        metric_entry = {"epoch": epoch, **metrics}
        self.training_history.append(metric_entry)
        self.updated_at = datetime.now()
    
    def add_validation_metric(self, epoch: int, metrics: Dict[str, float]):
        """Add validation metrics for an epoch."""
        metric_entry = {"epoch": epoch, **metrics}
        self.validation_history.append(metric_entry)
        self.updated_at = datetime.now()
    
    def get_best_epoch(self, metric: str = "accuracy") -> int:
        """Get epoch with best performance for given metric."""
        if not self.validation_history:
            return 0
        
        best_epoch = 0
        best_value = float('-inf') if metric in ['accuracy', 'precision', 'recall', 'f1'] else float('inf')
        
        for entry in self.validation_history:
            if metric in entry:
                value = entry[metric]
                if metric in ['accuracy', 'precision', 'recall', 'f1'] and value > best_value:
                    best_value = value
                    best_epoch = entry['epoch']
                elif metric in ['loss'] and value < best_value:
                    best_value = value
                    best_epoch = entry['epoch']
        
        return best_epoch