"""GPU-accelerated performance monitoring for anomaly detection."""

from __future__ import annotations

import logging
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque

try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPUtil = None
    GPU_UTIL_AVAILABLE = False

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    pynvml = None
    NVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """GPU performance metrics."""
    gpu_id: int
    name: str
    utilization: float  # Percentage
    memory_used: float  # MB
    memory_total: float  # MB
    memory_utilization: float  # Percentage
    temperature: float  # Celsius
    power_draw: float  # Watts
    power_limit: float  # Watts
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used: float  # MB
    memory_total: float  # MB
    disk_io_read: float  # MB/s
    disk_io_write: float  # MB/s
    network_sent: float  # MB/s
    network_recv: float  # MB/s
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelMetrics:
    """Model-specific performance metrics."""
    model_id: str
    training_time: Optional[float] = None
    inference_time: Optional[float] = None
    throughput: Optional[float] = None  # samples/second
    memory_footprint: Optional[float] = None  # MB
    gpu_memory_used: Optional[float] = None  # MB
    accuracy_score: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


class GPUPerformanceMonitor:
    """Advanced performance monitor with GPU acceleration support."""
    
    def __init__(
        self,
        monitoring_interval: float = 1.0,
        history_size: int = 1000,
        enable_gpu_monitoring: bool = True,
        enable_model_profiling: bool = True,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """Initialize GPU performance monitor.
        
        Args:
            monitoring_interval: Monitoring interval in seconds
            history_size: Maximum number of metrics to store
            enable_gpu_monitoring: Whether to monitor GPU metrics
            enable_model_profiling: Whether to profile model performance
            alert_thresholds: Alert thresholds for various metrics
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.enable_model_profiling = enable_model_profiling
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "cpu_percent": 90.0,
            "memory_percent": 90.0,
            "gpu_utilization": 95.0,
            "gpu_memory_percent": 90.0,
            "gpu_temperature": 85.0
        }
        
        # Metrics storage
        self.gpu_metrics_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.system_metrics_history: deque = deque(maxlen=history_size)
        self.model_metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        self._alert_callbacks: List[Callable] = []
        
        # GPU initialization
        self.available_gpus = []
        self.gpu_initialized = False
        
        if self.enable_gpu_monitoring:
            self._initialize_gpu_monitoring()
        
        # Performance counters
        self._last_disk_io = psutil.disk_io_counters()
        self._last_network_io = psutil.net_io_counters()
        self._last_measurement_time = time.time()
        
        logger.info(f"GPU Performance Monitor initialized with {len(self.available_gpus)} GPUs")
    
    def _initialize_gpu_monitoring(self) -> None:
        """Initialize GPU monitoring capabilities."""
        try:
            if NVML_AVAILABLE:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    self.available_gpus.append({
                        'id': i,
                        'handle': handle,
                        'name': name
                    })
                
                self.gpu_initialized = True
                logger.info(f"NVML initialized with {device_count} GPUs")
                
            elif GPU_UTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    self.available_gpus.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'gpu_obj': gpu
                    })
                
                self.gpu_initialized = True
                logger.info(f"GPUtil initialized with {len(gpus)} GPUs")
                
        except Exception as e:
            logger.warning(f"Failed to initialize GPU monitoring: {e}")
            self.gpu_initialized = False
    
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._monitoring:
            logger.warning("Monitoring already started")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Collect GPU metrics
                if self.gpu_initialized:
                    for gpu_info in self.available_gpus:
                        gpu_metrics = self._collect_gpu_metrics(gpu_info)
                        if gpu_metrics:
                            self.gpu_metrics_history[gpu_metrics.gpu_id].append(gpu_metrics)
                
                # Check for alerts
                self._check_alerts(system_metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        current_time = time.time()
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # Disk I/O
        current_disk_io = psutil.disk_io_counters()
        disk_read_rate = 0.0
        disk_write_rate = 0.0
        
        if self._last_disk_io:
            time_delta = current_time - self._last_measurement_time
            if time_delta > 0:
                read_bytes_delta = current_disk_io.read_bytes - self._last_disk_io.read_bytes
                write_bytes_delta = current_disk_io.write_bytes - self._last_disk_io.write_bytes
                
                disk_read_rate = (read_bytes_delta / time_delta) / (1024 * 1024)  # MB/s
                disk_write_rate = (write_bytes_delta / time_delta) / (1024 * 1024)  # MB/s
        
        # Network I/O
        current_network_io = psutil.net_io_counters()
        network_sent_rate = 0.0
        network_recv_rate = 0.0
        
        if self._last_network_io:
            time_delta = current_time - self._last_measurement_time
            if time_delta > 0:
                sent_bytes_delta = current_network_io.bytes_sent - self._last_network_io.bytes_sent
                recv_bytes_delta = current_network_io.bytes_recv - self._last_network_io.bytes_recv
                
                network_sent_rate = (sent_bytes_delta / time_delta) / (1024 * 1024)  # MB/s
                network_recv_rate = (recv_bytes_delta / time_delta) / (1024 * 1024)  # MB/s
        
        # Update for next iteration
        self._last_disk_io = current_disk_io
        self._last_network_io = current_network_io
        self._last_measurement_time = current_time
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used=memory.used / (1024 * 1024),  # MB
            memory_total=memory.total / (1024 * 1024),  # MB
            disk_io_read=disk_read_rate,
            disk_io_write=disk_write_rate,
            network_sent=network_sent_rate,
            network_recv=network_recv_rate
        )
    
    def _collect_gpu_metrics(self, gpu_info: Dict[str, Any]) -> Optional[GPUMetrics]:
        """Collect GPU performance metrics."""
        try:
            if NVML_AVAILABLE and 'handle' in gpu_info:
                return self._collect_nvml_metrics(gpu_info)
            elif GPU_UTIL_AVAILABLE and 'gpu_obj' in gpu_info:
                return self._collect_gputil_metrics(gpu_info)
        except Exception as e:
            logger.debug(f"Failed to collect GPU metrics for GPU {gpu_info['id']}: {e}")
        
        return None
    
    def _collect_nvml_metrics(self, gpu_info: Dict[str, Any]) -> GPUMetrics:
        """Collect GPU metrics using NVML."""
        handle = gpu_info['handle']
        
        # Utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        # Memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # Temperature
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            temperature = 0.0
        
        # Power
        try:
            power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
        except:
            power_draw = 0.0
            power_limit = 0.0
        
        return GPUMetrics(
            gpu_id=gpu_info['id'],
            name=gpu_info['name'],
            utilization=float(util.gpu),
            memory_used=mem_info.used / (1024 * 1024),  # MB
            memory_total=mem_info.total / (1024 * 1024),  # MB
            memory_utilization=float(mem_info.used / mem_info.total * 100),
            temperature=float(temperature),
            power_draw=float(power_draw),
            power_limit=float(power_limit)
        )
    
    def _collect_gputil_metrics(self, gpu_info: Dict[str, Any]) -> GPUMetrics:
        """Collect GPU metrics using GPUtil."""
        gpu = gpu_info['gpu_obj']
        
        return GPUMetrics(
            gpu_id=gpu.id,
            name=gpu.name,
            utilization=float(gpu.load * 100),
            memory_used=float(gpu.memoryUsed),
            memory_total=float(gpu.memoryTotal),
            memory_utilization=float(gpu.memoryUtil * 100),
            temperature=float(gpu.temperature),
            power_draw=0.0,  # Not available in GPUtil
            power_limit=0.0
        )
    
    def _check_alerts(self, system_metrics: SystemMetrics) -> None:
        """Check for performance alerts."""
        alerts = []
        
        # System alerts
        if system_metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {system_metrics.cpu_percent:.1f}%")
        
        if system_metrics.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {system_metrics.memory_percent:.1f}%")
        
        # GPU alerts
        for gpu_id, gpu_history in self.gpu_metrics_history.items():
            if gpu_history:
                latest_gpu_metrics = gpu_history[-1]
                
                if latest_gpu_metrics.utilization > self.alert_thresholds["gpu_utilization"]:
                    alerts.append(f"High GPU {gpu_id} utilization: {latest_gpu_metrics.utilization:.1f}%")
                
                if latest_gpu_metrics.memory_utilization > self.alert_thresholds["gpu_memory_percent"]:
                    alerts.append(f"High GPU {gpu_id} memory: {latest_gpu_metrics.memory_utilization:.1f}%")
                
                if latest_gpu_metrics.temperature > self.alert_thresholds["gpu_temperature"]:
                    alerts.append(f"High GPU {gpu_id} temperature: {latest_gpu_metrics.temperature:.1f}°C")
        
        # Trigger alert callbacks
        for alert in alerts:
            logger.warning(alert)
            for callback in self._alert_callbacks:
                try:
                    callback(alert, system_metrics)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def profile_model_training(self, model_id: str, training_function: Callable, *args, **kwargs) -> Any:
        """Profile model training performance."""
        if not self.enable_model_profiling:
            return training_function(*args, **kwargs)
        
        # Pre-training metrics
        start_time = time.time()
        initial_gpu_memory = self._get_current_gpu_memory()
        
        try:
            # Execute training
            result = training_function(*args, **kwargs)
            
            # Post-training metrics
            end_time = time.time()
            training_time = end_time - start_time
            final_gpu_memory = self._get_current_gpu_memory()
            
            # Calculate memory footprint
            gpu_memory_used = 0.0
            if initial_gpu_memory and final_gpu_memory:
                gpu_memory_used = max(0.0, final_gpu_memory - initial_gpu_memory)
            
            # Store metrics
            metrics = ModelMetrics(
                model_id=model_id,
                training_time=training_time,
                gpu_memory_used=gpu_memory_used
            )
            
            self.model_metrics_history[model_id].append(metrics)
            
            logger.info(f"Model {model_id} training completed in {training_time:.2f}s, "
                       f"GPU memory: {gpu_memory_used:.1f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def profile_model_inference(self, model_id: str, inference_function: Callable, 
                              data: Any, batch_size: Optional[int] = None) -> Any:
        """Profile model inference performance."""
        if not self.enable_model_profiling:
            return inference_function(data)
        
        # Determine data size
        if hasattr(data, '__len__'):
            data_size = len(data)
        elif hasattr(data, 'shape'):
            data_size = data.shape[0]
        else:
            data_size = 1
        
        start_time = time.time()
        
        try:
            # Execute inference
            result = inference_function(data)
            
            # Calculate metrics
            end_time = time.time()
            inference_time = end_time - start_time
            throughput = data_size / inference_time if inference_time > 0 else 0
            
            # Store metrics
            metrics = ModelMetrics(
                model_id=model_id,
                inference_time=inference_time,
                throughput=throughput
            )
            
            self.model_metrics_history[model_id].append(metrics)
            
            logger.debug(f"Model {model_id} inference: {inference_time:.4f}s, "
                        f"throughput: {throughput:.1f} samples/s")
            
            return result
            
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise
    
    def _get_current_gpu_memory(self) -> Optional[float]:
        """Get current GPU memory usage."""
        if not self.gpu_initialized or not self.available_gpus:
            return None
        
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            elif TF_AVAILABLE:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    # TensorFlow memory usage is harder to track
                    return 0.0
        except:
            pass
        
        return None
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        current_metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": {},
            "gpus": {},
            "models": {}
        }
        
        # Latest system metrics
        if self.system_metrics_history:
            latest_system = self.system_metrics_history[-1]
            current_metrics["system"] = {
                "cpu_percent": latest_system.cpu_percent,
                "memory_percent": latest_system.memory_percent,
                "memory_used_mb": latest_system.memory_used,
                "disk_io_read_mb_s": latest_system.disk_io_read,
                "disk_io_write_mb_s": latest_system.disk_io_write,
                "network_sent_mb_s": latest_system.network_sent,
                "network_recv_mb_s": latest_system.network_recv
            }
        
        # Latest GPU metrics
        for gpu_id, gpu_history in self.gpu_metrics_history.items():
            if gpu_history:
                latest_gpu = gpu_history[-1]
                current_metrics["gpus"][gpu_id] = {
                    "name": latest_gpu.name,
                    "utilization": latest_gpu.utilization,
                    "memory_used_mb": latest_gpu.memory_used,
                    "memory_utilization": latest_gpu.memory_utilization,
                    "temperature": latest_gpu.temperature,
                    "power_draw": latest_gpu.power_draw
                }
        
        # Model metrics summary
        for model_id, model_history in self.model_metrics_history.items():
            if model_history:
                latest_model = model_history[-1]
                current_metrics["models"][model_id] = {
                    "training_time": latest_model.training_time,
                    "inference_time": latest_model.inference_time,
                    "throughput": latest_model.throughput,
                    "gpu_memory_used": latest_model.gpu_memory_used
                }
        
        return current_metrics
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        summary = {
            "period_hours": hours,
            "system_summary": {},
            "gpu_summary": {},
            "model_summary": {}
        }
        
        # System summary
        recent_system_metrics = [
            m for m in self.system_metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        if recent_system_metrics:
            cpu_values = [m.cpu_percent for m in recent_system_metrics]
            memory_values = [m.memory_percent for m in recent_system_metrics]
            
            summary["system_summary"] = {
                "avg_cpu_percent": np.mean(cpu_values),
                "max_cpu_percent": np.max(cpu_values),
                "avg_memory_percent": np.mean(memory_values),
                "max_memory_percent": np.max(memory_values),
                "sample_count": len(recent_system_metrics)
            }
        
        # GPU summary
        for gpu_id, gpu_history in self.gpu_metrics_history.items():
            recent_gpu_metrics = [
                m for m in gpu_history 
                if m.timestamp > cutoff_time
            ]
            
            if recent_gpu_metrics:
                util_values = [m.utilization for m in recent_gpu_metrics]
                memory_values = [m.memory_utilization for m in recent_gpu_metrics]
                temp_values = [m.temperature for m in recent_gpu_metrics]
                
                summary["gpu_summary"][gpu_id] = {
                    "name": recent_gpu_metrics[0].name,
                    "avg_utilization": np.mean(util_values),
                    "max_utilization": np.max(util_values),
                    "avg_memory_utilization": np.mean(memory_values),
                    "max_memory_utilization": np.max(memory_values),
                    "avg_temperature": np.mean(temp_values),
                    "max_temperature": np.max(temp_values),
                    "sample_count": len(recent_gpu_metrics)
                }
        
        # Model summary
        for model_id, model_history in self.model_metrics_history.items():
            recent_model_metrics = [
                m for m in model_history 
                if m.timestamp > cutoff_time
            ]
            
            if recent_model_metrics:
                training_times = [m.training_time for m in recent_model_metrics if m.training_time]
                inference_times = [m.inference_time for m in recent_model_metrics if m.inference_time]
                throughputs = [m.throughput for m in recent_model_metrics if m.throughput]
                
                summary["model_summary"][model_id] = {
                    "training_count": len(training_times),
                    "avg_training_time": np.mean(training_times) if training_times else None,
                    "inference_count": len(inference_times),
                    "avg_inference_time": np.mean(inference_times) if inference_times else None,
                    "avg_throughput": np.mean(throughputs) if throughputs else None
                }
        
        return summary
    
    def add_alert_callback(self, callback: Callable[[str, SystemMetrics], None]) -> None:
        """Add callback for performance alerts."""
        self._alert_callbacks.append(callback)
    
    def optimize_gpu_memory(self) -> Dict[str, Any]:
        """Optimize GPU memory usage."""
        optimization_result = {
            "actions_taken": [],
            "memory_freed_mb": 0.0,
            "recommendations": []
        }
        
        try:
            # PyTorch optimization
            if TORCH_AVAILABLE and torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()
                
                # Clear cache
                torch.cuda.empty_cache()
                
                final_memory = torch.cuda.memory_allocated()
                freed_memory = (initial_memory - final_memory) / (1024 * 1024)
                
                optimization_result["actions_taken"].append("Cleared PyTorch CUDA cache")
                optimization_result["memory_freed_mb"] += freed_memory
            
            # TensorFlow optimization
            if TF_AVAILABLE:
                try:
                    # Enable memory growth
                    gpus = tf.config.list_physical_devices('GPU')
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    optimization_result["actions_taken"].append("Enabled TensorFlow memory growth")
                except:
                    pass
            
            # Add recommendations
            optimization_result["recommendations"].extend([
                "Use gradient checkpointing for large models",
                "Implement model parallelism for very large models",
                "Use mixed precision training to reduce memory usage",
                "Consider using smaller batch sizes during training"
            ])
            
        except Exception as e:
            logger.error(f"GPU memory optimization failed: {e}")
        
        return optimization_result
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()