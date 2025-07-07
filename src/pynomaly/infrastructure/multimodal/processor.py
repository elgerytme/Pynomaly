"""Multi-modal data processing and feature extraction pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.models.multimodal import (
    EncodingType,
    FusionLayer,
    FusionStrategy,
    ModalityConfig,
    ModalityEncoder,
    ModalityType,
    MultiModalData,
    MultiModalDetector,
)


class MultiModalProcessor:
    """Processor for handling multi-modal data streams."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Processing state
        self.registered_processors: Dict[ModalityType, Any] = {}
        self.processing_cache: Dict[str, np.ndarray] = {}
        self.max_cache_size = 1000
        
        # Performance tracking
        self.processing_stats: Dict[ModalityType, Dict[str, float]] = {}

    def register_modality_processor(
        self, 
        modality_type: ModalityType, 
        processor: Any
    ) -> None:
        """Register a processor for specific modality."""
        self.registered_processors[modality_type] = processor
        
        # Initialize stats
        self.processing_stats[modality_type] = {
            "total_processed": 0,
            "total_time": 0.0,
            "average_time": 0.0,
        }
        
        self.logger.info(f"Registered processor for modality: {modality_type.value}")

    async def process_tabular_data(
        self, 
        data: np.ndarray, 
        config: ModalityConfig
    ) -> np.ndarray:
        """Process tabular/structured data."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            processed_data = data.copy()
            
            # Apply preprocessing
            if config.preprocessing_params.get("normalize", False):
                # Min-max normalization
                data_min = np.min(processed_data, axis=0)
                data_max = np.max(processed_data, axis=0)
                data_range = data_max - data_min
                data_range[data_range == 0] = 1  # Avoid division by zero
                processed_data = (processed_data - data_min) / data_range
            
            if config.preprocessing_params.get("standardize", False):
                # Z-score standardization
                mean = np.mean(processed_data, axis=0)
                std = np.std(processed_data, axis=0)
                std[std == 0] = 1  # Avoid division by zero
                processed_data = (processed_data - mean) / std
            
            # Handle missing values
            if config.preprocessing_params.get("handle_missing", True):
                # Replace NaN with median
                for col in range(processed_data.shape[1]):
                    column_data = processed_data[:, col]
                    if np.any(np.isnan(column_data)):
                        median_val = np.nanmedian(column_data)
                        processed_data[:, col] = np.where(
                            np.isnan(column_data), median_val, column_data
                        )
            
            # Feature selection
            max_features = config.preprocessing_params.get("max_features")
            if max_features and processed_data.shape[1] > max_features:
                # Simple variance-based feature selection
                variances = np.var(processed_data, axis=0)
                top_indices = np.argsort(variances)[-max_features:]
                processed_data = processed_data[:, top_indices]
            
            return processed_data
            
        finally:
            # Update stats
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_processing_stats(ModalityType.TABULAR, processing_time)

    async def process_time_series_data(
        self, 
        data: np.ndarray, 
        config: ModalityConfig
    ) -> np.ndarray:
        """Process time series data."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            
            processed_data = data.copy()
            
            # Window-based processing
            window_size = config.preprocessing_params.get("window_size", 100)
            overlap = config.preprocessing_params.get("overlap", 0.5)
            
            if len(processed_data) > window_size:
                windows = self._create_sliding_windows(processed_data, window_size, overlap)
                processed_data = windows
            
            # Normalization
            normalization = config.preprocessing_params.get("normalization", "none")
            if normalization == "z_score":
                mean = np.mean(processed_data, axis=0)
                std = np.std(processed_data, axis=0)
                std[std == 0] = 1
                processed_data = (processed_data - mean) / std
            elif normalization == "min_max":
                data_min = np.min(processed_data, axis=0)
                data_max = np.max(processed_data, axis=0)
                data_range = data_max - data_min
                data_range[data_range == 0] = 1
                processed_data = (processed_data - data_min) / data_range
            
            # Feature extraction
            if config.preprocessing_params.get("extract_features", True):
                processed_data = self._extract_time_series_features(processed_data)
            
            return processed_data
            
        finally:
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_processing_stats(ModalityType.TIME_SERIES, processing_time)

    async def process_text_data(
        self, 
        data: str, 
        config: ModalityConfig
    ) -> np.ndarray:
        """Process text data."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            processed_text = data
            
            # Text preprocessing
            if config.preprocessing_params.get("lowercase", True):
                processed_text = processed_text.lower()
            
            if config.preprocessing_params.get("remove_punctuation", True):
                import string
                processed_text = processed_text.translate(
                    str.maketrans("", "", string.punctuation)
                )
            
            if config.preprocessing_params.get("remove_stopwords", False):
                # Simple stopword removal
                stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
                words = processed_text.split()
                processed_text = " ".join([word for word in words if word not in stopwords])
            
            # Length limiting
            max_length = config.preprocessing_params.get("max_length", 512)
            if len(processed_text) > max_length:
                processed_text = processed_text[:max_length]
            
            # Feature extraction
            if config.encoding_type == EncodingType.TFIDF:
                features = self._extract_tfidf_features(processed_text)
            elif config.encoding_type == EncodingType.WORD2VEC:
                features = self._extract_word2vec_features(processed_text)
            else:
                # Default: character-level features
                features = self._extract_char_features(processed_text)
            
            return features
            
        finally:
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_processing_stats(ModalityType.TEXT, processing_time)

    async def process_image_data(
        self, 
        data: np.ndarray, 
        config: ModalityConfig
    ) -> np.ndarray:
        """Process image data."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            processed_image = data.copy()
            
            # Resize if specified
            target_size = config.preprocessing_params.get("resize")
            if target_size and len(processed_image.shape) >= 2:
                # Simple resize using numpy (in practice would use cv2 or PIL)
                processed_image = self._simple_resize(processed_image, target_size)
            
            # Normalization
            if config.preprocessing_params.get("normalize", True):
                processed_image = processed_image.astype(np.float32) / 255.0
            
            # Data augmentation (if enabled)
            if config.preprocessing_params.get("augmentation", False):
                processed_image = self._apply_image_augmentation(processed_image)
            
            # Feature extraction
            if config.encoding_type == EncodingType.CNN:
                features = self._extract_cnn_features(processed_image)
            elif config.encoding_type == EncodingType.RESNET:
                features = self._extract_resnet_features(processed_image)
            else:
                # Default: flatten image
                features = processed_image.flatten()
            
            return features
            
        finally:
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_processing_stats(ModalityType.IMAGE, processing_time)

    async def process_audio_data(
        self, 
        data: np.ndarray, 
        config: ModalityConfig
    ) -> np.ndarray:
        """Process audio data."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            processed_audio = data.copy()
            
            # Audio preprocessing
            sample_rate = config.preprocessing_params.get("sample_rate", 22050)
            
            # Normalization
            if config.preprocessing_params.get("normalize", True):
                processed_audio = processed_audio / np.max(np.abs(processed_audio))
            
            # Feature extraction
            if config.encoding_type == EncodingType.MFCC:
                features = self._extract_mfcc_features(processed_audio, sample_rate)
            elif config.encoding_type == EncodingType.SPECTROGRAM:
                features = self._extract_spectrogram_features(processed_audio)
            elif config.encoding_type == EncodingType.MEL_SPECTROGRAM:
                features = self._extract_mel_spectrogram_features(processed_audio, sample_rate)
            else:
                # Default: statistical features
                features = self._extract_audio_statistical_features(processed_audio)
            
            return features
            
        finally:
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_processing_stats(ModalityType.AUDIO, processing_time)

    async def process_multimodal_sample(
        self, 
        sample: MultiModalData, 
        detector: MultiModalDetector
    ) -> Dict[ModalityType, np.ndarray]:
        """Process all modalities in a multi-modal sample."""
        processed_features = {}
        
        # Process each modality
        processing_tasks = []
        
        for modality_type in sample.get_available_modalities():
            if modality_type in detector.modality_configs:
                config = detector.modality_configs[modality_type]
                raw_data = sample.get_modality_data(modality_type)
                
                # Create processing task
                task = self._process_modality_async(modality_type, raw_data, config)
                processing_tasks.append((modality_type, task))
        
        # Wait for all processing to complete
        for modality_type, task in processing_tasks:
            try:
                processed_data = await task
                processed_features[modality_type] = processed_data
            except Exception as e:
                self.logger.error(f"Error processing {modality_type.value}: {e}")
                # Continue with other modalities
        
        return processed_features

    async def _process_modality_async(
        self, 
        modality_type: ModalityType, 
        data: Any, 
        config: ModalityConfig
    ) -> np.ndarray:
        """Process single modality asynchronously."""
        
        if modality_type == ModalityType.TABULAR:
            return await self.process_tabular_data(data, config)
        elif modality_type == ModalityType.TIME_SERIES:
            return await self.process_time_series_data(data, config)
        elif modality_type == ModalityType.TEXT:
            return await self.process_text_data(data, config)
        elif modality_type == ModalityType.IMAGE:
            return await self.process_image_data(data, config)
        elif modality_type == ModalityType.AUDIO:
            return await self.process_audio_data(data, config)
        else:
            # Default processing
            if isinstance(data, np.ndarray):
                return data.flatten()
            else:
                return np.array([float(data)])

    def _create_sliding_windows(
        self, 
        data: np.ndarray, 
        window_size: int, 
        overlap: float
    ) -> np.ndarray:
        """Create sliding windows from time series data."""
        step_size = int(window_size * (1 - overlap))
        step_size = max(1, step_size)
        
        windows = []
        for i in range(0, len(data) - window_size + 1, step_size):
            window = data[i:i + window_size]
            windows.append(window.flatten())
        
        if not windows:
            return data.flatten().reshape(1, -1)
        
        return np.array(windows)

    def _extract_time_series_features(self, data: np.ndarray) -> np.ndarray:
        """Extract statistical features from time series."""
        if len(data.shape) == 3:  # Multiple windows
            features = []
            for window in data:
                window_features = self._extract_single_window_features(window)
                features.append(window_features)
            return np.array(features).flatten()
        else:
            return self._extract_single_window_features(data)

    def _extract_single_window_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features from single time series window."""
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(data),
            np.std(data),
            np.min(data),
            np.max(data),
            np.median(data),
        ])
        
        # Distribution features
        features.extend([
            np.percentile(data, 25),
            np.percentile(data, 75),
            np.var(data),
        ])
        
        # Trend features
        if len(data) > 1:
            diff = np.diff(data.flatten())
            features.extend([
                np.mean(diff),
                np.std(diff),
                np.sum(diff > 0) / len(diff),  # Positive trend ratio
            ])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)

    def _extract_tfidf_features(self, text: str, max_features: int = 1000) -> np.ndarray:
        """Extract TF-IDF features from text."""
        words = text.split()
        
        # Simple TF-IDF implementation
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # TF calculation
        total_words = len(words)
        tf_features = np.zeros(max_features)
        
        for i, word in enumerate(word_counts.keys()):
            if i >= max_features:
                break
            tf = word_counts[word] / total_words
            tf_features[i] = tf
        
        return tf_features

    def _extract_word2vec_features(self, text: str, embedding_dim: int = 300) -> np.ndarray:
        """Extract Word2Vec-like features from text."""
        words = text.split()
        
        # Simple word embedding simulation
        features = np.zeros(embedding_dim)
        
        for word in words:
            # Hash-based pseudo-embedding
            word_hash = hash(word) % embedding_dim
            features[word_hash] += 1
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features

    def _extract_char_features(self, text: str, max_chars: int = 256) -> np.ndarray:
        """Extract character-level features from text."""
        features = np.zeros(max_chars)
        
        for i, char in enumerate(text[:max_chars]):
            features[i] = ord(char) / 255.0  # Normalize to [0, 1]
        
        return features

    def _simple_resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Simple image resizing (bilinear interpolation approximation)."""
        if len(image.shape) == 2:
            # Grayscale
            height, width = image.shape
            target_height, target_width = target_size
            
            # Simple downsampling/upsampling
            height_ratio = height / target_height
            width_ratio = width / target_width
            
            resized = np.zeros((target_height, target_width))
            
            for i in range(target_height):
                for j in range(target_width):
                    orig_i = min(int(i * height_ratio), height - 1)
                    orig_j = min(int(j * width_ratio), width - 1)
                    resized[i, j] = image[orig_i, orig_j]
            
            return resized
        else:
            # Color image - resize each channel
            channels = []
            for c in range(image.shape[2]):
                channel_resized = self._simple_resize(image[:, :, c], target_size)
                channels.append(channel_resized)
            
            return np.stack(channels, axis=2)

    def _apply_image_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply simple image augmentation."""
        augmented = image.copy()
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        augmented = np.clip(augmented * brightness_factor, 0, 1)
        
        # Random noise
        noise = np.random.normal(0, 0.01, augmented.shape)
        augmented = np.clip(augmented + noise, 0, 1)
        
        return augmented

    def _extract_cnn_features(self, image: np.ndarray, feature_dim: int = 512) -> np.ndarray:
        """Extract CNN-like features from image."""
        # Simplified CNN feature extraction
        
        # Apply simple filters (edge detection, etc.)
        features = []
        
        if len(image.shape) >= 2:
            # Horizontal edge filter
            horizontal_filter = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            if image.shape[0] >= 3 and image.shape[1] >= 3:
                h_edges = self._apply_filter(image, horizontal_filter)
                features.append(np.mean(h_edges))
                features.append(np.std(h_edges))
            
            # Vertical edge filter
            vertical_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            if image.shape[0] >= 3 and image.shape[1] >= 3:
                v_edges = self._apply_filter(image, vertical_filter)
                features.append(np.mean(v_edges))
                features.append(np.std(v_edges))
        
        # Add global statistics
        features.extend([
            np.mean(image),
            np.std(image),
            np.min(image),
            np.max(image),
        ])
        
        # Pad or truncate to desired dimension
        features = np.array(features)
        if len(features) > feature_dim:
            features = features[:feature_dim]
        elif len(features) < feature_dim:
            features = np.pad(features, (0, feature_dim - len(features)))
        
        return features

    def _apply_filter(self, image: np.ndarray, filter_kernel: np.ndarray) -> np.ndarray:
        """Apply convolution filter to image."""
        if len(image.shape) == 3:
            # Apply to first channel only for simplicity
            image = image[:, :, 0]
        
        filtered = np.zeros_like(image)
        k_height, k_width = filter_kernel.shape
        
        for i in range(k_height//2, image.shape[0] - k_height//2):
            for j in range(k_width//2, image.shape[1] - k_width//2):
                region = image[i-k_height//2:i+k_height//2+1, j-k_width//2:j+k_width//2+1]
                filtered[i, j] = np.sum(region * filter_kernel)
        
        return filtered

    def _extract_resnet_features(self, image: np.ndarray, feature_dim: int = 2048) -> np.ndarray:
        """Extract ResNet-like features from image."""
        # Simplified ResNet feature extraction
        features = self._extract_cnn_features(image, feature_dim // 4)
        
        # Add more complex features (simulated)
        complex_features = []
        
        # Multi-scale features
        if len(image.shape) >= 2:
            # Different pooling sizes
            for pool_size in [2, 4, 8]:
                if image.shape[0] >= pool_size and image.shape[1] >= pool_size:
                    pooled = self._max_pool(image, pool_size)
                    complex_features.extend([np.mean(pooled), np.std(pooled)])
        
        # Combine features
        all_features = np.concatenate([features, np.array(complex_features)])
        
        # Adjust to desired dimension
        if len(all_features) > feature_dim:
            all_features = all_features[:feature_dim]
        elif len(all_features) < feature_dim:
            all_features = np.pad(all_features, (0, feature_dim - len(all_features)))
        
        return all_features

    def _max_pool(self, image: np.ndarray, pool_size: int) -> np.ndarray:
        """Apply max pooling to image."""
        if len(image.shape) == 3:
            image = image[:, :, 0]  # First channel only
        
        height, width = image.shape
        pooled_height = height // pool_size
        pooled_width = width // pool_size
        
        pooled = np.zeros((pooled_height, pooled_width))
        
        for i in range(pooled_height):
            for j in range(pooled_width):
                region = image[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
                pooled[i, j] = np.max(region)
        
        return pooled

    def _extract_mfcc_features(
        self, 
        audio: np.ndarray, 
        sample_rate: int, 
        n_mfcc: int = 13
    ) -> np.ndarray:
        """Extract MFCC features from audio."""
        # Simplified MFCC extraction
        
        # Frame the audio
        frame_length = 1024
        hop_length = 512
        
        frames = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            frames.append(frame)
        
        if not frames:
            return np.zeros(n_mfcc)
        
        # Compute features for each frame
        mfcc_features = []
        
        for frame in frames:
            # Simple spectral features (approximating MFCC)
            # In practice, this would use proper MFCC computation
            
            # FFT
            fft = np.abs(np.fft.fft(frame))
            fft = fft[:len(fft)//2]  # Keep only positive frequencies
            
            # Log power spectrum
            log_power = np.log(fft + 1e-8)
            
            # DCT (simplified)
            dct_features = self._simple_dct(log_power, n_mfcc)
            mfcc_features.append(dct_features)
        
        if mfcc_features:
            # Average across frames
            return np.mean(mfcc_features, axis=0)
        else:
            return np.zeros(n_mfcc)

    def _simple_dct(self, x: np.ndarray, n_coeffs: int) -> np.ndarray:
        """Simple DCT implementation."""
        N = len(x)
        dct_coeffs = np.zeros(min(n_coeffs, N))
        
        for k in range(min(n_coeffs, N)):
            sum_val = 0.0
            for n in range(N):
                sum_val += x[n] * np.cos(np.pi * k * (2*n + 1) / (2*N))
            dct_coeffs[k] = sum_val
        
        return dct_coeffs

    def _extract_spectrogram_features(self, audio: np.ndarray, n_fft: int = 1024) -> np.ndarray:
        """Extract spectrogram features from audio."""
        # Simple spectrogram computation
        hop_length = n_fft // 4
        
        spectrograms = []
        for i in range(0, len(audio) - n_fft, hop_length):
            frame = audio[i:i + n_fft]
            fft = np.abs(np.fft.fft(frame))
            fft = fft[:len(fft)//2]  # Keep positive frequencies
            spectrograms.append(fft)
        
        if spectrograms:
            spectrogram = np.array(spectrograms).T
            # Extract statistical features
            features = [
                np.mean(spectrogram),
                np.std(spectrogram),
                np.max(spectrogram),
                np.min(spectrogram),
            ]
            
            # Add frequency band features
            n_bands = 10
            band_size = spectrogram.shape[0] // n_bands
            
            for i in range(n_bands):
                start_idx = i * band_size
                end_idx = min((i + 1) * band_size, spectrogram.shape[0])
                band_energy = np.mean(spectrogram[start_idx:end_idx])
                features.append(band_energy)
            
            return np.array(features)
        else:
            return np.zeros(14)  # 4 basic + 10 band features

    def _extract_mel_spectrogram_features(
        self, 
        audio: np.ndarray, 
        sample_rate: int, 
        n_mels: int = 80
    ) -> np.ndarray:
        """Extract mel-spectrogram features from audio."""
        # Simplified mel-spectrogram
        spectrogram_features = self._extract_spectrogram_features(audio)
        
        # Apply mel filter bank (simplified)
        mel_features = np.zeros(n_mels)
        
        # Map spectrogram features to mel scale
        for i in range(min(len(spectrogram_features), n_mels)):
            mel_features[i] = spectrogram_features[i]
        
        return mel_features

    def _extract_audio_statistical_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract statistical features from audio signal."""
        features = []
        
        # Time domain features
        features.extend([
            np.mean(audio),
            np.std(audio),
            np.min(audio),
            np.max(audio),
            np.median(audio),
            np.var(audio),
        ])
        
        # Energy features
        energy = np.sum(audio ** 2)
        features.append(energy)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        features.append(zero_crossings / len(audio))
        
        # Spectral centroid (simplified)
        fft = np.abs(np.fft.fft(audio))
        freqs = np.fft.fftfreq(len(audio))
        spectral_centroid = np.sum(freqs * fft) / np.sum(fft) if np.sum(fft) > 0 else 0
        features.append(spectral_centroid)
        
        return np.array(features)

    def _update_processing_stats(self, modality_type: ModalityType, processing_time: float) -> None:
        """Update processing statistics for modality."""
        stats = self.processing_stats[modality_type]
        stats["total_processed"] += 1
        stats["total_time"] += processing_time
        stats["average_time"] = stats["total_time"] / stats["total_processed"]

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        return {
            "modality_stats": {
                modality.value: stats 
                for modality, stats in self.processing_stats.items()
            },
            "cache_stats": {
                "cache_size": len(self.processing_cache),
                "max_cache_size": self.max_cache_size,
                "cache_utilization": len(self.processing_cache) / self.max_cache_size,
            },
            "registered_processors": [
                modality.value for modality in self.registered_processors.keys()
            ],
        }

    def clear_cache(self) -> None:
        """Clear processing cache."""
        cache_size = len(self.processing_cache)
        self.processing_cache.clear()
        self.logger.info(f"Cleared processing cache ({cache_size} entries)")