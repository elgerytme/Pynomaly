"""Text anomaly detection using NLP-based features."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import numpy as np
import numpy.typing as npt
import re
from collections import Counter
from dataclasses import dataclass

from simplified_services.core_detection_service import DetectionResult


@dataclass
class TextConfig:
    """Configuration for text anomaly detection."""
    contamination: float = 0.1
    min_text_length: int = 10
    max_features: int = 1000
    use_tfidf: bool = True
    detect_language_anomalies: bool = True
    detect_length_anomalies: bool = True
    detect_character_anomalies: bool = True


class TextAnomalyDetector:
    """Text anomaly detection using statistical and linguistic features.
    
    This detector identifies anomalous text documents based on:
    - Statistical text features (length, character distributions)
    - Linguistic patterns (vocabulary, n-grams)
    - Encoding and format anomalies
    - Language detection
    """

    def __init__(self, config: Optional[TextConfig] = None):
        """Initialize text anomaly detector."""
        self.config = config or TextConfig()
        self._vocabulary: Set[str] = set()
        self._char_frequencies: Dict[str, float] = {}
        self._trained = False

    def detect_anomalies(
        self,
        texts: List[str],
        **kwargs: Any
    ) -> DetectionResult:
        """Detect anomalies in text documents.
        
        Args:
            texts: List of text documents
            **kwargs: Additional parameters
            
        Returns:
            DetectionResult with text anomaly predictions
        """
        if not texts:
            return DetectionResult(
                predictions=np.array([], dtype=int),
                algorithm="text_anomaly",
                contamination=self.config.contamination
            )
        
        # Extract features from texts
        features = self._extract_text_features(texts)
        
        # Use statistical detection on extracted features
        from simplified_services.core_detection_service import CoreDetectionService
        detection_service = CoreDetectionService()
        
        result = detection_service.detect_anomalies(
            features,
            algorithm="iforest",
            contamination=self.config.contamination
        )
        
        # Add text-specific metadata
        result.algorithm = "text_anomaly"
        result.metadata.update({
            "method": "text_features",
            "n_features": features.shape[1],
            "avg_text_length": np.mean([len(text) for text in texts]),
            "unique_chars": len(self._char_frequencies),
            "vocabulary_size": len(self._vocabulary)
        })
        
        return result

    def _extract_text_features(self, texts: List[str]) -> npt.NDArray[np.floating]:
        """Extract numerical features from text documents."""
        features = []
        
        # Build vocabulary and character frequencies if not trained
        if not self._trained:
            self._build_vocabulary(texts)
            self._build_char_frequencies(texts)
            self._trained = True
        
        for text in texts:
            feature_vector = []
            
            # Basic length features
            feature_vector.extend(self._extract_length_features(text))
            
            # Character distribution features
            feature_vector.extend(self._extract_character_features(text))
            
            # Linguistic features
            feature_vector.extend(self._extract_linguistic_features(text))
            
            # Format features
            feature_vector.extend(self._extract_format_features(text))
            
            features.append(feature_vector)
        
        return np.array(features)

    def _extract_length_features(self, text: str) -> List[float]:
        """Extract length-based features."""
        words = text.split()
        sentences = text.split('.')
        
        return [
            len(text),                           # Total characters
            len(words),                          # Word count
            len(sentences),                      # Sentence count
            np.mean([len(word) for word in words]) if words else 0.0,  # Avg word length
            np.mean([len(sent) for sent in sentences]) if sentences else 0.0,  # Avg sentence length
        ]

    def _extract_character_features(self, text: str) -> List[float]:
        """Extract character distribution features."""
        if not text:
            return [0.0] * 10
        
        char_counts = Counter(text.lower())
        total_chars = len(text)
        
        features = [
            sum(1 for c in text if c.isalpha()) / total_chars,    # Alpha ratio
            sum(1 for c in text if c.isdigit()) / total_chars,    # Digit ratio
            sum(1 for c in text if c.isspace()) / total_chars,    # Space ratio
            sum(1 for c in text if c in '.,!?;:') / total_chars,  # Punctuation ratio
            sum(1 for c in text if c.isupper()) / total_chars,    # Uppercase ratio
        ]
        
        # Character frequency deviation from expected
        expected_frequencies = self._char_frequencies
        if expected_frequencies:
            deviations = []
            for char, expected_freq in list(expected_frequencies.items())[:5]:  # Top 5 chars
                actual_freq = char_counts.get(char, 0) / total_chars
                deviation = abs(actual_freq - expected_freq)
                deviations.append(deviation)
            
            features.extend(deviations)
        else:
            features.extend([0.0] * 5)
        
        return features

    def _extract_linguistic_features(self, text: str) -> List[float]:
        """Extract linguistic and vocabulary features."""
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return [0.0] * 5
        
        # Vocabulary overlap with training set
        word_set = set(words)
        vocabulary_overlap = len(word_set.intersection(self._vocabulary)) / len(word_set) if word_set else 0.0
        
        # Unique word ratio
        unique_word_ratio = len(set(words)) / len(words)
        
        # Average word frequency (simplified)
        word_freq_sum = sum(1 for word in words if word in self._vocabulary)
        avg_word_freq = word_freq_sum / len(words)
        
        # Repetition features
        word_counts = Counter(words)
        max_repetition = max(word_counts.values()) if word_counts else 0
        repetition_ratio = max_repetition / len(words)
        
        return [
            vocabulary_overlap,
            unique_word_ratio,
            avg_word_freq,
            max_repetition,
            repetition_ratio
        ]

    def _extract_format_features(self, text: str) -> List[float]:
        """Extract format and encoding features."""
        # Line breaks and formatting
        line_breaks = text.count('\n') / len(text) if text else 0.0
        tab_count = text.count('\t') / len(text) if text else 0.0
        
        # Special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_char_ratio = special_chars / len(text) if text else 0.0
        
        # URL/email patterns
        url_count = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        email_count = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        
        # Number patterns
        number_count = len(re.findall(r'\b\d+\b', text))
        number_ratio = number_count / len(text.split()) if text.split() else 0.0
        
        return [
            line_breaks,
            tab_count,
            special_char_ratio,
            url_count,
            email_count,
            number_ratio
        ]

    def _build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary from training texts."""
        word_counts = Counter()
        
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts.update(words)
        
        # Keep most common words up to max_features
        most_common = word_counts.most_common(self.config.max_features)
        self._vocabulary = {word for word, count in most_common}

    def _build_char_frequencies(self, texts: List[str]) -> None:
        """Build character frequency distribution from training texts."""
        all_chars = ''.join(texts).lower()
        char_counts = Counter(all_chars)
        total_chars = len(all_chars)
        
        # Convert to frequencies
        self._char_frequencies = {
            char: count / total_chars 
            for char, count in char_counts.most_common(50)  # Top 50 characters
        }

    def detect_language_anomalies(self, texts: List[str]) -> List[bool]:
        """Detect language-based anomalies (simplified language detection)."""
        anomalies = []
        
        # Simple language detection based on character patterns
        for text in texts:
            is_anomaly = False
            
            # Check for non-printable characters
            if any(ord(c) > 127 for c in text):
                is_anomaly = True
            
            # Check for unusual character patterns
            alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text) if text else 0
            if alpha_ratio < 0.3:  # Very low alphabetic content
                is_anomaly = True
            
            anomalies.append(is_anomaly)
        
        return anomalies

    def detect_encoding_anomalies(self, texts: List[str]) -> List[bool]:
        """Detect encoding and format anomalies."""
        anomalies = []
        
        for text in texts:
            is_anomaly = False
            
            # Check for encoding issues (simplified)
            try:
                text.encode('utf-8')
            except UnicodeEncodeError:
                is_anomaly = True
            
            # Check for unusual control characters
            control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
            if control_chars > 0:
                is_anomaly = True
            
            # Check for very unusual length
            if len(text) < self.config.min_text_length or len(text) > 100000:
                is_anomaly = True
            
            anomalies.append(is_anomaly)
        
        return anomalies

    def get_feature_importance(self, texts: List[str]) -> Dict[str, float]:
        """Get importance scores for different feature types."""
        if not texts:
            return {}
        
        features = self._extract_text_features(texts)
        
        # Calculate variance for each feature type
        feature_names = [
            "char_count", "word_count", "sentence_count", "avg_word_len", "avg_sent_len",
            "alpha_ratio", "digit_ratio", "space_ratio", "punct_ratio", "upper_ratio",
            "char_dev_1", "char_dev_2", "char_dev_3", "char_dev_4", "char_dev_5",
            "vocab_overlap", "unique_word_ratio", "avg_word_freq", "max_repetition", "repetition_ratio",
            "line_breaks", "tab_count", "special_char_ratio", "url_count", "email_count", "number_ratio"
        ]
        
        importance_scores = {}
        for i, name in enumerate(feature_names[:features.shape[1]]):
            if features.shape[1] > i:
                importance_scores[name] = float(np.var(features[:, i]))
        
        return importance_scores

    def reset(self) -> None:
        """Reset detector state."""
        self._vocabulary.clear()
        self._char_frequencies.clear()
        self._trained = False