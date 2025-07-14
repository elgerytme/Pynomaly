"""Tests for text anomaly detection adapter."""

import pytest

from pynomaly.domain.value_objects import AnomalyType
from pynomaly.infrastructure.adapters.text_adapter import (
    SentimentAnalyzer,
    TextAnomalyDetector,
    TextDetectionConfig,
    TextEncoder,
    TextPreprocessor,
    create_text_detector,
)


class TestTextDetectionConfig:
    """Test text detection configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TextDetectionConfig()
        assert config.algorithm == "isolation_forest"
        assert config.encoding_method == "tfidf"
        assert config.contamination == 0.1
        assert config.lowercase is True
        assert config.remove_stopwords is True

    def test_invalid_algorithm_raises_error(self):
        """Test that invalid algorithm raises error."""
        with pytest.raises(ValueError, match="Algorithm must be one of"):
            TextDetectionConfig(algorithm="invalid_algorithm")

    def test_invalid_encoding_method_raises_error(self):
        """Test that invalid encoding method raises error."""
        with pytest.raises(ValueError, match="Encoding method must be one of"):
            TextDetectionConfig(encoding_method="invalid_method")


class TestTextPreprocessor:
    """Test text preprocessing functionality."""

    @pytest.fixture
    def preprocessor(self):
        """Create text preprocessor."""
        config = TextDetectionConfig()
        return TextPreprocessor(config)

    def test_preprocess_text_basic(self, preprocessor):
        """Test basic text preprocessing."""
        text = "This is a TEST document with PUNCTUATION!"
        result = preprocessor.preprocess_text(text)

        # Should be lowercase, no punctuation, no short words
        expected_words = ["this", "test", "document", "with", "punctuation"]
        assert all(word in result for word in expected_words)
        assert result.islower()
        assert "!" not in result

    def test_preprocess_text_stopwords_removed(self, preprocessor):
        """Test that stop words are removed."""
        text = "This is a document with many stop words"
        result = preprocessor.preprocess_text(text)

        # Stop words should be removed
        stop_words = ["this", "is", "a", "with"]
        for stop_word in stop_words:
            assert stop_word not in result.split()

    def test_preprocess_corpus(self, preprocessor):
        """Test preprocessing multiple documents."""
        texts = [
            "First document with content",
            "Second document with different content",
            "Third document with similar content",
        ]

        results = preprocessor.preprocess_corpus(texts)
        assert len(results) == len(texts)
        assert all(isinstance(result, str) for result in results)


class TestTextEncoder:
    """Test text encoding functionality."""

    @pytest.fixture
    def tfidf_encoder(self):
        """Create TF-IDF encoder."""
        config = TextDetectionConfig(encoding_method="tfidf")
        return TextEncoder(config)

    @pytest.fixture
    def count_encoder(self):
        """Create count encoder."""
        config = TextDetectionConfig(encoding_method="count")
        return TextEncoder(config)

    def test_tfidf_encoding(self, tfidf_encoder):
        """Test TF-IDF encoding."""
        texts = [
            "document one with some words",
            "document two with different words",
            "document three with unique words",
        ]

        features = tfidf_encoder.fit_transform(texts)
        assert features.shape[0] == len(texts)
        assert features.shape[1] > 0
        assert tfidf_encoder.is_fitted

    def test_count_encoding(self, count_encoder):
        """Test count encoding."""
        texts = ["word word word", "different word here", "another different word"]

        features = count_encoder.fit_transform(texts)
        assert features.shape[0] == len(texts)
        assert features.shape[1] > 0
        assert count_encoder.is_fitted

    def test_transform_requires_fitted_encoder(self, tfidf_encoder):
        """Test that transform requires fitted encoder."""
        texts = ["test document"]

        with pytest.raises(ValueError, match="Encoder must be fitted"):
            tfidf_encoder.transform(texts)


class TestSentimentAnalyzer:
    """Test sentiment analysis functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create sentiment analyzer."""
        config = TextDetectionConfig()
        return SentimentAnalyzer(config)

    def test_positive_sentiment(self, analyzer):
        """Test positive sentiment detection."""
        text = "This is a great and wonderful document that I love"
        sentiment = analyzer.analyze_sentiment(text)
        assert sentiment > 0

    def test_negative_sentiment(self, analyzer):
        """Test negative sentiment detection."""
        text = "This is a terrible and awful document that I hate"
        sentiment = analyzer.analyze_sentiment(text)
        assert sentiment < 0

    def test_neutral_sentiment(self, analyzer):
        """Test neutral sentiment detection."""
        text = "This is a document with neutral words and content"
        sentiment = analyzer.analyze_sentiment(text)
        assert abs(sentiment) < 0.1  # Should be close to neutral

    def test_sentiment_anomaly_detection(self, analyzer):
        """Test sentiment anomaly detection."""
        texts = [
            "good document",
            "excellent content",
            "great work",
            "terrible awful horrible disgusting",  # Anomaly
        ]

        anomalies = analyzer.detect_sentiment_anomalies(texts)
        assert len(anomalies) == len(texts)
        assert isinstance(anomalies[0], bool)
        # Last document should be detected as anomaly
        assert anomalies[-1] is True


class TestTextAnomalyDetector:
    """Test main text anomaly detector."""

    @pytest.fixture
    def detector(self):
        """Create text anomaly detector."""
        config = TextDetectionConfig(
            algorithm="isolation_forest", encoding_method="tfidf", contamination=0.2
        )
        return TextAnomalyDetector(config)

    @pytest.fixture
    def sample_texts(self):
        """Create sample text data."""
        normal_texts = [
            "This is a normal document about technology",
            "Another normal document discussing algorithms",
            "A third normal document about machine learning",
            "Normal content about data science topics",
            "Regular document discussing software engineering",
        ]

        anomalous_texts = [
            "Completely different content about cooking recipes and food preparation methods",
            "Random text with unusual pattern xyz abc def qwe rty uio",
        ]

        return normal_texts + anomalous_texts

    def test_fit_detector(self, detector, sample_texts):
        """Test fitting the detector."""
        detector.fit(sample_texts)
        assert detector.is_fitted
        assert detector.encoder.is_fitted

    def test_predict_anomalies(self, detector, sample_texts):
        """Test predicting anomalies."""
        detector.fit(sample_texts)
        predictions = detector.predict(sample_texts)

        assert len(predictions) == len(sample_texts)
        assert all(isinstance(pred, bool) for pred in predictions)
        assert any(predictions)  # Should detect some anomalies

    def test_score_anomalies(self, detector, sample_texts):
        """Test scoring anomalies."""
        detector.fit(sample_texts)
        scores = detector.score(sample_texts)

        assert len(scores) == len(sample_texts)
        assert all(isinstance(score, float) for score in scores)
        assert all(0 <= score <= 1 for score in scores)

    def test_detect_method(self, detector, sample_texts):
        """Test main detect method."""
        detector.fit(sample_texts)
        result = detector.detect(sample_texts)

        assert result.anomaly_type == AnomalyType.LINGUISTIC
        assert len(result.anomaly_scores) == len(sample_texts)
        assert all(0 <= score.value <= 1 for score in result.anomaly_scores)
        assert "algorithm" in result.metadata
        assert "encoding_method" in result.metadata

    def test_detect_single_string(self, detector, sample_texts):
        """Test detection with single string input."""
        detector.fit(sample_texts)

        single_text = "This is a test document"
        result = detector.detect(single_text)

        assert len(result.anomaly_scores) == 1
        assert result.anomaly_type == AnomalyType.LINGUISTIC

    def test_detect_without_fitting_raises_error(self, detector):
        """Test that detection without fitting raises error."""
        with pytest.raises(ValueError, match="Detector must be fitted"):
            detector.predict(["test document"])

    def test_different_algorithms(self, sample_texts):
        """Test different detection algorithms."""
        algorithms = ["isolation_forest", "one_class_svm"]

        for algorithm in algorithms:
            config = TextDetectionConfig(algorithm=algorithm)
            detector = TextAnomalyDetector(config)

            detector.fit(sample_texts)
            predictions = detector.predict(sample_texts)

            assert len(predictions) == len(sample_texts)
            assert isinstance(predictions[0], bool)


class TestTextDetectorFactory:
    """Test text detector factory function."""

    def test_create_text_detector_default(self):
        """Test creating detector with default parameters."""
        detector = create_text_detector()
        assert isinstance(detector, TextAnomalyDetector)
        assert detector.config.algorithm == "isolation_forest"
        assert detector.config.encoding_method == "tfidf"

    def test_create_text_detector_custom(self):
        """Test creating detector with custom parameters."""
        detector = create_text_detector(
            algorithm="one_class_svm", encoding_method="count", contamination=0.15
        )

        assert detector.config.algorithm == "one_class_svm"
        assert detector.config.encoding_method == "count"
        assert detector.config.contamination == 0.15


@pytest.mark.integration
class TestTextDetectorIntegration:
    """Integration tests for text anomaly detection."""

    def test_complete_workflow(self):
        """Test complete text anomaly detection workflow."""
        # Create sample data with clear anomalies
        normal_docs = [
            "Machine learning algorithms are used for pattern recognition",
            "Deep learning models require large amounts of training data",
            "Neural networks can approximate complex mathematical functions",
            "Data preprocessing is crucial for model performance",
            "Feature engineering improves model accuracy significantly",
        ]

        anomalous_docs = [
            "The quick brown fox jumps over the lazy dog repeatedly",  # Different topic
            "xyzkjhwef asdkjfkjsdf qwerty asdfgh zxcvbn",  # Gibberish
        ]

        all_docs = normal_docs + anomalous_docs

        # Create and fit detector
        detector = create_text_detector(contamination=0.3)
        detector.fit(all_docs)

        # Perform detection
        result = detector.detect(all_docs)

        # Verify results
        assert len(result.anomaly_scores) == len(all_docs)
        assert len(result.anomaly_indices) <= len(all_docs)

        # Check that some anomalies were detected
        assert len(result.anomaly_indices) > 0

        # Verify metadata
        metadata = result.metadata
        assert metadata["num_documents"] == len(all_docs)
        assert metadata["contamination_rate"] >= 0
        assert "algorithm" in metadata
        assert "encoding_method" in metadata
