"""
Unit tests for text anomaly detection adapter.

Tests the text processing algorithms, similarity detection,
and advanced NLP features.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from monorepo.infrastructure.adapters.text_adapter import (
    TextAnomalyDetector,
    TextDetectionConfig,
    TextPreprocessor,
    TextEncoder,
    TextSimilarityAnalyzer,
    NamedEntityRecognizer,
    AdvancedTextEncoder,
    SentimentAnalyzer,
    TopicDriftDetector,
)


class TestTextDetectionConfig:
    """Test text detection configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TextDetectionConfig()
        
        assert config.algorithm == "isolation_forest"
        assert config.encoding_method == "tfidf"
        assert config.contamination == 0.1
        assert config.max_features == 5000
        assert config.lowercase is True
        assert config.remove_punctuation is True
        assert config.remove_stopwords is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TextDetectionConfig(
            algorithm="one_class_svm",
            encoding_method="bert",
            contamination=0.05,
            max_features=1000,
            lowercase=False
        )
        
        assert config.algorithm == "one_class_svm"
        assert config.encoding_method == "bert"
        assert config.contamination == 0.05
        assert config.max_features == 1000
        assert config.lowercase is False

    def test_invalid_algorithm(self):
        """Test validation of invalid algorithm."""
        with pytest.raises(ValueError, match="Algorithm must be one of"):
            TextDetectionConfig(algorithm="invalid_algorithm")

    def test_invalid_encoding_method(self):
        """Test validation of invalid encoding method."""
        with pytest.raises(ValueError, match="Encoding method must be one of"):
            TextDetectionConfig(encoding_method="invalid_encoding")


class TestTextPreprocessor:
    """Test text preprocessing functionality."""

    def test_basic_preprocessing(self):
        """Test basic text preprocessing."""
        config = TextDetectionConfig()
        preprocessor = TextPreprocessor(config)
        
        text = "Hello World! This is a TEST with PUNCTUATION."
        processed = preprocessor.preprocess_text(text)
        
        # Should be lowercased, punctuation removed, stop words removed
        expected_words = ["hello", "world", "test", "punctuation"]
        assert all(word in processed for word in expected_words)
        assert "this" not in processed  # stop word
        assert "is" not in processed    # stop word

    def test_preprocessing_with_custom_config(self):
        """Test preprocessing with custom configuration."""
        config = TextDetectionConfig(
            lowercase=False,
            remove_punctuation=False,
            remove_stopwords=False
        )
        preprocessor = TextPreprocessor(config)
        
        text = "Hello World! This is a TEST."
        processed = preprocessor.preprocess_text(text)
        
        # Should preserve case, punctuation, and stop words
        assert "Hello" in processed
        assert "!" in processed
        assert "This" in processed
        assert "is" in processed

    def test_preprocess_corpus(self):
        """Test preprocessing multiple texts."""
        config = TextDetectionConfig()
        preprocessor = TextPreprocessor(config)
        
        texts = [
            "First document with some text.",
            "Second document with different content."
        ]
        processed_texts = preprocessor.preprocess_corpus(texts)
        
        assert len(processed_texts) == 2
        assert "first" in processed_texts[0]
        assert "second" in processed_texts[1]

    def test_empty_text_handling(self):
        """Test handling of empty or None text."""
        config = TextDetectionConfig()
        preprocessor = TextPreprocessor(config)
        
        assert preprocessor.preprocess_text("") == ""
        assert preprocessor.preprocess_text(None) == ""


class TestTextSimilarityAnalyzer:
    """Test text similarity analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create text similarity analyzer."""
        config = TextDetectionConfig()
        analyzer = TextSimilarityAnalyzer(config)
        return analyzer

    @pytest.fixture
    def fitted_analyzer(self, analyzer):
        """Create fitted text similarity analyzer."""
        texts = [
            "machine learning algorithms",
            "artificial intelligence systems", 
            "data science methods"
        ]
        analyzer.fit(texts)
        return analyzer

    def test_cosine_similarity(self, fitted_analyzer):
        """Test cosine similarity calculation."""
        text1 = "machine learning algorithms"
        text2 = "machine learning methods"
        
        similarity = fitted_analyzer.cosine_similarity(text1, text2)
        assert 0 <= similarity <= 1
        assert similarity > 0.5  # Should be quite similar

    def test_jaccard_similarity(self, analyzer):
        """Test Jaccard similarity calculation."""
        text1 = "machine learning algorithms"
        text2 = "machine learning methods"
        
        similarity = analyzer.jaccard_similarity(text1, text2)
        assert 0 <= similarity <= 1
        assert similarity > 0  # Should have some overlap

    def test_edit_distance(self, analyzer):
        """Test edit distance calculation."""
        text1 = "hello"
        text2 = "hallo"
        
        distance = analyzer.edit_distance(text1, text2)
        assert distance == 1  # One character difference

    def test_n_gram_similarity(self, analyzer):
        """Test n-gram similarity calculation."""
        text1 = "machine learning"
        text2 = "machine methods"
        
        similarity = analyzer.n_gram_similarity(text1, text2, n=3)
        assert 0 <= similarity <= 1

    def test_similarity_matrix(self, fitted_analyzer):
        """Test similarity matrix calculation."""
        texts = [
            "machine learning algorithms",
            "artificial intelligence systems", 
            "data science methods"
        ]
        
        matrix = fitted_analyzer.calculate_similarity_matrix(texts, method="cosine")
        
        assert matrix.shape == (3, 3)
        assert np.allclose(np.diag(matrix), 1.0)  # Diagonal should be 1
        assert np.allclose(matrix, matrix.T)      # Should be symmetric

    def test_duplicate_detection(self, fitted_analyzer):
        """Test duplicate anomaly detection."""
        texts = [
            "machine learning algorithms",
            "machine learning algorithms",  # Duplicate
            "completely different content"
        ]
        
        anomalies = fitted_analyzer.detect_duplicate_anomalies(texts, threshold=0.9)
        
        assert len(anomalies) == 3
        assert anomalies[0] or anomalies[1]  # At least one duplicate detected

    def test_analyzer_not_fitted(self, analyzer):
        """Test error when analyzer not fitted."""
        with pytest.raises(ValueError, match="Analyzer must be fitted"):
            analyzer.cosine_similarity("text1", "text2")


class TestNamedEntityRecognizer:
    """Test named entity recognition."""

    @pytest.fixture
    def ner(self):
        """Create NER instance."""
        config = TextDetectionConfig()
        return NamedEntityRecognizer(config)

    @patch('spacy.load')
    def test_entity_extraction_with_spacy(self, mock_spacy_load, ner):
        """Test entity extraction with spaCy."""
        # Mock spaCy model
        mock_doc = Mock()
        mock_ent = Mock()
        mock_ent.label_ = "PERSON"
        mock_ent.text = "John Doe"
        mock_doc.ents = [mock_ent]
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        # Force reload of NLP model
        ner._nlp = None
        
        entities = ner.extract_entities("John Doe is a person.")
        
        assert "PERSON" in entities
        assert "John Doe" in entities["PERSON"]

    def test_entity_extraction_fallback(self, ner):
        """Test entity extraction fallback when spaCy unavailable."""
        # Force spaCy to be unavailable
        ner._nlp = None
        
        entities = ner.extract_entities("John Doe is a person.")
        
        # Should return empty entities structure
        assert isinstance(entities, dict)
        assert "PERSON" in entities
        assert len(entities["PERSON"]) == 0

    def test_entity_anomaly_detection(self, ner):
        """Test entity-based anomaly detection."""
        # Mock entity extraction
        ner.extract_entities = Mock(side_effect=[
            {"PERSON": ["John"], "ORG": [], "GPE": [], "MONEY": [], "DATE": []},
            {"PERSON": ["Jane"], "ORG": [], "GPE": [], "MONEY": [], "DATE": []},
            {"PERSON": ["A", "B", "C", "D", "E"], "ORG": [], "GPE": [], "MONEY": [], "DATE": []}  # Many entities
        ])
        
        texts = ["Text with John", "Text with Jane", "Text with many people"]
        anomalies = ner.detect_entity_anomalies(texts)
        
        assert len(anomalies) == 3
        assert anomalies[2]  # Third text should be anomalous (many entities)


class TestAdvancedTextEncoder:
    """Test advanced text encoding methods."""

    @pytest.fixture
    def encoder(self):
        """Create advanced text encoder."""
        config = TextDetectionConfig()
        return AdvancedTextEncoder(config)

    @patch('sentence_transformers.SentenceTransformer')
    def test_transformer_encoding(self, mock_transformer, encoder):
        """Test transformer-based encoding."""
        # Mock transformer model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_transformer.return_value = mock_model
        
        texts = ["First text", "Second text"]
        embeddings = encoder.advanced_encoding(texts, method="transformer")
        
        assert embeddings.shape == (2, 3)
        mock_transformer.assert_called_once()
        mock_model.encode.assert_called_once_with(texts)

    def test_invalid_encoding_method(self, encoder):
        """Test error for invalid encoding method."""
        with pytest.raises(ValueError, match="Unknown encoding method"):
            encoder.advanced_encoding(["text"], method="invalid_method")

    @patch('fasttext.train_unsupervised')
    def test_fasttext_encoding(self, mock_fasttext, encoder):
        """Test FastText encoding."""
        # Mock FastText model
        mock_model = Mock()
        mock_model.get_word_vector.return_value = np.array([0.1, 0.2, 0.3])
        mock_fasttext.return_value = mock_model
        
        texts = ["hello world", "test text"]
        
        with patch('tempfile.NamedTemporaryFile'), patch('os.unlink'):
            embeddings = encoder._fasttext_encoding(texts)
            
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0


class TestSentimentAnalyzer:
    """Test sentiment analysis functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create sentiment analyzer."""
        config = TextDetectionConfig()
        return SentimentAnalyzer(config)

    def test_positive_sentiment(self, analyzer):
        """Test positive sentiment detection."""
        text = "This is great and excellent work!"
        score = analyzer.analyze_sentiment(text)
        assert score > 0

    def test_negative_sentiment(self, analyzer):
        """Test negative sentiment detection."""
        text = "This is terrible and awful work!"
        score = analyzer.analyze_sentiment(text)
        assert score < 0

    def test_neutral_sentiment(self, analyzer):
        """Test neutral sentiment detection."""
        text = "This is some text without sentiment words."
        score = analyzer.analyze_sentiment(text)
        assert score == 0.0

    def test_sentiment_anomaly_detection(self, analyzer):
        """Test sentiment-based anomaly detection."""
        texts = [
            "good work",
            "nice job", 
            "terrible awful horrible disgusting"  # Very negative
        ]
        
        anomalies = analyzer.detect_sentiment_anomalies(texts)
        assert len(anomalies) == 3
        assert anomalies[2]  # Very negative text should be anomaly


class TestTopicDriftDetector:
    """Test topic drift detection."""

    @pytest.fixture
    def detector(self):
        """Create topic drift detector."""
        config = TextDetectionConfig(n_topics=2)
        return TopicDriftDetector(config)

    @patch('gensim.models.LdaModel')
    @patch('gensim.corpora.Dictionary')
    def test_topic_drift_detection(self, mock_dict, mock_lda, detector):
        """Test topic drift detection functionality."""
        # Mock LDA components
        mock_dictionary = Mock()
        mock_dictionary.doc2bow.return_value = [(0, 1), (1, 1)]
        mock_dict.return_value = mock_dictionary
        
        mock_model = Mock()
        mock_model.get_document_topics.return_value = [(0, 0.7), (1, 0.3)]
        mock_lda.return_value = mock_model
        
        # Set up detector
        detector.dictionary = mock_dictionary
        detector.lda_model = mock_model
        detector.baseline_topics = [[(0, 0.8), (1, 0.2)], [(0, 0.6), (1, 0.4)]]
        detector.is_fitted = True
        
        texts = ["new document with different topic distribution"]
        anomalies = detector.detect_topic_drift(texts)
        
        assert len(anomalies) == 1
        assert isinstance(anomalies[0], bool)


class TestTextAnomalyDetector:
    """Test main text anomaly detector."""

    @pytest.fixture
    def detector(self):
        """Create text anomaly detector."""
        config = TextDetectionConfig()
        return TextAnomalyDetector(config)

    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector.config is not None
        assert detector.preprocessor is not None
        assert detector.encoder is not None
        assert detector.sentiment_analyzer is not None
        assert detector.similarity_analyzer is not None
        assert detector.ner is not None
        assert detector.advanced_encoder is not None
        assert not detector.is_fitted

    def test_detector_properties(self, detector):
        """Test detector properties."""
        assert "TextAnomalyDetector" in detector.name
        assert detector.contamination_rate.value == 0.1
        assert not detector.is_fitted
        
        parameters = detector.parameters
        assert "algorithm" in parameters
        assert "encoding_method" in parameters
        assert "contamination" in parameters

    def test_text_extraction(self, detector):
        """Test text extraction from various input formats."""
        # String input
        texts = detector._extract_texts("single text")
        assert texts == ["single text"]
        
        # List input
        texts = detector._extract_texts(["text1", "text2"])
        assert texts == ["text1", "text2"]
        
        # DataFrame input
        import pandas as pd
        df = pd.DataFrame({"text": ["doc1", "doc2"]})
        texts = detector._extract_texts(df)
        assert texts == ["doc1", "doc2"]

    @patch('sklearn.ensemble.IsolationForest')
    def test_detector_fitting(self, mock_isolation_forest, detector):
        """Test detector fitting process."""
        # Mock sklearn components
        mock_vectorizer = Mock()
        mock_vectorizer.fit_transform.return_value = np.array([[1, 0], [0, 1]])
        mock_vectorizer.transform.return_value = np.array([[1, 0], [0, 1]])
        
        mock_detector_algo = Mock()
        mock_isolation_forest.return_value = mock_detector_algo
        
        # Mock encoder
        detector.encoder.fit_transform = Mock(return_value=np.array([[1, 0], [0, 1]]))
        detector.similarity_analyzer.fit = Mock()
        
        texts = ["first document", "second document"]
        result = detector.fit(texts)
        
        assert result is detector
        assert detector.is_fitted
        mock_detector_algo.fit.assert_called_once()

    def test_prediction_before_fitting(self, detector):
        """Test error when predicting before fitting."""
        with pytest.raises(ValueError, match="Detector must be fitted"):
            detector.predict(["test text"])

    def test_scoring_before_fitting(self, detector):
        """Test error when scoring before fitting."""
        with pytest.raises(ValueError, match="Detector must be fitted"):
            detector.score(["test text"])

    @patch('sklearn.ensemble.IsolationForest')
    def test_detection_workflow(self, mock_isolation_forest, detector):
        """Test complete detection workflow."""
        # Setup mocks
        mock_detector_algo = Mock()
        mock_detector_algo.predict.return_value = np.array([1, -1])  # One normal, one anomaly
        mock_detector_algo.decision_function.return_value = np.array([0.1, -0.5])
        mock_isolation_forest.return_value = mock_detector_algo
        
        # Mock all required components
        detector.encoder.fit_transform = Mock(return_value=np.array([[1, 0], [0, 1]]))
        detector.encoder.transform = Mock(return_value=np.array([[1, 0], [0, 1]]))
        detector.similarity_analyzer.fit = Mock()
        detector.similarity_analyzer.detect_duplicate_anomalies = Mock(return_value=[False, False])
        detector.ner.detect_entity_anomalies = Mock(return_value=[False, False])
        detector.sentiment_analyzer.detect_sentiment_anomalies = Mock(return_value=[False, False])
        
        # Fit and predict
        texts = ["normal text", "anomalous text"]
        detector.fit(texts)
        
        # Test prediction
        predictions = detector.predict(texts)
        assert len(predictions) == 2
        assert not predictions[0]  # Normal
        assert predictions[1]      # Anomaly
        
        # Test scoring
        scores = detector.score(texts)
        assert len(scores) == 2
        assert all(0 <= score <= 1 for score in scores)
        
        # Test detection
        result = detector.detect(texts)
        assert result.anomaly_indices == [1]
        assert len(result.anomaly_scores) == 2


class TestTextEncoders:
    """Test various text encoding methods."""

    def test_tfidf_encoder(self):
        """Test TF-IDF encoder."""
        config = TextDetectionConfig(encoding_method="tfidf")
        encoder = TextEncoder(config)
        
        texts = ["machine learning algorithms", "data science methods"]
        with patch('sklearn.feature_extraction.text.TfidfVectorizer') as mock_tfidf:
            mock_vectorizer = Mock()
            mock_vectorizer.fit_transform.return_value = Mock()
            mock_vectorizer.fit_transform.return_value.toarray.return_value = np.array([[1, 0], [0, 1]])
            mock_tfidf.return_value = mock_vectorizer
            
            features = encoder.fit_transform(texts)
            assert features.shape == (2, 2)

    def test_count_encoder(self):
        """Test Count encoder.""" 
        config = TextDetectionConfig(encoding_method="count")
        encoder = TextEncoder(config)
        
        texts = ["machine learning", "data science"]
        with patch('sklearn.feature_extraction.text.CountVectorizer') as mock_count:
            mock_vectorizer = Mock()
            mock_vectorizer.fit_transform.return_value = Mock()
            mock_vectorizer.fit_transform.return_value.toarray.return_value = np.array([[1, 0], [0, 1]])
            mock_count.return_value = mock_vectorizer
            
            features = encoder.fit_transform(texts)
            assert features.shape == (2, 2)

    def test_unsupported_encoder(self):
        """Test error for unsupported encoding method."""
        config = TextDetectionConfig(encoding_method="unsupported")
        encoder = TextEncoder(config)
        
        with pytest.raises(NotImplementedError):
            encoder.fit_transform(["text"])


if __name__ == "__main__":
    pytest.main([__file__])