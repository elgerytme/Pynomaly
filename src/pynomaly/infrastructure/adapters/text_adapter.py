"""
Text Anomaly Detection Adapter.

This module provides comprehensive text anomaly detection capabilities,
including document-level anomalies, topic drift, sentiment anomalies,
and linguistic pattern detection.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, field_validator

from pynomaly.domain.entities import DetectionResult
from pynomaly.domain.models.detection import DetectionConfig
from pynomaly.domain.value_objects import AnomalyScore, AnomalyType
from pynomaly.shared.protocols.detector_protocol import DetectorProtocol

logger = logging.getLogger(__name__)


class TextDetectionConfig(DetectionConfig):
    """Configuration for text anomaly detection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Text preprocessing options
    lowercase: bool = Field(default=True, description="Convert text to lowercase")
    remove_punctuation: bool = Field(default=True, description="Remove punctuation")
    remove_stopwords: bool = Field(default=True, description="Remove stop words")
    min_word_length: int = Field(default=2, description="Minimum word length")
    max_features: int = Field(default=5000, description="Maximum vocabulary size")

    # Detection algorithms
    algorithm: str = Field(default="isolation_forest", description="Text anomaly detection algorithm")
    encoding_method: str = Field(default="tfidf", description="Text encoding method")

    # Algorithm-specific parameters
    contamination: float = Field(default=0.1, description="Expected contamination rate")
    n_components: int = Field(default=100, description="Number of components for dimensionality reduction")
    random_state: int = Field(default=42, description="Random state for reproducibility")

    # Sentiment analysis
    enable_sentiment: bool = Field(default=True, description="Enable sentiment-based anomaly detection")
    sentiment_threshold: float = Field(default=2.0, description="Sentiment anomaly threshold")

    # Topic modeling
    enable_topic_detection: bool = Field(default=True, description="Enable topic drift detection")
    n_topics: int = Field(default=10, description="Number of topics for LDA")
    topic_threshold: float = Field(default=0.3, description="Topic coherence threshold")

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        valid_algorithms = {
            "isolation_forest", "one_class_svm", "local_outlier_factor",
            "elliptic_envelope", "autoencoder", "doc2vec_anomaly"
        }
        if v not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}")
        return v

    @field_validator("encoding_method")
    @classmethod
    def validate_encoding_method(cls, v: str) -> str:
        valid_methods = {"tfidf", "count", "doc2vec", "bert", "word2vec"}
        if v not in valid_methods:
            raise ValueError(f"Encoding method must be one of {valid_methods}")
        return v


class TextPreprocessor:
    """Text preprocessing utilities for anomaly detection."""

    def __init__(self, config: TextDetectionConfig):
        self.config = config
        self._stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set:
        """Load stop words (basic English set)."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'other', 'about', 'out', 'many', 'then', 'them'
        }

    def preprocess_text(self, text: str) -> str:
        """Preprocess a single text document."""
        if not isinstance(text, str):
            return ""

        # Lowercase
        if self.config.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.config.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenize and filter
        words = text.split()

        # Remove stop words
        if self.config.remove_stopwords:
            words = [w for w in words if w not in self._stopwords]

        # Filter by length
        words = [w for w in words if len(w) >= self.config.min_word_length]

        return ' '.join(words)

    def preprocess_corpus(self, texts: list[str]) -> list[str]:
        """Preprocess a corpus of texts."""
        return [self.preprocess_text(text) for text in texts]


class TextEncoder:
    """Text encoding utilities for feature extraction."""

    def __init__(self, config: TextDetectionConfig):
        self.config = config
        self.encoder = None
        self.is_fitted = False

    def _create_encoder(self):
        """Create the appropriate text encoder."""
        if self.config.encoding_method == "tfidf":
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                return TfidfVectorizer(
                    max_features=self.config.max_features,
                    stop_words='english' if self.config.remove_stopwords else None
                )
            except ImportError:
                raise ImportError("scikit-learn is required for TF-IDF encoding")

        elif self.config.encoding_method == "count":
            try:
                from sklearn.feature_extraction.text import CountVectorizer
                return CountVectorizer(
                    max_features=self.config.max_features,
                    stop_words='english' if self.config.remove_stopwords else None
                )
            except ImportError:
                raise ImportError("scikit-learn is required for count encoding")

        elif self.config.encoding_method == "doc2vec":
            return Doc2VecEncoder()

        elif self.config.encoding_method == "word2vec":
            return Word2VecEncoder()

        elif self.config.encoding_method == "bert":
            return BERTEncoder()

        else:
            raise NotImplementedError(f"Encoding method {self.config.encoding_method} not implemented")

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit encoder and transform texts to feature vectors."""
        self.encoder = self._create_encoder()
        features = self.encoder.fit_transform(texts)
        self.is_fitted = True
        return features.toarray() if hasattr(features, 'toarray') else features

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts using fitted encoder."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        features = self.encoder.transform(texts)
        return features.toarray() if hasattr(features, 'toarray') else features


class Doc2VecEncoder:
    """Doc2Vec encoder for document embeddings."""

    def __init__(self):
        self.model = None
        self.is_fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit Doc2Vec model and transform texts."""
        try:
            from gensim.models import Doc2Vec
            from gensim.models.doc2vec import TaggedDocument
        except ImportError:
            raise ImportError("gensim is required for Doc2Vec encoding. Install with: pip install gensim")

        # Prepare tagged documents
        documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(texts)]

        # Train Doc2Vec model
        self.model = Doc2Vec(
            documents,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
            epochs=20
        )

        # Get document vectors
        vectors = np.array([self.model.dv[i] for i in range(len(texts))])
        self.is_fitted = True
        return vectors

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts using fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")

        # Infer vectors for new documents
        vectors = []
        for text in texts:
            vector = self.model.infer_vector(text.split())
            vectors.append(vector)

        return np.array(vectors)


class Word2VecEncoder:
    """Word2Vec encoder with document averaging."""

    def __init__(self):
        self.model = None
        self.is_fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit Word2Vec model and transform texts."""
        try:
            from gensim.models import Word2Vec
        except ImportError:
            raise ImportError("gensim is required for Word2Vec encoding. Install with: pip install gensim")

        # Prepare sentences
        sentences = [text.split() for text in texts]

        # Train Word2Vec model
        self.model = Word2Vec(
            sentences,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4
        )

        # Get document vectors by averaging word vectors
        vectors = []
        for sentence in sentences:
            word_vectors = []
            for word in sentence:
                if word in self.model.wv:
                    word_vectors.append(self.model.wv[word])

            if word_vectors:
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                doc_vector = np.zeros(self.model.vector_size)

            vectors.append(doc_vector)

        self.is_fitted = True
        return np.array(vectors)

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts using fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")

        vectors = []
        for text in texts:
            words = text.split()
            word_vectors = []

            for word in words:
                if word in self.model.wv:
                    word_vectors.append(self.model.wv[word])

            if word_vectors:
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                doc_vector = np.zeros(self.model.vector_size)

            vectors.append(doc_vector)

        return np.array(vectors)


class BERTEncoder:
    """BERT encoder for text embeddings."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Initialize BERT and transform texts."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("transformers and torch are required for BERT encoding. Install with: pip install transformers torch")

        # Load pre-trained BERT model
        model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Encode texts
        vectors = self._encode_texts(texts)
        self.is_fitted = True
        return vectors

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts using BERT."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")

        return self._encode_texts(texts)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts using BERT."""
        import torch

        vectors = []

        # Process in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embeddings
                batch_vectors = outputs.last_hidden_state[:, 0, :].numpy()
                vectors.extend(batch_vectors)

        return np.array(vectors)


class TopicDriftDetector:
    """Topic drift detection for text anomalies."""

    def __init__(self, config: TextDetectionConfig):
        self.config = config
        self.lda_model = None
        self.dictionary = None
        self.baseline_topics = None
        self.is_fitted = False

    def fit(self, texts: list[str]) -> None:
        """Fit topic model on baseline texts."""
        try:
            from gensim import corpora, models
            from gensim.models import LdaModel
        except ImportError:
            raise ImportError("gensim is required for topic modeling. Install with: pip install gensim")

        # Preprocess texts
        processed_texts = [text.split() for text in texts]

        # Create dictionary and corpus
        self.dictionary = corpora.Dictionary(processed_texts)
        corpus = [self.dictionary.doc2bow(text) for text in processed_texts]

        # Train LDA model
        self.lda_model = LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.config.n_topics,
            random_state=self.config.random_state,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )

        # Store baseline topic distributions
        self.baseline_topics = [
            self.lda_model.get_document_topics(doc, minimum_probability=0)
            for doc in corpus
        ]

        self.is_fitted = True

    def detect_topic_drift(self, texts: list[str]) -> list[bool]:
        """Detect topic drift in new texts."""
        if not self.is_fitted:
            raise ValueError("Topic model must be fitted before detection")

        # Process new texts
        processed_texts = [text.split() for text in texts]
        new_corpus = [self.dictionary.doc2bow(text) for text in processed_texts]

        # Get topic distributions for new texts
        new_topics = [
            self.lda_model.get_document_topics(doc, minimum_probability=0)
            for doc in new_corpus
        ]

        # Calculate topic drift
        anomalies = []
        for topic_dist in new_topics:
            # Convert to probability vector
            topic_probs = np.zeros(self.config.n_topics)
            for topic_id, prob in topic_dist:
                topic_probs[topic_id] = prob

            # Compare with baseline
            is_anomaly = self._is_topic_anomaly(topic_probs)
            anomalies.append(is_anomaly)

        return anomalies

    def _is_topic_anomaly(self, topic_probs: np.ndarray) -> bool:
        """Check if topic distribution is anomalous."""
        # Calculate average baseline topic distribution
        baseline_avg = np.zeros(self.config.n_topics)

        for topic_dist in self.baseline_topics:
            topic_vector = np.zeros(self.config.n_topics)
            for topic_id, prob in topic_dist:
                topic_vector[topic_id] = prob
            baseline_avg += topic_vector

        baseline_avg /= len(self.baseline_topics)

        # Calculate KL divergence
        kl_div = self._kl_divergence(topic_probs + 1e-10, baseline_avg + 1e-10)

        return kl_div > self.config.topic_threshold

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence between two probability distributions."""
        return np.sum(p * np.log(p / q))


class SentimentAnalyzer:
    """Simple sentiment analysis for anomaly detection."""

    def __init__(self, config: TextDetectionConfig):
        self.config = config
        self._positive_words = self._load_positive_words()
        self._negative_words = self._load_negative_words()

    def _load_positive_words(self) -> set:
        """Load positive sentiment words."""
        return {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'perfect', 'love', 'like', 'happy', 'pleased', 'satisfied',
            'positive', 'brilliant', 'outstanding', 'superb', 'magnificent'
        }

    def _load_negative_words(self) -> set:
        """Load negative sentiment words."""
        return {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
            'dislike', 'angry', 'frustrated', 'disappointed', 'negative',
            'poor', 'worst', 'useless', 'pathetic', 'ridiculous', 'stupid'
        }

    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text. Returns score between -1 and 1."""
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in self._positive_words)
        negative_count = sum(1 for word in words if word in self._negative_words)

        total_words = len(words)
        if total_words == 0:
            return 0.0

        sentiment_score = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment_score))

    def detect_sentiment_anomalies(self, texts: list[str]) -> list[bool]:
        """Detect sentiment anomalies in text corpus."""
        sentiments = [self.analyze_sentiment(text) for text in texts]
        sentiment_mean = np.mean(sentiments)
        sentiment_std = np.std(sentiments)

        threshold = self.config.sentiment_threshold * sentiment_std
        anomalies = [abs(s - sentiment_mean) > threshold for s in sentiments]

        return anomalies


class TextAutoencoder:
    """Neural autoencoder for text anomaly detection."""

    def __init__(self, config: TextDetectionConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.threshold = None

    def fit(self, features: np.ndarray) -> None:
        """Fit autoencoder on text features."""
        try:
            import numpy as np
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("scikit-learn is required for autoencoder")

        # Normalize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        # Simple autoencoder using PCA for dimensionality reduction
        try:
            from sklearn.decomposition import PCA

            # Use PCA as a simple autoencoder
            n_components = min(self.config.n_components, features.shape[1] // 2)
            self.model = PCA(n_components=n_components)
            self.model.fit(features_scaled)

            # Calculate reconstruction error threshold
            reconstructed = self.model.inverse_transform(self.model.transform(features_scaled))
            reconstruction_errors = np.mean((features_scaled - reconstructed) ** 2, axis=1)
            self.threshold = np.percentile(reconstruction_errors, (1 - self.config.contamination) * 100)

        except ImportError:
            raise ImportError("scikit-learn is required for PCA autoencoder")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict anomalies using reconstruction error."""
        features_scaled = self.scaler.transform(features)
        reconstructed = self.model.inverse_transform(self.model.transform(features_scaled))
        reconstruction_errors = np.mean((features_scaled - reconstructed) ** 2, axis=1)

        # Return -1 for anomalies, 1 for normal
        return np.where(reconstruction_errors > self.threshold, -1, 1)


class Doc2VecAnomalyDetector:
    """Doc2Vec-based anomaly detection."""

    def __init__(self, config: TextDetectionConfig):
        self.config = config
        self.doc2vec_model = None
        self.anomaly_detector = None

    def fit(self, features: np.ndarray) -> None:
        """Fit Doc2Vec anomaly detector."""
        try:
            from sklearn.ensemble import IsolationForest

            # Use Isolation Forest on Doc2Vec features
            self.anomaly_detector = IsolationForest(
                contamination=self.config.contamination,
                random_state=self.config.random_state
            )
            self.anomaly_detector.fit(features)

        except ImportError:
            raise ImportError("scikit-learn is required for Doc2Vec anomaly detection")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict anomalies using Doc2Vec features."""
        return self.anomaly_detector.predict(features)


class TextAnomalyDetector(DetectorProtocol):
    """Main text anomaly detection adapter."""

    def __init__(self, config: TextDetectionConfig | None = None):
        self.config = config or TextDetectionConfig()
        self.preprocessor = TextPreprocessor(self.config)
        self.encoder = TextEncoder(self.config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.topic_detector = TopicDriftDetector(self.config) if self.config.enable_topic_detection else None
        self.similarity_analyzer = TextSimilarityAnalyzer(self.config)
        self.ner = NamedEntityRecognizer(self.config)
        self.advanced_encoder = AdvancedTextEncoder(self.config)
        self.detector = None
        self._is_fitted = False

    @property
    def name(self) -> str:
        """Get the name of the detector."""
        return f"TextAnomalyDetector_{self.config.algorithm}"

    @property
    def contamination_rate(self):
        """Get the contamination rate."""
        from pynomaly.domain.value_objects import ContaminationRate
        return ContaminationRate(value=self.config.contamination)

    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        return self._is_fitted

    @property
    def parameters(self) -> dict[str, Any]:
        """Get the current parameters of the detector."""
        return {
            "algorithm": self.config.algorithm,
            "encoding_method": self.config.encoding_method,
            "contamination": self.config.contamination,
            "max_features": self.config.max_features,
            "enable_sentiment": self.config.enable_sentiment,
            "enable_topic_detection": self.config.enable_topic_detection
        }

    def _extract_texts(self, data: Any) -> list[str]:
        """Extract texts from various input formats."""
        if isinstance(data, str):
            return [data]
        elif isinstance(data, list):
            return data
        elif isinstance(data, pd.DataFrame):
            # Assume first column contains text
            return data.iloc[:, 0].tolist()
        elif hasattr(data, 'data') and hasattr(data.data, 'iloc'):
            # Dataset object with DataFrame
            return data.data.iloc[:, 0].tolist()
        elif hasattr(data, 'data') and isinstance(data.data, list):
            # Dataset object with list
            return data.data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _create_detector(self):
        """Create the anomaly detection algorithm."""
        if self.config.algorithm == "isolation_forest":
            try:
                from sklearn.ensemble import IsolationForest
                return IsolationForest(
                    contamination=self.config.contamination,
                    random_state=self.config.random_state
                )
            except ImportError:
                raise ImportError("scikit-learn is required for Isolation Forest")

        elif self.config.algorithm == "one_class_svm":
            try:
                from sklearn.svm import OneClassSVM
                return OneClassSVM(nu=self.config.contamination)
            except ImportError:
                raise ImportError("scikit-learn is required for One-Class SVM")

        elif self.config.algorithm == "local_outlier_factor":
            try:
                from sklearn.neighbors import LocalOutlierFactor
                return LocalOutlierFactor(
                    contamination=self.config.contamination,
                    novelty=True
                )
            except ImportError:
                raise ImportError("scikit-learn is required for Local Outlier Factor")

        elif self.config.algorithm == "elliptic_envelope":
            try:
                from sklearn.covariance import EllipticEnvelope
                return EllipticEnvelope(contamination=self.config.contamination)
            except ImportError:
                raise ImportError("scikit-learn is required for Elliptic Envelope")

        elif self.config.algorithm == "autoencoder":
            return TextAutoencoder(self.config)

        elif self.config.algorithm == "doc2vec_anomaly":
            return Doc2VecAnomalyDetector(self.config)

        else:
            raise NotImplementedError(f"Algorithm {self.config.algorithm} not implemented")


class TextSimilarityAnalyzer:
    """Advanced text similarity algorithms for anomaly detection."""

    def __init__(self, config: TextDetectionConfig):
        self.config = config
        self._vectorizer = None
        self._is_fitted = False

    def fit(self, texts: list[str]) -> None:
        """Fit similarity analyzer on text corpus."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                stop_words='english' if self.config.remove_stopwords else None
            )
            self._vectorizer.fit(texts)
            self.is_fitted = True
        except ImportError:
            raise ImportError("scikit-learn is required for text similarity analysis")

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        if not self.is_fitted:
            raise ValueError("Analyzer must be fitted before calculating similarity")
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Vectorize texts
            vectors = self._vectorizer.transform([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except ImportError:
            raise ImportError("scikit-learn is required for cosine similarity")

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        # Tokenize texts
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def edit_distance(self, text1: str, text2: str) -> int:
        """Calculate edit distance (Levenshtein distance) between two texts."""
        try:
            import textdistance
            return textdistance.levenshtein(text1, text2)
        except ImportError:
            # Fallback implementation
            return self._levenshtein_distance(text1, text2)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Fallback implementation of Levenshtein distance."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using word embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Load pre-trained sentence transformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get embeddings
            embeddings = model.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except ImportError:
            # Fallback to TF-IDF cosine similarity
            return self.cosine_similarity(text1, text2)

    def fuzzy_similarity(self, text1: str, text2: str) -> float:
        """Calculate fuzzy string similarity."""
        try:
            from fuzzywuzzy import fuzz
            return fuzz.ratio(text1, text2) / 100.0
        except ImportError:
            # Fallback to Jaccard similarity
            return self.jaccard_similarity(text1, text2)

    def n_gram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """Calculate n-gram similarity between texts."""
        def get_ngrams(text: str, n: int) -> set:
            """Extract n-grams from text."""
            text = text.lower()
            return set([text[i:i+n] for i in range(len(text) - n + 1)])
        
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0

    def calculate_similarity_matrix(self, texts: list[str], method: str = "cosine") -> np.ndarray:
        """Calculate similarity matrix for a list of texts."""
        n_texts = len(texts)
        similarity_matrix = np.zeros((n_texts, n_texts))
        
        for i in range(n_texts):
            for j in range(i, n_texts):
                if i == j:
                    similarity = 1.0
                else:
                    if method == "cosine":
                        similarity = self.cosine_similarity(texts[i], texts[j])
                    elif method == "jaccard":
                        similarity = self.jaccard_similarity(texts[i], texts[j])
                    elif method == "semantic":
                        similarity = self.semantic_similarity(texts[i], texts[j])
                    elif method == "fuzzy":
                        similarity = self.fuzzy_similarity(texts[i], texts[j])
                    elif method == "ngram":
                        similarity = self.n_gram_similarity(texts[i], texts[j])
                    else:
                        raise ValueError(f"Unknown similarity method: {method}")
                
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        return similarity_matrix

    def detect_duplicate_anomalies(self, texts: list[str], threshold: float = 0.9) -> list[bool]:
        """Detect anomalously similar (duplicate) texts."""
        similarity_matrix = self.calculate_similarity_matrix(texts, method="cosine")
        
        # Find texts with high similarity to others (excluding self-similarity)
        anomalies = []
        for i in range(len(texts)):
            max_similarity = np.max([similarity_matrix[i][j] for j in range(len(texts)) if i != j])
            anomalies.append(max_similarity > threshold)
        
        return anomalies


class NamedEntityRecognizer:
    """Named Entity Recognition for text anomaly detection."""

    def __init__(self, config: TextDetectionConfig):
        self.config = config
        self._nlp = None
        self._is_fitted = False

    def _load_nlp_model(self):
        """Load spaCy NLP model."""
        try:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Fallback to basic English model
                logger.warning("en_core_web_sm model not found, using simple tokenizer")
                self._nlp = None
        except ImportError:
            logger.warning("spaCy not available, NER features disabled")
            self._nlp = None

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract named entities from text."""
        if self._nlp is None:
            self._load_nlp_model()
        
        if self._nlp is None:
            # Fallback: return empty entities
            return {"PERSON": [], "ORG": [], "GPE": [], "MONEY": [], "DATE": []}
        
        doc = self._nlp(text)
        entities = {"PERSON": [], "ORG": [], "GPE": [], "MONEY": [], "DATE": []}
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        return entities

    def detect_entity_anomalies(self, texts: list[str]) -> list[bool]:
        """Detect anomalies based on named entity patterns."""
        all_entities = [self.extract_entities(text) for text in texts]
        
        # Calculate entity statistics
        entity_counts = {}
        for entities in all_entities:
            for entity_type, entity_list in entities.items():
                if entity_type not in entity_counts:
                    entity_counts[entity_type] = []
                entity_counts[entity_type].append(len(entity_list))
        
        # Detect anomalies based on entity count deviations
        anomalies = []
        for i, entities in enumerate(all_entities):
            is_anomaly = False
            
            for entity_type, entity_list in entities.items():
                if entity_type in entity_counts and len(entity_counts[entity_type]) > 1:
                    count = len(entity_list)
                    mean_count = np.mean(entity_counts[entity_type])
                    std_count = np.std(entity_counts[entity_type])
                    
                    # Anomaly if count is more than 2 standard deviations from mean
                    if std_count > 0 and abs(count - mean_count) > 2 * std_count:
                        is_anomaly = True
                        break
            
            anomalies.append(is_anomaly)
        
        return anomalies


class AdvancedTextEncoder:
    """Advanced text encoding methods for anomaly detection."""

    def __init__(self, config: TextDetectionConfig):
        self.config = config
        self._encoder = None
        self._is_fitted = False

    def advanced_encoding(self, texts: list[str], method: str = "transformer") -> np.ndarray:
        """Implement advanced text encoding methods."""
        if method == "transformer":
            return self._transformer_encoding(texts)
        elif method == "universal_sentence_encoder":
            return self._use_encoding(texts)
        elif method == "fasttext":
            return self._fasttext_encoding(texts)
        elif method == "glove":
            return self._glove_encoding(texts)
        else:
            raise ValueError(f"Unknown encoding method: {method}")

    def _transformer_encoding(self, texts: list[str]) -> np.ndarray:
        """Encode texts using transformer models."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(texts)
            return embeddings
        except ImportError:
            raise ImportError("sentence-transformers is required for transformer encoding. Install with: pip install sentence-transformers")

    def _use_encoding(self, texts: list[str]) -> np.ndarray:
        """Encode texts using Universal Sentence Encoder."""
        try:
            import tensorflow_hub as hub
            import tensorflow as tf
            
            # Load Universal Sentence Encoder
            embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            embeddings = embed(texts)
            return embeddings.numpy()
        except ImportError:
            raise ImportError("tensorflow and tensorflow-hub are required for USE encoding")

    def _fasttext_encoding(self, texts: list[str]) -> np.ndarray:
        """Encode texts using FastText."""
        try:
            import fasttext
            import tempfile
            import os
            
            # Create temporary file for training
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                for text in texts:
                    f.write(text.replace('\n', ' ') + '\n')
                temp_file = f.name
            
            try:
                # Train FastText model
                model = fasttext.train_unsupervised(temp_file, model='cbow', dim=100)
                
                # Get sentence embeddings by averaging word vectors
                embeddings = []
                for text in texts:
                    words = text.split()
                    if words:
                        word_vectors = [model.get_word_vector(word) for word in words]
                        sentence_vector = np.mean(word_vectors, axis=0)
                    else:
                        sentence_vector = np.zeros(100)
                    embeddings.append(sentence_vector)
                
                return np.array(embeddings)
            finally:
                os.unlink(temp_file)
        except ImportError:
            raise ImportError("fasttext is required for FastText encoding. Install with: pip install fasttext")

    def _glove_encoding(self, texts: list[str]) -> np.ndarray:
        """Encode texts using GloVe embeddings."""
        # This would require pre-trained GloVe vectors
        # For now, fallback to TF-IDF
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=self.config.max_features)
            embeddings = vectorizer.fit_transform(texts)
            return embeddings.toarray()
        except ImportError:
            raise ImportError("scikit-learn is required for GloVe fallback encoding")

        logger.info(f"Fitting text anomaly detector with {len(texts)} documents")

        # Preprocess texts
        processed_texts = self.preprocessor.preprocess_corpus(texts)

        # Encode texts to features
        features = self.encoder.fit_transform(processed_texts)

        # Create and fit detector
        self.detector = self._create_detector()
        self.detector.fit(features)

        # Fit topic detector if enabled
        if self.topic_detector:
            self.topic_detector.fit(processed_texts)

        # Fit similarity analyzer
        self.similarity_analyzer.fit(processed_texts)

        self.is_fitted = True
        logger.info("Text anomaly detector fitted successfully")

        return self

    def predict(self, texts: list[str]) -> list[bool]:
        """Predict anomalies in texts."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        # Preprocess texts
        processed_texts = self.preprocessor.preprocess_corpus(texts)

        # Encode texts
        features = self.encoder.transform(processed_texts)

        # Predict anomalies
        predictions = self.detector.predict(features)

        # Convert sklearn format (-1 for anomaly, 1 for normal) to boolean
        anomalies = [pred == -1 for pred in predictions]

        # Combine with sentiment anomalies if enabled
        if self.config.enable_sentiment:
            sentiment_anomalies = self.sentiment_analyzer.detect_sentiment_anomalies(texts)
            anomalies = [a or s for a, s in zip(anomalies, sentiment_anomalies, strict=False)]

        # Combine with topic drift anomalies if enabled
        if self.topic_detector:
            processed_texts = self.preprocessor.preprocess_corpus(texts)
            topic_anomalies = self.topic_detector.detect_topic_drift(processed_texts)
            anomalies = [a or t for a, t in zip(anomalies, topic_anomalies, strict=False)]

        # Combine with similarity-based anomalies (duplicate detection)
        similarity_anomalies = self.similarity_analyzer.detect_duplicate_anomalies(texts)
        anomalies = [a or s for a, s in zip(anomalies, similarity_anomalies, strict=False)]

        # Combine with NER-based anomalies
        ner_anomalies = self.ner.detect_entity_anomalies(texts)
        anomalies = [a or n for a, n in zip(anomalies, ner_anomalies, strict=False)]

        return anomalies

    def score(self, texts: list[str]) -> list[float]:
        """Calculate anomaly scores for texts."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring")

        # Preprocess texts
        processed_texts = self.preprocessor.preprocess_corpus(texts)

        # Encode texts
        features = self.encoder.transform(processed_texts)

        # Calculate scores
        if hasattr(self.detector, 'decision_function'):
            scores = self.detector.decision_function(features)
            # Convert to anomaly scores (higher = more anomalous)
            scores = -scores
        elif hasattr(self.detector, 'score_samples'):
            scores = self.detector.score_samples(features)
            scores = -scores
        else:
            # Fallback: use distance-based scoring
            scores = np.ones(len(texts)) * 0.5

        # Normalize scores to [0, 1]
        if len(scores) > 1:
            min_score, max_score = np.min(scores), np.max(scores)
            if max_score > min_score:
                scores = (scores - min_score) / (max_score - min_score)

        return scores.tolist()

    def detect(self, data: Any, **kwargs) -> DetectionResult:
        """Main detection method following DetectorProtocol."""
        texts = self._extract_texts(data)

        anomalies = self.predict(texts)
        scores = self.score(texts)

        anomaly_indices = [i for i, is_anomaly in enumerate(anomalies) if is_anomaly]
        anomaly_scores = [AnomalyScore(value=score) for score in scores]

        return DetectionResult(
            anomaly_indices=anomaly_indices,
            anomaly_scores=anomaly_scores,
            anomaly_type=AnomalyType.LINGUISTIC,
            metadata={
                "algorithm": self.config.algorithm,
                "encoding_method": self.config.encoding_method,
                "num_documents": len(texts),
                "num_anomalies": len(anomaly_indices),
                "contamination_rate": len(anomaly_indices) / len(texts) if texts else 0
            }
        )


# Factory function for easy instantiation
def create_text_detector(
    algorithm: str = "isolation_forest",
    encoding_method: str = "tfidf",
    **kwargs
) -> TextAnomalyDetector:
    """Create a text anomaly detector with specified configuration."""
    config = TextDetectionConfig(
        algorithm=algorithm,
        encoding_method=encoding_method,
        **kwargs
    )
    return TextAnomalyDetector(config)
