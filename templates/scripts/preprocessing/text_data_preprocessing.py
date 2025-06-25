#!/usr/bin/env python3
"""
Text Data Preprocessing Pipeline Template

This template provides a comprehensive preprocessing pipeline specifically designed
for text data anomaly detection including NLP preprocessing, feature extraction, and anomaly-ready preparation.
"""

import json
import re
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# NLP and text processing imports
import logging

import nltk
import textstat
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

# Statistical imports
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

# Text feature extraction
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)

# Text similarity and clustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


class TextDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for text datasets in anomaly detection.

    Features:
    - Text cleaning and normalization
    - Advanced NLP preprocessing
    - Feature extraction (TF-IDF, N-grams, embeddings)
    - Readability and complexity metrics
    - Semantic similarity analysis
    - Anomaly-ready feature engineering
    """

    def __init__(
        self,
        config: dict[str, Any] = None,
        preserve_original: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize the text data preprocessor.

        Args:
            config: Configuration dictionary for preprocessing steps
            preserve_original: Whether to preserve original text columns
            verbose: Enable detailed logging
        """
        self.config = config or self._get_default_config()
        self.preserve_original = preserve_original
        self.verbose = verbose

        # Initialize NLP components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Initialize vectorizers and models
        self.vectorizers = {}
        self.topic_models = {}
        self.scalers = {}

        # Metadata tracking
        self.preprocessing_steps = []
        self.data_profile = {}
        self.vocabulary_stats = {}

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for text data preprocessing."""
        return {
            "text_cleaning": {
                "remove_html": True,
                "remove_urls": True,
                "remove_emails": True,
                "remove_phone_numbers": True,
                "remove_special_chars": True,
                "normalize_whitespace": True,
                "convert_to_lowercase": True,
                "remove_numbers": False,
                "min_length": 10,  # Minimum text length to keep
            },
            "tokenization": {
                "method": "word",  # 'word', 'sentence'
                "remove_stopwords": True,
                "remove_punctuation": True,
                "min_word_length": 2,
                "max_word_length": 50,
            },
            "normalization": {
                "method": "lemmatization",  # 'stemming', 'lemmatization', 'both'
                "preserve_pos": True,  # Preserve part-of-speech for lemmatization
                "handle_negations": True,
            },
            "feature_extraction": {
                "tfidf": {
                    "enable": True,
                    "max_features": 1000,
                    "ngram_range": (1, 2),
                    "min_df": 2,
                    "max_df": 0.95,
                },
                "count_vectorizer": {
                    "enable": False,
                    "max_features": 500,
                    "ngram_range": (1, 1),
                },
                "hashing_vectorizer": {"enable": False, "n_features": 1000},
            },
            "linguistic_features": {
                "readability_metrics": True,
                "complexity_metrics": True,
                "sentiment_features": True,
                "pos_tag_features": True,
                "named_entity_features": True,
            },
            "dimensionality_reduction": {
                "method": "svd",  # 'svd', 'lda', 'pca'
                "n_components": 100,
                "apply_to_tfidf": True,
            },
            "anomaly_features": {
                "document_similarity": True,
                "cluster_features": True,
                "statistical_features": True,
                "n_clusters": 10,
            },
            "validation": {
                "min_vocabulary_size": 100,
                "max_vocabulary_size": 10000,
                "min_documents": 10,
            },
        }

    def preprocess(
        self,
        data: pd.DataFrame,
        text_columns: list[str],
        metadata_columns: list[str] = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Apply comprehensive preprocessing pipeline to text data.

        Args:
            data: Input DataFrame
            text_columns: List of text column names
            metadata_columns: List of metadata columns to preserve

        Returns:
            Tuple of (processed_data, preprocessing_metadata)
        """
        logger.info("Starting text data preprocessing pipeline")

        # Create copy to avoid modifying original
        df = data.copy()
        original_shape = df.shape

        # Preserve metadata columns
        if metadata_columns is None:
            metadata_columns = [col for col in df.columns if col not in text_columns]

        # 1. Text Data Validation
        self._log_step("Text Data Validation")
        validation_results = self._validate_text_data(df, text_columns)

        # 2. Text Cleaning
        self._log_step("Text Cleaning and Normalization")
        df = self._clean_text_data(df, text_columns)

        # 3. Text Preprocessing
        self._log_step("Advanced Text Preprocessing")
        df = self._preprocess_text(df, text_columns)

        # 4. Feature Extraction
        self._log_step("Text Feature Extraction")
        df = self._extract_text_features(df, text_columns)

        # 5. Linguistic Features
        self._log_step("Linguistic Feature Engineering")
        df = self._engineer_linguistic_features(df, text_columns)

        # 6. Dimensionality Reduction
        self._log_step("Dimensionality Reduction")
        df = self._apply_dimensionality_reduction(df)

        # 7. Anomaly-Specific Features
        self._log_step("Anomaly-Specific Feature Engineering")
        df = self._engineer_anomaly_features(df, text_columns)

        # 8. Feature Scaling
        self._log_step("Feature Scaling")
        df = self._scale_features(df, metadata_columns)

        # 9. Final Validation
        self._log_step("Final Validation")
        final_validation = self._final_validation(df, original_shape)

        # Prepare metadata
        metadata = {
            "preprocessing_steps": self.preprocessing_steps,
            "data_profile": self.data_profile,
            "vocabulary_stats": self.vocabulary_stats,
            "validation_results": validation_results,
            "final_validation": final_validation,
            "original_shape": original_shape,
            "final_shape": df.shape,
            "config": self.config,
            "text_columns": text_columns,
            "metadata_columns": metadata_columns,
        }

        logger.info(f"Preprocessing complete: {original_shape} -> {df.shape}")
        return df, metadata

    def _validate_text_data(
        self, df: pd.DataFrame, text_columns: list[str]
    ) -> dict[str, Any]:
        """Validate text data for common issues."""
        validation_results = {
            "total_documents": len(df),
            "text_columns": len(text_columns),
            "text_analysis": {},
            "issues": [],
        }

        for col in text_columns:
            if col in df.columns:
                text_data = df[col].astype(str)

                # Basic text statistics
                text_stats = {
                    "missing_ratio": df[col].isnull().sum() / len(df),
                    "empty_strings": (text_data == "").sum(),
                    "avg_length": text_data.str.len().mean(),
                    "min_length": text_data.str.len().min(),
                    "max_length": text_data.str.len().max(),
                    "std_length": text_data.str.len().std(),
                    "unique_documents": text_data.nunique(),
                    "duplicate_ratio": 1 - (text_data.nunique() / len(text_data)),
                }

                # Check for issues
                if text_stats["missing_ratio"] > 0.1:
                    validation_results["issues"].append(
                        {
                            "type": "high_missing_text",
                            "column": col,
                            "ratio": text_stats["missing_ratio"],
                            "severity": "high",
                        }
                    )

                if (
                    text_stats["avg_length"]
                    < self.config["text_cleaning"]["min_length"]
                ):
                    validation_results["issues"].append(
                        {
                            "type": "short_documents",
                            "column": col,
                            "avg_length": text_stats["avg_length"],
                            "severity": "medium",
                        }
                    )

                if text_stats["duplicate_ratio"] > 0.3:
                    validation_results["issues"].append(
                        {
                            "type": "high_duplicates",
                            "column": col,
                            "ratio": text_stats["duplicate_ratio"],
                            "severity": "medium",
                        }
                    )

                validation_results["text_analysis"][col] = text_stats

        self.data_profile["validation"] = validation_results
        return validation_results

    def _clean_text_data(
        self, df: pd.DataFrame, text_columns: list[str]
    ) -> pd.DataFrame:
        """Clean and normalize text data."""
        cleaning_config = self.config["text_cleaning"]
        cleaning_info = {}

        for col in text_columns:
            if col in df.columns:
                original_col = f"{col}_original" if self.preserve_original else None
                if self.preserve_original:
                    df[original_col] = df[col].copy()

                # Convert to string and handle nulls
                df[col] = df[col].astype(str).fillna("")

                # Apply cleaning steps
                if cleaning_config["remove_html"]:
                    df[col] = df[col].apply(self._remove_html)

                if cleaning_config["remove_urls"]:
                    df[col] = df[col].apply(self._remove_urls)

                if cleaning_config["remove_emails"]:
                    df[col] = df[col].apply(self._remove_emails)

                if cleaning_config["remove_phone_numbers"]:
                    df[col] = df[col].apply(self._remove_phone_numbers)

                if cleaning_config["convert_to_lowercase"]:
                    df[col] = df[col].str.lower()

                if cleaning_config["remove_special_chars"]:
                    df[col] = df[col].apply(self._remove_special_chars)

                if cleaning_config["remove_numbers"]:
                    df[col] = df[col].apply(self._remove_numbers)

                if cleaning_config["normalize_whitespace"]:
                    df[col] = df[col].apply(self._normalize_whitespace)

                # Remove documents that are too short
                min_length = cleaning_config["min_length"]
                short_docs = df[col].str.len() < min_length
                cleaning_info[col] = {
                    "short_documents_removed": short_docs.sum(),
                    "documents_remaining": (~short_docs).sum(),
                }

                # Mark short documents for potential removal
                df.loc[short_docs, col] = ""

        self.preprocessing_steps.append(
            {
                "step": "text_cleaning",
                "cleaning_operations": list(cleaning_config.keys()),
                "cleaning_results": cleaning_info,
            }
        )

        return df

    def _preprocess_text(
        self, df: pd.DataFrame, text_columns: list[str]
    ) -> pd.DataFrame:
        """Apply advanced text preprocessing."""
        tokenization_config = self.config["tokenization"]
        normalization_config = self.config["normalization"]

        for col in text_columns:
            if col in df.columns:
                processed_col = f"{col}_processed"

                # Tokenize and preprocess each document
                df[processed_col] = df[col].apply(
                    lambda text: self._preprocess_document(
                        text, tokenization_config, normalization_config
                    )
                )

                # Update vocabulary statistics
                self._update_vocabulary_stats(df[processed_col], col)

        self.preprocessing_steps.append(
            {
                "step": "text_preprocessing",
                "tokenization_method": tokenization_config["method"],
                "normalization_method": normalization_config["method"],
            }
        )

        return df

    def _preprocess_document(
        self, text: str, tokenization_config: dict, normalization_config: dict
    ) -> str:
        """Preprocess a single document."""
        if not text or len(text.strip()) == 0:
            return ""

        # Tokenization
        if tokenization_config["method"] == "word":
            tokens = word_tokenize(text)
        else:
            # Sentence tokenization followed by word tokenization
            sentences = sent_tokenize(text)
            tokens = []
            for sentence in sentences:
                tokens.extend(word_tokenize(sentence))

        # Filter tokens
        min_length = tokenization_config["min_word_length"]
        max_length = tokenization_config["max_word_length"]

        filtered_tokens = []
        for token in tokens:
            # Remove punctuation if configured
            if tokenization_config["remove_punctuation"] and not token.isalnum():
                continue

            # Length filter
            if len(token) < min_length or len(token) > max_length:
                continue

            # Stopword filter
            if (
                tokenization_config["remove_stopwords"]
                and token.lower() in self.stop_words
            ):
                continue

            filtered_tokens.append(token)

        # Normalization
        if normalization_config["method"] in ["stemming", "both"]:
            filtered_tokens = [self.stemmer.stem(token) for token in filtered_tokens]

        if normalization_config["method"] in ["lemmatization", "both"]:
            if normalization_config["preserve_pos"]:
                # POS-aware lemmatization
                pos_tags = pos_tag(filtered_tokens)
                filtered_tokens = [
                    self.lemmatizer.lemmatize(token, self._get_wordnet_pos(pos))
                    for token, pos in pos_tags
                ]
            else:
                filtered_tokens = [
                    self.lemmatizer.lemmatize(token) for token in filtered_tokens
                ]

        return " ".join(filtered_tokens)

    def _extract_text_features(
        self, df: pd.DataFrame, text_columns: list[str]
    ) -> pd.DataFrame:
        """Extract numerical features from text data."""
        feature_config = self.config["feature_extraction"]

        for col in text_columns:
            processed_col = f"{col}_processed"
            if processed_col in df.columns:
                documents = df[processed_col].tolist()

                # Remove empty documents for feature extraction
                non_empty_docs = [doc for doc in documents if doc.strip()]
                if len(non_empty_docs) < self.config["validation"]["min_documents"]:
                    logger.warning(
                        f"Too few non-empty documents for {col}, skipping feature extraction"
                    )
                    continue

                # TF-IDF Features
                if feature_config["tfidf"]["enable"]:
                    tfidf_config = feature_config["tfidf"]
                    vectorizer = TfidfVectorizer(
                        max_features=tfidf_config["max_features"],
                        ngram_range=tfidf_config["ngram_range"],
                        min_df=tfidf_config["min_df"],
                        max_df=tfidf_config["max_df"],
                        stop_words="english",
                    )

                    # Handle empty documents
                    documents_for_tfidf = [
                        doc if doc.strip() else "empty_document" for doc in documents
                    ]
                    tfidf_features = vectorizer.fit_transform(documents_for_tfidf)

                    # Create feature names
                    feature_names = [
                        f"{col}_tfidf_{i}" for i in range(tfidf_features.shape[1])
                    ]
                    tfidf_df = pd.DataFrame(
                        tfidf_features.toarray(), columns=feature_names, index=df.index
                    )

                    # Add to main dataframe
                    df = pd.concat([df, tfidf_df], axis=1)
                    self.vectorizers[f"{col}_tfidf"] = vectorizer

                # Count Vectorizer Features
                if feature_config["count_vectorizer"]["enable"]:
                    count_config = feature_config["count_vectorizer"]
                    vectorizer = CountVectorizer(
                        max_features=count_config["max_features"],
                        ngram_range=count_config["ngram_range"],
                        stop_words="english",
                    )

                    documents_for_count = [
                        doc if doc.strip() else "empty_document" for doc in documents
                    ]
                    count_features = vectorizer.fit_transform(documents_for_count)

                    feature_names = [
                        f"{col}_count_{i}" for i in range(count_features.shape[1])
                    ]
                    count_df = pd.DataFrame(
                        count_features.toarray(), columns=feature_names, index=df.index
                    )

                    df = pd.concat([df, count_df], axis=1)
                    self.vectorizers[f"{col}_count"] = vectorizer

                # Hashing Vectorizer Features
                if feature_config["hashing_vectorizer"]["enable"]:
                    hash_config = feature_config["hashing_vectorizer"]
                    vectorizer = HashingVectorizer(
                        n_features=hash_config["n_features"], stop_words="english"
                    )

                    documents_for_hash = [
                        doc if doc.strip() else "empty_document" for doc in documents
                    ]
                    hash_features = vectorizer.transform(documents_for_hash)

                    feature_names = [
                        f"{col}_hash_{i}" for i in range(hash_features.shape[1])
                    ]
                    hash_df = pd.DataFrame(
                        hash_features.toarray(), columns=feature_names, index=df.index
                    )

                    df = pd.concat([df, hash_df], axis=1)
                    self.vectorizers[f"{col}_hash"] = vectorizer

        self.preprocessing_steps.append(
            {
                "step": "text_feature_extraction",
                "vectorizers_used": list(self.vectorizers.keys()),
            }
        )

        return df

    def _engineer_linguistic_features(
        self, df: pd.DataFrame, text_columns: list[str]
    ) -> pd.DataFrame:
        """Engineer linguistic and readability features."""
        linguistic_config = self.config["linguistic_features"]

        for col in text_columns:
            if col in df.columns:
                original_text = df[col].astype(str)

                # Readability metrics
                if linguistic_config["readability_metrics"]:
                    df[f"{col}_flesch_reading_ease"] = original_text.apply(
                        lambda x: textstat.flesch_reading_ease(x) if x.strip() else 0
                    )
                    df[f"{col}_flesch_kincaid_grade"] = original_text.apply(
                        lambda x: textstat.flesch_kincaid_grade(x) if x.strip() else 0
                    )
                    df[f"{col}_automated_readability_index"] = original_text.apply(
                        lambda x: (
                            textstat.automated_readability_index(x) if x.strip() else 0
                        )
                    )

                # Complexity metrics
                if linguistic_config["complexity_metrics"]:
                    df[f"{col}_sentence_count"] = original_text.apply(
                        lambda x: len(sent_tokenize(x)) if x.strip() else 0
                    )
                    df[f"{col}_word_count"] = original_text.apply(
                        lambda x: len(word_tokenize(x)) if x.strip() else 0
                    )
                    df[f"{col}_avg_sentence_length"] = original_text.apply(
                        lambda x: (
                            np.mean(
                                [len(word_tokenize(sent)) for sent in sent_tokenize(x)]
                            )
                            if x.strip() and sent_tokenize(x)
                            else 0
                        )
                    )
                    df[f"{col}_syllable_count"] = original_text.apply(
                        lambda x: textstat.syllable_count(x) if x.strip() else 0
                    )

                # Part-of-speech features
                if linguistic_config["pos_tag_features"]:
                    pos_features = original_text.apply(self._extract_pos_features)
                    pos_df = pd.DataFrame(
                        pos_features.tolist(),
                        index=df.index,
                        columns=[
                            f"{col}_pos_{tag}"
                            for tag in ["NN", "VB", "JJ", "RB", "PRP"]
                        ],
                    )
                    df = pd.concat([df, pos_df], axis=1)

                # Named entity features
                if linguistic_config["named_entity_features"]:
                    df[f"{col}_named_entities_count"] = original_text.apply(
                        self._count_named_entities
                    )

        self.preprocessing_steps.append(
            {
                "step": "linguistic_feature_engineering",
                "features_added": [
                    "readability",
                    "complexity",
                    "pos",
                    "named_entities",
                ],
            }
        )

        return df

    def _apply_dimensionality_reduction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply dimensionality reduction to high-dimensional text features."""
        reduction_config = self.config["dimensionality_reduction"]

        if not reduction_config["apply_to_tfidf"]:
            return df

        # Find TF-IDF columns
        tfidf_columns = [col for col in df.columns if "_tfidf_" in col]

        if len(tfidf_columns) > reduction_config["n_components"]:
            method = reduction_config["method"]
            n_components = reduction_config["n_components"]

            if method == "svd":
                reducer = TruncatedSVD(n_components=n_components, random_state=42)
                reduced_features = reducer.fit_transform(df[tfidf_columns])

                # Replace TF-IDF columns with reduced features
                df = df.drop(columns=tfidf_columns)

                reduced_df = pd.DataFrame(
                    reduced_features,
                    columns=[f"text_svd_{i}" for i in range(n_components)],
                    index=df.index,
                )
                df = pd.concat([df, reduced_df], axis=1)

                self.topic_models["svd"] = reducer

            elif method == "lda":
                # LDA requires non-negative features
                count_columns = [col for col in df.columns if "_count_" in col]
                if len(count_columns) > 0:
                    lda = LatentDirichletAllocation(
                        n_components=n_components, random_state=42, max_iter=10
                    )
                    lda_features = lda.fit_transform(df[count_columns])

                    lda_df = pd.DataFrame(
                        lda_features,
                        columns=[f"text_lda_{i}" for i in range(n_components)],
                        index=df.index,
                    )
                    df = pd.concat([df, lda_df], axis=1)

                    self.topic_models["lda"] = lda

            self.preprocessing_steps.append(
                {
                    "step": "dimensionality_reduction",
                    "method": method,
                    "original_features": len(tfidf_columns),
                    "reduced_features": n_components,
                }
            )

        return df

    def _engineer_anomaly_features(
        self, df: pd.DataFrame, text_columns: list[str]
    ) -> pd.DataFrame:
        """Engineer features specifically useful for anomaly detection."""
        anomaly_config = self.config["anomaly_features"]

        # Document similarity features
        if anomaly_config["document_similarity"]:
            for col in text_columns:
                processed_col = f"{col}_processed"
                if processed_col in df.columns:
                    # Calculate pairwise similarity
                    documents = df[processed_col].tolist()

                    if f"{col}_tfidf" in self.vectorizers:
                        # Use existing TF-IDF vectorizer
                        vectorizer = self.vectorizers[f"{col}_tfidf"]
                        doc_vectors = vectorizer.transform(documents)

                        # Calculate average similarity to all other documents
                        similarity_matrix = cosine_similarity(doc_vectors)

                        # Average similarity (excluding self-similarity)
                        np.fill_diagonal(similarity_matrix, 0)
                        df[f"{col}_avg_similarity"] = np.mean(similarity_matrix, axis=1)

                        # Minimum similarity (most dissimilar document)
                        df[f"{col}_min_similarity"] = np.min(similarity_matrix, axis=1)

                        # Maximum similarity (most similar document)
                        df[f"{col}_max_similarity"] = np.max(similarity_matrix, axis=1)

        # Cluster-based features
        if anomaly_config["cluster_features"]:
            # Find numerical text features for clustering
            text_feature_columns = [
                col
                for col in df.columns
                if any(
                    text_type in col
                    for text_type in ["_tfidf_", "_count_", "_svd_", "_lda_"]
                )
            ]

            if len(text_feature_columns) > 0:
                n_clusters = anomaly_config["n_clusters"]

                # Perform clustering
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(df[text_feature_columns])

                df["text_cluster"] = cluster_labels

                # Distance to cluster center
                cluster_centers = clusterer.cluster_centers_
                distances_to_centers = []

                for i, (_, row) in enumerate(df.iterrows()):
                    cluster_id = cluster_labels[i]
                    point = row[text_feature_columns].values
                    center = cluster_centers[cluster_id]
                    distance = np.linalg.norm(point - center)
                    distances_to_centers.append(distance)

                df["text_cluster_distance"] = distances_to_centers

                # Cluster size (smaller clusters might be more anomalous)
                cluster_sizes = pd.Series(cluster_labels).value_counts()
                df["text_cluster_size"] = df["text_cluster"].map(cluster_sizes)

        # Statistical features
        if anomaly_config["statistical_features"]:
            for col in text_columns:
                original_text = df[col].astype(str)

                # Character-level statistics
                df[f"{col}_char_count"] = original_text.str.len()
                df[f"{col}_uppercase_ratio"] = original_text.apply(
                    lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
                )
                df[f"{col}_digit_ratio"] = original_text.apply(
                    lambda x: sum(1 for c in x if c.isdigit()) / max(len(x), 1)
                )
                df[f"{col}_special_char_ratio"] = original_text.apply(
                    lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace())
                    / max(len(x), 1)
                )

                # Vocabulary diversity
                df[f"{col}_unique_words"] = original_text.apply(
                    lambda x: len(set(word_tokenize(x.lower()))) if x.strip() else 0
                )
                df[f"{col}_vocabulary_diversity"] = df[f"{col}_unique_words"] / (
                    df[f"{col}_word_count"] + 1
                )

        self.preprocessing_steps.append(
            {
                "step": "anomaly_feature_engineering",
                "features_added": ["similarity", "clustering", "statistical"],
            }
        )

        return df

    def _scale_features(
        self, df: pd.DataFrame, metadata_columns: list[str]
    ) -> pd.DataFrame:
        """Scale numerical features for anomaly detection."""
        # Identify numerical columns to scale (exclude metadata and original text)
        exclude_columns = set(
            metadata_columns + [col for col in df.columns if col.endswith("_original")]
        )
        numerical_columns = [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude_columns
        ]

        if len(numerical_columns) > 0:
            scaler = StandardScaler()
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            self.scalers["numerical"] = scaler

            self.preprocessing_steps.append(
                {
                    "step": "feature_scaling",
                    "method": "standard",
                    "features_scaled": len(numerical_columns),
                }
            )

        return df

    def _final_validation(
        self, df: pd.DataFrame, original_shape: tuple[int, int]
    ) -> dict[str, Any]:
        """Perform final validation of processed text data."""
        validation_results = {
            "shape_change": f"{original_shape} -> {df.shape}",
            "missing_values": df.isnull().sum().sum(),
            "infinite_values": np.isinf(df.select_dtypes(include=[np.number]))
            .sum()
            .sum(),
            "text_features_created": len(
                [
                    col
                    for col in df.columns
                    if any(
                        feat_type in col
                        for feat_type in [
                            "_tfidf_",
                            "_count_",
                            "_svd_",
                            "_lda_",
                            "_similarity",
                            "_cluster",
                        ]
                    )
                ]
            ),
            "linguistic_features_created": len(
                [
                    col
                    for col in df.columns
                    if any(
                        feat_type in col
                        for feat_type in [
                            "_flesch_",
                            "_sentence_",
                            "_word_",
                            "_pos_",
                            "_named_",
                        ]
                    )
                ]
            ),
            "data_types": dict(df.dtypes.astype(str)),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "processing_success": True,
        }

        # Check for any remaining issues
        issues = []
        if validation_results["missing_values"] > 0:
            issues.append("Missing values still present")
        if validation_results["infinite_values"] > 0:
            issues.append("Infinite values detected")
        if validation_results["text_features_created"] == 0:
            issues.append("No text features were created")

        validation_results["issues"] = issues
        validation_results["processing_success"] = len(issues) == 0

        return validation_results

    # Helper methods for text processing
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        return re.sub(r"<[^>]+>", "", text)

    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return re.sub(r"\S+@\S+", "", text)

    def _remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text."""
        return re.sub(
            r"(\+?\d{1,4}[-.\s]?)?(\(?\d{1,3}\)?[-.\s]?)?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
            "",
            text,
        )

    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters but keep basic punctuation."""
        return re.sub(r"[^\w\s.,!?;:\-\'\"()]", " ", text)

    def _remove_numbers(self, text: str) -> str:
        """Remove standalone numbers from text."""
        return re.sub(r"\b\d+\b", "", text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        return re.sub(r"\s+", " ", text).strip()

    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Convert treebank POS tag to WordNet POS tag."""
        if treebank_tag.startswith("J"):
            return "a"  # adjective
        elif treebank_tag.startswith("V"):
            return "v"  # verb
        elif treebank_tag.startswith("N"):
            return "n"  # noun
        elif treebank_tag.startswith("R"):
            return "r"  # adverb
        else:
            return "n"  # default to noun

    def _extract_pos_features(self, text: str) -> list[float]:
        """Extract part-of-speech tag features."""
        if not text.strip():
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)

        # Count different POS types
        pos_counts = {"NN": 0, "VB": 0, "JJ": 0, "RB": 0, "PRP": 0}

        for _, pos in pos_tags:
            for key in pos_counts:
                if pos.startswith(key):
                    pos_counts[key] += 1
                    break

        total_words = len(tokens)
        if total_words == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        # Return ratios
        return [
            pos_counts[key] / total_words for key in ["NN", "VB", "JJ", "RB", "PRP"]
        ]

    def _count_named_entities(self, text: str) -> int:
        """Count named entities in text."""
        if not text.strip():
            return 0

        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)

            entity_count = 0
            for chunk in chunks:
                if hasattr(chunk, "label"):
                    entity_count += 1

            return entity_count
        except:
            return 0

    def _update_vocabulary_stats(self, processed_texts: pd.Series, column_name: str):
        """Update vocabulary statistics for a column."""
        all_words = []
        for text in processed_texts:
            if text and text.strip():
                all_words.extend(text.split())

        if all_words:
            vocab_stats = {
                "vocabulary_size": len(set(all_words)),
                "total_words": len(all_words),
                "avg_words_per_doc": len(all_words) / len(processed_texts),
                "most_common_words": pd.Series(all_words)
                .value_counts()
                .head(10)
                .to_dict(),
            }
            self.vocabulary_stats[column_name] = vocab_stats

    def _log_step(self, step_name: str):
        """Log preprocessing step."""
        if self.verbose:
            logger.info(f"Executing: {step_name}")

    def save_pipeline(self, filepath: str):
        """Save the preprocessing pipeline configuration and fitted components."""
        pipeline_data = {
            "config": self.config,
            "preprocessing_steps": self.preprocessing_steps,
            "vocabulary_stats": self.vocabulary_stats,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(pipeline_data, f, indent=2, default=str)

        logger.info(f"Pipeline saved to {filepath}")

    def load_pipeline(self, filepath: str):
        """Load a saved preprocessing pipeline configuration."""
        with open(filepath) as f:
            pipeline_data = json.load(f)

        self.config = pipeline_data["config"]
        self.preprocessing_steps = pipeline_data.get("preprocessing_steps", [])
        self.vocabulary_stats = pipeline_data.get("vocabulary_stats", {})

        logger.info(f"Pipeline loaded from {filepath}")


def main():
    """Example usage of the Text Data Preprocessor."""
    # Create sample text data
    np.random.seed(42)
    n_samples = 1000

    # Generate synthetic text data with different characteristics
    normal_texts = [
        "This is a normal document with standard content about business operations.",
        "The quarterly report shows positive growth in revenue and customer satisfaction.",
        "Our team has successfully completed the project within the given timeline.",
        "The meeting was productive and we discussed various strategic initiatives.",
        "Customer feedback has been overwhelmingly positive for our new product line.",
    ]

    anomalous_texts = [
        "!!!URGENT!!! CLICK HERE NOW FOR AMAZING OFFERS!!!! LIMITED TIME ONLY!!!!",
        "Your account has been SUSPENDED. Please verify your information immediately at suspicious-link.com",
        "asdkjfh askdjfh aksjdhf aksjdhf aksjdhf random gibberish text here",
        "🎉🎉🎉 CONGRATULATIONS YOU WON $1000000!!! CLAIM NOW!!! 🎉🎉🎉",
        "ERROR ERROR ERROR SYSTEM MALFUNCTION CONTACT ADMINISTRATOR IMMEDIATELY",
    ]

    # Create dataset
    texts = []
    labels = []

    # Add normal texts (90%)
    for _ in range(int(n_samples * 0.9)):
        texts.append(np.random.choice(normal_texts))
        labels.append(0)

    # Add anomalous texts (10%)
    for _ in range(int(n_samples * 0.1)):
        texts.append(np.random.choice(anomalous_texts))
        labels.append(1)

    # Create additional metadata
    data = {
        "document_id": range(n_samples),
        "text_content": texts,
        "category": np.random.choice(["business", "technical", "marketing"], n_samples),
        "source": np.random.choice(["email", "report", "chat", "review"], n_samples),
        "timestamp": pd.date_range("2023-01-01", periods=n_samples, freq="1H"),
        "is_anomaly": labels,  # Ground truth for evaluation
    }

    df = pd.DataFrame(data)

    print("Original Data Shape:", df.shape)
    print("\nSample Text Content:")
    print(df["text_content"].head(3).tolist())

    # Initialize preprocessor with custom config
    config = {
        "text_cleaning": {
            "remove_html": True,
            "remove_urls": True,
            "convert_to_lowercase": True,
            "min_length": 5,
        },
        "feature_extraction": {
            "tfidf": {"enable": True, "max_features": 200, "ngram_range": (1, 2)}
        },
        "linguistic_features": {
            "readability_metrics": True,
            "complexity_metrics": True,
            "pos_tag_features": True,
        },
        "anomaly_features": {
            "document_similarity": True,
            "cluster_features": True,
            "statistical_features": True,
            "n_clusters": 5,
        },
        "dimensionality_reduction": {
            "method": "svd",
            "n_components": 50,
            "apply_to_tfidf": True,
        },
    }

    preprocessor = TextDataPreprocessor(config=config, verbose=True)

    # Apply preprocessing
    text_columns = ["text_content"]
    metadata_columns = ["document_id", "category", "source", "timestamp", "is_anomaly"]

    processed_df, metadata = preprocessor.preprocess(df, text_columns, metadata_columns)

    print(f"\nProcessed Data Shape: {processed_df.shape}")
    print("\nPreprocessing Steps Applied:")
    for i, step in enumerate(metadata["preprocessing_steps"], 1):
        print(f"{i}. {step['step']}")

    print("\nVocabulary Statistics:")
    for col, stats in metadata["vocabulary_stats"].items():
        print(
            f"- {col}: {stats['vocabulary_size']} unique words, {stats['avg_words_per_doc']:.1f} avg words/doc"
        )

    print(
        f"\nText Features Created: {metadata['final_validation']['text_features_created']}"
    )
    print(
        f"Linguistic Features Created: {metadata['final_validation']['linguistic_features_created']}"
    )
    print(f"Memory Usage: {metadata['final_validation']['memory_usage_mb']:.2f} MB")

    # Save pipeline for reuse
    preprocessor.save_pipeline("text_preprocessing_pipeline.json")

    print("\nText preprocessing pipeline completed successfully!")

    # Show some feature examples
    print("\nSample Features (first 5 rows):")
    feature_cols = [
        col
        for col in processed_df.columns
        if any(
            feat_type in col
            for feat_type in [
                "_tfidf_",
                "_flesch_",
                "_word_",
                "_similarity",
                "_cluster",
            ]
        )
    ][
        :10
    ]  # Show first 10 feature columns

    if feature_cols:
        print(processed_df[feature_cols].head())


if __name__ == "__main__":
    main()
