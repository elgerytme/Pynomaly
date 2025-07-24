"""Advanced NLP model entities for large language models and text processing."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


class LanguageTaskType(Enum):
    """Types of NLP tasks."""
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    QUESTION_ANSWERING = "question_answering"
    TEXT_SUMMARIZATION = "text_summarization"
    MACHINE_TRANSLATION = "machine_translation"
    LANGUAGE_DETECTION = "language_detection"
    TEXT_EMBEDDINGS = "text_embeddings"
    CONVERSATION = "conversation"
    CODE_GENERATION = "code_generation"
    FACT_CHECKING = "fact_checking"
    TOXICITY_DETECTION = "toxicity_detection"
    BIAS_DETECTION = "bias_detection"
    ZERO_SHOT_CLASSIFICATION = "zero_shot_classification"
    FEW_SHOT_LEARNING = "few_shot_learning"


class ModelArchitecture(Enum):
    """Language model architectures."""
    TRANSFORMER = "transformer"
    GPT = "gpt"
    BERT = "bert"
    ROBERTA = "roberta"
    T5 = "t5"
    BART = "bart"
    ELECTRA = "electra"
    DEBERTA = "deberta"
    LLAMA = "llama"
    CLAUDE = "claude"
    PALM = "palm"
    FALCON = "falcon"
    MISTRAL = "mistral"
    GEMINI = "gemini"
    CUSTOM = "custom"


class ModelSize(Enum):
    """Model size categories."""
    TINY = "tiny"          # < 100M parameters
    SMALL = "small"        # 100M - 1B parameters
    BASE = "base"          # 1B - 10B parameters
    LARGE = "large"        # 10B - 100B parameters
    XLARGE = "xlarge"      # 100B - 1T parameters
    XXLARGE = "xxlarge"    # > 1T parameters


class ResponseFormat(Enum):
    """Response format types."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    CODE = "code"
    STRUCTURED = "structured"


@dataclass
class TextSpan:
    """Represents a span of text with metadata."""
    start: int
    end: int
    text: str
    label: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate text span."""
        if self.start < 0 or self.end < 0:
            raise ValueError("Start and end positions must be non-negative")
        if self.start >= self.end:
            raise ValueError("Start position must be less than end position")
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")
    
    @property
    def length(self) -> int:
        """Get length of the text span."""
        return self.end - self.start
    
    def overlaps_with(self, other: 'TextSpan') -> bool:
        """Check if this span overlaps with another span."""
        return not (self.end <= other.start or other.end <= self.start)
    
    def contains(self, other: 'TextSpan') -> bool:
        """Check if this span contains another span."""
        return self.start <= other.start and other.end <= self.end


@dataclass
class Entity:
    """Represents a named entity."""
    span: TextSpan
    entity_type: str
    entity_id: Optional[str] = None
    normalized_form: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def text(self) -> str:
        """Get entity text."""
        return self.span.text
    
    @property
    def confidence(self) -> float:
        """Get entity confidence."""
        return self.span.confidence


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    subject: Entity
    predicate: str
    object: Entity
    confidence: float
    evidence_spans: List[TextSpan] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate relationship."""
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    
    # Advanced sampling parameters
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    diversity_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
    
    # Response format
    response_format: ResponseFormat = ResponseFormat.TEXT
    
    def __post_init__(self):
        """Validate generation configuration."""
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
        if not (0 <= self.temperature <= 2):
            raise ValueError("Temperature must be between 0 and 2")
        if not (0 <= self.top_p <= 1):
            raise ValueError("Top-p must be between 0 and 1")
        if self.top_k <= 0:
            raise ValueError("Top-k must be positive")


@dataclass
class ConversationMessage:
    """Represents a message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional structured content
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate message."""
        valid_roles = ["user", "assistant", "system", "function", "tool"]
        if self.role not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}")


@dataclass
class Conversation:
    """Represents a conversation with context."""
    conversation_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    system_prompt: Optional[str] = None
    context_window: int = 4096
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Conversation metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def add_message(self, message: ConversationMessage):
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_context(self, max_tokens: Optional[int] = None) -> List[ConversationMessage]:
        """Get conversation context within token limit."""
        if max_tokens is None:
            max_tokens = self.context_window
        
        # Simple token counting (rough approximation)
        current_tokens = 0
        context_messages = []
        
        # Add system prompt first if it exists
        if self.system_prompt:
            system_message = ConversationMessage(
                role="system",
                content=self.system_prompt
            )
            context_messages.append(system_message)
            current_tokens += len(self.system_prompt.split())
        
        # Add messages from most recent backwards
        for message in reversed(self.messages):
            message_tokens = len(message.content.split())
            if current_tokens + message_tokens > max_tokens:
                break
            context_messages.insert(-len([m for m in context_messages if m.role != "system"]), message)
            current_tokens += message_tokens
        
        return context_messages
    
    @property
    def message_count(self) -> int:
        """Get total number of messages."""
        return len(self.messages)
    
    @property
    def total_tokens_estimate(self) -> int:
        """Estimate total tokens in conversation."""
        total = 0
        if self.system_prompt:
            total += len(self.system_prompt.split())
        for message in self.messages:
            total += len(message.content.split())
        return total


@dataclass
class LanguagePrediction:
    """Represents a language model prediction."""
    task_type: LanguageTaskType
    input_text: str
    generated_text: Optional[str] = None
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Task-specific results
    classification_results: Optional[Dict[str, float]] = None
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    embeddings: Optional[np.ndarray] = None
    
    # Question answering specific
    answer: Optional[str] = None
    answer_span: Optional[TextSpan] = None
    context_relevance: Optional[float] = None
    
    # Generation specific
    generation_config: Optional[GenerationConfig] = None
    finish_reason: Optional[str] = None  # "stop", "length", "content_filter"
    token_count: Optional[int] = None
    
    # Quality metrics
    fluency_score: Optional[float] = None
    coherence_score: Optional[float] = None
    relevance_score: Optional[float] = None
    factuality_score: Optional[float] = None
    toxicity_score: Optional[float] = None
    bias_score: Optional[float] = None
    
    # Performance metrics
    tokens_per_second: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    
    # Model information
    model_version: Optional[str] = None
    model_size: Optional[ModelSize] = None
    
    def __post_init__(self):
        """Validate prediction."""
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")
        if self.processing_time_ms < 0:
            raise ValueError("Processing time must be non-negative")
    
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
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get entities of a specific type."""
        return [entity for entity in self.entities if entity.entity_type == entity_type]
    
    def calculate_quality_score(self) -> float:
        """Calculate overall quality score."""
        scores = []
        
        if self.fluency_score is not None:
            scores.append(self.fluency_score)
        if self.coherence_score is not None:
            scores.append(self.coherence_score)
        if self.relevance_score is not None:
            scores.append(self.relevance_score)
        if self.factuality_score is not None:
            scores.append(self.factuality_score)
        
        # Penalize for toxicity and bias
        if self.toxicity_score is not None:
            scores.append(1.0 - self.toxicity_score)
        if self.bias_score is not None:
            scores.append(1.0 - self.bias_score)
        
        return np.mean(scores) if scores else 0.0


@dataclass
class LanguageModelMetadata:
    """Metadata for language models."""
    model_id: str
    name: str
    version: str
    architecture: ModelArchitecture
    model_size: ModelSize
    parameter_count: int
    created_at: datetime
    
    # Model capabilities
    supported_tasks: List[LanguageTaskType]
    supported_languages: List[str]
    context_window: int
    max_generation_length: int
    
    # Performance characteristics
    inference_speed_tokens_per_sec: Optional[float] = None
    memory_requirements_gb: Optional[float] = None
    
    # Training information
    training_data_cutoff: Optional[datetime] = None
    training_tokens: Optional[int] = None
    training_compute_hours: Optional[float] = None
    
    # Quality metrics
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    safety_rating: Optional[str] = None
    bias_evaluation: Dict[str, float] = field(default_factory=dict)
    
    # Technical specifications
    vocabulary_size: Optional[int] = None
    tokenizer_type: Optional[str] = None
    attention_mechanism: Optional[str] = None
    activation_function: Optional[str] = None
    
    # Deployment information
    framework: str = "transformers"
    quantization: Optional[str] = None
    optimization_level: Optional[str] = None
    hardware_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Licensing and usage
    license: Optional[str] = None
    usage_restrictions: List[str] = field(default_factory=list)
    ethical_guidelines: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate metadata."""
        if self.parameter_count <= 0:
            raise ValueError("Parameter count must be positive")
        if self.context_window <= 0:
            raise ValueError("Context window must be positive")
        if self.max_generation_length <= 0:
            raise ValueError("Max generation length must be positive")
        if not self.supported_tasks:
            raise ValueError("Supported tasks cannot be empty")
        if not self.supported_languages:
            raise ValueError("Supported languages cannot be empty")
    
    def supports_task(self, task: LanguageTaskType) -> bool:
        """Check if model supports a specific task."""
        return task in self.supported_tasks
    
    def supports_language(self, language: str) -> bool:
        """Check if model supports a specific language."""
        return language in self.supported_languages
    
    def get_model_size_category(self) -> str:
        """Get human-readable model size category."""
        size_descriptions = {
            ModelSize.TINY: f"Tiny ({self.parameter_count/1e6:.0f}M parameters)",
            ModelSize.SMALL: f"Small ({self.parameter_count/1e9:.1f}B parameters)",
            ModelSize.BASE: f"Base ({self.parameter_count/1e9:.1f}B parameters)",
            ModelSize.LARGE: f"Large ({self.parameter_count/1e9:.0f}B parameters)",
            ModelSize.XLARGE: f"XLarge ({self.parameter_count/1e12:.1f}T parameters)",
            ModelSize.XXLARGE: f"XXLarge ({self.parameter_count/1e12:.1f}T parameters)"
        }
        return size_descriptions.get(self.model_size, f"Unknown ({self.parameter_count} parameters)")


@dataclass
class PromptTemplate:
    """Template for structured prompts."""
    template_id: str
    name: str
    description: str
    template_text: str
    variables: List[str]
    task_type: LanguageTaskType
    created_at: datetime = field(default_factory=datetime.now)
    
    # Template metadata
    author: Optional[str] = None
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    
    # Performance tracking
    usage_count: int = 0
    average_rating: Optional[float] = None
    success_rate: Optional[float] = None
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        try:
            return self.template_text.format(**kwargs)
        except KeyError as e:
            missing_var = str(e).strip("'")
            raise ValueError(f"Missing required variable: {missing_var}")
    
    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Validate that all required variables are provided."""
        missing = []
        for var in self.variables:
            if var not in variables:
                missing.append(var)
        return missing
    
    def get_variable_placeholders(self) -> List[str]:
        """Extract variable placeholders from template text."""
        import re
        pattern = r'\{([^}]+)\}'
        return re.findall(pattern, self.template_text)


@dataclass
class LanguageDataset:
    """Represents a language dataset."""
    dataset_id: str
    name: str
    task_type: LanguageTaskType
    version: str
    created_at: datetime
    
    # Dataset statistics
    total_samples: int
    training_samples: int
    validation_samples: int
    test_samples: int
    
    # Language characteristics
    languages: List[str]
    vocabulary_size: Optional[int] = None
    average_sequence_length: Optional[float] = None
    max_sequence_length: Optional[int] = None
    
    # Dataset quality metrics
    quality_score: Optional[float] = None
    diversity_score: Optional[float] = None
    bias_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Content characteristics
    domains: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    text_sources: List[str] = field(default_factory=list)
    
    # Annotation information
    annotation_guidelines: Optional[str] = None
    annotator_agreement: Optional[float] = None
    quality_control_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Storage and access
    storage_format: str = "jsonl"
    compressed_size_gb: float = 0.0
    uncompressed_size_gb: float = 0.0
    access_url: Optional[str] = None
    
    # Metadata
    description: str = ""
    citation: Optional[str] = None
    license: Optional[str] = None
    ethical_considerations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate dataset."""
        if self.total_samples != (self.training_samples + self.validation_samples + self.test_samples):
            raise ValueError("Total samples must equal sum of splits")
        if self.total_samples <= 0:
            raise ValueError("Total samples must be positive")
        if not self.languages:
            raise ValueError("Languages cannot be empty")
    
    @property
    def samples_per_language(self) -> float:
        """Average samples per language."""
        return self.total_samples / len(self.languages)
    
    def get_split_ratios(self) -> Tuple[float, float, float]:
        """Get train/validation/test split ratios."""
        total = self.total_samples
        return (
            self.training_samples / total,
            self.validation_samples / total,
            self.test_samples / total
        )


@dataclass
class LanguageExperiment:
    """Represents a language model experiment."""
    experiment_id: str
    name: str
    task_type: LanguageTaskType
    created_at: datetime
    updated_at: datetime
    
    # Experiment configuration
    model_architecture: ModelArchitecture
    dataset_id: str
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    
    # Results and metrics
    best_score: Optional[float] = None
    final_loss: Optional[float] = None
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_results: Dict[str, float] = field(default_factory=dict)
    
    # Model artifacts
    model_checkpoints: List[str] = field(default_factory=list)
    final_model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    
    # Experiment tracking
    status: str = "running"  # running, completed, failed, cancelled
    progress_percentage: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    
    # Resource usage
    training_time_hours: Optional[float] = None
    gpu_hours: Optional[float] = None
    compute_cost_usd: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    
    # Quality and safety metrics
    bias_evaluation: Dict[str, float] = field(default_factory=dict)
    safety_scores: Dict[str, float] = field(default_factory=dict)
    human_evaluation: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    researcher: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def add_training_step(self, step: int, metrics: Dict[str, float]):
        """Add training step metrics."""
        metric_entry = {"step": step, "timestamp": datetime.now(), **metrics}
        self.training_history.append(metric_entry)
        self.updated_at = datetime.now()
    
    def update_progress(self, epoch: int, total_epochs: int):
        """Update experiment progress."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        self.progress_percentage = (epoch / total_epochs) * 100 if total_epochs > 0 else 0
        self.updated_at = datetime.now()
    
    def get_best_checkpoint(self, metric: str = "loss") -> Optional[str]:
        """Get path to best model checkpoint based on metric."""
        if not self.training_history:
            return None
        
        best_step = 0
        best_value = float('inf') if metric == "loss" else float('-inf')
        
        for entry in self.training_history:
            if metric in entry:
                value = entry[metric]
                if metric == "loss" and value < best_value:
                    best_value = value
                    best_step = entry["step"]
                elif metric != "loss" and value > best_value:
                    best_value = value
                    best_step = entry["step"]
        
        # Find corresponding checkpoint
        for checkpoint in self.model_checkpoints:
            if f"step-{best_step}" in checkpoint or f"checkpoint-{best_step}" in checkpoint:
                return checkpoint
        
        return None