"""Advanced NLP service for large language models and text processing."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from datetime import datetime

from nlp_advanced.domain.entities.language_model import (
    LanguageTaskType, ModelArchitecture, LanguagePrediction, LanguageModelMetadata,
    GenerationConfig, ConversationMessage, Conversation, Entity, Relationship,
    TextSpan, PromptTemplate, LanguageDataset, LanguageExperiment
)


class NLPService:
    """Advanced NLP service for large language models and text processing."""
    
    def __init__(self, model_cache_size: int = 5, device: str = "auto"):
        """Initialize the NLP service.
        
        Args:
            model_cache_size: Maximum number of models to keep in memory
            device: Computing device (auto, cpu, cuda, mps)
        """
        self.model_cache_size = model_cache_size
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Model management
        self._models: Dict[str, Any] = {}
        self._model_metadata: Dict[str, LanguageModelMetadata] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._performance_metrics: Dict[str, List[float]] = {}
        
        # Conversation management
        self._conversations: Dict[str, Conversation] = {}
        self._prompt_templates: Dict[str, PromptTemplate] = {}
        
        # Setup device
        self._setup_device()
        
    def _setup_device(self):
        """Setup compute device for inference."""
        try:
            import torch
            
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            
            self.logger.info(f"Using device: {self.device}")
            
        except ImportError:
            self.device = "cpu"
            self.logger.warning("PyTorch not available, using CPU")
    
    async def load_model(
        self,
        model_id: str,
        model_path: Union[str, Path],
        architecture: ModelArchitecture,
        tokenizer_path: Optional[Union[str, Path]] = None,
        metadata: Optional[LanguageModelMetadata] = None
    ) -> bool:
        """Load a language model and tokenizer.
        
        Args:
            model_id: Unique identifier for the model
            model_path: Path to the model file or HuggingFace model name
            architecture: Model architecture
            tokenizer_path: Path to tokenizer (optional)
            metadata: Model metadata
            
        Returns:
            True if model loaded successfully
        """
        try:
            self.logger.info(f"Loading model {model_id} from {model_path}")
            
            # Load model and tokenizer based on architecture
            model, tokenizer = await self._load_model_implementation(
                model_path, architecture, tokenizer_path
            )
            
            if model is None or tokenizer is None:
                self.logger.error(f"Failed to load model {model_id}")
                return False
            
            # Manage cache size
            if len(self._models) >= self.model_cache_size:
                # Remove oldest model
                oldest_model = next(iter(self._models))
                await self.unload_model(oldest_model)
            
            # Store model, tokenizer, and metadata
            self._models[model_id] = model
            self._tokenizers[model_id] = tokenizer
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
        architecture: ModelArchitecture,
        tokenizer_path: Optional[Union[str, Path]] = None
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Load model implementation based on architecture."""
        try:
            model = None
            tokenizer = None
            
            if architecture == ModelArchitecture.GPT:
                model, tokenizer = await self._load_gpt_model(model_path, tokenizer_path)
            elif architecture == ModelArchitecture.BERT:
                model, tokenizer = await self._load_bert_model(model_path, tokenizer_path)
            elif architecture == ModelArchitecture.T5:
                model, tokenizer = await self._load_t5_model(model_path, tokenizer_path)
            elif architecture == ModelArchitecture.LLAMA:
                model, tokenizer = await self._load_llama_model(model_path, tokenizer_path)
            elif architecture == ModelArchitecture.CLAUDE:
                model, tokenizer = await self._load_claude_model(model_path, tokenizer_path)
            elif architecture == ModelArchitecture.MISTRAL:
                model, tokenizer = await self._load_mistral_model(model_path, tokenizer_path)
            else:
                # Generic transformers loading
                model, tokenizer = await self._load_transformers_model(model_path, tokenizer_path)
            
            if model is not None:
                model.eval()
                if hasattr(model, 'to'):
                    model = model.to(self.device)
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading model implementation: {str(e)}")
            return None, None
    
    async def _load_gpt_model(self, model_path: Union[str, Path], tokenizer_path: Optional[Union[str, Path]] = None):
        """Load GPT model."""
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            
            model = GPT2LMHeadModel.from_pretrained(str(model_path))
            tokenizer = GPT2Tokenizer.from_pretrained(str(tokenizer_path or model_path))
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading GPT model: {str(e)}")
            return None, None
    
    async def _load_bert_model(self, model_path: Union[str, Path], tokenizer_path: Optional[Union[str, Path]] = None):
        """Load BERT model."""
        try:
            from transformers import BertModel, BertTokenizer
            
            model = BertModel.from_pretrained(str(model_path))
            tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path or model_path))
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading BERT model: {str(e)}")
            return None, None
    
    async def _load_t5_model(self, model_path: Union[str, Path], tokenizer_path: Optional[Union[str, Path]] = None):
        """Load T5 model."""
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            model = T5ForConditionalGeneration.from_pretrained(str(model_path))
            tokenizer = T5Tokenizer.from_pretrained(str(tokenizer_path or model_path))
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading T5 model: {str(e)}")
            return None, None
    
    async def _load_llama_model(self, model_path: Union[str, Path], tokenizer_path: Optional[Union[str, Path]] = None):
        """Load LLaMA model."""
        try:
            from transformers import LlamaForCausalLM, LlamaTokenizer
            
            model = LlamaForCausalLM.from_pretrained(str(model_path))
            tokenizer = LlamaTokenizer.from_pretrained(str(tokenizer_path or model_path))
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading LLaMA model: {str(e)}")
            return None, None
    
    async def _load_claude_model(self, model_path: Union[str, Path], tokenizer_path: Optional[Union[str, Path]] = None):
        """Load Claude model (placeholder - would use Anthropic API)."""
        try:
            # Placeholder for Claude API integration
            self.logger.info("Claude model loading - would use Anthropic API")
            return None, None
            
        except Exception as e:
            self.logger.error(f"Error loading Claude model: {str(e)}")
            return None, None
    
    async def _load_mistral_model(self, model_path: Union[str, Path], tokenizer_path: Optional[Union[str, Path]] = None):
        """Load Mistral model."""
        try:
            from transformers import MistralForCausalLM, AutoTokenizer
            
            model = MistralForCausalLM.from_pretrained(str(model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path or model_path))
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading Mistral model: {str(e)}")
            return None, None
    
    async def _load_transformers_model(self, model_path: Union[str, Path], tokenizer_path: Optional[Union[str, Path]] = None):
        """Load generic transformers model."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            model = AutoModel.from_pretrained(str(model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path or model_path))
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading transformers model: {str(e)}")
            return None, None
    
    async def predict_text(
        self,
        model_id: str,
        input_text: str,
        task_type: LanguageTaskType,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> LanguagePrediction:
        """Perform text prediction with a language model.
        
        Args:
            model_id: ID of the model to use
            input_text: Input text for processing
            task_type: Type of NLP task to perform
            generation_config: Configuration for text generation
            **kwargs: Additional task-specific parameters
            
        Returns:
            Language prediction results
        """
        start_time = time.time()
        
        try:
            # Validate model exists
            if model_id not in self._models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model = self._models[model_id]
            tokenizer = self._tokenizers[model_id]
            
            # Preprocess input
            processed_input = await self._preprocess_text(
                input_text, task_type, tokenizer
            )
            preprocess_time = (time.time() - start_time) * 1000
            
            # Run inference
            inference_start = time.time()
            raw_output = await self._run_text_inference(
                model, tokenizer, processed_input, task_type, generation_config, **kwargs
            )
            inference_time = (time.time() - inference_start) * 1000
            
            # Post-process results
            postprocess_start = time.time()
            prediction = await self._postprocess_text_results(
                raw_output, input_text, task_type, model_id, **kwargs
            )
            postprocess_time = (time.time() - postprocess_start) * 1000
            
            # Update prediction with timing information
            total_time = (time.time() - start_time) * 1000
            prediction.processing_time_ms = inference_time
            prediction.model_version = self._model_metadata.get(model_id, {}).version if model_id in self._model_metadata else None
            
            # Track performance
            self._performance_metrics[model_id].append(total_time)
            if len(self._performance_metrics[model_id]) > 1000:
                self._performance_metrics[model_id] = self._performance_metrics[model_id][-1000:]
            
            self.logger.info(f"Text prediction completed in {total_time:.2f}ms")
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error during text prediction: {str(e)}")
            # Return error prediction
            total_time = (time.time() - start_time) * 1000
            return LanguagePrediction(
                task_type=task_type,
                input_text=input_text,
                confidence=0.0,
                processing_time_ms=total_time
            )
    
    async def _preprocess_text(
        self,
        text: str,
        task_type: LanguageTaskType,
        tokenizer: Any
    ) -> Dict[str, Any]:
        """Preprocess text for model input."""
        try:
            import torch
            
            # Basic tokenization
            if task_type == LanguageTaskType.TEXT_GENERATION:
                # For generation tasks
                inputs = tokenizer.encode(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                )
            elif task_type in [LanguageTaskType.TEXT_CLASSIFICATION, LanguageTaskType.SENTIMENT_ANALYSIS]:
                # For classification tasks
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
            elif task_type == LanguageTaskType.NAMED_ENTITY_RECOGNITION:
                # For NER tasks
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_offsets_mapping=True
                )
            else:
                # Default processing
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
            
            # Move to device
            if isinstance(inputs, dict):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.device)
            
            return inputs
            
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {str(e)}")
            raise
    
    async def _run_text_inference(
        self,
        model: Any,
        tokenizer: Any,
        inputs: Dict[str, Any],
        task_type: LanguageTaskType,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Any:
        """Run model inference on preprocessed text."""
        try:
            import torch
            
            with torch.no_grad():
                if task_type == LanguageTaskType.TEXT_GENERATION:
                    # Text generation
                    gen_config = generation_config or GenerationConfig()
                    
                    output = model.generate(
                        inputs if isinstance(inputs, torch.Tensor) else inputs['input_ids'],
                        max_new_tokens=gen_config.max_tokens,
                        temperature=gen_config.temperature,
                        top_p=gen_config.top_p,
                        top_k=gen_config.top_k,
                        do_sample=gen_config.temperature > 0,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                elif task_type in [
                    LanguageTaskType.TEXT_CLASSIFICATION,
                    LanguageTaskType.SENTIMENT_ANALYSIS,
                    LanguageTaskType.NAMED_ENTITY_RECOGNITION
                ]:
                    # Classification/NER tasks
                    if isinstance(inputs, dict):
                        output = model(**inputs)
                    else:
                        output = model(inputs)
                
                elif task_type == LanguageTaskType.TEXT_EMBEDDINGS:
                    # Embeddings extraction
                    if isinstance(inputs, dict):
                        output = model(**inputs)
                    else:
                        output = model(inputs)
                    
                    # Get embeddings from last hidden state
                    if hasattr(output, 'last_hidden_state'):
                        # Mean pooling
                        embeddings = output.last_hidden_state.mean(dim=1)
                        output = embeddings
                
                else:
                    # Generic forward pass
                    if isinstance(inputs, dict):
                        output = model(**inputs)
                    else:
                        output = model(inputs)
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error during text inference: {str(e)}")
            raise
    
    async def _postprocess_text_results(
        self,
        raw_output: Any,
        input_text: str,
        task_type: LanguageTaskType,
        model_id: str,
        **kwargs
    ) -> LanguagePrediction:
        """Post-process raw model output into structured prediction."""
        try:
            import torch
            import torch.nn.functional as F
            
            tokenizer = self._tokenizers[model_id]
            
            prediction = LanguagePrediction(
                task_type=task_type,
                input_text=input_text,
                confidence=0.0,
                processing_time_ms=0.0
            )
            
            if task_type == LanguageTaskType.TEXT_GENERATION:
                # Text generation post-processing
                if isinstance(raw_output, torch.Tensor):
                    generated_text = tokenizer.decode(
                        raw_output[0],
                        skip_special_tokens=True
                    )
                    
                    # Remove input text from generated output
                    if generated_text.startswith(input_text):
                        generated_text = generated_text[len(input_text):].strip()
                    
                    prediction.generated_text = generated_text
                    prediction.confidence = 1.0  # Placeholder
            
            elif task_type in [LanguageTaskType.TEXT_CLASSIFICATION, LanguageTaskType.SENTIMENT_ANALYSIS]:
                # Classification post-processing
                if hasattr(raw_output, 'logits'):
                    logits = raw_output.logits
                    probabilities = F.softmax(logits, dim=-1)
                    probs_np = probabilities.cpu().numpy()[0]
                    
                    # Get class names if available
                    metadata = self._model_metadata.get(model_id)
                    if metadata and hasattr(metadata, 'output_classes'):
                        class_names = metadata.output_classes
                    else:
                        class_names = [f"class_{i}" for i in range(len(probs_np))]
                    
                    # Create classification results
                    classification_results = {
                        class_names[i]: float(probs_np[i])
                        for i in range(min(len(probs_np), len(class_names)))
                    }
                    
                    prediction.classification_results = classification_results
                    prediction.confidence = float(np.max(probs_np))
                    
                    # For sentiment analysis, add specific fields
                    if task_type == LanguageTaskType.SENTIMENT_ANALYSIS:
                        # Assume binary sentiment (positive/negative) or three-class (pos/neg/neutral)
                        if len(probs_np) == 2:
                            prediction.sentiment_score = float(probs_np[1] - probs_np[0])  # positive - negative
                            prediction.sentiment_label = "positive" if probs_np[1] > probs_np[0] else "negative"
                        elif len(probs_np) == 3:
                            labels = ["negative", "neutral", "positive"]
                            max_idx = np.argmax(probs_np)
                            prediction.sentiment_label = labels[max_idx]
                            prediction.sentiment_score = float(probs_np[2] - probs_np[0])  # positive - negative
            
            elif task_type == LanguageTaskType.NAMED_ENTITY_RECOGNITION:
                # NER post-processing
                if hasattr(raw_output, 'logits'):
                    logits = raw_output.logits
                    predictions = torch.argmax(logits, dim=-1)
                    
                    # Convert token predictions to entities
                    entities = self._extract_entities_from_tokens(
                        predictions[0].cpu().numpy(),
                        input_text,
                        tokenizer,
                        model_id
                    )
                    
                    prediction.entities = entities
                    prediction.confidence = np.mean([e.confidence for e in entities]) if entities else 0.0
            
            elif task_type == LanguageTaskType.TEXT_EMBEDDINGS:
                # Embeddings post-processing
                if isinstance(raw_output, torch.Tensor):
                    embeddings = raw_output.cpu().numpy()
                    prediction.embeddings = embeddings
                    prediction.confidence = 1.0
            
            elif task_type == LanguageTaskType.QUESTION_ANSWERING:
                # QA post-processing
                if hasattr(raw_output, 'start_logits') and hasattr(raw_output, 'end_logits'):
                    start_logits = raw_output.start_logits
                    end_logits = raw_output.end_logits
                    
                    # Find best span
                    start_idx = torch.argmax(start_logits)
                    end_idx = torch.argmax(end_logits)
                    
                    if end_idx >= start_idx:
                        # Extract answer from input
                        tokens = tokenizer.tokenize(input_text)
                        answer_tokens = tokens[start_idx:end_idx+1]
                        answer = tokenizer.convert_tokens_to_string(answer_tokens)
                        
                        prediction.answer = answer
                        prediction.confidence = float(
                            torch.softmax(start_logits, dim=-1)[start_idx] *
                            torch.softmax(end_logits, dim=-1)[end_idx]
                        )
            
            else:
                # Generic post-processing
                if isinstance(raw_output, torch.Tensor):
                    prediction.confidence = 1.0
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error post-processing text results: {str(e)}")
            return LanguagePrediction(
                task_type=task_type,
                input_text=input_text,
                confidence=0.0,
                processing_time_ms=0.0
            )
    
    def _extract_entities_from_tokens(
        self,
        token_predictions: np.ndarray,
        input_text: str,
        tokenizer: Any,
        model_id: str
    ) -> List[Entity]:
        """Extract entities from token-level predictions."""
        entities = []
        
        try:
            # Get label names if available
            metadata = self._model_metadata.get(model_id)
            if metadata and hasattr(metadata, 'output_classes'):
                label_names = metadata.output_classes
            else:
                # Default BIO tags for NER
                label_names = [
                    "O", "B-PER", "I-PER", "B-ORG", "I-ORG", 
                    "B-LOC", "I-LOC", "B-MISC", "I-MISC"
                ]
            
            tokens = tokenizer.tokenize(input_text)
            current_entity = None
            current_tokens = []
            
            for i, (token, pred_id) in enumerate(zip(tokens, token_predictions)):
                if pred_id < len(label_names):
                    label = label_names[pred_id]
                    
                    if label.startswith("B-"):
                        # Begin new entity
                        if current_entity:
                            # Finish previous entity
                            entity_text = tokenizer.convert_tokens_to_string(current_tokens)
                            entities.append(current_entity)
                        
                        entity_type = label[2:]  # Remove "B-" prefix
                        current_entity = Entity(
                            span=TextSpan(
                                start=i,
                                end=i+1,
                                text=token,
                                label=entity_type,
                                confidence=0.9  # Placeholder
                            ),
                            entity_type=entity_type
                        )
                        current_tokens = [token]
                    
                    elif label.startswith("I-") and current_entity:
                        # Continue current entity
                        current_tokens.append(token)
                        current_entity.span.end = i + 1
                        current_entity.span.text = tokenizer.convert_tokens_to_string(current_tokens)
                    
                    else:
                        # Outside entity or other tag
                        if current_entity:
                            entities.append(current_entity)
                            current_entity = None
                            current_tokens = []
            
            # Finish last entity if exists
            if current_entity:
                entities.append(current_entity)
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
        
        return entities
    
    async def predict_batch(
        self,
        model_id: str,
        input_texts: List[str],
        task_type: LanguageTaskType,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[LanguagePrediction]:
        """Perform batch prediction on multiple texts."""
        start_time = time.time()
        
        try:
            if model_id not in self._models:
                raise ValueError(f"Model {model_id} not loaded")
            
            predictions = []
            batch_size = kwargs.get('batch_size', 8)
            
            # Process in batches
            for i in range(0, len(input_texts), batch_size):
                batch_texts = input_texts[i:i + batch_size]
                
                # Process each text in the batch
                batch_predictions = await asyncio.gather(*[
                    self.predict_text(
                        model_id, text, task_type, generation_config, **kwargs
                    )
                    for text in batch_texts
                ])
                
                predictions.extend(batch_predictions)
            
            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(input_texts)
            
            # Update timing for all predictions
            for pred in predictions:
                pred.processing_time_ms = avg_time
            
            self.logger.info(
                f"Batch prediction completed: {len(input_texts)} texts in {total_time:.2f}ms"
            )
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error during batch prediction: {str(e)}")
            return [
                LanguagePrediction(
                    task_type=task_type,
                    input_text=text,
                    confidence=0.0,
                    processing_time_ms=0.0
                )
                for text in input_texts
            ]
    
    # Conversation Management
    
    def create_conversation(
        self,
        conversation_id: str,
        system_prompt: Optional[str] = None,
        context_window: int = 4096,
        user_id: Optional[str] = None
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            conversation_id=conversation_id,
            system_prompt=system_prompt,
            context_window=context_window,
            user_id=user_id
        )
        
        self._conversations[conversation_id] = conversation
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get an existing conversation."""
        return self._conversations.get(conversation_id)
    
    def add_message_to_conversation(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a message to a conversation."""
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return False
        
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        conversation.add_message(message)
        return True
    
    async def continue_conversation(
        self,
        conversation_id: str,
        model_id: str,
        user_message: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> Optional[str]:
        """Continue a conversation with a new user message."""
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return None
        
        # Add user message
        self.add_message_to_conversation(conversation_id, "user", user_message)
        
        # Get conversation context
        context_messages = conversation.get_context()
        
        # Build prompt from context
        prompt_parts = []
        for msg in context_messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"Human: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt_parts.append("Assistant:")
        full_prompt = "\n\n".join(prompt_parts)
        
        # Generate response
        prediction = await self.predict_text(
            model_id,
            full_prompt,
            LanguageTaskType.TEXT_GENERATION,
            generation_config
        )
        
        if prediction.generated_text:
            # Add assistant response to conversation
            self.add_message_to_conversation(
                conversation_id, "assistant", prediction.generated_text
            )
            return prediction.generated_text
        
        return None
    
    # Prompt Template Management
    
    def add_prompt_template(self, template: PromptTemplate):
        """Add a prompt template."""
        self._prompt_templates[template.template_id] = template
    
    def get_prompt_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a prompt template."""
        return self._prompt_templates.get(template_id)
    
    def format_prompt_template(
        self,
        template_id: str,
        **variables
    ) -> Optional[str]:
        """Format a prompt template with variables."""
        template = self._prompt_templates.get(template_id)
        if not template:
            return None
        
        try:
            return template.format(**variables)
        except ValueError as e:
            self.logger.error(f"Error formatting template {template_id}: {str(e)}")
            return None
    
    # Model Management
    
    def get_model_info(self, model_id: str) -> Optional[LanguageModelMetadata]:
        """Get metadata for a loaded model."""
        return self._model_metadata.get(model_id)
    
    def get_performance_stats(self, model_id: str) -> Dict[str, float]:
        """Get performance statistics for a model."""
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
        """List all loaded model IDs."""
        return list(self._models.keys())
    
    def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        return list(self._conversations.keys())
    
    def list_prompt_templates(self) -> List[str]:
        """List all prompt template IDs."""
        return list(self._prompt_templates.keys())
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        try:
            if model_id in self._models:
                del self._models[model_id]
            
            if model_id in self._tokenizers:
                del self._tokenizers[model_id]
                
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
        """Perform health check on the NLP service."""
        try:
            import torch
            
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "device": self.device,
                "models_loaded": len(self._models),
                "conversations_active": len(self._conversations),
                "prompt_templates": len(self._prompt_templates),
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