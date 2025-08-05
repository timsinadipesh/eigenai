"""
Local ONNX Gemma 3n E2B inference engine with streaming support and proper chat template handling
FIXED: Better error handling and file processing
"""

import os
import gc
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
import numpy as np
import onnxruntime as ort
from fastapi import HTTPException
from transformers import AutoProcessor, AutoConfig

logger = logging.getLogger(__name__)


class LocalONNXGemmaEngine:
    """Local-only ONNX Gemma 3n E2B inference engine with streaming support and proper chat template handling"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        # Local cache directory (where download.py puts files)
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "onnx_models" / "gemma-3n-e2b"
        
        # Model components (will be loaded from local files)
        self.processor = None
        self.config = None
        self.embed_session = None
        self.audio_session = None
        self.vision_session = None
        self.decoder_session = None
        
        # Model state
        self.is_loaded = False
        
        # Configuration values (loaded from local config.json)
        self.num_key_value_heads = None
        self.head_dim = None
        self.num_hidden_layers = None
        self.eos_token_id = 106  # Gemma 3n specific
        self.image_token_id = None
        self.audio_token_id = None
        
        # ONNX Runtime settings optimized for CPU
        self.ort_providers = ['CPUExecutionProvider']
        self.ort_session_options = ort.SessionOptions()
        self.ort_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session_options.intra_op_num_threads = 0  # Use all available cores
    
    def _check_local_files(self) -> bool:
        """Check if all required local files exist"""
        
        # Check if cache directory exists
        if not self.cache_dir.exists():
            logger.error(f"Cache directory not found: {self.cache_dir}")
            return False
        
        # Required config files
        config_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "preprocessor_config.json"
        ]
        
        for file in config_files:
            if not (self.cache_dir / file).exists():
                logger.error(f"Missing config file: {file}")
                return False
        
        # Required ONNX model files (specific quantization from download.py)
        onnx_dir = self.cache_dir / "onnx"
        onnx_files = [
            "embed_tokens_quantized.onnx",      # 2.7 GB
            "audio_encoder.onnx",               # 1.4 GB (fp16)
            "vision_encoder_quantized.onnx",    # 0.3 GB  
            "decoder_model_merged_q4.onnx"      # 1.4 GB (q4f16)
        ]
        
        for file in onnx_files:
            file_path = onnx_dir / file
            if file_path.exists():
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"✓ Found {file} ({size_mb:.1f} MB)")
            else:
                logger.error(f"Missing ONNX model: {file}")
                return False
        
        return True
    
    def _load_local_config(self):
        """Load configuration from local files"""
        
        # Load main config
        config_path = self.cache_dir / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Extract text config values
        text_config = config_dict.get("text_config", {})
        self.num_key_value_heads = text_config.get("num_key_value_heads")
        self.head_dim = text_config.get("head_dim") 
        self.num_hidden_layers = text_config.get("num_hidden_layers")
        
        # Extract multimodal token IDs
        self.image_token_id = config_dict.get("image_token_id")
        self.audio_token_id = config_dict.get("audio_token_id")
        
        logger.info(f"Config loaded - layers: {self.num_hidden_layers}, kv_heads: {self.num_key_value_heads}")
        logger.info(f"Special tokens - image: {self.image_token_id}, audio: {self.audio_token_id}")
        
        return config_dict
    
    async def load_models(self):
        """Load all model components from local files only"""
        
        if self.is_loaded:
            logger.info("Models already loaded")
            return
        
        logger.info("="*60)
        logger.info("LOADING LOCAL ONNX GEMMA 3N E2B MODELS")
        logger.info("="*60)
        
        # Step 1: Check all files exist
        logger.info("🔍 Step 1: Checking local files...")
        if not self._check_local_files():
            raise HTTPException(
                status_code=500, 
                detail="Local model files not found. Run download.py first!"
            )
        logger.info("✓ All required files found locally")
        
        try:
            # Step 2: Load configuration
            logger.info("🔧 Step 2: Loading configuration...")
            config_dict = self._load_local_config()
            
            # Load processor from local files
            self.processor = AutoProcessor.from_pretrained(
                str(self.cache_dir),  # Load from local directory, not model ID
                local_files_only=True  # Force local-only loading
            )
            
            # Load config from local files  
            self.config = AutoConfig.from_pretrained(
                str(self.cache_dir),  # Load from local directory, not model ID
                local_files_only=True  # Force local-only loading
            )
            
            logger.info("✓ Processor and config loaded from local files")
            
            # Step 3: Load ONNX sessions
            logger.info("🤖 Step 3: Loading ONNX models...")
            onnx_dir = self.cache_dir / "onnx"
            
            # Load embed tokens (quantized, 2.7 GB)
            logger.info("   📥 Loading embed_tokens (quantized)...")
            self.embed_session = ort.InferenceSession(
                str(onnx_dir / "embed_tokens_quantized.onnx"),
                sess_options=self.ort_session_options,
                providers=self.ort_providers
            )
            
            # Load audio encoder (fp16, 1.4 GB)  
            logger.info("   📥 Loading audio_encoder (fp16)...")
            self.audio_session = ort.InferenceSession(
                str(onnx_dir / "audio_encoder.onnx"),
                sess_options=self.ort_session_options,
                providers=self.ort_providers
            )
            
            # Load vision encoder (quantized, 0.3 GB)
            logger.info("   📥 Loading vision_encoder (quantized)...")
            self.vision_session = ort.InferenceSession(
                str(onnx_dir / "vision_encoder_quantized.onnx"),
                sess_options=self.ort_session_options,
                providers=self.ort_providers
            )
            
            # Load decoder (q4f16, 1.4 GB)
            logger.info("   📥 Loading decoder (q4f16)...")
            self.decoder_session = ort.InferenceSession(
                str(onnx_dir / "decoder_model_merged_q4.onnx"),
                sess_options=self.ort_session_options,
                providers=self.ort_providers
            )
            
            self.is_loaded = True
            
            logger.info("="*60)
            logger.info("✅ ALL MODELS LOADED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info("🚀 Ready for inference with streaming support!")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.cleanup()
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")
    
    def cleanup(self):
        """Clean up loaded models"""
        logger.info("🧹 Cleaning up models...")
        
        sessions = [
            ('embed_session', self.embed_session),
            ('audio_session', self.audio_session), 
            ('vision_session', self.vision_session),
            ('decoder_session', self.decoder_session)
        ]
        
        for name, session in sessions:
            if session:
                del session
                setattr(self, name, None)
        
        self.is_loaded = False
        gc.collect()
        logger.info("✓ Cleanup complete")

    async def generate_response_stream(
        self, 
        messages: List[Dict], 
        max_new_tokens: int = 1000
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Advanced streaming with incremental text decoding using proper Gemma 3n chat template.
        Yields new text chunks as soon as they are available.
        FIXED: Better error handling and file processing
        """
        if not self.is_loaded:
            yield {'error': 'Models not loaded', 'finished': True}
            return

        max_new_tokens = min(max_new_tokens, 2048)
        logger.info(f"🎯 Starting advanced streaming generation (max_tokens: {max_new_tokens})")

        try:
            # CRITICAL: Normalize message format for Gemma 3n
            # Gemma 3n expects content to ALWAYS be a list of dictionaries
            normalized_messages = []
            for msg in messages:
                normalized_msg = {"role": msg["role"]}
                
                # Handle content formatting
                content = msg.get("content", [])
                if isinstance(content, str):
                    # Convert string content to proper list format
                    normalized_msg["content"] = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    # Content is already in list format, use as-is
                    normalized_msg["content"] = content
                else:
                    # Fallback for unexpected format
                    normalized_msg["content"] = [{"type": "text", "text": str(content)}]
                
                normalized_messages.append(normalized_msg)
            
            logger.info(f"📝 Processing normalized messages: {len(normalized_messages)} message(s)")
            for i, msg in enumerate(normalized_messages):
                content_types = [c.get('type', 'unknown') for c in msg.get('content', [])]
                logger.info(f"   Message {i}: role={msg.get('role')}, content_types={content_types}")
            
            # Use processor.apply_chat_template with normalized message structure
            # IMPORTANT: Must use return_tensors="pt" first for proper processing!
            try:
                inputs = self.processor.apply_chat_template(
                    normalized_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"  # PyTorch tensors required!
                )
            except Exception as template_error:
                logger.error(f"Chat template processing failed: {template_error}")
                yield {'error': f'Chat template processing failed: {template_error}', 'finished': True}
                return
            
            # Convert to numpy for ONNX
            input_ids = inputs["input_ids"].numpy()
            attention_mask = inputs["attention_mask"].numpy()
            position_ids = np.cumsum(attention_mask, axis=-1) - 1
            
            # Extract multimodal inputs if present
            pixel_values = inputs["pixel_values"].numpy() if "pixel_values" in inputs else None
            input_features = inputs["input_features"].numpy().astype(np.float32) if "input_features" in inputs else None
            input_features_mask = inputs["input_features_mask"].numpy() if "input_features_mask" in inputs else None
            
            logger.info(f"Input processed: {input_ids.shape}")
            if pixel_values is not None:
                logger.info(f"Image input: {pixel_values.shape}")
            if input_features is not None:
                logger.info(f"Audio input: {input_features.shape}")

            batch_size = input_ids.shape[0]
            past_key_values = {
                f"past_key_values.{layer}.{kv}": np.zeros(
                    [batch_size, self.num_key_value_heads, 0, self.head_dim], 
                    dtype=np.float32
                )
                for layer in range(self.num_hidden_layers)
                for kv in ("key", "value")
            }

            all_generated_tokens = []
            decoded_text = ""
            image_features = None
            audio_features = None

            for step in range(max_new_tokens):
                try:
                    # Get embeddings
                    inputs_embeds, per_layer_inputs = self.embed_session.run(
                        None, {"input_ids": input_ids}
                    )
                    
                    # Process image features (once)
                    if image_features is None and pixel_values is not None:
                        try:
                            image_features = self.vision_session.run(
                                ["image_features"],
                                {"pixel_values": pixel_values}
                            )[0]
                            
                            # Replace image tokens with image features
                            mask = (input_ids == self.image_token_id).reshape(-1)
                            if mask.any():
                                flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                                flat_embeds[mask] = image_features.reshape(-1, image_features.shape[-1])
                                inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)
                        except Exception as vision_error:
                            logger.error(f"Vision processing error: {vision_error}")
                            # Continue without image features
                    
                    # Process audio features (once)
                    if audio_features is None and input_features is not None and input_features_mask is not None:
                        try:
                            audio_features = self.audio_session.run(
                                ["audio_features"],
                                {
                                    "input_features": input_features,
                                    "input_features_mask": input_features_mask
                                }
                            )[0]
                            
                            # Replace audio tokens with audio features
                            mask = (input_ids == self.audio_token_id).reshape(-1)
                            if mask.any():
                                flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                                flat_embeds[mask] = audio_features.reshape(-1, audio_features.shape[-1])
                                inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)
                        except Exception as audio_error:
                            logger.error(f"Audio processing error: {audio_error}")
                            # Continue without audio features

                    # Run decoder
                    outputs = self.decoder_session.run(None, {
                        "inputs_embeds": inputs_embeds,
                        "per_layer_inputs": per_layer_inputs,
                        "position_ids": position_ids,
                        **past_key_values
                    })
                    logits = outputs[0]
                    present_key_values = outputs[1:]

                    # Greedy sampling
                    next_token = np.argmax(logits[:, -1], axis=-1, keepdims=True)
                    token_id = next_token[0, 0].item()
                    all_generated_tokens.append(token_id)

                    # Incremental decoding with better error handling
                    try:
                        new_decoded_text = self.processor.tokenizer.decode(
                            all_generated_tokens, 
                            skip_special_tokens=True
                        )
                        # Only yield the new part
                        if len(new_decoded_text) > len(decoded_text):
                            new_part = new_decoded_text[len(decoded_text):]
                            decoded_text = new_decoded_text
                            if new_part:
                                yield {
                                    'token': new_part,
                                    'finished': False,
                                    'total_tokens': len(all_generated_tokens),
                                    'step': step
                                }
                    except Exception as decode_error:
                        logger.warning(f"Decode error at step {step}: {decode_error}")
                        # Continue without yielding this token

                    await asyncio.sleep(0.001)  # Optional: for UI smoothness

                    # Prepare for next step
                    input_ids = next_token
                    attention_mask = np.ones_like(input_ids)
                    position_ids = position_ids[:, -1:] + 1
                    for i, key in enumerate(past_key_values.keys()):
                        past_key_values[key] = present_key_values[i]

                    # EOS check
                    if token_id == self.eos_token_id:
                        logger.info(f"🏁 EOS reached at step {step}")
                        break

                except Exception as step_error:
                    logger.error(f"Error at generation step {step}: {step_error}")
                    yield {
                        'error': f"Generation step {step} failed: {step_error}",
                        'finished': True
                    }
                    return

            # Final signal
            yield {
                'token': '',
                'finished': True,
                'total_tokens': len(all_generated_tokens),
                'step': step
            }
            logger.info(f"✅ Advanced streaming complete ({len(all_generated_tokens)} tokens)")

        except Exception as e:
            logger.error(f"💥 Advanced streaming failed: {e}")
            yield {
                'error': f"Generation failed: {e}",
                'finished': True
            }
        
    async def generate_response(self, messages: List[Dict], max_new_tokens: int = 1000) -> str:
        """
        Non-streaming generation for backward compatibility
        Collects all tokens from the streaming generator
        """
        
        full_response = ""
        async for chunk in self.generate_response_stream(messages, max_new_tokens):
            if chunk.get('error'):
                raise HTTPException(status_code=500, detail=chunk['error'])
            if not chunk['finished']:
                full_response += chunk['token']
        
        return full_response.strip()