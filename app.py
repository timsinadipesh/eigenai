"""
Enhanced ONNX Gemma 3n E2B with Token Streaming Support
This implementation adds real-time token streaming using Server-Sent Events (SSE)
"""

import os
import re
import gc
import json
import logging
import tempfile
import subprocess
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from transformers import AutoProcessor, AutoConfig
import uvicorn
from PIL import Image
from contextlib import asynccontextmanager
import librosa  # Add this import for audio processing
import soundfile as sf
from langdetect import detect, LangDetectException, DetectorFactory

# For Server-Sent Events (SSE)
from sse_starlette.sse import EventSourceResponse

# Set seed for consistent language detection results
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import os
import re
import hashlib
import subprocess
import logging
from pathlib import Path
from typing import Optional
from langdetect import detect, LangDetectException, DetectorFactory

# Set seed for consistent language detection results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


class PiperTTSEngine:
    """Piper TTS integration for text-to-speech with robust language detection"""
    
    def __init__(self):
        # Find the voices directory relative to this file
        self.voices_dir = Path(__file__).parent / "voices"
        self.cache_dir = Path.home() / ".cache" / "piper_audio"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Dynamically find all .onnx voice models in the voices directory
        self.voices = {}
        if self.voices_dir.exists():
            for onnx_file in self.voices_dir.glob("*.onnx"):
                # Example: en_US-lessac-high.onnx → "en_US" and "en"
                lang_full = onnx_file.name.split("-")[0]  # "en_US"
                lang_short = lang_full.split("_")[0]      # "en"
                self.voices[lang_full] = onnx_file
                self.voices[lang_short] = onnx_file
                logger.info(f"Found voice model: {lang_short} -> {onnx_file.name}")

        # Check if piper is available
        self.piper_available = self._check_piper_installation()
        
        # Pre-compile regex for text cleaning
        self._text_cleaner = re.compile(r'[^\w\s\.\,\!\?\-\']')
        
        # Language detection cache for performance
        self._detection_cache = {}
        self._cache_max_size = 100
        
        # Language-specific keywords for short text detection
        self.lang_keywords = {
            'fr': {
                'articles': ['le', 'la', 'les', 'un', 'une', 'des', 'du'],
                'common': ['et', 'est', 'avec', 'pour', 'dans', 'sur', 'que', 'qui', 'mais', 'vous', 'nous', 'ils'],
                'pronouns': ['je', 'tu', 'il', 'elle', 'on'],
            },
            'de': {
                'articles': ['der', 'die', 'das', 'ein', 'eine', 'den', 'dem'],
                'common': ['und', 'ist', 'mit', 'für', 'auf', 'von', 'zu', 'nicht', 'ich', 'sie', 'wir'],
                'pronouns': ['ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr'],
            }
        }
        
    def _check_piper_installation(self) -> bool:
        """Check if piper is installed and available"""
        try:
            result = subprocess.run(["piper", "--help"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Piper TTS not found. TTS functionality disabled.")
            return False
    
    def _detect_language_short_text(self, text: str) -> Optional[str]:
        """Detect language for short texts using keyword analysis"""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Score each language based on keyword matches
        scores = {}
        
        for lang, keywords in self.lang_keywords.items():
            score = 0
            # Check articles (strong indicators)
            for article in keywords['articles']:
                if article in words:
                    score += 3
            # Check common words
            for common in keywords['common']:
                if common in words:
                    score += 2
            # Check pronouns
            for pronoun in keywords['pronouns']:
                if pronoun in words:
                    score += 2
            
            if score > 0:
                scores[lang] = score
        
        # Return language with highest score if confident enough
        if scores:
            best_lang = max(scores, key=scores.get)
            if scores[best_lang] >= 4:  # Confidence threshold
                return best_lang
        
        return None
    
    def _detect_language(self, text: str) -> str:
        """
        Robust language detection with fallback strategies.
        Returns 'en', 'fr', or 'de' based on available voices.
        """
        try:
            # Check cache first
            text_sample = text[:200]  # Use first 200 chars for cache key
            cache_key = hash(text_sample)
            if cache_key in self._detection_cache:
                return self._detection_cache[cache_key]
            
            # Clean text for detection
            clean_text = self._text_cleaner.sub(' ', text).strip()
            
            # Strategy 1: Short text detection using keywords
            if len(clean_text) < 30:
                detected = self._detect_language_short_text(clean_text)
                if detected and detected in self.voices:
                    logger.info(f"Short text detection: {detected}")
                    self._detection_cache[cache_key] = detected
                    return detected
                # For very short texts without clear indicators, default to English
                if len(clean_text) < 15:
                    logger.info("Very short text, defaulting to English")
                    return 'en'
            
            # Strategy 2: Use langdetect for longer texts
            try:
                detected = detect(clean_text)
                logger.info(f"Langdetect result: {detected}")
                
                # Map detected language to available voices
                if detected in ['en', 'fr', 'de']:
                    if detected in self.voices and self.voices[detected].exists():
                        # Cache the result
                        if len(self._detection_cache) < self._cache_max_size:
                            self._detection_cache[cache_key] = detected
                        return detected
                    else:
                        logger.warning(f"Voice for {detected} not available")
                
                # Handle other language codes that might map to our supported languages
                lang_mapping = {
                    'english': 'en',
                    'french': 'fr', 
                    'german': 'de',
                    'deutsch': 'de',
                    'francais': 'fr',
                    'français': 'fr'
                }
                
                if detected.lower() in lang_mapping:
                    mapped_lang = lang_mapping[detected.lower()]
                    if mapped_lang in self.voices:
                        self._detection_cache[cache_key] = mapped_lang
                        return mapped_lang
                
            except LangDetectException as e:
                logger.warning(f"Langdetect failed: {e}")
            
            # Strategy 3: Fallback keyword detection for any length
            if len(clean_text) >= 30:
                detected = self._detect_language_short_text(clean_text)
                if detected and detected in self.voices:
                    logger.info(f"Fallback keyword detection: {detected}")
                    self._detection_cache[cache_key] = detected
                    return detected
            
            # Default to English
            logger.info("All detection strategies failed, defaulting to English")
            return 'en'
            
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return 'en'
    
    def _get_cache_path(self, text: str, language: str) -> Path:
        """Generate cache file path based on text hash and language"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"{text_hash}_{language}.wav"
    
    async def synthesize(self, text: str, language: Optional[str] = None) -> Optional[Path]:
        """
        Synthesize text to speech using Piper.
        
        Args:
            text: The text to synthesize
            language: Optional language code ('en', 'fr', 'de'). If None, auto-detects.
        
        Returns:
            Path to the synthesized audio file, or None if synthesis failed
        """
        
        if not self.piper_available:
            logger.error("Piper TTS is not available")
            return None
        
        # Handle language parameter
        if language is not None:
            # Validate provided language
            if language not in self.voices:
                logger.warning(f"Requested language '{language}' not available, auto-detecting instead")
                language = self._detect_language(text)
            elif not self.voices[language].exists():
                logger.warning(f"Voice model for '{language}' not found, auto-detecting instead")
                language = self._detect_language(text)
            else:
                logger.info(f"Using requested language: {language}")
        else:
            # Auto-detect language
            language = self._detect_language(text)
            logger.info(f"Auto-detected language: {language}")
        
        # Final validation
        if language not in self.voices or not self.voices[language].exists():
            logger.warning(f"Voice model for {language} not found, falling back to English")
            language = "en"
            if language not in self.voices or not self.voices[language].exists():
                logger.error("No voice models available")
                return None
        
        # Check cache first
        cache_path = self._get_cache_path(text, language)
        if cache_path.exists():
            logger.info(f"Using cached audio: {cache_path}")
            return cache_path
        
        try:
            # Clean text for TTS
            clean_text = self._clean_text_for_tts(text)
            
            # Run Piper TTS
            cmd = [
                "piper",
                "--model", str(self.voices[language]),
                "--output_file", str(cache_path)
            ]
            
            logger.info(f"Running Piper TTS with model: {self.voices[language].name}")
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=clean_text, timeout=30)
            
            if process.returncode == 0 and cache_path.exists():
                file_size = cache_path.stat().st_size
                logger.info(f"TTS synthesis complete: {cache_path} ({file_size} bytes)")
                return cache_path
            else:
                logger.error(f"Piper TTS failed with return code {process.returncode}")
                if stderr:
                    logger.error(f"Piper stderr: {stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("Piper TTS timed out after 30 seconds")
            return None
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return None
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for better TTS synthesis - no length limits"""
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic  
        text = re.sub(r'#{1,6}\s', '', text)          # Headers
        
        # Remove code blocks
        text = re.sub(r'```[^`]*```', ' ', text)
        
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Replace smart quotes and other special characters
        replacements = {
            '"': '"', '"': '"', ''': "'", ''': "'",
            '–': '-', '—': '-', '…': '...'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def clear_cache(self, language: Optional[str] = None):
        """Clear TTS cache, optionally for a specific language"""
        try:
            if language:
                # Clear only files for specific language
                pattern = f"*_{language}.wav"
                files = list(self.cache_dir.glob(pattern))
                for file in files:
                    file.unlink()
                logger.info(f"Cleared {len(files)} cached files for language: {language}")
            else:
                # Clear all cache
                files = list(self.cache_dir.glob("*.wav"))
                for file in files:
                    file.unlink()
                logger.info(f"Cleared {len(files)} cached files")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


class LocalONNXGemmaEngine:
    """Local-only ONNX Gemma 3n E2B inference engine with streaming support"""
    
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
        Advanced streaming with incremental text decoding.
        Yields new text chunks as soon as they are available.
        """
        if not self.is_loaded:
            yield {'error': 'Models not loaded', 'finished': True}
            return

        max_new_tokens = min(max_new_tokens, 2048)
        logger.info(f"🎯 Starting advanced streaming generation (max_tokens: {max_new_tokens})")

        try:
            # No need to process audio here - just pass messages as-is to processor
            # The processor will handle audio file loading internally
            
            # IMPORTANT: Must use return_tensors="pt" first!
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"  # PyTorch tensors required!
            )
            
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
                # Get embeddings
                inputs_embeds, per_layer_inputs = self.embed_session.run(
                    None, {"input_ids": input_ids}
                )
                
                # Process image features (once)
                if image_features is None and pixel_values is not None:
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
                
                # Process audio features (once)
                if audio_features is None and input_features is not None and input_features_mask is not None:
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

                # Incremental decoding
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
                    logger.warning(f"Advanced decode error at step {step}: {decode_error}")

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


# Initialize the inference engine and TTS
inference_engine = LocalONNXGemmaEngine()
tts_engine = PiperTTSEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    try:
        await inference_engine.load_models()
        logger.info("🎉 Application startup complete!")
        yield
    finally:
        inference_engine.cleanup()
        logger.info("👋 Application shutdown complete")


# FastAPI application
app = FastAPI(
    title="Local ONNX Gemma 3n E2B with Streaming",
    description="Multimodal AI inference with real-time token streaming using locally cached ONNX models",
    version="2.1.0",
    lifespan=lifespan
)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if inference_engine.is_loaded else "loading",
        "models_loaded": inference_engine.is_loaded,
        "tts_available": tts_engine.piper_available,
        "streaming_supported": True,
        "cache_location": str(inference_engine.cache_dir),
        "quantization": {
            "embed_tokens": "quantized",
            "audio_encoder": "fp16", 
            "vision_encoder": "quantized",
            "decoder": "q4f16"
        }
    }


@app.get("/chat")
async def chat_interface():
    """Serve the chat interface"""
    return FileResponse("static/chat.html")


@app.post("/api/generate")
async def generate_response(
    text: str = Form(...),
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
    max_tokens: int = Form(1000)
):
    """Generate response from text, image, and/or audio inputs (non-streaming)"""
    
    # Limit max tokens
    max_tokens = min(max_tokens, 2048)
    
    content = [{"type": "text", "text": text}]
    temp_files = []
    
    try:
        # Handle image upload
        if image and image.filename:
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Invalid image format")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                content_bytes = await image.read()
                tmp_file.write(content_bytes)
                temp_files.append(tmp_file.name)
            
            img = Image.open(temp_files[-1])
            if img.mode != 'RGB':
                img = img.convert('RGB')
            content.append({"type": "image", "image": img})
            logger.info(f"🖼️  Image processed: {img.size}")
        
        # Handle audio upload
        if audio and audio.filename:
            if not audio.content_type.startswith("audio/"):
                raise HTTPException(status_code=400, detail="Invalid audio format")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                content_bytes = await audio.read()
                tmp_file.write(content_bytes)
                temp_files.append(tmp_file.name)
            
            content.append({"type": "audio", "audio": temp_files[-1]})
            logger.info(f"🎵 Audio processed: {temp_files[-1]}")
        
        # Generate response
        messages = [{"role": "user", "content": content}]
        response = await inference_engine.generate_response(messages, max_new_tokens=max_tokens)
        
        return {
            "response": response,
            "input_modalities": {
                "text": True,
                "image": image is not None and image.filename,
                "audio": audio is not None and audio.filename
            },
            "tokens_generated": len(response.split())  # Rough estimate
        }
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass


@app.post("/api/generate/stream")
async def generate_response_stream(
    text: str = Form(...),
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
    max_tokens: int = Form(1000)
):
    """
    Generate streaming response from text, image, and/or audio inputs
    Uses Server-Sent Events (SSE) for real-time token streaming
    """
    
    # Limit max tokens
    max_tokens = min(max_tokens, 2048)
    
    content = [{"type": "text", "text": text}]
    temp_files = []
    
    try:
        # Handle image upload
        if image and image.filename:
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Invalid image format")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                content_bytes = await image.read()
                tmp_file.write(content_bytes)
                temp_files.append(tmp_file.name)
            
            img = Image.open(temp_files[-1])
            if img.mode != 'RGB':
                img = img.convert('RGB')
            content.append({"type": "image", "image": img})
            logger.info(f"🖼️  Image processed: {img.size}")
        
        # Handle audio upload
        if audio and audio.filename:
            if not audio.content_type.startswith("audio/"):
                raise HTTPException(status_code=400, detail="Invalid audio format")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                content_bytes = await audio.read()
                tmp_file.write(content_bytes)
                temp_files.append(tmp_file.name)
            
            content.append({"type": "audio", "audio": temp_files[-1]})
            logger.info(f"🎵 Audio processed: {temp_files[-1]}")
        
        # Create the generator for streaming
        messages = [{"role": "user", "content": content}]
        
        async def event_generator():
            """Generator function for Server-Sent Events"""
            try:
                async for chunk in inference_engine.generate_response_stream(messages, max_new_tokens=max_tokens):
                    # Format the chunk as SSE data
                    yield {
                        "event": "token" if not chunk['finished'] else "done",
                        "data": json.dumps(chunk)
                    }
                    
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e), "finished": True})
                }
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except FileNotFoundError:
                        pass
        
        # Return Server-Sent Events response
        return EventSourceResponse(event_generator())
        
    except Exception as e:
        # Clean up temporary files if there's an early error
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass
        raise HTTPException(status_code=500, detail=f"Request processing failed: {e}")


@app.post("/api/tts")
async def text_to_speech(
    text: str = Form(...),
    language: str = Form(None)
):
    """Convert text to speech using Piper TTS"""
    
    if not tts_engine.piper_available:
        raise HTTPException(status_code=503, detail="TTS service not available")
    
    try:
        # Synthesize audio
        audio_path = await tts_engine.synthesize(text, language)
        
        if not audio_path or not audio_path.exists():
            raise HTTPException(status_code=500, detail="TTS synthesis failed")
        
        # Stream the audio file
        def audio_stream():
            with open(audio_path, 'rb') as audio_file:
                while chunk := audio_file.read(8192):
                    yield chunk
        
        return StreamingResponse(
            audio_stream(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=tts_audio.wav",
                "Accept-Ranges": "bytes"
            }
        )
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


@app.get("/")
async def root():
    """Serve the main index page"""
    return FileResponse("static/index.html")


def main():
    """Run the application"""
    print("="*70)
    print("🚀 LOCAL ONNX GEMMA 3N E2B MULTIMODAL SERVER WITH STREAMING")
    print("="*70)
    print("📁 Cache directory:", inference_engine.cache_dir)
    print("🔧 ONNX Runtime providers:", inference_engine.ort_providers)
    print("📊 Using CPU inference with all available cores")
    print("🎵 TTS Engine:", "Enabled" if tts_engine.piper_available else "Disabled")
    print("🌊 Token Streaming: Enabled via Server-Sent Events")
    print("="*70)
    print()
    print("Starting server...")
    print("- Health check: http://localhost:8000/health")
    print("- API endpoint (non-streaming): http://localhost:8000/api/generate")
    print("- API endpoint (streaming): http://localhost:8000/api/generate/stream")
    print("- TTS endpoint: http://localhost:8000/api/tts")
    print("- Interactive docs: http://localhost:8000/docs")
    print()
    
    # Run the FastAPI server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
