"""
ONNX Gemma 3n E2B Web Application - Working Implementation
Based on official Hugging Face ONNX community example
"""

import os
import gc
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, AutoConfig
import uvicorn
from PIL import Image
import torch
from contextlib import asynccontextmanager
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXGemmaEngine:
    """ONNX-optimized Gemma 3n E2B inference engine"""
    
    def __init__(self, model_cache_dir: Optional[str] = None):
        # Model configuration
        self.model_id = "google/gemma-3n-E2B-it"
        
        # Cache directory
        if model_cache_dir:
            self.cache_dir = Path(model_cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "onnx_models" / "gemma-3n-e2b"
        
        # Model components
        self.processor = None
        self.config = None
        self.embed_session = None
        self.audio_session = None
        self.vision_session = None
        self.decoder_session = None
        
        # Model state
        self.is_loaded = False
        
        # Configuration values (will be set after loading config)
        self.num_key_value_heads = None
        self.head_dim = None
        self.num_hidden_layers = None
        self.eos_token_id = 106  # Gemma 3n specific - NOT config.text_config.eos_token_id
        self.image_token_id = None
        self.audio_token_id = None
        
        # ONNX Runtime settings
        self.ort_providers = ['CPUExecutionProvider']
        self.ort_session_options = ort.SessionOptions()
        self.ort_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    def _get_model_paths(self) -> Dict[str, Path]:
        """Get paths to ONNX model files"""
        onnx_dir = self.cache_dir / "onnx"
        
        # Try different quantization options
        quantization_options = [
            {
                "embed": "embed_tokens_quantized.onnx",
                "audio": "audio_encoder.onnx",
                "vision": "vision_encoder.onnx", 
                "decoder": "decoder_model_merged_q4.onnx"
            },
            {
                "embed": "embed_tokens.onnx",
                "audio": "audio_encoder.onnx",
                "vision": "vision_encoder.onnx",
                "decoder": "decoder_model_merged.onnx"
            }
        ]
        
        for option in quantization_options:
            paths = {
                "embed": onnx_dir / option["embed"],
                "audio": onnx_dir / option["audio"], 
                "vision": onnx_dir / option["vision"],
                "decoder": onnx_dir / option["decoder"]
            }
            
            if all(path.exists() for path in paths.values()):
                logger.info(f"Using quantization: {option}")
                return paths
        
        raise FileNotFoundError(f"ONNX model files not found in {onnx_dir}")
    
    async def load_models(self):
        """Load all model components"""
        if self.is_loaded:
            return
        
        logger.info("Loading ONNX Gemma 3n E2B models...")
        
        try:
            # Load processor and config
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir.parent / "transformers"
            )
            
            self.config = AutoConfig.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir.parent / "transformers"
            )
            
            # Set configuration values
            self.num_key_value_heads = self.config.text_config.num_key_value_heads
            self.head_dim = self.config.text_config.head_dim
            self.num_hidden_layers = self.config.text_config.num_hidden_layers
            self.image_token_id = self.config.image_token_id
            self.audio_token_id = self.config.audio_token_id
            
            logger.info(f"Config loaded - layers: {self.num_hidden_layers}, kv_heads: {self.num_key_value_heads}")
            
            # Get model paths and load ONNX sessions
            model_paths = self._get_model_paths()
            
            self.embed_session = ort.InferenceSession(
                str(model_paths['embed']),
                sess_options=self.ort_session_options,
                providers=self.ort_providers
            )
            
            self.audio_session = ort.InferenceSession(
                str(model_paths['audio']),
                sess_options=self.ort_session_options,
                providers=self.ort_providers
            )
            
            self.vision_session = ort.InferenceSession(
                str(model_paths['vision']),
                sess_options=self.ort_session_options,
                providers=self.ort_providers
            )
            
            self.decoder_session = ort.InferenceSession(
                str(model_paths['decoder']),
                sess_options=self.ort_session_options,
                providers=self.ort_providers
            )
            
            self.is_loaded = True
            logger.info("✓ All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.cleanup()
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")
    
    def cleanup(self):
        """Clean up loaded models"""
        if self.embed_session:
            del self.embed_session
            self.embed_session = None
        if self.audio_session:
            del self.audio_session
            self.audio_session = None
        if self.vision_session:
            del self.vision_session 
            self.vision_session = None
        if self.decoder_session:
            del self.decoder_session
            self.decoder_session = None
        self.is_loaded = False
        gc.collect()
    
    async def generate_response(self, messages: List[Dict], max_new_tokens: int = 32768) -> str:
        """Generate response using ONNX models - based on official example"""
        if not self.is_loaded:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        logger.info(f"Starting generation with max_new_tokens={max_new_tokens}")
        
        try:
            # Process input with chat template - MUST use return_tensors="pt"
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
            
            # Extract multimodal inputs
            pixel_values = inputs["pixel_values"].numpy() if "pixel_values" in inputs else None
            input_features = inputs["input_features"].numpy().astype(np.float32) if "input_features" in inputs else None
            input_features_mask = inputs["input_features_mask"].numpy() if "input_features_mask" in inputs else None
            
            logger.info(f"Input processed: {input_ids.shape}")
            if pixel_values is not None:
                logger.info(f"Image input: {pixel_values.shape}")
            if input_features is not None:
                logger.info(f"Audio input: {input_features.shape}")
            
            # Initialize generation state
            batch_size = input_ids.shape[0]
            past_key_values = {
                f"past_key_values.{layer}.{kv}": np.zeros([batch_size, self.num_key_value_heads, 0, self.head_dim], dtype=np.float32)
                for layer in range(self.num_hidden_layers)
                for kv in ("key", "value")
            }
            
            generated_tokens = np.array([[]], dtype=np.int64)
            image_features = None
            audio_features = None
            
            # Generation loop - following official example exactly
            for i in range(max_new_tokens):
                # Get embeddings
                inputs_embeds, per_layer_inputs = self.embed_session.run(None, {"input_ids": input_ids})
                
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
                logits, *present_key_values = self.decoder_session.run(None, dict(
                    inputs_embeds=inputs_embeds,
                    per_layer_inputs=per_layer_inputs,
                    position_ids=position_ids,
                    **past_key_values,
                ))
                
                # Update values for next generation loop
                input_ids = logits[:, -1].argmax(-1, keepdims=True)
                attention_mask = np.ones_like(input_ids)
                position_ids = position_ids[:, -1:] + 1
                
                for j, key in enumerate(past_key_values):
                    past_key_values[key] = present_key_values[j]
                
                generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
                
                # Check for EOS
                if (input_ids == self.eos_token_id).all():
                    logger.info(f"EOS reached at step {i}")
                    break
                
                # Log progress
                if i % 100 == 0 and i > 0:
                    logger.info(f"Generated {i} tokens...")
            
            # Decode output
            response = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            logger.info(f"Generation completed: {len(generated_tokens[0])} tokens")
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

# Initialize engine
inference_engine = ONNXGemmaEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await inference_engine.load_models()
    logger.info("Application ready")
    yield
    inference_engine.cleanup()

# FastAPI app
app = FastAPI(
    title="ONNX Gemma 3n E2B Multimodal",
    description="Working ONNX inference implementation",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "models_loaded": inference_engine.is_loaded,
        "cache_dir": str(inference_engine.cache_dir)
    }

@app.post("/api/generate")
async def generate_response(
    text: str = Form(...),
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
    max_tokens: int = Form(32768)  # Fixed: was 32678
):
    """Single unified endpoint for any modality combination"""
    max_tokens = min(max_tokens, 32768)
    
    content = [{"type": "text", "text": text}]
    temp_files = []
    
    try:
        # Handle image if provided
        if image and image.filename:  # Check filename to avoid empty uploads
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
            logger.info(f"Image processed: {img.size}")
        
        # Handle audio if provided  
        if audio and audio.filename:  # Check filename to avoid empty uploads
            if not audio.content_type.startswith("audio/"):
                raise HTTPException(status_code=400, detail="Invalid audio format")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                content_bytes = await audio.read()
                tmp_file.write(content_bytes)
                temp_files.append(tmp_file.name)
            
            content.append({"type": "audio", "audio": temp_files[-1]})
            logger.info(f"Audio file processed: {temp_files[-1]}")
        
        messages = [{"role": "user", "content": content}]
        response = await inference_engine.generate_response(messages, max_new_tokens=max_tokens)
        
        return {
            "response": response,
            "modalities_used": {
                "text": True,
                "image": image is not None and image.filename,
                "audio": audio is not None and audio.filename
            }
        }
        
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass


# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

def main():
    """Run the application"""
    print("="*60)
    print("ONNX GEMMA 3N E2B MULTIMODAL - WORKING VERSION")
    print("="*60)
    print(f"Cache directory: {inference_engine.cache_dir}")
    print("Max output tokens: 32768")
    print("Multimodal support: Text, Image, Audio")
    print("-" * 60)
    print("Starting server...")
    print("Health: http://localhost:8000/health")
    print("="*60)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()