"""
ONNX Gemma 3n E2B Local Inference Application
Clean implementation that works with locally downloaded ONNX models only
"""

import os
import gc
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from transformers import AutoProcessor, AutoConfig
import uvicorn
from PIL import Image
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalONNXGemmaEngine:
    """Local-only ONNX Gemma 3n E2B inference engine"""
    
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
            if not file_path.exists():
                logger.error(f"Missing ONNX model: {file}")
                return False
            else:
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"✓ Found {file} ({size_mb:.1f} MB)")
        
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
            logger.info("🚀 Ready for inference!")
            
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
    
    async def generate_response(self, messages: List[Dict], max_new_tokens: int = 1000) -> str:
        """Generate response using local ONNX models"""
        
        if not self.is_loaded:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Limit max tokens to prevent excessive generation
        max_new_tokens = min(max_new_tokens, 2048)
        logger.info(f"🎯 Starting generation (max_tokens: {max_new_tokens})")
        
        try:
            # Process input with chat template
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"  # PyTorch tensors required
            )
            
            # Convert to numpy for ONNX
            input_ids = inputs["input_ids"].numpy()
            attention_mask = inputs["attention_mask"].numpy()
            position_ids = np.cumsum(attention_mask, axis=-1) - 1
            
            # Extract multimodal inputs if present
            pixel_values = inputs["pixel_values"].numpy() if "pixel_values" in inputs else None
            input_features = inputs["input_features"].numpy().astype(np.float32) if "input_features" in inputs else None
            input_features_mask = inputs["input_features_mask"].numpy() if "input_features_mask" in inputs else None
            
            logger.info(f"📊 Input shape: {input_ids.shape}")
            if pixel_values is not None:
                logger.info(f"🖼️  Image input: {pixel_values.shape}")
            if input_features is not None:
                logger.info(f"🎵 Audio input: {input_features.shape}")
            
            # Initialize generation state
            batch_size = input_ids.shape[0]
            past_key_values = {
                f"past_key_values.{layer}.{kv}": np.zeros(
                    [batch_size, self.num_key_value_heads, 0, self.head_dim], 
                    dtype=np.float32
                )
                for layer in range(self.num_hidden_layers)
                for kv in ("key", "value")
            }
            
            generated_tokens = np.empty((batch_size, 0), dtype=np.int64)
            image_features = None
            audio_features = None
            
            # Generation loop
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
                    
                    # Replace image tokens with features
                    mask = (input_ids == self.image_token_id)
                    if mask.any():
                        inputs_embeds[mask] = image_features.reshape(-1, image_features.shape[-1])
                
                # Process audio features (once)  
                if audio_features is None and input_features is not None and input_features_mask is not None:
                    audio_features = self.audio_session.run(
                        ["audio_features"],
                        {
                            "input_features": input_features,
                            "input_features_mask": input_features_mask
                        }
                    )[0]
                    
                    # Replace audio tokens with features
                    mask = (input_ids == self.audio_token_id)
                    if mask.any():
                        inputs_embeds[mask] = audio_features.reshape(-1, audio_features.shape[-1])
                
                # Run decoder
                outputs = self.decoder_session.run(None, {
                    "inputs_embeds": inputs_embeds,
                    "per_layer_inputs": per_layer_inputs,
                    "position_ids": position_ids,
                    **past_key_values
                })
                
                logits = outputs[0]
                present_key_values = outputs[1:]
                
                # Sample next token (greedy for now)
                next_token = np.argmax(logits[:, -1], axis=-1, keepdims=True)
                generated_tokens = np.concatenate([generated_tokens, next_token], axis=-1)
                
                # Update for next iteration
                input_ids = next_token
                attention_mask = np.ones_like(input_ids)
                position_ids = position_ids[:, -1:] + 1
                
                # Update key-value cache
                for i, key in enumerate(past_key_values.keys()):
                    past_key_values[key] = present_key_values[i]
                
                # Check for EOS token
                if (next_token == self.eos_token_id).all():
                    logger.info(f"🏁 EOS reached at step {step}")
                    break
                
                # Progress logging
                if step > 0 and step % 50 == 0:
                    logger.info(f"⏳ Generated {step} tokens...")
            
            # Decode the generated tokens
            response = self.processor.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )[0]
            
            logger.info(f"✅ Generation complete ({len(generated_tokens[0])} tokens)")
            return response.strip()
            
        except Exception as e:
            logger.error(f"💥 Generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


# Initialize the inference engine
inference_engine = LocalONNXGemmaEngine()


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
    title="Local ONNX Gemma 3n E2B",
    description="Multimodal AI inference using locally cached ONNX models",
    version="2.0.0",
    lifespan=lifespan
)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if inference_engine.is_loaded else "loading",
        "models_loaded": inference_engine.is_loaded,
        "cache_location": str(inference_engine.cache_dir),
        "quantization": {
            "embed_tokens": "quantized",
            "audio_encoder": "fp16", 
            "vision_encoder": "quantized",
            "decoder": "q4f16"
        }
    }


app.get("/chat")
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
    """Generate response from text, image, and/or audio inputs"""
    
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


@app.get("/")
async def root():
    """Serve the main index page"""
    return FileResponse("static/index.html")


def main():
    """Run the application"""
    print("="*70)
    print("🚀 LOCAL ONNX GEMMA 3N E2B MULTIMODAL SERVER")
    print("="*70)
    print("📁 Cache directory:", inference_engine.cache_dir)
    print("🔧 ONNX Runtime providers:", inference_engine.ort_providers)
    print("📊 Using CPU inference with all available cores")
    print("="*70)
    print()
    print("Starting server...")
    print("- Health check: http://localhost:8000/health")
    print("- API endpoint: http://localhost:8000/api/generate")
    print("- Interactive docs: http://localhost:8000/docs")
    print()
    
    # Run the FastAPI server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
