"""
ONNX Gemma 3n E2B Web Application
High-performance CPU inference with persistent model loading
"""

import os
import gc
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, AutoConfig
import uvicorn
from PIL import Image
import soundfile as sf
import tempfile
import psutil
from contextlib import asynccontextmanager
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor system memory usage"""
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current memory usage"""
        memory = psutil.virtual_memory()
        return {
            "ram_used_gb": memory.used / 1024**3,
            "ram_available_gb": memory.available / 1024**3,
            "ram_total_gb": memory.total / 1024**3,
            "ram_percent": memory.percent
        }
    
    @staticmethod
    def log_memory(context: str):
        """Log memory usage with context"""
        mem = MemoryMonitor.get_memory_info()
        logger.info(f"{context} - RAM: {mem['ram_used_gb']:.1f}GB/{mem['ram_total_gb']:.1f}GB ({mem['ram_percent']:.1f}%)")

class ONNXGemmaEngine:
    """ONNX-optimized Gemma 3n E2B inference engine with persistent loading"""
    
    def __init__(self, model_cache_dir: Optional[str] = None):
        # Model configuration
        self.model_id = "google/gemma-3n-e2b-it"
        self.onnx_model_id = "onnx-community/gemma-3n-E2B-it-ONNX"
        
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
        self.load_time = None
        
        # Configuration values (will be set after loading config)
        self.num_key_value_heads = None
        self.head_dim = None
        self.num_hidden_layers = None
        self.eos_token_id = 106  # Gemma 3n specific
        self.image_token_id = None
        self.audio_token_id = None
        
        # ONNX Runtime settings for CPU optimization
        self.ort_providers = ['CPUExecutionProvider']
        self.ort_session_options = ort.SessionOptions()
        self.ort_session_options.enable_cpu_mem_arena = False  # Reduce memory overhead
        self.ort_session_options.enable_mem_pattern = False
        self.ort_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    def _get_model_paths(self) -> Dict[str, Path]:
        """Get paths to ONNX model files"""
        onnx_dir = self.cache_dir / "onnx"
        
        # Try different quantization options in order of preference
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
            
            # Check if all files exist
            if all(path.exists() for path in paths.values()):
                logger.info(f"Using quantization: {option}")
                return paths
        
        raise FileNotFoundError(f"ONNX model files not found in {onnx_dir}. Please run the download script first.")
    
    async def load_models(self):
        """Load all model components with persistent loading"""
        if self.is_loaded:
            logger.info("Models already loaded")
            return
        
        start_time = time.time()
        logger.info("Loading ONNX Gemma 3n E2B models...")
        MemoryMonitor.log_memory("Before model loading")
        
        try:
            # 1. Load processor and config
            logger.info("Loading processor and config...")
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
            
            logger.info("✓ Processor and config loaded")
            
            # 2. Get model file paths
            model_paths = self._get_model_paths()
            
            # 3. Load ONNX sessions
            logger.info("Loading ONNX inference sessions...")
            
            # Embed tokens model
            logger.info(f"Loading embed model: {model_paths['embed'].name}")
            self.embed_session = ort.InferenceSession(
                str(model_paths['embed']),
                sess_options=self.ort_session_options,
                providers=self.ort_providers
            )
            
            # Audio encoder
            logger.info(f"Loading audio model: {model_paths['audio'].name}")
            self.audio_session = ort.InferenceSession(
                str(model_paths['audio']),
                sess_options=self.ort_session_options,
                providers=self.ort_providers
            )
            
            # Vision encoder  
            logger.info(f"Loading vision model: {model_paths['vision'].name}")
            self.vision_session = ort.InferenceSession(
                str(model_paths['vision']),
                sess_options=self.ort_session_options,
                providers=self.ort_providers
            )
            
            # Decoder (largest model)
            logger.info(f"Loading decoder model: {model_paths['decoder'].name}")
            self.decoder_session = ort.InferenceSession(
                str(model_paths['decoder']),
                sess_options=self.ort_session_options,
                providers=self.ort_providers
            )
            
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            MemoryMonitor.log_memory("After model loading")
            logger.info(f"✓ All models loaded successfully in {self.load_time:.1f}s")
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.cleanup()
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")
    
    def cleanup(self):
        """Clean up loaded models"""
        logger.info("Cleaning up models...")
        
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
        MemoryMonitor.log_memory("After cleanup")
    
    async def generate_response(self, messages: List[Dict], max_new_tokens: int = 256) -> str:
        """Generate response using ONNX models"""
        if not self.is_loaded:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        start_time = time.time()
        logger.info(f"Starting generation with max_new_tokens={max_new_tokens}")
        
        try:
            # 1. Process input with chat template
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Convert to numpy
            input_ids = inputs["input_ids"].numpy()
            attention_mask = inputs["attention_mask"].numpy()
            position_ids = np.cumsum(attention_mask, axis=-1) - 1
            
            # Extract multimodal inputs
            pixel_values = inputs["pixel_values"].numpy() if "pixel_values" in inputs else None
            input_features = inputs["input_features"].numpy().astype(np.float32) if "input_features" in inputs else None
            input_features_mask = inputs["input_features_mask"].numpy() if "input_features_mask" in inputs else None
            
            logger.info(f"Input processed: {input_ids.shape}")
            
            # 2. Initialize generation state
            batch_size = input_ids.shape[0]
            past_key_values = {
                f"past_key_values.{layer}.{kv}": np.zeros([batch_size, self.num_key_value_heads, 0, self.head_dim], dtype=np.float32)
                for layer in range(self.num_hidden_layers)
                for kv in ("key", "value")
            }
            
            generated_tokens = []
            image_features = None  
            audio_features = None
            
            # 3. Generation loop
            for step in range(max_new_tokens):
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
                if audio_features is None and input_features is not None:
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
                decoder_inputs = {
                    "inputs_embeds": inputs_embeds,
                    "per_layer_inputs": per_layer_inputs,
                    "position_ids": position_ids,
                    **past_key_values
                }
                
                decoder_outputs = self.decoder_session.run(None, decoder_inputs)
                logits = decoder_outputs[0]
                present_key_values = decoder_outputs[1:]
                
                # Sample next token (greedy for now)
                next_token_id = np.argmax(logits[:, -1], axis=-1, keepdims=True)
                generated_tokens.append(next_token_id[0, 0])
                
                # Check for EOS
                if next_token_id[0, 0] == self.eos_token_id:
                    break
                
                # Update for next iteration
                input_ids = next_token_id
                attention_mask = np.ones_like(input_ids)
                position_ids = position_ids[:, -1:] + 1
                
                # Update past key values
                for j, key in enumerate(past_key_values.keys()):
                    past_key_values[key] = present_key_values[j]
            
            # 4. Decode generated tokens
            if generated_tokens:
                response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                response = "No response generated"
            
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f}s ({len(generated_tokens)} tokens)")
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

# Initialize engine
inference_engine = ONNXGemmaEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - load models persistently
    await inference_engine.load_models()
    logger.info("Application ready - models loaded and ready for inference")
    yield
    # Shutdown
    inference_engine.cleanup()

# Initialize FastAPI app
app = FastAPI(
    title="ONNX Gemma 3n E2B",
    description="High-performance ONNX inference with persistent model loading",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check with model status"""
    memory_info = MemoryMonitor.get_memory_info()
    return {
        "status": "healthy",
        "models_loaded": inference_engine.is_loaded,
        "load_time_seconds": inference_engine.load_time,
        "cache_dir": str(inference_engine.cache_dir),
        "memory": memory_info
    }

@app.post("/api/generate/text")
async def generate_text(text: str = Form(...)):
    """Generate text-only response"""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }
    ]
    
    response = await inference_engine.generate_response(messages)
    return {"response": response, "type": "text"}

@app.post("/api/generate/multimodal")
async def generate_multimodal(
    text: str = Form(...),
    image: UploadFile = File(None),
    audio: UploadFile = File(None)
):
    """Generate multimodal response"""
    content = [{"type": "text", "text": text}]
    temp_files = []
    
    try:
        # Handle image
        if image:
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Invalid image format")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                content_bytes = await image.read()
                tmp_file.write(content_bytes)
                temp_files.append(tmp_file.name)
            
            img = Image.open(temp_files[-1])
            content.append({"type": "image", "image": img})
        
        # Handle audio
        if audio:
            if not audio.content_type.startswith("audio/"):
                raise HTTPException(status_code=400, detail="Invalid audio format")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                content_bytes = await audio.read()
                tmp_file.write(content_bytes)
                temp_files.append(tmp_file.name)
            
            # Load audio data
            audio_data, sample_rate = sf.read(temp_files[-1])
            content.append({"type": "audio", "audio": temp_files[-1]})
        
        messages = [{"role": "user", "content": content}]
        response = await inference_engine.generate_response(messages)
        
        return {"response": response, "type": "multimodal"}
        
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass

@app.get("/api/memory")
async def get_memory_info():
    """Get current memory usage"""
    return MemoryMonitor.get_memory_info()

# Serve static files (you'll need to create a static directory with HTML)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

def main():
    """Run the ONNX application"""
    print("="*60)
    print("ONNX GEMMA 3N E2B WEB APPLICATION")
    print("="*60)
    
    # System info
    memory_info = MemoryMonitor.get_memory_info()
    print(f"System RAM: {memory_info['ram_total_gb']:.1f}GB")
    print(f"Available RAM: {memory_info['ram_available_gb']:.1f}GB")
    print(f"Cache directory: {inference_engine.cache_dir}")
    print("Model loading: Persistent (models stay in memory)")
    print("-" * 60)
    print("Starting server...")
    print("Application: http://localhost:8000")
    print("Health check: http://localhost:8000/health")
    print("Memory info: http://localhost:8000/api/memory")
    print("Press Ctrl+C to stop")
    print("="*60)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()
