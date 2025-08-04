"""
Enhanced ONNX Gemma 3n E2B with Token Streaming Support
This implementation adds real-time token streaming using Server-Sent Events (SSE)
"""

import os
import json
import logging
import tempfile
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from contextlib import asynccontextmanager
import uvicorn
from PIL import Image

# For Server-Sent Events (SSE)
from sse_starlette.sse import EventSourceResponse

# Import our modular components
from gemma3n_onnx_engine import LocalONNXGemmaEngine
from piper_tts_engine import PiperTTSEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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