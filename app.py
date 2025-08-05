"""
Enhanced ONNX Gemma 3n E2B with Token Streaming Support and Proper Chat Template Handling
This implementation adds real-time token streaming using Server-Sent Events (SSE)
and properly handles Gemma 3n chat templates for system prompts
FIXED: File attachment "I/O operation on closed file" error
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
    title="Local ONNX Gemma 3n E2B with Streaming and Proper Chat Templates",
    description="Multimodal AI inference with real-time token streaming using locally cached ONNX models and proper Gemma 3n chat template handling",
    version="2.2.1",
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
        "chat_template_support": True,
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


async def process_messages_and_attachments(text: str, image: UploadFile = None, audio: UploadFile = None, messages_json: str = None) -> tuple:
    """
    Process input messages and attachments into proper Gemma 3n message format.
    Supports both direct text input and structured message format from frontend.
    FIXED: Properly handle file reading to avoid "I/O operation on closed file" error
    """
    
    temp_files = []
    
    try:
        # Option 1: Structured messages from frontend (e.g., translator)
        if messages_json:
            try:
                messages = json.loads(messages_json)
                logger.info(f"📝 Using structured messages: {len(messages)} message(s)")
                
                # Process any file attachments that need to be linked to the messages
                if image and image.filename:
                    # ✅ FIXED: Read file content once and create temp file properly
                    content_bytes = await image.read()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(content_bytes)
                        temp_files.append(tmp_file.name)
                    
                    # Open the temp file with PIL (avoiding closed file issue)
                    img = Image.open(temp_files[-1])
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Find and update image content in messages
                    for message in messages:
                        if isinstance(message.get('content'), list):
                            for content_item in message['content']:
                                if content_item.get('type') == 'image' and 'image' not in content_item:
                                    content_item['image'] = img
                                    logger.info(f"🖼️  Image attached to message: {img.size}")
                
                if audio and audio.filename:
                    # ✅ FIXED: Read audio content once and create temp file properly
                    content_bytes = await audio.read()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(content_bytes)
                        temp_files.append(tmp_file.name)
                    
                    # Find and update audio content in messages
                    for message in messages:
                        if isinstance(message.get('content'), list):
                            for content_item in message['content']:
                                if content_item.get('type') == 'audio' and isinstance(content_item.get('audio'), str):
                                    content_item['audio'] = temp_files[-1]
                                    logger.info(f"🎵 Audio attached to message: {temp_files[-1]}")
                
                return messages, temp_files
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse messages JSON: {e}")
                # Fall back to text processing
        
        # Option 2: Direct text input (legacy format for chat interface)
        content = [{"type": "text", "text": text}]
        
        # Handle image upload
        if image and image.filename:
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Invalid image format")
            
            # ✅ FIXED: Read file content once and create temp file properly
            content_bytes = await image.read()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(content_bytes)
                temp_files.append(tmp_file.name)
            
            # Open the temp file with PIL (avoiding closed file issue)
            img = Image.open(temp_files[-1])
            if img.mode != 'RGB':
                img = img.convert('RGB')
            content.append({"type": "image", "image": img})
            logger.info(f"🖼️  Image processed: {img.size}")
        
        # Handle audio upload
        if audio and audio.filename:
            if not audio.content_type.startswith("audio/"):
                raise HTTPException(status_code=400, detail="Invalid audio format")
            
            # ✅ FIXED: Read audio content once and create temp file properly
            content_bytes = await audio.read()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(content_bytes)
                temp_files.append(tmp_file.name)
            
            content.append({"type": "audio", "audio": temp_files[-1]})
            logger.info(f"🎵 Audio processed: {temp_files[-1]}")
        
        # Create simple user message
        messages = [{"role": "user", "content": content}]
        logger.info(f"📝 Created simple user message with {len(content)} content item(s)")
        
        return messages, temp_files
        
    except Exception as e:
        # Clean up on error
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass
        raise e


@app.post("/api/generate")
async def generate_response(
    text: str = Form(default=""),
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
    messages: str = Form(None),  # JSON string of structured messages
    max_tokens: int = Form(1000)
):
    """Generate response from text, image, and/or audio inputs (non-streaming) with proper chat template support"""
    
    # Limit max tokens
    max_tokens = min(max_tokens, 2048)
    
    temp_files = []
    
    try:
        # Process input into proper message format
        processed_messages, temp_files = await process_messages_and_attachments(text, image, audio, messages)
        
        # Generate response using proper message structure
        response = await inference_engine.generate_response(processed_messages, max_new_tokens=max_tokens)
        
        return {
            "response": response,
            "input_modalities": {
                "text": bool(text),
                "image": image is not None and image.filename,
                "audio": audio is not None and audio.filename,
                "structured_messages": messages is not None
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
    text: str = Form(default=""),
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
    messages: str = Form(None),  # JSON string of structured messages
    max_tokens: int = Form(1000)
):
    """
    Generate streaming response from text, image, and/or audio inputs with proper chat template support
    Uses Server-Sent Events (SSE) for real-time token streaming
    FIXED: Process files before creating the generator to avoid closed file errors
    """
    
    # Limit max tokens
    max_tokens = min(max_tokens, 2048)
    
    # Process files BEFORE creating the generator
    # This ensures file reading happens while they're still open
    try:
        processed_messages, temp_files = await process_messages_and_attachments(text, image, audio, messages)
    except Exception as e:
        logger.error(f"Failed to process messages: {e}")
        # Return error as SSE stream
        async def error_generator():
            yield {
                "event": "error",
                "data": json.dumps({"error": f"Failed to process input: {str(e)}", "finished": True})
            }
        return EventSourceResponse(error_generator())
    
    async def event_generator():
        """Generator function for Server-Sent Events"""
        try:
            # Use already processed messages (no file operations here!)
            async for chunk in inference_engine.generate_response_stream(processed_messages, max_new_tokens=max_tokens):
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
            # Clean up temporary files after streaming is complete
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                    logger.debug(f"Cleaned up temp file: {temp_file}")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
    
    # Return Server-Sent Events response
    return EventSourceResponse(event_generator())


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
    print("💬 Chat Template Support: Enabled for proper system prompts")
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