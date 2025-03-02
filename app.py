from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import os
import io
import time
import torch
import torchaudio
import numpy as np
from typing import List, Optional, Dict, Any, Set
import httpx
import logging
from pathlib import Path
import asyncio
import uvicorn
import json
import boto3
import base64

from dotenv import load_dotenv
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import subprocess
import sys

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Install DeepSpeed at runtime if needed
USE_DEEPSPEED = os.getenv("USE_DEEPSPEED", "True").lower() in ("true", "1", "t")
if USE_DEEPSPEED:
    try:
        import deepspeed
        logger.info("DeepSpeed already installed")
    except ImportError:
        logger.info("Installing DeepSpeed...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "deepspeed==0.16.4", "--no-deps"])
        logger.info("DeepSpeed installed successfully")

# App configuration
app = FastAPI(
    title="Coqui TTS API",
    description="A FastAPI service for Coqui's XTTS text-to-speech model with streaming support",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=json.loads(os.getenv("ALLOWED_ORIGINS", '["*"]')),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "./models/xtts_v2")
MODEL_PATH = LOCAL_MODEL_DIR
CHECKPOINT_DIR = LOCAL_MODEL_DIR
# Only use DeepSpeed if it's available and enabled
USE_DEEPSPEED = os.getenv("USE_DEEPSPEED", "True").lower() in ("true", "1", "t")
USE_CUDA = torch.cuda.is_available() and os.getenv("USE_CUDA", "True").lower() in ("true", "1", "t")
SAMPLE_RATE = 24000  # XTTS uses 24kHz

# S3 configuration
USE_S3 = os.getenv("USE_S3", "False").lower() in ("true", "1", "t")
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_MODEL_PATH = os.getenv("S3_MODEL_PATH", "xtts_v2")

# Cache for speaker embeddings
speaker_cache = {}
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "100"))

# Supabase authentication config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Use service key for backend
if not (SUPABASE_URL and SUPABASE_KEY):
    logger.warning("Supabase credentials not fully configured. Authentication will not work correctly.")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # Map user_id to set of connection_ids

    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str):
        # Don't accept again - the websocket is already accepted in the handler
        self.active_connections[connection_id] = websocket
        
        # Add connection to user's list
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        logger.info(f"Client connected: {connection_id} (user: {user_id})")

    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            # Find and remove from user connections
            for user_id, connections in self.user_connections.items():
                if connection_id in connections:
                    connections.remove(connection_id)
                    if len(connections) == 0:
                        # Clean up empty user entries
                        del self.user_connections[user_id]
                    break
            
            # Remove from active connections
            del self.active_connections[connection_id]
            logger.info(f"Client disconnected: {connection_id}")

    async def send_audio_chunk(self, connection_id: str, chunk: bytes):
        if connection_id in self.active_connections:
            # Encode audio chunk as base64 for sending over WebSocket
            base64_chunk = base64.b64encode(chunk).decode('utf-8')
            await self.active_connections[connection_id].send_json({
                "type": "audio_chunk",
                "data": base64_chunk
            })

    async def send_message(self, connection_id: str, message: Dict):
        if connection_id in self.active_connections:
            await self.active_connections[connection_id].send_json(message)

    async def send_error(self, connection_id: str, error_message: str):
        if connection_id in self.active_connections:
            await self.active_connections[connection_id].send_json({
                "type": "error",
                "message": error_message
            })

# Initialize connection manager
manager = ConnectionManager()

# Request models
class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    speaker_wav: Optional[List[str]] = None
    speaker_name: Optional[str] = None
    enable_text_splitting: bool = True

    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, this is a test of the text to speech system.",
                "language": "en",
                "speaker_name": "Ana Florence"
            }
        }


class TokenValidationResponse(BaseModel):
    user_id: str
    role: str


# Global model variable
model = None


def download_model_from_s3():
    """Download model files from Digital Ocean Spaces if they don't exist locally"""
    if not USE_S3:
        logger.info("S3 downloading disabled, skipping")
        return True

    try:
        logger.info(f"Checking for model files in {LOCAL_MODEL_DIR}")
        local_dir = Path(LOCAL_MODEL_DIR)
        local_dir.mkdir(parents=True, exist_ok=True)

        # Check if model files already exist locally
        config_file = local_dir / "config.json"
        if config_file.exists():
            # Check for additional required files
            required_files = ["model.pth", "vocab.json"]
            all_files_exist = all((local_dir / file).exists() for file in required_files)

            if all_files_exist:
                logger.info("Model files already exist locally, skipping download")
                return True
            else:
                logger.info("Some model files are missing, downloading from Digital Ocean Spaces")

        if not S3_BUCKET:
            logger.error("S3_BUCKET environment variable not set")
            return False

        logger.info(f"Downloading model files from DO Spaces bucket {S3_BUCKET}")

        # Set up Digital Ocean Spaces connection
        session = boto3.session.Session()
        s3_client = session.client('s3',
                                   region_name=os.getenv('DO_REGION', 'nyc3'),
                                   endpoint_url=os.getenv('DO_ENDPOINT_URL', 'https://nyc3.digitaloceanspaces.com'),
                                   aws_access_key_id=os.getenv('SPACES_KEY'),
                                   aws_secret_access_key=os.getenv('SPACES_SECRET'))

        # List objects in the bucket with the specified prefix
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=S3_MODEL_PATH
        )

        if 'Contents' not in response:
            logger.error(f"No files found in {S3_BUCKET}/{S3_MODEL_PATH}")
            return False

        # Download each object
        for obj in response['Contents']:
            key = obj['Key']

            # Skip the directory itself
            if key.endswith('/'):
                continue

            # Create the relative path for the local file
            target_path = key.replace(S3_MODEL_PATH, '')
            if target_path.startswith('/'):
                target_path = target_path[1:]

            if not target_path:  # Skip the directory itself
                continue

            target = local_dir / target_path
            target.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {key} to {target}")
            s3_client.download_file(S3_BUCKET, key, str(target))

        logger.info("Model download complete")
        return True

    except Exception as e:
        logger.error(f"Unexpected error during model download: {e}")
        return False

# Helper functions for authentication
async def validate_token(token: str) -> TokenValidationResponse:
    if not token:
        raise ValueError("Missing authentication token")

    async with httpx.AsyncClient() as client:
        try:
            # Verify the JWT with Supabase Auth
            response = await client.get(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "apiKey": SUPABASE_KEY
                }
            )

            if response.status_code != 200:
                raise ValueError("Invalid authentication token")

            user_data = response.json()
            return TokenValidationResponse(
                user_id=user_data.get("id"),
                role=user_data.get("role", "user")
            )

        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise ValueError(f"Authentication failed: {str(e)}")


async def validate_http_token(request: Request) -> TokenValidationResponse:
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    token = auth_header.replace("Bearer ", "")
    try:
        return await validate_token(token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


# Dependency for authentication
async def get_current_user(request: Request) -> TokenValidationResponse:
    return await validate_http_token(request)


# Load model on startup
@app.on_event("startup")
async def startup_event():
    global model

    try:
        # First download the model if needed
        if not download_model_from_s3():
            logger.error("Failed to download model files from S3")
            # We continue anyway, in case the files are already present

        logger.info("Loading XTTS model...")
        # Load XTTS model with non-blocking operation
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(None, load_model)
        logger.info("XTTS model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load XTTS model: {str(e)}")
        # We don't exit - the app will return 503 errors until model is loaded


def load_model():
    """Load the XTTS model"""
    config = XttsConfig()
    config_path = os.path.join(MODEL_PATH, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    config.load_json(config_path)
    model = Xtts.init_from_config(config)

    # Log DeepSpeed status
    if USE_DEEPSPEED:
        logger.info("Loading model with DeepSpeed optimization")
    else:
        logger.info("DeepSpeed is not available or is disabled, using PyTorch only")

    checkpoint_path = CHECKPOINT_DIR
    model.load_checkpoint(config, checkpoint_dir=checkpoint_path, use_deepspeed=USE_DEEPSPEED)

    if USE_CUDA:
        model.cuda()

    return model


async def get_speaker_embedding(speaker_key, speaker_files=None, speaker_name=None):
    """Get speaker embedding from cache or generate new ones"""
    global speaker_cache

    # Return from cache if exists
    if speaker_key in speaker_cache:
        return speaker_cache[speaker_key]

    # Generate new embedding
    loop = asyncio.get_event_loop()
    if speaker_name:
        # Use built-in speaker
        gpt_cond_latent, speaker_embedding = await loop.run_in_executor(
            None, lambda: model.speaker_manager.speakers[speaker_name].values()
        )
    elif speaker_files:
        # Use provided audio files for voice cloning
        gpt_cond_latent, speaker_embedding = await loop.run_in_executor(
            None, lambda: model.get_conditioning_latents(audio_path=speaker_files)
        )
    else:
        raise ValueError("Either speaker_name or speaker_files must be provided")

    # Cache the embedding
    if len(speaker_cache) >= CACHE_MAX_SIZE:
        # Remove oldest item when cache is full
        speaker_cache.pop(next(iter(speaker_cache)))

    speaker_cache[speaker_key] = (gpt_cond_latent, speaker_embedding)
    return gpt_cond_latent, speaker_embedding


async def generate_audio_chunks(
        text, language, gpt_cond_latent, speaker_embedding, enable_text_splitting=True
):
    """Generate audio in chunks for streaming - simplified to match Coqui's example"""
    try:
        # Use executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        # Get generator from model with minimal parameters, following Coqui's example
        chunk_generator = await loop.run_in_executor(
            None,
            lambda: model.inference_stream(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                enable_text_splitting=enable_text_splitting
            )
        )

        # Stream chunks as they're generated
        for chunk in chunk_generator:
            # Convert chunk to WAV format
            wav_chunk = chunk.cpu().numpy().astype(np.float32)

            # Create wave file in memory
            buffer = io.BytesIO()
            with io.BytesIO() as audio_buffer:
                torchaudio.save(
                    audio_buffer,
                    torch.tensor(wav_chunk).unsqueeze(0),
                    SAMPLE_RATE,
                    format="wav"
                )
                audio_buffer.seek(0)
                buffer.write(audio_buffer.read())

            buffer.seek(0)
            yield buffer.read()

    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise


async def send_audio_chunks_to_websocket(
        connection_id: str,
        text: str, 
        language: str, 
        gpt_cond_latent, 
        speaker_embedding,
        enable_text_splitting=True
):
    """Generate audio and send chunks to a WebSocket connection"""
    try:
        # Send a start message
        await manager.send_message(connection_id, {
            "type": "start",
            "message": "Starting audio generation"
        })
        
        # Log parameters for debugging
        logger.info(f"TTS parameters: text='{text[:50]}...', language={language}")
        
        # Instead of using inference_stream which might be causing issues,
        # let's adapt the approach from the working REST implementation
        # We'll use run_in_executor to make it non-blocking
        
        loop = asyncio.get_event_loop()
        
        # Create a function that will wrap the inference and chunk handling 
        async def generate_and_send():
            try:
                # Perform inference using the basic inference method, which we know works
                out = await loop.run_in_executor(
                    None,
                    lambda: model.inference(
                        text=text,
                        language=language,
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        enable_text_splitting=enable_text_splitting
                    )
                )
                
                # Get the waveform
                wav = out["wav"]
                
                # Split the waveform into chunks for streaming
                # Using 0.5 second chunks (sample_rate * 0.5 samples)
                chunk_size = int(SAMPLE_RATE * 0.5)
                wav_len = len(wav)
                
                # Create and send chunks
                for i in range(0, wav_len, chunk_size):
                    # Get the chunk
                    chunk_end = min(i + chunk_size, wav_len)
                    wav_chunk = wav[i:chunk_end]
                    
                    # Convert to tensor and create WAV
                    buffer = io.BytesIO()
                    with io.BytesIO() as audio_buffer:
                        torchaudio.save(
                            audio_buffer,
                            torch.tensor(wav_chunk).unsqueeze(0),
                            SAMPLE_RATE,
                            format="wav"
                        )
                        audio_buffer.seek(0)
                        buffer.write(audio_buffer.read())
                    
                    buffer.seek(0)
                    
                    # Send the chunk
                    await manager.send_audio_chunk(connection_id, buffer.read())
                    
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.05)
                
                # Send completion message
                await manager.send_message(connection_id, {
                    "type": "complete",
                    "message": "Audio generation complete"
                })
                
            except Exception as e:
                logger.error(f"Error generating audio chunks: {str(e)}")
                await manager.send_error(connection_id, f"Error generating audio: {str(e)}")
        
        # Start the generation and sending process
        asyncio.create_task(generate_and_send())
            
    except Exception as e:
        logger.error(f"Error in WebSocket streaming: {str(e)}")
        await manager.send_error(connection_id, f"Error generating audio: {str(e)}")


@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """WebSocket endpoint for real-time TTS streaming"""
    if model is None:
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Server is initializing")
        return

    connection_id = f"conn_{int(time.time() * 1000)}_{id(websocket)}"
    user_id = None
    
    try:
        # First, accept the connection
        await websocket.accept()
        logger.info(f"WebSocket connection accepted: {connection_id}")
        
        # Wait for authentication message
        auth_message = await websocket.receive_json()
        
        # Validate authentication token
        if "token" not in auth_message:
            await websocket.send_json({"type": "error", "message": "Missing authentication token"})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
            
        try:
            # Validate the token
            user_data = await validate_token(auth_message["token"])
            user_id = user_data.user_id
            
            # Send welcome message directly
            await websocket.send_json({
                "type": "connected",
                "message": "Connection established",
                "user_id": user_id
            })
            
            # Now add to connection manager
            await manager.connect(websocket, connection_id, user_id)
            
            # Main message processing loop
            while True:
                # Wait for TTS request
                logger.info(f"Waiting for TTS request from client: {connection_id}")
                message = await websocket.receive_json()
                logger.info(f"Received message from client: {connection_id}")
                
                if "request" not in message:
                    logger.error(f"Invalid message format: {message}")
                    await manager.send_error(connection_id, "Invalid request format")
                    continue
                
                try:
                    # Extract the request data
                    request_data = message["request"]
                    
                    # Validate required fields
                    if "text" not in request_data or not request_data["text"]:
                        await manager.send_error(connection_id, "Text is required")
                        continue
                        
                    if "speaker_name" not in request_data and ("speaker_wav" not in request_data or not request_data["speaker_wav"]):
                        await manager.send_error(connection_id, "Either speaker_name or speaker_wav must be provided")
                        continue
                    
                    # Extract parameters
                    text = request_data["text"]
                    language = request_data.get("language", "en")
                    enable_text_splitting = request_data.get("enable_text_splitting", True)
                    
                    # Get speaker embeddings
                    if "speaker_name" in request_data and request_data["speaker_name"]:
                        speaker_name = request_data["speaker_name"]
                        speaker_key = f"name:{speaker_name}"
                        gpt_cond_latent, speaker_embedding = await get_speaker_embedding(
                            speaker_key=speaker_key,
                            speaker_name=speaker_name
                        )
                    else:
                        # For voice cloning
                        speaker_wav = request_data["speaker_wav"]
                        speaker_key = f"wav:{speaker_wav[0]}"
                        gpt_cond_latent, speaker_embedding = await get_speaker_embedding(
                            speaker_key=speaker_key,
                            speaker_files=speaker_wav
                        )
                    
                    # Process the request
                    await send_audio_chunks_to_websocket(
                        connection_id=connection_id,
                        text=text,
                        language=language,
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        enable_text_splitting=enable_text_splitting
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing WebSocket request: {str(e)}")
                    await manager.send_error(connection_id, f"Error processing request: {str(e)}")
                
        except ValueError as e:
            # Authentication failed
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
    finally:
        # Always clean up the connection
        if connection_id:
            manager.disconnect(connection_id)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Add DeepSpeed status to health check
    return {
        "status": "healthy", 
        "model": "xtts_v2",
        "deepspeed_enabled": USE_DEEPSPEED,
        "cuda_available": torch.cuda.is_available(),
        "cuda_enabled": USE_CUDA
    }


@app.get("/speakers")
async def list_speakers(current_user: TokenValidationResponse = Depends(get_current_user)):
    """List available built-in speakers"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        speakers = list(model.speaker_manager.speakers.keys())
        return {"speakers": speakers}
    except Exception as e:
        logger.error(f"Error retrieving speakers: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve speakers")


@app.get("/languages")
async def list_languages(current_user: TokenValidationResponse = Depends(get_current_user)):
    """List supported languages"""
    return {
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "pl", "name": "Polish"},
            {"code": "tr", "name": "Turkish"},
            {"code": "ru", "name": "Russian"},
            {"code": "nl", "name": "Dutch"},
            {"code": "cs", "name": "Czech"},
            {"code": "ar", "name": "Arabic"},
            {"code": "zh-cn", "name": "Chinese"},
            {"code": "ja", "name": "Japanese"},
            {"code": "hu", "name": "Hungarian"},
            {"code": "ko", "name": "Korean"}
        ]
    }


@app.post("/tts/stream")
async def stream_tts(
        request: TTSRequest,
        current_user: TokenValidationResponse = Depends(get_current_user)
):
    """Stream TTS audio with low latency"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Validate input
        if not request.text:
            raise HTTPException(status_code=400, detail="Text is required")

        if not request.speaker_name and not request.speaker_wav:
            raise HTTPException(status_code=400, detail="Either speaker_name or speaker_wav must be provided")

        # Determine speaker key for caching
        if request.speaker_name:
            speaker_key = f"name:{request.speaker_name}"
            gpt_cond_latent, speaker_embedding = await get_speaker_embedding(
                speaker_key=speaker_key,
                speaker_name=request.speaker_name
            )
        else:
            # For voice cloning, use the first file path as key
            speaker_key = f"wav:{request.speaker_wav[0]}"
            gpt_cond_latent, speaker_embedding = await get_speaker_embedding(
                speaker_key=speaker_key,
                speaker_files=request.speaker_wav
            )

        # Start streaming response
        stream = generate_audio_chunks(
            text=request.text,
            language=request.language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            enable_text_splitting=request.enable_text_splitting
        )

        # Return streaming response
        return StreamingResponse(
            stream,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tts_stream.wav"
            }
        )

    except Exception as e:
        logger.error(f"Error in stream_tts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts")
async def generate_tts(
        request: TTSRequest,
        current_user: TokenValidationResponse = Depends(get_current_user)
):
    """Generate TTS audio (non-streaming, single file)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Validate input
        if not request.text:
            raise HTTPException(status_code=400, detail="Text is required")

        if not request.speaker_name and not request.speaker_wav:
            raise HTTPException(status_code=400, detail="Either speaker_name or speaker_wav must be provided")

        # Determine speaker key for caching
        if request.speaker_name:
            speaker_key = f"name:{request.speaker_name}"
            gpt_cond_latent, speaker_embedding = await get_speaker_embedding(
                speaker_key=speaker_key,
                speaker_name=request.speaker_name
            )
        else:
            # For voice cloning, use the first file path as key
            speaker_key = f"wav:{request.speaker_wav[0]}"
            gpt_cond_latent, speaker_embedding = await get_speaker_embedding(
                speaker_key=speaker_key,
                speaker_files=request.speaker_wav
            )

        # Run inference in background to avoid blocking
        loop = asyncio.get_event_loop()
        out = await loop.run_in_executor(
            None,
            lambda: model.inference(
                text=request.text,
                language=request.language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding
            )
        )

        # Convert output to WAV format
        wav = out["wav"]

        # Create in-memory buffer
        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            torch.tensor(wav).unsqueeze(0),
            SAMPLE_RATE,
            format="wav"
        )
        buffer.seek(0)

        # Return as downloadable file
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.wav"
            }
        )

    except Exception as e:
        logger.error(f"Error in generate_tts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """WebSocket endpoint for real-time TTS streaming"""
    if model is None:
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Server is initializing")
        return

    connection_id = f"conn_{int(time.time() * 1000)}_{id(websocket)}"
    user_id = None
    
    try:
        # First, accept the connection
        await websocket.accept()
        logger.info(f"WebSocket connection accepted: {connection_id}")
        
        # Wait for authentication message
        auth_message = await websocket.receive_json()
        
        # Validate authentication token
        if "token" not in auth_message:
            await websocket.send_json({"type": "error", "message": "Missing authentication token"})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
            
        try:
            # Validate the token
            user_data = await validate_token(auth_message["token"])
            user_id = user_data.user_id
            
            # Send welcome message directly
            await websocket.send_json({
                "type": "connected",
                "message": "Connection established",
                "user_id": user_id
            })
            
            # Now add to connection manager
            await manager.connect(websocket, connection_id, user_id)
            
            # Main message processing loop
            while True:
                # Wait for TTS request
                logger.info(f"Waiting for TTS request from client: {connection_id}")
                message = await websocket.receive_json()
                logger.info(f"Received message from client: {connection_id}")
                
                if "request" not in message:
                    logger.error(f"Invalid message format: {message}")
                    await manager.send_error(connection_id, "Invalid request format")
                    continue
                
                try:
                    # Parse the TTS request - match the full REST API parameters
                    request_data = message["request"]
                    
                    # Validate required fields manually
                    if "text" not in request_data or not request_data["text"]:
                        await manager.send_error(connection_id, "Text is required")
                        continue
                        
                    if "speaker_name" not in request_data and ("speaker_wav" not in request_data or not request_data["speaker_wav"]):
                        await manager.send_error(connection_id, "Either speaker_name or speaker_wav must be provided")
                        continue
                    
                    # Get text and language
                    text = request_data["text"]
                    language = request_data.get("language", "en")
                    enable_text_splitting = request_data.get("enable_text_splitting", True)
                    
                    # Determine speaker key for caching
                    if "speaker_name" in request_data and request_data["speaker_name"]:
                        speaker_name = request_data["speaker_name"]
                        speaker_key = f"name:{speaker_name}"
                        gpt_cond_latent, speaker_embedding = await get_speaker_embedding(
                            speaker_key=speaker_key,
                            speaker_name=speaker_name
                        )
                    else:
                        # For voice cloning, use the first file path as key
                        speaker_wav = request_data["speaker_wav"]
                        speaker_key = f"wav:{speaker_wav[0]}"
                        gpt_cond_latent, speaker_embedding = await get_speaker_embedding(
                            speaker_key=speaker_key,
                            speaker_files=speaker_wav
                        )
                    
                    # Process the request in the background to avoid blocking the WebSocket
                    asyncio.create_task(send_audio_chunks_to_websocket(
                        connection_id=connection_id,
                        text=text,
                        language=language,
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        enable_text_splitting=enable_text_splitting
                    ))
                    
                except Exception as e:
                    logger.error(f"Error processing WebSocket request: {str(e)}")
                    await manager.send_error(connection_id, f"Error processing request: {str(e)}")
                
        except ValueError as e:
            # Authentication failed
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
    finally:
        # Always clean up the connection
        if connection_id:
            manager.disconnect(connection_id)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)