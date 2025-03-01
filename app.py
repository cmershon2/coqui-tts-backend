from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import os
import io
import time
import torch
import torchaudio
import numpy as np
from typing import List, Optional, Dict, Any
import httpx
import logging
from pathlib import Path
import asyncio
import uvicorn
import json
import boto3

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
USE_DEEPSPEED = HAS_DEEPSPEED and os.getenv("USE_DEEPSPEED", "True").lower() in ("true", "1", "t")
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


# Request models
class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    speaker_wav: Optional[List[str]] = None
    speaker_name: Optional[str] = None
    speed: float = 1.0
    temperature: float = 0.65
    length_penalty: float = 1.0
    repetition_penalty: float = 2.0
    top_k: int = 50
    top_p: float = 0.8
    enable_text_splitting: bool = True

    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, this is a test of the text to speech system.",
                "language": "en",
                "speaker_name": "Ana Florence",
                "speed": 1.0,
                "temperature": 0.65
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
async def validate_token(request: Request) -> TokenValidationResponse:
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    token = auth_header.replace("Bearer ", "")

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
                raise HTTPException(status_code=401, detail="Invalid authentication token")

            user_data = response.json()
            return TokenValidationResponse(
                user_id=user_data.get("id"),
                role=user_data.get("role", "user")
            )

        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(status_code=401, detail="Authentication failed")


# Dependency for authentication
async def get_current_user(request: Request) -> TokenValidationResponse:
    return await validate_token(request)


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
        if HAS_DEEPSPEED:
            logger.info("DeepSpeed is available but disabled by configuration")
        else:
            logger.info("DeepSpeed is not available, using PyTorch only")

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
        text, language, gpt_cond_latent, speaker_embedding,
        speed=1.0, temperature=0.65, length_penalty=1.0,
        repetition_penalty=2.0, top_k=50, top_p=0.8, **kwargs
):
    """Generate audio in chunks for streaming"""
    try:
        # Use executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        # Get generator from model
        chunk_generator = await loop.run_in_executor(
            None,
            lambda: model.inference_stream(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                speed=speed,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                **kwargs
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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Add DeepSpeed status to health check
    return {
        "status": "healthy", 
        "model": "xtts_v2",
        "deepspeed_available": HAS_DEEPSPEED,
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
            speed=request.speed,
            temperature=request.temperature,
            length_penalty=request.length_penalty,
            repetition_penalty=request.repetition_penalty,
            top_k=request.top_k,
            top_p=request.top_p,
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
                speaker_embedding=speaker_embedding,
                speed=request.speed,
                temperature=request.temperature,
                length_penalty=request.length_penalty,
                repetition_penalty=request.repetition_penalty,
                top_k=request.top_k,
                top_p=request.top_p,
                enable_text_splitting=request.enable_text_splitting
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


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)