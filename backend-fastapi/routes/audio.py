"""
ARIHANT SOC - Audio Detection Routes
====================================
API endpoints for audio spoof/deepfake detection
"""

from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from typing import Optional
from datetime import datetime
import tempfile
import os
import numpy as np

from schemas.response_schemas import AudioDetectionResponse
from services.prediction_service import PredictionService
from services.elastic_service import elastic_service

# Try importing audio processing libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

router = APIRouter()

# Audio processing constants
SAMPLE_RATE = 16000
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1024
MAX_DURATION = 30  # seconds


def extract_audio_features(audio_path: str) -> Optional[np.ndarray]:
    """
    Extract mel spectrogram features from audio file
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Mel spectrogram as numpy array or None if failed
    """
    if not LIBROSA_AVAILABLE:
        return None
    
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_norm = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_norm
        
    except Exception as e:
        print(f"[AUDIO] Feature extraction failed: {e}")
        return None


@router.post(
    "/detect",
    response_model=AudioDetectionResponse,
    summary="Detect Audio Spoof/Deepfake",
    description="Analyze audio file to detect synthetic or manipulated audio"
)
async def detect_audio_spoof(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC)"),
    filename: Optional[str] = Form(None)
):
    """
    Detect if audio is real or fake (deepfake/spoof)
    
    - **file**: Audio file upload (WAV, MP3, FLAC supported)
    - **filename**: Original filename (optional)
    
    Returns prediction (Real/Fake), confidence, and recommendations
    """
    model_loader = request.app.state.model_loader
    ws_manager = request.app.state.ws_manager
    
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    # Validate file type
    allowed_types = ["audio/wav", "audio/wave", "audio/x-wav", "audio/mpeg", 
                     "audio/mp3", "audio/flac", "audio/x-flac", "audio/ogg"]
    
    content_type = file.content_type or ""
    if not any(t in content_type.lower() for t in ["audio", "wav", "mp3", "flac", "ogg"]):
        # Check file extension as fallback
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: WAV, MP3, FLAC, OGG"
            )
    
    # Check file size (max 50MB)
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 50MB"
        )
    
    # Save to temp file for processing
    temp_path = None
    try:
        ext = os.path.splitext(file.filename or ".wav")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Extract features
        audio_features = None
        audio_info = {
            "filename": filename or file.filename,
            "file_size_bytes": file_size,
            "content_type": content_type
        }
        
        if LIBROSA_AVAILABLE:
            audio_features = extract_audio_features(temp_path)
            
            # Get audio duration
            try:
                y, sr = librosa.load(temp_path, sr=None, duration=1)
                duration = librosa.get_duration(path=temp_path)
                audio_info["duration_seconds"] = round(duration, 2)
                audio_info["sample_rate"] = sr
            except:
                pass
        
        # Run prediction
        prediction_service = PredictionService(model_loader)
        result = await prediction_service.predict_audio(audio_features)
        
        # Add audio info to result
        result["audio_features"] = audio_info
        
        # Log to Elasticsearch
        await elastic_service.log_prediction(
            index=elastic_service.INDEX_AUDIO,
            input_data=audio_info,
            prediction=result
        )
        
        # Broadcast alert if fake detected with high confidence
        if result["is_fake"] and result["confidence"] > 0.8:
            import uuid
            await ws_manager.broadcast_alert(
                alert_id=str(uuid.uuid4()),
                attack_type="Audio Deepfake",
                severity=result["severity"],
                confidence=result["confidence"],
                source="AUDIO",
                details={
                    "filename": audio_info.get("filename"),
                    "duration": audio_info.get("duration_seconds")
                }
            )
        
        return AudioDetectionResponse(
            success=True,
            timestamp=datetime.utcnow(),
            processing_time_ms=result["processing_time_ms"],
            prediction=result["prediction"],
            is_fake=result["is_fake"],
            confidence=result["confidence"],
            severity=result["severity"],
            audio_features=audio_info,
            recommendation=result["recommendation"]
        )
        
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass


@router.post(
    "/detect/features",
    response_model=AudioDetectionResponse,
    summary="Detect from Pre-extracted Features",
    description="Analyze pre-extracted audio features (mel spectrogram)"
)
async def detect_from_features(
    request: Request,
    features: list
):
    """
    Detect audio spoof from pre-extracted features
    
    - **features**: Pre-extracted mel spectrogram features as flat array
    
    Useful when audio processing is done client-side
    """
    model_loader = request.app.state.model_loader
    
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    prediction_service = PredictionService(model_loader)
    
    # Convert to numpy array
    features_array = np.array(features)
    
    result = await prediction_service.predict_audio(features_array)
    
    return AudioDetectionResponse(
        success=True,
        timestamp=datetime.utcnow(),
        processing_time_ms=result["processing_time_ms"],
        prediction=result["prediction"],
        is_fake=result["is_fake"],
        confidence=result["confidence"],
        severity=result["severity"],
        audio_features={"source": "pre-extracted"},
        recommendation=result["recommendation"]
    )


@router.get(
    "/stats",
    summary="Audio Detection Statistics",
    description="Get statistics for audio detections"
)
async def get_audio_stats():
    """Get audio detection statistics"""
    stats = await elastic_service.get_stats(
        elastic_service.INDEX_AUDIO,
        time_range="24h"
    )
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "stats": stats,
        "librosa_available": LIBROSA_AVAILABLE
    }


@router.get(
    "/history",
    summary="Audio Detection History",
    description="Get recent audio detection history"
)
async def get_audio_history(
    limit: int = 50,
    time_range: str = "24h"
):
    """Get audio detection history"""
    logs = await elastic_service.search_logs(
        elastic_service.INDEX_AUDIO,
        time_range=time_range,
        size=limit
    )
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "count": len(logs),
        "logs": logs
    }


@router.get(
    "/supported-formats",
    summary="Get Supported Audio Formats",
    description="List of supported audio file formats"
)
async def get_supported_formats():
    """Get list of supported audio formats"""
    return {
        "success": True,
        "formats": [
            {"extension": ".wav", "mime_type": "audio/wav", "description": "Waveform Audio"},
            {"extension": ".mp3", "mime_type": "audio/mpeg", "description": "MPEG Audio Layer 3"},
            {"extension": ".flac", "mime_type": "audio/flac", "description": "Free Lossless Audio Codec"},
            {"extension": ".ogg", "mime_type": "audio/ogg", "description": "Ogg Vorbis"},
            {"extension": ".m4a", "mime_type": "audio/mp4", "description": "MPEG-4 Audio"}
        ],
        "max_file_size_mb": 50,
        "max_duration_seconds": MAX_DURATION,
        "processing_available": LIBROSA_AVAILABLE
    }
