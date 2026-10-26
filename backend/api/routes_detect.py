"""
Audio Detection and Inference API Routes
Accepts audio files, validates format and size, runs real PyTorch inference,
and computes chunk-level scores and thresholds.
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.schemas.detection import DetectionComparisonResult, DetectionResult
from backend.services.audio_service import AudioService
from backend.services.inference_service import InferenceService
from backend.services.model_service import ModelService

router = APIRouter(tags=["Audio Detection"])

_model_service = ModelService()
_audio_service = AudioService()
_inference_service = InferenceService(_model_service, _audio_service)


def get_inference_service():
    return _inference_service


def get_audio_service():
    return _audio_service


@router.post("/detect", response_model=DetectionResult)
async def detect_audio(
    file: UploadFile = File(..., description="Audio file in WAV, FLAC, MP3, OGG, or M4A format"),
    model: str = Form(default="vit_b16", description="'vit_b16' or 'efficientnet_b0'"),
    threshold: Optional[float] = Form(default=None, description="Optional custom decision threshold"),
    inference_service: InferenceService = Depends(get_inference_service),
    audio_service: AudioService = Depends(get_audio_service),
):
    """
    Runs single-model audio deepfake detection.
    Validates file, applies standardized preprocessing, executes model forward pass,
    aggregates chunk probabilities, and compares against the frozen threshold.
    """
    file_bytes = await file.read()
    filename = file.filename or "uploaded_audio.wav"

    try:
        audio_service.validate_upload(filename, file_bytes)
        waveform, metadata = audio_service.load_and_inspect_audio(file_bytes, filename)
        result = inference_service.run_detection(
            waveform=waveform,
            audio_metadata=metadata,
            model_id=model,
            custom_threshold=threshold,
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during audio analysis: {str(e)}",
        )


@router.post("/detect/compare", response_model=DetectionComparisonResult)
async def detect_and_compare(
    file: UploadFile = File(..., description="Audio file in WAV, FLAC, MP3, OGG, or M4A format"),
    threshold_eff: Optional[float] = Form(default=None, description="Optional custom threshold for EfficientNet"),
    threshold_vit: Optional[float] = Form(default=None, description="Optional custom threshold for ViT"),
    inference_service: InferenceService = Depends(get_inference_service),
    audio_service: AudioService = Depends(get_audio_service),
):
    """
    Runs dual-model comparative detection (EfficientNet-B0 vs ViT-B/16).
    Computes predictions, latencies, and chunk-level scores for both models,
    flagging any classification disagreement between CNN and Transformer.
    """
    file_bytes = await file.read()
    filename = file.filename or "uploaded_audio.wav"

    try:
        audio_service.validate_upload(filename, file_bytes)
        waveform, metadata = audio_service.load_and_inspect_audio(file_bytes, filename)
        comparison = inference_service.run_dual_comparison(
            waveform=waveform,
            audio_metadata=metadata,
            custom_threshold_eff=threshold_eff,
            custom_threshold_vit=threshold_vit,
        )
        return comparison
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during dual-model comparison: {str(e)}",
        )
