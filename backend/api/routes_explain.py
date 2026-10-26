"""
Explainability API Routes
Generates real visual explanations (spectrograms, Grad-CAM, ViT attention rollout)
for uploaded audio samples.
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.services.audio_service import AudioService
from backend.services.explainability_service import ExplainabilityService
from backend.services.model_service import ModelService

router = APIRouter(tags=["Explainability"])

_model_service = ModelService()
_audio_service = AudioService()
_explainability_service = ExplainabilityService(_model_service, _audio_service)


def get_explainability_service():
    return _explainability_service


def get_audio_service():
    return _audio_service


@router.post("/explain")
async def explain_audio(
    file: UploadFile = File(..., description="Audio file to generate acoustic heatmaps for"),
    chunk_index: int = Form(default=0, ge=0, description="0-indexed audio chunk to visualize"),
    explainability_service: ExplainabilityService = Depends(get_explainability_service),
    audio_service: AudioService = Depends(get_audio_service),
):
    """
    Computes genuine visual acoustic explanations:
    1. Log-mel Spectrogram (128x128 frequency-time map)
    2. CNN Grad-CAM Heatmap & Spectrogram Overlay
    3. Vision Transformer Self-Attention / Spatial Patch Attribution Map & Overlay
    """
    file_bytes = await file.read()
    filename = file.filename or "uploaded_audio.wav"

    try:
        audio_service.validate_upload(filename, file_bytes)
        waveform, _ = audio_service.load_and_inspect_audio(file_bytes, filename)
        explanations = explainability_service.explain_audio(waveform, chunk_idx=chunk_index)
        return explanations
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating visual explanations: {str(e)}",
        )
