"""
Robustness Testing API Routes
Applies controlled acoustic perturbations to uploaded audio and measures model score deltas.
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.schemas.robustness import RobustnessTestResponse
from backend.services.audio_service import AudioService
from backend.services.inference_service import InferenceService
from backend.services.model_service import ModelService
from backend.services.robustness_service import RobustnessService

router = APIRouter(tags=["Robustness"])

_model_service = ModelService()
_audio_service = AudioService()
_inference_service = InferenceService(_model_service, _audio_service)
_robustness_service = RobustnessService(_model_service, _audio_service, _inference_service)


def get_robustness_service():
    return _robustness_service


def get_audio_service():
    return _audio_service


@router.post("/robustness/test", response_model=RobustnessTestResponse)
async def test_robustness(
    file: UploadFile = File(..., description="Audio file to stress test"),
    model: str = Form(default="vit_b16", description="'vit_b16' or 'efficientnet_b0'"),
    robustness_service: RobustnessService = Depends(get_robustness_service),
    audio_service: AudioService = Depends(get_audio_service),
):
    """
    Stress-tests the model against 6 synthetic acoustic corruptions:
    - Additive Gaussian Noise (SNR 20 dB, 10 dB, 5 dB)
    - Volume Gain (+6 dB) and Attenuation (-6 dB)
    - Telephony 8 kHz Bandpass Resampling
    Also returns population-level benchmark statistics from results/robustness_results.json.
    """
    file_bytes = await file.read()
    filename = file.filename or "uploaded_audio.wav"

    try:
        audio_service.validate_upload(filename, file_bytes)
        waveform, metadata = audio_service.load_and_inspect_audio(file_bytes, filename)
        result = robustness_service.test_audio_robustness(
            waveform=waveform,
            audio_metadata=metadata,
            model_id=model,
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing robustness testing: {str(e)}",
        )
