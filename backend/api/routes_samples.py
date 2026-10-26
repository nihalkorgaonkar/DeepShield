"""
Preloaded Audio Samples API Routes
Provides immediate sample audio files (bonafide and spoof) for fast 1-click verification.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from backend.config import SAMPLES_DIR

router = APIRouter(tags=["Audio Samples"])


@router.get("/samples")
def list_samples() -> List[Dict[str, Any]]:
    """Returns list of preloaded audio samples available for immediate testing."""
    samples = []
    if SAMPLES_DIR.exists():
        for file in SAMPLES_DIR.glob("*.flac"):
            is_bonafide = "bonafide" in file.name.lower()
            samples.append({
                "filename": file.name,
                "label": "bonafide" if is_bonafide else "spoof",
                "label_display": "Genuine / Bona Fide (Authentic Human Voice)" if is_bonafide else "Spoof / Synthetic Speech (TTS A01)",
                "size_bytes": file.stat().st_size,
                "url": f"/api/samples/{file.name}",
            })
    return sorted(samples, key=lambda s: s["label"])


@router.get("/samples/{filename}")
def get_sample_audio(filename: str):
    """Streams a sample audio file."""
    # Prevent directory traversal
    clean_name = os.path.basename(filename)
    file_path = SAMPLES_DIR / clean_name

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample '{clean_name}' not found.",
        )

    return FileResponse(path=str(file_path), media_type="audio/flac", filename=clean_name)
