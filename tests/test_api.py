"""
Test Flask Backend REST API
Verifies:
1. GET /api/health endpoint
2. POST /api/predict endpoint with synthetic audio
3. Response JSON structure and probability bounds
"""

import sys
import os
import io
import torch
import torchaudio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.app import create_app


def test_api_endpoints():
    print("=" * 60)
    print("RUNNING API BACKEND VERIFICATION")
    print("=" * 60)

    app = create_app()
    client = app.test_client()

    # 1. Health check
    res_health = client.get("/api/health")
    assert res_health.status_code == 200, f"Health check failed with {res_health.status_code}"
    health_data = res_health.get_json()
    print(f"Health Response: {health_data}")
    assert health_data["status"] == "healthy"

    # 2. Predict endpoint with synthetic audio
    # Generate 4-second WAV buffer
    sr = 16000
    t = torch.linspace(0, 4.0, int(sr * 4.0))
    wav_tensor = (0.5 * torch.sin(2 * torch.pi * 440 * t)).unsqueeze(0)
    
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, wav_tensor.squeeze().numpy(), sr, format="WAV")
    buf.seek(0)

    data = {
        "file": (buf, "test_sample.wav"),
        "model": "vit",
    }

    res_predict = client.post(
        "/api/predict",
        data=data,
        content_type="multipart/form-data",
    )
    assert res_predict.status_code == 200, f"Predict failed with {res_predict.status_code}: {res_predict.get_data(as_text=True)}"
    pred_data = res_predict.get_json()
    print(f"Predict Response: {pred_data}")

    # Assert required schema
    for key in ["prediction", "confidence", "genuine_probability", "spoof_probability", "model", "processing_time_ms", "chunks_analyzed"]:
        assert key in pred_data, f"Missing key '{key}' in prediction response!"

    assert pred_data["prediction"] in ["bonafide", "spoof"]
    assert 0.0 <= pred_data["confidence"] <= 1.0
    assert 0.0 <= pred_data["genuine_probability"] <= 1.0
    assert 0.0 <= pred_data["spoof_probability"] <= 1.0
    assert abs((pred_data["genuine_probability"] + pred_data["spoof_probability"]) - 1.0) < 1e-3

    print("PASS: Flask API endpoints and response schema verified successfully!")
    return True


if __name__ == "__main__":
    test_api_endpoints()
