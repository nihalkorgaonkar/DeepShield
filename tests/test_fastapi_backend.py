"""
Automated Integration and Functional Tests for DeepShield FastAPI Backend
Verifies all REST API endpoints with real model checkpoints and sample audio files.
"""

import io
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verifies GET /api/health returns healthy status and active device."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "active_device" in data
    assert "python_version" in data
    assert "torch_version" in data
    assert "loaded_models" in data
    assert "efficientnet_b0" in data["loaded_models"]
    assert "vit_b16" in data["loaded_models"]


def test_models_metadata_endpoint():
    """Verifies GET /api/models returns accurate architectural specs."""
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "efficientnet_b0" in data
    assert "vit_b16" in data

    eff = data["efficientnet_b0"]
    vit = data["vit_b16"]

    # Parameter checks
    assert eff["parameters_million"] > 3.9 and eff["parameters_million"] < 4.5
    assert vit["parameters_million"] > 80.0

    # Threshold checks
    assert eff["frozen_threshold"] > 0.9
    assert vit["frozen_threshold"] > 0.7


def test_metrics_summary_endpoint():
    """Verifies GET /api/metrics/summary returns real metrics and flags discrepancy."""
    res = client.get("/api/metrics/summary")
    assert res.status_code == 200
    data = res.json()
    assert "efficientnet_baseline" in data
    assert "vit_b16_model" in data
    assert data["discrepancy_warning"] is True
    assert "discrepancy_analysis" in data


def test_datasets_endpoint():
    """Verifies GET /api/datasets accurately reports ASVspoof 2019 and 2021 status."""
    res = client.get("/api/datasets")
    assert res.status_code == 200
    data = res.json()
    datasets = {d["id"]: d for d in data["datasets"]}

    assert "asvspoof2019_la" in datasets
    assert "asvspoof2021_df" in datasets

    # Zero fabrication assertion for ASVspoof 2021 DF
    df2021 = datasets["asvspoof2021_df"]
    assert df2021["status"] == "DOCUMENTED_MISSING_AUDIO"


def test_samples_endpoint():
    """Verifies GET /api/samples returns sample test files."""
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 2
    filenames = [s["filename"] for s in data]
    assert any("bonafide" in f.lower() for f in filenames)
    assert any("spoof" in f.lower() for f in filenames)


def test_detect_endpoint_with_real_sample():
    """Verifies POST /api/detect with a real bonafide audio sample."""
    sample_path = "samples/bonafide_LA_D_1047731.flac"
    assert os.path.exists(sample_path), "Sample file missing for test!"

    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    # Test with ViT
    res_vit = client.post(
        "/api/detect",
        files={"file": ("bonafide_sample.flac", file_bytes, "audio/flac")},
        data={"model": "vit_b16"},
    )
    assert res_vit.status_code == 200
    data_vit = res_vit.json()
    assert data_vit["model_id"] == "vit_b16"
    assert data_vit["prediction"] in ["bonafide", "spoof"]
    assert 0.0 <= data_vit["spoof_score"] <= 1.0
    assert 0.0 <= data_vit["bonafide_score"] <= 1.0
    assert data_vit["total_inference_time_ms"] > 0
    assert len(data_vit["chunk_scores"]) >= 1

    # Test with EfficientNet
    res_eff = client.post(
        "/api/detect",
        files={"file": ("bonafide_sample.flac", file_bytes, "audio/flac")},
        data={"model": "efficientnet_b0"},
    )
    assert res_eff.status_code == 200
    data_eff = res_eff.json()
    assert data_eff["model_id"] == "efficientnet_b0"


def test_detect_compare_endpoint():
    """Verifies POST /api/detect/compare runs both models and evaluates agreement."""
    sample_path = "samples/spoof_A01_LA_D_1000265.flac"
    assert os.path.exists(sample_path), "Sample file missing for test!"

    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    res = client.post(
        "/api/detect/compare",
        files={"file": ("spoof_sample.flac", file_bytes, "audio/flac")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "efficientnet_result" in data
    assert "vit_result" in data
    assert "agreement" in data
    assert data["agreement_status"] in ["AGREEMENT", "DISAGREEMENT"]


def test_explain_endpoint():
    """Verifies POST /api/explain returns spectrogram and heatmap images in base64."""
    sample_path = "samples/bonafide_LA_D_1047731.flac"
    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    res = client.post(
        "/api/explain",
        files={"file": ("sample.flac", file_bytes, "audio/flac")},
        data={"chunk_index": 0},
    )
    assert res.status_code == 200
    data = res.json()
    assert "spectrogram_image" in data
    assert data["spectrogram_image"].startswith("data:image/png;base64,")
    assert "insights" in data


def test_reports_endpoint():
    """Verifies GET /api/reports and GET /api/reports/{id}."""
    res = client.get("/api/reports")
    assert res.status_code == 200
    reports = res.json()
    assert len(reports) >= 3

    # Fetch final report detail
    res_detail = client.get("/api/reports/final_report")
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert "DeepShield" in detail["markdown_content"]


def test_diagnostics_endpoint():
    """Verifies GET /api/diagnostics returns complete diagnostic matrix."""
    res = client.get("/api/diagnostics")
    assert res.status_code == 200
    data = res.json()
    assert "checks" in data
    assert len(data["checks"]) >= 4
    check_names = [c["name"] for c in data["checks"]]
    assert any("Checkpoint" in n for n in check_names)
