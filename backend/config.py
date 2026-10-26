"""
DeepShield Backend Configuration
Centralizes configuration for model paths, dataset roots, thresholds, and server settings.
"""

from pathlib import Path
import os
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
DOCS_DIR = BASE_DIR / "docs"
REPORTS_DIR = BASE_DIR / "reports"
SAMPLES_DIR = BASE_DIR / "samples"
EVAL_RUNS_DIR = RESULTS_DIR / "eval_runs"

# Ensure runtime directories exist
EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Dataset paths
DEFAULT_DATA_ROOT = Path("C:/DeepShieldData/ASVspoof2019_LA")
DEFAULT_CACHE_DIR = Path("C:/DeepShieldData/cache_spec")
DEFAULT_DF2021_ROOT = Path("C:/DeepShieldData/ASVspoof2021_DF")

# Checkpoint paths
EFFICIENTNET_CHECKPOINT = CHECKPOINTS_DIR / "best_efficientnet_b0.pt"
VIT_CHECKPOINT = CHECKPOINTS_DIR / "best_vit_b16.pt"

# Threshold files
THRESHOLDS_JSON = RESULTS_DIR / "thresholds.json"
THRESHOLDS_VIT_JSON = RESULTS_DIR / "thresholds_vit.json"

# Metric files
MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
MODEL_COMPARISON_VIT_CSV = RESULTS_DIR / "model_comparison_vit.csv"
ROBUSTNESS_RESULTS_JSON = RESULTS_DIR / "robustness_results.json"
GENERALIZATION_STATUS_JSON = RESULTS_DIR / "generalization_2021_df_status.json"
TRAINING_HISTORY_EFF = RESULTS_DIR / "training_history_efficientnet_b0.json"
TRAINING_HISTORY_VIT = RESULTS_DIR / "training_history_vit_b16.json"

# Audio preprocessing constants (strictly preserved 14-point protocol)
SAMPLE_RATE = 16000
CHUNK_DURATION = 4.0
HOP_DURATION = 2.0
TRIM_SILENCE = True
TOP_DB = 30.0
NORMALIZE_PEAK = True
TARGET_PEAK = 0.95
IMAGE_SIZE = (128, 128)

# Hardware device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Security and limits
MAX_UPLOAD_SIZE_BYTES = 30 * 1024 * 1024  # 30 MB
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# Server settings
HOST = os.getenv("DEEPSHIELD_HOST", "127.0.0.1")
PORT = int(os.getenv("DEEPSHIELD_PORT", "8000"))
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
