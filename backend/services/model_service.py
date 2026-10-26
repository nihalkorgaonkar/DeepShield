"""
Model Management Service
Thread-safe singleton holding EfficientNet-B0 and ViT-B/16 PyTorch models in memory.
Extracts model metadata, parameters, and verifies SHA-256 hashes.
"""

import hashlib
import json
import logging
import os
import threading
from typing import Dict, Optional, Tuple

import torch

from backend.config import (
    DEVICE,
    EFFICIENTNET_CHECKPOINT,
    IMAGE_SIZE,
    THRESHOLDS_JSON,
    THRESHOLDS_VIT_JSON,
    VIT_CHECKPOINT,
)
from backend.schemas.models import CheckpointMetadata, ModelInfo
from src.models.efficientnet_model import EfficientNetDeepShield
from src.models.vit_model import VisionTransformerDeepShield

logger = logging.getLogger("deepshield.models")


class ModelService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.device = DEVICE
        self.models: Dict[str, torch.nn.Module] = {}
        self.metadata: Dict[str, CheckpointMetadata] = {}
        self.load_errors: Dict[str, str] = {}
        self._infer_lock = threading.Lock()

        logger.info(f"Initializing ModelService on target device: {self.device}")
        self._load_all_models()
        self._initialized = True

    _sha_cache: Dict[str, Tuple[float, str]] = {}

    @classmethod
    def compute_sha256(cls, filepath: str) -> str:
        """Calculates cryptographic SHA-256 hash of a checkpoint file with mtime caching."""
        if not os.path.exists(filepath):
            return ""
        mtime = os.path.getmtime(filepath)
        if filepath in cls._sha_cache and cls._sha_cache[filepath][0] == mtime:
            return cls._sha_cache[filepath][1]
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(1048576):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        cls._sha_cache[filepath] = (mtime, digest)
        return digest

    def _load_all_models(self):
        # 1. EfficientNet-B0
        self._load_efficientnet()
        # 2. Vision Transformer ViT-B/16
        self._load_vit()

    def _load_efficientnet(self):
        model_id = "efficientnet_b0"
        ckpt_path = str(EFFICIENTNET_CHECKPOINT)

        try:
            model = EfficientNetDeepShield(num_classes=2, pretrained=False).to(self.device)
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict)
                model.eval()
                self.models[model_id] = model

                file_size = os.path.getsize(ckpt_path)
                sha = self.compute_sha256(ckpt_path)
                mtime = os.path.getmtime(ckpt_path)

                self.metadata[model_id] = CheckpointMetadata(
                    model_name=model_id,
                    exists=True,
                    path=ckpt_path,
                    size_bytes=file_size,
                    size_mb=round(file_size / (1024 * 1024), 2),
                    sha256_hash=sha,
                    epoch=ckpt.get("epoch") if isinstance(ckpt, dict) else None,
                    dev_eer=float(ckpt.get("dev_eer")) if isinstance(ckpt, dict) and "dev_eer" in ckpt else None,
                    dev_threshold=float(ckpt.get("dev_threshold")) if isinstance(ckpt, dict) and "dev_threshold" in ckpt else None,
                    seed=ckpt.get("seed") if isinstance(ckpt, dict) else None,
                    last_modified=str(mtime),
                )
                logger.info(f"Loaded EfficientNet-B0 checkpoint successfully (SHA256: {sha[:12]}...)")
            else:
                self.load_errors[model_id] = f"Checkpoint file missing at {ckpt_path}"
                logger.warning(self.load_errors[model_id])
        except Exception as e:
            self.load_errors[model_id] = str(e)
            logger.error(f"Failed to load EfficientNet-B0 checkpoint: {e}", exc_info=True)

    def _load_vit(self):
        model_id = "vit_b16"
        ckpt_path = str(VIT_CHECKPOINT)

        try:
            model = VisionTransformerDeepShield(
                num_classes=2,
                pretrained=False,
                img_size=IMAGE_SIZE,
            ).to(self.device)

            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict)
                model.eval()
                self.models[model_id] = model

                file_size = os.path.getsize(ckpt_path)
                sha = self.compute_sha256(ckpt_path)
                mtime = os.path.getmtime(ckpt_path)

                self.metadata[model_id] = CheckpointMetadata(
                    model_name=model_id,
                    exists=True,
                    path=ckpt_path,
                    size_bytes=file_size,
                    size_mb=round(file_size / (1024 * 1024), 2),
                    sha256_hash=sha,
                    epoch=ckpt.get("epoch") if isinstance(ckpt, dict) else None,
                    dev_eer=float(ckpt.get("dev_eer")) if isinstance(ckpt, dict) and "dev_eer" in ckpt else None,
                    dev_threshold=float(ckpt.get("dev_threshold")) if isinstance(ckpt, dict) and "dev_threshold" in ckpt else None,
                    seed=ckpt.get("seed") if isinstance(ckpt, dict) else None,
                    last_modified=str(mtime),
                )
                logger.info(f"Loaded ViT-B/16 checkpoint successfully (SHA256: {sha[:12]}...)")
            else:
                self.load_errors[model_id] = f"Checkpoint file missing at {ckpt_path}"
                logger.warning(self.load_errors[model_id])
        except Exception as e:
            self.load_errors[model_id] = str(e)
            logger.error(f"Failed to load ViT-B/16 checkpoint: {e}", exc_info=True)

    def get_model(self, model_id: str) -> Optional[torch.nn.Module]:
        norm_id = "efficientnet_b0" if "eff" in model_id.lower() else "vit_b16"
        return self.models.get(norm_id)

    def get_frozen_threshold(self, model_id: str) -> float:
        norm_id = "efficientnet_b0" if "eff" in model_id.lower() else "vit_b16"

        # Check thresholds_vit.json first for ViT if available
        if norm_id == "vit_b16" and os.path.exists(THRESHOLDS_VIT_JSON):
            try:
                with open(THRESHOLDS_VIT_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "vit_b16" in data and "threshold" in data["vit_b16"]:
                        return float(data["vit_b16"]["threshold"])
            except Exception:
                pass

        # Fallback to thresholds.json
        if os.path.exists(THRESHOLDS_JSON):
            try:
                with open(THRESHOLDS_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if norm_id in data and "threshold" in data[norm_id]:
                        return float(data[norm_id]["threshold"])
            except Exception:
                pass

        # Check metadata from checkpoint
        if norm_id in self.metadata and self.metadata[norm_id].dev_threshold is not None:
            return float(self.metadata[norm_id].dev_threshold)

        return 0.5

    def get_models_info(self) -> Dict[str, ModelInfo]:
        info_dict = {}

        # EfficientNet-B0
        eff_meta = self.metadata.get(
            "efficientnet_b0",
            CheckpointMetadata(model_name="efficientnet_b0", exists=False, path=str(EFFICIENTNET_CHECKPOINT)),
        )
        eff_model = self.models.get("efficientnet_b0")
        eff_params = sum(p.numel() for p in eff_model.parameters()) if eff_model else 4010110

        info_dict["efficientnet_b0"] = ModelInfo(
            id="efficientnet_b0",
            name="EfficientNet-B0 (CNN Baseline)",
            architecture="EfficientNet-B0",
            family="Convolutional Neural Network",
            parameters=eff_params,
            parameters_million=round(eff_params / 1_000_000, 2),
            input_resolution="128x128x3",
            feature_extractor="MBConv blocks with Squeeze-and-Excitation",
            classification_head="Linear(1280, 2)",
            frozen_threshold=self.get_frozen_threshold("efficientnet_b0"),
            threshold_criterion="min_eer on Dev",
            dev_eer=eff_meta.dev_eer if eff_meta.dev_eer is not None else 0.0644,
            eval_eer=0.2149,
            eval_roc_auc=0.8682,
            avg_inference_latency_ms=1.22,
            checkpoint=eff_meta,
        )

        # ViT-B/16
        vit_meta = self.metadata.get(
            "vit_b16",
            CheckpointMetadata(model_name="vit_b16", exists=False, path=str(VIT_CHECKPOINT)),
        )
        vit_model = self.models.get("vit_b16")
        vit_params = sum(p.numel() for p in vit_model.parameters()) if vit_model else 85799426

        info_dict["vit_b16"] = ModelInfo(
            id="vit_b16",
            name="Vision Transformer ViT-B/16",
            architecture="ViT-B/16 with 2D Bicubic Pos-Embed Interpolation",
            family="Vision Transformer",
            parameters=vit_params,
            parameters_million=round(vit_params / 1_000_000, 1),
            input_resolution="128x128x3 (8x8 patch grid, patch size 16)",
            feature_extractor="12 Transformer Encoder Layers (12 heads, 768 dim)",
            classification_head="Linear(768, 2)",
            frozen_threshold=self.get_frozen_threshold("vit_b16"),
            threshold_criterion="min_eer on Dev",
            dev_eer=vit_meta.dev_eer if vit_meta.dev_eer is not None else 0.0155,
            eval_eer=0.2708,
            eval_roc_auc=0.8201,
            avg_inference_latency_ms=1.18,
            checkpoint=vit_meta,
        )

        return info_dict
