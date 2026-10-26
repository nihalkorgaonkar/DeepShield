"""
Inference Service
Runs neural network inference for single models or dual-model comparisons.
Applies unweighted chunk-probability mean pooling and frozen decision thresholds.
"""

import logging
import time
from typing import Dict, List, Tuple

import torch

from backend.config import DEVICE
from backend.schemas.detection import (
    AudioMetadata,
    ChunkScore,
    DetectionComparisonResult,
    DetectionResult,
)
from backend.services.audio_service import AudioService
from backend.services.model_service import ModelService

logger = logging.getLogger("deepshield.inference")


class InferenceService:
    def __init__(self, model_service: ModelService, audio_service: AudioService):
        self.model_service = model_service
        self.audio_service = audio_service
        self.device = model_service.device

    def run_detection(
        self,
        waveform: torch.Tensor,
        audio_metadata: AudioMetadata,
        model_id: str = "vit_b16",
        custom_threshold: float = None,
    ) -> DetectionResult:
        """
        Runs complete deepfake detection for a single model on an audio waveform.
        """
        norm_id = "efficientnet_b0" if "eff" in model_id.lower() else "vit_b16"
        model = self.model_service.get_model(norm_id)
        if model is None:
            raise RuntimeError(f"Model '{norm_id}' is not loaded or available.")

        # 1. Preprocess and chunk
        chunks, batch_specs = self.audio_service.preprocess_to_chunks(waveform)
        batch_specs = batch_specs.to(self.device)

        # 2. Forward pass with timing
        t0 = time.perf_counter()
        with self.model_service._infer_lock:
            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = model(batch_specs)
                else:
                    logits = model(batch_specs)

                probs = torch.softmax(logits, dim=1)  # (N_chunks, 2)
                # Unweighted arithmetic mean across chunks
                avg_probs = torch.mean(probs, dim=0)
                bonafide_prob = float(avg_probs[0].item())
                spoof_prob = float(avg_probs[1].item())

        inference_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # 3. Decision threshold
        if custom_threshold is not None:
            threshold = float(custom_threshold)
        else:
            threshold = self.model_service.get_frozen_threshold(norm_id)

        is_spoof = spoof_prob >= threshold
        prediction = "spoof" if is_spoof else "bonafide"
        prediction_label = "Synthetic / Spoof Speech" if is_spoof else "Genuine Speech"
        confidence_percent = round((spoof_prob if is_spoof else bonafide_prob) * 100.0, 2)

        # 4. Per-chunk breakdown
        chunk_scores: List[ChunkScore] = []
        hop_sec = self.audio_service.preprocessor.hop_duration
        chunk_len_sec = self.audio_service.preprocessor.chunk_duration

        for i in range(len(chunks)):
            c_start = round(i * hop_sec, 2)
            c_end = round(min(audio_metadata.duration_sec, c_start + chunk_len_sec), 2)
            c_spoof = float(probs[i, 1].item())
            c_bona = float(probs[i, 0].item())
            chunk_scores.append(
                ChunkScore(
                    chunk_index=i + 1,
                    start_time_sec=c_start,
                    end_time_sec=c_end,
                    spoof_score=round(c_spoof, 4),
                    bonafide_score=round(c_bona, 4),
                    is_spoof=c_spoof >= threshold,
                )
            )

        model_name = (
            "EfficientNet-B0 (CNN Baseline)"
            if norm_id == "efficientnet_b0"
            else "Vision Transformer ViT-B/16"
        )

        preprocessing_summary = {
            "sampling_rate": "16,000 Hz (mono)",
            "chunk_window": "4.0s (64,000 samples)",
            "hop_window": "2.0s (50% overlap)",
            "silence_trimming": "30 dB energy envelope",
            "peak_normalization": "Target 0.95 amplitude",
            "spectrogram_dim": "128x128 log-mel (3-channel ImageNet normalized)",
            "score_aggregation": "Unweighted arithmetic mean of chunk posteriors",
        }

        return DetectionResult(
            model_id=norm_id,
            model_name=model_name,
            prediction=prediction,
            prediction_label=prediction_label,
            spoof_score=round(spoof_prob, 6),
            bonafide_score=round(bonafide_prob, 6),
            confidence_percent=confidence_percent,
            configured_threshold=round(threshold, 6),
            is_above_threshold=is_spoof,
            total_inference_time_ms=inference_time_ms,
            num_chunks_analyzed=len(chunks),
            chunk_scores=chunk_scores,
            audio_metadata=audio_metadata,
            preprocessing_summary=preprocessing_summary,
        )

    def run_dual_comparison(
        self,
        waveform: torch.Tensor,
        audio_metadata: AudioMetadata,
        custom_threshold_eff: float = None,
        custom_threshold_vit: float = None,
    ) -> DetectionComparisonResult:
        """
        Runs both EfficientNet-B0 and ViT-B/16 on the exact same audio input
        and returns a side-by-side comparison with agreement analysis.
        """
        eff_result = self.run_detection(
            waveform,
            audio_metadata,
            model_id="efficientnet_b0",
            custom_threshold=custom_threshold_eff,
        )
        vit_result = self.run_detection(
            waveform,
            audio_metadata,
            model_id="vit_b16",
            custom_threshold=custom_threshold_vit,
        )

        agreement = eff_result.prediction == vit_result.prediction
        agreement_status = "AGREEMENT" if agreement else "DISAGREEMENT"

        disagreement_warning = None
        if not agreement:
            disagreement_warning = (
                f"Model Disagreement: EfficientNet-B0 classified this audio as '{eff_result.prediction_label}' "
                f"(score: {eff_result.spoof_score:.4f}, τ: {eff_result.configured_threshold:.4f}), "
                f"while ViT-B/16 classified it as '{vit_result.prediction_label}' "
                f"(score: {vit_result.spoof_score:.4f}, τ: {vit_result.configured_threshold:.4f}). "
                "Per DeepShield research protocol, neither model is arbitrarily favored; "
                "further acoustic and explainability analysis is recommended."
            )

        faster_model = (
            "EfficientNet-B0"
            if eff_result.total_inference_time_ms <= vit_result.total_inference_time_ms
            else "ViT-B/16"
        )
        latency_diff = round(
            abs(eff_result.total_inference_time_ms - vit_result.total_inference_time_ms), 2
        )

        return DetectionComparisonResult(
            efficientnet_result=eff_result,
            vit_result=vit_result,
            agreement=agreement,
            agreement_status=agreement_status,
            disagreement_warning=disagreement_warning,
            faster_model=faster_model,
            latency_difference_ms=latency_diff,
            audio_metadata=audio_metadata,
        )
