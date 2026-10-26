"""
Robustness Service
Applies real acoustic perturbations to uploaded audio and measures model degradation.
Also provides population-level benchmark results loaded directly from results/robustness_results.json.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from backend.config import ROBUSTNESS_RESULTS_JSON
from backend.schemas.detection import AudioMetadata
from backend.schemas.robustness import PerturbationResult, RobustnessTestResponse
from backend.services.audio_service import AudioService
from backend.services.inference_service import InferenceService
from backend.services.model_service import ModelService
from src.evaluation.robustness import AudioCorruptor

logger = logging.getLogger("deepshield.robustness")


class RobustnessService:
    def __init__(
        self,
        model_service: ModelService,
        audio_service: AudioService,
        inference_service: InferenceService,
    ):
        self.model_service = model_service
        self.audio_service = audio_service
        self.inference_service = inference_service

    @staticmethod
    def load_benchmark_results() -> Dict[str, Any]:
        """Loads recorded population robustness benchmarks from results/robustness_results.json."""
        if os.path.exists(ROBUSTNESS_RESULTS_JSON):
            try:
                with open(ROBUSTNESS_RESULTS_JSON, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read robustness_results.json: {e}")
        return {}

    def test_audio_robustness(
        self,
        waveform: torch.Tensor,
        audio_metadata: AudioMetadata,
        model_id: str = "vit_b16",
    ) -> RobustnessTestResponse:
        """
        Runs the clean audio through the model, then applies 6 acoustic perturbations
        and re-evaluates the model on each condition.
        """
        norm_id = "efficientnet_b0" if "eff" in model_id.lower() else "vit_b16"

        # 1. Clean reference prediction
        clean_result = self.inference_service.run_detection(
            waveform, audio_metadata, model_id=norm_id
        )
        clean_score = clean_result.spoof_score
        clean_pred = clean_result.prediction

        # Perturbation definitions
        perturbation_configs = [
            (
                "Noise (SNR 20dB)",
                "Mild additive white Gaussian noise simulating good ambient indoor conditions.",
                "SNR = 20 dB",
                lambda wf: AudioCorruptor.add_gaussian_noise(wf, snr_db=20.0),
            ),
            (
                "Noise (SNR 10dB)",
                "Moderate additive white Gaussian noise simulating outdoor urban traffic or air conditioning.",
                "SNR = 10 dB",
                lambda wf: AudioCorruptor.add_gaussian_noise(wf, snr_db=10.0),
            ),
            (
                "Noise (SNR 5dB)",
                "Severe acoustic noise simulating harsh environmental conditions with high noise floor.",
                "SNR = 5 dB",
                lambda wf: AudioCorruptor.add_gaussian_noise(wf, snr_db=5.0),
            ),
            (
                "Volume Boost (+6dB)",
                "Acoustic gain amplification doubling amplitude (risk of non-linear clipping).",
                "Gain = +6.0 dB",
                lambda wf: AudioCorruptor.scale_volume(wf, gain_db=6.0),
            ),
            (
                "Volume Attenuation (-6dB)",
                "Signal attenuation halving signal amplitude toward background floor.",
                "Gain = -6.0 dB",
                lambda wf: AudioCorruptor.scale_volume(wf, gain_db=-6.0),
            ),
            (
                "Telephony (8kHz Resample)",
                "PSTN / mobile telephony bandpass simulation restricting frequency content to [0, 4kHz].",
                "Downsample to 8,000 Hz, upsample to 16,000 Hz",
                lambda wf: AudioCorruptor.resample_telephony(wf, orig_sr=16000, low_sr=8000),
            ),
        ]

        perturbation_results: List[PerturbationResult] = []

        for name, desc, param_desc, perturb_fn in perturbation_configs:
            try:
                corrupted_wf = perturb_fn(waveform.clone())
                # Run through detection
                res = self.inference_service.run_detection(
                    corrupted_wf, audio_metadata, model_id=norm_id
                )
                delta = round(res.spoof_score - clean_score, 4)
                downsampled_envelope = self.audio_service._compute_waveform_envelope(
                    corrupted_wf, num_points=60
                )

                perturbation_results.append(
                    PerturbationResult(
                        condition_name=name,
                        description=desc,
                        parameter_description=param_desc,
                        spoof_score=res.spoof_score,
                        bonafide_score=res.bonafide_score,
                        prediction=res.prediction,
                        score_delta_from_clean=delta,
                        waveform_downsampled=downsampled_envelope,
                    )
                )
            except Exception as e:
                logger.warning(f"Perturbation {name} evaluation failed: {e}")

        # Load population benchmark results
        benchmarks = self.load_benchmark_results()
        model_key = "ViT-B/16" if norm_id == "vit_b16" else "EfficientNet-B0"
        bench_summary = benchmarks.get(model_key, {})

        return RobustnessTestResponse(
            filename=audio_metadata.filename,
            model_id=norm_id,
            clean_score=clean_score,
            clean_prediction=clean_pred,
            perturbations=perturbation_results,
            benchmark_dataset_summary=bench_summary,
        )
