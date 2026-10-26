"""
DeepShield Flask Backend REST API
Serves audio deepfake detection predictions with chunk aggregation,
dual model selection (EfficientNet-B0 vs ViT), and explainability heatmaps.
"""

import os
import sys
import time
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.audio import AudioPreprocessor
from src.preprocessing.spectrogram import SpectrogramExtractor
from src.models.efficientnet_model import EfficientNetDeepShield
from src.models.vit_model import VisionTransformerDeepShield
from src.evaluation.threshold import ThresholdOptimizer


def create_app():
    app = Flask(__name__)
    CORS(app)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing DeepShield API Backend on device: {device}")

    preprocessor = AudioPreprocessor()
    spec_extractor = SpectrogramExtractor(image_size=(128, 128))
    threshold_optimizer = ThresholdOptimizer(save_path="results/thresholds.json")

    # Load models
    models_dict = {}
    
    # 1. EfficientNet
    eff_model = EfficientNetDeepShield(num_classes=2, pretrained=True).to(device)
    eff_ckpt = "models/checkpoints/best_efficientnet_b0.pt"
    if os.path.exists(eff_ckpt):
        try:
            ckpt = torch.load(eff_ckpt, map_location=device)
            eff_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        except Exception as e:
            print(f"Warning: Could not load EfficientNet checkpoint ({e}), using initialized weights.")
    eff_model.eval()
    models_dict["efficientnet"] = eff_model

    # 2. Vision Transformer
    vit_model = VisionTransformerDeepShield(model_name="vit_base_patch16_224", num_classes=2, pretrained=True, img_size=(128, 128)).to(device)
    vit_ckpt = "models/checkpoints/best_vit_b16.pt"
    if os.path.exists(vit_ckpt):
        try:
            ckpt = torch.load(vit_ckpt, map_location=device)
            vit_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        except Exception as e:
            print(f"Warning: Could not load ViT checkpoint ({e}), using initialized weights.")
    vit_model.eval()
    models_dict["vit"] = vit_model

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "device": str(device),
            "available_models": list(models_dict.keys()),
            "default_model": "vit",
        })

    @app.route("/api/predict", methods=["POST"])
    def predict():
        t0 = time.perf_counter()
        
        if "audio" not in request.files and "file" not in request.files:
            return jsonify({"error": "No audio file provided in request."}), 400

        audio_file = request.files.get("audio") or request.files.get("file")
        model_choice = request.form.get("model", "vit").lower()
        if model_choice not in models_dict:
            model_choice = "vit"

        # Save to temp file
        suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            # 1. Load and preprocess audio
            waveform = preprocessor.load_audio(tmp_path)
            processed_wf = preprocessor.process_waveform(waveform)
            chunks = preprocessor.chunk_waveform(processed_wf)

            if len(chunks) == 0:
                return jsonify({"error": "Audio could not be processed into valid chunks."}), 400

            # 2. Extract Spectrograms for each chunk
            spec_tensors = [spec_extractor(chunk) for chunk in chunks]
            batch_spec = torch.stack(spec_tensors).to(device)  # (N_chunks, 3, 128, 128)

            # 3. Model Inference
            selected_model = models_dict[model_choice]
            with torch.no_grad():
                if device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = selected_model(batch_spec)
                else:
                    logits = selected_model(batch_spec)

                probs = torch.softmax(logits, dim=1)  # (N_chunks, 2)
                # Chunk-level probability averaging
                avg_probs = torch.mean(probs, dim=0)  # (2,)
                bonafide_prob = float(avg_probs[0].item())
                spoof_prob = float(avg_probs[1].item())

            # 4. Decision Threshold
            model_key = "efficientnet_b0" if model_choice == "efficientnet" else "vit_b16"
            threshold = threshold_optimizer.load_threshold(model_name=model_key, default=0.5)

            prediction = "spoof" if spoof_prob >= threshold else "bonafide"
            confidence = spoof_prob if prediction == "spoof" else bonafide_prob

            processing_time_ms = round((time.perf_counter() - t0) * 1000.0, 1)

            return jsonify({
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "genuine_probability": round(bonafide_prob, 4),
                "spoof_probability": round(spoof_prob, 4),
                "decision_threshold": round(threshold, 4),
                "model": model_key,
                "processing_time_ms": processing_time_ms,
                "chunks_analyzed": len(chunks),
            })

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
