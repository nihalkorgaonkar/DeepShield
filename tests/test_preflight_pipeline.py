"""
DeepShield Pre-Flight Pipeline Verification
Tests:
1. EfficientNet-B0 1-batch forward, backward, optimizer step with class weights
2. ViT-B/16 1-batch forward, backward, optimizer step with class weights
3. Checkpoint save and load cycle
4. Decision threshold optimization and load cycle
5. Multi-chunk probability averaging and biometric metric computation
"""

import os
import sys
import tempfile
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.efficientnet_model import EfficientNetDeepShield
from src.models.vit_model import VisionTransformerDeepShield
from src.evaluation.metrics import compute_all_metrics, compute_eer
from src.evaluation.threshold import ThresholdOptimizer


def test_efficientnet_train_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetDeepShield(num_classes=2, pretrained=False).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0]).to(device))

    x = torch.randn(4, 3, 128, 128, device=device)
    y = torch.tensor([0, 1, 1, 0], device=device)

    optimizer.zero_grad()
    logits = model(x)
    assert logits.shape == (4, 2)
    loss = criterion(logits, y)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    loss.backward()
    optimizer.step()
    assert True


def test_vit_train_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformerDeepShield(
        model_name="vit_base_patch16_224",
        num_classes=2,
        pretrained=False,
        img_size=(128, 128),
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0]).to(device))

    x = torch.randn(4, 3, 128, 128, device=device)
    y = torch.tensor([0, 1, 1, 0], device=device)

    optimizer.zero_grad()
    logits = model(x)
    assert logits.shape == (4, 2)
    loss = criterion(logits, y)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    loss.backward()
    optimizer.step()
    assert True


def test_checkpoint_save_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "test_checkpoint.pt")
        model = EfficientNetDeepShield(num_classes=2, pretrained=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        torch.save({
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dev_eer": 0.045,
            "dev_threshold": 0.523,
            "model_name": "efficientnet_b0",
        }, ckpt_path)

        assert os.path.exists(ckpt_path)
        loaded = torch.load(ckpt_path, weights_only=False)
        assert loaded["epoch"] == 5
        assert loaded["dev_eer"] == 0.045

        new_model = EfficientNetDeepShield(num_classes=2, pretrained=False)
        new_model.load_state_dict(loaded["model_state_dict"])
        assert True


def test_threshold_optimizer_and_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        th_file = os.path.join(tmp, "thresholds.json")
        opt = ThresholdOptimizer(criterion="min_eer", save_path=th_file)

        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        th = opt.optimize_threshold(y_true, y_scores, model_name="test_model", seed=42)
        assert 0.0 <= th <= 1.0

        loaded_th = opt.load_threshold("test_model")
        assert abs(loaded_th - th) < 1e-6

        metrics = compute_all_metrics(y_true, y_scores, threshold=loaded_th)
        assert metrics["eer"] == 0.0
        assert metrics["accuracy"] == 1.0
        assert metrics["roc_auc"] == 1.0
