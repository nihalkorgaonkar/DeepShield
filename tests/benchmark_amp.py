"""
Benchmark Automatic Mixed Precision (AMP) and Inference on NVIDIA RTX 5050 (8 GB VRAM)
Measures:
- model
- batch_size
- input_resolution
- AMP (Enabled/Disabled)
- GPU_memory_allocated (MB)
- GPU_memory_reserved (MB)
- inference_time (ms per batch)
- samples_per_second
"""

import sys
import os
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.efficientnet_model import EfficientNetDeepShield
from src.models.vit_model import VisionTransformerDeepShield


def benchmark_model_config(model_class, model_kwargs, model_name, input_res=(128, 128), batch_sizes=[8, 16, 32], use_amp=True, num_batches=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"BENCHMARK: {model_name} | Res: {input_res} | AMP: {use_amp} | Device: {device}")
    print(f"=======================================================")

    results = []

    for bs in batch_sizes:
        if not torch.cuda.is_available():
            print("CUDA is not available, skipping.")
            break

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = model_class(**model_kwargs).to(device)
        model.eval()

        dummy_x = torch.randn(bs, 3, input_res[0], input_res[1], device=device)

        # 1. Benchmark Inference (Forward only)
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        _ = model(dummy_x)
                else:
                    _ = model(dummy_x)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_batches):
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        _ = model(dummy_x)
                else:
                    _ = model(dummy_x)
        torch.cuda.synchronize()
        infer_time_total = time.perf_counter() - t0
        infer_time_ms = (infer_time_total / num_batches) * 1000.0
        infer_samples_sec = (bs * num_batches) / infer_time_total

        mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)

        print(f"[Inference] BS: {bs:2d} | Time: {infer_time_ms:6.2f} ms/batch | {infer_samples_sec:6.1f} samples/s | VRAM Alloc: {mem_allocated:6.1f} MB | Reserved: {mem_reserved:6.1f} MB")

        results.append({
            "model": model_name,
            "batch_size": bs,
            "input_resolution": f"{input_res[0]}x{input_res[1]}",
            "AMP": use_amp,
            "GPU_memory_allocated_MB": round(mem_allocated, 2),
            "GPU_memory_reserved_MB": round(mem_reserved, 2),
            "inference_time_ms": round(infer_time_ms, 2),
            "samples_per_second": round(infer_samples_sec, 1)
        })

        del model, dummy_x
        torch.cuda.empty_cache()

    return results


def run_all_benchmarks():
    all_results = []

    # EfficientNet-B0 AMP=True and AMP=False
    for use_amp in [True, False]:
        r = benchmark_model_config(
            model_class=EfficientNetDeepShield,
            model_kwargs={"num_classes": 2, "pretrained": True},
            model_name="EfficientNet-B0",
            input_res=(128, 128),
            batch_sizes=[8, 16, 32],
            use_amp=use_amp,
        )
        all_results.extend(r)

    # ViT-B/16 (128x128) AMP=True and AMP=False
    for use_amp in [True, False]:
        r = benchmark_model_config(
            model_class=VisionTransformerDeepShield,
            model_kwargs={"model_name": "vit_base_patch16_224", "num_classes": 2, "pretrained": True, "img_size": (128, 128)},
            model_name="ViT-B/16",
            input_res=(128, 128),
            batch_sizes=[8, 16, 32],
            use_amp=use_amp,
        )
        all_results.extend(r)

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Model':<16} | {'BS':<4} | {'Res':<8} | {'AMP':<6} | {'VRAM Alloc (MB)':<15} | {'VRAM Res (MB)':<13} | {'Infer Time (ms)':<15} | {'Throughput (img/s)':<18}"
    print(header)
    print("-" * len(header))
    for res in all_results:
        print(f"{res['model']:<16} | {res['batch_size']:<4} | {res['input_resolution']:<8} | {str(res['AMP']):<6} | {res['GPU_memory_allocated_MB']:<15.1f} | {res['GPU_memory_reserved_MB']:<13.1f} | {res['inference_time_ms']:<15.2f} | {res['samples_per_second']:<18.1f}")


if __name__ == "__main__":
    run_all_benchmarks()

