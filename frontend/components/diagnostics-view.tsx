"use client";

import React, { useEffect, useState } from "react";
import {
  Cpu,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Info,
  RefreshCw,
  Zap,
  HardDrive,
  ShieldCheck,
  Terminal,
} from "lucide-react";
import { fetchDiagnostics, FullDiagnostics } from "@/lib/api";

export function DiagnosticsView() {
  const [diagnostics, setDiagnostics] = useState<FullDiagnostics | null>(null);
  const [loading, setLoading] = useState(true);
  const [testingInference, setTestingInference] = useState(false);
  const [inferenceTestResult, setInferenceTestResult] = useState<any | null>(null);

  const loadDiagnostics = () => {
    setLoading(true);
    fetchDiagnostics()
      .then(setDiagnostics)
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadDiagnostics();
  }, []);

  const handleTestInference = async () => {
    try {
      setTestingInference(true);
      setInferenceTestResult(null);

      // Generate a 1-second synthetic 440 Hz WAV tone in browser
      const sampleRate = 16000;
      const numSamples = sampleRate * 1;
      const buffer = new Float32Array(numSamples);
      for (let i = 0; i < numSamples; i++) {
        buffer[i] = 0.5 * Math.sin((2 * Math.PI * 440 * i) / sampleRate);
      }

      // Encode as WAV
      const wavBytes = encodeWav(buffer, sampleRate);
      const blob = new Blob([wavBytes], { type: "audio/wav" });
      const file = new File([blob], "synthetic_diagnostic_tone.wav", { type: "audio/wav" });

      const form = new FormData();
      form.append("file", file);
      form.append("model", "vit_b16");

      const t0 = performance.now();
      const res = await fetch("/api/detect", { method: "POST", body: form });
      const latency = Math.round(performance.now() - t0);
      const data = await res.json();

      setInferenceTestResult({
        status: res.ok ? "SUCCESS" : "ERROR",
        totalRoundtripMs: latency,
        inferenceTimeMs: data.total_inference_time_ms,
        model: data.model_id,
        prediction: data.prediction,
        confidence: data.confidence_percent,
        spoofProb: (data.spoof_score * 100).toFixed(2),
      });
    } catch (err: any) {
      setInferenceTestResult({ status: "ERROR", message: err.message });
    } finally {
      setTestingInference(false);
    }
  };

  // Minimal WAV encoder helper
  const encodeWav = (samples: Float32Array, sampleRate: number) => {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeString(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, samples.length * 2, true);
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return buffer;
  };

  if (loading && !diagnostics) {
    return (
      <div className="flex items-center justify-center min-h-[350px] text-slate-400">
        <div className="w-8 h-8 border-2 border-sky-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const sys = diagnostics?.system_health;

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Action Header */}
      <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
            <Cpu className="w-5 h-5 text-sky-400" />
            System Diagnostics & Execution Health
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Real-time verification of GPU tensors, CUDA driver, PyTorch state dicts, and file caches.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={loadDiagnostics}
            disabled={loading}
            className="px-4 py-2 rounded-xl bg-slate-900 border border-slate-800 hover:border-slate-700 text-xs font-semibold text-slate-300 hover:text-white flex items-center gap-2 transition-all"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-sky-400" : ""}`} />
            Run Diagnostics
          </button>

          <button
            onClick={handleTestInference}
            disabled={testingInference}
            className="px-4 py-2 rounded-xl bg-sky-500 hover:bg-sky-400 text-slate-950 text-xs font-bold flex items-center gap-2 transition-all shadow-sm"
          >
            <Zap className="w-3.5 h-3.5 fill-current" />
            {testingInference ? "Testing..." : "Test Model Forward Pass"}
          </button>
        </div>
      </div>

      {/* Synthetic Inference Test Output */}
      {inferenceTestResult && (
        <div className="p-5 rounded-2xl bg-[#0f172a] border border-sky-500/40 space-y-3">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <span className="text-xs font-bold text-sky-300 uppercase tracking-wider flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-sky-400" /> Live Forward Pass Latency Test
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
              {inferenceTestResult.status}
            </span>
          </div>

          <p className="text-xs text-slate-400 leading-snug">
            Synthetic 440 Hz sinusoidal waveform (1.0s) was transmitted to the backend, preprocessed to 128×128 log-mel spectrogram, and evaluated through {inferenceTestResult.model}.
            <span className="text-amber-400 block mt-1 text-[11px]">
              Note: Synthetic tones measure compute latency only and do not validate detection accuracy.
            </span>
          </p>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono pt-1">
            <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
              <span className="text-slate-400 block text-[10px]">PyTorch Forward Time</span>
              <span className="text-white font-bold">{inferenceTestResult.inferenceTimeMs} ms</span>
            </div>
            <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
              <span className="text-slate-400 block text-[10px]">HTTP Total Roundtrip</span>
              <span className="text-white font-bold">{inferenceTestResult.totalRoundtripMs} ms</span>
            </div>
            <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
              <span className="text-slate-400 block text-[10px]">Spoof Probability</span>
              <span className="text-white font-bold">{inferenceTestResult.spoofProb}%</span>
            </div>
            <div className="p-2.5 rounded bg-slate-900 border border-slate-800">
              <span className="text-slate-400 block text-[10px]">Verdict</span>
              <span className="text-sky-300 font-bold uppercase">{inferenceTestResult.prediction}</span>
            </div>
          </div>
        </div>
      )}

      {/* System Telemetry Specs */}
      {sys && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-[#0f172a] border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-mono">Python Runtime</span>
            <span className="text-base font-bold text-white font-mono block">{sys.python_version}</span>
            <span className="text-[10px] text-slate-400 font-mono">PyTorch {sys.torch_version}</span>
          </div>

          <div className="p-4 rounded-xl bg-[#0f172a] border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-mono">CUDA Hardware</span>
            <span className="text-base font-bold text-white font-mono block">
              {sys.cuda_available ? "NVIDIA RTX 5050" : "Disabled"}
            </span>
            <span className="text-[10px] text-sky-400 font-mono">
              {sys.cuda_available ? `${sys.vram_allocated_gb} / ${sys.vram_total_gb} GB VRAM` : "CPU Mode"}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-[#0f172a] border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-mono">Loaded Weights</span>
            <span className="text-base font-bold text-white font-mono block">
              {sys.loaded_models.length} Model Checkpoints
            </span>
            <span className="text-[10px] text-emerald-400 font-mono">EfficientNet-B0 + ViT</span>
          </div>

          <div className="p-4 rounded-xl bg-[#0f172a] border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-mono">Dataset Status</span>
            <span className="text-base font-bold text-white font-mono block">{sys.dataset_status}</span>
            <span className="text-[10px] text-purple-400 font-mono">{sys.preprocessed_cache_status}</span>
          </div>
        </div>
      )}

      {/* Diagnostic Checks Matrix */}
      {diagnostics?.checks && (
        <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <h4 className="font-semibold text-white text-sm">Automated Diagnostic Check Results</h4>
            <p className="text-xs text-slate-400">Continuous sanity checks covering models, storage, and protocols</p>
          </div>

          <div className="space-y-3">
            {diagnostics.checks.map((c, idx) => (
              <div
                key={idx}
                className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 flex items-start justify-between gap-4"
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-white">{c.name}</span>
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-slate-400 border border-slate-700">
                      {c.category}
                    </span>
                  </div>
                  <p className="text-xs text-slate-300 leading-snug">{c.message}</p>
                </div>

                <div className="shrink-0">
                  {c.status === "PASS" && (
                    <span className="text-[11px] font-mono font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" /> PASS
                    </span>
                  )}
                  {c.status === "WARN" && (
                    <span className="text-[11px] font-mono font-bold px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30 flex items-center gap-1">
                      <AlertTriangle className="w-3 h-3" /> WARN
                    </span>
                  )}
                  {c.status === "FAIL" && (
                    <span className="text-[11px] font-mono font-bold px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30 flex items-center gap-1">
                      <XCircle className="w-3 h-3" /> FAIL
                    </span>
                  )}
                  {c.status === "INFO" && (
                    <span className="text-[11px] font-mono font-bold px-2 py-0.5 rounded bg-sky-500/20 text-sky-300 border border-sky-500/30 flex items-center gap-1">
                      <Info className="w-3 h-3" /> INFO
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
