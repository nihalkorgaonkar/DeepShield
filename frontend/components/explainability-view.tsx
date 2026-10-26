"use client";

import React, { useState, useEffect } from "react";
import {
  Layers,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Info,
  Maximize2,
} from "lucide-react";
import {
  explainAudio,
  fetchSamples,
  AudioSample,
  ExplainResult,
} from "@/lib/api";

export function ExplainabilityView() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [chunkIndex, setChunkIndex] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ExplainResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [samples, setSamples] = useState<AudioSample[]>([]);

  useEffect(() => {
    fetchSamples()
      .then(setSamples)
      .catch(() => {});
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleLoadSample = async (sample: AudioSample) => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(sample.url);
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: "audio/flac" });
      setSelectedFile(file);
      setResult(null);
    } catch (err: any) {
      setError(err.message || "Failed to load sample audio.");
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateExplanations = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await explainAudio(selectedFile, selectedFile.name, chunkIndex);
      setResult(res);
    } catch (err: any) {
      setError(err.message || "Failed to generate visual saliency explanations.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Header Card */}
      <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-6">
        <div>
          <h3 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
            <Layers className="w-5 h-5 text-sky-400" />
            Explainability & Acoustic Saliency Workbench
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Compare Convolutional gradient activations (Grad-CAM) against Vision Transformer self-attention saliency maps on identical 128×128 log-mel spectrogram features.
          </p>
        </div>

        {/* Quick Sample Selector */}
        <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-slate-800/80">
          <span className="text-xs text-slate-400 font-semibold mr-2">Load sample audio:</span>
          {samples.map((s) => (
            <button
              key={s.filename}
              onClick={() => handleLoadSample(s)}
              disabled={loading}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all flex items-center gap-1.5 ${
                s.label === "bonafide"
                  ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/20"
                  : "bg-rose-500/10 border-rose-500/30 text-rose-300 hover:bg-rose-500/20"
              }`}
            >
              {s.label === "bonafide" ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              ) : (
                <XCircle className="w-3.5 h-3.5 text-rose-400" />
              )}
              {s.label === "bonafide" ? "Load Genuine Speech" : "Load Synthetic Speech (TTS A01)"}
            </button>
          ))}
        </div>

        {/* File and Chunk Selectors */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs pt-1">
          <div className="sm:col-span-2">
            <label className="text-slate-400 font-semibold block mb-1.5 uppercase tracking-wider text-[11px]">
              Audio Recording
            </label>
            <div className="flex items-center gap-3">
              <input
                type="file"
                onChange={handleFileChange}
                accept=".wav,.flac,.mp3"
                className="block w-full text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700"
              />
              {selectedFile && (
                <span className="text-xs font-mono text-sky-400 truncate max-w-xs shrink-0">
                  {selectedFile.name}
                </span>
              )}
            </div>
          </div>

          <div>
            <label className="text-slate-400 font-semibold block mb-1.5 uppercase tracking-wider text-[11px]">
              Temporal Chunk Index
            </label>
            <select
              value={chunkIndex}
              onChange={(e) => setChunkIndex(parseInt(e.target.value))}
              disabled={loading}
              className="w-full p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-white font-medium focus:border-sky-500 focus:outline-none"
            >
              <option value="0">Chunk 1 (0.0s – 4.0s)</option>
              <option value="1">Chunk 2 (2.0s – 6.0s)</option>
              <option value="2">Chunk 3 (4.0s – 8.0s)</option>
            </select>
          </div>
        </div>

        {/* Action Button */}
        <button
          onClick={handleGenerateExplanations}
          disabled={!selectedFile || loading}
          className={`px-6 py-2.5 rounded-xl font-bold text-xs uppercase tracking-wider transition-all flex items-center justify-center gap-2 ${
            !selectedFile || loading
              ? "bg-slate-800 text-slate-400 cursor-not-allowed border border-slate-700"
              : "bg-sky-500 hover:bg-sky-400 text-slate-950 shadow-md shadow-sky-500/20"
          }`}
        >
          {loading ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              Computing Gradient & Attention Attributions...
            </>
          ) : (
            "Generate Acoustic Saliency Maps"
          )}
        </button>

        {error && (
          <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-400 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Visual Heatmaps Results */}
      {result && (
        <div className="space-y-6">
          {/* 3-Column Visual Map Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* 1. Base Log-Mel Spectrogram */}
            <div className="p-5 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs font-bold text-white uppercase tracking-wider">
                  Log-Mel Spectrogram (Input)
                </span>
                <span className="text-[10px] font-mono text-slate-400">128 × 128 Bins</span>
              </div>
              <div className="aspect-square w-full rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center">
                <img
                  src={result.spectrogram_image}
                  alt="Log-Mel Spectrogram"
                  className="w-full h-full object-cover"
                />
              </div>
              <p className="text-[11px] text-slate-400 leading-snug">
                Normalized ImageNet representation. Frequency bins (0–8 kHz) on vertical axis, time frames on horizontal axis.
              </p>
            </div>

            {/* 2. CNN Grad-CAM Overlay */}
            <div className="p-5 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs font-bold text-sky-400 uppercase tracking-wider">
                  CNN Grad-CAM (EfficientNet-B0)
                </span>
                <span className="text-[10px] font-mono text-slate-400">MBConv Saliency</span>
              </div>
              <div className="aspect-square w-full rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center">
                {result.efficientnet_overlay_image ? (
                  <img
                    src={result.efficientnet_overlay_image}
                    alt="EfficientNet Grad-CAM Heatmap"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <span className="text-xs text-slate-500">Grad-CAM unavailable</span>
                )}
              </div>
              <p className="text-[11px] text-slate-400 leading-snug">
                Gradient-weighted class activations highlight localized low-to-mid frequency formants (0–2 kHz).
              </p>
            </div>

            {/* 3. ViT Attention Overlay */}
            <div className="p-5 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs font-bold text-purple-400 uppercase tracking-wider">
                  ViT Self-Attention (ViT-B/16)
                </span>
                <span className="text-[10px] font-mono text-slate-400">Patch Rollout</span>
              </div>
              <div className="aspect-square w-full rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center">
                {result.vit_overlay_image ? (
                  <img
                    src={result.vit_overlay_image}
                    alt="ViT Attention Map"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <span className="text-xs text-slate-500">Attention map unavailable</span>
                )}
              </div>
              <p className="text-[11px] text-slate-400 leading-snug">
                Multi-head self-attention rollout captures non-local high-frequency spectral artifacts and synthesis phase boundaries.
              </p>
            </div>
          </div>

          {/* Qualitative Insights Card */}
          <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
            <h4 className="font-semibold text-white text-sm border-b border-slate-800 pb-3">
              Acoustic Attribution Analysis (Chunk {result.insights.chunk_analyzed} of {result.insights.total_chunks})
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-1.5">
                <span className="font-bold text-sky-400">CNN Receptive Field Dynamics</span>
                <p className="text-slate-300 leading-relaxed">{result.insights.efficientnet_focus}</p>
              </div>

              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-1.5">
                <span className="font-bold text-purple-400">Vision Transformer Attention Dynamics</span>
                <p className="text-slate-300 leading-relaxed">{result.insights.vit_focus}</p>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 text-[11px] text-slate-400 flex items-start gap-2">
              <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
              <span>{result.insights.disclaimer}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
