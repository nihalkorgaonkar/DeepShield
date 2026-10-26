"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  UploadCloud,
  Play,
  Pause,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Clock,
  Layers,
  Sparkles,
  Volume2,
  FileAudio,
  Sliders,
  RefreshCw,
  Info,
} from "lucide-react";
import {
  detectAudio,
  detectAndCompare,
  fetchSamples,
  AudioSample,
  DetectionResult,
  DetectionComparisonResult,
} from "@/lib/api";

export function DetectionView() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [mode, setMode] = useState<"single" | "compare">("compare");
  const [model, setModel] = useState<string>("vit_b16");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [samples, setSamples] = useState<AudioSample[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);

  // Results
  const [singleResult, setSingleResult] = useState<DetectionResult | null>(null);
  const [compareResult, setCompareResult] = useState<DetectionComparisonResult | null>(null);

  // Threshold overrides
  const [customThreshold, setCustomThreshold] = useState<number | undefined>(undefined);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    fetchSamples()
      .then(setSamples)
      .catch(() => {});
  }, []);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    setSingleResult(null);
    setCompareResult(null);
    setError(null);
    setIsPlaying(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleLoadSample = async (sample: AudioSample) => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(sample.url);
      if (!res.ok) throw new Error("Failed to load sample audio from server.");
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: "audio/flac" });
      handleFileSelect(file);
    } catch (err: any) {
      setError(err.message || "Failed to load sample audio.");
    } finally {
      setLoading(false);
    }
  };

  const togglePlayback = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setSingleResult(null);
    setCompareResult(null);

    try {
      if (mode === "single") {
        const res = await detectAudio(
          selectedFile,
          selectedFile.name,
          model,
          customThreshold
        );
        setSingleResult(res);
      } else {
        const res = await detectAndCompare(
          selectedFile,
          selectedFile.name,
          undefined,
          customThreshold
        );
        setCompareResult(res);
      }
    } catch (err: any) {
      setError(err.message || "Audio inference failed.");
    } finally {
      setLoading(false);
    }
  };

  // Helper for active metadata
  const activeMetadata =
    singleResult?.audio_metadata ||
    compareResult?.audio_metadata ||
    null;

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Top Banner: Quick Sample Selectors */}
      <div className="p-4 rounded-xl bg-[#0f172a] border border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <FileAudio className="w-4 h-4 text-sky-400" />
          <span className="text-xs font-semibold text-slate-300">Quick Test Samples:</span>
          <span className="text-xs text-slate-400">Load authentic or synthetic speech directly:</span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
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
      </div>

      {/* Main Grid: Upload & Analysis on Left, Results on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column (5 Cols): Upload, Controls, Audio Player */}
        <div className="lg:col-span-5 space-y-6">
          {/* Drag & Drop Zone */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-slate-700 hover:border-sky-500 bg-slate-900/40 hover:bg-slate-900/80 rounded-2xl p-8 text-center cursor-pointer transition-all space-y-3 group"
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              accept=".wav,.flac,.mp3,.ogg,.m4a"
              className="hidden"
            />
            <div className="w-12 h-12 mx-auto rounded-xl bg-slate-800 group-hover:bg-sky-500/10 border border-slate-700 group-hover:border-sky-500/30 flex items-center justify-center text-slate-400 group-hover:text-sky-400 transition-colors">
              <UploadCloud className="w-6 h-6" />
            </div>
            <div>
              <p className="text-sm font-semibold text-white">
                {selectedFile ? selectedFile.name : "Drop audio file or browse"}
              </p>
              <p className="text-xs text-slate-400 mt-1">
                Supports WAV, FLAC, MP3, OGG (16 kHz mono normalized)
              </p>
            </div>
            {selectedFile && (
              <span className="inline-block text-[11px] font-mono px-2 py-0.5 rounded bg-sky-500/20 text-sky-300 border border-sky-500/30">
                {(selectedFile.size / 1024).toFixed(1)} KB
              </span>
            )}
          </div>

          {/* Audio Player and Waveform Preview */}
          {audioUrl && (
            <div className="p-4 rounded-xl bg-[#0f172a] border border-slate-800 space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                  <Volume2 className="w-3.5 h-3.5 text-sky-400" /> Audio Playback
                </span>
                <span className="font-mono text-slate-300">{selectedFile?.name}</span>
              </div>

              {/* Native audio element */}
              <audio
                ref={audioRef}
                src={audioUrl}
                onEnded={() => setIsPlaying(false)}
                className="w-full h-8"
                controls
              />

              {/* Waveform Visualization Envelope */}
              {activeMetadata && activeMetadata.waveform_downsampled.length > 0 && (
                <div className="pt-2">
                  <p className="text-[10px] text-slate-400 uppercase font-mono mb-1">
                    Waveform Envelope (RMS Normalized)
                  </p>
                  <div className="h-12 w-full bg-slate-950/80 rounded-lg p-1.5 flex items-end gap-[2px] overflow-hidden border border-slate-800/80">
                    {activeMetadata.waveform_downsampled.map((val, idx) => (
                      <div
                        key={idx}
                        className="flex-1 bg-sky-500/70 hover:bg-sky-400 rounded-t-sm transition-all"
                        style={{ height: `${Math.max(8, val * 100)}%` }}
                        title={`Sample ${idx}: ${(val * 100).toFixed(1)}%`}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Metadata telemetry */}
              {activeMetadata && (
                <div className="grid grid-cols-3 gap-2 pt-1 text-[11px] font-mono text-slate-300">
                  <div className="p-2 rounded bg-slate-900 border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">Duration</span>
                    {activeMetadata.duration_sec}s
                  </div>
                  <div className="p-2 rounded bg-slate-900 border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">Sample Rate</span>
                    {activeMetadata.sample_rate_hz} Hz
                  </div>
                  <div className="p-2 rounded bg-slate-900 border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">Channels</span>
                    {activeMetadata.num_channels === 1 ? "Mono" : "Stereo"}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Inference Mode & Settings */}
          <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-4">
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-2">
                Inference Mode
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setMode("compare")}
                  className={`p-2.5 rounded-lg text-xs font-medium border text-left transition-all ${
                    mode === "compare"
                      ? "bg-sky-500/15 border-sky-500 text-white"
                      : "bg-slate-900 border-slate-800 text-slate-400 hover:text-white"
                  }`}
                >
                  <div className="font-semibold text-white">Dual-Model Comparison</div>
                  <div className="text-[11px] text-slate-400 mt-0.5">CNN vs ViT Agreement</div>
                </button>
                <button
                  type="button"
                  onClick={() => setMode("single")}
                  className={`p-2.5 rounded-lg text-xs font-medium border text-left transition-all ${
                    mode === "single"
                      ? "bg-sky-500/15 border-sky-500 text-white"
                      : "bg-slate-900 border-slate-800 text-slate-400 hover:text-white"
                  }`}
                >
                  <div className="font-semibold text-white">Single Architecture</div>
                  <div className="text-[11px] text-slate-400 mt-0.5">Evaluate one model</div>
                </button>
              </div>
            </div>

            {/* Model Selector if single mode */}
            {mode === "single" && (
              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-2">
                  Select Model
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setModel("vit_b16")}
                    className={`p-2 rounded-lg text-xs font-medium border ${
                      model === "vit_b16"
                        ? "bg-purple-500/15 border-purple-500 text-purple-300"
                        : "bg-slate-900 border-slate-800 text-slate-400"
                    }`}
                  >
                    ViT-B/16 (128x128)
                  </button>
                  <button
                    type="button"
                    onClick={() => setModel("efficientnet_b0")}
                    className={`p-2 rounded-lg text-xs font-medium border ${
                      model === "efficientnet_b0"
                        ? "bg-sky-500/15 border-sky-500 text-sky-300"
                        : "bg-slate-900 border-slate-800 text-slate-400"
                    }`}
                  >
                    EfficientNet-B0 (CNN)
                  </button>
                </div>
              </div>
            )}

            {/* Submit Action */}
            <button
              onClick={handleAnalyze}
              disabled={!selectedFile || loading}
              className={`w-full py-3 px-4 rounded-xl font-bold text-xs uppercase tracking-wider transition-all flex items-center justify-center gap-2 ${
                !selectedFile || loading
                  ? "bg-slate-800 text-slate-400 cursor-not-allowed border border-slate-700/50"
                  : "bg-sky-500 hover:bg-sky-400 text-slate-950 shadow-md shadow-sky-500/20"
              }`}
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Running Neural Forward Pass...
                </>
              ) : mode === "compare" ? (
                "Run Dual Comparative Inference"
              ) : (
                "Analyze Audio Authenticity"
              )}
            </button>

            {error && (
              <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-400 text-xs flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column (7 Cols): Detection Results Panel */}
        <div className="lg:col-span-7">
          {compareResult ? (
            /* Dual Model Comparative Results Card */
            <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-6">
              {/* Agreement Header */}
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
                <div>
                  <span className="text-[11px] uppercase tracking-wider font-semibold text-slate-400">
                    Dual Architecture Consensus
                  </span>
                  <div className="flex items-center gap-2 mt-1">
                    <span
                      className={`text-xl font-extrabold tracking-tight ${
                        compareResult.agreement ? "text-emerald-400" : "text-amber-400"
                      }`}
                    >
                      {compareResult.agreement
                        ? "MODELS IN AGREEMENT"
                        : "MODEL DISAGREEMENT DETECTED"}
                    </span>
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-[10px] text-slate-400 uppercase font-mono block">Latency Delta</span>
                  <span className="text-xs font-mono font-medium text-slate-300">
                    {compareResult.faster_model} faster by {compareResult.latency_difference_ms} ms
                  </span>
                </div>
              </div>

              {/* Disagreement Warning if any */}
              {compareResult.disagreement_warning && (
                <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-200 text-xs flex items-start gap-2.5">
                  <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <span>{compareResult.disagreement_warning}</span>
                </div>
              )}

              {/* Side-by-Side Model Comparison Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* EfficientNet Card */}
                <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-xs font-semibold text-sky-400">EfficientNet-B0</span>
                    <span className="text-[10px] font-mono text-slate-400">
                      {compareResult.efficientnet_result.total_inference_time_ms} ms
                    </span>
                  </div>

                  <div>
                    <span
                      className={`text-sm font-bold block ${
                        compareResult.efficientnet_result.prediction === "bonafide"
                          ? "text-emerald-400"
                          : "text-rose-400"
                      }`}
                    >
                      {compareResult.efficientnet_result.prediction_label}
                    </span>
                    <span className="text-[11px] text-slate-400 font-mono">
                      Confidence: {compareResult.efficientnet_result.confidence_percent}%
                    </span>
                  </div>

                  {/* Progress Bars */}
                  <div className="space-y-1.5 pt-1 text-[11px]">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Spoof Posterior (P_spoof)</span>
                      <span className="font-mono text-rose-400">
                        {(compareResult.efficientnet_result.spoof_score * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="bg-rose-500 h-full rounded-full transition-all"
                        style={{
                          width: `${Math.min(100, compareResult.efficientnet_result.spoof_score * 100)}%`,
                        }}
                      />
                    </div>
                    <div className="flex justify-between text-[10px] text-slate-400 pt-0.5">
                      <span>Frozen Threshold:</span>
                      <span className="font-mono text-slate-300">
                        {compareResult.efficientnet_result.configured_threshold}
                      </span>
                    </div>
                  </div>
                </div>

                {/* ViT Card */}
                <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-xs font-semibold text-purple-400">Vision Transformer ViT-B/16</span>
                    <span className="text-[10px] font-mono text-slate-400">
                      {compareResult.vit_result.total_inference_time_ms} ms
                    </span>
                  </div>

                  <div>
                    <span
                      className={`text-sm font-bold block ${
                        compareResult.vit_result.prediction === "bonafide"
                          ? "text-emerald-400"
                          : "text-rose-400"
                      }`}
                    >
                      {compareResult.vit_result.prediction_label}
                    </span>
                    <span className="text-[11px] text-slate-400 font-mono">
                      Confidence: {compareResult.vit_result.confidence_percent}%
                    </span>
                  </div>

                  {/* Progress Bars */}
                  <div className="space-y-1.5 pt-1 text-[11px]">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Spoof Posterior (P_spoof)</span>
                      <span className="font-mono text-rose-400">
                        {(compareResult.vit_result.spoof_score * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="bg-purple-500 h-full rounded-full transition-all"
                        style={{
                          width: `${Math.min(100, compareResult.vit_result.spoof_score * 100)}%`,
                        }}
                      />
                    </div>
                    <div className="flex justify-between text-[10px] text-slate-400 pt-0.5">
                      <span>Frozen Threshold:</span>
                      <span className="font-mono text-slate-300">
                        {compareResult.vit_result.configured_threshold}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Per-chunk timeline for ViT */}
              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800/80 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-300">Temporal Chunk Decomposition (4.0s sliding window)</span>
                  <span className="font-mono text-slate-400 text-[11px]">
                    {compareResult.vit_result.num_chunks_analyzed} chunks analyzed
                  </span>
                </div>
                <div className="space-y-1.5 pt-1">
                  {compareResult.vit_result.chunk_scores.map((c) => (
                    <div
                      key={c.chunk_index}
                      className="p-2 rounded bg-slate-950/60 flex items-center justify-between text-xs font-mono"
                    >
                      <span className="text-slate-400">
                        Chunk {c.chunk_index} ({c.start_time_sec}s – {c.end_time_sec}s)
                      </span>
                      <div className="flex items-center gap-3">
                        <span className="text-slate-400">
                          P_spoof: <strong className="text-white">{c.spoof_score}</strong>
                        </span>
                        <span
                          className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
                            c.is_spoof ? "bg-rose-500/20 text-rose-400" : "bg-emerald-500/20 text-emerald-400"
                          }`}
                        >
                          {c.is_spoof ? "SPOOF" : "BONAFIDE"}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Academic Disclaimer Box */}
              <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 text-[11px] text-slate-400 flex items-start gap-2">
                <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
                <span>{compareResult.disclaimer}</span>
              </div>
            </div>
          ) : singleResult ? (
            /* Single Model Results Card */
            <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-6">
              {/* Verdict Header */}
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
                <div>
                  <span className="text-[11px] uppercase tracking-wider font-semibold text-slate-400">
                    Detection Verdict ({singleResult.model_name})
                  </span>
                  <div className="flex items-center gap-3 mt-1">
                    <span
                      className={`text-2xl font-extrabold tracking-tight ${
                        singleResult.prediction === "bonafide" ? "text-emerald-400" : "text-rose-400"
                      }`}
                    >
                      {singleResult.prediction_label}
                    </span>
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-[10px] text-slate-400 uppercase font-mono block">Confidence</span>
                  <span className="text-2xl font-mono font-bold text-white">
                    {singleResult.confidence_percent}%
                  </span>
                </div>
              </div>

              {/* Posterior Probabilities */}
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1">
                    <span className="text-slate-400">Genuine / Bona Fide Probability</span>
                    <span className="text-emerald-400 font-mono">
                      {(singleResult.bonafide_score * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                    <div
                      className="bg-emerald-500 h-full rounded-full transition-all"
                      style={{ width: `${singleResult.bonafide_score * 100}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs font-medium mb-1">
                    <span className="text-slate-400">Spoof / Deepfake Probability</span>
                    <span className="text-rose-400 font-mono">
                      {(singleResult.spoof_score * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                    <div
                      className="bg-rose-500 h-full rounded-full transition-all"
                      style={{ width: `${singleResult.spoof_score * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Telemetry Metrics */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Configured τ</span>
                  <span className="text-white font-bold">{singleResult.configured_threshold}</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Inference Time</span>
                  <span className="text-white font-bold">{singleResult.total_inference_time_ms} ms</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Chunks Evaluated</span>
                  <span className="text-white font-bold">{singleResult.num_chunks_analyzed}</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Decision Rule</span>
                  <span className="text-white font-bold">
                    {singleResult.is_above_threshold ? "Score ≥ τ" : "Score < τ"}
                  </span>
                </div>
              </div>

              {/* Chunk breakdown */}
              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800/80 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-300">Temporal Chunk Decomposition</span>
                  <span className="font-mono text-slate-400 text-[11px]">
                    {singleResult.num_chunks_analyzed} window(s)
                  </span>
                </div>
                <div className="space-y-1.5 pt-1">
                  {singleResult.chunk_scores.map((c) => (
                    <div
                      key={c.chunk_index}
                      className="p-2 rounded bg-slate-950/60 flex items-center justify-between text-xs font-mono"
                    >
                      <span className="text-slate-400">
                        Chunk {c.chunk_index} ({c.start_time_sec}s – {c.end_time_sec}s)
                      </span>
                      <div className="flex items-center gap-3">
                        <span className="text-slate-400">
                          P_spoof: <strong className="text-white">{c.spoof_score}</strong>
                        </span>
                        <span
                          className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
                            c.is_spoof ? "bg-rose-500/20 text-rose-400" : "bg-emerald-500/20 text-emerald-400"
                          }`}
                        >
                          {c.is_spoof ? "SPOOF" : "BONAFIDE"}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Academic Disclaimer Box */}
              <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 text-[11px] text-slate-400 flex items-start gap-2">
                <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
                <span>{singleResult.disclaimer}</span>
              </div>
            </div>
          ) : (
            /* Empty State */
            <div className="min-h-[420px] rounded-2xl bg-[#0f172a]/60 border border-slate-800/80 flex flex-col items-center justify-center p-8 text-center space-y-3">
              <div className="w-14 h-14 rounded-2xl bg-slate-800/50 border border-slate-700/50 flex items-center justify-center text-slate-400">
                <Sparkles className="w-6 h-6 text-slate-400" />
              </div>
              <h3 className="font-semibold text-white text-base">Inference Ready</h3>
              <p className="text-xs text-slate-400 max-w-sm leading-relaxed">
                Upload an audio file on the left or select one of the preloaded sample test recordings to evaluate
                speech authenticity using real PyTorch checkpoints.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
