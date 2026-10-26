"use client";

import React, { useState, useEffect } from "react";
import {
  Activity,
  AlertTriangle,
  FileAudio,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Volume2,
  Info,
  Radio,
} from "lucide-react";
import {
  testRobustness,
  fetchSamples,
  AudioSample,
  RobustnessTestResult,
} from "@/lib/api";

export function RobustnessView() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [model, setModel] = useState<string>("vit_b16");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RobustnessTestResult | null>(null);
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

  const handleRunRobustness = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await testRobustness(selectedFile, selectedFile.name, model);
      setResult(res);
    } catch (err: any) {
      setError(err.message || "Robustness stress test failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Configuration Header Card */}
      <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-6">
        <div>
          <h3 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
            <Activity className="w-5 h-5 text-sky-400" />
            Acoustic Corruption & Perturbation Suite
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Evaluate model resilience against additive white Gaussian noise, amplitude scaling, and telephony 8 kHz bandpass downsampling.
          </p>
        </div>

        {/* Quick Sample Selector */}
        <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-slate-800/80">
          <span className="text-xs text-slate-400 font-semibold mr-2">Test with sample audio:</span>
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

        {/* Upload and Model Controls */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs pt-1">
          <div className="sm:col-span-2">
            <label className="text-slate-400 font-semibold block mb-1.5 uppercase tracking-wider text-[11px]">
              Audio File
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
              Evaluated Model
            </label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={loading}
              className="w-full p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-white font-medium focus:border-sky-500 focus:outline-none"
            >
              <option value="vit_b16">ViT-B/16 (Vision Transformer)</option>
              <option value="efficientnet_b0">EfficientNet-B0 (CNN Baseline)</option>
            </select>
          </div>
        </div>

        {/* Submit */}
        <button
          onClick={handleRunRobustness}
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
              Applying Acoustic Perturbations...
            </>
          ) : (
            "Execute Perturbation Stress Test"
          )}
        </button>

        {error && (
          <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-400 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Real Single-Sample Stress Results */}
      {result && (
        <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <span className="text-[11px] font-mono text-sky-400 uppercase font-semibold">
                Stress Test Subject: {result.filename}
              </span>
              <h3 className="text-lg font-bold text-white mt-0.5">
                Baseline Clean Score: {(result.clean_score * 100).toFixed(2)}% ({result.clean_prediction.toUpperCase()})
              </h3>
            </div>
            <span className="text-xs font-mono px-3 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300">
              Model: {result.model_id}
            </span>
          </div>

          {/* Perturbation Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {result.perturbations.map((p, idx) => {
              const delta = p.score_delta_from_clean;
              const isFlipped = p.prediction !== result.clean_prediction;

              return (
                <div
                  key={idx}
                  className={`p-4 rounded-xl border space-y-3 transition-all ${
                    isFlipped
                      ? "bg-rose-500/10 border-rose-500/40"
                      : "bg-slate-900/80 border-slate-800"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-white">{p.condition_name}</span>
                    <span
                      className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
                        p.prediction === "bonafide"
                          ? "bg-emerald-500/20 text-emerald-300"
                          : "bg-rose-500/20 text-rose-300"
                      }`}
                    >
                      {p.prediction.toUpperCase()}
                    </span>
                  </div>

                  <p className="text-[11px] text-slate-400 leading-snug">{p.description}</p>

                  <div className="space-y-1 font-mono text-xs pt-1 border-t border-slate-800/80">
                    <div className="flex justify-between">
                      <span className="text-slate-400">P_spoof:</span>
                      <span className="text-white font-bold">{(p.spoof_score * 100).toFixed(2)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Score Delta (Δ):</span>
                      <span
                        className={`font-bold ${
                          delta > 0.05
                            ? "text-rose-400"
                            : delta < -0.05
                            ? "text-emerald-400"
                            : "text-slate-300"
                        }`}
                      >
                        {delta > 0 ? `+${(delta * 100).toFixed(2)}%` : `${(delta * 100).toFixed(2)}%`}
                      </span>
                    </div>
                  </div>

                  {/* Waveform envelope for corrupted audio */}
                  {p.waveform_downsampled.length > 0 && (
                    <div className="h-8 w-full bg-slate-950/60 rounded p-1 flex items-end gap-[1px] overflow-hidden border border-slate-800/60">
                      {p.waveform_downsampled.map((v, i) => (
                        <div
                          key={i}
                          className={`flex-1 rounded-t-sm ${isFlipped ? "bg-rose-400/80" : "bg-sky-400/70"}`}
                          style={{ height: `${Math.max(10, v * 100)}%` }}
                        />
                      ))}
                    </div>
                  )}

                  {isFlipped && (
                    <div className="text-[10px] font-semibold text-rose-300 flex items-center gap-1">
                      <AlertTriangle className="w-3 h-3" /> Classification flipped from clean baseline!
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Academic Disclaimer */}
          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 text-[11px] text-slate-400 flex items-start gap-2">
            <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
            <span>{result.disclaimer}</span>
          </div>
        </div>
      )}

      {/* Population-Level Benchmark Matrix from results/robustness_results.json */}
      {result?.benchmark_dataset_summary && Object.keys(result.benchmark_dataset_summary).length > 0 && (
        <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <h4 className="font-semibold text-white text-sm">
              Population-Level Benchmark Matrix ({model === "vit_b16" ? "ViT-B/16" : "EfficientNet-B0"})
            </h4>
            <p className="text-xs text-slate-400">
              Evaluated across 500 balanced test utterances per condition (from results/robustness_results.json)
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px]">
                  <th className="py-2.5 px-3">Condition</th>
                  <th className="py-2.5 px-3">EER</th>
                  <th className="py-2.5 px-3">ROC-AUC</th>
                  <th className="py-2.5 px-3">Accuracy</th>
                  <th className="py-2.5 px-3">F1 Macro</th>
                  <th className="py-2.5 px-3">FPR</th>
                  <th className="py-2.5 px-3">FNR</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {Object.entries(result.benchmark_dataset_summary).map(([cond, vals]: [string, any]) => (
                  <tr key={cond} className="hover:bg-slate-900/50">
                    <td className="py-2.5 px-3 font-semibold text-white font-sans">{cond}</td>
                    <td className="py-2.5 px-3 text-purple-400">{(vals.eer * 100).toFixed(2)}%</td>
                    <td className="py-2.5 px-3">{vals.roc_auc ? vals.roc_auc.toFixed(4) : "N/A"}</td>
                    <td className="py-2.5 px-3 text-emerald-400">{(vals.accuracy * 100).toFixed(1)}%</td>
                    <td className="py-2.5 px-3">{vals.f1_macro ? vals.f1_macro.toFixed(4) : "N/A"}</td>
                    <td className="py-2.5 px-3 font-mono">{(vals.false_positive_rate * 100).toFixed(1)}%</td>
                    <td className="py-2.5 px-3 font-mono">{(vals.false_negative_rate * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
