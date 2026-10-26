"use client";

import React, { useEffect, useState } from "react";
import {
  GitCompare,
  Layers,
  Cpu,
  Clock,
  HelpCircle,
  TrendingDown,
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import {
  fetchMetricsSummary,
  fetchModels,
  MetricsSummary,
  ModelInfo,
} from "@/lib/api";

export function ComparisonView() {
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([fetchModels(), fetchMetricsSummary()])
      .then(([m, met]) => {
        setModels(m);
        setMetrics(met);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[350px] text-slate-400">
        <div className="w-8 h-8 border-2 border-sky-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const eff = models["efficientnet_b0"];
  const vit = models["vit_b16"];
  const effMetric = metrics?.efficientnet_baseline;
  const vitMetric = metrics?.vit_b16_model;

  const eerComparisonData = [
    {
      split: "Development Split (Dev)",
      "EfficientNet-B0": eff ? (eff.dev_eer * 100).toFixed(2) : 6.44,
      "ViT-B/16": vit ? (vit.dev_eer * 100).toFixed(2) : 1.55,
    },
    {
      split: "Official Evaluation (Eval)",
      "EfficientNet-B0": effMetric ? (effMetric.eer * 100).toFixed(2) : 21.49,
      "ViT-B/16": vitMetric ? (vitMetric.eer * 100).toFixed(2) : 27.08,
    },
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Overview Comparison Header Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* CNN Baseline Card */}
        <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <span className="text-[10px] font-mono text-sky-400 uppercase tracking-wider font-semibold">
                Convolutional Baseline
              </span>
              <h2 className="text-lg font-bold text-white">EfficientNet-B0</h2>
            </div>
            <span className="text-xs font-mono px-2.5 py-1 rounded bg-sky-500/15 text-sky-300 border border-sky-500/30">
              4.01M Params
            </span>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Architecture Family</span>
              <span className="text-white font-medium">Convolutional Neural Network (CNN)</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Feature Extraction</span>
              <span className="text-white font-medium">MBConv blocks with Squeeze-and-Excitation</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Input Spectrogram</span>
              <span className="text-white font-mono">128 × 128 (3-channel normalized)</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Classification Head</span>
              <span className="text-white font-mono">Dropout(0.2) + Linear(1280, 2)</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Frozen Dev Operating Threshold</span>
              <span className="text-sky-400 font-mono font-bold">0.967773</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-400">Verified Checkpoint SHA-256</span>
              <span className="text-slate-400 font-mono text-[11px] truncate max-w-[180px]">
                {eff?.checkpoint?.sha256_hash ? eff.checkpoint.sha256_hash.slice(0, 16) + "..." : "1bd3841c7f..."}
              </span>
            </div>
          </div>
        </div>

        {/* Vision Transformer Card */}
        <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <span className="text-[10px] font-mono text-purple-400 uppercase tracking-wider font-semibold">
                Transformer Architecture
              </span>
              <h2 className="text-lg font-bold text-white">Vision Transformer (ViT-B/16)</h2>
            </div>
            <span className="text-xs font-mono px-2.5 py-1 rounded bg-purple-500/15 text-purple-300 border border-purple-500/30">
              85.8M Params (21.4×)
            </span>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Architecture Family</span>
              <span className="text-white font-medium">Vision Transformer (Self-Attention)</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Positional Embeddings</span>
              <span className="text-white font-medium">2D Bicubic Interpolation to (8×8 grid)</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Input Spectrogram</span>
              <span className="text-white font-mono">128 × 128 (Patch Size 16 = 64 tokens)</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Classification Head</span>
              <span className="text-white font-mono">Dropout(0.1) + Linear(768, 2)</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Frozen Dev Operating Threshold</span>
              <span className="text-purple-400 font-mono font-bold">0.986328</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-400">Verified Checkpoint SHA-256</span>
              <span className="text-slate-400 font-mono text-[11px] truncate max-w-[180px]">
                {vit?.checkpoint?.sha256_hash ? vit.checkpoint.sha256_hash.slice(0, 16) + "..." : "14dfd3451c..."}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Biometric Benchmark Matrix Table */}
      <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
        <div className="border-b border-slate-800 pb-3">
          <h3 className="font-semibold text-white text-sm">Official Biometric Metric Matrix</h3>
          <p className="text-xs text-slate-400">
            Recorded from ASVspoof 2019 LA partition under strict zero-leakage protocol
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-mono uppercase text-[11px]">
                <th className="py-3 px-4">Evaluation Metric</th>
                <th className="py-3 px-4">EfficientNet-B0 (CNN)</th>
                <th className="py-3 px-4">ViT-B/16 (Transformer)</th>
                <th className="py-3 px-4">Comparison Analysis</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-200">
              <tr>
                <td className="py-3 px-4 font-semibold text-white">Equal Error Rate (EER) - Dev</td>
                <td className="py-3 px-4 font-mono text-sky-400 font-bold">6.44%</td>
                <td className="py-3 px-4 font-mono text-purple-400 font-bold">1.55%</td>
                <td className="py-3 px-4 text-slate-400">ViT reduced error by 76% on in-distribution dev split</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-semibold text-white">Equal Error Rate (EER) - Eval</td>
                <td className="py-3 px-4 font-mono text-sky-400 font-bold">21.49%</td>
                <td className="py-3 px-4 font-mono text-purple-400 font-bold">27.08%</td>
                <td className="py-3 px-4 text-slate-400">CNN generalized better to 13 unseen vocoders (A07–A19)</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-semibold text-white">ROC-AUC (Discrimination)</td>
                <td className="py-3 px-4 font-mono text-sky-400">0.8682</td>
                <td className="py-3 px-4 font-mono text-purple-400">0.8201</td>
                <td className="py-3 px-4 text-slate-400">Threshold-independent discrimination capacity</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-semibold text-white">Balanced Accuracy (at frozen τ)</td>
                <td className="py-3 px-4 font-mono text-sky-400">73.58%</td>
                <td className="py-3 px-4 font-mono text-purple-400">72.15%</td>
                <td className="py-3 px-4 text-slate-400">Mean of authentic and spoof recall at frozen threshold</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-semibold text-white">Macro F1 Score (at frozen τ)</td>
                <td className="py-3 px-4 font-mono text-sky-400">0.4739</td>
                <td className="py-3 px-4 font-mono text-purple-400">0.4693</td>
                <td className="py-3 px-4 text-slate-400">Severe penalty from threshold shift on evaluation split</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-semibold text-white">Inference Latency</td>
                <td className="py-3 px-4 font-mono text-sky-400">1.22 ms / utterance</td>
                <td className="py-3 px-4 font-mono text-purple-400">1.18 ms / utterance</td>
                <td className="py-3 px-4 text-slate-400">Both architectures achieve sub-2ms realtime throughput</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Side-by-Side Confusion Matrices */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* CNN Matrix */}
        <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
          <div className="border-b border-slate-800 pb-2">
            <h4 className="text-xs font-semibold text-sky-400 uppercase tracking-wider">
              EfficientNet-B0 Confusion Matrix (71,237 Utterances)
            </h4>
            <p className="text-[11px] text-slate-400">At frozen threshold τ = 0.967773</p>
          </div>

          <div className="grid grid-cols-2 gap-3 text-center text-xs font-mono pt-2">
            <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
              <span className="text-[10px] text-emerald-400 block font-sans">True Negatives (TN)</span>
              <span className="text-lg font-bold text-white">
                {effMetric?.true_negatives || 7331}
              </span>
              <span className="text-[10px] text-slate-400 block">Bonafide Correct</span>
            </div>
            <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20">
              <span className="text-[10px] text-rose-400 block font-sans">False Positives (FP)</span>
              <span className="text-lg font-bold text-white">
                {effMetric?.false_positives || 24}
              </span>
              <span className="text-[10px] text-slate-400 block">FPR: 0.33%</span>
            </div>
            <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20">
              <span className="text-[10px] text-rose-400 block font-sans">False Negatives (FN)</span>
              <span className="text-lg font-bold text-white">
                {effMetric?.false_negatives || 33544}
              </span>
              <span className="text-[10px] text-slate-400 block">FNR: 52.51%</span>
            </div>
            <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
              <span className="text-[10px] text-emerald-400 block font-sans">True Positives (TP)</span>
              <span className="text-lg font-bold text-white">
                {effMetric?.true_positives || 30338}
              </span>
              <span className="text-[10px] text-slate-400 block">Spoofs Detected</span>
            </div>
          </div>
        </div>

        {/* ViT Matrix */}
        <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
          <div className="border-b border-slate-800 pb-2">
            <h4 className="text-xs font-semibold text-purple-400 uppercase tracking-wider">
              ViT-B/16 Confusion Matrix (71,237 Utterances)
            </h4>
            <p className="text-[11px] text-slate-400">At frozen threshold τ = 0.986328</p>
          </div>

          <div className="grid grid-cols-2 gap-3 text-center text-xs font-mono pt-2">
            <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
              <span className="text-[10px] text-emerald-400 block font-sans">True Negatives (TN)</span>
              <span className="text-lg font-bold text-white">
                {vitMetric?.true_negatives || 7123}
              </span>
              <span className="text-[10px] text-slate-400 block">Bonafide Correct</span>
            </div>
            <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20">
              <span className="text-[10px] text-rose-400 block font-sans">False Positives (FP)</span>
              <span className="text-lg font-bold text-white">
                {vitMetric?.false_positives || 232}
              </span>
              <span className="text-[10px] text-slate-400 block">FPR: 3.15%</span>
            </div>
            <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20">
              <span className="text-[10px] text-rose-400 block font-sans">False Negatives (FN)</span>
              <span className="text-lg font-bold text-white">
                {vitMetric?.false_negatives || 33569}
              </span>
              <span className="text-[10px] text-slate-400 block">FNR: 52.55%</span>
            </div>
            <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
              <span className="text-[10px] text-emerald-400 block font-sans">True Positives (TP)</span>
              <span className="text-lg font-bold text-white">
                {vitMetric?.true_positives || 30313}
              </span>
              <span className="text-[10px] text-slate-400 block">Spoofs Detected</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
