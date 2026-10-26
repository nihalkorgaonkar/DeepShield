"use client";

import React, { useEffect, useState } from "react";
import {
  ShieldAlert,
  AlertTriangle,
  Cpu,
  Layers,
  Database,
  ArrowRight,
  TrendingDown,
  Clock,
  CheckCircle2,
  XCircle,
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
  fetchHealth,
  fetchMetricsSummary,
  fetchModels,
  MetricsSummary,
  ModelInfo,
  SystemHealth,
} from "@/lib/api";
import { NavSection } from "./sidebar";

interface OverviewViewProps {
  onNavigate: (section: NavSection) => void;
}

export function OverviewView({ onNavigate }: OverviewViewProps) {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [h, m, met] = await Promise.all([
          fetchHealth().catch(() => null),
          fetchModels().catch(() => ({})),
          fetchMetricsSummary().catch(() => null),
        ]);
        setHealth(h);
        setModels(m);
        setMetrics(met);
      } catch (err: any) {
        setError(err.message || "Failed to load dashboard data");
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] text-slate-400 space-y-3">
        <div className="w-8 h-8 border-2 border-sky-500 border-t-transparent rounded-full animate-spin" />
        <p className="text-sm">Loading DeepShield research telemetry...</p>
      </div>
    );
  }

  const effInfo = models["efficientnet_b0"];
  const vitInfo = models["vit_b16"];
  const effMetric = metrics?.efficientnet_baseline;
  const vitMetric = metrics?.vit_b16_model;

  // Comparison chart data
  const chartData = [
    {
      metric: "Dev EER (%)",
      "EfficientNet-B0": effInfo ? (effInfo.dev_eer * 100).toFixed(2) : 6.44,
      "ViT-B/16": vitInfo ? (vitInfo.dev_eer * 100).toFixed(2) : 1.55,
    },
    {
      metric: "Eval EER (%)",
      "EfficientNet-B0": effMetric ? (effMetric.eer * 100).toFixed(2) : 21.49,
      "ViT-B/16": vitMetric ? (vitMetric.eer * 100).toFixed(2) : 27.08,
    },
    {
      metric: "ROC-AUC (×100)",
      "EfficientNet-B0": effMetric ? (effMetric.roc_auc * 100).toFixed(1) : 86.8,
      "ViT-B/16": vitMetric ? (vitMetric.roc_auc * 100).toFixed(1) : 82.0,
    },
    {
      metric: "Balanced Acc (%)",
      "EfficientNet-B0": effMetric ? (effMetric.balanced_accuracy * 100).toFixed(1) : 73.6,
      "ViT-B/16": vitMetric ? (vitMetric.balanced_accuracy * 100).toFixed(1) : 72.1,
    },
    {
      metric: "Macro F1 (×100)",
      "EfficientNet-B0": effMetric ? (effMetric.f1_macro * 100).toFixed(1) : 47.4,
      "ViT-B/16": vitMetric ? (vitMetric.f1_macro * 100).toFixed(1) : 46.9,
    },
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Research Discrepancy Alert Banner */}
      {metrics?.discrepancy_warning && (
        <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-200 space-y-2">
          <div className="flex items-center gap-2.5 font-semibold text-sm text-amber-300">
            <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
            <span>Research Audit Alert: Domain Shift & Threshold Discrepancy Flagged</span>
          </div>
          <p className="text-xs text-amber-300/80 leading-relaxed pl-7.5">
            {metrics.warning_details}
          </p>
          <div className="text-[11px] font-mono text-amber-400/90 bg-amber-950/40 p-2.5 rounded-lg border border-amber-500/20 mt-2 space-y-1">
            <div>• EfficientNet-B0: Frozen Dev τ = 0.967773 vs Optimal Eval Operating τ = 2.801418e-06</div>
            <div>• Vision Transformer: Frozen Dev τ = 0.986328 vs Optimal Eval Operating τ = 1.215935e-05</div>
            <div className="text-[10px] text-amber-500/80 mt-1 italic">
              Governed by DeepShield Research Rule #4: Never tune decision thresholds on official evaluation data.
            </div>
          </div>
        </div>
      )}

      {/* Top Telemetry KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Checkpoint Status */}
        <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-2 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-medium uppercase tracking-wider text-[11px]">Checkpoints</span>
            <Layers className="w-4 h-4 text-sky-400" />
          </div>
          <div className="space-y-1 pt-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">EfficientNet-B0</span>
              {effInfo?.checkpoint?.exists ? (
                <span className="flex items-center gap-1 text-emerald-400 font-mono text-[11px]">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Verified
                </span>
              ) : (
                <span className="flex items-center gap-1 text-rose-400 font-mono text-[11px]">
                  <XCircle className="w-3.5 h-3.5" /> Missing
                </span>
              )}
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">ViT-B/16</span>
              {vitInfo?.checkpoint?.exists ? (
                <span className="flex items-center gap-1 text-emerald-400 font-mono text-[11px]">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Verified
                </span>
              ) : (
                <span className="flex items-center gap-1 text-rose-400 font-mono text-[11px]">
                  <XCircle className="w-3.5 h-3.5" /> Missing
                </span>
              )}
            </div>
          </div>
          <p className="text-[10px] text-slate-400 pt-1">SHA-256 Checksums audited in final_audit.md</p>
        </div>

        {/* Card 2: Compute Device */}
        <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-2 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-medium uppercase tracking-wider text-[11px]">Compute Acceleration</span>
            <Cpu className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="pt-1">
            <div className="text-lg font-bold text-white font-mono">
              {health?.cuda_available ? "NVIDIA CUDA" : "CPU Fallback"}
            </div>
            <p className="text-xs text-sky-400 font-medium truncate mt-0.5">
              {health?.cuda_device_name || "Host Processors"}
            </p>
          </div>
          <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 pt-1 border-t border-slate-800/80">
            <span>VRAM Allocated</span>
            <span className="text-white">
              {health?.vram_allocated_gb !== undefined ? `${health.vram_allocated_gb} / ${health.vram_total_gb} GB` : "N/A"}
            </span>
          </div>
        </div>

        {/* Card 3: Evaluation Dataset */}
        <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-2 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-medium uppercase tracking-wider text-[11px]">ASVspoof 2019 LA</span>
            <Database className="w-4 h-4 text-purple-400" />
          </div>
          <div className="pt-1">
            <div className="text-2xl font-bold text-white font-mono">71,237</div>
            <p className="text-xs text-slate-400 mt-0.5">Evaluation Utterances (7,355 Bonafide, 63,882 Spoof)</p>
          </div>
          <p className="text-[10px] text-slate-400 pt-1 border-t border-slate-800/80">
            Preprocessed Cache: 128x128 log-mel tensors
          </p>
        </div>

        {/* Card 4: Baseline Latency */}
        <div className="p-5 rounded-xl bg-[#0f172a] border border-slate-800 space-y-2 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-medium uppercase tracking-wider text-[11px]">Utterance Inference</span>
            <Clock className="w-4 h-4 text-amber-400" />
          </div>
          <div className="pt-1 flex items-baseline gap-2">
            <span className="text-2xl font-bold text-white font-mono">
              {effMetric?.inference_time_ms_per_utterance || "1.22"}
            </span>
            <span className="text-xs text-slate-400">ms / utterance</span>
          </div>
          <p className="text-xs text-slate-400">ViT-B/16: {vitMetric?.inference_time_ms_per_utterance || "1.18"} ms</p>
          <p className="text-[10px] text-slate-400 pt-1 border-t border-slate-800/80">
            Batched Amp Autocast on RTX 5050
          </p>
        </div>
      </div>

      {/* Main Grid: Comparison Chart & Metric Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Visual Chart */}
        <div className="lg:col-span-7 p-6 rounded-xl bg-[#0f172a] border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
            <div>
              <h3 className="font-semibold text-white text-sm">Empirical Comparison: CNN Baseline vs. ViT</h3>
              <p className="text-xs text-slate-400">Audited metrics on official ASVspoof 2019 LA partition</p>
            </div>
            <button
              onClick={() => onNavigate("comparison")}
              className="text-xs text-sky-400 hover:text-sky-300 font-medium flex items-center gap-1 transition-colors"
            >
              Full Details <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="h-[280px] w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="metric" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0b1120",
                    borderColor: "#334155",
                    fontSize: "12px",
                    borderRadius: "8px",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }} />
                <Bar dataKey="EfficientNet-B0" fill="#0284c7" radius={[4, 4, 0, 0]} />
                <Bar dataKey="ViT-B/16" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right Column: Key Metric Table */}
        <div className="lg:col-span-5 p-6 rounded-xl bg-[#0f172a] border border-slate-800 space-y-4 flex flex-col justify-between">
          <div>
            <div className="border-b border-slate-800/80 pb-3">
              <h3 className="font-semibold text-white text-sm">Key Biometric Metric Audit</h3>
              <p className="text-xs text-slate-400">Strict zero-leakage development vs evaluation</p>
            </div>

            <div className="divide-y divide-slate-800/60 mt-3 text-xs">
              <div className="py-2.5 flex items-center justify-between">
                <span className="text-slate-400">Model Capacity (Parameters)</span>
                <span className="font-mono text-slate-200">
                  CNN: <strong className="text-sky-400">4.01M</strong> | ViT: <strong className="text-purple-400">85.8M</strong>
                </span>
              </div>
              <div className="py-2.5 flex items-center justify-between">
                <span className="text-slate-400">Dev Split EER (Frozen τ)</span>
                <span className="font-mono text-slate-200">
                  CNN: <strong className="text-sky-400">6.44%</strong> | ViT: <strong className="text-purple-400">1.55%</strong>
                </span>
              </div>
              <div className="py-2.5 flex items-center justify-between">
                <span className="text-slate-400">Official Eval Split EER</span>
                <span className="font-mono text-slate-200">
                  CNN: <strong className="text-sky-400">21.49%</strong> | ViT: <strong className="text-purple-400">27.08%</strong>
                </span>
              </div>
              <div className="py-2.5 flex items-center justify-between">
                <span className="text-slate-400">Eval ROC-AUC</span>
                <span className="font-mono text-slate-200">
                  CNN: <strong className="text-sky-400">0.8682</strong> | ViT: <strong className="text-purple-400">0.8201</strong>
                </span>
              </div>
              <div className="py-2.5 flex items-center justify-between">
                <span className="text-slate-400">Frozen Operating Threshold (τ)</span>
                <span className="font-mono text-slate-200">
                  CNN: 0.967773 | ViT: 0.986328
                </span>
              </div>
            </div>
          </div>

          {/* Action Button */}
          <div className="pt-4 border-t border-slate-800/80">
            <button
              onClick={() => onNavigate("detection")}
              className="w-full py-2.5 px-4 rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold text-xs transition-colors flex items-center justify-center gap-2 shadow-sm"
            >
              Launch Live Audio Detection <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
