"use client";

import React from "react";
import { Cpu, ShieldCheck, AlertCircle, RefreshCw } from "lucide-react";
import { NavSection } from "./sidebar";
import { SystemHealth } from "@/lib/api";

interface HeaderProps {
  currentSection: NavSection;
  health: SystemHealth | null;
  onRefreshHealth: () => void;
  loadingHealth: boolean;
}

const SECTION_TITLES: Record<NavSection, { title: string; subtitle: string }> = {
  overview: {
    title: "Research Overview",
    subtitle: "Empirical comparison of CNN Baseline and Vision Transformer on ASVspoof benchmarks",
  },
  detection: {
    title: "Audio Deepfake Detection",
    subtitle: "Real-time acoustic analysis and spoofing probability estimation via PyTorch inference",
  },
  comparison: {
    title: "Model Architecture Comparison",
    subtitle: "Direct biometric metric and capacity evaluation: EfficientNet-B0 vs. ViT-B/16",
  },
  evaluation: {
    title: "Evaluation & Diagnostic Lab",
    subtitle: "Reproducible partition testing with confusion matrix and threshold shift analysis",
  },
  robustness: {
    title: "Acoustic Robustness Suite",
    subtitle: "Stress testing against noise, volume perturbations, and telephony bandpass downsampling",
  },
  explainability: {
    title: "Explainability & Saliency Analysis",
    subtitle: "CNN Grad-CAM formants and Vision Transformer self-attention attribution heatmaps",
  },
  datasets: {
    title: "Dataset Explorer & Inventory",
    subtitle: "ASVspoof 2019 LA partition manifests and verified 2021 DF resource status",
  },
  diagnostics: {
    title: "System Diagnostics & Telemetry",
    subtitle: "CUDA GPU runtime, VRAM allocation, checkpoint integrity, and hardware health",
  },
  reports: {
    title: "Research Reports & Audit Logs",
    subtitle: "Academic reports, provenance verification, and SHA-256 checkpoint audit tables",
  },
  settings: {
    title: "System Configuration & Settings",
    subtitle: "Inference parameters, threshold source preferences, and server configuration",
  },
};

export function Header({ currentSection, health, onRefreshHealth, loadingHealth }: HeaderProps) {
  const current = SECTION_TITLES[currentSection];

  return (
    <header className="h-16 border-b border-slate-800/80 bg-[#090d16]/90 backdrop-blur-md sticky top-0 z-20 px-8 flex items-center justify-between">
      <div>
        <h1 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
          {current.title}
        </h1>
        <p className="text-xs text-slate-400 hidden sm:block">{current.subtitle}</p>
      </div>

      <div className="flex items-center gap-3">
        {/* Device Badge */}
        {health && (
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <Cpu className="w-3.5 h-3.5 text-sky-400" />
            <span className="text-slate-400">Device:</span>
            <span className="font-mono text-white font-medium">
              {health.cuda_available ? health.cuda_device_name || "CUDA" : "CPU Fallback"}
            </span>
            {health.vram_allocated_gb !== undefined && health.vram_total_gb !== undefined && (
              <span className="text-[11px] font-mono text-slate-400 border-l border-slate-800 pl-1.5 ml-1">
                {health.vram_allocated_gb} / {health.vram_total_gb} GB
              </span>
            )}
          </div>
        )}

        {/* Refresh Diagnostics */}
        <button
          onClick={onRefreshHealth}
          disabled={loadingHealth}
          title="Refresh Backend Telemetry"
          className="p-2 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-400 hover:text-white transition-all disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loadingHealth ? "animate-spin text-sky-400" : ""}`} />
        </button>
      </div>
    </header>
  );
}
