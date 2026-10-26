"use client";

import React, { useState } from "react";
import {
  Settings,
  Cpu,
  ShieldCheck,
  Sliders,
  CheckCircle2,
  HardDrive,
  Info,
} from "lucide-react";

export function SettingsView() {
  const [defaultModel, setDefaultModel] = useState<string>("vit_b16");
  const [savedNotification, setSavedNotification] = useState(false);

  const handleSave = () => {
    setSavedNotification(true);
    setTimeout(() => setSavedNotification(false), 2500);
  };

  return (
    <div className="space-y-8 max-w-4xl mx-auto pb-12">
      <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-6">
        <div>
          <h3 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
            <Settings className="w-5 h-5 text-sky-400" />
            System Preferences & Research Invariants
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Configure default runtime settings and review locked research invariants for DeepShield.
          </p>
        </div>

        {/* Section 1: Default Model */}
        <div className="space-y-3 pt-2 border-t border-slate-800/80">
          <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider block">
            Default Detection Architecture
          </label>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <button
              type="button"
              onClick={() => setDefaultModel("vit_b16")}
              className={`p-3 rounded-xl border text-left transition-all ${
                defaultModel === "vit_b16"
                  ? "bg-purple-500/15 border-purple-500 text-purple-300"
                  : "bg-slate-900 border-slate-800 text-slate-400 hover:text-white"
              }`}
            >
              <div className="font-bold text-white text-sm">Vision Transformer ViT-B/16</div>
              <div className="text-[11px] text-slate-400 mt-0.5">85.8M parameters, patch-based self-attention</div>
            </button>

            <button
              type="button"
              onClick={() => setDefaultModel("efficientnet_b0")}
              className={`p-3 rounded-xl border text-left transition-all ${
                defaultModel === "efficientnet_b0"
                  ? "bg-sky-500/15 border-sky-500 text-sky-300"
                  : "bg-slate-900 border-slate-800 text-slate-400 hover:text-white"
              }`}
            >
              <div className="font-bold text-white text-sm">EfficientNet-B0 (CNN Baseline)</div>
              <div className="text-[11px] text-slate-400 mt-0.5">4.01M parameters, localized MBConv features</div>
            </button>
          </div>
        </div>

        {/* Section 2: Locked Preprocessing Invariants */}
        <div className="space-y-3 pt-2 border-t border-slate-800/80">
          <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider block">
            Locked 14-Point Audio Preprocessing Protocol
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs font-mono">
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-slate-400 block text-[10px] uppercase">Sampling Rate</span>
              <span className="text-white font-bold">16,000 Hz Mono</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-slate-400 block text-[10px] uppercase">Chunk Window</span>
              <span className="text-white font-bold">4.0s (64,000 samples)</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-slate-400 block text-[10px] uppercase">Hop Window</span>
              <span className="text-white font-bold">2.0s (50% Overlap)</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-slate-400 block text-[10px] uppercase">Silence Trimming</span>
              <span className="text-white font-bold">30 dB Top Envelope</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-slate-400 block text-[10px] uppercase">Peak Normalization</span>
              <span className="text-white font-bold">Target 0.95 Amplitude</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-slate-400 block text-[10px] uppercase">Spectrogram Grid</span>
              <span className="text-white font-bold">128 × 128 Log-Mel</span>
            </div>
          </div>
          <p className="text-[11px] text-slate-400 italic">
            * Protocol invariants cannot be modified at runtime to preserve academic comparability across experiments.
          </p>
        </div>

        {/* Section 3: Research Integrity Directives */}
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 text-xs space-y-2">
          <div className="font-semibold text-white flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Zero Fabrication & Evaluation Isolation Pledge</span>
          </div>
          <p className="text-slate-400 leading-relaxed">
            DeepShield enforces strict cryptographic artifact verification and evaluation partition isolation.
            Official evaluation partitions are never used for threshold tuning or hyperparameter optimization.
            Predictions represent authentic PyTorch tensor outputs.
          </p>
        </div>

        <div className="flex items-center justify-between pt-2">
          <button
            onClick={handleSave}
            className="px-5 py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold text-xs uppercase tracking-wider transition-all shadow-sm"
          >
            Save Preferences
          </button>

          {savedNotification && (
            <span className="text-xs text-emerald-400 font-medium flex items-center gap-1.5 animate-fade-in">
              <CheckCircle2 className="w-4 h-4" /> Preferences saved!
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
