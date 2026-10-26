"use client";

import React, { useEffect, useState } from "react";
import {
  Database,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Layers,
  HardDrive,
  ShieldCheck,
  Info,
} from "lucide-react";
import { fetchDatasets, DatasetsResponse } from "@/lib/api";

export function DatasetView() {
  const [data, setData] = useState<DatasetsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDatasets()
      .then(setData)
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

  const ds19 = data?.datasets.find((d) => d.id === "asvspoof2019_la");
  const ds21 = data?.datasets.find((d) => d.id === "asvspoof2021_df");

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Top Banner: Storage Status */}
      <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
        <div className="flex flex-wrap items-center justify-between border-b border-slate-800 pb-3 gap-2">
          <div>
            <h3 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
              <Database className="w-5 h-5 text-sky-400" />
              Dataset Storage & Preprocessed Cache Telemetry
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Strict audit of local audio benchmarks and pre-extracted spectrogram caches.
            </p>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-sky-500/15 text-sky-300 border border-sky-500/30">
            {data?.cache_available ? "Cache Active (128×128)" : "Cache Inactive"}
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
          <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
            <span className="text-slate-400 block text-[10px] uppercase">Cache Directory</span>
            <span className="text-white font-medium text-[11px] truncate block mt-0.5">
              {data?.preprocessed_cache_dir}
            </span>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
            <span className="text-slate-400 block text-[10px] uppercase">Resolution Standard</span>
            <span className="text-white font-medium text-[11px] block mt-0.5">
              128 Mel Bins × 128 Time Frames (3-Ch RGB)
            </span>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
            <span className="text-slate-400 block text-[10px] uppercase">Acoustic Segmentation</span>
            <span className="text-white font-medium text-[11px] block mt-0.5">
              4.0s Window / 2.0s Overlap (50% Hop)
            </span>
          </div>
        </div>
      </div>

      {/* Dataset 1: ASVspoof 2019 Logical Access */}
      {ds19 && (
        <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <span className="text-[11px] font-mono text-sky-400 uppercase font-semibold">
                Primary Benchmark
              </span>
              <h3 className="text-lg font-bold text-white mt-0.5">{ds19.name}</h3>
              <p className="text-xs text-slate-400 mt-1">{ds19.description}</p>
            </div>
            <span className="text-xs font-mono px-3 py-1 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" /> AVAILABLE
            </span>
          </div>

          {/* Splits Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase font-mono text-[10px]">
                  <th className="py-2.5 px-3">Split</th>
                  <th className="py-2.5 px-3">Bonafide (0)</th>
                  <th className="py-2.5 px-3">Spoof (1)</th>
                  <th className="py-2.5 px-3">Total Utterances</th>
                  <th className="py-2.5 px-3">Pre-cached Chunks</th>
                  <th className="py-2.5 px-3">Attack Algorithms</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-200">
                {ds19.splits.map((s) => (
                  <tr key={s.split} className="hover:bg-slate-900/50">
                    <td className="py-3 px-3 font-semibold text-white uppercase font-mono">{s.split}</td>
                    <td className="py-3 px-3 font-mono text-emerald-400">{s.bonafide_count.toLocaleString()}</td>
                    <td className="py-3 px-3 font-mono text-rose-400">{s.spoof_count.toLocaleString()}</td>
                    <td className="py-3 px-3 font-mono font-bold">{s.total_count.toLocaleString()}</td>
                    <td className="py-3 px-3 font-mono text-sky-400">
                      {s.cached_chunks_count ? s.cached_chunks_count.toLocaleString() : "Cached on Disk"}
                    </td>
                    <td className="py-3 px-3 font-mono text-[11px] text-slate-400">
                      {s.attack_types.join(", ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 text-[11px] text-slate-400 flex items-start gap-2">
            <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
            <span>{ds19.notes}</span>
          </div>
        </div>
      )}

      {/* Dataset 2: ASVspoof 2021 Deepfake (Documented Missing Audio) */}
      {ds21 && (
        <div className="p-6 rounded-2xl bg-[#0f172a] border border-amber-500/30 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <span className="text-[11px] font-mono text-amber-400 uppercase font-semibold">
                Cross-Dataset Generalization
              </span>
              <h3 className="text-lg font-bold text-white mt-0.5">{ds21.name}</h3>
              <p className="text-xs text-slate-400 mt-1">{ds21.description}</p>
            </div>
            <span className="text-xs font-mono px-3 py-1 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30 flex items-center gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5" /> {ds21.status}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-500/20 text-xs text-amber-200 space-y-2">
            <div className="font-semibold text-amber-300 flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-amber-400" />
              <span>Zero Fabrication Directive Compliance</span>
            </div>
            <p className="text-amber-200/90 leading-relaxed">{ds21.notes}</p>
            <p className="text-[11px] font-mono text-amber-300/80 pt-1 border-t border-amber-500/20">
              {ds21.zero_fabrication_statement}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
