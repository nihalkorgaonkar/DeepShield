"use client";

import React from "react";
import {
  ShieldAlert,
  Mic,
  GitCompare,
  FlaskConical,
  Activity,
  Layers,
  Database,
  Cpu,
  FileText,
  Settings,
  ShieldCheck,
  Radio,
} from "lucide-react";

export type NavSection =
  | "overview"
  | "detection"
  | "comparison"
  | "evaluation"
  | "robustness"
  | "explainability"
  | "datasets"
  | "diagnostics"
  | "reports"
  | "settings";

interface SidebarProps {
  currentSection: NavSection;
  onSelectSection: (section: NavSection) => void;
  apiOnline: boolean;
  activeDevice: string;
}

const NAV_ITEMS: { id: NavSection; label: string; icon: React.ElementType; badge?: string }[] = [
  { id: "overview", label: "Overview", icon: ShieldAlert },
  { id: "detection", label: "Audio Detection", icon: Mic, badge: "Live" },
  { id: "comparison", label: "Model Comparison", icon: GitCompare },
  { id: "evaluation", label: "Evaluation Lab", icon: FlaskConical },
  { id: "robustness", label: "Robustness Testing", icon: Activity },
  { id: "explainability", label: "Explainability", icon: Layers },
  { id: "datasets", label: "Dataset Explorer", icon: Database },
  { id: "diagnostics", label: "System Diagnostics", icon: Cpu },
  { id: "reports", label: "Research Reports", icon: FileText },
  { id: "settings", label: "Settings", icon: Settings },
];

export function Sidebar({ currentSection, onSelectSection, apiOnline, activeDevice }: SidebarProps) {
  return (
    <aside className="w-64 bg-[#0b1120] border-r border-slate-800/80 flex flex-col h-screen fixed left-0 top-0 z-30 select-none">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800/80">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-sky-500/10 border border-sky-500/30 flex items-center justify-center text-sky-400 shadow-sm shadow-sky-500/10">
            <ShieldCheck className="w-6 h-6 text-sky-400" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-lg tracking-tight text-white">DeepShield</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-300 border border-sky-500/30">
                v2.0
              </span>
            </div>
            <p className="text-[11px] text-slate-400">Audio Deepfake Research</p>
          </div>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <div className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
          Research Console
        </div>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = currentSection === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectSection(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                isActive
                  ? "bg-sky-500/15 text-white border border-sky-500/30 shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
              }`}
            >
              <div className="flex items-center gap-3">
                <Icon className={`w-4 h-4 ${isActive ? "text-sky-400" : "text-slate-400"}`} />
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span
                  className={`text-[10px] font-mono px-1.5 py-0.2 rounded-full ${
                    isActive
                      ? "bg-sky-500 text-slate-950 font-semibold"
                      : "bg-slate-800 text-slate-400 border border-slate-700"
                  }`}
                >
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer System Status */}
      <div className="p-4 border-t border-slate-800/80 bg-slate-950/40 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400 flex items-center gap-1.5">
            <Radio className="w-3.5 h-3.5 text-slate-400" />
            Backend API
          </span>
          <div className="flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                apiOnline ? "bg-emerald-400 shadow-sm shadow-emerald-400/50 animate-pulse" : "bg-rose-500"
              }`}
            />
            <span className="font-mono text-[11px] text-slate-300">
              {apiOnline ? "Online" : "Offline"}
            </span>
          </div>
        </div>

        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400">Compute Device</span>
          <span className="font-mono text-[11px] text-sky-400 font-medium truncate max-w-[110px]">
            {activeDevice.toUpperCase()}
          </span>
        </div>
      </div>
    </aside>
  );
}
