"use client";

import React, { useEffect, useState, useRef } from "react";
import {
  FlaskConical,
  Play,
  Download,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Layers,
  Database,
  RefreshCw,
  FileSpreadsheet,
} from "lucide-react";
import {
  startEvaluation,
  fetchEvaluationStatus,
  listEvaluations,
  EvaluationRunResult,
} from "@/lib/api";

export function EvaluationView() {
  const [model, setModel] = useState<string>("vit_b16");
  const [split, setSplit] = useState<string>("dev");
  const [maxSamples, setMaxSamples] = useState<number>(50);
  const [customThreshold, setCustomThreshold] = useState<string>("");

  const [isRunning, setIsRunning] = useState(false);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ percent: number; processed: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [activeResult, setActiveResult] = useState<EvaluationRunResult | null>(null);
  const [historicalRuns, setHistoricalRuns] = useState<EvaluationRunResult[]>([]);

  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    listEvaluations()
      .then(setHistoricalRuns)
      .catch(() => {});

    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  const handleStartRun = async () => {
    setError(null);
    setIsRunning(true);
    setProgress({ percent: 0, processed: 0, total: maxSamples });
    setActiveResult(null);

    try {
      const th = customThreshold.trim() !== "" ? parseFloat(customThreshold) : undefined;
      const { run_id } = await startEvaluation(model, split, maxSamples, th);
      setActiveRunId(run_id);

      // Start polling status
      pollIntervalRef.current = setInterval(async () => {
        try {
          const data = await fetchEvaluationStatus(run_id);
          if (data.status === "completed" || data.total_utterances !== undefined) {
            // Completed!
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
            setIsRunning(false);
            setActiveResult(data);
            listEvaluations().then(setHistoricalRuns).catch(() => {});
          } else if (data.status === "failed") {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
            setIsRunning(false);
            setError(data.error_message || "Evaluation run failed.");
          } else {
            setProgress({
              percent: data.percent_complete || 0,
              processed: data.processed_samples || 0,
              total: data.total_samples || maxSamples,
            });
          }
        } catch (err: any) {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          setIsRunning(false);
          setError("Lost connection to evaluation task.");
        }
      }, 1000);
    } catch (err: any) {
      setIsRunning(false);
      setError(err.message || "Failed to start evaluation.");
    }
  };

  const exportAsJSON = () => {
    if (!activeResult) return;
    const blob = new Blob([JSON.stringify(activeResult, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `deepshield_${activeResult.run_id}.json`;
    a.click();
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Control Panel */}
      <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-6">
        <div>
          <h3 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
            <FlaskConical className="w-5 h-5 text-sky-400" />
            Evaluation Lab & Diagnostic Workbench
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Execute reproducible diagnostic runs on ASVspoof 2019 partitions. Results are recorded as isolated run artifacts.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          {/* Model Choice */}
          <div>
            <label className="text-slate-400 font-semibold block mb-1.5 uppercase tracking-wider text-[11px]">
              Model Checkpoint
            </label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={isRunning}
              className="w-full p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-white font-medium focus:border-sky-500 focus:outline-none"
            >
              <option value="vit_b16">ViT-B/16 (Vision Transformer)</option>
              <option value="efficientnet_b0">EfficientNet-B0 (CNN Baseline)</option>
            </select>
          </div>

          {/* Dataset Split */}
          <div>
            <label className="text-slate-400 font-semibold block mb-1.5 uppercase tracking-wider text-[11px]">
              Dataset Split
            </label>
            <select
              value={split}
              onChange={(e) => setSplit(e.target.value)}
              disabled={isRunning}
              className="w-full p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-white font-medium focus:border-sky-500 focus:outline-none"
            >
              <option value="dev">Development Split (dev)</option>
              <option value="eval">Evaluation Split (eval)</option>
            </select>
          </div>

          {/* Subset Sample Limit */}
          <div>
            <label className="text-slate-400 font-semibold block mb-1.5 uppercase tracking-wider text-[11px]">
              Sample Limit (Stratified)
            </label>
            <select
              value={maxSamples}
              onChange={(e) => setMaxSamples(parseInt(e.target.value))}
              disabled={isRunning}
              className="w-full p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-white font-medium focus:border-sky-500 focus:outline-none"
            >
              <option value="20">20 Samples (Fast Diagnostic)</option>
              <option value="50">50 Samples (Standard Verification)</option>
              <option value="100">100 Samples (Detailed Benchmark)</option>
              <option value="250">250 Samples (Deep Audit)</option>
            </select>
          </div>

          {/* Custom Threshold */}
          <div>
            <label className="text-slate-400 font-semibold block mb-1.5 uppercase tracking-wider text-[11px]">
              Custom Decision Threshold (Optional)
            </label>
            <input
              type="text"
              placeholder="Default: Frozen Dev τ"
              value={customThreshold}
              onChange={(e) => setCustomThreshold(e.target.value)}
              disabled={isRunning}
              className="w-full p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-white placeholder-slate-400 focus:border-sky-500 focus:outline-none"
            >
            </input>
          </div>
        </div>

        {/* Action Button & Live Progress */}
        <div className="pt-2 flex flex-col sm:flex-row items-center justify-between gap-4">
          <button
            onClick={handleStartRun}
            disabled={isRunning}
            className={`w-full sm:w-auto px-6 py-2.5 rounded-xl font-bold text-xs uppercase tracking-wider transition-all flex items-center justify-center gap-2 ${
              isRunning
                ? "bg-slate-800 text-slate-400 cursor-not-allowed border border-slate-700"
                : "bg-sky-500 hover:bg-sky-400 text-slate-950 shadow-md shadow-sky-500/20"
            }`}
          >
            {isRunning ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Evaluating Batches...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                Execute Diagnostic Evaluation Run
              </>
            )}
          </button>

          {isRunning && progress && (
            <div className="w-full sm:w-72 space-y-1">
              <div className="flex justify-between text-xs text-slate-400 font-mono">
                <span>Evaluating Audio Chunks...</span>
                <span>{progress.processed} / {progress.total} ({progress.percent}%)</span>
              </div>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                <div
                  className="bg-sky-500 h-full rounded-full transition-all duration-300"
                  style={{ width: `${progress.percent}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-400 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Completed Run Results Display */}
      {activeResult && (
        <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-6">
          <div className="flex flex-wrap items-center justify-between border-b border-slate-800/80 pb-4 gap-4">
            <div>
              <span className="text-[11px] font-mono text-sky-400 uppercase font-semibold">
                Run ID: {activeResult.run_id} | Split: {activeResult.split.toUpperCase()}
              </span>
              <h3 className="text-xl font-bold text-white mt-0.5">
                Diagnostic Results: {activeResult.model === "vit_b16" ? "ViT-B/16" : "EfficientNet-B0"}
              </h3>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={exportAsJSON}
                className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white text-xs font-medium flex items-center gap-1.5 transition-all"
              >
                <Download className="w-3.5 h-3.5 text-sky-400" /> Export JSON
              </button>
            </div>
          </div>

          {/* Metric KPI Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
              <span className="text-[10px] text-slate-400 uppercase font-mono block">Equal Error Rate (EER)</span>
              <span className="text-2xl font-bold font-mono text-sky-400">
                {activeResult.eer_percent}%
              </span>
              <span className="text-[10px] text-slate-400 block font-mono">
                EER Operating τ: {activeResult.eer_threshold.toExponential(4)}
              </span>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
              <span className="text-[10px] text-slate-400 uppercase font-mono block">ROC-AUC Score</span>
              <span className="text-2xl font-bold font-mono text-purple-400">
                {activeResult.roc_auc.toFixed(4)}
              </span>
              <span className="text-[10px] text-slate-400 block font-mono">
                PR-AUC: {activeResult.pr_auc.toFixed(4)}
              </span>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
              <span className="text-[10px] text-slate-400 uppercase font-mono block">Balanced Accuracy</span>
              <span className="text-2xl font-bold font-mono text-emerald-400">
                {activeResult.balanced_accuracy_percent}%
              </span>
              <span className="text-[10px] text-slate-400 block font-mono">
                Raw Accuracy: {activeResult.accuracy_percent}%
              </span>
            </div>

            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
              <span className="text-[10px] text-slate-400 uppercase font-mono block">Macro F1 Score</span>
              <span className="text-2xl font-bold font-mono text-amber-400">
                {activeResult.macro_f1.toFixed(4)}
              </span>
              <span className="text-[10px] text-slate-400 block font-mono">
                τ Used: {activeResult.threshold_used}
              </span>
            </div>
          </div>

          {/* Confusion Matrix */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
            <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Confusion Matrix ({activeResult.total_utterances} Utterances)
            </h4>
            <div className="grid grid-cols-2 gap-3 text-center text-xs font-mono">
              <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                <span className="text-[10px] text-emerald-400 block font-sans">True Negatives (TN)</span>
                <span className="text-xl font-bold text-white">{activeResult.confusion_matrix.true_negatives}</span>
                <span className="text-[10px] text-slate-400 block">Bonafide Correct</span>
              </div>
              <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20">
                <span className="text-[10px] text-rose-400 block font-sans">False Positives (FP)</span>
                <span className="text-xl font-bold text-white">{activeResult.confusion_matrix.false_positives}</span>
                <span className="text-[10px] text-slate-400 block">FPR: {(activeResult.confusion_matrix.fpr * 100).toFixed(2)}%</span>
              </div>
              <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20">
                <span className="text-[10px] text-rose-400 block font-sans">False Negatives (FN)</span>
                <span className="text-xl font-bold text-white">{activeResult.confusion_matrix.false_negatives}</span>
                <span className="text-[10px] text-slate-400 block">FNR: {(activeResult.confusion_matrix.fnr * 100).toFixed(2)}%</span>
              </div>
              <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                <span className="text-[10px] text-emerald-400 block font-sans">True Positives (TP)</span>
                <span className="text-xl font-bold text-white">{activeResult.confusion_matrix.true_positives}</span>
                <span className="text-[10px] text-slate-400 block">Spoofs Detected</span>
              </div>
            </div>
          </div>

          {/* Warnings & Diagnostics */}
          {activeResult.warnings.length > 0 && (
            <div className="space-y-2">
              {activeResult.warnings.map((w, idx) => (
                <div key={idx} className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-200 text-xs flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <span>{w}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Historical Runs List */}
      {historicalRuns.length > 0 && (
        <div className="p-6 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-4">
          <div className="border-b border-slate-800 pb-3 flex items-center justify-between">
            <div>
              <h4 className="font-semibold text-white text-sm">Completed Diagnostic Runs History</h4>
              <p className="text-xs text-slate-400">Persisted in results/eval_runs/*.json</p>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px]">
                  <th className="py-2.5 px-3">Run ID</th>
                  <th className="py-2.5 px-3">Model</th>
                  <th className="py-2.5 px-3">Split</th>
                  <th className="py-2.5 px-3">Samples</th>
                  <th className="py-2.5 px-3">EER</th>
                  <th className="py-2.5 px-3">ROC-AUC</th>
                  <th className="py-2.5 px-3">Balanced Acc</th>
                  <th className="py-2.5 px-3">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {historicalRuns.map((r) => (
                  <tr key={r.run_id} className="hover:bg-slate-900/50">
                    <td className="py-2.5 px-3 font-semibold text-sky-400">{r.run_id}</td>
                    <td className="py-2.5 px-3">{r.model}</td>
                    <td className="py-2.5 px-3">{r.split}</td>
                    <td className="py-2.5 px-3">{r.total_utterances}</td>
                    <td className="py-2.5 px-3 text-purple-400">{r.eer_percent}%</td>
                    <td className="py-2.5 px-3">{r.roc_auc.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-emerald-400">{r.balanced_accuracy_percent}%</td>
                    <td className="py-2.5 px-3">
                      <button
                        onClick={() => setActiveResult(r)}
                        className="text-xs text-sky-400 hover:text-sky-300 font-sans font-medium"
                      >
                        Inspect
                      </button>
                    </td>
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
