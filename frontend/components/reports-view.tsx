"use client";

import React, { useEffect, useState } from "react";
import {
  FileText,
  Download,
  ShieldCheck,
  Layers,
  ArrowRight,
  ExternalLink,
  BookOpen,
} from "lucide-react";
import {
  fetchReports,
  fetchReportDetail,
  ReportDetail,
  ReportSummary,
} from "@/lib/api";

export function ReportsView() {
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [selectedReportId, setSelectedReportId] = useState<string>("final_report");
  const [currentDetail, setCurrentDetail] = useState<ReportDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingContent, setLoadingContent] = useState(false);

  useEffect(() => {
    fetchReports()
      .then((reps) => {
        setReports(reps);
        if (reps.length > 0) {
          loadDetail(reps[0].id);
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const loadDetail = (id: string) => {
    setSelectedReportId(id);
    setLoadingContent(true);
    fetchReportDetail(id)
      .then(setCurrentDetail)
      .catch(() => {})
      .finally(() => setLoadingContent(false));
  };

  const handleDownload = () => {
    if (!currentDetail) return;
    const blob = new Blob([currentDetail.markdown_content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = currentDetail.filename;
    a.click();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[350px] text-slate-400">
        <div className="w-8 h-8 border-2 border-sky-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Header and Report Selector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column (4 cols): Reports Index */}
        <div className="lg:col-span-4 space-y-3">
          <div className="p-4 rounded-xl bg-[#0f172a] border border-slate-800">
            <h3 className="font-semibold text-white text-sm flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-sky-400" />
              Research Reports & Audits
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Select a research document to read and verify raw findings.
            </p>
          </div>

          <div className="space-y-2">
            {reports.map((r) => {
              const isSelected = selectedReportId === r.id;
              return (
                <button
                  key={r.id}
                  onClick={() => loadDetail(r.id)}
                  className={`w-full p-4 rounded-xl border text-left transition-all ${
                    isSelected
                      ? "bg-sky-500/10 border-sky-500/50 shadow-sm"
                      : "bg-[#0f172a] border-slate-800 hover:border-slate-700 hover:bg-slate-900"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                      {r.category}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400">{r.filename}</span>
                  </div>
                  <h4 className="text-xs font-bold text-white mt-2 leading-snug">{r.title}</h4>
                  <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">{r.description}</p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right Column (8 cols): Document Reader */}
        <div className="lg:col-span-8">
          {loadingContent ? (
            <div className="flex items-center justify-center min-h-[400px] text-slate-400">
              <div className="w-8 h-8 border-2 border-sky-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : currentDetail ? (
            <div className="p-6 sm:p-8 rounded-2xl bg-[#0f172a] border border-slate-800 space-y-6">
              {/* Report Header */}
              <div className="flex flex-wrap items-center justify-between border-b border-slate-800 pb-4 gap-4">
                <div>
                  <span className="text-[10px] font-mono text-sky-400 uppercase font-semibold">
                    {currentDetail.category} Document
                  </span>
                  <h2 className="text-xl font-bold text-white mt-1">{currentDetail.title}</h2>
                  <p className="text-xs text-slate-400 font-mono mt-0.5">
                    File: {currentDetail.filename} | Updated: {currentDetail.last_modified}
                  </p>
                </div>

                <button
                  onClick={handleDownload}
                  className="px-4 py-2 rounded-xl bg-slate-900 border border-slate-800 hover:border-slate-700 text-xs font-semibold text-slate-300 hover:text-white flex items-center gap-2 transition-all shadow-sm"
                >
                  <Download className="w-3.5 h-3.5 text-sky-400" /> Download Document
                </button>
              </div>

              {/* Rendered Document Content */}
              <div className="p-6 rounded-xl bg-slate-950/70 border border-slate-800/80 font-mono text-xs leading-relaxed text-slate-300 whitespace-pre-wrap overflow-x-auto max-h-[600px] overflow-y-auto">
                {currentDetail.markdown_content}
              </div>

              {/* Associated Artifacts Footer */}
              {currentDetail.associated_artifacts.length > 0 && (
                <div className="pt-3 border-t border-slate-800/80 flex flex-wrap items-center gap-2 text-xs">
                  <span className="text-slate-400 font-semibold text-[11px]">Related Artifacts:</span>
                  {currentDetail.associated_artifacts.map((art, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 font-mono text-[10px] text-slate-400"
                    >
                      {art}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="min-h-[400px] rounded-2xl bg-[#0f172a] border border-slate-800 flex items-center justify-center text-slate-400 text-xs">
              Select a research report on the left.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
