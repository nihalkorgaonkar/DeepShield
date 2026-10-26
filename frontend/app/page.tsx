"use client";

import React, { useEffect, useState } from "react";
import { Sidebar, NavSection } from "@/components/sidebar";
import { Header } from "@/components/header";
import { OverviewView } from "@/components/overview-view";
import { DetectionView } from "@/components/detection-view";
import { ComparisonView } from "@/components/comparison-view";
import { EvaluationView } from "@/components/evaluation-view";
import { RobustnessView } from "@/components/robustness-view";
import { ExplainabilityView } from "@/components/explainability-view";
import { DatasetView } from "@/components/dataset-view";
import { DiagnosticsView } from "@/components/diagnostics-view";
import { ReportsView } from "@/components/reports-view";
import { SettingsView } from "@/components/settings-view";
import { fetchHealth, SystemHealth } from "@/lib/api";

export default function DeepShieldApp() {
  const [currentSection, setCurrentSection] = useState<NavSection>("overview");
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean>(false);
  const [loadingHealth, setLoadingHealth] = useState<boolean>(false);

  const refreshHealth = async () => {
    try {
      setLoadingHealth(true);
      const h = await fetchHealth();
      setHealth(h);
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    } finally {
      setLoadingHealth(false);
    }
  };

  useEffect(() => {
    refreshHealth();
    // Periodic health telemetry check every 15 seconds
    const interval = setInterval(refreshHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#090d16] flex">
      {/* Persistent Navigation Sidebar */}
      <Sidebar
        currentSection={currentSection}
        onSelectSection={setCurrentSection}
        apiOnline={apiOnline}
        activeDevice={health?.active_device || "CPU"}
      />

      {/* Main Content Area */}
      <div className="flex-1 ml-64 flex flex-col min-h-screen">
        <Header
          currentSection={currentSection}
          health={health}
          onRefreshHealth={refreshHealth}
          loadingHealth={loadingHealth}
        />

        <main className="flex-1 p-8">
          {currentSection === "overview" && (
            <OverviewView onNavigate={setCurrentSection} />
          )}
          {currentSection === "detection" && <DetectionView />}
          {currentSection === "comparison" && <ComparisonView />}
          {currentSection === "evaluation" && <EvaluationView />}
          {currentSection === "robustness" && <RobustnessView />}
          {currentSection === "explainability" && <ExplainabilityView />}
          {currentSection === "datasets" && <DatasetView />}
          {currentSection === "diagnostics" && <DiagnosticsView />}
          {currentSection === "reports" && <ReportsView />}
          {currentSection === "settings" && <SettingsView />}
        </main>
      </div>
    </div>
  );
}
