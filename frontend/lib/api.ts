/**
 * DeepShield Typed API Client
 * Connects frontend directly to the FastAPI backend endpoints.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export interface SystemHealth {
  status: "healthy" | "degraded" | "error";
  timestamp: string;
  python_version: string;
  torch_version: string;
  cuda_available: boolean;
  cuda_device_name?: string;
  cuda_device_count: number;
  vram_total_gb?: number;
  vram_allocated_gb?: number;
  vram_reserved_gb?: number;
  active_device: string;
  loaded_models: string[];
  dataset_status: string;
  preprocessed_cache_status: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  architecture: string;
  family: string;
  parameters: number;
  parameters_million: number;
  input_resolution: string;
  feature_extractor: string;
  classification_head: string;
  frozen_threshold: number;
  threshold_criterion: string;
  dev_eer: number;
  eval_eer?: number;
  eval_roc_auc?: number;
  avg_inference_latency_ms?: number;
  checkpoint: {
    model_name: string;
    exists: boolean;
    path: string;
    size_bytes: number;
    size_mb: number;
    sha256_hash: string;
    epoch?: number;
    dev_eer?: number;
    dev_threshold?: number;
    seed?: number;
    last_modified?: string;
  };
}

export interface MetricRecord {
  model: string;
  dataset: string;
  split: string;
  parameters_m: number;
  inference_time_ms_per_utterance: number;
  eer: number;
  eer_percent: number;
  eer_threshold: number;
  eval_threshold: number;
  accuracy: number;
  accuracy_percent: number;
  balanced_accuracy: number;
  balanced_accuracy_percent: number;
  f1_macro: number;
  precision_macro: number;
  recall_macro: number;
  bonafide_precision: number;
  bonafide_recall: number;
  bonafide_f1: number;
  spoof_precision: number;
  spoof_recall: number;
  spoof_f1: number;
  roc_auc: number;
  pr_auc: number;
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
  true_positives: number;
  false_positive_rate: number;
  false_negative_rate: number;
}

export interface MetricsSummary {
  efficientnet_baseline?: MetricRecord;
  vit_b16_model?: MetricRecord;
  discrepancy_analysis: Record<string, string>;
  discrepancy_warning: boolean;
  warning_details?: string;
}

export interface ChunkScore {
  chunk_index: number;
  start_time_sec: number;
  end_time_sec: number;
  spoof_score: number;
  bonafide_score: number;
  is_spoof: boolean;
}

export interface AudioMetadata {
  filename: string;
  duration_sec: number;
  sample_rate_hz: number;
  num_channels: number;
  num_samples: number;
  file_size_bytes: number;
  file_size_formatted: string;
  rms_energy: number;
  waveform_downsampled: number[];
}

export interface DetectionResult {
  model_id: string;
  model_name: string;
  prediction: "bonafide" | "spoof";
  prediction_label: string;
  spoof_score: number;
  bonafide_score: number;
  confidence_percent: number;
  configured_threshold: number;
  is_above_threshold: boolean;
  total_inference_time_ms: number;
  num_chunks_analyzed: number;
  chunk_scores: ChunkScore[];
  audio_metadata: AudioMetadata;
  preprocessing_summary: Record<string, string>;
  disclaimer: string;
}

export interface DetectionComparisonResult {
  efficientnet_result: DetectionResult;
  vit_result: DetectionResult;
  agreement: boolean;
  agreement_status: "AGREEMENT" | "DISAGREEMENT";
  disagreement_warning?: string;
  faster_model: string;
  latency_difference_ms: number;
  audio_metadata: AudioMetadata;
  disclaimer: string;
}

export interface ExplainResult {
  spectrogram_image: string;
  efficientnet_gradcam_image?: string;
  efficientnet_overlay_image?: string;
  vit_attention_image?: string;
  vit_overlay_image?: string;
  insights: {
    chunk_analyzed: number;
    total_chunks: number;
    efficientnet_focus: string;
    vit_focus: string;
    disclaimer: string;
  };
}

export interface PerturbationResult {
  condition_name: string;
  description: string;
  parameter_description: string;
  spoof_score: number;
  bonafide_score: number;
  prediction: string;
  score_delta_from_clean: number;
  waveform_downsampled: number[];
}

export interface RobustnessTestResult {
  filename: string;
  model_id: string;
  clean_score: number;
  clean_prediction: string;
  perturbations: PerturbationResult[];
  benchmark_dataset_summary?: Record<string, Record<string, number>>;
  disclaimer: string;
}

export interface DatasetSplit {
  split: string;
  available: boolean;
  bonafide_count: number;
  spoof_count: number;
  total_count: number;
  cached_chunks_count?: number;
  has_manifest: boolean;
  audio_dir_exists: boolean;
  protocol_file_exists: boolean;
  attack_types: string[];
}

export interface DatasetInfo {
  id: string;
  name: string;
  description: string;
  status: string;
  data_root: string;
  splits: DatasetSplit[];
  notes: string;
  zero_fabrication_statement?: string;
}

export interface DatasetsResponse {
  datasets: DatasetInfo[];
  preprocessed_cache_dir: string;
  cache_available: boolean;
}

export interface AudioSample {
  filename: string;
  label: "bonafide" | "spoof";
  label_display: string;
  size_bytes: number;
  url: string;
}

export interface EvaluationRunResult {
  run_id: string;
  timestamp: string;
  model: string;
  dataset: string;
  split: string;
  total_utterances: number;
  bonafide_count: number;
  spoof_count: number;
  threshold_used: number;
  threshold_source: string;
  eer: number;
  eer_percent: number;
  eer_threshold: number;
  accuracy: number;
  accuracy_percent: number;
  balanced_accuracy: number;
  balanced_accuracy_percent: number;
  macro_f1: number;
  macro_precision: number;
  macro_recall: number;
  roc_auc: number;
  pr_auc: number;
  confusion_matrix: {
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    true_positives: number;
    fpr: number;
    fnr: number;
  };
  avg_latency_ms_per_utterance: number;
  total_duration_sec: number;
  score_distribution: Record<string, Record<string, number>>;
  warnings: string[];
}

export interface ReportSummary {
  id: string;
  title: string;
  category: string;
  description: string;
  filename: string;
  last_modified: string;
  size_bytes: number;
}

export interface ReportDetail {
  id: string;
  title: string;
  category: string;
  filename: string;
  markdown_content: string;
  last_modified: string;
  associated_artifacts: string[];
}

export interface DiagnosticCheckItem {
  name: string;
  category: string;
  status: "PASS" | "WARN" | "FAIL" | "INFO";
  message: string;
  details?: Record<string, any>;
}

export interface FullDiagnostics {
  overall_health: string;
  timestamp: string;
  system_health: SystemHealth;
  checks: DiagnosticCheckItem[];
  recent_errors: string[];
}

export async function fetchHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error("Backend connection failed.");
  return res.json();
}

export async function fetchModels(): Promise<Record<string, ModelInfo>> {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) throw new Error("Failed to load model specifications.");
  return res.json();
}

export async function fetchMetricsSummary(): Promise<MetricsSummary> {
  const res = await fetch(`${API_BASE}/api/metrics/summary`);
  if (!res.ok) throw new Error("Failed to load metrics summary.");
  return res.json();
}

export async function fetchDatasets(): Promise<DatasetsResponse> {
  const res = await fetch(`${API_BASE}/api/datasets`);
  if (!res.ok) throw new Error("Failed to fetch dataset inventory.");
  return res.json();
}

export async function fetchSamples(): Promise<AudioSample[]> {
  const res = await fetch(`${API_BASE}/api/samples`);
  if (!res.ok) throw new Error("Failed to load sample library.");
  return res.json();
}

export async function detectAudio(
  file: File | Blob,
  fileName: string,
  model: string = "vit_b16",
  threshold?: number
): Promise<DetectionResult> {
  const form = new FormData();
  form.append("file", file, fileName);
  form.append("model", model);
  if (threshold !== undefined) form.append("threshold", threshold.toString());

  const res = await fetch(`${API_BASE}/api/detect`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Inference failed" }));
    throw new Error(err.detail || "Audio detection failed.");
  }
  return res.json();
}

export async function detectAndCompare(
  file: File | Blob,
  fileName: string,
  thresholdEff?: number,
  thresholdVit?: number
): Promise<DetectionComparisonResult> {
  const form = new FormData();
  form.append("file", file, fileName);
  if (thresholdEff !== undefined) form.append("threshold_eff", thresholdEff.toString());
  if (thresholdVit !== undefined) form.append("threshold_vit", thresholdVit.toString());

  const res = await fetch(`${API_BASE}/api/detect/compare`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Comparison failed" }));
    throw new Error(err.detail || "Comparative detection failed.");
  }
  return res.json();
}

export async function explainAudio(
  file: File | Blob,
  fileName: string,
  chunkIndex: number = 0
): Promise<ExplainResult> {
  const form = new FormData();
  form.append("file", file, fileName);
  form.append("chunk_index", chunkIndex.toString());

  const res = await fetch(`${API_BASE}/api/explain`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Explanation failed" }));
    throw new Error(err.detail || "Failed to generate visual explanation.");
  }
  return res.json();
}

export async function testRobustness(
  file: File | Blob,
  fileName: string,
  model: string = "vit_b16"
): Promise<RobustnessTestResult> {
  const form = new FormData();
  form.append("file", file, fileName);
  form.append("model", model);

  const res = await fetch(`${API_BASE}/api/robustness/test`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Robustness test failed" }));
    throw new Error(err.detail || "Failed to execute robustness suite.");
  }
  return res.json();
}

export async function startEvaluation(
  model: string,
  split: string,
  maxSamples: number = 100,
  threshold?: number
): Promise<{ run_id: string; status: string; message: string }> {
  const res = await fetch(`${API_BASE}/api/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      split,
      max_samples: maxSamples,
      threshold,
      dataset: "ASVspoof2019_LA",
    }),
  });
  if (!res.ok) throw new Error("Failed to start evaluation.");
  return res.json();
}

export async function fetchEvaluationStatus(runId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/api/evaluate/${runId}`);
  if (!res.ok) throw new Error(`Run ${runId} not found.`);
  return res.json();
}

export async function listEvaluations(): Promise<EvaluationRunResult[]> {
  const res = await fetch(`${API_BASE}/api/evaluate`);
  if (!res.ok) throw new Error("Failed to list evaluations.");
  return res.json();
}

export async function fetchReports(): Promise<ReportSummary[]> {
  const res = await fetch(`${API_BASE}/api/reports`);
  if (!res.ok) throw new Error("Failed to load reports list.");
  return res.json();
}

export async function fetchReportDetail(reportId: string): Promise<ReportDetail> {
  const res = await fetch(`${API_BASE}/api/reports/${reportId}`);
  if (!res.ok) throw new Error(`Failed to load report '${reportId}'.`);
  return res.json();
}

export async function fetchDiagnostics(): Promise<FullDiagnostics> {
  const res = await fetch(`${API_BASE}/api/diagnostics`);
  if (!res.ok) throw new Error("Failed to load diagnostics.");
  return res.json();
}
