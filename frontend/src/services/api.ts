import type { ApiError, DatasetStats, Explanation, ExperimentCurves, Metrics, ModelInfo, ModelName, Prediction } from "../types/api";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new Error("Cannot reach the API. Start the FastAPI backend and try again.");
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail as ApiError | undefined;
    throw new Error(detail?.error ?? data.error ?? `Request failed (${response.status}).`);
  }
  return data as T;
}

export const api = {
  health: () => request<{ status: string; platform: string; available_models: number; models: ModelInfo[] }>("/health"),
  models: () => request<{ models: ModelInfo[]; minimum_article_words: number }>("/models"),
  metrics: () => request<{ models: Record<string, Metrics>; comparison: Record<string, unknown> | null; experiments: ExperimentCurves | null }>("/metrics"),
  dataset: () => request<DatasetStats>("/dataset/statistics"),
  modelMetrics: (name: ModelName) => request<Metrics>(`/model/${name}/metrics`),
  predict: (headline: string, article: string, model: ModelName) =>
    request<Prediction>("/predict", { method: "POST", body: JSON.stringify({ headline, article, model }) }),
  compare: (headline: string, article: string, model: ModelName) =>
    request<{ results: Array<Prediction | { model: string; error: string; code: string }> }>("/compare", {
      method: "POST", body: JSON.stringify({ headline, article, model }),
    }),
  explain: (headline: string, article: string, model: ModelName) =>
    request<Explanation>("/explain", { method: "POST", body: JSON.stringify({ headline, article, model }) }),
};
