import type {
  CO2Dataset,
  EvaluationResponse,
  ModelName,
  ModelsResponse,
  PredictionResponse,
  TrainingJobRequest,
  TrainingJobSnapshot,
} from "./types";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text || path}`);
  }
  return (await res.json()) as T;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text || path}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => get<{ status: string }>("/api/health"),
  models: () => get<ModelsResponse>("/api/models"),
  dataset: (forceRefresh = false) =>
    get<CO2Dataset>(
      `/api/datasets/co2${forceRefresh ? "?force_refresh=true" : ""}`
    ),
  predict: (model: ModelName) =>
    post<PredictionResponse>("/api/predictions", { model }),
  evaluate: (models: ModelName[]) =>
    post<EvaluationResponse>("/api/evaluations", { models }),
  createTrainingJob: (req: TrainingJobRequest) =>
    post<{ job_id: string; status: string }>("/api/training/jobs", req),
  listTrainingJobs: () =>
    get<{ jobs: TrainingJobSnapshot[] }>("/api/training/jobs"),
  getTrainingJob: (id: string) =>
    get<TrainingJobSnapshot>(`/api/training/jobs/${id}`),
  cancelTrainingJob: (id: string) =>
    fetch(`/api/training/jobs/${id}`, { method: "DELETE" }).then(
      (r) =>
        r.json() as Promise<{ id: string; status: string; cancelled: boolean }>
    ),
};
