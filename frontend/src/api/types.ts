// Type aliases over the auto-generated `schema.d.ts` (from FastAPI /openapi.json).
// Regenerate schema via: `npm run types:gen` (dev) or `make types` (from repo root).
//
// NOTE: a few literal unions and SSE event payloads are kept hand-written
// because OpenAPI does not model them (ModelName enum + SSE event bodies).

import type { components } from "./schema";

type Schemas = components["schemas"];

// --- datasets ---
export type CO2Dataset = Schemas["CO2Dataset"];

// --- models ---
export type ModelName = "lstm" | "transformer" | "hybrid" | "ensemble";
export type ModelInfo = Schemas["ModelInfo"];
export type ModelsResponse = Schemas["ModelsResponse"];

// --- predictions ---
export type PredictionRequest = Omit<Schemas["PredictionRequest"], "model"> & {
  model: ModelName;
};
export type PredictionResponse = Omit<
  Schemas["PredictionResponse"],
  "model_name"
> & { model_name: ModelName };

// --- evaluations ---
export type EvaluationRequest = Omit<Schemas["EvaluationRequest"], "models"> & {
  models: ModelName[];
};
export type PerModelEvaluation = Omit<
  Schemas["PerModelEvaluation"],
  "model_name"
> & { model_name: ModelName };
export type EvaluationResponse = Omit<
  Schemas["EvaluationResponse"],
  "results" | "best_by_r2"
> & {
  results: PerModelEvaluation[];
  best_by_r2: ModelName | null;
};

// --- training ---
// Strip `null` from the generated Optional[int] → number | null | undefined
// so the form state stays `number | undefined`.
type StripNull<T> = {
  [K in keyof T]: Exclude<T[K], null>;
};
export type TrainingOverrides = StripNull<Schemas["TrainingOverrides"]>;
export type TrainingJobRequest = Omit<
  Schemas["TrainingJobRequest"],
  "model"
> & { model: ModelName };
export type TrainingJobSnapshot = Omit<
  Schemas["TrainingJobSnapshot"],
  "model" | "status"
> & {
  model: ModelName;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
};

// --- SSE event payloads (not in OpenAPI) ---
export interface ProgressEvent {
  epoch: number;
  total_epochs: number;
  train_loss: number;
  val_loss: number;
  best_val_loss: number;
  val_metrics: Record<string, number>;
}

export interface LogEvent {
  message: string;
}

export interface CompletedEvent {
  reason: string;
  best_val_loss: number;
  training_time_s: number;
  saved_to: string;
}

export interface ErrorEvent {
  message: string;
  traceback?: string;
}

export interface DoneEvent {
  status: string;
  stopped_reason?: string | null;
  error?: string | null;
}
