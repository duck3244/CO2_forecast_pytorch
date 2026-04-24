import { useCallback, useMemo, useRef, useState } from "react";
import { api } from "@/api/client";
import type {
  CompletedEvent,
  ErrorEvent,
  LogEvent,
  ModelName,
  ProgressEvent,
  TrainingOverrides,
} from "@/api/types";
import {
  TimeSeriesChart,
  type SeriesPoint,
} from "@/components/charts/TimeSeriesChart";
import { ModelSelect } from "@/components/ui/ModelSelect";
import { useSSE } from "@/hooks/useSSE";

interface LossPoint extends SeriesPoint {
  date: string; // epoch as string for chart x-axis
  train: number;
  val: number;
  best: number;
}

interface LogLine {
  ts: number;
  level: "log" | "error" | "completed";
  text: string;
}

export function Training() {
  const [model, setModel] = useState<ModelName>("lstm");
  const [overrides, setOverrides] = useState<TrainingOverrides>({
    epochs: 30,
    patience: 10,
  });
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [lossHistory, setLossHistory] = useState<LossPoint[]>([]);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [status, setStatus] = useState<"idle" | "running" | "done" | "failed">(
    "idle"
  );
  const [submitError, setSubmitError] = useState<string | null>(null);
  const logScrollRef = useRef<HTMLDivElement | null>(null);

  const resetState = () => {
    setProgress(null);
    setLossHistory([]);
    setLogs([]);
    setSubmitError(null);
  };

  const appendLog = useCallback((line: LogLine) => {
    setLogs((prev) => {
      const next = [...prev, line];
      queueMicrotask(() => {
        if (logScrollRef.current) {
          logScrollRef.current.scrollTop = logScrollRef.current.scrollHeight;
        }
      });
      return next.slice(-500);
    });
  }, []);

  const sseUrl = jobId ? `/api/training/jobs/${jobId}/events` : null;

  useSSE(sseUrl, {
    enabled: Boolean(jobId),
    onEvent: ({ type, data }) => {
      if (type === "progress") {
        const p = data as ProgressEvent;
        setProgress(p);
        setLossHistory((prev) => [
          ...prev,
          {
            date: String(p.epoch),
            train: p.train_loss,
            val: p.val_loss,
            best: p.best_val_loss,
          },
        ]);
      } else if (type === "log") {
        const l = data as LogEvent;
        appendLog({ ts: Date.now(), level: "log", text: l.message });
      } else if (type === "completed") {
        const c = data as CompletedEvent;
        appendLog({
          ts: Date.now(),
          level: "completed",
          text: `completed (${c.reason}) · best_val_loss=${c.best_val_loss.toFixed(
            6
          )} · ${c.training_time_s.toFixed(1)}s · saved → ${c.saved_to}`,
        });
      } else if (type === "error") {
        const e = data as ErrorEvent;
        appendLog({ ts: Date.now(), level: "error", text: e.message });
      }
    },
    onDone: () => {
      // terminal event — mark status based on last known info
      setStatus((s) => (s === "running" ? "done" : s));
    },
  });

  async function handleStart() {
    resetState();
    setStatus("running");
    try {
      const { job_id } = await api.createTrainingJob({
        model,
        overrides: Object.fromEntries(
          Object.entries(overrides).filter(([, v]) => v !== undefined)
        ) as TrainingOverrides,
      });
      setJobId(job_id);
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
      setStatus("idle");
    }
  }

  async function handleCancel() {
    if (!jobId) return;
    try {
      await api.cancelTrainingJob(jobId);
      appendLog({
        ts: Date.now(),
        level: "log",
        text: "cancellation requested (will stop at end of current epoch)",
      });
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    }
  }

  const pct = useMemo(() => {
    if (!progress) return 0;
    return Math.round((progress.epoch / progress.total_epochs) * 100);
  }, [progress]);

  const running = status === "running";

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold text-slate-900">Training</h1>
        <p className="text-sm text-slate-500 mt-1">
          Start a training job; epoch-level loss streams over SSE
        </p>
      </header>

      <section className="bg-white border border-slate-200 rounded-lg p-4 space-y-4">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <label className="text-xs text-slate-500">model</label>
            <ModelSelect value={model} onChange={setModel} disabled={running} />
          </div>
          <NumberField
            label="epochs"
            value={overrides.epochs}
            onChange={(v) => setOverrides((o) => ({ ...o, epochs: v }))}
            disabled={running}
          />
          <NumberField
            label="patience"
            value={overrides.patience}
            onChange={(v) => setOverrides((o) => ({ ...o, patience: v }))}
            disabled={running}
          />
          <NumberField
            label="lr"
            step={0.0001}
            value={overrides.learning_rate}
            onChange={(v) =>
              setOverrides((o) => ({ ...o, learning_rate: v }))
            }
            disabled={running}
          />
          <NumberField
            label="batch size"
            value={overrides.batch_size}
            onChange={(v) => setOverrides((o) => ({ ...o, batch_size: v }))}
            disabled={running}
          />
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleStart}
            disabled={running}
            className="px-4 py-1.5 rounded-md bg-slate-900 text-white text-sm font-medium hover:bg-slate-800 disabled:opacity-50"
          >
            {running ? "running…" : "Start training"}
          </button>
          {running && (
            <button
              onClick={handleCancel}
              className="px-4 py-1.5 rounded-md border border-slate-300 text-slate-700 text-sm font-medium hover:bg-slate-50"
            >
              Cancel
            </button>
          )}
          {jobId && (
            <span className="text-xs text-slate-500 ml-2 font-mono">
              job: {jobId.slice(0, 8)}
            </span>
          )}
        </div>
        {submitError && (
          <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded px-3 py-2">
            {submitError}
          </div>
        )}
      </section>

      {(running || progress) && (
        <section className="bg-white border border-slate-200 rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="text-sm font-medium text-slate-700">Progress</div>
            <div className="text-xs text-slate-500">
              {progress
                ? `epoch ${progress.epoch} / ${progress.total_epochs}`
                : "waiting for first epoch…"}{" "}
              · status: <span className="font-medium">{status}</span>
            </div>
          </div>
          <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden">
            <div
              className="h-full bg-slate-900 transition-all"
              style={{ width: `${pct}%` }}
            />
          </div>
          {progress && (
            <div className="grid grid-cols-3 gap-3 text-xs">
              <Stat label="train loss" value={progress.train_loss.toFixed(6)} />
              <Stat label="val loss" value={progress.val_loss.toFixed(6)} />
              <Stat
                label="best val loss"
                value={progress.best_val_loss.toFixed(6)}
              />
            </div>
          )}
        </section>
      )}

      {lossHistory.length > 0 && (
        <section className="bg-white border border-slate-200 rounded-lg p-4">
          <div className="text-sm font-medium text-slate-700 mb-3">Loss</div>
          <TimeSeriesChart
            data={lossHistory}
            series={[
              { key: "train", name: "train", color: "#3b82f6" },
              { key: "val", name: "val", color: "#ef4444" },
              { key: "best", name: "best val", color: "#10b981" },
            ]}
            yLabel="MSE"
            height={260}
          />
        </section>
      )}

      {logs.length > 0 && (
        <section className="bg-white border border-slate-200 rounded-lg p-4">
          <div className="text-sm font-medium text-slate-700 mb-3">Log</div>
          <div
            ref={logScrollRef}
            className="max-h-64 overflow-auto bg-slate-900 rounded px-3 py-2 font-mono text-[11px] text-slate-100 space-y-0.5"
          >
            {logs.map((l, i) => (
              <div
                key={i}
                className={
                  l.level === "error"
                    ? "text-red-300"
                    : l.level === "completed"
                      ? "text-emerald-300"
                      : "text-slate-200"
                }
              >
                {l.text}
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
  disabled,
  step,
}: {
  label: string;
  value: number | undefined;
  onChange: (v: number | undefined) => void;
  disabled?: boolean;
  step?: number;
}) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-xs text-slate-500">{label}</label>
      <input
        type="number"
        step={step}
        disabled={disabled}
        value={value ?? ""}
        onChange={(e) => {
          const raw = e.target.value;
          onChange(raw === "" ? undefined : Number(raw));
        }}
        className="w-24 border border-slate-300 rounded px-2 py-1 text-sm disabled:bg-slate-50 disabled:text-slate-400"
      />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-50 rounded px-3 py-2">
      <div className="text-[10px] uppercase text-slate-400 tracking-wide">
        {label}
      </div>
      <div className="text-sm font-semibold text-slate-900 tabular-nums">
        {value}
      </div>
    </div>
  );
}
