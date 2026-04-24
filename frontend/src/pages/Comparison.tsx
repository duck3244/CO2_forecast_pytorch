import { useMemo, useState } from "react";
import { api } from "@/api/client";
import type { EvaluationResponse, ModelName } from "@/api/types";
import {
  TimeSeriesChart,
  type SeriesPoint,
} from "@/components/charts/TimeSeriesChart";

const ALL_MODELS: ModelName[] = ["lstm", "transformer", "hybrid", "ensemble"];
const COLORS: Record<ModelName, string> = {
  lstm: "#ef4444",
  transformer: "#3b82f6",
  hybrid: "#10b981",
  ensemble: "#a855f7",
};
const METRIC_ORDER = ["R2", "MSE", "RMSE", "MAE", "MAPE"];

function formatMetric(name: string, v: number): string {
  if (!Number.isFinite(v)) return "-";
  if (name === "R2") return v.toFixed(4);
  if (name === "MAPE") return `${v.toFixed(3)}%`;
  return v.toFixed(4);
}

export function Comparison() {
  const [selected, setSelected] = useState<ModelName[]>(ALL_MODELS);
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function toggle(m: ModelName) {
    setSelected((prev) =>
      prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]
    );
  }

  async function run() {
    if (selected.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const r = await api.evaluate(selected);
      setData(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const chartData = useMemo<SeriesPoint[]>(() => {
    if (!data) return [];
    return data.dates.map((date, i) => {
      const row: SeriesPoint = {
        date,
        actual: data.actual[i],
      };
      for (const r of data.results) {
        row[r.model_name] = r.predicted[i];
      }
      return row;
    });
  }, [data]);

  const chartSeries = useMemo(() => {
    if (!data) return [];
    const s = [{ key: "actual", name: "Actual", color: "#0f172a" }];
    for (const r of data.results) {
      s.push({
        key: r.model_name,
        name: r.model_name,
        color: COLORS[r.model_name as ModelName],
      });
    }
    return s;
  }, [data]);

  const metricsShown = useMemo(() => {
    if (!data) return [];
    const all = new Set<string>();
    for (const r of data.results)
      for (const k of Object.keys(r.metrics)) all.add(k);
    return METRIC_ORDER.filter((k) => all.has(k));
  }, [data]);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold text-slate-900">Comparison</h1>
        <p className="text-sm text-slate-500 mt-1">
          Select models to compare metrics and overlay test-set predictions
        </p>
      </header>

      <div className="flex items-center justify-between gap-4 bg-white border border-slate-200 rounded-lg px-4 py-3">
        <div className="flex gap-2 flex-wrap">
          {ALL_MODELS.map((m) => {
            const active = selected.includes(m);
            return (
              <button
                key={m}
                type="button"
                onClick={() => toggle(m)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
                  active
                    ? "bg-slate-900 text-white border-slate-900"
                    : "bg-white text-slate-600 border-slate-200 hover:border-slate-400"
                }`}
              >
                {m}
              </button>
            );
          })}
        </div>
        <button
          type="button"
          onClick={run}
          disabled={selected.length === 0 || loading}
          className="px-4 py-1.5 rounded-md bg-slate-900 text-white text-sm font-medium hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "evaluating…" : "Run comparison"}
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded px-4 py-3">
          {error}
        </div>
      )}

      {data && (
        <>
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">
                Metrics
              </h2>
              {data.best_by_r2 && (
                <div className="text-xs text-slate-500">
                  best by R²:{" "}
                  <span className="font-semibold text-slate-900">
                    {data.best_by_r2}
                  </span>
                </div>
              )}
            </div>
            <div className="overflow-auto bg-white border border-slate-200 rounded-lg">
              <table className="min-w-full text-sm">
                <thead className="bg-slate-50 text-xs uppercase text-slate-500">
                  <tr>
                    <th className="text-left px-4 py-2 font-medium">model</th>
                    {metricsShown.map((k) => (
                      <th
                        key={k}
                        className="text-right px-4 py-2 font-medium"
                      >
                        {k}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {data.results.map((r) => (
                    <tr key={r.model_name}>
                      <td className="px-4 py-2 font-medium text-slate-900">
                        {r.model_name}
                        {r.model_name === data.best_by_r2 && (
                          <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-100 text-emerald-700 font-semibold">
                            BEST
                          </span>
                        )}
                      </td>
                      {metricsShown.map((k) => (
                        <td
                          key={k}
                          className="px-4 py-2 text-right tabular-nums text-slate-700"
                        >
                          {formatMetric(k, r.metrics[k])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section>
            <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-3">
              Predictions overlay
            </h2>
            <div className="bg-white rounded-lg border border-slate-200 p-4">
              <TimeSeriesChart
                data={chartData}
                series={chartSeries}
                yLabel="ppm"
                height={400}
              />
            </div>
          </section>
        </>
      )}
    </div>
  );
}
