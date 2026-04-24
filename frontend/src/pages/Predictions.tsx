import { useEffect, useMemo, useState } from "react";
import { api } from "@/api/client";
import type { ModelName, PredictionResponse } from "@/api/types";
import { TimeSeriesChart } from "@/components/charts/TimeSeriesChart";
import { MetricsStrip } from "@/components/ui/MetricsStrip";
import { ModelSelect } from "@/components/ui/ModelSelect";

export function Predictions() {
  const [model, setModel] = useState<ModelName>("lstm");
  const [data, setData] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .predict(model)
      .then((r) => {
        if (!cancelled) setData(r);
      })
      .catch((e: unknown) => {
        if (!cancelled)
          setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [model]);

  const chartData = useMemo(() => {
    if (!data) return [];
    return data.dates.map((date, i) => ({
      date,
      actual: data.actual[i],
      predicted: data.predicted[i],
    }));
  }, [data]);

  return (
    <div className="space-y-6">
      <header className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Predictions</h1>
          <p className="text-sm text-slate-500 mt-1">
            Test-set first-step forecast (horizon {data?.horizon ?? 12})
          </p>
        </div>
        <ModelSelect value={model} onChange={setModel} disabled={loading} />
      </header>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded px-4 py-3">
          {error}
        </div>
      )}

      {data && <MetricsStrip metrics={data.metrics} />}

      <div className="bg-white rounded-lg border border-slate-200 p-4 relative">
        {loading && (
          <div className="absolute inset-0 bg-white/60 flex items-center justify-center text-sm text-slate-500 z-10">
            running inference…
          </div>
        )}
        {chartData.length > 0 ? (
          <TimeSeriesChart
            data={chartData}
            series={[
              { key: "actual", name: "Actual", color: "#0f172a" },
              { key: "predicted", name: "Predicted", color: "#ef4444" },
            ]}
            yLabel="ppm"
            height={400}
          />
        ) : (
          <div className="h-[400px] flex items-center justify-center text-slate-400 text-sm">
            {loading ? "" : "no data"}
          </div>
        )}
      </div>

      {data && (
        <div className="text-xs text-slate-500">
          {data.n_sequences} sequences · {data.dates[0]} → {data.dates[data.dates.length - 1]}
        </div>
      )}
    </div>
  );
}
