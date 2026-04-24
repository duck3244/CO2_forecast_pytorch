const METRIC_ORDER = ["R2", "MSE", "RMSE", "MAE", "MAPE"];

function formatMetric(name: string, value: number): string {
  if (!Number.isFinite(value)) return "-";
  if (name === "R2") return value.toFixed(4);
  if (name === "MAPE") return `${value.toFixed(3)}%`;
  return value.toFixed(4);
}

export function MetricsStrip({ metrics }: { metrics: Record<string, number> }) {
  const shown = METRIC_ORDER.filter((k) => k in metrics);
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
      {shown.map((k) => (
        <div
          key={k}
          className="bg-white border border-slate-200 rounded-md px-3 py-2"
        >
          <div className="text-[10px] uppercase tracking-wide text-slate-400">
            {k}
          </div>
          <div className="text-sm font-semibold text-slate-900 mt-0.5">
            {formatMetric(k, metrics[k])}
          </div>
        </div>
      ))}
    </div>
  );
}
