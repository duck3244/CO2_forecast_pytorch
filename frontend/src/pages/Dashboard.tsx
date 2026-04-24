import { useEffect, useMemo, useState } from "react";
import { api } from "@/api/client";
import type { CO2Dataset, ModelInfo, ModelsResponse } from "@/api/types";
import { TimeSeriesChart } from "@/components/charts/TimeSeriesChart";
import { formatBytes, formatDateISO } from "@/lib/format";

function ModelCard({ model }: { model: ModelInfo }) {
  const trained = model.trained;
  return (
    <div className="bg-white rounded-lg border border-slate-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-sm font-semibold text-slate-900 uppercase">
          {model.name}
        </div>
        <div
          className={`text-[11px] px-2 py-0.5 rounded-full font-medium ${
            trained
              ? "bg-emerald-100 text-emerald-800"
              : "bg-slate-100 text-slate-500"
          }`}
        >
          {trained ? "trained" : "not trained"}
        </div>
      </div>
      <dl className="text-xs text-slate-600 space-y-1">
        <div className="flex justify-between">
          <dt className="text-slate-400">size</dt>
          <dd>{formatBytes(model.file_size_bytes)}</dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-slate-400">saved at</dt>
          <dd>{formatDateISO(model.saved_at)}</dd>
        </div>
      </dl>
    </div>
  );
}

export function Dashboard() {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [dataset, setDataset] = useState<CO2Dataset | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.models(), api.dataset()])
      .then(([m, d]) => {
        setModels(m);
        setDataset(d);
      })
      .catch((e: unknown) =>
        setError(e instanceof Error ? e.message : String(e))
      );
  }, []);

  const chartData = useMemo(() => {
    if (!dataset) return [];
    return dataset.dates.map((date, i) => ({
      date,
      co2: dataset.values[i],
    }));
  }, [dataset]);

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-2xl font-semibold text-slate-900">Dashboard</h1>
        <p className="text-sm text-slate-500 mt-1">
          Mauna Loa monthly CO<sub>2</sub> · trained model inventory
        </p>
      </header>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded px-4 py-3">
          {error}
        </div>
      )}

      <section>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">
            Models
          </h2>
          {models && (
            <div className="text-xs text-slate-500">
              device: <span className="font-medium text-slate-700">
                {models.device}
              </span>
              {models.device_name ? ` · ${models.device_name}` : ""}
            </div>
          )}
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {models?.models.map((m) => (
            <ModelCard key={m.name} model={m} />
          ))}
          {!models &&
            Array.from({ length: 4 }).map((_, i) => (
              <div
                key={i}
                className="bg-white rounded-lg border border-slate-200 p-4 h-24 animate-pulse"
              />
            ))}
        </div>
      </section>

      <section>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">
            CO<sub>2</sub> series
          </h2>
          {dataset && (
            <div className="text-xs text-slate-500">
              {dataset.n_records.toLocaleString()} records · {dataset.start_date}
              {" → "}
              {dataset.end_date} · source: {dataset.source}
            </div>
          )}
        </div>
        <div className="bg-white rounded-lg border border-slate-200 p-4">
          {chartData.length > 0 ? (
            <TimeSeriesChart
              data={chartData}
              series={[
                { key: "co2", name: "CO₂ (ppm)", color: "#0ea5e9" },
              ]}
              yLabel="ppm"
              height={360}
            />
          ) : (
            <div className="h-[360px] flex items-center justify-center text-slate-400 text-sm">
              loading…
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
