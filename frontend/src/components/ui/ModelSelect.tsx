import type { ModelName } from "@/api/types";

const OPTIONS: { value: ModelName; label: string }[] = [
  { value: "lstm", label: "LSTM" },
  { value: "transformer", label: "Transformer" },
  { value: "hybrid", label: "Hybrid" },
  { value: "ensemble", label: "Ensemble" },
];

interface ModelSelectProps {
  value: ModelName;
  onChange: (v: ModelName) => void;
  disabled?: boolean;
}

export function ModelSelect({ value, onChange, disabled }: ModelSelectProps) {
  return (
    <div className="inline-flex rounded-md bg-slate-100 p-1">
      {OPTIONS.map((opt) => (
        <button
          key={opt.value}
          type="button"
          disabled={disabled}
          onClick={() => onChange(opt.value)}
          className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
            value === opt.value
              ? "bg-white text-slate-900 shadow-sm"
              : "text-slate-600 hover:text-slate-900"
          } disabled:opacity-50 disabled:cursor-not-allowed`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
