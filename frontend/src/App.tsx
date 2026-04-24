import { lazy, Suspense } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";

const Dashboard = lazy(() =>
  import("@/pages/Dashboard").then((m) => ({ default: m.Dashboard }))
);
const Predictions = lazy(() =>
  import("@/pages/Predictions").then((m) => ({ default: m.Predictions }))
);
const Comparison = lazy(() =>
  import("@/pages/Comparison").then((m) => ({ default: m.Comparison }))
);
const Training = lazy(() =>
  import("@/pages/Training").then((m) => ({ default: m.Training }))
);

function PageFallback() {
  return (
    <div className="flex items-center justify-center h-64 text-sm text-slate-400">
      loading…
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="predictions" element={<Predictions />} />
            <Route path="comparison" element={<Comparison />} />
            <Route path="training" element={<Training />} />
          </Route>
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
