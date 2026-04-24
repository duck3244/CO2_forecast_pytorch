import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Dashboard" },
  { to: "/predictions", label: "Predictions" },
  { to: "/comparison", label: "Comparison" },
  { to: "/training", label: "Training" },
];

export function Sidebar() {
  return (
    <aside className="w-56 shrink-0 bg-white border-r border-slate-200 h-full py-6 px-4">
      <div className="px-2 mb-8">
        <div className="text-base font-semibold text-slate-900">
          CO<sub>2</sub> Forecast
        </div>
        <div className="text-xs text-slate-500 mt-0.5">MVP · v0.1</div>
      </div>
      <nav className="flex flex-col gap-1">
        {links.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.to === "/"}
            className={({ isActive }) =>
              `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                isActive
                  ? "bg-slate-900 text-white"
                  : "text-slate-600 hover:bg-slate-100"
              }`
            }
          >
            {l.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
