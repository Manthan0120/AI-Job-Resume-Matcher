import { ChevronDown } from "lucide-react";
import { useState } from "react";
import SignalRing from "./SignalRing";

export default function MatchCard({ title, subtitle, pct, matching = [], missing = [], recommendations = [], assessment }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-4 p-4 text-left hover:bg-[var(--color-surface-raised)] transition-colors"
      >
        <SignalRing pct={pct} size={48} />
        <div className="flex-1 min-w-0">
          <p className="font-medium truncate">{title}</p>
          {subtitle && <p className="text-sm text-[var(--color-text-dim)] truncate">{subtitle}</p>}
        </div>
        <ChevronDown size={16} className={`text-[var(--color-text-faint)] transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <div className="px-5 pb-5 pt-1 border-t border-[var(--color-border)] grid grid-cols-2 gap-6 text-sm">
          <div>
            <p className="font-[var(--font-mono)] text-[10px] uppercase tracking-widest text-[var(--color-signal-high)] mb-2">Matching</p>
            <ul className="space-y-1">
              {matching.length ? matching.slice(0, 6).map((s, i) => <li key={i} className="text-[var(--color-text-dim)]">• {s}</li>) : <li className="text-[var(--color-text-faint)]">None surfaced</li>}
            </ul>
          </div>
          <div>
            <p className="font-[var(--font-mono)] text-[10px] uppercase tracking-widest text-[var(--color-signal-low)] mb-2">Gaps</p>
            <ul className="space-y-1">
              {missing.length ? missing.slice(0, 6).map((s, i) => <li key={i} className="text-[var(--color-text-dim)]">• {s}</li>) : <li className="text-[var(--color-text-faint)]">None surfaced</li>}
            </ul>
          </div>
          {recommendations.length > 0 && (
            <div className="col-span-2">
              <p className="font-[var(--font-mono)] text-[10px] uppercase tracking-widest text-[var(--color-signal-mid)] mb-2">Recommendations</p>
              <ul className="space-y-1">
                {recommendations.slice(0, 3).map((r, i) => <li key={i} className="text-[var(--color-text-dim)]">• {r}</li>)}
              </ul>
            </div>
          )}
          {assessment && (
            <p className="col-span-2 text-[var(--color-text-dim)] leading-relaxed">{assessment}</p>
          )}
        </div>
      )}
    </div>
  );
}
