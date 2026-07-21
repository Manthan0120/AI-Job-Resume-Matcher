// SignalRing — the signature element of this UI. A resume-to-job match
// isn't a generic progress bar; it's read as a "signal strength" the
// same way you'd read a connection indicator. Color and fill both
// encode the strength, so the ring communicates before the number does.

function colorForPct(pct) {
  if (pct >= 75) return "var(--color-signal-high)";
  if (pct >= 45) return "var(--color-signal-mid)";
  return "var(--color-signal-low)";
}

export default function SignalRing({ pct = 0, size = 64, label }) {
  const stroke = 5;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const offset = c - (Math.min(Math.max(pct, 0), 100) / 100) * c;
  const color = colorForPct(pct);

  return (
    <div className="flex flex-col items-center gap-1.5" style={{ width: size }}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke="var(--color-border)"
            strokeWidth={stroke}
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={c}
            strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 0.6s ease, stroke 0.3s ease" }}
          />
        </svg>
        <div
          className="absolute inset-0 flex items-center justify-center font-[var(--font-mono)] text-sm font-bold"
          style={{ color }}
        >
          {Math.round(pct)}
        </div>
      </div>
      {label && (
        <span className="text-[10px] tracking-widest uppercase text-[var(--color-text-faint)] font-[var(--font-mono)]">
          {label}
        </span>
      )}
    </div>
  );
}
