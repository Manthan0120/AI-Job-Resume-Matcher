import { useEffect, useState } from "react";
import { BarChart3, FileText, Briefcase, Layers } from "lucide-react";
import { api } from "../api";

function Stat({ icon: Icon, label, value }) {
  return (
    <div className="bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg p-5">
      <div className="flex items-center gap-2 text-[var(--color-text-faint)] mb-3">
        <Icon size={14} />
        <span className="text-[10px] uppercase tracking-widest font-[var(--font-mono)]">{label}</span>
      </div>
      <p className="text-3xl font-[var(--font-mono)] font-bold">{value}</p>
    </div>
  );
}

export default function Analytics() {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.analyticsMetrics().then(setMetrics).catch((e) => setError(e.message));
  }, []);

  return (
    <div>
      <header className="mb-8 flex items-center gap-2">
        <BarChart3 size={20} className="text-[var(--color-signal-mid)]" />
        <h1 className="text-2xl font-semibold">Analytics</h1>
      </header>

      {error && <p className="text-sm text-[var(--color-danger)]">{error}</p>}

      {metrics && (
        <>
          <div className="grid grid-cols-3 gap-5">
            <Stat icon={FileText} label="Resumes" value={metrics.total_resumes} />
            <Stat icon={Briefcase} label="Jobs" value={metrics.total_jobs} />
            <Stat icon={Layers} label="Document Chunks" value={metrics.total_documents} />
          </div>

          {metrics.total_documents === 0 ? (
            <p className="text-sm text-[var(--color-text-dim)] mt-6">
              No data loaded yet — head to the Data tab to index some resumes and jobs.
            </p>
          ) : (
            <p className="text-sm text-[var(--color-text-dim)] mt-6">
              Vector store contains {metrics.total_documents} chunks across {metrics.total_resumes} resumes and {metrics.total_jobs} jobs.
            </p>
          )}
        </>
      )}
    </div>
  );
}
