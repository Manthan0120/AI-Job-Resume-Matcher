import { useState } from "react";
import { Upload, FileText, Table2, CheckCircle2, AlertCircle } from "lucide-react";
import { api } from "../api";

function Panel({ icon: Icon, title, children }) {
  return (
    <div className="bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg p-6">
      <div className="flex items-center gap-2 mb-4">
        <Icon size={16} className="text-[var(--color-signal-mid)]" />
        <h3 className="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--color-text-dim)]">
          {title}
        </h3>
      </div>
      {children}
    </div>
  );
}

function DropZone({ accept, multiple, onFiles, hint }) {
  const [drag, setDrag] = useState(false);
  return (
    <label
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        onFiles(Array.from(e.dataTransfer.files));
      }}
      className={`flex flex-col items-center justify-center gap-2 border border-dashed rounded-md py-8 cursor-pointer transition-colors ${
        drag ? "border-[var(--color-signal-high)] bg-[var(--color-surface-raised)]" : "border-[var(--color-border)] hover:border-[var(--color-text-faint)]"
      }`}
    >
      <Upload size={20} className="text-[var(--color-text-faint)]" />
      <span className="text-sm text-[var(--color-text-dim)]">{hint}</span>
      <input
        type="file"
        accept={accept}
        multiple={multiple}
        className="hidden"
        onChange={(e) => onFiles(Array.from(e.target.files))}
      />
    </label>
  );
}

export default function DataManagement() {
  const [resumeStatus, setResumeStatus] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [busy, setBusy] = useState(false);

  async function handleResumes(files) {
    if (!files.length) return;
    setBusy(true);
    try {
      const result = await api.uploadResumes(files);
      setResumeStatus({ ok: true, ...result });
    } catch (e) {
      setResumeStatus({ ok: false, error: e.message });
    } finally {
      setBusy(false);
    }
  }

  async function handleJobs(files) {
    if (!files.length) return;
    setBusy(true);
    try {
      const result = await api.uploadJobs(files[0]);
      setJobStatus({ ok: true, ...result });
    } catch (e) {
      setJobStatus({ ok: false, error: e.message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <header className="mb-8">
        <h1 className="text-2xl font-semibold">Data</h1>
        <p className="text-[var(--color-text-dim)] text-sm mt-1">
          Load resumes and job postings into the vector index before matching against them.
        </p>
      </header>

      <div className="grid grid-cols-2 gap-5">
        <Panel icon={FileText} title="Resumes">
          <DropZone
            accept=".pdf,.docx,.txt"
            multiple
            onFiles={handleResumes}
            hint="Drop PDF / DOCX / TXT resumes, or click to browse"
          />
          {resumeStatus && (
            <div className={`mt-3 flex items-start gap-2 text-sm ${resumeStatus.ok ? "text-[var(--color-signal-high)]" : "text-[var(--color-danger)]"}`}>
              {resumeStatus.ok ? <CheckCircle2 size={16} className="mt-0.5" /> : <AlertCircle size={16} className="mt-0.5" />}
              <span>
                {resumeStatus.ok
                  ? `Indexed ${resumeStatus.processed} resume${resumeStatus.processed === 1 ? "" : "s"}: ${resumeStatus.filenames.join(", ")}`
                  : resumeStatus.error}
              </span>
            </div>
          )}
        </Panel>

        <Panel icon={Table2} title="Job Postings (CSV)">
          <DropZone
            accept=".csv"
            multiple={false}
            onFiles={handleJobs}
            hint="Drop a jobs CSV, or click to browse"
          />
          {jobStatus && (
            <div className={`mt-3 flex items-start gap-2 text-sm ${jobStatus.ok ? "text-[var(--color-signal-high)]" : "text-[var(--color-danger)]"}`}>
              {jobStatus.ok ? <CheckCircle2 size={16} className="mt-0.5" /> : <AlertCircle size={16} className="mt-0.5" />}
              <span>
                {jobStatus.ok
                  ? `Indexed ${jobStatus.processed} job${jobStatus.processed === 1 ? "" : "s"}${jobStatus.skipped ? ` (${jobStatus.skipped} skipped — insufficient content)` : ""}`
                  : jobStatus.error}
              </span>
            </div>
          )}
        </Panel>
      </div>

      <p className="text-xs text-[var(--color-text-faint)] mt-6 font-[var(--font-mono)]">
        Tip: try sample_data/sample_resume.txt and sample_data/sample_jobs.csv from the repo to test this without real data.
      </p>
      {busy && <p className="text-xs text-[var(--color-signal-mid)] mt-2">Indexing…</p>}
    </div>
  );
}
