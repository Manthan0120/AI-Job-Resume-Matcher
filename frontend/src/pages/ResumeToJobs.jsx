import { useState } from "react";
import { Target } from "lucide-react";
import { api } from "../api";
import MatchCard from "../components/MatchCard";

export default function ResumeToJobs() {
  const [text, setText] = useState("");
  const [topK, setTopK] = useState(5);
  const [matches, setMatches] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function run() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const { matches } = await api.matchResumeToJobs(text, topK);
      setMatches(matches);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <header className="mb-8 flex items-center gap-2">
        <Target size={20} className="text-[var(--color-signal-mid)]" />
        <div>
          <h1 className="text-2xl font-semibold">Find Jobs for Resume</h1>
          <p className="text-[var(--color-text-dim)] text-sm mt-1">Fixed pipeline: vector search → threshold filter → LLM analysis.</p>
        </div>
      </header>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste resume content here..."
        rows={8}
        className="w-full bg-[var(--color-surface)] border border-[var(--color-border)] rounded-md p-4 text-sm outline-none focus:border-[var(--color-signal-mid)] resize-none"
      />

      <div className="flex items-center gap-4 mt-4">
        <label className="flex items-center gap-2 text-sm text-[var(--color-text-dim)]">
          Matches
          <input
            type="number" min={1} max={10} value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-14 bg-[var(--color-surface)] border border-[var(--color-border)] rounded px-2 py-1 text-center"
          />
        </label>
        <button
          onClick={run}
          disabled={loading || !text.trim()}
          className="ml-auto px-4 py-2 rounded-md bg-[var(--color-signal-high)] text-[var(--color-bg)] text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? "Searching…" : "Find Job Matches"}
        </button>
      </div>

      {error && <p className="text-sm text-[var(--color-danger)] mt-4">{error}</p>}

      {matches && (
        <div className="mt-8 space-y-3">
          {matches.length === 0 && <p className="text-sm text-[var(--color-text-dim)]">No matches above the similarity threshold.</p>}
          {matches.map((m, i) => (
            <MatchCard
              key={i}
              title={m.title}
              subtitle={m.company}
              pct={m.similarity_score * 100}
              matching={m.analysis?.matching_skills}
              missing={m.analysis?.missing_qualifications}
              recommendations={m.analysis?.recommendations}
            />
          ))}
        </div>
      )}
    </div>
  );
}
