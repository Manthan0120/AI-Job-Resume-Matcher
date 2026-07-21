import { useState } from "react";
import { Sparkles, PenLine } from "lucide-react";
import { api } from "../api";
import SignalRing from "../components/SignalRing";
import MatchCard from "../components/MatchCard";

export default function AgenticMatch() {
  const [text, setText] = useState("");
  const [topK, setTopK] = useState(3);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function run() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.agentCareerMatch(text, topK);
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <header className="mb-8 flex items-center gap-2">
        <Sparkles size={20} className="text-[var(--color-signal-high)]" />
        <div>
          <h1 className="text-2xl font-semibold">Agentic Career Match</h1>
          <p className="text-[var(--color-text-dim)] text-sm mt-1">
            The model plans its own tool calls — search, extraction, gap verification, bullet rewrites —
            looping until it's confident, rather than following one fixed sequence.
          </p>
        </div>
      </header>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste resume content here..."
        rows={8}
        className="w-full bg-[var(--color-surface)] border border-[var(--color-border)] rounded-md p-4 text-sm outline-none focus:border-[var(--color-signal-high)] resize-none"
      />

      <div className="flex items-center gap-4 mt-4">
        <label className="flex items-center gap-2 text-sm text-[var(--color-text-dim)]">
          Matches
          <input
            type="number" min={1} max={5} value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-14 bg-[var(--color-surface)] border border-[var(--color-border)] rounded px-2 py-1 text-center"
          />
        </label>
        <button
          onClick={run}
          disabled={loading || !text.trim()}
          className="ml-auto px-4 py-2 rounded-md bg-[var(--color-signal-high)] text-[var(--color-bg)] text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? "Agent is working…" : "Run Career Agent"}
        </button>
      </div>

      {loading && (
        <p className="text-xs text-[var(--color-text-faint)] mt-3 font-[var(--font-mono)]">
          watch your backend terminal — you'll see search_jobs, extract_skills, compare_skills,
          verify_claim being called as the model decides its own next step.
        </p>
      )}

      {error && <p className="text-sm text-[var(--color-danger)] mt-4">{error}</p>}

      {result?.raw_output && (
        <div className="mt-6 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-md p-4 text-sm text-[var(--color-text-dim)]">
          <p className="text-[10px] uppercase tracking-widest text-[var(--color-signal-low)] font-[var(--font-mono)] mb-2">
            Agent returned unstructured output
          </p>
          {result.raw_output}
        </div>
      )}

      {result?.top_matches && (
        <div className="mt-8 space-y-3">
          {result.top_matches.length === 0 && (
            <p className="text-sm text-[var(--color-text-dim)]">Agent found no matches.</p>
          )}
          {result.top_matches.map((m, i) => (
            <MatchCard
              key={i}
              title={m.title}
              subtitle={m.company}
              pct={m.match_percentage}
              matching={m.matching_skills}
              missing={[...(m.missing_skills || []), ...(m.verified_gaps || [])]}
              recommendations={m.recommendations}
            />
          ))}
        </div>
      )}

      {result?.suggested_bullet_rewrites?.length > 0 && (
        <div className="mt-8">
          <div className="flex items-center gap-2 mb-3">
            <PenLine size={14} className="text-[var(--color-signal-mid)]" />
            <h3 className="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--color-text-dim)]">
              Suggested Bullet Rewrites
            </h3>
          </div>
          <div className="space-y-3">
            {result.suggested_bullet_rewrites.map((rw, i) => (
              <div key={i} className="bg-[var(--color-surface)] border border-[var(--color-border)] rounded-md p-4 text-sm space-y-2">
                <p className="text-[var(--color-text-faint)] line-through decoration-[var(--color-danger)]/50">{rw.original}</p>
                <p className="text-[var(--color-signal-high)]">{rw.rewritten}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {result?.overall_assessment && (
        <div className="mt-8 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-md p-5">
          <h3 className="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--color-text-dim)] mb-2">
            Overall Assessment
          </h3>
          <p className="text-sm text-[var(--color-text-dim)] leading-relaxed">{result.overall_assessment}</p>
        </div>
      )}
    </div>
  );
}
