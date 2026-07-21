import { NavLink, Outlet } from "react-router-dom";
import { useState } from "react";
import {
  Radar, FolderInput, Target, Users, Sparkles, MessageSquare, BarChart3, Key,
} from "lucide-react";
import { api } from "../api";

const NAV = [
  { to: "/", label: "Data", icon: FolderInput, end: true },
  { to: "/resume-to-jobs", label: "Resume → Jobs", icon: Target },
  { to: "/job-to-resumes", label: "Job → Resumes", icon: Users },
  { to: "/agent", label: "Agentic Match", icon: Sparkles },
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
];

export default function Shell() {
  const [apiKey, setApiKey] = useState("");
  const [keyStatus, setKeyStatus] = useState("unset"); // unset | saving | saved | error

  async function saveKey(e) {
    e.preventDefault();
    if (!apiKey) return;
    setKeyStatus("saving");
    try {
      await api.setApiKey(apiKey);
      setKeyStatus("saved");
    } catch {
      setKeyStatus("error");
    }
  }

  return (
    <div className="min-h-screen flex bg-[var(--color-bg)] text-[var(--color-text)]">
      <aside className="w-64 shrink-0 border-r border-[var(--color-border)] flex flex-col">
        <div className="px-5 py-6 border-b border-[var(--color-border)]">
          <div className="flex items-center gap-2">
            <Radar size={20} className="text-[var(--color-signal-high)]" />
            <span className="font-[var(--font-mono)] font-bold tracking-tight text-sm">
              RESUME<span className="text-[var(--color-signal-high)]">/</span>MATCH
            </span>
          </div>
          <p className="text-[10px] text-[var(--color-text-faint)] mt-1 font-[var(--font-mono)] tracking-widest uppercase">
            signal alignment engine
          </p>
        </div>

        <nav className="flex-1 py-4 px-3 space-y-1">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-md text-sm transition-colors ${
                  isActive
                    ? "bg-[var(--color-surface-raised)] text-[var(--color-text)] border border-[var(--color-border)]"
                    : "text-[var(--color-text-dim)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface)]"
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        <form onSubmit={saveKey} className="p-3 border-t border-[var(--color-border)]">
          <label className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-[var(--color-text-faint)] font-[var(--font-mono)] mb-2">
            <Key size={11} /> OpenAI API Key
          </label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => { setApiKey(e.target.value); setKeyStatus("unset"); }}
            placeholder="sk-..."
            className="w-full bg-[var(--color-surface)] border border-[var(--color-border)] rounded px-2.5 py-1.5 text-xs font-[var(--font-mono)] outline-none focus:border-[var(--color-signal-mid)]"
          />
          <button
            type="submit"
            className="mt-2 w-full text-xs py-1.5 rounded border border-[var(--color-border)] hover:border-[var(--color-signal-high)] transition-colors text-[var(--color-text-dim)] hover:text-[var(--color-signal-high)]"
          >
            {keyStatus === "saving" ? "Saving…" : keyStatus === "saved" ? "✓ Connected" : keyStatus === "error" ? "Failed — retry" : "Connect"}
          </button>
        </form>
      </aside>

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-5xl mx-auto px-8 py-10">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
