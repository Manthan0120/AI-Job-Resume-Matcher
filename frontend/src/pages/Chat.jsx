import { useState, useRef, useEffect } from "react";
import { MessageSquare, Send } from "lucide-react";
import { api } from "../api";

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    if (!input.trim() || loading) return;
    const question = input;
    const nextMessages = [...messages, { role: "user", content: question }];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);
    try {
      const result = await api.chat(question, nextMessages);
      setMessages([...nextMessages, { role: "assistant", content: result.answer, sources: result.source_documents }]);
    } catch (e) {
      setMessages([...nextMessages, { role: "assistant", content: `Error: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-5rem)]">
      <header className="mb-6 flex items-center gap-2">
        <MessageSquare size={20} className="text-[var(--color-signal-mid)]" />
        <h1 className="text-2xl font-semibold">Chat</h1>
      </header>

      <div className="flex-1 overflow-y-auto space-y-4 pr-1">
        {messages.length === 0 && (
          <p className="text-sm text-[var(--color-text-faint)]">Ask about resume-job matching, once you've loaded some data.</p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`max-w-[80%] ${m.role === "user" ? "ml-auto text-right" : ""}`}>
            <div
              className={`inline-block px-4 py-2.5 rounded-lg text-sm ${
                m.role === "user"
                  ? "bg-[var(--color-signal-mid)] text-[var(--color-bg)]"
                  : "bg-[var(--color-surface)] border border-[var(--color-border)]"
              }`}
            >
              {m.content}
            </div>
            {m.sources?.length > 0 && (
              <p className="text-[10px] text-[var(--color-text-faint)] mt-1 font-[var(--font-mono)]">
                sources: {m.sources.map((s) => s.filename || s.title).join(", ")}
              </p>
            )}
          </div>
        ))}
        {loading && <p className="text-sm text-[var(--color-text-faint)]">Searching documents…</p>}
        <div ref={bottomRef} />
      </div>

      <div className="flex items-center gap-2 mt-4 border-t border-[var(--color-border)] pt-4">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask me anything about resume-job matching..."
          className="flex-1 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-md px-4 py-2.5 text-sm outline-none focus:border-[var(--color-signal-mid)]"
        />
        <button
          onClick={send}
          disabled={loading || !input.trim()}
          className="p-2.5 rounded-md bg-[var(--color-signal-high)] text-[var(--color-bg)] disabled:opacity-40"
        >
          <Send size={16} />
        </button>
      </div>
    </div>
  );
}
