"use client";

import { useMemo, useState } from "react";
import { useEffect } from "react";
import SourceModal from "../components/source-modal";
import { getSource, queryRag, sendFeedback } from "../lib/api";
import type { QueryResponse, SourcePreview } from "../lib/types";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type ChatMsg =
  | { role: "user"; content: string; at: string }
  | { role: "assistant"; content: string; at: string; response: QueryResponse };

function newSessionId(): string {
  return crypto.randomUUID();
}

export default function Page() {
  const [sessionId, setSessionId] = useState<string>("");
  const [input, setInput] = useState("");
  const [verbose, setVerbose] = useState(false);
  const [useWeb, setUseWeb] = useState(false);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [sourceData, setSourceData] = useState<SourcePreview | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  useEffect(() => {
    setSessionId(newSessionId());
  }, []);

  async function onAsk() {
    const question = input.trim();
    if (!question) return;
    if (!sessionId) return;

    setError(null);
    setLoading(true);
    setInput("");

    const userMsg: ChatMsg = { role: "user", content: question, at: new Date().toISOString() };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const response = await queryRag({
        query: question,
        session_id: sessionId,
        verbose,
        use_web: useWeb,
      });

      const assistant: ChatMsg = {
        role: "assistant",
        content: response.answer,
        at: new Date().toISOString(),
        response,
      };
      setMessages((prev) => [...prev, assistant]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  async function openSource(path: string) {
    try {
      const data = await getSource(path);
      setSourceData(data);
      setModalOpen(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load source");
    }
  }

  function viewableSources(m: Extract<ChatMsg, { role: "assistant" }>): string[] {
    return m.response.local_source_paths || m.response.local_sources || [];
  }

  async function vote(answerId: string, thumb: "up" | "down") {
    try {
      await sendFeedback({
        answer_id: answerId,
        session_id: sessionId,
        thumb,
        reason_tags: thumb === "down" ? ["irrelevant"] : [],
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to submit feedback");
    }
  }

  return (
    <main>
      <div className="card col">
        <h2 style={{ margin: 0 }}>Obsidian RAG</h2>
        <p className="muted" style={{ margin: 0 }}>
          FastAPI backend + Next.js frontend, typed API contracts.
        </p>
        <div className="row">
          <label className="row">
            <input type="checkbox" checked={verbose} onChange={(e) => setVerbose(e.target.checked)} />
            Verbose
          </label>
          <label className="row">
            <input type="checkbox" checked={useWeb} onChange={(e) => setUseWeb(e.target.checked)} />
            Use trusted web docs
          </label>
          <span className="muted">Session: {sessionId ? sessionId.slice(0, 8) : "..."}</span>
        </div>
      </div>

      <div className="chat">
        {messages.map((m, idx) => (
          <div key={`${m.at}-${idx}`} className={`msg ${m.role === "user" ? "user" : "assistant"}`}>
            <div className="muted">{m.role.toUpperCase()}</div>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
            {m.role === "assistant" && (
              <>
                <div className="sources">
                  {viewableSources(m).map((src) => (
                    <button key={src} className="chip" onClick={() => openSource(src)}>
                      {src.split("/").pop() || src}
                    </button>
                  ))}
                </div>
                {(m.response.web_sources || []).length > 0 && (
                  <details style={{ marginTop: 8 }}>
                    <summary>Online resources used</summary>
                    <ul>
                      {(m.response.web_sources || []).map((url) => (
                        <li key={url}>
                          <a href={url} target="_blank" rel="noreferrer">
                            {url}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
                <details style={{ marginTop: 8 }}>
                  <summary>Thinking and logs</summary>
                  <p className="muted">{String(m.response.debug?.thinking_summary || "No summary available")}</p>
                  <pre>{JSON.stringify(m.response.telemetry || {}, null, 2)}</pre>
                </details>
                <div className="row" style={{ marginTop: 8 }}>
                  <button onClick={() => vote(m.response.answer_id, "up")}>Helpful</button>
                  <button onClick={() => vote(m.response.answer_id, "down")}>Not Helpful</button>
                </div>
              </>
            )}
          </div>
        ))}
      </div>

      <div className="card col">
        <textarea
          rows={4}
          placeholder="Ask a question about your notes..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <div className="row">
          <button disabled={!canSend} onClick={onAsk}>
            {loading ? "Thinking..." : "Ask"}
          </button>
        </div>
        {error && <p style={{ color: "#9f1d1d" }}>{error}</p>}
      </div>

      <SourceModal open={modalOpen} data={sourceData} onClose={() => setModalOpen(false)} />
    </main>
  );
}
