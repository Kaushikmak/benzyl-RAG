"use client";

import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import SourceModal from "../components/source-modal";
import {
  getGpuStats,
  getHistory,
  getSource,
  getVaultTree,
  listSessions,
  queryRag,
  sendFeedback,
} from "../lib/api";
import type { GpuStats, QueryResponse, SessionInfo, SourcePreview, VaultTreeNode } from "../lib/types";

type ChatMsg =
  | { role: "user"; content: string; at: string }
  | { role: "assistant"; content: string; at: string; response: QueryResponse };

function newSessionId(): string {
  return crypto.randomUUID();
}

function TreeNode({ name, node, level = 0 }: { name: string; node: VaultTreeNode; level?: number }) {
  return (
    <details open={level < 1} className="tree-node" style={{ marginLeft: level * 10 }}>
      <summary>{name}</summary>
      {Object.entries(node.folders).map(([childName, childNode]) => (
        <TreeNode key={`${name}/${childName}`} name={childName} node={childNode} level={level + 1} />
      ))}
      {node.files.map((file) => (
        <div key={`${name}/${file}`} className="tree-file">
          {file}
        </div>
      ))}
    </details>
  );
}

export default function Page() {
  const [sessionId, setSessionId] = useState<string>("");
  const [input, setInput] = useState("");
  const [verbose, setVerbose] = useState(false);
  const [useWeb, setUseWeb] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [sourceData, setSourceData] = useState<SourcePreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [gpu, setGpu] = useState<GpuStats | null>(null);
  const [vaultTree, setVaultTree] = useState<Record<string, VaultTreeNode>>({});

  const canSend = useMemo(() => input.trim().length > 0 && !loading && !!sessionId, [input, loading, sessionId]);

  useEffect(() => {
    const id = newSessionId();
    setSessionId(id);
  }, []);

  useEffect(() => {
    async function loadSidebarData() {
      try {
        const [sess, gpuStats, tree] = await Promise.all([listSessions(), getGpuStats(), getVaultTree()]);
        setSessions(sess);
        setGpu(gpuStats);
        setVaultTree(tree.folders || {});
      } catch {
        // no-op
      }
    }

    loadSidebarData();
    const timer = setInterval(async () => {
      try {
        const gpuStats = await getGpuStats();
        setGpu(gpuStats);
      } catch {
        // no-op
      }
    }, 5000);

    return () => clearInterval(timer);
  }, []);

  async function loadSession(existingSessionId: string) {
    try {
      const history = await getHistory(existingSessionId);
      const rebuilt: ChatMsg[] = [];
      history
        .slice()
        .reverse()
        .forEach((h) => {
          rebuilt.push({ role: "user", content: h.query, at: h.timestamp });
          rebuilt.push({
            role: "assistant",
            content: h.answer,
            at: h.timestamp,
            response: {
              answer_id: `history-${h.id}`,
              answer: h.answer,
              sources: h.sources || [],
            },
          });
        });
      setSessionId(existingSessionId);
      setMessages(rebuilt);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load session");
    }
  }

  async function onAsk() {
    const question = input.trim();
    if (!question || !sessionId) return;

    setError(null);
    setLoading(true);
    setInput("");

    const userMsg: ChatMsg = { role: "user", content: question, at: new Date().toISOString() };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const response = await queryRag({ query: question, session_id: sessionId, verbose, use_web: useWeb });
      const assistant: ChatMsg = { role: "assistant", content: response.answer, at: new Date().toISOString(), response };
      setMessages((prev) => [...prev, assistant]);
      const refreshedSessions = await listSessions();
      setSessions(refreshedSessions);
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
    if (answerId.startsWith("history-")) return;
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
    <main className="app-shell">
      <aside className="sidebar card">
        <h3>Workspace</h3>
        <div className="sidebar-block">
          <div className="muted">Session</div>
          <div className="session-id">{sessionId ? sessionId.slice(0, 8) : "..."}</div>
        </div>
        <div className="sidebar-block">
          <div className="muted">Live GPU</div>
          {gpu?.available ? (
            gpu.gpus.map((g) => (
              <div key={g.name} className="gpu-card">
                <strong>{g.name}</strong>
                <div className="muted">Util: {g.utilization_gpu}%</div>
                <div className="muted">Mem: {g.memory_used_mb}/{g.memory_total_mb} MB</div>
                <div className="muted">Temp: {g.temperature_c}°C</div>
              </div>
            ))
          ) : (
            <div className="muted">GPU stats unavailable</div>
          )}
        </div>

        <div className="sidebar-block">
          <div className="muted">Previous Sessions</div>
          <div className="session-list">
            {sessions.map((s) => (
              <button key={s.session_id} className="session-btn" onClick={() => loadSession(s.session_id)}>
                {s.session_id.slice(0, 8)} ({s.message_count})
              </button>
            ))}
          </div>
        </div>

        <div className="sidebar-block">
          <div className="muted">Vault Structure</div>
          <div className="tree-scroll">
            {Object.entries(vaultTree).map(([name, node]) => (
              <TreeNode key={name} name={name} node={node} />
            ))}
          </div>
        </div>
      </aside>

      <section className="content">
        <div className="card col hero">
          <h2 style={{ margin: 0 }}>Obsidian RAG</h2>
          <p className="muted" style={{ margin: 0 }}>Ask, inspect retrieval behavior, and continue sessions.</p>
          <div className="row wrap-row">
            <label className="row"><input type="checkbox" checked={verbose} onChange={(e) => setVerbose(e.target.checked)} />Verbose</label>
            <label className="row"><input type="checkbox" checked={useWeb} onChange={(e) => setUseWeb(e.target.checked)} />Use trusted web</label>
            <label className="row"><input type="checkbox" checked={showDebug} onChange={(e) => setShowDebug(e.target.checked)} />Show logs/thinking</label>
            <button onClick={() => { setMessages([]); setSessionId(newSessionId()); }}>New Session</button>
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
                      <button key={src} className="chip" onClick={() => openSource(src)}>{src.split("/").pop() || src}</button>
                    ))}
                  </div>

                  {(m.response.web_sources || m.response.web_candidate_sources || []).length > 0 && (
                    <details style={{ marginTop: 8 }}>
                      <summary>Online resources</summary>
                      <ul>
                        {[...(m.response.web_sources || []), ...(m.response.web_candidate_sources || [])]
                          .filter((v, i, arr) => arr.indexOf(v) === i)
                          .map((url) => (
                            <li key={url}><a href={url} target="_blank" rel="noreferrer">{url}</a></li>
                          ))}
                      </ul>
                    </details>
                  )}

                  {showDebug && (
                    <details style={{ marginTop: 8 }} open={idx === messages.length - 1}>
                      <summary>Thinking and logs</summary>
                      <p className="muted">{String(m.response.debug?.thinking_summary || "No summary available")}</p>
                      <pre>{JSON.stringify(m.response.telemetry || {}, null, 2)}</pre>
                    </details>
                  )}

                  <div className="row" style={{ marginTop: 8 }}>
                    <button onClick={() => vote(m.response.answer_id, "up")}>Helpful</button>
                    <button onClick={() => vote(m.response.answer_id, "down")}>Not Helpful</button>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>

        <div className="card col composer">
          <textarea rows={4} placeholder="Ask a question about your notes..." value={input} onChange={(e) => setInput(e.target.value)} />
          <div className="row">
            <button disabled={!canSend} onClick={onAsk}>{loading ? "Thinking..." : "Ask"}</button>
          </div>
          {error && <p style={{ color: "#9f1d1d" }}>{error}</p>}
        </div>
      </section>

      <SourceModal open={modalOpen} data={sourceData} onClose={() => setModalOpen(false)} />
    </main>
  );
}
