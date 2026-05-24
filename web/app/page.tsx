"use client";

import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { toast } from "sonner";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import SourceModal from "../components/source-modal";
import {
  getHistory,
  getReady,
  getSource,
  getSystemMetric,
  getVaultTree,
  listSessions,
  queryRag,
  sendFeedback,
  streamQueryRag,
} from "../lib/api";
import type {
  QueryResponse,
  SessionInfo,
  SourcePreview,
  SystemMetric,
  VaultFile,
  VaultTree,
  VaultTreeNode,
} from "../lib/types";

type ChatMsg =
  | { role: "user"; content: string; at: string }
  | { role: "assistant"; content: string; at: string; response: QueryResponse };

type LiveTrace = {
  active: boolean;
  stageLogs: string[];
  raw: string;
  thinking: string;
  answer: string;
};

function newSessionId(): string {
  return crypto.randomUUID();
}

function extractThinking(raw: string): { thinking: string; answer: string } {
  const thinkingMatch = raw.match(/<thinking>([\s\S]*?)(?:<\/thinking>|$)/i);
  const answerMatch = raw.match(/<answer>([\s\S]*?)(?:<\/answer>|$)/i);
  return {
    thinking: thinkingMatch?.[1]?.trim() || "",
    answer: answerMatch?.[1]?.trim() || "",
  };
}

function ExplorerNode({
  node,
  depth,
  expanded,
  toggleExpanded,
  onOpen,
  selectedPath,
}: {
  node: VaultTreeNode;
  depth: number;
  expanded: Set<string>;
  toggleExpanded: (path: string) => void;
  onOpen: (f: VaultFile) => void;
  selectedPath: string;
}) {
  const isOpen = expanded.has(node.path);
  return (
    <div className="explorer-node">
      <button
        className="explorer-row folder-row"
        style={{ paddingLeft: `${8 + depth * 12}px` }}
        onClick={() => toggleExpanded(node.path)}
      >
        <span className="chev">{isOpen ? "▾" : "▸"}</span>
        <span className="icon">📁</span>
        <span>{node.name}</span>
      </button>
      {isOpen && (
        <>
          {node.folders.map((folder) => (
            <ExplorerNode
              key={folder.path}
              node={folder}
              depth={depth + 1}
              expanded={expanded}
              toggleExpanded={toggleExpanded}
              onOpen={onOpen}
              selectedPath={selectedPath}
            />
          ))}
          {node.files.map((f) => (
            <button
              key={f.path}
              className={`explorer-row file-row ${selectedPath === f.path ? "selected" : ""}`}
              style={{ paddingLeft: `${28 + depth * 12}px` }}
              onClick={() => onOpen(f)}
            >
              <span className="icon">📄</span>
              <span>{f.name}</span>
            </button>
          ))}
        </>
      )}
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="loading-shell">
      <div className="loading-card shimmer" />
      <div className="loading-grid">
        <div className="loading-card shimmer tall" />
        <div className="loading-card shimmer tall" />
      </div>
      <p className="muted center">Booting backend models and indexes. UI will unlock when ready.</p>
    </div>
  );
}

export default function Page() {
  const [backendReady, setBackendReady] = useState(false);
  const [sessionId, setSessionId] = useState<string>("");
  const [input, setInput] = useState("");
  const [useWeb, setUseWeb] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [showThinking, setShowThinking] = useState(true);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [sourceData, setSourceData] = useState<SourcePreview | null>(null);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [vaultTree, setVaultTree] = useState<VaultTree | null>(null);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [selectedFilePath, setSelectedFilePath] = useState<string>("");
  const [metrics, setMetrics] = useState<SystemMetric[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [liveTrace, setLiveTrace] = useState<LiveTrace>({
    active: false,
    stageLogs: [],
    raw: "",
    thinking: "",
    answer: "",
  });

  const canSend = useMemo(
    () => input.trim().length > 0 && !loading && !!sessionId && backendReady,
    [input, loading, sessionId, backendReady],
  );

  async function refreshWorkspace() {
    try {
      const [sess, tree] = await Promise.all([listSessions(), getVaultTree()]);
      setSessions(sess);
      setVaultTree(tree);
    } catch (e) {
      toast.error("Failed to refresh workspace");
    }
  }

  useEffect(() => {
    setSessionId(newSessionId());
    let mounted = true;

    async function checkReady() {
      try {
        const r = await getReady();
        if (mounted && r.ready) {
          setBackendReady(true);
          await refreshWorkspace();
          toast.success("Backend is ready");
          return true;
        }
      } catch {
        // backend not ready yet
      }
      return false;
    }

    checkReady();
    const readyTimer = setInterval(async () => {
      const ok = await checkReady();
      if (ok) clearInterval(readyTimer);
    }, 2000);

    return () => {
      mounted = false;
      clearInterval(readyTimer);
    };
  }, []);

  useEffect(() => {
    if (!backendReady) return;

    let mounted = true;
    const tick = async () => {
      try {
        const m = await getSystemMetric();
        if (!mounted) return;
        setMetrics((prev) => {
          const next = [...prev, m];
          return next.slice(-60);
        });
      } catch {
        // no-op
      }
    };

    tick();
    const metricTimer = setInterval(tick, 1000);
    return () => {
      mounted = false;
      clearInterval(metricTimer);
    };
  }, [backendReady]);

  useEffect(() => {
    if (vaultTree?.root?.path) {
      setExpandedFolders(new Set([vaultTree.root.path]));
    }
  }, [vaultTree?.root?.path]);

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
      toast.success("Session loaded");
    } catch (e) {
      toast.error("Failed to load session");
    }
  }

  async function openSource(path: string) {
    try {
      setSelectedFilePath(path);
      const data = await getSource(path);
      setSourceData(data);
      setModalOpen(true);
    } catch {
      toast.error("Failed to open file");
    }
  }

  function toggleExpanded(path: string) {
    setExpandedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
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
      if (thumb === "up") toast.success("Feedback saved");
      if (thumb === "down") toast.error("Marked as not helpful");
    } catch {
      toast.error("Feedback failed");
    }
  }

  async function onAsk() {
    const question = input.trim();
    if (!question || !sessionId) return;

    setError(null);
    setLoading(true);
    setInput("");
    setLiveTrace({ active: showThinking, stageLogs: [], raw: "", thinking: "", answer: "" });

    const userMsg: ChatMsg = { role: "user", content: question, at: new Date().toISOString() };
    setMessages((prev) => [...prev, userMsg]);

    try {
      if (showThinking) {
        let finalResult: QueryResponse | undefined;
        await streamQueryRag(
          {
            query: question,
            session_id: sessionId,
            use_web: useWeb,
            include_thinking: true,
          },
          (event) => {
            const type = String(event.event || "");
            if (type === "stage") {
              const name = String(event.name || "stage");
              const count = Number(event.count || 0);
              setLiveTrace((prev) => ({ ...prev, stageLogs: [...prev.stageLogs, `${name}: ${count}`] }));
              return;
            }
            if (type === "token") {
              const delta = String(event.delta || "");
              setLiveTrace((prev) => {
                const raw = prev.raw + delta;
                const parsed = extractThinking(raw);
                return {
                  ...prev,
                  raw,
                  thinking: parsed.thinking,
                  answer: parsed.answer,
                };
              });
              return;
            }
            if (type === "done" && event.result) {
              finalResult = event.result as QueryResponse;
            }
          },
        );

        if (finalResult) {
          const resolved = finalResult;
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: resolved.answer || "",
              at: new Date().toISOString(),
              response: resolved,
            },
          ]);
        }
      } else {
        const response = await queryRag({ query: question, session_id: sessionId, use_web: useWeb });
        setMessages((prev) => [...prev, { role: "assistant", content: response.answer, at: new Date().toISOString(), response }]);
      }
      await refreshWorkspace();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
      toast.error("Query failed");
    } finally {
      setLoading(false);
      setLiveTrace((prev) => ({ ...prev, active: false }));
    }
  }

  if (!backendReady) {
    return (
      <>
        <header className="top-nav">
          <div className="brand">Obsidian RAG</div>
          <div className="muted">Waiting for backend...</div>
        </header>
        <main>
          <LoadingSkeleton />
        </main>
      </>
    );
  }

  return (
    <>
      <header className="top-nav">
        <div className="brand">Obsidian RAG</div>
        <div className="nav-controls">
          <label className="row"><input type="checkbox" checked={useWeb} onChange={(e) => setUseWeb(e.target.checked)} />Use web refs</label>
          <label className="row"><input type="checkbox" checked={showLogs} onChange={(e) => setShowLogs(e.target.checked)} />Show logs</label>
          <label className="row"><input type="checkbox" checked={showThinking} onChange={(e) => setShowThinking(e.target.checked)} />Live thinking</label>
          <button onClick={() => { setMessages([]); setSessionId(newSessionId()); toast.success("New session started"); }}>New Session</button>
        </div>
      </header>

      <main className="app-shell">
        <aside className="sidebar card">
          <h3>Workspace</h3>
          <div className="sidebar-block">
            <div className="muted">Session</div>
            <div className="session-id">{sessionId.slice(0, 8)}</div>
          </div>

          <div className="sidebar-block">
            <div className="muted">Resources (CPU/RAM/GPU)</div>
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={210}>
                <LineChart data={metrics.map((m, i) => ({ i, cpu: m.cpu_percent, ram: m.ram_percent, gpu: m.gpu_percent }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ddd" />
                  <XAxis dataKey="i" tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="cpu" stroke="#d12f2f" dot={false} name="CPU %" />
                  <Line type="monotone" dataKey="ram" stroke="#2f9d57" dot={false} name="RAM %" />
                  <Line type="monotone" dataKey="gpu" stroke="#2b2b2b" dot={false} name="GPU %" />
                </LineChart>
              </ResponsiveContainer>
            </div>
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
            <div className="muted">Vault File Browser</div>
            <div className="tree-scroll">
              {vaultTree?.root ? (
                <ExplorerNode
                  node={vaultTree.root}
                  depth={0}
                  expanded={expandedFolders}
                  toggleExpanded={toggleExpanded}
                  onOpen={(f) => openSource(f.path)}
                  selectedPath={selectedFilePath}
                />
              ) : (
                <div className="muted">No files</div>
              )}
            </div>
          </div>
        </aside>

        <section className="content">
          {liveTrace.active && (
            <div className="card live-trace">
              <div className="row" style={{ justifyContent: "space-between" }}>
                <strong>Realtime Model Stream</strong>
                <span className="pulse-dot" />
              </div>
              <div className="muted">{liveTrace.stageLogs.join("  ->  ") || "Starting retrieval..."}</div>
              <div className="trace-grid">
                <div>
                  <h4>Thinking</h4>
                  <pre>{liveTrace.thinking || "Thinking stream will appear here..."}</pre>
                </div>
                <div>
                  <h4>Draft Answer</h4>
                  <pre>{liveTrace.answer || "Answer stream will appear here..."}</pre>
                </div>
              </div>
            </div>
          )}

          <div className="chat">
            {messages.map((m, idx) => (
              <div key={`${m.at}-${idx}`} className={`msg ${m.role === "user" ? "user" : "assistant"}`}>
                <div className="muted">{m.role.toUpperCase()}</div>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                {m.role === "assistant" && (
                  <>
                    <div className="sources">
                      {(m.response.local_source_paths || []).map((src) => (
                        <button key={src} className="chip" onClick={() => openSource(src)}>{src.split("/").pop() || src}</button>
                      ))}
                    </div>
                    {(m.response.web_sources || m.response.web_candidate_sources || []).length > 0 && (
                      <details style={{ marginTop: 8 }}>
                        <summary>Online references</summary>
                        <ul>
                          {[...(m.response.web_sources || []), ...(m.response.web_candidate_sources || [])]
                            .filter((v, i, arr) => arr.indexOf(v) === i)
                            .map((url) => (
                              <li key={url}><a href={url} target="_blank" rel="noreferrer">{url}</a></li>
                            ))}
                        </ul>
                      </details>
                    )}

                    {showLogs && (
                      <details style={{ marginTop: 8 }}>
                        <summary>Logs</summary>
                        <pre>{JSON.stringify(m.response.telemetry || {}, null, 2)}</pre>
                      </details>
                    )}

                    <div className="row" style={{ marginTop: 8 }}>
                      <button className="good-btn" onClick={() => vote(m.response.answer_id, "up")}>Helpful</button>
                      <button className="bad-btn" onClick={() => vote(m.response.answer_id, "down")}>Not Helpful</button>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>

          <div className="card col composer">
            <textarea rows={4} placeholder="Ask a question about your vault..." value={input} onChange={(e) => setInput(e.target.value)} />
            <div className="row">
              <button disabled={!canSend} onClick={onAsk}>{loading ? "Generating..." : "Ask"}</button>
            </div>
            {error && <p className="error-text">{error}</p>}
          </div>
        </section>

        <SourceModal open={modalOpen} data={sourceData} onClose={() => setModalOpen(false)} />
      </main>
    </>
  );
}
