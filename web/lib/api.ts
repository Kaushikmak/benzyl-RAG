import type {
  FeedbackRequest,
  GpuStats,
  HistoryItem,
  QueryRequest,
  QueryResponse,
  SessionInfo,
  SourcePreview,
  VaultTree,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }

  return (await res.json()) as T;
}

export function queryRag(payload: QueryRequest): Promise<QueryResponse> {
  return http<QueryResponse>("/query", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function sendFeedback(payload: FeedbackRequest): Promise<{ message: string }> {
  return http<{ message: string }>("/feedback", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getSource(path: string): Promise<SourcePreview> {
  return http<SourcePreview>(`/source?path=${encodeURIComponent(path)}`);
}

export function listSessions(): Promise<SessionInfo[]> {
  return http<SessionInfo[]>("/sessions");
}

export function getHistory(sessionId: string): Promise<HistoryItem[]> {
  return http<HistoryItem[]>(`/history/${encodeURIComponent(sessionId)}?limit=100`);
}

export function getVaultTree(): Promise<VaultTree> {
  return http<VaultTree>("/vault/tree");
}

export function getGpuStats(): Promise<GpuStats> {
  return http<GpuStats>("/system/gpu");
}

export async function streamQueryRag(
  payload: QueryRequest & { include_thinking?: boolean },
  onEvent: (event: Record<string, unknown>) => void,
): Promise<void> {
  const res = await fetch(`${API_BASE}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok || !res.body) {
    throw new Error(`Stream request failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";

    for (const part of parts) {
      const line = part
        .split("\n")
        .find((l) => l.startsWith("data: "));
      if (!line) continue;
      const data = line.slice(6);
      try {
        onEvent(JSON.parse(data));
      } catch {
        // ignore malformed events
      }
    }
  }
}
