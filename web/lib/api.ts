import type { FeedbackRequest, QueryRequest, QueryResponse, SourcePreview } from "./types";

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
