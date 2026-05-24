export type QueryRequest = {
  query: string;
  session_id: string;
  verbose?: boolean;
  use_web?: boolean;
};

export type QueryResponse = {
  answer_id: string;
  answer: string;
  sources?: string[];
  local_sources?: string[];
  web_sources?: string[];
  web_candidate_sources?: string[];
  local_source_paths?: string[];
  chunk_ids?: string[];
  telemetry?: Record<string, unknown>;
  debug?: Record<string, unknown>;
  verbose_output?: Record<string, unknown> | null;
};

export type FeedbackRequest = {
  answer_id: string;
  session_id: string;
  thumb: "up" | "down";
  reason_tags?: string[];
  note?: string;
};

export type SourcePreview = {
  path: string;
  name: string;
  content: string;
};

export type SessionInfo = {
  session_id: string;
  last_timestamp: string;
  message_count: number;
};

export type HistoryItem = {
  id: number;
  timestamp: string;
  query: string;
  answer: string;
  verbose_output?: string | null;
  sources?: string[] | null;
};

export type VaultTree = {
  root: string;
  folders: Record<string, VaultTreeNode>;
  files: string[];
};

export type VaultTreeNode = {
  folders: Record<string, VaultTreeNode>;
  files: string[];
};

export type GpuStats = {
  available: boolean;
  gpus: Array<{
    name: string;
    utilization_gpu: number;
    memory_used_mb: number;
    memory_total_mb: number;
    temperature_c: number;
  }>;
};
