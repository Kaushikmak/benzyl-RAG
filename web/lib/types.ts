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
  chunk_ids?: string[];
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
