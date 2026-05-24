"use client";

import type { SourcePreview } from "../lib/types";

type Props = {
  open: boolean;
  data: SourcePreview | null;
  onClose: () => void;
};

export default function SourceModal({ open, data, onClose }: Props) {
  if (!open || !data) return null;

  return (
    <div className="modal" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="row modal-header" style={{ justifyContent: "space-between" }}>
          <strong>{data.name}</strong>
          <button onClick={onClose}>Close</button>
        </div>
        <p className="muted">{data.path}</p>
        <pre>{data.content}</pre>
      </div>
    </div>
  );
}
