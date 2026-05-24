import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import config
from app.feedback import (
    compute_feedback_priors,
    feedback_stats,
    init_feedback_tables,
    save_answer_record,
    save_feedback,
)
from app.rag import ImprovedRAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "conversations.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            verbose_output TEXT,
            sources TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_session
        ON conversations(session_id, timestamp)
        """
    )
    init_feedback_tables(conn)
    conn.commit()
    conn.close()


init_db()
logger.info("Initializing RAG system...")
rag = ImprovedRAG()
logger.info("RAG system ready!")


class QueryRequest(BaseModel):
    query: str
    session_id: str
    verbose: bool = False
    use_web: bool = False


class QueryResponse(BaseModel):
    answer_id: str
    answer: str
    verbose_output: Optional[Dict] = None
    sources: Optional[List[str]] = None
    local_sources: Optional[List[str]] = None
    web_sources: Optional[List[str]] = None
    chunk_ids: Optional[List[str]] = None


class FeedbackRequest(BaseModel):
    answer_id: str
    session_id: str
    thumb: str = Field(pattern="^(up|down)$")
    reason_tags: List[str] = []
    note: str = ""


class ConversationHistory(BaseModel):
    id: int
    timestamp: str
    query: str
    answer: str
    verbose_output: Optional[str] = None
    sources: Optional[List[str]] = None


class GraphRequest(BaseModel):
    note_query: str
    depth: int = 1


class GraphResponse(BaseModel):
    matched_note: str
    neighbors: Dict[int, List[Dict[str, str]]]


@app.get("/")
async def root():
    return {
        "message": "RAG API is running",
        "endpoints": {
            "/query": "POST - Ask questions",
            "/feedback": "POST - Submit thumbs/reason feedback",
            "/feedback/stats": "GET - Feedback quality metrics",
            "/source": "GET - Preview cited source in app",
            "/history/{session_id}": "GET - Get conversation history",
            "/graph": "POST - Explore note connections",
            "/clear/{session_id}": "DELETE - Clear session history",
        },
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        verbose_data = {}

        if request.verbose:
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                result = rag.answer(request.query, verbose=True, use_web=request.use_web)
            verbose_output = f.getvalue()
            verbose_data = {"raw": verbose_output, "telemetry": result.get("telemetry", {})}
        else:
            result = rag.answer(request.query, verbose=False, use_web=request.use_web)
            verbose_output = None
            verbose_data = None

        answer = result["answer"]
        sources = result.get("sources", [])
        answer_id = result["answer_id"]
        local_sources = result.get("local_sources", [])
        web_sources = result.get("web_sources", [])
        chunk_ids = result.get("chunk_ids", [])

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO conversations
            (session_id, timestamp, query, answer, verbose_output, sources)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                request.session_id,
                datetime.now().isoformat(),
                request.query,
                answer,
                verbose_output,
                json.dumps(sources),
            ),
        )
        save_answer_record(
            conn,
            answer_id=answer_id,
            session_id=request.session_id,
            query=request.query,
            answer=answer,
            chunk_ids=chunk_ids,
            local_sources=local_sources,
            web_sources=web_sources,
        )
        conn.commit()
        conn.close()

        return QueryResponse(
            answer_id=answer_id,
            answer=answer,
            verbose_output=verbose_data if request.verbose else None,
            sources=sources,
            local_sources=local_sources,
            web_sources=web_sources,
            chunk_ids=chunk_ids,
        )

    except Exception as e:
        logger.error("Error processing query: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    try:
        conn = sqlite3.connect(DB_PATH)
        save_feedback(
            conn,
            answer_id=request.answer_id,
            session_id=request.session_id,
            thumb=request.thumb,
            reason_tags=request.reason_tags,
            note=request.note,
        )

        priors = compute_feedback_priors(conn)
        rag.set_feedback_priors(priors)
        conn.close()

        return {"message": "Feedback saved", "active_priors": len(priors)}
    except Exception as e:
        logger.error("Error saving feedback: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/stats")
async def get_feedback_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        stats = feedback_stats(conn)
        conn.close()
        return stats
    except Exception as e:
        logger.error("Error getting feedback stats: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/source")
async def get_source(path: str):
    try:
        candidate = Path(path).resolve()
        vault_root = Path(config.VAULT_PATH).resolve()

        if vault_root not in candidate.parents and candidate != vault_root:
            raise HTTPException(status_code=400, detail="Source path outside vault")

        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Source not found")

        content = candidate.read_text(encoding="utf-8", errors="ignore")
        content = content[: config.MAX_SOURCE_PREVIEW_CHARS]
        return {
            "path": str(candidate),
            "name": candidate.name,
            "content": content,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error loading source: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}", response_model=List[ConversationHistory])
async def get_history(session_id: str, limit: int = 50):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, query, answer, verbose_output, sources
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (session_id, limit),
        )
        rows = cursor.fetchall()
        conn.close()

        history = []
        for row in rows:
            sources_list = None
            if len(row) > 5 and row[5]:
                try:
                    sources_list = json.loads(row[5])
                except Exception:
                    pass
            history.append(
                ConversationHistory(
                    id=row[0],
                    timestamp=row[1],
                    query=row[2],
                    answer=row[3],
                    verbose_output=row[4],
                    sources=sources_list,
                )
            )

        return history

    except Exception as e:
        logger.error("Error fetching history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear/{session_id}")
async def clear_history(session_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM conversations
            WHERE session_id = ?
            """,
            (session_id,),
        )

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return {"message": f"Cleared {deleted_count} messages from session {session_id}"}

    except Exception as e:
        logger.error("Error clearing history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph", response_model=GraphResponse)
async def explore_graph(request: GraphRequest):
    try:
        node_map_lower = {node.lower(): node for node in rag.graph.nodes()}
        results = rag.node_db.similarity_search(request.note_query, k=3)

        if not results:
            raise HTTPException(status_code=404, detail="No matching notes found")

        matched_node = results[0].page_content
        actual_node = node_map_lower.get(matched_node.lower())

        if not actual_node:
            raise HTTPException(status_code=404, detail=f"'{matched_node}' not found in graph")

        visited = set()
        current = {actual_node}
        neighbors_by_depth = {}

        for d in range(request.depth):
            next_nodes = set()
            depth_neighbors = []

            for node in current:
                for neighbor in list(rag.graph.neighbors(node)):
                    if neighbor not in visited:
                        depth_neighbors.append({"from": node, "to": neighbor})
                        next_nodes.add(neighbor)
                        visited.add(neighbor)

            if depth_neighbors:
                neighbors_by_depth[d + 1] = depth_neighbors

            current = next_nodes

        return GraphResponse(matched_note=actual_node, neighbors=neighbors_by_depth)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error exploring graph: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(DISTINCT session_id) as sessions, COUNT(*) as total_queries
            FROM conversations
            """
        )

        stats = cursor.fetchone()
        conn.close()

        return {
            "total_sessions": stats[0],
            "total_queries": stats[1],
            "graph_nodes": len(rag.graph.nodes),
            "graph_edges": len(rag.graph.edges),
            "chunks": len(rag.bm25_chunks),
        }

    except Exception as e:
        logger.error("Error getting stats: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
