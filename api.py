import logging
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from main import ImprovedRAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API")

# CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DB_PATH = "conversations.db"


def init_db():
    """Initialize SQLite database for conversation history"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            verbose_output TEXT,
            sources TEXT
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_session 
        ON conversations(session_id, timestamp)
    """)
    
    conn.commit()
    conn.close()


# Initialize database on startup
init_db()

# Initialize RAG system
logger.info("Initializing RAG system...")
rag = ImprovedRAG()
logger.info("RAG system ready!")


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    session_id: str
    verbose: bool = False


class QueryResponse(BaseModel):
    answer: str
    verbose_output: Optional[Dict] = None
    sources: Optional[List[str]] = None


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
            "/history/{session_id}": "GET - Get conversation history",
            "/graph": "POST - Explore note connections",
            "/clear/{session_id}": "DELETE - Clear session history"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer a query and store in conversation history"""
    try:
        verbose_data = {}
        
        if request.verbose:
            # Capture verbose output
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                result = rag.answer(request.query, verbose=True)
            
            verbose_output = f.getvalue()
            
            # Parse verbose output into structured format
            verbose_data = {
                "raw": verbose_output
            }
        else:
            result = rag.answer(request.query, verbose=False)
            verbose_output = None
            verbose_data = None
            
        answer = result["answer"]
        sources = result["sources"]
        
        # Store in database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        import json
        sources_json = json.dumps(sources)
        
        cursor.execute("""
            INSERT INTO conversations 
            (session_id, timestamp, query, answer, verbose_output, sources)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            request.session_id,
            datetime.now().isoformat(),
            request.query,
            answer,
            verbose_output,
            sources_json
        ))
        
        conn.commit()
        conn.close()
        
        return QueryResponse(
            answer=answer,
            verbose_output=verbose_data if request.verbose else None,
            sources=sources
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}", response_model=List[ConversationHistory])
async def get_history(session_id: str, limit: int = 50):
    """Get conversation history for a session"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, timestamp, query, answer, verbose_output, sources
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        import json
        history = []
        for row in rows:
            sources_list = None
            if len(row) > 5 and row[5]:
                try:
                    sources_list = json.loads(row[5])
                except:
                    pass
            history.append(
                ConversationHistory(
                    id=row[0],
                    timestamp=row[1],
                    query=row[2],
                    answer=row[3],
                    verbose_output=row[4],
                    sources=sources_list
                )
            )
        
        return history
    
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for a session"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM conversations
            WHERE session_id = ?
        """, (session_id,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return {
            "message": f"Cleared {deleted_count} messages from session {session_id}"
        }
    
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph", response_model=GraphResponse)
async def explore_graph(request: GraphRequest):
    """Explore note connections using graph"""
    try:
        # Get node mapping
        node_map_lower = {
            node.lower(): node
            for node in rag.graph.nodes()
        }
        
        # Search for matching notes
        results = rag.node_db.similarity_search(request.note_query, k=3)
        
        if not results:
            raise HTTPException(status_code=404, detail="No matching notes found")
        
        matched_node = results[0].page_content
        actual_node = node_map_lower.get(matched_node.lower())
        
        if not actual_node:
            raise HTTPException(
                status_code=404,
                detail=f"'{matched_node}' not found in graph"
            )
        
        # Build neighbor structure
        visited = set()
        current = {actual_node}
        neighbors_by_depth = {}
        
        for d in range(request.depth):
            next_nodes = set()
            depth_neighbors = []
            
            for node in current:
                node_neighbors = list(rag.graph.neighbors(node))
                for neighbor in node_neighbors:
                    if neighbor not in visited:
                        depth_neighbors.append({
                            "from": node,
                            "to": neighbor
                        })
                        next_nodes.add(neighbor)
                        visited.add(neighbor)
            
            if depth_neighbors:
                neighbors_by_depth[d + 1] = depth_neighbors
            
            current = next_nodes
        
        return GraphResponse(
            matched_note=actual_node,
            neighbors=neighbors_by_depth
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exploring graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT session_id) as sessions,
                COUNT(*) as total_queries
            FROM conversations
        """)
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            "total_sessions": stats[0],
            "total_queries": stats[1],
            "graph_nodes": len(rag.graph.nodes),
            "graph_edges": len(rag.graph.edges),
            "chunks": len(rag.bm25_chunks)
        }
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )