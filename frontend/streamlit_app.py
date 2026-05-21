import streamlit as st
import requests
import uuid
from datetime import datetime
import json

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .timestamp {
        font-size: 0.75rem;
        color: #666;
        margin-bottom: 0.25rem;
    }
    .verbose-section {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-top: 0.5rem;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .graph-edge {
        padding: 0.25rem 0.5rem;
        background-color: #e8f5e9;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-family: monospace;
    }
    .stat-box {
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "verbose_mode" not in st.session_state:
    st.session_state.verbose_mode = False

if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False


def load_history():
    """Load conversation history from backend"""
    try:
        response = requests.get(
            f"{API_URL}/history/{st.session_state.session_id}",
            params={"limit": 50}
        )
        
        if response.status_code == 200:
            history = response.json()
            # Reverse to show oldest first
            history.reverse()
            
            st.session_state.messages = []
            for item in history:
                st.session_state.messages.append({
                    "role": "user",
                    "content": item["query"],
                    "timestamp": item["timestamp"]
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": item["answer"],
                    "timestamp": item["timestamp"],
                    "verbose_output": item.get("verbose_output"),
                    "sources": item.get("sources")
                })
            
            st.session_state.history_loaded = True
            return True
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        return False


def clear_history():
    """Clear conversation history"""
    try:
        response = requests.delete(
            f"{API_URL}/clear/{st.session_state.session_id}"
        )
        
        if response.status_code == 200:
            st.session_state.messages = []
            st.success(" History cleared!")
            return True
    except Exception as e:
        st.error(f"Failed to clear history: {e}")
        return False


def get_stats():
    """Get system statistics"""
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            return response.json()
    except:
        return None


def query_rag(question: str, verbose: bool = False):
    """Send query to RAG backend"""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "query": question,
                "session_id": st.session_state.session_id,
                "verbose": verbose
            },
            timeout=3000
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
        return None


def explore_graph(note_query: str, depth: int = 1):
    """Explore graph connections"""
    try:
        response = requests.post(
            f"{API_URL}/graph",
            json={
                "note_query": note_query,
                "depth": depth
            },
            timeout=300
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Graph exploration failed: {e}")
        return None


# Sidebar
with st.sidebar:
    st.title(" Settings")
    
    # Verbose mode toggle
    st.session_state.verbose_mode = st.toggle(
        " Verbose Mode",
        value=st.session_state.verbose_mode,
        help="Show detailed retrieval process"
    )
    
    st.divider()
    
    # Session info
    st.subheader(" Session Info")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
    st.caption(f"Messages: {len(st.session_state.messages)}")
    
    # Load history button
    if st.button(" Load History", use_container_width=True):
        with st.spinner("Loading history..."):
            if load_history():
                st.success(" History loaded!")
                st.rerun()
    
    # Clear history button
    if st.button(" Clear History", use_container_width=True):
        if clear_history():
            st.rerun()
    
    # New session button
    if st.button(" New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.history_loaded = False
        st.success(" New session started!")
        st.rerun()
    
    st.divider()
    
    # System stats
    st.subheader(" System Stats")
    stats = get_stats()
    if stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sessions", stats["total_sessions"])
            st.metric("Queries", stats["total_queries"])
        with col2:
            st.metric("Nodes", stats["graph_nodes"])
            st.metric("Chunks", stats["chunks"])


# Main content
st.title(" RAG Assistant")
st.caption("Ask questions about your Obsidian vault")

# Tabs
tab1, tab2 = st.tabs([" Chat", " Graph Explorer"])

# CHAT TAB
with tab1:
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # Timestamp
                if "timestamp" in message:
                    ts = datetime.fromisoformat(message["timestamp"])
                    st.caption(ts.strftime("%Y-%m-%d %H:%M:%S"))
                
                # Message content
                st.markdown(message["content"])
                
                # Sources (if any)
                if message.get("role") == "assistant" and message.get("sources"):
                    sources_md = "\n**Sources:**\n"
                    import os
                    import urllib.parse
                    for src in message["sources"]:
                        abs_path = os.path.abspath(src)
                        filename = os.path.basename(src)
                        url_path = urllib.parse.quote(abs_path)
                        sources_md += f"- [{filename}](obsidian://open?path={url_path})\n"
                    st.markdown(sources_md)
                
                # Verbose output (if any)
                if message["role"] == "assistant" and message.get("verbose_output"):
                    with st.expander(" View Retrieval Details"):
                        st.code(message["verbose_output"], language="text")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your notes..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_rag(prompt, st.session_state.verbose_mode)
                
                if response:
                    # Display answer
                    st.markdown(response["answer"])
                    
                    # Display sources if available
                    sources_md = ""
                    if response.get("sources"):
                        sources_md = "\n**Sources:**\n"
                        import os
                        import urllib.parse
                        for src in response["sources"]:
                            abs_path = os.path.abspath(src)
                            filename = os.path.basename(src)
                            url_path = urllib.parse.quote(abs_path)
                            sources_md += f"- [{filename}](obsidian://open?path={url_path})\n"
                        st.markdown(sources_md)
                    
                    # Display verbose output if available
                    verbose_output = None
                    if response.get("verbose_output"):
                        verbose_output = response["verbose_output"].get("raw")
                        with st.expander(" View Retrieval Details"):
                            st.code(verbose_output, language="text")
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "timestamp": datetime.now().isoformat(),
                        "verbose_output": verbose_output,
                        "sources": response.get("sources")
                    })
                else:
                    st.error(" Failed to get response from backend")

# GRAPH EXPLORER TAB
with tab2:
    st.subheader(" Explore Note Connections")
    st.caption("Discover how your notes are connected through the knowledge graph")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        note_query = st.text_input(
            "Search for a note:",
            placeholder="e.g., machine learning, python, etc."
        )
    
    with col2:
        depth = st.number_input(
            "Depth:",
            min_value=1,
            max_value=5,
            value=1,
            help="How many levels of connections to explore"
        )
    
    if st.button(" Explore Connections", type="primary", use_container_width=True):
        if note_query:
            with st.spinner("Exploring graph..."):
                graph_result = explore_graph(note_query, depth)
                
                if graph_result:
                    st.success(f"Found: **{graph_result['matched_note']}**")
                    
                    neighbors = graph_result.get("neighbors", {})
                    
                    if neighbors:
                        st.divider()
                        st.subheader("Connections")
                        
                        for depth_level, edges in neighbors.items():
                            with st.expander(f"Level {depth_level} ({len(edges)} connections)", expanded=True):
                                for edge in edges:
                                    st.markdown(
                                        f"<div class='graph-edge'>"
                                        f" <b>{edge['from']}</b> → {edge['to']}"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                    else:
                        st.info("No connections found at this depth")
                else:
                    st.warning("No matching notes found")
        else:
            st.warning("Please enter a note name or query")
    
    # Example queries
    st.divider()
    st.caption("**Example queries:** machine learning, algorithms, python programming")


# Footer
st.divider()
st.caption(
    "**Tip:** Toggle verbose mode in the sidebar to see detailed retrieval process | "
    "Use the Graph Explorer to discover note connections"
)