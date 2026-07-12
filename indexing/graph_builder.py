import os
import re
import logging
import networkx as nx

logger = logging.getLogger(__name__)

def build_graph(documents):
    logger.info("Building universal document relationship graph")
    graph = nx.Graph()
    doc_map = {}
    doc_map_lower = {}

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        filename = os.path.basename(source)
        node_name = os.path.splitext(filename)[0]
        doc_type = doc.metadata.get("doc_type", "document")
        graph.add_node(node_name, doc_type=doc_type)

        doc_map[node_name] = doc
        doc_map_lower[node_name.lower()] = node_name

    # Universal cross-reference pattern: markdown/HTML links [text](path) or explicit filename references
    md_link_pattern = r"\[.*?\]\((.*?)\)"

    for node_name, doc in doc_map.items():
        content = doc.page_content
        content_lower = content.lower()

        # 1. Check markdown links
        links = re.findall(md_link_pattern, content)
        for link in links:
            clean_link = os.path.splitext(os.path.basename(link.split("#")[0].strip()))[0]
            target = doc_map_lower.get(clean_link.lower())
            if target and target != node_name:
                graph.add_edge(node_name, target)

        # 2. Check explicit document name references (for stems >= 3 chars to avoid noise)
        for target_lower, target_name in doc_map_lower.items():
            if target_name == node_name or len(target_lower) < 3:
                continue
            if re.search(r'\b' + re.escape(target_lower) + r'\b', content_lower):
                graph.add_edge(node_name, target_name)

    logger.info(
        f"Graph created with "
        f"{len(graph.nodes)} nodes and "
        f"{len(graph.edges)} edges"
    )
    return graph
