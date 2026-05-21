import os
import re
import logging
import networkx as nx

logger = logging.getLogger(__name__)

def build_graph(documents):
    logger.info("Building Obsidian graph")
    graph = nx.Graph()
    note_map = {}
    note_map_lower = {}
    for doc in documents:
        source = doc.metadata["source"]
        filename = os.path.basename(source)
        note_name = filename.replace(".md", "")
        note_map[note_name] = doc
        note_map_lower[note_name.lower()] = note_name
        graph.add_node(note_name)

    wikilink_pattern = r"\[\[(.*?)\]\]"

    for note_name, doc in note_map.items():
        links = re.findall(wikilink_pattern,doc.page_content)
        for link in links:
            clean_link = link.split("|")[0].strip()
            normalized_link = note_map_lower.get(clean_link.lower())
            if normalized_link:
                graph.add_edge(note_name,normalized_link)

    logger.info(
        f"Graph created with "
        f"{len(graph.nodes)} nodes and "
        f"{len(graph.edges)} edges"
    )
    return graph