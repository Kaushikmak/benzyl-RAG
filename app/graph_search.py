import os
from typing import List
from app.data import ScoredDoc


def expand_with_graph(graph,chunk_index,docs: List[ScoredDoc],max_neighbors: int = 2,max_chunks_per_neighbor: int = 2) -> List[ScoredDoc]:

    expanded = []

    seen = set()

    for doc in docs:

        if doc.content not in seen:
            expanded.append(doc)
            seen.add(doc.content)

        source = doc.doc.metadata.get("source", "")

        filename = os.path.basename(source)

        note_name = filename.replace(".md", "")

        if note_name in graph:

            neighbors = list(
                graph.neighbors(note_name)
            )[:max_neighbors]

            for neighbor in neighbors:

                neighbor_chunks = chunk_index.get(neighbor, [])

                for chunk in neighbor_chunks[:max_chunks_per_neighbor]:

                    if chunk.page_content not in seen:

                        expanded.append(
                            ScoredDoc(
                                content=chunk.page_content,
                                doc=chunk,
                                source=os.path.basename(
                                    chunk.metadata.get(
                                        "source",
                                        "Unknown"
                                    )
                                )
                            )
                        )

                        seen.add(chunk.page_content)

    return expanded