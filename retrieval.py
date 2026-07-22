"""
Retrieval layer for the teacher agent: queries the local ChromaDB
"knn_lecture" collection (built by build_vectordb.py) for the lecture
chunks most relevant to a given segment, restricted to that segment's
knowledge dimension so the teacher only pulls from the right part of
the lecture.
"""
import chromadb
from chromadb.utils import embedding_functions

from knowledge_state import primary_conceptual_dimension

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "knn_lecture"

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        _collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    return _collection


def retrieve_for_segment(segment, n_results=3):
    """Query the knn_lecture collection using the segment's concept text,
    filtered to chunks tagged with this segment's primary conceptual
    dimension (the Group C entry in its `dimensions` list).

    Returns a list of dicts: {"chunk_id", "section_title", "chunk_text"},
    ordered most relevant first. Empty list if the dimension has no chunks.
    """
    collection = _get_collection()
    dim = primary_conceptual_dimension(segment["dimensions"])
    results = collection.query(
        query_texts=[segment["concept"]],
        n_results=n_results,
        where={"segment_dimension": dim},
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return [
        {"chunk_id": cid, "section_title": meta["section_title"], "chunk_text": doc}
        for cid, doc, meta in zip(ids, documents, metadatas)
    ]


def format_context(chunks):
    """Format retrieved chunks into one text block for prompt injection."""
    if not chunks:
        return ""
    parts = [f"[{c['section_title']}]\n{c['chunk_text']}" for c in chunks]
    return "\n\n---\n\n".join(parts)
