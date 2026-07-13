"""
Embeds the lecture chunks (from chunk_lecture.py) into a local, persistent
ChromaDB collection so the teacher agent can eventually retrieve real
lecture content instead of relying on the LLM's generic KNN knowledge.

Uses Chroma's default embedding function (sentence-transformers
all-MiniLM-L6-v2, runs locally, no API key needed).

Run:
    pip install chromadb
    python build_vectordb.py
"""
import json

import chromadb
from chromadb.utils import embedding_functions

CHUNKS_FILE = "chunks.json"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "knn_lecture"


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    # Recreate the collection each run so re-running after re-chunking
    # doesn't leave stale entries behind.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet -- nothing to clear
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["chunk_text"] for c in chunks],
        metadatas=[
            {
                "section_title": c["section_title"],
                "segment_dimension": c["segment_dimension"],
            }
            for c in chunks
        ],
    )

    print(f"Embedded {collection.count()} chunks into collection "
          f"'{COLLECTION_NAME}' at {PERSIST_DIR}")


if __name__ == "__main__":
    main()
