"""
Quick CLI sanity check for the "knn_lecture" ChromaDB collection built by
build_vectordb.py. Lets you query the DB and eyeball which chunks come
back before wiring retrieval into the teacher agent.

Run:
    python test_retrieval.py "why does knn need a distance metric"
    python test_retrieval.py "how do you choose k" --n 5
    python test_retrieval.py "feature scaling" --dimension feature_scaling
"""
import argparse

import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "knn_lecture"


def main():
    parser = argparse.ArgumentParser(description="Query the knn_lecture ChromaDB collection.")
    parser.add_argument("query", help="Natural-language query text")
    parser.add_argument("--n", type=int, default=3, help="Number of results to return (default 3)")
    parser.add_argument("--dimension", default=None,
                         help="Restrict results to a single segment_dimension")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    where = {"segment_dimension": args.dimension} if args.dimension else None
    results = collection.query(query_texts=[args.query], n_results=args.n, where=where)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not ids:
        print("No results.")
        return

    print(f'Query: "{args.query}"' + (f" (dimension={args.dimension})" if args.dimension else ""))
    print(f"Top {len(ids)} result(s):\n")

    for rank, (chunk_id, doc, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        print(f"#{rank}  chunk_id={chunk_id}  distance={dist:.4f}")
        print(f"    section: {meta['section_title']}")
        print(f"    dimension: {meta['segment_dimension']}")
        preview = doc.replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"    text: {preview}\n")


if __name__ == "__main__":
    main()
