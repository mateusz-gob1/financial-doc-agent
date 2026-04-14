import json
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

COLLECTION_NAME = "red_flags"
CHROMA_PATH = "data/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)


def build_red_flags_index(red_flags_path: str = "data/red_flags/red_flags.json") -> None:
    """
    Embed all red flags into ChromaDB. Safe to call multiple times — skips
    already-indexed entries by ID.
    """
    with open(red_flags_path, encoding="utf-8") as f:
        red_flags = json.load(f)

    collection = get_collection()
    existing_ids = set(collection.get()["ids"])

    to_add = [rf for rf in red_flags if rf["id"] not in existing_ids]
    if not to_add:
        print(f"Red flags index already up to date ({len(existing_ids)} entries).")
        return

    collection.add(
        ids=[rf["id"] for rf in to_add],
        documents=[rf["description"] for rf in to_add],
        metadatas=[{"title": rf["title"], "category": rf["category"]} for rf in to_add],
    )
    print(f"Indexed {len(to_add)} red flags into ChromaDB.")


def retrieve_red_flags(query: str, k: int = 5) -> list[str]:
    """
    Retrieve the top-k most relevant red flags for a given query.
    Returns formatted strings: "CATEGORY — Title: Description"
    """
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=k)

    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        output.append(f"{meta['category']} — {meta['title']}: {doc}")
    return output
