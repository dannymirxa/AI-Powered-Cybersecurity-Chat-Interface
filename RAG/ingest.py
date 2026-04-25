"""
Module 2: Markdown → Vectors → Milvus
Embeds text chunks with Ollama (nomic-embed-text) and stores them in Milvus.

Requirements:
  - Milvus running via Docker Compose (see docker-compose.yml)
  - Ollama running locally with nomic-embed-text pulled

Install: pip install pymilvus requests
Pull model: ollama pull nomic-embed-text

Milvus Docker Compose:
  https://milvus.io/docs/install_standalone-docker-compose.md
"""

import requests
import json
from pymilvus import MilvusClient, DataType
from pdf_to_markdown import chunk_markdown
from typing import Optional

# ── Config ──────────────────────────────────────────────────────────────────
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "cybersec_kb"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768          # nomic-embed-text output dimension
OLLAMA_URL = "http://localhost:11434"
BATCH_SIZE = 32          # chunks per Milvus insert call


# ── Embedding ────────────────────────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    """Call Ollama /api/embeddings and return the embedding vector."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, one call per text (Ollama has no batch endpoint)."""
    return [get_embedding(t) for t in texts]


# ── Milvus helpers ───────────────────────────────────────────────────────────
def get_client() -> MilvusClient:
    return MilvusClient(uri=MILVUS_URI)


def create_collection(client: MilvusClient, recreate: bool = False) -> None:
    """
    Create the cybersec_kb collection with schema:
      id (auto INT64 PK) | vector (FLOAT_VECTOR) | text (VARCHAR) | source (VARCHAR) | chunk_index (INT64)
    """
    if client.has_collection(COLLECTION_NAME):
        if recreate:
            client.drop_collection(COLLECTION_NAME)
            print(f"🗑️  Dropped existing collection '{COLLECTION_NAME}'")
        else:
            print(f"ℹ️  Collection '{COLLECTION_NAME}' already exists — skipping creation.")
            return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id",          DataType.INT64,        is_primary=True)
    schema.add_field("vector",      DataType.FLOAT_VECTOR, dim=EMBED_DIM)
    schema.add_field("text",        DataType.VARCHAR,       max_length=4096)
    schema.add_field("source",      DataType.VARCHAR,       max_length=256)
    schema.add_field("chunk_index", DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"✅ Created collection '{COLLECTION_NAME}'")


def insert_chunks(client: MilvusClient, chunks: list[dict]) -> int:
    """
    Embed and insert chunks into Milvus in batches.

    Args:
        client: Connected MilvusClient.
        chunks: Output of chunk_markdown() — list of {text, chunk_index, source}.

    Returns:
        Total number of records inserted.
    """
    total = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        print(f"  🔢 Embedding chunks {i}–{i+len(batch)-1}...")
        vectors = get_embeddings_batch(texts)

        data = [
            {
                "vector":      vec,
                "text":        chunk["text"],
                "source":      chunk["source"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk, vec in zip(batch, vectors)
        ]
        client.insert(collection_name=COLLECTION_NAME, data=data)
        total += len(batch)
        print(f"  ✅ Inserted {total}/{len(chunks)} chunks")

    return total


# ── Main ingestion pipeline ──────────────────────────────────────────────────
def ingest_markdown(md_path: str, recreate: bool = False) -> None:
    """
    Full pipeline: Markdown file → chunks → embeddings → Milvus.

    Args:
        md_path:  Path to the .md file produced by pdf_to_markdown.
        recreate: If True, drop and recreate the collection first.
    """
    print(f"\n📂 Loading markdown: {md_path}")
    chunks = chunk_markdown(md_path)
    print(f"📄 {len(chunks)} chunks ready for ingestion")

    client = get_client()
    create_collection(client, recreate=recreate)

    print(f"\n⬆️  Inserting into Milvus '{COLLECTION_NAME}'...")
    total = insert_chunks(client, chunks)
    print(f"\n🎉 Done! {total} chunks stored in Milvus.")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Ingest markdown into Milvus")
    parser.add_argument("md_path", help="Path to .md file")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Milvus collection",
    )
    args = parser.parse_args()
    ingest_markdown(args.md_path, recreate=args.recreate)