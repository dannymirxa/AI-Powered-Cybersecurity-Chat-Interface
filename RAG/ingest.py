"""
RAG/ingest.py
─────────────
Pipeline: Markdown → chunks → embeddings → Milvus.

All config is read from environment variables (see .env.example).

Usage:
  python RAG/ingest.py path/to/file.md
  python RAG/ingest.py path/to/file.md --recreate

Inside Docker:
  docker compose exec app python RAG/ingest.py RAG/your_doc.md
"""

import os
import requests
from pymilvus import MilvusClient, DataType
from dotenv import load_dotenv
from RAG.pdf_to_markdown import chunk_markdown

load_dotenv()

MILVUS_URI      = os.getenv("MILVUS_URI",      "http://localhost:19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cybersec_kb")
EMBED_MODEL     = os.getenv("EMBED_MODEL",      "nomic-embed-text")
OLLAMA_URL      = os.getenv("OLLAMA_URL",       "http://localhost:11434")
EMBED_TIMEOUT   = int(os.getenv("EMBED_TIMEOUT", "60"))
BATCH_SIZE      = int(os.getenv("INGEST_BATCH_SIZE", "32"))

EMBED_DIM = 768  # nomic-embed-text output dimension


def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=EMBED_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    return [get_embedding(t) for t in texts]


def get_client() -> MilvusClient:
    return MilvusClient(uri=MILVUS_URI)


def create_collection(client: MilvusClient, recreate: bool = False) -> None:
    if client.has_collection(COLLECTION_NAME):
        if recreate:
            client.drop_collection(COLLECTION_NAME)
            print(f"Dropped collection '{COLLECTION_NAME}'")
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
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
    print(f"Created collection '{COLLECTION_NAME}'")


def insert_chunks(client: MilvusClient, chunks: list[dict]) -> int:
    total = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch   = chunks[i : i + BATCH_SIZE]
        vectors = get_embeddings_batch([c["text"] for c in batch])
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
        print(f"  Inserted {total}/{len(chunks)} chunks")
    return total


def ingest_markdown(md_path: str, recreate: bool = False) -> None:
    print(f"Loading: {md_path}")
    chunks = chunk_markdown(md_path)
    print(f"{len(chunks)} chunks ready")

    client = get_client()
    create_collection(client, recreate=recreate)

    print(f"Inserting into '{COLLECTION_NAME}' ...")
    total = insert_chunks(client, chunks)
    print(f"Done. {total} chunks stored.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest markdown into Milvus")
    parser.add_argument("md_path", help="Path to .md file")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Milvus collection before ingesting",
    )
    args = parser.parse_args()
    ingest_markdown(args.md_path, recreate=args.recreate)
