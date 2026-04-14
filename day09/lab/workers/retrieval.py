"""
workers/retrieval.py — Retrieval Worker
Sprint 2: Dense retrieval từ ChromaDB (cùng index Day 08: collection rag_lab).

Gọi độc lập:
    python workers/retrieval.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from dotenv import load_dotenv

load_dotenv(LAB_DIR / ".env")

from index import CHROMA_DB_DIR, get_embedding

WORKER_NAME = "retrieval_worker"
DEFAULT_TOP_K = 5


def _normalize_source(meta: dict) -> str:
    raw = meta.get("source", "unknown") if meta else "unknown"
    if not isinstance(raw, str):
        return str(raw)
    name = Path(raw.replace("\\", "/")).name
    return name or "unknown"


def retrieve_dense(query: str, top_k: int = DEFAULT_TOP_K) -> list:
    """Dense retrieval: embed query → ChromaDB rag_lab → top_k chunks."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = client.get_collection("rag_lab")
    except Exception:
        return []

    n_docs = collection.count()
    if n_docs == 0:
        return []

    q_emb = get_embedding(query)
    k = min(top_k, n_docs)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "distances", "metadatas"],
    )
    if not results["ids"] or not results["ids"][0]:
        return []

    out = []
    for i, cid in enumerate(results["ids"][0]):
        dist = float(results["distances"][0][i])
        score = max(0.0, min(1.0, 1.0 - dist))
        meta = results["metadatas"][0][i] or {}
        out.append({
            "text": results["documents"][0][i],
            "source": _normalize_source(meta),
            "score": round(score, 4),
            "metadata": dict(meta),
        })
    return out


def run(state: dict) -> dict:
    task = state.get("task", "")
    top_k = int(state.get("retrieval_top_k", DEFAULT_TOP_K))

    state.setdefault("workers_called", [])
    state.setdefault("history", [])
    state.setdefault("worker_io_logs", [])

    state["workers_called"].append(WORKER_NAME)

    worker_io = {
        "worker": WORKER_NAME,
        "input": {"task": task, "top_k": top_k},
        "output": None,
        "error": None,
    }

    try:
        chunks = retrieve_dense(task, top_k=top_k)
        sources = list({c["source"] for c in chunks})
        state["retrieved_chunks"] = chunks
        state["retrieved_sources"] = sources
        worker_io["output"] = {"chunks_count": len(chunks), "sources": sources}
        state["history"].append(
            f"[{WORKER_NAME}] retrieved {len(chunks)} chunks from {sources}"
        )
    except Exception as e:
        worker_io["error"] = {"code": "RETRIEVAL_FAILED", "reason": str(e)}
        state["retrieved_chunks"] = []
        state["retrieved_sources"] = []
        state["history"].append(f"[{WORKER_NAME}] ERROR: {e}")

    state["worker_io_logs"].append(worker_io)
    return state


if __name__ == "__main__":
    print("=" * 50)
    print("Retrieval Worker — Standalone Test")
    print("=" * 50)

    for query in [
        "SLA ticket P1 là bao lâu?",
        "Điều kiện được hoàn tiền là gì?",
        "Ai phê duyệt cấp quyền Level 3?",
    ]:
        print(f"\n▶ Query: {query}")
        result = run({"task": query})
        print(f"  Retrieved: {len(result.get('retrieved_chunks', []))} chunks")
        for c in result.get("retrieved_chunks", [])[:2]:
            print(f"    [{c['score']:.3f}] {c['source']}: {c['text'][:80]}...")
        print(f"  Sources: {result.get('retrieved_sources', [])}")

    print("\n✅ retrieval_worker test done.")
