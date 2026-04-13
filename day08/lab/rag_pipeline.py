"""
rag_pipeline.py — Indexing + retrieval + grounded answer (end-to-end)
======================================================================
Deliverable Day 08 (gộp Sprint 1–3): một entry point để build index và hỏi đáp.

Ví dụ:
  python rag_pipeline.py index          # build Chroma index
  python rag_pipeline.py ask "SLA ticket P1 là bao lâu?"
  python rag_pipeline.py ask --hybrid "Approval Matrix cấp quyền là gì?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from index import build_index, list_chunks, inspect_metadata_coverage
from rag_answer import rag_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline Day 08")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Preprocess → chunk → embed → ChromaDB")
    p_index.add_argument("--list", action="store_true", help="In vài chunk sau khi build")
    p_index.add_argument("--inspect", action="store_true", help="Phân phối metadata")

    p_ask = sub.add_parser("ask", help="Một câu hỏi qua RAG")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument(
        "--mode",
        choices=("dense", "hybrid", "sparse"),
        default="dense",
        help="Chiến lược retrieval",
    )
    p_ask.add_argument("--rerank", action="store_true", help="Bật cross-encoder rerank")

    args = parser.parse_args()

    if args.cmd == "index":
        build_index()
        if args.list:
            list_chunks()
        if args.inspect:
            inspect_metadata_coverage()
        return

    if args.cmd == "ask":
        out = rag_answer(
            args.question,
            retrieval_mode=args.mode,
            use_rerank=args.rerank,
            verbose=True,
        )
        print("\n--- Answer ---\n")
        print(out["answer"])
        print("\nSources:", out.get("sources", []))
        return


if __name__ == "__main__":
    main()
