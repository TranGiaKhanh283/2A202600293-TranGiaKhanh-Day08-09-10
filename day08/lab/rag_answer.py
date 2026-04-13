"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from index import CHROMA_DB_DIR, get_embedding

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10
TOP_K_SELECT = 3

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# 0 = chỉ abstain khi không có chunk hoặc mã lỗi ERR-* không có trong context; >0 chỉ áp dụng với retrieval dense
MIN_DENSE_SIM_FOR_ANSWER = float(os.getenv("MIN_DENSE_SIM_FOR_ANSWER", "0"))
RRF_K = 60
# Khi điểm retrieval thấp (embedding yếu / RRF nhỏ), đưa thêm chunk vào prompt
LOW_RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("LOW_RETRIEVAL_SCORE_THRESHOLD", "0.18"))
EXPAND_WEAK_RETRIEVAL = os.getenv("EXPAND_WEAK_RETRIEVAL", "1") == "1"
MAX_SELECT_WHEN_WEAK = int(os.getenv("MAX_SELECT_WHEN_WEAK", "6"))

_CROSS_ENCODER = None
_BM25_INDEX = None
_BM25_CHUNKS: Optional[List[Dict[str, Any]]] = None


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CROSS_ENCODER


def _ensure_bm25() -> None:
    global _BM25_INDEX, _BM25_CHUNKS
    if _BM25_INDEX is not None and _BM25_CHUNKS is not None:
        return
    import chromadb
    from rank_bm25 import BM25Okapi

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")
    data = collection.get(include=["documents", "metadatas"])
    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"] or [{}] * len(ids)

    tokenized_corpus = [d.lower().split() for d in docs]
    _BM25_INDEX = BM25Okapi(tokenized_corpus)
    _BM25_CHUNKS = []
    for i, cid in enumerate(ids):
        _BM25_CHUNKS.append({
            "id": cid,
            "text": docs[i],
            "metadata": dict(metas[i]) if metas[i] else {},
        })


# =============================================================================
# RETRIEVAL — DENSE
# =============================================================================


def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")
    n_docs = collection.count()
    if n_docs == 0:
        return []
    q_emb = get_embedding(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, n_docs),
        include=["documents", "metadatas", "distances"],
    )
    if not results["ids"] or not results["ids"][0]:
        return []

    out: List[Dict[str, Any]] = []
    for i, cid in enumerate(results["ids"][0]):
        dist = float(results["distances"][0][i])
        score = max(0.0, min(1.0, 1.0 - dist))
        out.append({
            "id": cid,
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i] or {},
            "score": score,
        })
    return out


# =============================================================================
# RETRIEVAL — SPARSE / HYBRID
# =============================================================================


def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    _ensure_bm25()
    assert _BM25_INDEX is not None and _BM25_CHUNKS is not None

    tokenized_query = query.lower().split()
    scores = _BM25_INDEX.get_scores(tokenized_query)
    n = len(scores)
    top_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)[:top_k]

    out: List[Dict[str, Any]] = []
    for rank, idx in enumerate(top_indices):
        ch = dict(_BM25_CHUNKS[idx])
        ch["score"] = float(scores[idx])
        ch["rank"] = rank + 1
        out.append(ch)
    return out


def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    dense = retrieve_dense(query, top_k=max(top_k * 2, 20))
    sparse = retrieve_sparse(query, top_k=max(top_k * 2, 20))

    rrf: Dict[str, float] = {}
    by_id: Dict[str, Dict[str, Any]] = {}

    for rank, ch in enumerate(dense):
        cid = ch["id"]
        by_id[cid] = ch
        rrf[cid] = rrf.get(cid, 0.0) + dense_weight * (1.0 / (RRF_K + rank + 1))

    for rank, ch in enumerate(sparse):
        cid = ch["id"]
        if cid not in by_id:
            by_id[cid] = ch
        rrf[cid] = rrf.get(cid, 0.0) + sparse_weight * (1.0 / (RRF_K + rank + 1))

    sorted_ids = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)[:top_k]
    out: List[Dict[str, Any]] = []
    for cid in sorted_ids:
        ch = dict(by_id[cid])
        ch["score"] = rrf[cid]
        ch["rrf_score"] = rrf[cid]
        out.append(ch)
    return out


# =============================================================================
# RERANK
# =============================================================================


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    try:
        model = _get_cross_encoder()
        pairs = [[query, c["text"]] for c in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        out: List[Dict[str, Any]] = []
        for ch, s in ranked[:top_k]:
            ch = dict(ch)
            ch["rerank_score"] = float(s)
            ch["score"] = float(s)
            out.append(ch)
        return out
    except Exception as e:
        print(f"[warn] rerank CrossEncoder không khả dụng ({e}); giữ thứ tự retrieval.")
        return candidates[:top_k]


# =============================================================================
# QUERY TRANSFORM
# =============================================================================


def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    return [query]


# =============================================================================
# GENERATION
# =============================================================================


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        score = chunk.get("score", 0)
        text = chunk.get("text", "")
        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if score:
            header += f" | score={float(score):.4f}"
        context_parts.append(f"{header}\n{text}")
    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    prompt = f"""Bạn là trợ lý nội bộ CS/IT. Chỉ được dùng các đoạn context bên dưới.

Quy tắc:
1) Nếu ít nhất một đoạn có thông tin liên quan câu hỏi (kể cả gián tiếp: tên tài liệu đổi, alias như "Approval Matrix", mục SLA/refund/access), bạn PHẢI trả lời ngắn gọn, đúng fact, và trích dẫn [1], [2], … theo đoạn dùng.
2) Chỉ từ chối khi toàn bộ đoạn context đều không liên quan câu hỏi. Khi đó hãy nói một câu tiếng Việt: không đủ thông tin trong tài liệu nội bộ đã cung cấp (không bịa chi tiết).
3) Không được trả lời "không đủ dữ liệu" nếu context đã nhắc đến chủ đề hoặc tài liệu trả lời được câu hỏi (kể cả tên khác với cách hỏi của người dùng).
4) Không bịa số SLA, chính sách, hoặc giải thích mã lỗi không có trong context.

Câu hỏi: {query}

Context:
{context_block}

Trả lời:"""
    return prompt


def call_llm(prompt: str) -> str:
    if os.getenv("SKIP_LLM") == "1":
        return (
            "Theo tài liệu [1], vui lòng tham chiếu các đoạn context đã cung cấp (SKIP_LLM=1)."
        )

    if LLM_PROVIDER == "gemini":
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY required for LLM_PROVIDER=gemini")
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text or ""

    from openai_client import format_model_for_inference, get_openai_client

    client = get_openai_client()
    model = format_model_for_inference(LLM_MODEL)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


def _best_dense_similarity(query: str) -> float:
    top = retrieve_dense(query, top_k=1)
    return float(top[0]["score"]) if top else 0.0


def _query_has_error_code_not_in_context(query: str, candidates: List[Dict[str, Any]]) -> bool:
    codes = re.findall(r"ERR-[A-Z0-9-]+", query.upper())
    if not codes:
        return False
    blob = "\n".join(c.get("text", "") for c in candidates).upper()
    return not any(c in blob for c in codes)


def should_abstain(
    query: str,
    candidates: List[Dict[str, Any]],
    best_dense_similarity: float,
    retrieval_mode: str,
) -> bool:
    if not candidates:
        return True
    if _query_has_error_code_not_in_context(query, candidates):
        return True
    # Chỉ lọc theo dense top-1 khi đang dùng dense; hybrid/sparse có thang điểm khác — tránh abstain nhầm
    if (
        MIN_DENSE_SIM_FOR_ANSWER > 0
        and retrieval_mode == "dense"
        and best_dense_similarity < MIN_DENSE_SIM_FOR_ANSWER
    ):
        return True
    return False


def _effective_top_k_select(
    candidates: List[Dict[str, Any]],
    requested: int,
) -> int:
    """Khi tín hiệu retrieval yếu, tăng số chunk đưa vào LLM (embedding fallback / RRF nhỏ)."""
    if not candidates or not EXPAND_WEAK_RETRIEVAL:
        return requested
    head = candidates[: max(8, requested)]
    scores = [float(c.get("score", 0) or 0) for c in head]
    mx = max(scores) if scores else 0.0
    if mx >= LOW_RETRIEVAL_SCORE_THRESHOLD:
        return requested
    bumped = min(MAX_SELECT_WHEN_WEAK, len(candidates), max(requested, 5))
    return bumped


def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
    }

    best_dense_sim = _best_dense_similarity(query)

    if retrieval_mode == "dense":
        candidates = retrieve_dense(query, top_k=top_k_search)
    elif retrieval_mode == "sparse":
        candidates = retrieve_sparse(query, top_k=top_k_search)
    elif retrieval_mode == "hybrid":
        candidates = retrieve_hybrid(query, top_k=top_k_search)
    else:
        raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:3]):
            print(f"  [{i+1}] score={c.get('score', 0):.4f} | {c['metadata'].get('source', '?')}")

    select_k = _effective_top_k_select(candidates, top_k_select)

    if should_abstain(query, candidates, best_dense_sim, retrieval_mode):
        return {
            "query": query,
            "answer": "Không đủ dữ liệu trong tài liệu nội bộ để trả lời câu hỏi này.",
            "sources": [],
            "chunks_used": [],
            "config": config,
            "abstained": True,
        }

    if use_rerank:
        candidates = rerank(query, candidates, top_k=select_k)
    else:
        candidates = candidates[:select_k]

    if verbose:
        print(f"[RAG] After select: {len(candidates)} chunks")

    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"\n[RAG] Prompt:\n{prompt[:500]}...\n")

    answer = call_llm(prompt)

    sources = list({
        c["metadata"].get("source", "unknown")
        for c in candidates
    })

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": candidates,
        "config": config,
        "abstained": False,
    }


def compare_retrieval_strategies(query: str) -> None:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print("=" * 60)

    strategies = ["dense", "hybrid"]

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        try:
            result = rag_answer(query, retrieval_mode=strategy, verbose=False)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except Exception as e:
            print(f"Lỗi: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 + 3: RAG Answer Pipeline")
    print("=" * 60)

    test_queries = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
        "Ai phải phê duyệt để cấp quyền Level 3?",
        "ERR-403-AUTH là lỗi gì?",
    ]

    print("\n--- Sprint 2: Test Baseline (Dense) ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except Exception as e:
            print(f"Lỗi: {e}")

    print("\n--- Sprint 3: So sánh strategies (cần index + API LLM) ---")
    try:
        compare_retrieval_strategies("Approval Matrix để cấp quyền là tài liệu nào?")
    except Exception as e:
        print(e)
