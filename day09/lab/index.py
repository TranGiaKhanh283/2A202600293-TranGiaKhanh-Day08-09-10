"""
index.py — Sprint 1: Build RAG Index
====================================
Mục tiêu Sprint 1 (60 phút):
  - Đọc và preprocess tài liệu từ data/docs/
  - Chunk tài liệu theo cấu trúc tự nhiên (heading/section)
  - Gắn metadata: source, section, department, effective_date, access
  - Embed và lưu vào vector store (ChromaDB)

Definition of Done Sprint 1:
  ✓ Script chạy được và index đủ docs
  ✓ Có ít nhất 3 metadata fields hữu ích cho retrieval
  ✓ Có thể kiểm tra chunk bằng list_chunks()
"""

import os
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2",
)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

_ST_MODEL = None
_USE_DETERMINISTIC_EMBEDDINGS = False


def _deterministic_embedding(text: str, dim: int = 384) -> List[float]:
    """Fallback khi không load được sentence-transformers (môi trường torch lỗi). Chỉ dùng smoke test."""
    import hashlib

    import numpy as np

    vec = np.zeros(dim, dtype=np.float64)
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    for i in range(dim):
        block = hashlib.sha256(seed + i.to_bytes(4, "little", signed=False)).digest()
        v = int.from_bytes(block[:4], "little") / float(2**32)
        vec[i] = v * 2.0 - 1.0
    n = float(np.linalg.norm(vec)) + 1e-12
    vec /= n
    return vec.astype(np.float32).tolist()


def _get_sentence_transformer():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _ST_MODEL = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    return _ST_MODEL


# =============================================================================
# STEP 1: PREPROCESS
# =============================================================================


def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    """Preprocess một tài liệu: extract metadata từ header và làm sạch nội dung."""
    lines = raw_text.strip().split("\n")
    metadata = {
        "source": filepath,
        "section": "",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    content_lines: List[str] = []
    header_done = False

    for line in lines:
        if not header_done:
            if line.startswith("Source:"):
                metadata["source"] = line.replace("Source:", "").strip()
            elif line.startswith("Department:"):
                metadata["department"] = line.replace("Department:", "").strip()
            elif line.startswith("Effective Date:"):
                metadata["effective_date"] = line.replace("Effective Date:", "").strip()
            elif line.startswith("Access:"):
                metadata["access"] = line.replace("Access:", "").strip()
            elif line.startswith("==="):
                header_done = True
                content_lines.append(line)
            elif line.strip() == "" or line.isupper():
                continue
        else:
            content_lines.append(line)

    cleaned_text = "\n".join(content_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return {
        "text": cleaned_text,
        "metadata": metadata,
    }


# =============================================================================
# STEP 2: CHUNK
# =============================================================================


def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chunk một tài liệu đã preprocess theo section heading."""
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks: List[Dict[str, Any]] = []

    sections = re.split(r"(===.*?===)", text)

    current_section = "General"
    current_section_text = ""

    for part in sections:
        if re.match(r"===.*?===", part):
            if current_section_text.strip():
                section_chunks = _split_by_size(
                    current_section_text.strip(),
                    base_metadata=base_metadata,
                    section=current_section,
                )
                chunks.extend(section_chunks)
            current_section = part.strip("= ").strip()
            current_section_text = ""
        else:
            current_section_text += part

    if current_section_text.strip():
        section_chunks = _split_by_size(
            current_section_text.strip(),
            base_metadata=base_metadata,
            section=current_section,
        )
        chunks.extend(section_chunks)

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict,
    section: str,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """Split text thành chunks; ưu tiên ranh giới đoạn (paragraph)."""
    if len(text) <= chunk_chars:
        return [{
            "text": text,
            "metadata": {**base_metadata, "section": section},
        }]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [{
            "text": text[:chunk_chars],
            "metadata": {**base_metadata, "section": section},
        }]

    chunks: List[Dict[str, Any]] = []
    buf: List[str] = []
    buf_len = 0

    def flush_buf() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        chunk_text = "\n\n".join(buf)
        chunks.append({
            "text": chunk_text,
            "metadata": {**base_metadata, "section": section},
        })
        buf = []
        buf_len = 0

    for para in paragraphs:
        plen = len(para) + (2 if buf else 0)
        if buf_len + plen <= chunk_chars:
            buf.append(para)
            buf_len += plen
            continue

        if buf:
            flush_buf()

        if len(para) <= chunk_chars:
            buf = [para]
            buf_len = len(para)
            continue

        start = 0
        while start < len(para):
            end = min(start + chunk_chars, len(para))
            if end < len(para):
                cut = para.rfind("\n", start + chunk_chars // 2, end)
                if cut == -1:
                    cut = para.rfind(" ", start + chunk_chars // 2, end)
                if cut > start:
                    end = cut
            piece = para[start:end].strip()
            if piece:
                chunks.append({
                    "text": piece,
                    "metadata": {**base_metadata, "section": section},
                })
            if end >= len(para):
                break
            nstart = end - overlap_chars
            if nstart <= start:
                nstart = start + max(1, chunk_chars // 4)
            start = nstart

    flush_buf()

    return chunks


# =============================================================================
# STEP 3: EMBED + STORE
# =============================================================================


def get_embedding(text: str) -> List[float]:
    """Tạo embedding vector; OpenAI hoặc Sentence Transformers theo EMBEDDING_PROVIDER."""
    global _USE_DETERMINISTIC_EMBEDDINGS

    if EMBEDDING_PROVIDER == "openai":
        from openai_client import get_openai_client

        client = get_openai_client()
        response = client.embeddings.create(
            input=text,
            model=OPENAI_EMBEDDING_MODEL,
        )
        return list(response.data[0].embedding)

    if _USE_DETERMINISTIC_EMBEDDINGS:
        return _deterministic_embedding(text)

    try:
        model = _get_sentence_transformer()
        vec = model.encode(text, convert_to_numpy=True)
        return vec.tolist()
    except Exception as e:
        print(
            f"[warn] Không load được sentence-transformers ({e}). "
            "Chuyển sang deterministic embedding (chất lượng kém — chỉ để chạy thử). "
            "Khuyến nghị: OPENAI_API_KEY + EMBEDDING_PROVIDER=openai hoặc sửa torch/torchvision.",
        )
        _USE_DETERMINISTIC_EMBEDDINGS = True
        return _deterministic_embedding(text)


def _normalize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma chỉ chấp nhận str, int, float, bool."""
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    """Pipeline: đọc docs → preprocess → chunk → embed → store."""
    import chromadb

    print(f"Đang build index từ: {docs_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        client.delete_collection("rag_lab")
    except Exception:
        pass
    collection = client.create_collection(
        name="rag_lab",
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    doc_files = sorted(docs_dir.glob("*.txt"))

    if not doc_files:
        print(f"Không tìm thấy file .txt trong {docs_dir}")
        return

    for filepath in doc_files:
        print(f"  Processing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw_text, str(filepath))
        chunks = chunk_document(doc)

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            cid = f"{filepath.stem}_{i}"
            ids.append(cid)
            embeddings.append(get_embedding(chunk["text"]))
            documents.append(chunk["text"])
            metadatas.append(_normalize_metadata(chunk["metadata"]))

        if ids:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        total_chunks += len(chunks)
        print(f"    → {len(chunks)} chunks đã embed và lưu")

    print(f"\nHoàn thành! Tổng số chunks: {total_chunks}")


# =============================================================================
# STEP 4: INSPECT
# =============================================================================


def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 5) -> None:
    """In ra n chunk đầu tiên trong ChromaDB."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(limit=n, include=["documents", "metadatas"])

        print(f"\n=== Top {n} chunks trong index ===\n")
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            print(f"[Chunk {i+1}]")
            print(f"  Source: {meta.get('source', 'N/A')}")
            print(f"  Section: {meta.get('section', 'N/A')}")
            print(f"  Effective Date: {meta.get('effective_date', 'N/A')}")
            print(f"  Text preview: {doc[:120]}...")
            print()
    except Exception as e:
        print(f"Lỗi khi đọc index: {e}")
        print("Hãy chạy build_index() trước.")


def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR) -> None:
    """Kiểm tra phân phối metadata trong toàn bộ index."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(include=["metadatas"])

        print(f"\nTổng chunks: {len(results['metadatas'])}")

        departments: Dict[str, int] = {}
        missing_date = 0
        for meta in results["metadatas"]:
            dept = meta.get("department", "unknown")
            departments[dept] = departments.get(dept, 0) + 1
            if meta.get("effective_date") in ("unknown", "", None):
                missing_date += 1

        print("Phân bố theo department:")
        for dept, count in sorted(departments.items()):
            print(f"  {dept}: {count} chunks")
        print(f"Chunks thiếu effective_date: {missing_date}")

    except Exception as e:
        print(f"Lỗi: {e}. Hãy chạy build_index() trước.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build RAG index (Sprint 1)")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Chạy build_index() (cần embedding đã cấu hình)",
    )
    parser.add_argument("--list", action="store_true", help="list_chunks() sau khi build")
    parser.add_argument("--inspect", action="store_true", help="inspect_metadata_coverage()")
    args = parser.parse_args()

    print("=" * 60)
    print("Sprint 1: Build RAG Index")
    print("=" * 60)

    doc_files = list(DOCS_DIR.glob("*.txt"))
    print(f"\nTìm thấy {len(doc_files)} tài liệu:")
    for f in doc_files:
        print(f"  - {f.name}")

    print("\n--- Test preprocess + chunking (không cần API key với local embedding) ---")
    for filepath in doc_files[:1]:
        raw = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(filepath))
        chunks = chunk_document(doc)
        print(f"\nFile: {filepath.name}")
        print(f"  Metadata: {doc['metadata']}")
        print(f"  Số chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n  [Chunk {i+1}] Section: {chunk['metadata']['section']}")
            print(f"  Text: {chunk['text'][:150]}...")

    if args.build:
        print("\n--- Build Full Index ---")
        build_index()
    if args.list:
        list_chunks()
    if args.inspect:
        inspect_metadata_coverage()

    if not args.build and not args.list and not args.inspect:
        print("\nSprint 1: chạy `python index.py --build` để index đầy đủ.")
        print("Embedding: EMBEDDING_PROVIDER=local (mặc định) hoặc openai + OPENAI_API_KEY/GITHUB_TOKEN (+ OPENAI_BASE_URL nếu cần).")
