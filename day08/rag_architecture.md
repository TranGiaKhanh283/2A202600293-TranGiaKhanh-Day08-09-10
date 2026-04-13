# RAG Architecture — Day 08 (CS + IT Helpdesk Assistant)

## Tổng quan

Hệ trả lời câu hỏi nội bộ dựa trên **ChromaDB** (vector store cục bộ), **embedding** (OpenAI `text-embedding-3-small` hoặc `sentence-transformers` local), và **LLM** (OpenAI / Gemini) với prompt **grounded** (chỉ trả lời từ context, trích dẫn `[1]`, abstain khi thiếu bằng chứng hoặc mã lỗi không có trong docs).

```
[data/docs/*.txt] → preprocess → chunk (theo === section ===) → embed → ChromaDB
       → query → dense | hybrid (RRF) → (optional rerank) → grounded prompt → LLM → answer
```

## Indexing (`index.py`)

| Tham số | Giá trị |
|---------|---------|
| Chunk size | ~400 token (ước lượng bằng ký tự/4), overlap ~80 |
| Chiến lược | Tách theo heading `=== ... ===`, gom paragraph; cắt dài theo đoạn |
| Metadata | `source`, `section`, `department`, `effective_date`, `access` |
| Vector DB | `chroma_db/`, collection `rag_lab`, cosine |

**Fallback:** nếu không load được `sentence-transformers` (lỗi torch/torchvision trên Windows), pipeline dùng **deterministic embedding** chỉ để smoke test; khuyến nghị `EMBEDDING_PROVIDER=openai` + `OPENAI_API_KEY`.

## Retrieval (`rag_answer.py`)

| Mode | Mô tả |
|------|--------|
| **dense** | Query embedding + Chroma similarity |
| **sparse** | BM25 trên toàn corpus (load từ Chroma) |
| **hybrid** | RRF kết hợp dense + sparse (trọng số 0.6 / 0.4) |
| **rerank** | Cross-encoder (tùy chọn); lỗi load → giữ thứ tự retrieval |

**Abstain:** không chunk; hoặc mã `ERR-*` trong câu hỏi không xuất hiện trong context đã lấy; hoặc (tùy cấu hình) điểm dense quá thấp (`MIN_DENSE_SIM_FOR_ANSWER`).

## Generation

- Prompt: trả lời đúng ngôn ngữ câu hỏi, trích dẫn `[n]`, câu magic **Không đủ dữ liệu** khi không đủ context.
- `SKIP_LLM=1`: không gọi API (chỉ để chạy eval / CI).

## Evaluation (`eval.py`)

- Metrics heuristic: faithfulness (từ overlap answer–context), relevance, context recall (expected source), completeness (overlap với `expected_answer`).
- `eval_scorecard.csv`: gộp baseline + variant với cột `run`.

## File liên quan

| File | Vai trò |
|------|---------|
| `lab/index.py` | Build index |
| `lab/rag_answer.py` | Retrieve + generate |
| `lab/rag_pipeline.py` | CLI end-to-end |
| `lab/eval.py` | Scorecard + A/B |
| `day08/rag_pipeline.py` | Wrapper gọi `lab/rag_pipeline.py` |
