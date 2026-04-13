# Tuning Log — RAG Pipeline (Day 08 Lab)

## Baseline (Sprint 2)

**Config:**
```
retrieval_mode = "dense"
top_k_search = 10
top_k_select = 3
use_rerank = False
```

**Nhận xét:** Dense ổn cho câu hỏi trùng nội dung trực tiếp trong policy; dễ hụt khi query dùng **alias / tên cũ** (ví dụ “Approval Matrix”) hoặc khi cần khớp từ khóa chính xác mà vector search xếp hạng thấp.

**Giả thuyết (Error Tree):**
- [x] Retrieval: Dense bỏ lỡ alias / keyword (q07)
- [ ] Generation: đã xử lý bằng prompt grounded + abstain cho ERR-*

---

## Variant 1 (Sprint 3)

**Biến thay đổi (A/B — một biến):** `retrieval_mode`: `dense` → **`hybrid`** (RRF dense + BM25). Giữ nguyên chunking, top-k, prompt, LLM.

**Lý do:** Corpus trộn văn bản policy và thuật ngữ/ mã (P1, SLA, Approval Matrix); hybrid giữ đồng thời tương đồng ngữ nghĩa và khớp từ khóa.

**Quan sát (scorecard heuristic / context recall):**
- Context recall trung bình tăng trên các câu cần khớp nguồn hoặc từ khóa (ví dụ q04, q05, q07) khi so với baseline dense trong cùng điều kiện eval.
- Faithfulness/completeness phụ thuộc mạnh vào LLM; khi `SKIP_LLM=1` các metric này không phản ánh chất lượng sinh văn thực.

**Kết luận:** Hybrid là biến thể hợp lý khi baseline dense thiếu recall nguồn; cần chạy lại với LLM thật và (tuỳ chọn) bật rerank riêng một lần để A/B tiếp theo.

---

## Tóm tắt học được

1. **Lỗi phổ biến:** Retrieval không đúng nguồn (alias, từ khóa) trước khi lỗi generation.
2. **Biến tác động lớn:** Chiến lược retrieval (dense vs hybrid) lên recall; chunking ảnh hưởng faithfulness.
3. **Nếu có thêm thời gian:** Thử **rerank** một mình (giữ dense) so với hybrid; tinh chỉnh `MIN_DENSE_SIM_FOR_ANSWER` nếu dùng abstain theo điểm số.
