# Báo Cáo Nhóm — Lab Day 09: Multi-Agent Orchestration

**Tên nhóm:** (nhóm lab)  
**Thành viên:**
| Tên | Vai trò | Email |
|-----|---------|-------|
| Trần Gia Khánh | End-to-end (supervisor + workers + MCP + trace) | (điền) |

**Ngày nộp:** 14/04/2026  
**Repo:** `2A202600293-TranGiaKhanh-Day08-09-10`  
**Độ dài:** ~900 từ

---

## 1. Kiến trúc nhóm đã xây dựng (150–200 từ)

Hệ thống gồm **một supervisor** trong `graph.py` và **ba worker**: `retrieval_worker` (Chroma `rag_lab`, embedding đồng bộ `index.py` từ Day 08), `policy_tool_worker` (không gọi Chroma trực tiếp — chỉ **MCP `search_kb`** và phân tích exception rule-based), và `synthesis_worker` (LLM grounded qua `openai_client`). **MCP mock** trong `mcp_server.py` expose `search_kb`, `get_ticket_info`, `check_access_permission`, `create_ticket`; policy worker dùng tối thiểu hai tool đầu trong luồng thật.

**Routing logic:** rule-based theo từ khóa: ưu tiên **policy** khi có store credit / ngày temporal (31/01, 07/02) / contractor + access / Level 2 emergency / Level 3 approval; ngược lại **retrieval** khi có P1, SLA, ticket, escalation, hoặc mã ERR-403 (để khớp câu abstain). Nhánh `human_review` dành cho mã `ERR-*` không thuộc mẫu lab (vd. ERR-999).

**Ví dụ trace MCP:** Với câu policy, trace ghi `mcp_tools_used` chứa object có `tool: search_kb` và timestamp; có thể thêm `get_ticket_info` khi task chứa ticket/P1.

---

## 2. Quyết định kỹ thuật quan trọng nhất (200–250 từ)

**Quyết định:** Policy worker **không** import ChromaDB; mọi evidence KB đi qua **MCP `search_kb`** (mock gọi lại `retrieve_dense` trong process).

**Bối cảnh:** Đề bài Sprint 3 yêu cầu policy không truy cập vector store trực tiếp để mô phỏng ranh giới MCP trong production.

**Phương án đã cân nhắc:**

| Phương án | Ưu điểm | Nhược điểm |
|-----------|---------|-----------|
| Policy gọi Chroma giống retrieval | Nhanh, ít lớp | Vi phạm đề bài, khó đổi backend sau |
| MCP in-process (`dispatch_tool`) | Đủ điểm, trace giống protocol | Không phải HTTP thật |
| MCP HTTP (FastAPI) | Bonus +2 | Thời gian lab không đủ |

**Đã chọn:** MCP in-process + `search_kb` delegate retrieval — vừa đạt contract, vừa tái dùng embedding/index Day 08.

**Bằng chứng (trace):** Trong `artifacts/traces/*.json`, nhánh `policy_tool_worker` có `history` chứa dòng `MCP search_kb` và `mcp_tools_used` không rỗng.

---

## 3. Kết quả grading questions (150–200 từ)

**Pipeline:** `python eval_trace.py --grading` sinh `artifacts/grading_run.jsonl` với 10 câu mẫu (`data/grading_questions.json`).

**Tổng điểm raw ước tính:** (phụ thuộc LLM và embedding thật; chạy `SKIP_LLM=1` chỉ để kiểm tra không crash).

**Câu dễ kỳ vọng tốt:** gq01, gq05 (SLA rõ trong `sla_p1_2026`).

**Câu khó:** gq07 (abstain nếu không có mức phạt trong docs), gq09 (multi-hop SLA + Level 2).

**gq07:** Synthesis được prompt hướng dẫn không bịa số — trả lời “không có trong tài liệu” nếu context không chứa fact.

**gq09:** Route `policy_tool_worker` + MCP `top_k` cao; trace nên ghi `policy_tool_worker` + `synthesis_worker`.

---

## 4. So sánh Day 08 vs Day 09 (150–200 từ)

Từ `artifacts/eval_report.json`: **latency trung bình ~710 ms** (15 câu, có một câu first-hit chậm do load thư viện), **định tuyến 60% retrieval / 40% policy**. Day 08 không có `route_reason`; Day 09 có — **debuggability** cải thiện rõ.

**Bất ngờ:** Overhead supervisor nhỏ so với một lần LLM; chi phí chính vẫn là embedding + LLM.

**Multi-agent không giúp** nếu routing sai hoặc index/embed yếu — cần sửa gốc retrieval giống Day 08.

---

## 5. Phân công và đánh giá nhóm (100–150 từ)

| Thành viên | Phần đã làm | Sprint |
|------------|-------------|--------|
| Trần Gia Khánh | graph, workers, MCP, eval_trace, docs, reports | 1–4 |

**Làm tốt:** Repo chạy end-to-end, trace + grading JSONL, tài liệu điền số liệu thật.

**Chưa tốt:** Phụ thuộc môi trường Windows/torch; khi ST lỗi phải dùng embedding fallback hoặc OpenAI embeddings.

---

## 6. Nếu có thêm 1 ngày (50–100 từ)

1. Bật `EMBEDDING_PROVIDER=openai` hoặc sửa torch để bỏ deterministic embedding.  
2. Thử LLM classifier nhẹ cho routing thay vì keyword thuần trên câu multi-hop.

---

*File: `reports/group_report.md`*
