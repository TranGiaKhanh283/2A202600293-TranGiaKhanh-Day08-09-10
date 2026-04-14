# System Architecture — Lab Day 09

**Nhóm:** (nhóm lab)  
**Ngày:** 14/04/2026  
**Version:** 1.0

---

## 1. Tổng quan kiến trúc

**Pattern đã chọn:** Supervisor–Worker (Python thuần, không bắt buộc LangGraph).

**Lý do chọn thay vì single agent (Day 08):** Monolithic RAG khó phân tách lỗi (retrieve vs policy vs generate). Supervisor tách **định tuyến** và **ghi trace** (`route_reason`, `workers_called`, MCP) nên debug và mở rộng (thêm MCP tool) không phải sửa một prompt khổng lồ.

---

## 2. Sơ đồ Pipeline

```
User task
    │
    ▼
┌─────────────────┐
│   Supervisor    │  supervisor_route, route_reason, needs_tool, risk_high
│   (graph.py)    │
└────────┬────────┘
         │
    route_decision
         │
    ┌────┴──────────────────────────────┐
    │                                   │
    ▼                                   ▼
retrieval_worker              policy_tool_worker
(Chroma rag_lab)              (MCP search_kb + analyze_policy + optional get_ticket_info)
    │                                   │
    └──────────────┬────────────────────┘
                   ▼
            synthesis_worker
            (grounded LLM, citations [1]…)
                   │
                   ▼
            final_answer, confidence, sources
```

---

## 3. Vai trò từng thành phần

### Supervisor (`graph.py`)

| Thuộc tính | Mô tả |
|-----------|-------|
| **Nhiệm vụ** | Chọn `retrieval_worker`, `policy_tool_worker`, hoặc `human_review` theo từ khóa |
| **Input** | `task` (chuỗi câu hỏi) |
| **Output** | `supervisor_route`, `route_reason`, `risk_high`, `needs_tool` |
| **Routing logic** | Rule-based: ưu tiên policy khi có store credit/temporal/contractor+access/Level 2–3 emergency; P1/SLA/ticket/ERR-403 → retrieval |

### Retrieval Worker (`workers/retrieval.py`)

| Thuộc tính | Mô tả |
|-----------|-------|
| **Nhiệm vụ** | Dense retrieval trên Chroma collection `rag_lab` |
| **Embedding** | Cùng `index.py`/`get_embedding` với Day 08 |
| **Top-k** | Mặc định 5; policy path có thể tăng qua `retrieval_top_k` |
| **Stateless?** | Có (chỉ đọc `task`, ghi chunks vào state) |

### Policy Tool Worker (`workers/policy_tool.py`)

| Thuộc tính | Mô tả |
|-----------|-------|
| **Nhiệm vụ** | Lấy evidence qua **MCP `search_kb`** (không import ChromaDB), phân tích exception (Flash Sale, digital, …) |
| **MCP** | `search_kb`, `get_ticket_info` khi có ticket/P1 |
| **Exception** | Flash Sale, digital product, activated — rule-based + `policy_result` |

### Synthesis Worker (`workers/synthesis.py`)

| Thuộc tính | Mô tả |
|-----------|-------|
| **LLM** | `LLM_MODEL` + `openai_client` (GitHub Models / OpenAI) |
| **Temperature** | 0 |
| **Grounding** | Chỉ context đã đưa; nhắc abstain khi thiếu fact (vd. phạt SLA) |
| **Output** | `final_answer`, `sources`, `confidence`

### MCP Server (`mcp_server.py`)

| Tool | Input | Output |
|------|-------|--------|
| search_kb | query, top_k | chunks, sources |
| get_ticket_info | ticket_id | mock ticket + notifications |
| check_access_permission | access_level, requester_role, is_emergency | approvers, emergency_override |

---

## 4. Shared State Schema

| Field | Type | Mô tả | Ai ghi |
|-------|------|-------|--------|
| task | str | Câu hỏi | supervisor đọc |
| supervisor_route | str | Worker đã chọn | supervisor |
| route_reason | str | Lý do route (bắt buộc rõ) | supervisor |
| retrieved_chunks | list | Evidence | retrieval hoặc MCP (policy) |
| policy_result | dict | Kết quả policy | policy_tool |
| mcp_tools_used | list | Tool calls (JSON có `tool`, `input`, `output`) | policy_tool |
| worker_io_logs | list | Log I/O từng worker | workers |
| final_answer | str | Câu trả lời | synthesis |
| confidence | float | 0–1 | synthesis |

---

## 5. Lý do chọn Supervisor-Worker so với Single Agent (Day 08)

| Tiêu chí | Single Agent (Day 08) | Supervisor-Worker (Day 09) |
|----------|----------------------|------------------------------|
| Debug khi sai | Khó — một pipeline | Dễ — trace + test worker riêng |
| Routing visibility | Không | Có `route_reason` |
| Thêm external API | Sửa prompt/code lẫn | Thêm MCP tool + route rule |
| Policy vs retrieval | Trộn trong một hàm | Tách worker + MCP |

---

## 6. Giới hạn và cải tiến

1. Routing keyword có thể chồng lấn (multi-hop): cần classifier hoặc multi-step graph.
2. `sentence-transformers` lỗi môi trường → fallback embedding yếu; nên dùng `EMBEDDING_PROVIDER=openai` hoặc sửa torch.
3. Confidence hiện là heuristic; có thể thêm LLM-as-judge.
