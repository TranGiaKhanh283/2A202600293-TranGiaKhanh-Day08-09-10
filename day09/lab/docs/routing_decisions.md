# Routing Decisions Log — Lab Day 09

**Nhóm:** (nhóm lab)  
**Ngày:** 14/04/2026

Các quyết định dưới đây lấy từ trace thực tế trong `artifacts/traces/` (sau khi chạy `python eval_trace.py`).

---

## Routing Decision #1

**Task đầu vào:**
> SLA xử lý ticket P1 là bao lâu?

**Worker được chọn:** `retrieval_worker`  
**Route reason (từ trace):** `task contains P1/SLA/ticket/escalation/incident/err code → retrieval`  
**MCP tools được gọi:** (không)  
**Workers called sequence:** `retrieval_worker` → `synthesis_worker`

**Kết quả thực tế:**
- final_answer: grounded placeholder khi `SKIP_LLM=1`, hoặc câu trả lời có trích [1] khi bật LLM
- confidence: ~0.25 (SKIP_LLM + heuristic)
- Correct routing? Yes (đúng expected retrieval cho câu SLA)

**Nhận xét:** Route đúng: cần tài liệu SLA, không cần policy tool trừ khi có ngoại lệ refund/access.

---

## Routing Decision #2

**Task đầu vào:**
> Ai phải phê duyệt để cấp quyền Level 3?

**Worker được chọn:** `policy_tool_worker`  
**Route reason:** `policy_tool_worker | Level 3 access / approval policy | MCP: search_kb for evidence (not direct Chroma in policy worker)`  
**MCP tools được gọi:** `search_kb`  
**Workers called sequence:** `policy_tool_worker` → `synthesis_worker`

**Kết quả thực tế:**
- Evidence lấy qua MCP `search_kb` (delegate tới retrieval dense trong server)
- confidence: theo synthesis
- Correct routing? Yes (khớp `expected_route` trong `test_questions.json`)

**Nhận xét:** Policy path buộc evidence qua MCP để đáp ứng yêu cầu “không Chroma trực tiếp trong policy_tool”.

---

## Routing Decision #3

**Task đầu vào:**
> Ticket P1 lúc 2am. Cần cấp Level 2 access tạm thời cho contractor để thực hiện emergency fix. Đồng thời cần notify stakeholders theo SLA. Nêu đủ cả hai quy trình.

**Worker được chọn:** `policy_tool_worker`  
**Route reason:** chứa `contractor` + `access` / `level` → cross-doc policy path; `top_k` MCP tăng lên ≥8  
**MCP tools được gọi:** `search_kb` (và có thể `get_ticket_info` nếu match ticket/P1)  
**Workers called sequence:** `policy_tool_worker` → `synthesis_worker`

**Kết quả thực tế:**
- `retrieved_sources` thường gồm cả SLA và access control nếu retrieve đủ chunk
- Correct routing? Yes (multi-hop mong đợi policy route + synthesis hợp nhất)

**Nhận xét:** Đây là câu khó; chất lượng phụ thuộc embedding + top_k, không chỉ routing.

---

## Tổng kết

### Routing Distribution (15 câu test, `eval_trace.py`)

| Worker | Số câu | % |
|--------|--------|---|
| retrieval_worker | 9 | 60% |
| policy_tool_worker | 6 | 40% |
| human_review | 0 (trong batch test) | 0% |

### Lesson Learned

- Ưu tiên **policy** khi có tín hiệu temporal (31/01, 07/02), store credit, contractor + access, hoặc Level 2 emergency.
- **Retrieval** cho P1/SLA/ticket/ERR-403 để khớp câu abstain và SLA thuần.
