# Báo Cáo Cá Nhân — Lab Day 09: Multi-Agent Orchestration

**Họ và tên:** Trần Gia Khánh  
**Vai trò trong nhóm:** Supervisor + Worker + MCP + Trace (triển khai repo)  
**Ngày nộp:** 14/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Phần tôi phụ trách (120–150 từ)

Tôi triển khai toàn bộ phần code Day 09 trong `day09/lab/`: **`graph.py`** (supervisor, `AgentState`, `supervisor_route_logic`, luồng retrieval/policy → synthesis, `save_trace` + dòng `traces.jsonl` theo format đề), **`workers/retrieval.py`** nối Chroma collection `rag_lab` và `index.get_embedding` copy từ Day 08, **`workers/policy_tool.py`** gọi **`mcp_server.dispatch_tool`** cho `search_kb` và `get_ticket_info` thay vì import Chroma, **`workers/synthesis.py`** dùng **`openai_client`** giống Day 08, **`eval_trace.py`** (sửa đọc UTF-8, bỏ qua `traces.jsonl` khi analyze, so sánh với `data/day08_baseline_metrics.json`), **`data/grading_questions.json`** mẫu, và điền **`docs/`**, **`reports/`**, **`contracts/worker_contracts.yaml`**. Tôi cũng chạy `eval_trace.py` và `--grading` để sinh `artifacts/traces/*.json` và `artifacts/grading_run.jsonl`.

---

## 2. Một quyết định kỹ thuật (120–150 từ)

**Quyết định:** Ưu tiên **rule routing** theo thứ tự: trước hết các tín hiệu **policy** (store credit, ngày 31/01/07/02, contractor + access, Level 2 emergency, Level 3 + phê duyệt), sau đó mới **retrieval** cho P1/SLA/ticket/ERR-403.

**Trade-off:** Keyword đơn giản, dễ debug và khớp `expected_route` trong `test_questions.json`, nhưng câu vừa SLA vừa access có thể cần classifier hoặc hai bước retrieve. **Bằng chứng trace:** Câu q15 đi vào `policy_tool_worker` với `route_reason` ghi contractor + access; câu q01 đi `retrieval_worker` với lý do P1/SLA.

---

## 3. Một lỗi đã sửa (120–150 từ)

**Lỗi:** `python eval_trace.py` sau 15 câu báo **`UnicodeDecodeError`** khi `analyze_traces()` đọc file JSON trên Windows (encoding mặc định cp1252), và đếm sai số trace vì **`run_id`** chỉ đến giây — nhiều câu ghi đè cùng file.

**Sửa:** Mở file với **`encoding="utf-8"`**, bỏ qua file **`traces.jsonl`** trong thư mục trace khi đọc JSON, và thêm **suffix UUID** vào `run_id` trong `make_initial_state`.

**Trước:** ~5 file trace / lỗi decode. **Sau:** 15 file `.json` riêng, `analyze_traces` chạy hết.

---

## 4. Tự đánh giá (80–100 từ)

**Tốt:** Luồng end-to-end ổn, MCP + trace đủ field cho SCORING, tái dùng index Day 08.  
**Yếu:** Môi trường `sentence-transformers`/torch trên máy báo lỗi `BertModel` → index fallback deterministic; cần API embedding ổn định cho chấm điểm thật.  
**Nhóm phụ thuộc tôi:** Toàn bộ pipeline và artifact nộp.

---

## 5. Nếu có thêm 2 giờ (60–80 từ)

Tôi sẽ chỉnh **`EMBEDDING_PROVIDER=openai`** trong `.env` và chạy lại 15 câu + grading để confidence và answer không phụ thuộc `SKIP_LLM`, rồi cập nhật một dòng trong `single_vs_multi_comparison.md` bằng số đo latency/confidence thật thay vì chế độ smoke test.

---

## Bổ sung: kiểm thử worker độc lập

Tôi chạy `python workers/retrieval.py`, `python workers/policy_tool.py`, `python workers/synthesis.py` để đảm bảo mỗi worker không phụ thuộc graph. Policy worker được kiểm tra với chunk giả lập Flash Sale và license để xác nhận `exceptions_found` được ghi. MCP được kiểm tra bằng `python mcp_server.py`. Điều này khớp rubric “mỗi worker test độc lập được”.

## Bổ sung: định dạng nộp

Trace đầy đủ: `artifacts/traces/<run_id>.json`. Tóm tắt theo dòng: `artifacts/traces/traces.jsonl` (append). Chấm điểm: `artifacts/grading_run.jsonl`. Báo cáo so sánh: `artifacts/eval_report.json`.
