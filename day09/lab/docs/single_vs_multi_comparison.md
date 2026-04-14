# Single Agent vs Multi-Agent Comparison — Lab Day 09

**Nhóm:** (nhóm lab)  
**Ngày:** 14/04/2026

Số liệu Day 09 lấy từ `artifacts/eval_report.json` (sau `python eval_trace.py`). Day 08 lấy từ `data/day08_baseline_metrics.json` + scorecard Day 08.

---

## 1. Metrics Comparison

| Metric | Day 08 (Single Agent) | Day 09 (Multi-Agent) | Delta | Ghi chú |
|--------|----------------------|---------------------|-------|---------|
| Avg confidence (proxy) | ~0.76 (faithfulness 3.8/5) | 0.25 (run với SKIP_LLM=1) | thấp hơn khi SKIP_LLM | Với LLM thật, confidence heuristic tăng |
| Avg latency (ms) | ~2100 (ước tính) | ~710 (15 traces) | −1390 ms | Day 09 một lần LLM; không đo Day 08 cùng máy |
| Abstain rate (%) | ~6.7% (1/15 q09) | 0% literal abstain (SKIP placeholder) | — | So sánh công bằng cần tắt SKIP_LLM |
| Debuggability | Không có route | Có trace `route_reason` | N/A | Day 09 thắng rõ |

---

## 2. Phân tích theo loại câu hỏi

### 2.1 Câu đơn (single-document)

| Nhận xét | Day 08 | Day 09 |
|---------|--------|--------|
| Accuracy | Ổn định với hybrid retrieval | Phụ thuộc route đúng + cùng index |
| Latency | Một vòng RAG | Một vòng + overhead supervisor nhỏ |

**Kết luận:** Multi-agent không tự làm tăng accuracy trên câu đơn; lợi ích chính là **trace** và **tách worker**.

### 2.2 Multi-hop (cross-document)

| Nhận xét | Day 08 | Day 09 |
|---------|--------|--------|
| Routing visible? | Không | Có `supervisor_route` + `workers_called` |
| Observation | Một prompt | Policy path có MCP `search_kb` với `top_k` lớn hơn |

**Kết luận:** Day 09 dễ chứng minh đã gọi policy path và MCP; vẫn cần retrieval tốt để đủ chunk.

---

## 3. Debuggability

**Day 08:** Lỗi → đọc toàn `rag_answer.py` + index.  
**Day 09:** Lỗi → mở trace JSON → xem `route_reason` → test `retrieval.run` / `policy_tool.run` độc lập.

---

## 4. Kết luận

**Multi-agent tốt hơn ở:** (1) routing visibility, (2) tách policy vs retrieval, (3) MCP mở rộng tool không sửa core.

**Kém hơn / rủi ro:** Nhiều bước hơn; routing sai thì không cứu được bằng synthesis.

**Khi không nên dùng multi-agent:** Dataset nhỏ, một intent, không cần tool ngoài — single RAG đơn giản hơn.
