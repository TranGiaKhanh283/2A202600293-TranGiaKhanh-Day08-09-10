# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Trần Gia Khánh 
**Vai trò trong nhóm:** Retrieval Owner + Eval/Documentation (triển khai end-to-end)  
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab Day 08, tôi triển khai pipeline RAG theo 4 sprint, từ code lõi tới tài liệu và kết quả đánh giá. Ở Sprint 1, tôi hoàn thiện `index.py`: preprocess metadata (`source`, `department`, `effective_date`, `access`), chunk theo section + paragraph, build index vào ChromaDB, thêm chức năng inspect chunk/metadata. Ở Sprint 2-3, tôi hoàn thiện `rag_answer.py`: dense retrieval, sparse BM25, hybrid retrieval bằng RRF, rerank (có fallback), grounded prompt, abstain có kiểm soát cho câu hỏi không có dữ liệu thật. Tôi thêm `openai_client.py` để hỗ trợ gọi model bằng GitHub token (endpoint mới `models.github.ai/inference`) và tự map model name. Ở Sprint 4, tôi hoàn thiện `eval.py` để chạy baseline/variant, tính metric, so sánh A/B, export `eval_scorecard.csv`. Tôi cũng tạo `rag_pipeline.py` để chạy end-to-end bằng CLI.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, tôi hiểu rõ nhất hai điểm: **retrieval quyết định chất lượng hơn generation**, và **grounded prompting cần cân bằng giữa an toàn và hữu dụng**. Ban đầu tôi nghĩ chỉ cần prompt tốt là model trả lời ổn, nhưng thực tế nếu chunk không đúng nguồn hoặc top-k thiếu context, model sẽ hoặc hallucinate, hoặc từ chối quá mức. Khi thêm hybrid retrieval (dense + BM25), context recall tăng rõ ở các câu có alias/keyword như “Approval Matrix”. Tôi cũng hiểu sâu hơn về abstain logic: nếu ép model “luôn trả lời không đủ dữ liệu” quá cứng thì nó dễ từ chối cả câu có context liên quan; ngược lại, nếu không ràng buộc thì dễ bịa. Vì vậy tôi chỉnh prompt theo hướng “chỉ từ chối khi toàn bộ context không liên quan” và bổ sung rule kiểm tra mã lỗi `ERR-*` không xuất hiện trong docs để chặn hallucination.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Khó khăn lớn nhất là vấn đề môi trường Python/ML trên Windows: `sentence-transformers` phụ thuộc `torch/torchvision` bị lệch version nên không load được `BertModel`, kéo theo retrieval quality giảm mạnh khi phải dùng embedding fallback. Ban đầu tôi giả thuyết lỗi do prompt hoặc do Chroma query sai, nhưng sau khi trace stack thì gốc lỗi nằm ở dependency runtime. Ngoài ra, tôi gặp lỗi API 404 khi gọi OpenAI SDK bằng GitHub token; tưởng token sai nhưng thực tế endpoint cũ `models.inference.ai.azure.com` đã deprecated. Sau khi chuyển sang `https://models.github.ai/inference`, thêm header API version, và map model sang dạng `openai/gpt-4o-mini`, request hoạt động đúng. Một điểm bất ngờ khác là khi retrieval score thấp, model dễ trả lời “không đủ dữ liệu”; tôi xử lý bằng cách tự động mở rộng số chunk đầu vào khi tín hiệu retrieval yếu.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** `q07 - "Approval Matrix để cấp quyền hệ thống là tài liệu nào?"`

**Phân tích:**

Đây là câu rất đại diện cho lỗi retrieval theo alias. Trong tài liệu thật, tên hiện tại là `Access Control SOP`, còn “Approval Matrix” là tên cũ được nhắc trong ghi chú. Ở baseline dense, hệ có thể trả về chunk liên quan access control nhưng không ổn định; khi score thấp và context chưa đủ chứa dòng alias, model dễ trả lời “không đủ dữ liệu”. Điểm lỗi chính nằm ở **retrieval** chứ không phải generation: indexing vẫn có đúng tài liệu, nhưng top-k chưa chắc lấy đúng đoạn nối alias -> tên mới. Khi chuyển sang variant hybrid (dense + BM25 + RRF), recall nguồn đúng cải thiện vì BM25 bắt được keyword “Approval Matrix”. Kết quả là hệ có xác suất cao hơn để đưa đúng chunk vào prompt và trả lời đúng mối quan hệ tài liệu cũ/mới. Tôi còn chỉnh prompt để giảm over-abstain: nếu context có bằng chứng gián tiếp về alias thì phải trả lời, không được từ chối máy móc. Bài học chính: với enterprise docs có đổi tên tài liệu, hybrid retrieval gần như bắt buộc.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi sẽ làm hai việc: (1) sửa triệt để môi trường embedding local (torch/transformers) hoặc chuyển hẳn sang OpenAI embeddings để loại bỏ deterministic fallback; (2) tách một vòng rerank độc lập để A/B chuẩn hơn (baseline dense vs dense+rerank vs hybrid), từ đó biết biến nào cải thiện nhiều nhất theo từng nhóm câu hỏi (SLA, policy, alias, insufficient-context). Tôi cũng muốn thêm test tự động cho ngưỡng abstain để tránh trả lời “không đủ dữ liệu” quá mức.

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*
*Ví dụ: `reports/individual/nguyen_van_a.md`*
