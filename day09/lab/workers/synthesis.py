"""
workers/synthesis.py — Synthesis Worker
Tổng hợp câu trả lời grounded; LLM qua openai_client (tương thích GitHub Models).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from dotenv import load_dotenv

load_dotenv(LAB_DIR / ".env")

WORKER_NAME = "synthesis_worker"

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

SYSTEM_PROMPT = """Bạn là trợ lý IT Helpdesk nội bộ.

Quy tắc:
1. CHỈ dùng thông tin trong phần CONTEXT. Không bịa số, không suy diễn ngoài tài liệu.
2. Trả lời tiếng Việt, ngắn gọn. Mỗi ý quan trọng kèm trích dẫn dạng [1], [2], … trùng với số đoạn trong CONTEXT.
3. Nếu CONTEXT không chứa đủ thông tin để trả lời → một câu: không đủ thông tin trong tài liệu nội bộ đã cung cấp (không thêm chi tiết giả định).
4. Nếu câu hỏi về mức phạt/số tiền mà CONTEXT không nêu → từ chối trả lời số, nói rõ không có trong tài liệu.
"""


def _call_llm(prompt: str) -> str:
    if os.getenv("SKIP_LLM") == "1":
        return (
            "Theo [1] trong context (SKIP_LLM=1), vui lòng xem các đoạn đã cung cấp."
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
        messages=[{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + prompt}],
        temperature=0,
        max_tokens=700,
    )
    return (response.choices[0].message.content or "").strip()


def _build_user_prompt(task: str, chunks: list, policy_result: dict) -> str:
    parts = ["CONTEXT (chỉ được dùng các đoạn sau):\n"]
    if chunks:
        for i, chunk in enumerate(chunks, 1):
            src = chunk.get("source", "unknown")
            parts.append(f"[{i}] Nguồn: {src}\n{chunk.get('text', '')}\n")
    else:
        parts.append("(Không có đoạn nào)\n")

    if policy_result and policy_result.get("exceptions_found"):
        parts.append("\nGHI CHÚ POLICY (exceptions):\n")
        for ex in policy_result["exceptions_found"]:
            parts.append(f"- {ex.get('rule', '')}\n")

    if policy_result and policy_result.get("policy_version_note"):
        parts.append(f"\nTemporal / version: {policy_result['policy_version_note']}\n")

    parts.append(f"\nCÂU HỎI: {task}\n\nTrả lời theo quy tắc hệ thống.")
    return "".join(parts)


def _estimate_confidence(chunks: list, answer: str, policy_result: dict) -> float:
    if not chunks:
        return 0.15
    low = answer.lower()
    if "không đủ thông tin" in low or "không có trong tài liệu" in low:
        return 0.35
    if chunks:
        avg_score = sum(float(c.get("score", 0) or 0) for c in chunks) / len(chunks)
    else:
        avg_score = 0.0
    penalty = 0.04 * len(policy_result.get("exceptions_found") or [])
    conf = min(0.95, max(0.2, avg_score - penalty))
    if re.search(r"\[[0-9]+\]", answer):
        conf = min(0.95, conf + 0.05)
    return round(conf, 2)


def synthesize(task: str, chunks: list, policy_result: dict) -> dict:
    prompt = _build_user_prompt(task, chunks, policy_result or {})
    answer = _call_llm(prompt)
    sources = list({c.get("source", "unknown") for c in chunks}) if chunks else []
    confidence = _estimate_confidence(chunks, answer, policy_result or {})
    return {"answer": answer, "sources": sources, "confidence": confidence}


def run(state: dict) -> dict:
    task = state.get("task", "")
    chunks = state.get("retrieved_chunks") or []
    policy_result = state.get("policy_result") or {}

    state.setdefault("workers_called", [])
    state.setdefault("history", [])
    state.setdefault("worker_io_logs", [])

    state["workers_called"].append(WORKER_NAME)

    worker_io = {
        "worker": WORKER_NAME,
        "input": {
            "task": task,
            "chunks_count": len(chunks),
            "has_policy": bool(policy_result),
        },
        "output": None,
        "error": None,
    }

    try:
        result = synthesize(task, chunks, policy_result)
        state["final_answer"] = result["answer"]
        state["sources"] = result["sources"]
        state["confidence"] = result["confidence"]
        if result["confidence"] < 0.4:
            state["hitl_triggered"] = True

        worker_io["output"] = {
            "answer_length": len(result["answer"]),
            "sources": result["sources"],
            "confidence": result["confidence"],
        }
        state["history"].append(
            f"[{WORKER_NAME}] confidence={result['confidence']} sources={result['sources']}"
        )
    except Exception as e:
        worker_io["error"] = {"code": "SYNTHESIS_FAILED", "reason": str(e)}
        state["final_answer"] = f"SYNTHESIS_ERROR: {e}"
        state["confidence"] = 0.0
        state["hitl_triggered"] = True
        state["history"].append(f"[{WORKER_NAME}] ERROR: {e}")

    state["worker_io_logs"].append(worker_io)
    return state


if __name__ == "__main__":
    st = {
        "task": "SLA ticket P1 là bao lâu?",
        "retrieved_chunks": [
            {
                "text": "P1: phản hồi 15 phút, xử lý 4 giờ.",
                "source": "sla_p1_2026.txt",
                "score": 0.92,
            }
        ],
        "policy_result": {},
    }
    print(run(st))
