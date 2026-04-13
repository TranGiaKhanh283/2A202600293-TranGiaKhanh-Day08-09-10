"""
OpenAI SDK client — OpenAI chính thức hoặc GitHub Models (inference).

GitHub Models (PAT có scope `models`):
  - Base URL hiện tại: https://models.github.ai/inference
    (SDK sẽ gọi .../inference/chat/completions)
  - Endpoint cũ https://models.inference.ai.azure.com đã deprecated → thường trả 404.

Model id trên GitHub thường có dạng openai/gpt-4o, openai/gpt-4o-mini (xem marketplace/models).
"""

from __future__ import annotations

import os

# Không dùng /v1 — path OpenAI SDK là base + "/chat/completions"
DEFAULT_GITHUB_INFERENCE_BASE = os.getenv(
    "GITHUB_INFERENCE_BASE",
    "https://models.github.ai/inference",
)

GITHUB_API_VERSION = os.getenv("GITHUB_API_VERSION", "2022-11-28")


def _normalize_inference_base(base: str) -> str:
    b = base.rstrip("/")
    lower = b.lower()
    if "models.inference.ai.azure.com" in lower:
        print(
            "[warn] OPENAI_BASE_URL trỏ tới Azure GitHub Models (deprecated). "
            "Đổi sang https://models.github.ai/inference — đang tự chuyển.",
        )
        return DEFAULT_GITHUB_INFERENCE_BASE
    return b


def resolved_openai_base_url() -> str:
    """Base URL thực tế (sau khi áp dụng default GitHub / normalize)."""
    base = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if not base and os.getenv("GITHUB_TOKEN") and not (os.getenv("OPENAI_API_KEY") or "").strip():
        base = DEFAULT_GITHUB_INFERENCE_BASE
    if not base:
        return ""
    return _normalize_inference_base(base)


def is_github_models_endpoint() -> bool:
    return "models.github.ai" in resolved_openai_base_url().lower()


def format_model_for_inference(model: str) -> str:
    """
    GitHub Models cần id dạng vendor/model (vd openai/gpt-4o-mini).
    OpenAI.com chấp nhận gpt-4o-mini — giữ nguyên khi không dùng GitHub.
    """
    m = (model or "").strip()
    if not m:
        return m
    if not is_github_models_endpoint():
        return m
    if "/" in m:
        return m
    return f"openai/{m}"


def get_openai_client():
    """
    - OPENAI_API_KEY hoặc GITHUB_TOKEN (ưu tiên OPENAI_API_KEY nếu có cả hai).
    - OPENAI_BASE_URL: tuỳ chọn; nếu chỉ có GITHUB_TOKEN → dùng models.github.ai/inference.
    """
    from openai import OpenAI

    api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_TOKEN") or "").strip()
    if not api_key:
        raise RuntimeError(
            "Cần OPENAI_API_KEY hoặc GITHUB_TOKEN trong .env (hoặc biến môi trường).",
        )

    base = resolved_openai_base_url()

    kwargs: dict = {"api_key": api_key}
    if base:
        kwargs["base_url"] = base
    if is_github_models_endpoint():
        kwargs["default_headers"] = {
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }

    return OpenAI(**kwargs)
