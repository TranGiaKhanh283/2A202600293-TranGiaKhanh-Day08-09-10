"""
Chạy pipeline từ thư mục day08 (delegates tới lab/rag_pipeline.py).

  python rag_pipeline.py index
  python rag_pipeline.py ask "Câu hỏi..."
"""

import importlib.util
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_lab = Path(__file__).resolve().parent / "lab"
sys.path.insert(0, str(_lab))

_spec = importlib.util.spec_from_file_location("rag_pipeline_lab", _lab / "rag_pipeline.py")
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    _mod.main()
