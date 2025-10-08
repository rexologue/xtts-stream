"""Compatibility shim for the legacy `tokenizer` module."""

from pathlib import Path
import runpy
import sys

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if __name__ == "__main__":
    runpy.run_module("xtts_stream.core.tokenizer", run_name="__main__")
else:
    _module = __import__("xtts_stream.core.tokenizer", fromlist=["*"])
    globals().update({k: getattr(_module, k) for k in dir(_module) if not k.startswith("__")})
    if hasattr(_module, "__all__"):
        __all__ = _module.__all__  # type: ignore
    sys.modules.setdefault(__name__, _module)
