"""Compatibility shim for the legacy `shared_configs` module."""

from pathlib import Path
import runpy
import sys

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if __name__ == "__main__":
    runpy.run_module("xtts_stream.core.shared_configs", run_name="__main__")
else:
    _module = __import__("xtts_stream.core.shared_configs", fromlist=["*"])
    globals().update({k: getattr(_module, k) for k in dir(_module) if not k.startswith("__")})
    if hasattr(_module, "__all__"):
        __all__ = _module.__all__  # type: ignore
    sys.modules.setdefault(__name__, _module)
