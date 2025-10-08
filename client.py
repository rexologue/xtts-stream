"""Re-export the example websocket client for convenience."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xtts_stream.client.example import main

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
