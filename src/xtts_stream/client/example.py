# client_min.py
import os, json, base64, asyncio, websockets, sys, inspect
from urllib.parse import urlencode

BASE_URI = os.environ.get(
    "WS_URI_BASE",
    "ws://0.0.0.0:60215/v1/text-to-speech/VOICE123/stream-input",
)

query_params = {
    "output_format": os.environ.get("WS_OUTPUT_FORMAT", "pcm_24000"),
    "inactivity_timeout": os.environ.get("WS_INACTIVITY_TIMEOUT", "20"),
    "sync_alignment": os.environ.get("WS_SYNC_ALIGNMENT", "false"),
    "stream_chunk_size": os.environ.get("WS_STREAM_CHUNK_SIZE", "20"),
    "overlap_wav_len": os.environ.get("WS_OVERLAP_WAV_LEN", "512"),
    "speed": os.environ.get("WS_SPEED", "1.0"),
}

left_ctx = os.environ.get("WS_LEFT_CONTEXT_SECONDS")
if left_ctx is not None:
    query_params["left_context_seconds"] = left_ctx

URI = os.environ.get("WS_URI", f"{BASE_URI}?{urlencode(query_params)}")

async def main():
    headers = {"xi-api-key": "dummy"}
    connect_params = inspect.signature(websockets.connect.__init__).parameters
    header_kwarg = "additional_headers" if "additional_headers" in connect_params else "extra_headers"
    
    async with websockets.connect(URI, **{header_kwarg: headers}) as ws:
        # init: пробел + расписание триггера (как у 11labs)
        await ws.send(json.dumps({
            "text": " ",
            "generation_config": {"chunk_length_schedule": [80, 120, 160, 200]}
        }))

        # шлём фразу кусками
        await ws.send(json.dumps({"text": "Привет! Это тест потокового TTS. "}))
        await ws.send(json.dumps({"flush": True}))
        await ws.send(json.dumps({"text": "Продолжаем говорить без задержек. "}))
        await ws.send(json.dumps({"text": ""}))  # завершить текущую реплику

        # читаем поток и пишем сырой PCM в stdout (можно слушать через ffplay)
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("isFinal"):
                break
            audio = base64.b64decode(msg["audio"])
            sys.stdout.buffer.write(audio)
            sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(main())
