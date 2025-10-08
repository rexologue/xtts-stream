# client_min.py
import os, json, base64, asyncio, websockets, sys

URI = os.environ.get(
    "WS_URI",
    "ws://127.0.0.1:8000/v1/text-to-speech/VOICE123/stream-input?model_id=eleven_flash_v2_5&output_format=pcm_24000"
)

async def main():
    async with websockets.connect(URI, extra_headers={"xi-api-key": "dummy"}) as ws:
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
