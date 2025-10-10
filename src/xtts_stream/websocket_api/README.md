# WebSocket API XTTS

В директории `xtts_stream/websocket_api` собраны все артефакты, связанные с потоковым WebSocket API:

- `server/` — FastAPI-приложение с эндпоинтом ElevenLabs-совместимого протокола `stream-input`.
- `client/` — референсный клиент `example.py`, который демонстрирует handshake, работу с расписанием и live-проигрывателем.
- `README.md` (этот файл) — краткое описание структуры и ссылки на ключевые команды.

## Запуск сервера

```bash
PYTHONPATH=src python -m xtts_stream.websocket_api.server.app
```

Перед запуском убедитесь, что переменная `XTTS_SETTINGS_FILE` указывает на YAML с путями к весам модели, или положите `config.yaml` в корень репозитория.

## Клиентский пример

```bash
python -m xtts_stream.websocket_api.client.example \
  --host 127.0.0.1 --port 60215 --voice-id demo \
  --sr 24000 --text "Привет, это тестовый запрос" --play --no-save
```

Скрипт умеет разбивать текст на чанки, подстраивать параметры генерации и, при наличии `ffplay`, воспроизводить поток в реальном времени.
