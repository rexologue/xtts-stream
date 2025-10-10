# XTTS Stream Inference

Пакет предназначен для развёртывания inference-стека XTTS и потокового WebSocket API, совместимого с протоколом ElevenLabs `stream-input`.

## Возможности

- Клонирование голосов и извлечение эмбеддингов из референсного аудио.
- Мультиязычная нормализация текста, сплиттер предложений и кэширование голосов.
- CLI для офлайн-генерации WAV и потоковый режим выдачи.
- WebSocket-сервис на FastAPI, который повторяет контракт ElevenLabs и может использоваться «из коробки».

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Добавьте репозиторий в `PYTHONPATH` перед запуском любых модулей:

```bash
export PYTHONPATH="$(pwd)/src"
```

Убедитесь, что в системе установлены подходящие колёса `torch`/`torchaudio` с поддержкой вашей GPU (или CPU).

## Конфигурация модели

Сервис и CLI читают настройки из YAML-файла. Скопируйте пример и отредактируйте пути к весам:

```bash
cp config.example.yaml config.yaml
```

Минимально необходимы файлы:

- `config.json`
- `model.pth`
- `dvae.pth`
- `mel_stats.pth`
- `vocab.json`

Положите их в одну директорию и укажите путь в секции `model` вашего `config.yaml`. Для использования альтернативного пути задайте переменную окружения:

```bash
export XTTS_SETTINGS_FILE=/абсолютный/путь/к/config.yaml
```

## Офлайн-использование

Убедитесь, что скачали директорию с весами XTTS (`model.pth`, `config.json`, `vocab.json`, опционально `speakers_xtts.pth`) и подготовили референсную озвучку ≥3 секунд. Запустите CLI в потоковом или пакетном режиме:

```bash
PYTHONPATH=src python -m xtts_stream.inference.infer_xtts \
  --config /path/to/config.json \
  --checkpoint /path/to/model.pth \
  --tokenizer /path/to/vocab.json \
  --speakers /path/to/speakers_xtts.pth \
  --text "Текст для синтеза" \
  --language ru \
  --reference /path/to/reference.wav \
  --output ./generated.wav
```

Флаг `--stream` включает вывод аудио по мере генерации:

```bash
PYTHONPATH=src python -m xtts_stream.inference.infer_xtts \
  --config /path/to/config.json \
  --checkpoint /path/to/model.pth \
  --tokenizer /path/to/vocab.json \
  --reference /path/to/reference.wav \
  --output ./generated.wav \
  --stream
```

Подробнее о параметрах см. `python -m xtts_stream.inference.infer_xtts --help`.

## Потоковый WebSocket-сервис

Все компоненты собраны в `src/xtts_stream/websocket_api/`:

- `server/app.py` — FastAPI-приложение.
- `server/settings.py` — загрузка и валидация конфигурации.
- `client/example.py` — пример клиента, совместимого с протоколом ElevenLabs.
- `README.md` — краткое описание запуска.

Запуск сервера:

```bash
PYTHONPATH=src python -m xtts_stream.websocket_api.server.app
```

Пример клиента:

```bash
python -m xtts_stream.websocket_api.client.example \
  --host 127.0.0.1 --port 60215 --voice-id demo \
  --sr 24000 --text "Привет, это тест" --play --no-save
```

## Структура репозитория

- `src/xtts_stream/inference/` — реализация inference-стека XTTS (GPT, Perceiver, HiFi-GAN, токенизаторы, вспомогательные утилиты).
- `src/xtts_stream/websocket_api/` — сервер и клиент WebSocket API.
- `src/xtts_stream/wrappers/` — абстракции для подключения других моделей к потоковому сервису.
- `src/xtts_stream/resources/` — дополнительные данные (например, конфиги языков).

## Полезные заметки

- Для опции `--split-text` CLI использует пайплайны spaCy — установите соответствующие модели.
- Потоковый вывод по умолчанию включает шумоподавление; отключите вызов в `xtts_stream.inference.xtts::_apply_noise_reduction`, если нужен «сырой» сигнал.
- Архитектура XTTS была изначально опубликована компанией Coqui.
