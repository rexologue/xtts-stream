import traceback 
from queue import Queue
from dataclasses import dataclass

import numpy as np
from .tone import StreamingCTCPipeline

FLUSH = object()   # спец-сентинел: завершить текущую фразу, НЕ выходить
CLOSE = None       # сентинел: финализировать и выйти

@dataclass
class Phrase:
    text: str
    start_time: float | None
    end_time: float | None


class ASRWorker:
    def __init__(self, audio_queue: Queue, result_queue: Queue, chunk_size: int = 2400):
        self.audio_queue = audio_queue
        self.result_queue = result_queue
        self.chunk_size = chunk_size
        
        # hidden 
        self.pipeline = None
        self.streaming_state = None
        self.internal_buffer = np.array([], dtype=np.int32)

    def _initialize_pipeline(self) -> bool:
        print("ASR worker: Initializing pipeline...")

        try:
            self.pipeline = StreamingCTCPipeline.from_hugging_face()
            print("ASR worker: Pipeline initialized successfully.")
            return True
        
        except Exception as e:
            print(f"ASR worker: FATAL - Failed to initialize pipeline: {e}")
            traceback.print_exc()
            return False

    def _process_buffer(self):
        while len(self.internal_buffer) >= self.chunk_size:
            chunk_to_process = self.internal_buffer[:self.chunk_size]
            self.internal_buffer = self.internal_buffer[self.chunk_size:]

            new_phrases, self.streaming_state = self.pipeline.forward(
                chunk_to_process, self.streaming_state
            )

            if new_phrases:
                # Отдаём список словарей с таймкодами
                payload = [
                    {
                        "text": getattr(p, "text", ""),
                        "start_time": getattr(p, "start_time", None),
                        "end_time": getattr(p, "end_time", None),
                    }
                    for p in new_phrases
                ]
                self.result_queue.put(payload)

    def _finalize_one(self):
        """Финализировать текущую фразу, отдать хвост и СБРОСИТЬ состояние, но НЕ выходить."""
        if len(self.internal_buffer) > 0:
            new_phrases, self.streaming_state = self.pipeline.forward(
                self.internal_buffer, self.streaming_state
            )
            self.internal_buffer = np.array([], dtype=np.int32)
            if new_phrases:
                payload = [
                    {"text": getattr(p, "text", ""),
                     "start_time": getattr(p, "start_time", None),
                     "end_time": getattr(p, "end_time", None)}
                    for p in new_phrases
                ]
                self.result_queue.put(payload)

        if self.streaming_state:
            final_phrases, _ = self.pipeline.finalize(self.streaming_state)
            self.streaming_state = None  # Сброс для следующего запроса
            if final_phrases:
                payload = [
                    {"text": getattr(p, "text", ""),
                     "start_time": getattr(p, "start_time", None),
                     "end_time": getattr(p, "end_time", None)}
                    for p in final_phrases
                ]
                self.result_queue.put(payload)

    def _finalize_and_close(self):
        """Финализировать и завершить процесс."""
        self._finalize_one()

    def run(self):
        if not self._initialize_pipeline():
            return

        print("ASR worker: Process started, entering main loop.")
        while True:
            try:
                incoming_audio_chunk = self.audio_queue.get()
                if incoming_audio_chunk is CLOSE:  # None
                    print("ASR worker: Received CLOSE signal.")
                    self._finalize_and_close()
                    break

                if incoming_audio_chunk is FLUSH:
                    print("ASR worker: Received FLUSH signal.")
                    self._finalize_one()
                    continue

                # обычный аудио-чанк
                self.internal_buffer = np.concatenate([self.internal_buffer, incoming_audio_chunk])
                self._process_buffer()

            except Exception as e:
                print(f"ASR worker: CRITICAL ERROR in main loop: {e}")
                traceback.print_exc()
                break

        print("ASR worker: Process finished.")