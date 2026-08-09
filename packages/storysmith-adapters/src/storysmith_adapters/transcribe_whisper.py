from __future__ import annotations

import asyncio
import tempfile

from faster_whisper import WhisperModel

_MODEL_SIZE = "base"


class WhisperTranscribe:
    """TranscribePort via faster-whisper, running locally on CPU -- no per-call
    API cost, unlike every other port (§5)."""

    def __init__(self) -> None:
        self._model: WhisperModel | None = None

    def _get_model(self) -> WhisperModel:
        # Lazy: constructing WhisperModel loads (and on first use, downloads)
        # model weights, so defer that until transcribe() is actually called.
        if self._model is None:
            self._model = WhisperModel(_MODEL_SIZE, device="cpu", compute_type="int8")
        return self._model

    async def transcribe(self, *, audio: bytes) -> tuple[list[dict[str, str | float]], float]:
        return await asyncio.to_thread(self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: bytes) -> tuple[list[dict[str, str | float]], float]:
        model = self._get_model()
        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
            tmp.write(audio)
            tmp.flush()
            segments, _info = model.transcribe(tmp.name, word_timestamps=True)
            words: list[dict[str, str | float]] = []
            for segment in segments:
                for word in segment.words or []:
                    words.append({"word": word.word.strip(), "start": word.start, "end": word.end})
        return words, 0.0
