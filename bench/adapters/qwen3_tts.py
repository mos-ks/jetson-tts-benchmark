"""Qwen3-TTS 12Hz 0.6B Base on the GPU through faster-qwen3-tts (CUDA graphs, no flash-attn),
cloning the designed reference, streaming. QWEN_CHUNK: codec frames per chunk (default 2, the
smallest that its author found real time on constrained hardware); QWEN_MODEL to try another."""
import os

from _util import mono

MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
CHUNK = int(os.environ.get("QWEN_CHUNK", "2"))
LANG = {"en": "English", "de": "German"}
_shared = {}


class Engine:
    def __init__(self, lang, voice):
        from faster_qwen3_tts import FasterQwen3TTS

        if "m" not in _shared:  # one model for both languages
            m = FasterQwen3TTS.from_pretrained(MODEL)
            m.warmup(prefill_len=100)
            _shared["m"] = m
        self.m = _shared["m"]
        self.lang = LANG[lang]
        self.voice = voice
        self.sample_rate = 24000
        self.info = {"model": MODEL, "chunk_size": CHUNK, "device": "cuda", "voice": "cloned: " + os.path.basename(voice["wav"]) if voice else None}
        if not voice:
            raise ValueError("Qwen3-TTS Base clones a voice: run with --voices")

    def stream(self, text):
        for chunk, sr, _timing in self.m.generate_voice_clone_streaming(
            text=text, language=self.lang, ref_audio=self.voice["wav"], ref_text=self.voice["text"], chunk_size=CHUNK
        ):
            self.sample_rate = sr
            yield mono(chunk)


def make(lang, voice=None):
    return Engine(lang, voice)
