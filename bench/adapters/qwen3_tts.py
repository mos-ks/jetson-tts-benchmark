"""Qwen3-TTS 12Hz 0.6B Base on the GPU through faster-qwen3-tts (CUDA graphs, no flash-attn),
cloning the designed reference, streaming. QWEN_CHUNK: codec frames per chunk (default 2, the
smallest that its author found real time on constrained hardware); QWEN_MODEL to try another.

One fix to its package (qwen-tts-hf 0.1.1): its RoPE shim for Transformers 5 reads config.rope_theta,
which Transformers 5 keeps in config.rope_parameters for the Mimi codec, so it fails to load
("'MimiConfig' object has no attribute 'rope_theta'"); the shim here reads either."""
import os

from _util import mono

MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
CHUNK = int(os.environ.get("QWEN_CHUNK", "2"))
LANG = {"en": "English", "de": "German"}
_shared = {}


def _rope_theta_either_place():
    from qwen_tts import _transformers_compat as compat
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    compat.patch_transformers_rope_registry()
    shim = compat._default_rope_parameters

    def default_rope(config, *args, **kwargs):
        if not hasattr(config, "rope_theta"):
            config.rope_theta = (getattr(config, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
        return shim(config, *args, **kwargs)

    if ROPE_INIT_FUNCTIONS.get("default") is shim:
        ROPE_INIT_FUNCTIONS["default"] = default_rope


class Engine:
    def __init__(self, lang, voice):
        from faster_qwen3_tts import FasterQwen3TTS

        _rope_theta_either_place()
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
