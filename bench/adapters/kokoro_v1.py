"""Kokoro-82M v1.0 (kokoro-onnx 0.6+, CPU), one sentence at a time, voice af_heart: the cheapest
upgrade from v0.19. No German, no cloning."""
from _util import models, sentences, mono

VOICE = "af_heart"


class Engine:
    sample_rate = 24000
    info = {"voice": VOICE, "device": "cpu (onnxruntime)", "model": "kokoro-v1.0.onnx"}

    def __init__(self):
        from kokoro_onnx import Kokoro

        self.k = Kokoro(models("kokoro_v1", "kokoro-v1.0.onnx"), models("kokoro_v1", "voices-v1.0.bin"))

    def stream(self, text):
        for s in sentences(text):
            samples, sr = self.k.create(s, voice=VOICE, speed=1.0, lang="en-us")
            self.sample_rate = sr
            yield mono(samples)


def make(lang, voice=None):
    if lang != "en":
        raise NotImplementedError("Kokoro has no German voice")
    return Engine()
