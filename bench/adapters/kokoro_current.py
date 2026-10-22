"""The baseline in English: Kokoro-82M v0.19 through kokoro-onnx 0.3.3 on the CPU, one sentence at
a time, voice af_bella: a common setup on a Jetson today. Kokoro has no German."""
from _util import models, sentences, mono

VOICE = "af_bella"


class Engine:
    sample_rate = 24000
    info = {"voice": VOICE, "device": "cpu (onnxruntime)", "model": "kokoro-v0_19.onnx"}

    def __init__(self):
        from kokoro_onnx import Kokoro

        self.k = Kokoro(models("kokoro_current", "kokoro-v0_19.onnx"), models("kokoro_current", "voices.json"))

    def stream(self, text):
        for s in sentences(text):
            samples, sr = self.k.create(s, voice=VOICE, speed=1.0, lang="en-us")
            self.sample_rate = sr
            yield mono(samples)


def make(lang, voice=None):
    if lang != "en":
        raise NotImplementedError("Kokoro has no German voice")
    return Engine()
