"""Kokoro-82M v1.0 on the GPU: kokoro-onnx 0.6+ over onnxruntime-gpu (NVIDIA's Jetson build, CUDA
execution provider), one sentence at a time, voice af_heart. English only."""
from _util import models, sentences, mono

VOICE = "af_heart"


class Engine:
    sample_rate = 24000

    def __init__(self):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        session = ort.InferenceSession(models("kokoro_v1", "kokoro-v1.0.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        provider = session.get_providers()[0]
        if provider != "CUDAExecutionProvider":  # onnxruntime falls back to the CPU quietly
            raise RuntimeError(f"Kokoro's session runs on {provider}, not the GPU")
        self.k = Kokoro.from_session(session, models("kokoro_v1", "voices-v1.0.bin"))
        self.info = {"voice": VOICE, "device": "gpu (onnxruntime CUDA)", "model": "kokoro-v1.0.onnx"}

    def stream(self, text):
        for s in sentences(text):
            samples, sr = self.k.create(s, voice=VOICE, speed=1.0, lang="en-us")
            self.sample_rate = sr
            yield mono(samples)


def make(lang, voice=None):
    if lang != "en":
        raise NotImplementedError("Kokoro has no German voice")
    return Engine()
