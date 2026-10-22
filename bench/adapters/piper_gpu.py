"""Piper on the GPU (onnxruntime-gpu, CUDA execution provider), its chunks as they come. Female
voices: de_DE-ramona-low, en_US-amy-medium."""
from _util import models, mono

PICK = {"de": "de_DE-ramona-low", "en": "en_US-amy-medium"}


class Engine:
    def __init__(self, name):
        from piper import PiperVoice

        self.v = PiperVoice.load(models("piper", name + ".onnx"), config_path=models("piper", name + ".onnx.json"), use_cuda=True)
        provider = self.v.session.get_providers()[0]
        if provider != "CUDAExecutionProvider":  # onnxruntime falls back to the CPU quietly
            raise RuntimeError(f"Piper's session runs on {provider}, not the GPU")
        self.sample_rate = self.v.config.sample_rate
        self.info = {"voice": name, "device": "gpu (onnxruntime CUDA)"}

    def stream(self, text):
        for chunk in self.v.synthesize(text):
            if len(chunk.audio_float_array):
                yield mono(chunk.audio_float_array)


def make(lang, voice=None):
    return Engine(PICK[lang])
