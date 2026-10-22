"""OmniVoice (k2-fsa, 0.6B, non-autoregressive) on the GPU in fp16, cloning the designed reference,
one sentence at a time (a whole sentence comes out at once)."""
import os

from _util import sentences, mono


class Engine:
    sample_rate = 24000

    def __init__(self, lang, voice):
        import torch
        from omnivoice import OmniVoice

        if not voice:
            raise ValueError("OmniVoice clones a voice here: run with --voices")
        self.m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
        self.voice = voice
        self.info = {"device": "cuda fp16", "voice": "cloned: " + os.path.basename(voice["wav"])}

    def stream(self, text):
        for s in sentences(text):
            audio = self.m.generate(text=s, ref_audio=self.voice["wav"], ref_text=self.voice["text"])
            yield mono(audio[0])


def make(lang, voice=None):
    return Engine(lang, voice)
