"""Chatterbox on the GPU, cloning the designed reference, one sentence at a time (it does not
stream): Turbo (350M) for English, Multilingual for German (and English with CHATTERBOX_ML=1)."""
import os

from _util import sentences, mono


class Engine:
    def __init__(self, lang, voice):
        if not voice:
            raise ValueError("Chatterbox clones a voice: run with --voices")
        self.voice = voice
        self.lang = lang
        if lang == "en" and os.environ.get("CHATTERBOX_ML") != "1":
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            self.m = ChatterboxTurboTTS.from_pretrained(device="cuda")
            self.kind = "turbo"
        else:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            self.m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
            self.kind = "multilingual"
        self.sample_rate = self.m.sr
        self.info = {"model": self.kind, "device": "cuda", "voice": "cloned: " + os.path.basename(voice["wav"])}

    def stream(self, text):
        for s in sentences(text):
            if self.kind == "turbo":
                wav = self.m.generate(s, audio_prompt_path=self.voice["wav"])
            else:
                wav = self.m.generate(s, language_id=self.lang, audio_prompt_path=self.voice["wav"])
            yield mono(wav)


def make(lang, voice=None):
    return Engine(lang, voice)
