"""Kyutai Pocket TTS (100M, CPU, streaming). Clones the designed reference where its cloning
weights are available (a gated Hugging Face repo); else a built-in voice. Threads: TORCH_THREADS
(default: all cores)."""
import os

from _util import mono

BUILTIN = {"en": "alba", "de": "juergen"}


class Engine:
    def __init__(self, lang, voice):
        import torch
        from pocket_tts import TTSModel

        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", os.cpu_count() or 4)))
        self.m = TTSModel.load_model(language="german") if lang == "de" else TTSModel.load_model()
        self.sample_rate = self.m.sample_rate
        self.info = {"device": f"cpu ({torch.get_num_threads()} threads)"}
        self.state = None
        if voice:
            try:
                self.state = self.m.get_state_for_audio_prompt(voice["wav"])
                self.info["voice"] = "cloned: " + os.path.basename(voice["wav"])
            except Exception as e:  # cloning weights gated, or not this version's API
                self.info["clone_error"] = f"{type(e).__name__}: {e}"[:200]
        if self.state is None:
            self.state = self.m.get_state_for_audio_prompt(BUILTIN[lang])
            self.info["voice"] = "built-in: " + BUILTIN[lang]

    def stream(self, text):
        for chunk in self.m.generate_audio_stream(self.state, text):
            yield mono(chunk)


def make(lang, voice=None):
    return Engine(lang, voice)
