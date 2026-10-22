"""Kyutai Pocket TTS's small English model (english_18m_2026-10: 15.6M flow LM with a one-step head
and a 2.6M decoder), on the CPU, built-in voice alba. English only: there is no small German model.
TORCH_THREADS as in pocket_tts.py."""
import importlib.util, os

CONFIG = "english_18m_2026-10"
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("adapter_pocket_tts_cpu", os.path.join(_here, "pocket_tts.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def make(lang, voice=None, device="cpu"):
    if lang != "en":
        raise NotImplementedError("Pocket TTS has no small German model")
    engine = _base.Engine(lang, None, device=device, config=CONFIG)
    engine.info["model"] = CONFIG
    return engine
