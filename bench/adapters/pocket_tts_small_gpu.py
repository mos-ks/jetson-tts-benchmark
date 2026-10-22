"""Kyutai Pocket TTS's small English model (english_18m_2026-10) on the GPU: pocket_tts_small.py on
CUDA. English only."""
import importlib.util, os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("adapter_pocket_tts_small", os.path.join(_here, "pocket_tts_small.py"))
_small = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_small)


def make(lang, voice=None):
    return _small.make(lang, voice, device="cuda")
