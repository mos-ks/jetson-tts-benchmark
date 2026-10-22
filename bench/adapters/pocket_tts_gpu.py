"""Kyutai Pocket TTS on the GPU (NVIDIA's CUDA PyTorch for Jetson), streaming: the CPU adapter
(pocket_tts.py) with the model moved to CUDA, as its own command line does with --device cuda."""
import importlib.util, os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("adapter_pocket_tts_cpu", os.path.join(_here, "pocket_tts.py"))
_cpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cpu)


def make(lang, voice=None):
    return _cpu.Engine(lang, voice, device="cuda")
