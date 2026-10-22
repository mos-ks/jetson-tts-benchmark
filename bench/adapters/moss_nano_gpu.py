"""MOSS-TTS-Nano-100M on the GPU: its own ONNX runtime with onnxruntime-gpu's CUDA execution
provider (the CPU adapter, moss_nano.py, with MOSS_EP=cuda), cloning the designed reference."""
import importlib.util, os, sys

os.environ["MOSS_EP"] = "cuda"
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("adapter_moss_nano_cpu", os.path.join(_here, "moss_nano.py"))
_cpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cpu)


def make(lang, voice=None):
    return _cpu.make(lang, voice)
