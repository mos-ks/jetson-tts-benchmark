"""Kokoro v0.19 on the GPU: the CPU adapter (kokoro_current.py) with onnxruntime-gpu's CUDA
execution provider, chosen through kokoro-onnx's own ONNX_PROVIDER setting."""
import importlib.util, os

os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("adapter_kokoro_current_cpu", os.path.join(_here, "kokoro_current.py"))
_cpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cpu)


def make(lang, voice=None):
    engine = _cpu.make(lang, voice)
    provider = engine.k.sess.get_providers()[0]
    if provider != "CUDAExecutionProvider":
        raise RuntimeError(f"Kokoro's session runs on {provider}, not the GPU")
    engine.info = dict(getattr(engine, "info", None) or {}, device="gpu (onnxruntime CUDA)")
    return engine
