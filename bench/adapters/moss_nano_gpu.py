"""MOSS-TTS-Nano-100M on the GPU: its own ONNX runtime with onnxruntime-gpu's CUDA execution
provider (the CPU adapter, moss_nano.py, with MOSS_EP=cuda), cloning the designed reference.

With ONNX Runtime's CUDA defaults it ran an 8 GB board out of memory (its arena grows in powers of
two, and cuDNN takes its largest workspace), so its sessions get the lean settings: the arena grows
by what is asked, convolution algorithms are picked by heuristic, cuDNN's workspace is not maxed."""
import importlib.util, os, sys

os.environ["MOSS_EP"] = "cuda"
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("adapter_moss_nano_cpu", os.path.join(_here, "moss_nano.py"))
_cpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cpu)

LEAN_CUDA = ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": "0"})


def make(lang, voice=None):
    sys.path.insert(0, _cpu.SRC)
    import ort_cpu_runtime as rt

    if not getattr(rt, "_lean", False):
        resolve = rt._resolve_ort_providers

        def lean(execution_provider):
            providers = resolve(execution_provider)
            return [LEAN_CUDA if p == "CUDAExecutionProvider" else p for p in providers]

        rt._resolve_ort_providers, rt._lean = lean, True
    engine = _cpu.make(lang, voice)
    engine.info["device"] = "gpu (onnxruntime CUDA, lean arena)"
    return engine
