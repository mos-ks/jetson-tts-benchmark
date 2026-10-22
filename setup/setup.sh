#!/usr/bin/env bash
# One model's own Python environment and files on a Jetson (JetPack 6.x, Python 3.10), apart from
# everything else on the board:
#   bash setup/setup.sh <model>      # kokoro_current | piper_current | kokoro_v1 | pocket_tts |
#                                    # moss_nano | qwen3_tts | chatterbox | omnivoice
# Environments go to $BENCH/envs/<model>, model files to $BENCH/models/<model> (BENCH defaults to
# ~/ttsbench). GPU models take PyTorch from NVIDIA's Jetson AI Lab index (jp6/cu126); the CPU models
# take nothing from it. Hugging Face downloads happen on first use, into the usual cache.
set -euo pipefail
MODEL="${1:?which model?}"
BENCH="${BENCH:-$HOME/ttsbench}"
JETSON_INDEX="${JETSON_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
ENV="$BENCH/envs/$MODEL"
DIR="$BENCH/models/$MODEL"
mkdir -p "$DIR"
[ -x "$ENV/bin/python" ] || python3 -m venv "$ENV"
PIP="$ENV/bin/pip"
"$PIP" install -q -U pip wheel
fetch() { [ -s "$2" ] || curl -fL --retry 3 -o "$2" "$1"; }
# NVIDIA's CUDA build of PyTorch for Jetson, over whatever generic build a package pulled in (PyPI's
# aarch64 torch 2.8.0 is CPU-only and has the same version number, so pip keeps it): run AFTER the
# model's own install, and refuse to go on without CUDA.
jetson_torch() {
  "$PIP" install -q --force-reinstall --no-deps "torch==2.8.0" "torchaudio==2.8.0" --index-url "$JETSON_INDEX"
  "$PIP" install -q "numpy<2"   # that build is compiled against NumPy 1.x
  "$ENV/bin/python" -c "import torch, sys; ok = torch.cuda.is_available(); torch.zeros(2).numpy(); print('[setup] torch', torch.__version__, 'cuda', ok); sys.exit(0 if ok else 3)"
}

case "$MODEL" in
  kokoro_current)
    # Kokoro v0.19 as many existing setups run it: kokoro-onnx 0.3.3 reads that model (with numpy < 2)
    "$PIP" install -q "kokoro-onnx==0.3.3" "numpy<2"
    fetch https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx "$DIR/kokoro-v0_19.onnx"
    fetch https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json "$DIR/voices.json"
    ;;
  piper_current)
    "$PIP" install -q piper-tts "numpy<2"
    DIR="$BENCH/models/piper"; mkdir -p "$DIR"
    for v in de/de_DE/ramona/low/de_DE-ramona-low en/en_US/amy/medium/en_US-amy-medium; do
      n="$(basename "$v")"
      fetch "https://huggingface.co/rhasspy/piper-voices/resolve/main/$v.onnx" "$DIR/$n.onnx"
      fetch "https://huggingface.co/rhasspy/piper-voices/resolve/main/$v.onnx.json" "$DIR/$n.onnx.json"
    done
    ;;
  kokoro_v1)
    "$PIP" install -q "kokoro-onnx>=0.6.1"
    fetch https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx "$DIR/kokoro-v1.0.onnx"
    fetch https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin "$DIR/voices-v1.0.bin"
    ;;
  pocket_tts)
    # CPU only: PyTorch's own aarch64 CPU wheels
    "$PIP" install -q torch --index-url https://download.pytorch.org/whl/cpu
    "$PIP" install -q pocket-tts
    ;;
  moss_nano)
    # its ONNX runtime: onnxruntime on the CPU (CPU torch only to read the reference clip; 2.8,
    # since torchaudio 2.9 reads audio only through torchcodec)
    "$PIP" install -q "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cpu
    "$PIP" install -q numpy sentencepiece soundfile onnxruntime huggingface_hub
    [ -d "$DIR/src" ] || git clone -q --depth 1 https://github.com/OpenMOSS/MOSS-TTS-Nano.git "$DIR/src"
    ;;
  qwen3_tts)
    "$PIP" install -q faster-qwen3-tts
    jetson_torch
    ;;
  chatterbox)
    [ -d "$DIR/src" ] || git clone -q https://github.com/resemble-ai/chatterbox "$DIR/src"
    # it pins torch 2.6; the Jetson index has 2.8 (as jetson-containers does it: unpin)
    sed -i 's/==/>=/g' "$DIR/src/pyproject.toml"
    "$PIP" install -q -e "$DIR/src"
    jetson_torch
    ;;
  omnivoice)
    "$PIP" install -q omnivoice
    jetson_torch
    ;;
  *) echo "unknown model: $MODEL" >&2; exit 2 ;;
esac
echo "[setup] $MODEL ready: $ENV"
