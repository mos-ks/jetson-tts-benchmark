#!/usr/bin/env bash
# Sets up and measures each model in turn, one at a time (they share the board), each in its own
# environment, every result under results/<model>/:
#   bash run.sh [model ...]          # default: all of them
# VOICES: the designed reference voices (voices.json and the clips); DESIGN: which one (warm).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="${BENCH:-$HOME/ttsbench}"
VOICES="${VOICES:-$HERE/voices}"
DESIGN="${DESIGN:-warm}"
MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || MODELS=(kokoro_current piper_current kokoro_v1 pocket_tts moss_nano qwen3_tts chatterbox omnivoice)
for m in "${MODELS[@]}"; do
  echo "=== $m"
  if ! bash "$HERE/setup/setup.sh" "$m" > "$HERE/results/setup_$m.log" 2>&1; then
    echo "setup failed: see results/setup_$m.log"; tail -5 "$HERE/results/setup_$m.log"; continue
  fi
  BENCH="$BENCH" "$BENCH/envs/$m/bin/python" "$HERE/bench/bench.py" --model "$m" --out "$HERE/results" --voices "$VOICES" --design "$DESIGN" 2>&1 | tee "$HERE/results/run_$m.log" | grep -v "^\s*$" | tail -12
done
