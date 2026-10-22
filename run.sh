#!/usr/bin/env bash
# Sets up and measures each model in turn, one at a time (they share the board), each in its own
# environment, every result under results/<model>/:
#   bash run.sh [model ...]          # default: all of them
# VOICES: the designed reference voices (voices.json and the clips); DESIGN: which one (warm).
#
# Safe on a board that serves other work: a model runs only when the memory free now covers what it
# needs (NEED_MB below, measured here; a GPU model takes it from the same 8 GB), else it is skipped
# and says why; and it runs in a systemd scope capped at that need plus HEADROOM_MB, so a model that
# takes more is stopped by the kernel, never the board's other work. (An earlier run without these
# took 7.5 GB with Qwen3-TTS and left the board swapping until it was power-cycled.)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="${BENCH:-$HOME/ttsbench}"
VOICES="${VOICES:-$HERE/voices}"
DESIGN="${DESIGN:-warm}"
HEADROOM_MB="${HEADROOM_MB:-512}"
declare -A NEED_MB=([kokoro_current]=700 [piper_current]=400 [kokoro_v1]=700 [pocket_tts]=1300 [moss_nano]=2500 [qwen3_tts]=4500 [chatterbox]=4000 [omnivoice]=3500)
MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || MODELS=(kokoro_current piper_current kokoro_v1 pocket_tts moss_nano qwen3_tts chatterbox omnivoice)
free_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
for m in "${MODELS[@]}"; do
  echo "=== $m"
  need="${NEED_MB[$m]:-3000}"
  have="$(free_mb)"
  if [ "$have" -lt $((need + HEADROOM_MB)) ]; then
    echo "skipped: needs ~${need} MB, ${have} MB free now"
    mkdir -p "$HERE/results/$m"
    printf '{"model": "%s", "skipped": "needs about %s MB; %s MB were free on this board"}\n' "$m" "$need" "$have" > "$HERE/results/$m/result.json"
    continue
  fi
  if ! bash "$HERE/setup/setup.sh" "$m" > "$HERE/results/setup_$m.log" 2>&1; then
    echo "setup failed: see results/setup_$m.log"; tail -5 "$HERE/results/setup_$m.log"; continue
  fi
  run=(env BENCH="$BENCH" "$BENCH/envs/$m/bin/python" "$HERE/bench/bench.py" --model "$m" --out "$HERE/results" --voices "$VOICES" --design "$DESIGN")
  if systemd-run --user --scope --quiet -p MemoryMax=1G true 2>/dev/null; then
    run=(systemd-run --user --scope --quiet -p "MemoryMax=$((need + HEADROOM_MB))M" -p MemorySwapMax=0 "${run[@]}")
  else
    echo "(no systemd user scope here: running without a memory cap)"
  fi
  "${run[@]}" 2>&1 | tee "$HERE/results/run_$m.log" | grep -v "^\s*$" | tail -12
done
