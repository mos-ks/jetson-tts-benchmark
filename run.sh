#!/usr/bin/env bash
# Sets up and measures each model in turn, one at a time (they share the board), each in its own
# environment, every result under results/<model>/:
#   bash run.sh [model ...]          # default: all of them
# VOICES: the designed reference voices (voices.json and the clips); DESIGN: which one (warm).
# LLM: an Ollama model (OLLAMA_URL) kept generating during each run, as when an assistant speaks
# while its reply streams in; results then go under results/<model>_llm/.
# TAG: results under results/<model>_<TAG>/ instead (a variant, e.g. TORCH_THREADS=2 TAG=t2).
#
# Safe on a board that serves other work:
# - a model runs only when the memory free now covers what it needs (NEED_MB below; a GPU model
#   takes it from the same 8 GB), else it is skipped and says why;
# - it runs in a systemd scope capped at that need plus HEADROOM_MB, so a model that takes more CPU
#   memory is stopped by the kernel, and with the kernel's out-of-memory killer pointed at it first;
# - and a watchdog stops it the moment the board's free memory falls below FLOOR_MB, whatever took
#   it: on a Jetson the GPU's memory is the same RAM, and the CUDA allocations are not charged to
#   the scope, so the cap alone cannot hold a GPU model.
# (An earlier run without these took 7.5 GB with Qwen3-TTS and left the board swapping until it was
# power-cycled.)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="${BENCH:-$HOME/ttsbench}"
VOICES="${VOICES:-$HERE/voices}"
DESIGN="${DESIGN:-warm}"
HEADROOM_MB="${HEADROOM_MB:-512}"
FLOOR_MB="${FLOOR_MB:-800}"
LLM="${LLM:-}"
TAG="${TAG:-${LLM:+llm}}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
declare -A NEED_MB=(
  [kokoro_current]=700 [piper_current]=400 [kokoro_v1]=700 [pocket_tts]=1300 [moss_nano]=2500
  [kokoro_current_gpu]=1500 [piper_gpu]=1200 [kokoro_v1_gpu]=1500 [pocket_tts_gpu]=1300 [moss_nano_gpu]=3000
  [pocket_tts_small]=600 [pocket_tts_small_gpu]=1100
  [qwen3_tts]=2500 [chatterbox]=3600 [omnivoice]=3000
)
MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || MODELS=(kokoro_current piper_current kokoro_v1 pocket_tts moss_nano kokoro_current_gpu piper_gpu kokoro_v1_gpu pocket_tts_gpu moss_nano_gpu omnivoice qwen3_tts chatterbox)
free_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
note() {  # a model that did not run to the end, and why
  mkdir -p "$HERE/results/$1"
  printf '{"model": "%s", "skipped": "%s"}\n' "$1" "$2" > "$HERE/results/$1/result.json"
}
# kill the run (its own process group) when the board's free memory falls below FLOOR_MB; says so
watchdog() {
  local pid="$1" model="$2" low
  while kill -0 "$pid" 2>/dev/null; do
    low="$(free_mb)"
    if [ "$low" -lt "$FLOOR_MB" ]; then
      kill -KILL -- "-$pid" 2>/dev/null
      echo "$low" > "$HERE/results/.stopped_$model"
      return
    fi
    sleep 0.2
  done
}
capped=0
systemd-run --user --scope --quiet -p MemoryMax=1G true 2>/dev/null && capped=1
[ "$capped" = 1 ] || echo "(no systemd user scope here: the watchdog alone guards the memory)"
for m in "${MODELS[@]}"; do
  name="$m"; extra=()
  [ -z "$TAG" ] || { name="${m}_${TAG}"; extra=(--name "$name"); }
  if [ -n "$LLM" ]; then
    extra+=(--llm "$LLM")
    # the LLM in memory first, so the room left is what the TTS model really gets
    curl -sf -o /dev/null "$OLLAMA_URL/api/generate" -d "{\"model\": \"$LLM\", \"prompt\": \"\", \"keep_alive\": \"10m\"}" || { echo "=== $name: $LLM did not load"; continue; }
  fi
  echo "=== $name"
  need="${NEED_MB[$m]:-3000}"
  have="$(free_mb)"
  if [ "$have" -lt $((need + HEADROOM_MB)) ]; then
    echo "skipped: needs ~${need} MB, ${have} MB free now"
    note "$name" "needs about $need MB; $have MB were free on this board"
    continue
  fi
  if ! bash "$HERE/setup/setup.sh" "$m" > "$HERE/results/setup_$m.log" 2>&1; then
    echo "setup failed: see results/setup_$m.log"; tail -5 "$HERE/results/setup_$m.log"; continue
  fi
  run=(env BENCH="$BENCH" "$BENCH/envs/$m/bin/python" "$HERE/bench/bench.py" --model "$m" --out "$HERE/results" --voices "$VOICES" --design "$DESIGN" "${extra[@]}")
  [ "$capped" = 1 ] && run=(systemd-run --user --scope --quiet -p "MemoryMax=$((need + HEADROOM_MB))M" -p MemorySwapMax=0 "${run[@]}")
  rm -f "$HERE/results/.stopped_$name"
  # its own process group (setsid), the first the out-of-memory killer takes (oom_score_adj 1000)
  setsid bash -c 'echo 1000 > /proc/self/oom_score_adj; exec "$@"' _ "${run[@]}" > "$HERE/results/run_$name.log" 2>&1 &
  pid=$!
  watchdog "$pid" "$name" &
  dog=$!
  wait "$pid"
  kill "$dog" 2>/dev/null; wait "$dog" 2>/dev/null
  if [ -s "$HERE/results/.stopped_$name" ]; then
    low="$(cat "$HERE/results/.stopped_$name")"; rm -f "$HERE/results/.stopped_$name"
    echo "stopped: the board's free memory fell to $low MB"
    note "$name" "stopped when the board's free memory fell to $low MB (it started with $have MB free)"
  fi
  grep -v "^\s*$" "$HERE/results/run_$name.log" | tail -12
done
