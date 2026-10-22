# Jetson TTS Benchmark

How fast, how heavy and how natural open text-to-speech models are on a **Jetson Orin Nano Super (8 GB)**, measured the way a voice assistant would use them. Every model says the same six lines (three English, three German); every model that can clone a voice speaks with the same designed reference voice, so the voices compare like for like.

**Listen to every clip and see the speed-to-realism plot: <https://mos-ks.github.io/jetson-tts-benchmark/>**

It was made to pick a self-hosted voice for a personal assistant that runs on a Jetson, in place of Kokoro-82M v0.19 (English) and Piper (German): a voice of its own, fast enough to answer without a pause, on the same board.

## Results

<!-- results -->
| Model | Runs on | Language | First audio | Speed | Realism (UTMOS) | Extra memory | Power | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| Kokoro-82M v0.19 | CPU, onnxruntime | English | 1,742 ms | 1.08x | 4.07 | +568 MB | 7.7 W | measured |
| Kokoro-82M v0.19 | GPU, onnxruntime CUDA | English | 441 ms | 4.05x | 4.10 | +2,108 MB | 6.9 W | measured |
| Piper (ramona-low, amy-medium) | CPU, onnxruntime | English | 181 ms | 12.11x | 4.22 | +180 MB | 8.1 W | measured |
| Piper (ramona-low, amy-medium) | CPU, onnxruntime | German | 129 ms | 15.41x | 2.46 | +180 MB | 8.1 W | measured |
| Piper (ramona-low, amy-medium) | GPU, onnxruntime CUDA | English | 392 ms | 4.64x | 4.48 | +469 MB | 6.1 W | measured |
| Piper (ramona-low, amy-medium) | GPU, onnxruntime CUDA | German | 389 ms | 5.66x | 2.48 | +469 MB | 6.1 W | measured |
| Kokoro-82M v1.0 | CPU, onnxruntime | English | 1,266 ms | 1.43x | 4.44 | +435 MB | 9.7 W | measured |
| Kokoro-82M v1.0 | GPU, onnxruntime CUDA | English | 340 ms | 5.83x | 4.44 | +697 MB | 9.1 W | measured |
| Kyutai Pocket TTS | CPU, PyTorch | English | 408 ms | 0.51x | 4.44 | +922 MB | 7.0 W | measured |
| Kyutai Pocket TTS | CPU, PyTorch | German | 386 ms | 0.51x | 3.34 | +922 MB | 7.0 W | measured |
| Kyutai Pocket TTS | GPU, PyTorch | English | 194 ms | 1.53x | 4.42 | +1,109 MB | 7.8 W | measured |
| Kyutai Pocket TTS | GPU, PyTorch | German | 193 ms | 1.53x | 3.28 | +1,109 MB | 7.8 W | measured |
| Kyutai Pocket TTS small (english_18m) | CPU, PyTorch | English | 258 ms | 1.05x | 4.42 | +330 MB | 6.5 W | measured |
| Kyutai Pocket TTS small (english_18m) | GPU, PyTorch | English | 224 ms | 1.36x | 4.36 | +841 MB | 7.1 W | measured |
| MOSS-TTS-Nano-100M | CPU, onnxruntime | English | 1,941 ms | 0.50x | 3.79 | +1,572 MB | 9.5 W | measured |
| MOSS-TTS-Nano-100M | CPU, onnxruntime | German | 1,492 ms | 0.51x | 3.10 | +1,572 MB | 9.5 W | measured |
| MOSS-TTS-Nano-100M | GPU, onnxruntime CUDA |  |  |  |  |  |  | too big: stopped when the board's free memory fell to 789 MB (it started with 4269 MB free) |
| Qwen3-TTS 0.6B Base | GPU, CUDA graphs |  |  |  |  |  |  | too big: stopped when the board's free memory fell to 738 MB (it started with 4319 MB free) |
| Chatterbox Turbo / Multilingual | GPU, PyTorch |  |  |  |  |  |  | too big: stopped when the board's free memory fell to 781 MB (it started with 4364 MB free) |
| OmniVoice | GPU, fp16 |  |  |  |  |  |  | too big: stopped when the board's free memory fell to 736 MB (it started with 5158 MB free) |

_11 of 15 measured, 2026-10-22 21:00._
<!-- /results -->

**First audio**: the first streamed chunk (for a model that does not stream: the first sentence). **Speed**: seconds of speech per second of synthesis (above 1x it keeps up with itself). **Realism**: UTMOS (utmos22_strong), a predictor of listeners' mean opinion score from 1 to 5, trained on English, so German scores are rougher; listen to the clips. **Extra memory**: the board's used memory at its highest during the run, over what it was before (CPU and GPU share the 8 GB). **Power**: the board's input power (VDD_IN, tegrastats) while it speaks. Medians over the three lines per language.

## Speaking while the LLM writes

An assistant speaks its first sentence while the rest of its reply is still being written, and on a Jetson the TTS model and the LLM share one GPU (`LLM=qwen2.5:3b bash run.sh <model>`).

<!-- llm -->
While qwen2.5:3b (Ollama) keeps generating on the same GPU; each cell alone -> together. The LLM's rate comes from one to four 200-token answers per run, so read it as rough.

| Model | Runs on | Language | First audio | Speed | LLM tokens/s |
|---|---|---|---:|---:|---:|
| Kokoro-82M v0.19 | CPU, onnxruntime | English | 1,742 ms -> 1,993 ms | 1.08x -> 0.88x | 20.7 -> 17.9 |
| Piper (ramona-low, amy-medium) | CPU, onnxruntime | English | 181 ms -> 229 ms | 12.11x -> 10.02x | 21.2 -> 23.8 |
| Piper (ramona-low, amy-medium) | CPU, onnxruntime | German | 129 ms -> 194 ms | 15.41x -> 11.74x | 21.2 -> 23.8 |
| Kokoro-82M v1.0 | GPU, onnxruntime CUDA | English | 340 ms -> 544 ms | 5.83x -> 3.32x | 15.9 -> 22.1 |
| Kyutai Pocket TTS | GPU, PyTorch | English | 194 ms -> 235 ms | 1.53x -> 0.91x | 21.2 -> 16.9 |
| Kyutai Pocket TTS | GPU, PyTorch | German | 193 ms -> 224 ms | 1.53x -> 0.88x | 21.2 -> 16.9 |
| Kyutai Pocket TTS small (english_18m) | CPU, PyTorch | English | 258 ms -> 226 ms | 1.05x -> 1.09x | 20.9 -> 21.7 |
| Kyutai Pocket TTS small (english_18m) | GPU, PyTorch | English | 224 ms -> 377 ms | 1.36x -> 0.65x | 21.2 -> 17.2 |
<!-- /llm -->

## The board

Jetson Orin Nano Engineering Reference Developer Kit Super, 8 GB (7.4 GiB usable), JetPack 6.2 (L4T R36.4.3), CUDA 12.6, Python 3.10, 6 Cortex-A78AE cores, power mode MAXN SUPER. It was serving other work during the runs (a web backend, and Ollama with its model unloaded unless a run says otherwise), as it would in use; 3.9 to 5 GiB was free.

## Run it yourself

```bash
git clone https://github.com/mos-ks/jetson-tts-benchmark && cd jetson-tts-benchmark
bash run.sh                       # every model, one at a time
bash run.sh kokoro_v1 pocket_tts  # or just these
```

**On a board that serves other work**, `run.sh` runs a model only when the memory free covers what it needs (`NEED_MB`), inside a systemd scope capped at that, with the out-of-memory killer pointed at it first, and with a watchdog that stops it the moment the board's free memory falls below 800 MB (`FLOOR_MB`). The watchdog is the one that counts for GPU models: on a Jetson the GPU's memory is the same RAM, and CUDA's allocations are not charged to the scope. (A first run without these pushed an 8 GB board running a 3B chat model into swap with Qwen3-TTS, at 7.5 GB, until it was power-cycled.)

The small models run twice, on the CPU and on the GPU (`<model>_gpu`: onnxruntime-gpu's CUDA execution provider, or PyTorch on CUDA).

`setup/setup.sh <model>` makes each model its own virtual environment under `~/ttsbench/envs/` (GPU runs take PyTorch 2.8 or onnxruntime-gpu 1.24 from NVIDIA's Jetson AI Lab index, `jp6/cu126`) and fetches its files; `bench/bench.py` measures it and writes `results/<model>/result.json` and a WAV per line. Then, on any machine with PyTorch (a GPU helps):

```bash
python tools/mos.py          # UTMOS for every clip -> results/mos.json
python tools/build_page.py   # docs/index.html and this README's table (needs ffmpeg)
```

Adding a model: an adapter in `bench/adapters/<name>.py` with `make(lang, voice)` returning an object with `sample_rate` and `stream(text)` (an iterator of float32 chunks), and its install in `setup/setup.sh`.

## The reference voices

`voices/` holds three voices designed from a text description with Qwen3-TTS 1.7B VoiceDesign (`design/design.py`, run once on a desktop GPU), each in English and in German (the German clip clones the English one with Qwen3-TTS 1.7B Base). They are no real person's voice. The cloning models speak with `warm`.

## Licences

The scripts are MIT. The models keep their own licences (listed in the table on the page); the clips here are their output, shared to compare them, and some of those licences are non-commercial.
