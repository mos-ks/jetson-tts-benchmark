# Jetson TTS Benchmark

How fast, how heavy and how natural open text-to-speech models are on a **Jetson Orin Nano Super (8 GB)**, measured the way a voice assistant would use them. Every model says the same six lines (three English, three German); every model that can clone a voice speaks with the same designed reference voice, so the voices compare like for like.

**Listen to every clip and see the speed-to-realism plot: <https://mos-ks.github.io/jetson-tts-benchmark/>**

It was made to pick a self-hosted voice for a personal assistant that runs on a Jetson, in place of Kokoro-82M v0.19 (English) and Piper (German): a voice of its own, fast enough to answer without a pause, on the same board.

## Results

<!-- results -->
| Model | Runs on | Language | First audio | Speed | Realism (UTMOS) | Extra memory | Power | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| Kokoro-82M v0.19 | CPU, onnxruntime | English | 1,742 ms | 1.08x | 4.07 | +568 MB | 7.7 W | measured |
| Piper (ramona-low, amy-medium) | CPU, onnxruntime | English | 181 ms | 12.11x | 4.22 | +180 MB | 8.1 W | measured |
| Piper (ramona-low, amy-medium) | CPU, onnxruntime | German | 129 ms | 15.41x | 2.46 | +180 MB | 8.1 W | measured |
| Kokoro-82M v1.0 | CPU, onnxruntime | English | 1,266 ms | 1.43x | 4.44 | +435 MB | 9.7 W | measured |
| Kyutai Pocket TTS | CPU, PyTorch | English | 408 ms | 0.51x | 4.44 | +922 MB | 7.0 W | measured |
| Kyutai Pocket TTS | CPU, PyTorch | German | 386 ms | 0.51x | 3.34 | +922 MB | 7.0 W | measured |
| MOSS-TTS-Nano-100M | CPU, onnxruntime |  |  |  |  |  |  | queued |
| Qwen3-TTS 0.6B Base | GPU, CUDA graphs |  |  |  |  |  |  | queued |
| Chatterbox Turbo / Multilingual | GPU, PyTorch |  |  |  |  |  |  | queued |
| OmniVoice | GPU, fp16 |  |  |  |  |  |  | queued |

_4 of 8 measured, 2026-10-22 17:01._
<!-- /results -->

**First audio**: the first streamed chunk (for a model that does not stream: the first sentence). **Speed**: seconds of speech per second of synthesis (above 1x it keeps up with itself). **Realism**: UTMOS (utmos22_strong), a predictor of listeners' mean opinion score from 1 to 5, trained on English, so German scores are rougher; listen to the clips. **Extra memory**: the board's used memory at its highest during the run, over what it was before (CPU and GPU share the 8 GB). **Power**: the board's input power (VDD_IN, tegrastats) while it speaks. Medians over the three lines per language.

## The board

Jetson Orin Nano Engineering Reference Developer Kit Super, 8 GB (7.4 GiB usable), JetPack 6.2 (L4T R36.4.3), CUDA 12.6, Python 3.10, 6 Cortex-A78AE cores, default power mode. It was serving other work during the runs (a web backend and an idle Ollama), as it would in use; about 3.9 GiB was free.

## Run it yourself

```bash
git clone https://github.com/mos-ks/jetson-tts-benchmark && cd jetson-tts-benchmark
bash run.sh                       # every model, one at a time
bash run.sh kokoro_v1 pocket_tts  # or just these
```

`setup/setup.sh <model>` makes each model its own virtual environment under `~/ttsbench/envs/` (GPU models take PyTorch 2.8 from NVIDIA's Jetson AI Lab index, `jp6/cu126`) and fetches its files; `bench/bench.py` measures it and writes `results/<model>/result.json` and a WAV per line. Then, on any machine with PyTorch (a GPU helps):

```bash
python tools/mos.py          # UTMOS for every clip -> results/mos.json
python tools/build_page.py   # docs/index.html and this README's table (needs ffmpeg)
```

Adding a model: an adapter in `bench/adapters/<name>.py` with `make(lang, voice)` returning an object with `sample_rate` and `stream(text)` (an iterator of float32 chunks), and its install in `setup/setup.sh`.

## The reference voices

`voices/` holds three voices designed from a text description with Qwen3-TTS 1.7B VoiceDesign (`design/design.py`, run once on a desktop GPU), each in English and in German (the German clip clones the English one with Qwen3-TTS 1.7B Base). They are no real person's voice. The cloning models speak with `warm`.

## Licences

The scripts are MIT. The models keep their own licences (listed in the table on the page); the clips here are their output, shared to compare them, and some of those licences are non-commercial.
