#!/usr/bin/env python3
"""One TTS model on this Jetson, measured the same way as every other.

    python3 bench.py --model <name> --out results/ [--langs en,de] [--voices <dir> --design warm]

For each sentence (sentences.json): one untimed warm-up of the first sentence, then each sentence
once, timed: time to the first audio (the first chunk out of a streaming model; the whole clip for
one that does not stream), the time for the whole clip, the clip's length, and so how many times
faster than real time it speaks. Also the model's load time, the most memory the process held
(VmHWM, plus the system's used memory at its highest during the run: the Jetson's GPU memory is
the same RAM), and the board's input power while it speaks (tegrastats VDD_IN, where readable).
Writes results/<model>/<sentence>.wav and results/<model>/result.json.

Adapters live in adapters/<name>.py and define make(lang, voice) -> an object with
    .sample_rate: int
    .stream(text) -> iterator of float32 numpy chunks (mono, -1..1)
    .info (optional): what it ran with (its voice, its device), kept in the result
A model that cannot stream yields one chunk (or one per sentence, as an app would speak it).
`voice` is the reference a voice-cloning model speaks with: {"wav": path, "text": its transcript}
for the language (--voices: a folder with voices.json, as design/design.py writes it), or None.
"""
import argparse, importlib, importlib.util, json, os, re, subprocess, sys, threading, time, wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def used_mb():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0])
    return (info["MemTotal"] - info["MemAvailable"]) / 1024


def hwm_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return None


class Watch:
    """The system's used memory and the board's input power, sampled while something runs."""

    def __init__(self):
        self.peak_used = used_mb()
        self.power = []
        self._stop = threading.Event()
        self._proc = None

    def start(self):
        threading.Thread(target=self._mem, daemon=True).start()
        try:
            self._proc = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._tegra, daemon=True).start()
        except Exception:
            self._proc = None
        return self

    def _mem(self):
        while not self._stop.is_set():
            self.peak_used = max(self.peak_used, used_mb())
            time.sleep(0.05)

    def _tegra(self):
        for line in self._proc.stdout:
            m = re.search(r"VDD_IN (\d+)mW", line)
            if m:
                self.power.append(int(m.group(1)))
            if self._stop.is_set():
                break

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()


def write_wav(path, audio, sr):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def speak(engine, text):
    t0 = time.perf_counter()
    first = None
    chunks = []
    for chunk in engine.stream(text):
        if first is None:
            first = time.perf_counter() - t0
        chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
    total = time.perf_counter() - t0
    audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    return audio, first if first is not None else total, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--langs", default="en,de")
    ap.add_argument("--voices", default=None, help="a folder of designed reference clips (voices.json)")
    ap.add_argument("--design", default="warm", help="which designed voice the cloning models speak with")
    args = ap.parse_args()
    sentences = json.load(open(os.path.join(HERE, "sentences.json"), encoding="utf-8"))
    refs = {}
    if args.voices:
        designs = json.load(open(os.path.join(args.voices, "voices.json"), encoding="utf-8"))
        for lang, ref in designs[args.design].items():
            if isinstance(ref, dict) and "wav" in ref:
                refs[lang] = {"wav": os.path.join(args.voices, ref["wav"]), "text": ref["text"]}
    out = os.path.join(args.out, args.model)
    os.makedirs(out, exist_ok=True)
    # an adapter is named after its model, often its package's own name (pocket_tts, chatterbox):
    # loaded under a name of its own, with the installed packages found first
    sys.path.append(os.path.join(HERE, "adapters"))
    spec = importlib.util.spec_from_file_location(f"adapter_{args.model}", os.path.join(HERE, "adapters", args.model + ".py"))
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    result = {"model": args.model, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "base_used_mb": round(used_mb()), "langs": {}, "lines": []}
    watch = Watch().start()
    for lang in args.langs.split(","):
        lines = sentences.get(lang, [])
        try:
            t0 = time.perf_counter()
            engine = adapter.make(lang, refs.get(lang))
            load = time.perf_counter() - t0
            speak(engine, lines[0]["text"])  # warm-up, untimed
        except Exception as e:  # a model without this language, or one that fails to load
            result["langs"][lang] = {"error": f"{type(e).__name__}: {e}"[:400]}
            continue
        result["langs"][lang] = {"load_s": round(load, 2), "sample_rate": engine.sample_rate, "info": getattr(engine, "info", None)}
        for line in lines:
            try:
                audio, ttfa, total = speak(engine, line["text"])
            except Exception as e:
                result["lines"].append({"id": line["id"], "error": f"{type(e).__name__}: {e}"[:400]})
                continue
            secs = len(audio) / engine.sample_rate
            write_wav(os.path.join(out, line["id"] + ".wav"), audio, engine.sample_rate)
            result["lines"].append({"id": line["id"], "lang": lang, "ttfa_s": round(ttfa, 3), "total_s": round(total, 3), "audio_s": round(secs, 3), "x_realtime": round(secs / total, 2) if total > 0 else None})
            print(f"{args.model} {line['id']}: first audio {ttfa*1000:.0f} ms, {secs:.1f} s of speech in {total:.2f} s ({secs/total:.2f}x real time)", flush=True)
        del engine
        import gc
        gc.collect()
        try:  # a GPU model's memory back before the next language's model loads
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    watch.stop()
    result["peak_used_mb"] = round(watch.peak_used)
    result["process_hwm_mb"] = round(hwm_mb() or 0)
    if watch.power:
        result["power_mw"] = {"mean": round(sum(watch.power) / len(watch.power)), "max": max(watch.power)}
    ok = [l for l in result["lines"] if "x_realtime" in l]
    if ok:
        result["summary"] = {
            "ttfa_ms_median": round(1000 * sorted(l["ttfa_s"] for l in ok)[len(ok) // 2]),
            "x_realtime_median": sorted(l["x_realtime"] for l in ok)[len(ok) // 2],
        }
    json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(json.dumps(result.get("summary", {})), "peak used", result["peak_used_mb"], "MB")


if __name__ == "__main__":
    main()
