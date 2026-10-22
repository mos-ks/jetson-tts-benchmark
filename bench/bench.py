#!/usr/bin/env python3
"""One TTS model on this Jetson, measured the same way as every other.

    python3 bench.py --model <name> --out results/ [--langs en,de] [--voices <dir> --design warm]
                     [--llm <ollama model> --name <results folder>]

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

--llm: a local Ollama model (OLLAMA_URL, default http://localhost:11434) keeps generating the whole
time the TTS model loads and speaks, as an assistant's reply streams in while its first sentence
is spoken: on a Jetson both share one GPU. Its tokens per second alone (before the TTS model loads)
and while the TTS model runs go into the result as "llm".
"""
import argparse, importlib, importlib.util, json, os, re, subprocess, sys, threading, time, urllib.request, wave

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


class LLMLoad:
    """A local Ollama model generating, over and over, while something else runs: its tokens per
    second, counted as they stream in (one streamed piece is one token)."""

    PROMPT = "Tell me, in detail and at length, how a small town changes over the four seasons of a year."

    def __init__(self, model, url):
        self.model, self.url = model, url.rstrip("/")
        self.rates = []
        self._stop = threading.Event()
        self._thread = None

    def _generate(self, max_tokens, stop=None):
        body = json.dumps({"model": self.model, "prompt": self.PROMPT, "stream": True, "keep_alive": "10m", "options": {"num_predict": max_tokens}}).encode()
        req = urllib.request.Request(self.url + "/api/generate", data=body, headers={"Content-Type": "application/json"})
        first, n = None, 0
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                piece = json.loads(line)
                if piece.get("response"):
                    n += 1
                    if first is None:
                        first = time.perf_counter()
                if piece.get("done") or (stop is not None and stop.is_set()):
                    break
        took = time.perf_counter() - first if first else 0
        return (n - 1) / took if n > 1 and took > 0 else None

    def alone(self):
        self._generate(1)  # loaded and warm
        return self._generate(200)

    def start(self):
        def loop():
            while not self._stop.is_set():
                try:
                    rate = self._generate(200, self._stop)
                except Exception as e:  # the server busy or away: say so, keep trying
                    print(f"llm: {type(e).__name__}: {e}"[:200], flush=True)
                    time.sleep(1)
                    continue
                if rate:
                    self.rates.append(rate)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=60)


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
    ap.add_argument("--llm", default=None, help="an Ollama model kept generating while the TTS model runs")
    ap.add_argument("--name", default=None, help="the results folder (default: the model's name)")
    args = ap.parse_args()
    sentences = json.load(open(os.path.join(HERE, "sentences.json"), encoding="utf-8"))
    refs = {}
    if args.voices:
        designs = json.load(open(os.path.join(args.voices, "voices.json"), encoding="utf-8"))
        for lang, ref in designs[args.design].items():
            if isinstance(ref, dict) and "wav" in ref:
                refs[lang] = {"wav": os.path.join(args.voices, ref["wav"]), "text": ref["text"]}
    out = os.path.join(args.out, args.name or args.model)
    os.makedirs(out, exist_ok=True)
    llm = None
    if args.llm:
        llm = LLMLoad(args.llm, os.environ.get("OLLAMA_URL", "http://localhost:11434"))
        llm_alone = llm.alone()
        llm.start()
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
        # what it holds once loaded and warm (its peak, while loading, can be well above this)
        held = round(used_mb() - result["base_used_mb"])
        result["langs"][lang] = {"load_s": round(load, 2), "held_mb": held, "sample_rate": engine.sample_rate, "info": getattr(engine, "info", None)}
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
    if llm:
        llm.stop()
        during = sorted(llm.rates)
        result["llm"] = {
            "model": args.llm,
            "tok_s_alone": round(llm_alone, 1) if llm_alone else None,
            "tok_s_during": round(during[len(during) // 2], 1) if during else None,
            "runs_during": len(during),
        }
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
