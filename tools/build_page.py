"""The listening page (docs/index.html, with its audio in docs/audio/) and the README's results table,
from results/<model>/result.json, results/mos.json and the designed voices. Run it again whenever a
result comes in.
    python tools/build_page.py
Needs ffmpeg on PATH (the clips go out as MP3)."""
import html, json, math, os, re, shutil, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
VOICES = os.path.join(ROOT, "voices")
DOCS = os.path.join(ROOT, "docs")
AUDIO = os.path.join(DOCS, "audio")
REPO = "https://github.com/mos-ks/jetson-tts-benchmark"

# Every model in the run, each small one's GPU run after its CPU run; the page shows the ones not
# yet measured as queued.
MODELS = [
    ("kokoro_current", "Kokoro-82M v0.19", "82M", "CPU, onnxruntime", "Apache-2.0", "a preset voice", "today's English voice (baseline)"),
    ("kokoro_current_gpu", "Kokoro-82M v0.19", "82M", "GPU, onnxruntime CUDA", "Apache-2.0", "a preset voice", "the baseline on the GPU"),
    ("piper_current", "Piper (ramona-low, amy-medium)", "15-60M", "CPU, onnxruntime", "MIT (voices vary)", "a preset voice", "today's German voice (baseline)"),
    ("piper_gpu", "Piper (ramona-low, amy-medium)", "15-60M", "GPU, onnxruntime CUDA", "MIT (voices vary)", "a preset voice", "the baseline on the GPU"),
    ("kokoro_v1", "Kokoro-82M v1.0", "82M", "CPU, onnxruntime", "Apache-2.0", "a preset voice", "English only"),
    ("kokoro_v1_gpu", "Kokoro-82M v1.0", "82M", "GPU, onnxruntime CUDA", "Apache-2.0", "a preset voice", "English only"),
    ("pocket_tts", "Kyutai Pocket TTS", "100M", "CPU, PyTorch", "CC-BY-4.0", "clones the reference", "streams"),
    ("pocket_tts_gpu", "Kyutai Pocket TTS", "100M", "GPU, PyTorch", "CC-BY-4.0", "clones the reference", "streams"),
    ("pocket_tts_small", "Kyutai Pocket TTS small (english_18m)", "18M", "CPU, PyTorch", "CC-BY-4.0", "a preset voice", "English only, streams"),
    ("pocket_tts_small_gpu", "Kyutai Pocket TTS small (english_18m)", "18M", "GPU, PyTorch", "CC-BY-4.0", "a preset voice", "English only, streams"),
    ("moss_nano", "MOSS-TTS-Nano-100M", "100M", "CPU, onnxruntime", "Apache-2.0", "clones the reference", "streams, 48 kHz"),
    ("moss_nano_gpu", "MOSS-TTS-Nano-100M", "100M", "GPU, onnxruntime CUDA", "Apache-2.0", "clones the reference", "streams, 48 kHz"),
    ("qwen3_tts", "Qwen3-TTS 0.6B Base", "0.6B", "GPU, CUDA graphs", "Apache-2.0", "clones the reference", "streams (faster-qwen3-tts)"),
    ("chatterbox", "Chatterbox Turbo / Multilingual", "350M / 500M", "GPU, PyTorch", "MIT", "clones the reference", "sentence by sentence"),
    ("omnivoice", "OmniVoice", "0.6B", "GPU, fp16", "Apache-2.0", "clones the reference", "sentence by sentence"),
]
DESIGNS = ["warm", "bright", "soft"]


def load(path, default=None):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def mp3(src, dst):
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-b:a", "96k", dst], check=True)


def esc(s):
    return html.escape(str(s), quote=True)


sentences = load(os.path.join(ROOT, "bench", "sentences.json"), {})
text_of = {s["id"]: s["text"] for lang in ("en", "de") for s in sentences.get(lang, [])}
mos = load(os.path.join(RESULTS, "mos.json"), {})
designs = load(os.path.join(VOICES, "voices.json"), {})
os.makedirs(AUDIO, exist_ok=True)

rows, points, players = [], [], []
measured = 0
for key, name, size, runs, lic, voice, note in MODELS:
    r = load(os.path.join(RESULTS, key, "result.json"))
    if not r:
        rows.append({"key": key, "name": name, "size": size, "runs": runs, "lang": "", "status": "queued", "note": note, "lic": lic, "voice": voice})
        continue
    if r.get("skipped"):  # it did not fit in the memory free on the board (skipped, or stopped by the watchdog)
        rows.append({"key": key, "name": name, "size": size, "runs": runs, "lang": "", "status": "too big", "note": r["skipped"], "lic": lic, "voice": voice})
        continue
    measured += 1
    clips = []
    for lang in ("en", "de"):
        info = r.get("langs", {}).get(lang)
        lines = [l for l in r.get("lines", []) if l.get("lang") == lang and "x_realtime" in l]
        if not info or "error" in info or not lines:
            err = (info or {}).get("error", "") if info else ""
            if "no German" in err or "NotImplemented" in err:
                continue  # a model without that language: no row
            rows.append({"key": key, "name": name, "size": size, "runs": runs, "lang": lang, "status": "failed", "note": err[:140] or "no output", "lic": lic, "voice": voice})
            continue
        m = [mos.get(key, {}).get(l["id"]) for l in lines]
        row = {
            "key": key, "name": name, "size": size, "runs": runs, "lang": lang, "status": "measured", "note": note, "lic": lic,
            "voice": (info.get("info") or {}).get("voice") or voice,
            "ttfa": median([l["ttfa_s"] * 1000 for l in lines]),
            "speed": median([l["x_realtime"] for l in lines]),
            "mos": median(m),
            "extra_mb": (r.get("peak_used_mb", 0) - r.get("base_used_mb", 0)) if r.get("peak_used_mb") else None,
            "watts": (r.get("power_mw") or {}).get("mean", 0) / 1000 if r.get("power_mw") else None,
            "load": info.get("load_s"),
        }
        rows.append(row)
        if row["speed"] and row["mos"]:
            points.append(row)
        for l in lines:
            src = os.path.join(RESULTS, key, l["id"] + ".wav")
            if os.path.exists(src):
                dst = f"{key}_{l['id']}.mp3"
                mp3(src, os.path.join(AUDIO, dst))
                clips.append((l["id"], dst, mos.get(key, {}).get(l["id"]), l))
    if clips:
        players.append((key, name, runs, clips))

ref_scores = []
design_cards = []
for d in DESIGNS:
    if d not in designs:
        continue
    items = []
    for lang in ("en", "de"):
        ref = designs[d].get(lang)
        if not ref:
            continue
        src = os.path.join(VOICES, ref["wav"])
        dst = f"voice_{d}_{lang}.mp3"
        mp3(src, os.path.join(AUDIO, dst))
        s = mos.get("voices", {}).get(ref["wav"][:-4])
        if s and lang == "en":
            ref_scores.append(s)
        items.append((lang, dst, ref["text"], s))
    design_cards.append((d, designs[d]["instruct"], items))
ref_mos = median(ref_scores)


# ── speaking while an LLM writes (results/<model>_llm, run.sh with LLM=...) ────
def lang_stats(r, lang):
    lines = [l for l in (r or {}).get("lines", []) if l.get("lang") == lang and "x_realtime" in l]
    if not lines:
        return None, None
    return median([l["ttfa_s"] * 1000 for l in lines]), median([l["x_realtime"] for l in lines])


llm_rows = []
for key, name, size, runs, lic, voice, note in MODELS:
    rl = load(os.path.join(RESULTS, key + "_llm", "result.json"))
    if not rl:
        continue
    if rl.get("skipped"):
        llm_rows.append({"name": name, "runs": runs, "lang": "", "note": rl["skipped"]})
        continue
    r0 = load(os.path.join(RESULTS, key, "result.json"))
    llm = rl.get("llm") or {}
    for lang in ("en", "de"):
        ttfa1, speed1 = lang_stats(rl, lang)
        if speed1 is None:
            continue
        ttfa0, speed0 = lang_stats(r0, lang)
        llm_rows.append({"name": name, "runs": runs, "lang": lang, "ttfa0": ttfa0, "ttfa1": ttfa1, "speed0": speed0, "speed1": speed1,
                         "tok0": llm.get("tok_s_alone"), "tok1": llm.get("tok_s_during"), "llm": llm.get("model")})
llm_model = next((r["llm"] for r in llm_rows if r.get("llm")), None)


# ── the speed to realism plot (SVG, drawn to one scale) ──────────────────────
def plot():
    W, H, L, R, T, B = 760, 380, 58, 20, 18, 52
    xmin, xmax = math.log10(0.25), math.log10(32)
    ymin, ymax = 1.5, 4.5
    X = lambda v: L + (math.log10(v) - xmin) / (xmax - xmin) * (W - L - R)
    Y = lambda v: T + (ymax - v) / (ymax - ymin) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Speed against realism for each model and language" class="plot">']
    # slower than speech
    out.append(f'<rect x="{L}" y="{T}" width="{X(1) - L:.1f}" height="{H - T - B}" class="slow"/>')
    out.append(f'<text x="{L + 8}" y="{H - B - 10}" class="zone">slower than real time</text>')
    for v in (0.25, 0.5, 1, 2, 4, 8, 16, 32):
        x = X(v)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{H - B}" class="{"edge" if v == 1 else "grid"}"/>')
        out.append(f'<text x="{x:.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{v:g}&#215;</text>')
    for v in (1.5, 2, 2.5, 3, 3.5, 4, 4.5):
        y = Y(v)
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{v:g}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 8}" class="axis" text-anchor="middle">speed: seconds of speech per second of work (log scale)</text>')
    out.append(f'<text transform="translate(14 {(T + H - B) / 2}) rotate(-90)" class="axis" text-anchor="middle">realism (UTMOS, 1-5)</text>')
    if ref_mos:
        y = Y(ref_mos)
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" class="ref"/>')
        out.append(f'<text x="{W - R - 6}" y="{y - 6:.1f}" class="reflabel" text-anchor="end">the designed reference voice ({ref_mos:.2f})</text>')
    placed = []
    for p in sorted(points, key=lambda p: (p["speed"], p["mos"])):
        x, y = X(min(max(p["speed"], 0.26), 31)), Y(min(max(p["mos"], ymin + 0.02), ymax - 0.02))
        cls = "en" if p["lang"] == "en" else "de"
        if p["lang"] == "en":
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" class="pt {cls}"><title>{esc(p["name"])} ({p["lang"]}): {p["speed"]:.2f}x, UTMOS {p["mos"]:.2f}</title></circle>')
        else:
            out.append(f'<rect x="{x - 5.5:.1f}" y="{y - 5.5:.1f}" width="11" height="11" class="pt {cls}"><title>{esc(p["name"])} ({p["lang"]}): {p["speed"]:.2f}x, UTMOS {p["mos"]:.2f}</title></rect>')
        label = f'{p["name"].split(" (")[0]}{" GPU" if p["key"].endswith("_gpu") else ""} {p["lang"].upper()}'
        ly = y - 10
        for (px, py) in placed:  # keep labels apart
            if abs(px - x) < 120 and abs(py - ly) < 14:
                ly = py + 16
        anchor, lx = ("end", x - 10) if x > W - 190 else ("start", x + 10)
        placed.append((x, ly))
        out.append(f'<text x="{lx:.1f}" y="{ly:.1f}" class="plabel" text-anchor="{anchor}">{esc(label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def fmt(v, f, suffix=""):
    return "&#8211;" if v is None else (f.format(v) + suffix)


table = []
for r in rows:
    st = r["status"]
    pill = {"measured": "ok", "queued": "wait", "too big": "wait", "failed": "bad"}[st]
    lang = {"en": "English", "de": "German", "": ""}[r["lang"]]
    table.append(
        "<tr>"
        f'<th scope="row">{esc(r["name"])}<span class="sub">{esc(r["size"])} &#183; {esc(r["lic"])}</span></th>'
        f"<td>{esc(r['runs'])}</td><td>{lang}</td>"
        f'<td class="num">{fmt(r.get("ttfa"), "{:,.0f}", " ms")}</td>'
        f'<td class="num">{fmt(r.get("speed"), "{:.2f}", "&#215;")}</td>'
        f'<td class="num">{fmt(r.get("mos"), "{:.2f}")}</td>'
        f'<td class="num">{fmt(r.get("extra_mb"), "{:+,.0f}", " MB")}</td>'
        f'<td class="num">{fmt(r.get("watts"), "{:.1f}", " W")}</td>'
        f"<td>{esc(r.get('voice') or '')}</td>"
        f'<td><span class="pill {pill}">{st}</span>{"<span class=sub>" + esc(r["note"]) + "</span>" if r.get("note") and st != "measured" else ""}</td>'
        "</tr>"
    )

def pair(a, b, f, suffix=""):
    return f'{fmt(a, f, suffix)} &#8594; <b>{fmt(b, f, suffix)}</b>'


llm_table = []
for r in llm_rows:
    lang = {"en": "English", "de": "German", "": ""}[r["lang"]]
    if "note" in r:
        llm_table.append(f'<tr><th scope="row">{esc(r["name"])}</th><td>{esc(r["runs"])}</td><td></td><td colspan="3"><span class="pill wait">too big</span><span class="sub">{esc(r["note"])}</span></td></tr>')
        continue
    llm_table.append(
        f'<tr><th scope="row">{esc(r["name"])}</th><td>{esc(r["runs"])}</td><td>{lang}</td>'
        f'<td class="num">{pair(r["ttfa0"], r["ttfa1"], "{:,.0f}", " ms")}</td>'
        f'<td class="num">{pair(r["speed0"], r["speed1"], "{:.2f}", "&#215;")}</td>'
        f'<td class="num">{pair(r["tok0"], r["tok1"], "{:.1f}")}</td></tr>'
    )
llm_section = "" if not llm_table else f"""
  <section class="block" aria-labelledby="h-llm">
    <h2 id="h-llm">Speaking while the LLM writes</h2>
    <p>An assistant speaks its first sentence while the rest of its reply is still being written, and on a Jetson the TTS model and the LLM share one GPU. Here {esc(llm_model or "a local LLM")} (Ollama) kept generating the whole time each TTS model loaded and spoke. Each cell: alone &#8594; <b>together</b>. The LLM's rate comes from one to four 200-token answers per run, so read it as rough (a few tokens per second either way); the TTS columns are medians over three lines.</p>
    <div class="scroll panel"><table>
      <thead><tr><th scope="col">Model</th><th scope="col">Runs on</th><th scope="col">Language</th><th scope="col" class="num">First audio</th><th scope="col" class="num">Speed</th><th scope="col" class="num">LLM tokens/s</th></tr></thead>
      <tbody>{''.join(llm_table)}</tbody>
    </table></div>
  </section>
"""

listen = []
for key, name, runs, clips in players:
    items = []
    for sid, dst, s, l in clips:
        items.append(
            f'<li><p class="said" lang="{l["lang"]}">{esc(text_of.get(sid, sid))}</p>'
            f'<audio controls preload="none" src="audio/{dst}"></audio>'
            f'<p class="meta">first audio {l["ttfa_s"] * 1000:,.0f} ms &#183; {l["x_realtime"]:.2f}&#215; real time{" &#183; UTMOS %.2f" % s if s else ""}</p></li>'
        )
    listen.append(f'<section class="model" id="{key}"><h3>{esc(name)}</h3><p class="runs">{esc(runs)}</p><ul class="clips">{"".join(items)}</ul></section>')

cards = []
for d, instruct, items in design_cards:
    li = "".join(
        f'<li><p class="said" lang="{lang}">{esc(text)}</p><audio controls preload="none" src="audio/{dst}"></audio>'
        f'<p class="meta">{"English" if lang == "en" else "German (cloned from the English clip)"}{" &#183; UTMOS %.2f" % s if s else ""}</p></li>'
        for lang, dst, text, s in items
    )
    cards.append(f'<section class="voice"><h3>{esc(d.capitalize())}</h3><p class="brief">&#8220;{esc(instruct)}&#8221;</p><ul class="clips">{li}</ul></section>')

total = len(MODELS)
stamp = time.strftime("%Y-%m-%d %H:%M")
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Jetson TTS Benchmark</title>
<meta name="description" content="Open text-to-speech models on an 8 GB Jetson Orin Nano Super: first-audio latency, speed, memory, power and predicted naturalness, with every clip to listen to.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Atkinson+Hyperlegible:wght@400;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* One column that reads top to bottom: the answer (plot), the numbers (table), then the evidence (every clip). */
:root {{
  --bg: #f5f4f8; --panel: #ffffff; --fg: #1c1a24; --muted: #66617a; --line: #dddae6;
  --accent: #6a4dc4; --olive: #6f7c22; --slow: #f1e9f7; --good: #2c7a4b; --warn: #a8661f; --bad: #b23a3a;
  --display: "Bricolage Grotesque", "Segoe UI", system-ui, sans-serif;
  --body: "Atkinson Hyperlegible", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "Cascadia Mono", monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #14121b; --panel: #1c1a25; --fg: #ece9f5; --muted: #a39cb6; --line: #2f2b3c;
  --accent: #a891ff; --olive: #c2cf6c; --slow: #251d33; --good: #6fcf97; --warn: #e3a35a; --bad: #ef7d7d; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #14121b; --panel: #1c1a25; --fg: #ece9f5; --muted: #a39cb6; --line: #2f2b3c;
  --accent: #a891ff; --olive: #c2cf6c; --slow: #251d33; --good: #6fcf97; --warn: #e3a35a; --bad: #ef7d7d; color-scheme: dark; }}
html {{ -webkit-text-size-adjust: 100%; }}
body {{ margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.55 var(--body); }}
.wrap {{ max-width: 1040px; margin: 0 auto; padding-block: 40px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 40px; }}
h1, h2, h3 {{ font-family: var(--display); text-wrap: balance; margin: 0; line-height: 1.15; }}
h1 {{ font-size: clamp(30px, 5vw, 46px); font-weight: 700; letter-spacing: -0.01em; }}
h2 {{ font-size: 24px; font-weight: 700; }}
h3 {{ font-size: 18px; font-weight: 700; }}
p {{ margin: 0; max-width: 68ch; }}
.lede {{ display: grid; gap: 12px; }}
.lede p {{ color: var(--muted); font-size: 18px; }}
.status {{ font-family: var(--mono); font-size: 13px; color: var(--muted); letter-spacing: 0.02em; }}
.status b {{ color: var(--accent); }}
section.block {{ display: grid; gap: 14px; min-width: 0; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; min-width: 0; }}
.plot {{ width: 100%; height: auto; display: block; }}
.plot .grid {{ stroke: var(--line); stroke-width: 1; }}
.plot .edge {{ stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 4 3; }}
.plot .slow {{ fill: var(--slow); }}
.plot .zone, .plot .tick, .plot .axis {{ fill: var(--muted); font: 12px var(--mono); }}
.plot .axis {{ font-size: 12px; }}
.plot .ref {{ stroke: var(--accent); stroke-width: 1.5; stroke-dasharray: 2 4; }}
.plot .reflabel {{ fill: var(--accent); font: 12px var(--body); }}
.plot .pt.en {{ fill: var(--accent); }}
.plot .pt.de {{ fill: var(--olive); }}
.plot .plabel {{ fill: var(--fg); font: 600 12px var(--body); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 18px; font-size: 14px; color: var(--muted); }}
.legend i {{ display: inline-block; width: 11px; height: 11px; margin-right: 6px; vertical-align: -1px; }}
.legend .en {{ background: var(--accent); border-radius: 50%; }}
.legend .de {{ background: var(--olive); }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; min-width: 860px; font-size: 14px; }}
th, td {{ text-align: left; padding: 10px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
thead th {{ font: 600 12px var(--mono); color: var(--muted); letter-spacing: 0.04em; text-transform: uppercase; }}
tbody th {{ font-weight: 700; }}
.num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }}
.sub {{ display: block; font-weight: 400; color: var(--muted); font-size: 12px; margin-top: 2px; }}
.pill {{ display: inline-block; font: 600 11px var(--mono); padding: 2px 8px; border-radius: 99px; border: 1px solid currentColor; text-transform: uppercase; letter-spacing: 0.04em; }}
.pill.ok {{ color: var(--good); }} .pill.wait {{ color: var(--warn); }} .pill.bad {{ color: var(--bad); }}
.grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
.model, .voice {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; display: grid; gap: 10px; min-width: 0; }}
.runs, .brief {{ color: var(--muted); font-size: 14px; }}
.clips {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 14px; }}
.said {{ font-size: 15px; }}
audio {{ width: 100%; height: 36px; }}
.meta {{ font: 12px var(--mono); color: var(--muted); }}
.notes {{ display: grid; gap: 10px; color: var(--muted); font-size: 15px; }}
.notes b {{ color: var(--fg); }}
a {{ color: var(--accent); }}
a:focus-visible, audio:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
</style>
</head>
<body>
<div class="wrap">
  <header class="lede">
    <h1>Jetson TTS Benchmark</h1>
    <p>Open text-to-speech models on an 8 GB Jetson Orin Nano Super, measured the way a voice assistant would use them: how soon the first audio comes, how many times faster than real time it speaks, how much memory and power it takes, and how natural it sounds. Every model says the same six lines, three in English and three in German; the ones that clone a voice all speak with the same designed reference voice.</p>
    <p class="status"><b>{measured} of {total}</b> models measured &#183; updated {stamp} &#183; <a href="{REPO}">scripts and raw results</a></p>
  </header>

  <section class="block" aria-labelledby="h-plot">
    <h2 id="h-plot">Speed against realism</h2>
    <div class="panel">{plot()}</div>
    <div class="legend"><span><i class="en"></i>English</span><span><i class="de"></i>German</span><span>Right of the dashed line it speaks faster than real time. Realism is UTMOS, a predictor of listeners' opinion scores (1 to 5), trained on English: German scores are rougher.</span></div>
  </section>

  <section class="block" aria-labelledby="h-table">
    <h2 id="h-table">Results</h2>
    <div class="scroll panel"><table>
      <thead><tr><th scope="col">Model</th><th scope="col">Runs on</th><th scope="col">Language</th><th scope="col" class="num">First audio</th><th scope="col" class="num">Speed</th><th scope="col" class="num">Realism</th><th scope="col" class="num">Extra memory</th><th scope="col" class="num">Power</th><th scope="col">Voice</th><th scope="col">Status</th></tr></thead>
      <tbody>{''.join(table)}</tbody>
    </table></div>
    <p class="status">Medians over the three lines per language, after one untimed warm-up. First audio: the first streamed chunk (for a model that does not stream: the first sentence). Speed: seconds of speech per second of synthesis. Extra memory: the board's used memory at its highest during the run, over what it was before (CPU and GPU share the 8 GB). Power: the board's input (VDD_IN) while it speaks.</p>
  </section>

{llm_section}
  <section class="block" aria-labelledby="h-listen">
    <h2 id="h-listen">Listen</h2>
    <div class="grid2">{''.join(listen) or '<p>No clips yet.</p>'}</div>
  </section>

  <section class="block" aria-labelledby="h-voices">
    <h2 id="h-voices">The designed reference voices</h2>
    <p>Designed from a description with Qwen3-TTS 1.7B VoiceDesign (once, on a desktop GPU), then the same voice in German by cloning the English clip with Qwen3-TTS 1.7B Base. No real person's voice. The cloning models in this benchmark speak with <b>Warm</b>.</p>
    <div class="grid2">{''.join(cards)}</div>
  </section>

  <section class="block notes" aria-labelledby="h-how">
    <h2 id="h-how">How it was measured</h2>
    <p><b>Board.</b> Jetson Orin Nano Engineering Reference Developer Kit Super, 8 GB (7.4 GiB usable), JetPack 6.2 (L4T R36.4.3), CUDA 12.6, Python 3.10, 6 CPU cores, power mode MAXN SUPER. The board was serving other work at the time (a web backend and an idle Ollama), as it would in use.</p>
    <p><b>Each model</b> in its own virtual environment (GPU runs with PyTorch 2.8 or onnxruntime-gpu 1.24 from NVIDIA's Jetson AI Lab index), loaded once per language, warmed up once, then each line spoken once and timed. One run at a time, and stopped if the board's free memory falls below 800 MB.</p>
    <p><b>Threads.</b> The CPU runs use all 6 cores. Pocket TTS was also run with 1, 2 and 4 threads: the 100M model stays at 0.51 to 0.56&#215; at any count, and the small English model is fastest on one thread (1.23&#215;, against 1.05&#215; on six).</p>
    <p><b>Too big.</b> A run is stopped when the board's free memory falls below 800 MB. OmniVoice, Qwen3-TTS 0.6B, Chatterbox and MOSS-TTS-Nano on the GPU (also with ONNX Runtime's lean arena settings) each reached that floor while loading or warming up, starting from 4.3 to 5.2 GB free: on an 8 GB Orin Nano they need the board nearly to themselves.</p>
    <p><b>Realism</b> is predicted, not a listening test: UTMOS (utmos22_strong) on every clip. It rewards clean, neutral read-aloud speech, so a plain preset voice can outscore a livelier one, and it says nothing about whether a voice sounds like anyone in particular. Listen to the clips to judge for yourself.</p>
  </section>
</div>
</body>
</html>
"""
open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8").write(page)

# The README's table, between its markers
md = ["| Model | Runs on | Language | First audio | Speed | Realism (UTMOS) | Extra memory | Power | Status |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
for r in rows:
    md.append(
        f"| {r['name']} | {r['runs']} | {dict(en='English', de='German').get(r['lang'], '')} | "
        f"{'' if r.get('ttfa') is None else f'{r['ttfa']:,.0f} ms'} | {'' if r.get('speed') is None else f'{r['speed']:.2f}x'} | "
        f"{'' if r.get('mos') is None else f'{r['mos']:.2f}'} | {'' if r.get('extra_mb') is None else f'{r['extra_mb']:+,.0f} MB'} | "
        f"{'' if r.get('watts') is None else f'{r['watts']:.1f} W'} | {r['status']}{': ' + r['note'] if r['status'] in ('too big', 'failed') and r.get('note') else ''} |"
    )
readme = os.path.join(ROOT, "README.md")
if os.path.exists(readme):
    s = open(readme, encoding="utf-8").read()
    s = re.sub(r"<!-- results -->.*<!-- /results -->", "<!-- results -->\n" + "\n".join(md) + f"\n\n_{measured} of {total} measured, {stamp}._\n<!-- /results -->", s, flags=re.S)
    if llm_rows:
        md2 = [f"While {llm_model or 'a local LLM'} (Ollama) keeps generating on the same GPU; each cell alone -> together. "
               "The LLM's rate comes from one to four 200-token answers per run, so read it as rough.", "",
               "| Model | Runs on | Language | First audio | Speed | LLM tokens/s |", "|---|---|---|---:|---:|---:|"]
        for r in llm_rows:
            if "note" in r:
                md2.append(f"| {r['name']} | {r['runs']} | | too big: {r['note']} | | |")
                continue
            f = lambda v, spec, suf="": "–" if v is None else spec.format(v) + suf
            md2.append(f"| {r['name']} | {r['runs']} | {dict(en='English', de='German')[r['lang']]} | {f(r['ttfa0'], '{:,.0f}', ' ms')} -> {f(r['ttfa1'], '{:,.0f}', ' ms')} | "
                       f"{f(r['speed0'], '{:.2f}', 'x')} -> {f(r['speed1'], '{:.2f}', 'x')} | {f(r['tok0'], '{:.1f}')} -> {f(r['tok1'], '{:.1f}')} |")
        s = re.sub(r"<!-- llm -->.*<!-- /llm -->", lambda _: "<!-- llm -->\n" + "\n".join(md2) + "\n<!-- /llm -->", s, flags=re.S)
    open(readme, "w", encoding="utf-8").write(s)
print(f"page: {measured} of {total} measured, {len(points)} points, {sum(len(c) for *_, c in players)} clips")
