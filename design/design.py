"""Ami's voice, designed from words (Qwen3-TTS 1.7B VoiceDesign), then the same voice in German (the
1.7B Base model cloning the English clip). One clip per design and language: the reference every
voice-cloning model in the benchmark speaks with.
    .venv/Scripts/python.exe design.py [out_dir]
"""
import json, os, sys

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
os.makedirs(OUT, exist_ok=True)

DESIGNS = {
    "warm": "A warm, friendly young woman in her twenties. Soft, bright and gently playful, with a smile in her voice. Calm, encouraging and clear, at a relaxed natural pace. Cozy and caring, never childish, never breathy or whispery.",
    "bright": "A cheerful young woman with a light, clear, medium-high voice and lively, upbeat energy. Cute and playful but natural, with expressive intonation and a quick smile. Crisp, clear diction.",
    "soft": "A gentle, soothing young woman with a soft, slightly low, velvety voice. Patient, grounding and reassuring, unhurried, with warm natural intonation, like a close friend sitting next to you.",
}
REF = {
    "en": "Hey, it's me, Ami. I'm so glad you're here. Let's take today one small step at a time, and I'll keep track of the rest for you.",
    "de": "Hey, ich bin's, Ami. Schön, dass du da bist. Lass uns den Tag Schritt für Schritt angehen, und ich behalte den Rest für dich im Blick.",
}

design = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cuda:0", dtype=torch.bfloat16)
made = {}
for i, (name, instruct) in enumerate(DESIGNS.items()):
    torch.manual_seed(1000 + i)
    wavs, sr = design.generate_voice_design(text=REF["en"], instruct=instruct, language="English")
    path = os.path.join(OUT, f"ami_{name}_en.wav")
    sf.write(path, np.asarray(wavs[0], dtype=np.float32), sr)
    made[name] = {"instruct": instruct, "en": {"wav": os.path.basename(path), "text": REF["en"], "sample_rate": sr}}
    print(f"designed {name}: {len(wavs[0]) / sr:.1f} s", flush=True)
del design
torch.cuda.empty_cache()

base = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16)
for i, name in enumerate(DESIGNS):
    torch.manual_seed(2000 + i)
    en = os.path.join(OUT, made[name]["en"]["wav"])
    wavs, sr = base.generate_voice_clone(text=REF["de"], language="German", ref_audio=en, ref_text=REF["en"])
    path = os.path.join(OUT, f"ami_{name}_de.wav")
    sf.write(path, np.asarray(wavs[0], dtype=np.float32), sr)
    made[name]["de"] = {"wav": os.path.basename(path), "text": REF["de"], "sample_rate": sr}
    print(f"cloned {name} into German: {len(wavs[0]) / sr:.1f} s", flush=True)
json.dump(made, open(os.path.join(OUT, "voices.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("->", OUT)
