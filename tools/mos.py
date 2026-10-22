"""How natural each clip sounds, as UTMOS predicts it (utmos22_strong, SpeechMOS v1.2.0: a model of
listeners' mean opinion score, 1 to 5, trained on English speech, so German scores are rougher).
Runs anywhere with PyTorch (a GPU helps); not part of the timing on the Jetson.
    python tools/mos.py [results_dir] [voices_dir]
Writes <results_dir>/mos.json: {model: {sentence id: score}} and {"voices": {clip: score}}."""
import json, os, sys

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
VOICES = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "voices")
device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()


def score(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    with torch.no_grad():
        return round(float(predictor(torch.from_numpy(a).unsqueeze(0).to(device), sr)), 3)


out = {}
for model in sorted(os.listdir(RESULTS)):
    d = os.path.join(RESULTS, model)
    if not os.path.isdir(d):
        continue
    clips = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
    if clips:
        out[model] = {f[:-4]: score(os.path.join(d, f)) for f in clips}
        print(model, out[model], flush=True)
if os.path.isdir(VOICES):
    out["voices"] = {f[:-4]: score(os.path.join(VOICES, f)) for f in sorted(os.listdir(VOICES)) if f.endswith(".wav")}
    print("voices", out["voices"])
json.dump(out, open(os.path.join(RESULTS, "mos.json"), "w"), indent=1)
