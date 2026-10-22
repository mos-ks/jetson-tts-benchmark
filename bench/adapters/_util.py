"""What the adapters share: where the models live, and splitting text into sentences."""
import os, re

import numpy as np

BENCH = os.environ.get("BENCH", os.path.expanduser("~/ttsbench"))


def models(*parts):
    return os.path.join(BENCH, "models", *parts)


def sentences(text):
    """Sentence by sentence, as an app speaks a reply: a model that cannot stream starts the
    first sentence's audio while the rest are still to come."""
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def mono(x):
    """A model's output (a tensor or array, any shape) as a 1-D float32 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    a = np.asarray(x, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=0) if a.shape[0] < a.shape[-1] else a.mean(axis=-1)
    return a.reshape(-1)
