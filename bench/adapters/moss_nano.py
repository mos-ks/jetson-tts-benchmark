"""MOSS-TTS-Nano-100M through its own ONNX runtime (onnx_tts_runtime.py, no PyTorch model), cloning
the designed reference, streaming: its codec decodes frames as they are generated, and each decoded
piece is handed out as it comes (its runtime keeps them to the end otherwise). 48 kHz stereo, mixed
to mono. MOSS_EP: cpu (default) or cuda (onnxruntime-gpu); MOSS_THREADS: CPU threads. Its
WeTextProcessing normaliser (pynini) is left off: its own light normaliser still runs."""
import os, queue, sys, threading

from _util import models, mono

SRC = models("moss_nano", "src")


class Engine:
    def __init__(self, lang, voice):
        if not voice:
            raise ValueError("MOSS-TTS-Nano clones a voice here: run with --voices")
        sys.path.insert(0, SRC)
        from onnx_tts_runtime import OnnxTtsRuntime

        ep = os.environ.get("MOSS_EP", "cpu")
        threads = int(os.environ.get("MOSS_THREADS", os.cpu_count() or 4))
        os.makedirs(models("moss_nano", "out"), exist_ok=True)
        self.rt = OnnxTtsRuntime(thread_count=threads, execution_provider=ep, output_dir=models("moss_nano", "out"))
        self.sample_rate = int(self.rt.codec_meta["codec_config"]["sample_rate"])
        self.voice = voice
        self.info = {"device": f"{ep} (onnxruntime, {threads} threads)", "voice": "cloned: " + os.path.basename(voice["wav"])}
        self.q = None
        session = self.rt.codec_streaming_session
        decode = session.run_frames

        def run_frames(frames):
            out = decode(frames)
            if out is not None and self.q is not None:
                audio, n = out
                if n > 0:
                    self.q.put(audio[0, :, :n].mean(axis=0))
            return out

        session.run_frames = run_frames

    def stream(self, text):
        self.q = queue.Queue()
        done, failed = object(), []

        def work():
            try:
                self.rt.synthesize(
                    text=text,
                    prompt_audio_path=self.voice["wav"],
                    streaming=True,
                    enable_wetext=False,
                    enable_normalize_tts_text=True,
                    output_audio_path=models("moss_nano", "out", "last.wav"),
                )
            except Exception as e:
                failed.append(e)
            finally:
                self.q.put(done)

        t = threading.Thread(target=work, daemon=True)
        t.start()
        while True:
            item = self.q.get()
            if item is done:
                break
            yield mono(item)
        t.join()
        self.q = None
        if failed:
            raise failed[0]


def make(lang, voice=None):
    return Engine(lang, voice)
