"""Local English speech recognition for demodulated FM audio."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
from scipy import signal
from scipy.io import wavfile


SPEECH_SECONDS = 6


class Whisper:
    def __init__(self):
        self.cli = shutil.which("whisper-cli")
        default = Path.home() / "Developer/whisper.cpp/models/ggml-tiny.en.bin"
        self.model = Path(os.environ.get("WHISPER_MODEL", default)).expanduser()
        if not self.cli or not self.model.is_file():
            raise RuntimeError("Speech mode needs whisper-cli on PATH and a local model. "
                               "See the README setup or set WHISPER_MODEL to the model file.")

    def __call__(self, audio):
        mono = signal.resample_poly(audio - audio.mean(), 1, 2)  # 32 kHz to 16 kHz
        with tempfile.TemporaryDirectory(prefix="rds-poet-speech-") as folder:
            path = Path(folder) / "station.wav"
            wavfile.write(path, 16_000, np.rint(32767 * np.clip(mono, -1, 1)).astype("<i2"))
            result = subprocess.run([self.cli, "-m", str(self.model), "-f", str(path),
                                     "-l", "en", "-t", "3", "-np", "-nt"],
                                    capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError("Whisper failed: " + result.stderr.strip()[-500:])
        # Whisper can return nonverbal captions such as [Music] or (applause).
        text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", result.stdout)
        return " ".join(text.split())[:350] if re.search(r"[A-Za-z]", text) else ""


def speech_cue(cue, audio, recognizer):
    if recognizer is None:
        return cue
    if "nearly still" in cue or "mostly radio hiss" in cue:
        return cue + "\nWhisper skipped silence or hiss."
    transcript = recognizer(audio)
    return cue + ("\nWhisper transcript: " + transcript if transcript else "\nWhisper returned no clear words.")
