"""Read, edit and synthesize the pitched Strudel subset used by the FM performer."""

from dataclasses import dataclass
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import numpy as np


RATE = 48_000
STARTER = Path(__file__).resolve().parent / "examples" / "airwaves.strudel"
TOKEN = r"(?:[a-g]#?[2-6]|~)"
SYNTHS = "sine|triangle|square|sawtooth"
NOTES = rf"{TOKEN}(?: {TOKEN}){{0,15}}"
BARE_PATTERN = rf'note\("({NOTES})"\)\.s\("({SYNTHS})"\)'
FRACTION = r"(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
PATTERN = (BARE_PATTERN + rf'(?:\.clip\(({FRACTION})\))?'
           + r'(?:\.fast\(([1-4])\))?' + rf'(?:\.gain\(({FRACTION})\))?')


@dataclass(frozen=True)
class Layer:
    notes: str
    synth: str
    clip: float
    fast: int
    gain: float

    def code(self):
        return (f'note("{self.notes}").s("{self.synth}")'
                f'.clip({self.clip}).fast({self.fast}).gain({self.gain})')


def parse_layers(source):
    """Parse a single note expression or a flat stack of up to eight layers."""
    if not isinstance(source, str):
        raise ValueError("Missing Strudel sketch")
    code = re.sub(r"//[^\n]*", "", source).strip()
    code = re.sub(r"^setcps\(\s*\d+(?:\.\d+)?\s*\)\s*;?\s*", "", code)
    code = code.removesuffix(";").strip()
    stack = re.fullmatch(r"stack\s*\((.*)\)", code, re.DOTALL)
    parts = stack[1].strip().removesuffix(",").split(",") if stack else [code]
    if len(parts) > 8:
        raise ValueError("A sketch supports up to eight layers")
    layers = []
    for part in parts:
        # Normalise quoted note sequences and whitespace between method calls.
        part = re.sub(r"\s*\.\s*(?=[a-z])", ".", part.strip().replace("'", '"'))
        match = re.fullmatch(PATTERN, part)
        if not match:
            raise ValueError(f"Unsupported Strudel layer: {part[:120]}. See examples/airwaves.strudel")
        layer = Layer(match[1], match[2], float(match[3] or 1), int(match[4] or 1), float(match[5] or 1))
        if not (0 <= layer.clip <= 1 and 0 <= layer.gain <= 1):
            raise ValueError("Layer clip and gain must be between 0 and 1")
        layers.append(layer)
    if not any(layer.notes.replace("~", "").strip() and layer.gain and layer.clip for layer in layers):
        raise ValueError("The sketch needs at least one sounding note")
    return tuple(layers)


def stack_code(layers):
    if len(layers) == 1:
        return layers[0].code()
    return "stack(\n  " + ",\n  ".join(layer.code() for layer in layers) + "\n)"


def save_sketch(path, code, seconds):
    source = f"setcps({1 / seconds})\n{stack_code(parse_layers(code))}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(snapshot.name)
    try:
        with snapshot:
            snapshot.write(source)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def render_layer(layer, seconds):
    notes = layer.notes.split() * layer.fast
    gate = layer.clip
    audio = np.zeros(round(RATE * seconds), np.float32)
    pitches = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
    for step, note in enumerate(notes):
        if note == "~":
            continue
        midi = 12 * (int(note[-1]) + 1) + pitches[note[0]] + ("#" in note)
        frequency = 440 * 2 ** ((midi - 69) / 12)
        start = step * len(audio) // len(notes)
        end = (step + 1) * len(audio) // len(notes)
        end = start + round((end - start) * gate)
        t = np.arange(end - start) / RATE
        phase = 2 * np.pi * frequency * t
        wave = np.sin(phase)
        if layer.synth == "triangle":
            wave = 2 / np.pi * np.arcsin(wave)
        elif layer.synth in ("square", "sawtooth"):
            # Limited harmonics leave headroom for the FM audio pre-emphasis.
            stride = 2 if layer.synth == "square" else 1
            for harmonic in range(1 + stride, min(15, int(12_000 / frequency)) + 1, stride):
                wave += np.sin(harmonic * phase) / harmonic
            wave *= 0.65 if layer.synth == "sawtooth" else 1
        envelope = np.minimum(1, t / 0.008) * np.clip((len(t) / RATE - t) / 0.05, 0, 1)
        audio[start:end] = 0.28 * layer.gain * envelope * wave
    return audio


def mix(layers, seconds):
    audio = render_layer(layers[0], seconds)
    for layer in layers[1:]:
        audio += render_layer(layer, seconds)
    # Custom sketches may set every gain to maximum. Scale the bus only when needed.
    peak = float(np.max(np.abs(audio)))
    if peak > 0.9:
        audio *= 0.9 / peak
    return audio
