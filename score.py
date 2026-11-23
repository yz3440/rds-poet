"""Local model inference and the small Strudel vocabulary used on air."""

from dataclasses import dataclass
import json
import re
from urllib.request import Request, urlopen

import numpy as np

from radio.live_rds import validate_title


RATE = 48_000
TOKEN = r"(?:[a-g]#?[2-6]|~)"
PATTERN = rf'note\("({TOKEN}(?: {TOKEN}){{0,15}})"\)\.s\("(sine|triangle)"\)'
# Literal double quotes in a constrained string can make Ollama emit invalid JSON.
# Request valid single-quoted Strudel, then use the saved score's double-quote style.
MODEL_PATTERN = PATTERN.replace('"', "'")
SCHEMA = {
    "type": "object",
    "properties": {
        "line": {"type": "string", "minLength": 1, "maxLength": 64},
        "strudel": {"type": "string", "pattern": "^" + MODEL_PATTERN + "$"},
    },
    "required": ["line", "strudel"],
    "additionalProperties": False,
}
INSTRUCTIONS = """Compose one line of an ongoing radio poem and its musical answer.
Write a short concrete image (2 to 8 words), in printable ASCII. Listen to the
provided cue. Carry one detail forward from the recent lines, changing what it
means. Avoid repeating lines. Return a JSON object with line and strudel.
Compose Strudel like note('d3 ~ a3 c4').s('triangle'). Each space divides the
cycle into equal steps. Use 1-16 tokens: lowercase notes a-g in octaves 2-6,
optional sharp, and ~ for a rest. Use sine or triangle. Include a sounding
note. Change rhythm, register or timbre in response to the poem and sound."""


@dataclass
class Phrase:
    line: str
    strudel: str
    audio: np.ndarray


def render(reply, seconds):
    """Render a single cycle of the accepted Strudel mini-notation at 48 kHz."""
    if not isinstance(reply, dict) or not isinstance(reply.get("line"), str):
        raise ValueError("Missing poem line")
    line = validate_title(reply["line"].strip())
    code = reply.get("strudel")
    match = re.fullmatch(PATTERN, code) if isinstance(code, str) else None
    if not match:
        raise ValueError("Expected a note pattern with a sine or triangle synth")
    notes = match[1].split()
    if all(note == "~" for note in notes):
        raise ValueError("The pattern needs at least one sounding note")
    audio = np.zeros(round(RATE * seconds), np.float32)
    pitches = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
    for step, note in enumerate(notes):
        if note == "~":
            continue
        midi = 12 * (int(note[-1]) + 1) + pitches[note[0]] + ("#" in note)
        frequency = 440 * 2 ** ((midi - 69) / 12)
        start = step * len(audio) // len(notes)
        end = (step + 1) * len(audio) // len(notes)
        t = np.arange(end - start) / RATE
        wave = np.sin(2 * np.pi * frequency * t)
        if match[2] == "triangle":
            wave = 2 / np.pi * np.arcsin(wave)
        envelope = np.minimum(1, t / 0.008) * np.clip((len(t) / RATE - t) / 0.05, 0, 1)
        audio[start:end] = 0.28 * envelope * wave
    return Phrase(line, code, audio)


def generate(model, theme, cue, history):
    body = {
        "model": model,
        "stream": False,
        "format": SCHEMA,
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": json.dumps({
                "theme": theme, "air": cue, "recent_lines": history,
            })},
        ],
        "options": {"temperature": 0.85, "num_ctx": 2048, "num_predict": 140, "num_thread": 3},
        "keep_alive": "15m",
    }
    request = Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        message = json.load(response)
    if "error" in message:
        raise ValueError(message["error"])
    reply = json.loads(message["message"]["content"])
    code = reply["strudel"]
    if not isinstance(code, str) or not re.fullmatch(MODEL_PATTERN, code):
        raise ValueError("Model returned an invalid Strudel expression")
    reply["strudel"] = code.replace("'", '"')
    return reply
