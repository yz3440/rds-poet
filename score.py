"""Local model inference and the small Strudel vocabulary used on air."""

from dataclasses import dataclass
import json
import random
import re
from urllib.request import Request, urlopen

import numpy as np

from radio.live_rds import validate_title


RATE = 48_000
TOKEN = r"(?:[a-g]#?[2-6]|~)"
SYNTHS = "sine|triangle|square|sawtooth"
NOTES = rf'{TOKEN}(?: {TOKEN}){{0,15}}'
BARE_PATTERN = rf'note\("({NOTES})"\)\.s\("({SYNTHS})"\)'
PATTERN = BARE_PATTERN + r'(?:\.clip\((0\.[1-9][0-9]?|1)\)\.fast\(([1-4])\))?'
# Literal double quotes in a constrained string can make Ollama emit invalid JSON.
# Request valid single-quoted Strudel, then use the saved score's double-quote style.
MODEL_PATTERN = BARE_PATTERN.replace('"', "'")
INSTRUCTIONS = """Compose one line of an ongoing radio poem and its musical answer.
Write a short concrete image (2 to 8 words), in printable ASCII. Listen to the
provided cue. Show a physical detail or action. Vary your opening words.
Keep musical directions in the score.
Echo the pulse or brightness without naming the station. Return line and strudel.
Compose Strudel like note('d3 ~ a3 c4').s('triangle'). Each space divides the
cycle into equal steps. Use 1-16 tokens: lowercase notes a-g in octaves 2-6,
optional sharp, and ~ for a rest. Use the requested synth. Include a sounding
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
        raise ValueError("Expected a supported note pattern, synth and optional clip/fast controls")
    gate = float(match[3] or 1)
    notes = match[1].split() * int(match[4] or 1)
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
        end = start + round((end - start) * gate)
        t = np.arange(end - start) / RATE
        phase = 2 * np.pi * frequency * t
        wave = np.sin(phase)
        if match[2] == "triangle":
            wave = 2 / np.pi * np.arcsin(wave)
        elif match[2] in ("square", "sawtooth"):
            # Limited harmonics leave headroom for the FM audio pre-emphasis.
            stride = 2 if match[2] == "square" else 1
            for harmonic in range(1 + stride, min(15, int(12_000 / frequency)) + 1, stride):
                wave += np.sin(harmonic * phase) / harmonic
            wave *= 0.65 if match[2] == "sawtooth" else 1
        envelope = np.minimum(1, t / 0.008) * np.clip((len(t) / RATE - t) / 0.05, 0, 1)
        audio[start:end] = 0.28 * envelope * wave
    return Phrase(line, code, audio)


def generate(model, theme, cue, history, synth=None):
    pattern = MODEL_PATTERN if synth is None else MODEL_PATTERN.replace(SYNTHS, synth)
    schema = {
        "type": "object",
        "properties": {
            "line": {"type": "string", "minLength": 1, "maxLength": 64},
            "strudel": {"type": "string", "pattern": "^" + pattern + "$"},
        },
        "required": ["line", "strudel"],
        "additionalProperties": False,
    }
    prompt = {"theme": theme, "air": cue, "synth": synth or "choose sine, triangle, sawtooth or square"}
    # A small model tends to recite whole lines when given its own transcript.
    # Occasionally carry one word across the cut while keeping the cue central.
    if history and random.randrange(3) == 0:
        words = re.findall(r"[a-zA-Z]{3,}", history[-1])
        if words:
            prompt["optional_echo"] = words[-1]
    body = {
        "model": model,
        "stream": False,
        "format": schema,
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": json.dumps(prompt)},
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
    if not isinstance(code, str) or not re.fullmatch(pattern, code):
        raise ValueError("Model returned an invalid Strudel expression")
    reply["strudel"] = code.replace("'", '"')
    return reply


# Each voice gives the model a different musical and poetic gesture.
VOICES = {
    "sine": ("Follow a fading sound. Write widely spaced low notes.", (0.85, 0.95), (1,)),
    "triangle": ("Zoom in on a small movement. Write a skipping melody.", (0.35, 0.5), (1, 2)),
    "square": ("Listen for interruptions. Write high taps separated by rests.", (0.15, 0.25), (2, 3)),
    "sawtooth": ("Find what is vibrating. Write a rising or falling bass figure.", (0.55, 0.7), (1, 2)),
}


def opening(line):
    words = re.findall(r"[a-z0-9']+", line.lower())
    return next((word for word in words if word not in
                 {"a", "an", "the", "in", "at", "on", "by", "of", "under", "only"}), "")


def image_words(line):
    common = {"about", "after", "again", "from", "into", "like", "only", "over", "that",
              "their", "them", "then", "there", "this", "through", "under", "what", "when",
              "where", "while", "with", "without", "your"}
    return {word.removesuffix("s") for word in re.findall(r"[a-z]{4,}", line.lower())
            if word not in common}


class Composer:
    def __init__(self, model):
        self.model = model
        self.pocket = None
        if model == "pocket":
            from pocket import PocketPoet
            self.pocket = PocketPoet()
        self.history = []
        self.voices = []
        self.last_voice = None

    def compose(self, theme, cue, seconds):
        if not self.voices:
            self.voices = random.sample(list(VOICES), len(VOICES))
            if self.voices[-1] == self.last_voice:
                self.voices[0], self.voices[-1] = self.voices[-1], self.voices[0]
        synth = self.voices[-1]
        direction, gates, speeds = VOICES[synth]
        prompt = theme + " " + direction
        for attempt in range(3):
            try:
                if self.pocket:
                    reply = self.pocket.reply(prompt, cue, self.history)
                else:
                    reply = generate(self.model, prompt, cue, self.history, synth)
                if not isinstance(reply["line"], str):
                    raise ValueError("Missing poem line")
                line = validate_title(reply["line"].strip())
                subject = opening(line)
                # Let a motif echo once. Catch it again even if it has moved
                # to the middle of a line or changed from singular to plural.
                recent_images = [image_words(old) for old in self.history[-5:]]
                tired_image = any(sum(word in old for old in recent_images) >= 2
                                  for word in image_words(line))
                if (not subject or line.casefold() in (old.casefold() for old in self.history)
                        or subject in (opening(old) for old in self.history[-8:]) or tired_image):
                    prompt = f"{theme} {direction} Move to a new place. Begin with an action."
                    raise ValueError("repeated line or opening image")
                match = re.fullmatch(BARE_PATTERN, reply["strudel"])
                if not match:
                    raise ValueError("unusable note pattern")
                code = (f'note("{match[1]}").s("{synth}")'
                        f'.clip({random.choice(gates)}).fast({random.choice(speeds)})')
                phrase = render({"line": line, "strudel": code}, seconds)
            except (ValueError, TypeError, KeyError) as exc:
                if attempt == 2:
                    raise ValueError(f"No playable phrase after three replies: {exc}") from exc
            else:
                self.history.append(line)
                del self.history[:-32]
                self.last_voice = self.voices.pop()
                return phrase
