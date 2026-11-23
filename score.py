"""Local model inference and the small Strudel vocabulary used on air."""

from dataclasses import dataclass, replace
import json
import random
import re
from urllib.request import Request, urlopen

import numpy as np

from arrangement import arrange, choose_section, edit_motif
from radio.live_rds import validate_title
from sketch import BARE_PATTERN, RATE, STARTER, SYNTHS, mix, parse_layers, stack_code


# Literal double quotes in a constrained string can make Ollama emit invalid JSON.
# Request valid single-quoted Strudel, then use the saved score's double-quote style.
MODEL_PATTERN = BARE_PATTERN.replace('"', "'")
INSTRUCTIONS = """Write a brief, concrete image in printable ASCII and a pitched Strudel edit.
Overheard speech is source material: transform one detail into a physical image in your own words.
Let the station's pulse or brightness shape the music. Use the requested register and rhythm.
Use lowercase note names with octaves, ~ for a rest, and the requested synth.
Return JSON with line and strudel."""


@dataclass
class Phrase:
    line: str
    strudel: str
    audio: np.ndarray
    cue: str | None = None
    edited_layer: int | None = None
    section_name: str | None = None
    answer_layer: int | None = None


def render(reply, seconds):
    """Render a single cycle of the accepted Strudel mini-notation at 48 kHz."""
    if not isinstance(reply, dict) or not isinstance(reply.get("line"), str):
        raise ValueError("Missing poem line")
    line = validate_title(reply["line"].strip())
    code = reply.get("strudel")
    audio = mix(parse_layers(code), seconds)
    cue = reply.get("cue")
    return Phrase(line, code, audio, cue if isinstance(cue, str) else None)


def generate(model, theme, cue, history, synth=None, *, layers=None, target=None, step=0, section=None):
    pattern = MODEL_PATTERN if synth is None else MODEL_PATTERN.replace(SYNTHS, synth)
    if section is not None:
        minimum, maximum = section.steps
        pattern = pattern.replace("{0,15}", f"{{{minimum - 1},{maximum - 1}}}")
        pattern = pattern.replace("[2-6]", f"[{section.octave}-{section.octave + 1}]")
    elif layers is not None:
        limit = min(3, len(layers[target].notes.split()))
        pattern = pattern.replace("{0,15}", f"{{0,{limit - 1}}}")
        octaves = [int(n[-1]) for n in layers[target].notes.split() if n != "~"]
        if octaves:
            pattern = pattern.replace("[2-6]", f"[{min(octaves)}-{max(octaves)}]")
    schema = {
        "type": "object",
        "properties": {
            "line": {"type": "string", "minLength": 1, "maxLength": 64},
            "strudel": {"type": "string", "pattern": "^" + pattern + "$"},
        },
        "required": ["line", "strudel"],
        "additionalProperties": False,
    }
    acoustic, _, speech = cue.partition("\nWhisper transcript: ")
    prompt = {"theme": theme, "air": acoustic, "synth": synth or "choose sine, triangle, sawtooth or square"}
    if speech:
        prompt["overheard_speech"] = speech
    if layers is not None:
        prompt.update(current_sketch=[layer.code() for layer in layers], edit_layer=target + 1)
        if section is None:
            prompt.update(start_step=step + 1, task=f"Write a new 3-6 word image. Return up to {limit} replacement notes/rests from this step. Stay in the layer's register.")
        else:
            prompt["task"] = (f"Write a new 3-6 word image and a complete {minimum}-{maximum} step motif "
                              f"for {section.name}, octaves {section.octave}-{section.octave + 1}. "
                              "Echo one pitch from the existing sketch. Include sounding notes.")
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
    def __init__(self, model, sketch=None, edit_size="bold"):
        if edit_size not in ("bold", "subtle"):
            raise ValueError("Choose bold or subtle sketch edits")
        self.edit_size = edit_size
        self.last_section = None
        self.used_sections = set()
        self.model = model
        self.sketch = sketch
        self.source_layers = parse_layers((sketch or STARTER).read_text(encoding="utf-8"))
        self.layers = self.source_layers
        self.cursors = [0] * len(self.layers)
        self.edited = []
        self.last_layer = None
        self.pocket = None
        if model == "pocket":
            from pocket import PocketPoet
            self.pocket = PocketPoet()
        self.history = []

    def refresh_sketch(self):
        """Merge hand-edited parts into the model's current arrangement."""
        if self.sketch is None:
            return set()
        try:
            new = parse_layers(self.sketch.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"Cannot read sketch: {exc}") from exc
        if len(new) != len(self.layers):
            self.layers = self.source_layers = new
            self.cursors = [0] * len(new)
            self.edited.clear()
            self.last_layer = None
            return set()
        changed = {i for i, layer in enumerate(new) if self.source_layers[i] != layer}
        for i in changed:
            self.cursors[i] = 0
        self.layers = tuple(new[i] if i in changed else v for i, v in enumerate(self.layers))
        self.source_layers = new
        return changed

    def next_layer(self, cue, hand_edits):
        cue = cue.partition("\n")[0]
        last = len(self.layers) - 1
        if "pulsing" in cue:
            preferred = min(1, last)
        elif "bright and thin" in cue or "mostly radio hiss" in cue:
            preferred = last
        else:
            preferred = 0
        eligible = [i for i, layer in enumerate(self.layers) if self.edit_size == "subtle" or layer.audible]
        available = [i for i in eligible if i not in self.edited]
        if not available or (all(i in hand_edits for i in available) and any(i not in hand_edits for i in eligible)):
            self.edited.clear()
            available = eligible
        available = [i for i in available if i not in hand_edits] or available
        if len(available) > 1 and self.last_layer in available:
            available.remove(self.last_layer)
        return preferred if preferred in available else available[0]

    def compose(self, theme, cue, seconds):
        hand_edits = self.refresh_sketch()
        target = self.next_layer(cue, hand_edits)
        layer = self.layers[target]
        section = choose_section(cue, self.last_section, self.used_sections) if self.edit_size == "bold" else None
        step = self.cursors[target]
        prompt = theme
        for attempt in range(3):
            try:
                if self.pocket:
                    reply = self.pocket.reply(prompt, cue, self.history)
                else:
                    reply = generate(self.model, prompt, cue, self.history, section.synth if section else layer.synth,
                                     layers=self.layers, target=target, step=step, section=section)
                if not isinstance(reply["line"], str):
                    raise ValueError("Missing poem line")
                line = validate_title(reply["line"].strip())
                _, _, speech = cue.partition("\nWhisper transcript: ")
                words = re.findall(r"[a-z0-9']+", line.lower())
                heard = re.findall(r"[a-z0-9']+", speech.lower())
                if len(words) >= 4 and f" {' '.join(words)} " in f" {' '.join(heard)} ":
                    prompt = theme + " Take one detail from the speaker and place it somewhere new. Use your own words."
                    raise ValueError("Poem line quotes the transcript verbatim")
                subject = opening(line)
                # Let a motif echo once. Catch it again even if it has moved
                # to the middle of a line or changed from singular to plural.
                recent_images = [image_words(old) for old in self.history[-5:]]
                tired_image = any(sum(word in old for old in recent_images) >= 2
                                  for word in image_words(line))
                if (not subject or line.casefold() in (old.casefold() for old in self.history)
                        or subject in (opening(old) for old in self.history[-8:]) or tired_image):
                    prompt = f"{theme} Move to a new place. Begin with an action."
                    raise ValueError("repeated line or opening image")
                match = re.fullmatch(BARE_PATTERN, reply["strudel"])
                if not match:
                    raise ValueError("Invalid layer edit")
                fragment = match[1].split()
                answer = None
                if section:
                    layers, answer = arrange(self.layers, target, hand_edits, fragment, section)
                else:
                    prompt = theme + " Change at least one note or rest in the selected layer."
                    updated, next_step = edit_motif(layer, step, fragment)
                    layers = list(self.layers)
                    layers[target] = replace(layer, notes=updated)
                phrase = Phrase(line, stack_code(layers), mix(layers, seconds), cue, target,
                                section_name=section.name if section else None, answer_layer=answer)
            except (ValueError, TypeError, KeyError) as exc:
                if attempt == 2:
                    raise ValueError(f"No playable phrase after three replies: {exc}") from exc
            else:
                self.history.append(line)
                del self.history[:-32]
                self.layers = tuple(layers)
                if section:
                    self.last_section = section
                    self.used_sections.add(section)
                    if len(self.used_sections) == 4:
                        self.used_sections.clear()
                else:
                    self.cursors[target] = next_step
                self.last_layer = target
                self.edited.append(target)
                if len(self.edited) == len(self.layers):
                    self.edited.clear()
                return phrase
