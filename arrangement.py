"""Rearrange a lead and an answering voice while carrying the sketch forward."""

from dataclasses import dataclass

from sketch import Layer


@dataclass(frozen=True)
class Section:
    name: str
    synth: str
    octave: int
    steps: tuple[int, int]
    clip: float
    fast: int


LONG_BASS = Section("sustained bass", "sine", 2, (2, 3), 0.94, 1)
HIGH_BELLS = Section("spaced bells", "triangle", 4, (3, 4), 0.48, 1)
RUNNING_BASS = Section("running bass", "sawtooth", 2, (5, 8), 0.60, 3)
HIGH_TAPS = Section("syncopated taps", "square", 4, (6, 8), 0.20, 4)


def choose_section(cue, last, used_sections):
    acoustic = cue.partition("\n")[0].lower()
    bright = "bright" in acoustic or "hiss" in acoustic
    pulsing = "pulsing" in acoustic
    busy = last.fast == 1 if last else pulsing
    if busy:
        first, other = (HIGH_TAPS, RUNNING_BASS) if bright or pulsing else (RUNNING_BASS, HIGH_TAPS)
    else:
        first, other = (HIGH_BELLS, LONG_BASS) if bright else (LONG_BASS, HIGH_BELLS)
    return other if first in used_sections else first


def in_register(notes, octave):
    """Move a figure as a unit, within two octaves of the chosen register."""
    bottom = min(int(note[-1]) for note in notes if note != "~")
    return ["~" if note == "~" else note[:-1] + str(octave + min(1, int(note[-1]) - bottom))
            for note in notes]


def arrange(layers, target, hand_edits, fragment, section):
    pitches = [note for note in fragment if note != "~"]
    if not pitches:
        raise ValueError("The new motif needs a sounding note")
    minimum, maximum = section.steps
    # Pocket can return a shorter motif than the structured Ollama response.
    count = min(maximum, max(minimum, len(fragment)))
    notes = [fragment[i % len(fragment)] for i in range(count)]
    if all(note == "~" for note in notes):
        notes[0] = pitches[0]
    if section.fast > 1:
        # Keep a perceptible pulse even when the model supplies mostly rests.
        for i in range(0, count, 2):
            if notes[i] == "~":
                notes[i] = pitches[(i // 2) % len(pitches)]
    lead = Layer(" ".join(in_register(notes, section.octave)), section.synth,
                 section.clip, section.fast, 0.62)
    result = list(layers)
    result[target] = lead

    others = [i for i, layer in enumerate(layers) if i != target and layer.audible]
    editable = [i for i in others if i not in hand_edits]
    answer = None
    if len(others) >= 2 and editable:
        # Keep at least one audible part exactly as played in the preceding bar.
        answer = max(editable, key=lambda i: layers[i].gain)
        old = [note for note in layers[answer].notes.split() if note != "~"]
        a, b = in_register([old[0], old[-1]], 4 if section.octave == 2 else 2)
        if section.fast == 1:
            result[answer] = Layer(f"{a} ~ {b}", "sine", 0.86, 1, 0.30)
        else:
            result[answer] = Layer(f"~ {a} ~ {b} {a} ~", "triangle", 0.34, 2, 0.36)
    return result, answer


def edit_motif(layer, step, fragment):
    """Splice a small edit into the current motif, wrapping at the bar line."""
    notes = layer.notes.split()
    fragment = fragment[:min(3, len(notes))]
    for offset, note in enumerate(fragment):
        notes[(step + offset) % len(notes)] = note
    updated = " ".join(notes)
    if updated == layer.notes or all(note == "~" for note in notes):
        raise ValueError("Layer edit must change the motif and leave a sounding note")
    return updated, (step + len(fragment)) % len(notes)
