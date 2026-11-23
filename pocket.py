"""A small, offline neural language model for this score's own vocabulary."""

from pathlib import Path
import re

import numpy as np


class PocketPoet:
    """A conditional next-token network trained on paired verse and note sequences."""

    def __init__(self):
        records = []
        path = Path(__file__).with_name("examples") / "air-poems.tsv"
        for row in path.read_text().splitlines():
            if not row or row.startswith("#"):
                continue
            air, line, synth, notes = row.split("|")
            register, pulse = air.split()
            records.append((register, pulse, "verse", line.split()))
            records.append((register, pulse, "sound", [synth]))
            records.append((register, pulse, "notes", notes.split()))
        vocabulary = {"<start>", "<end>"}
        for reg, pulse, part, words in records:
            vocabulary.update([reg, pulse, part, *words])
        self.words = sorted(vocabulary)
        self.ids = {word: i for i, word in enumerate(self.words)}
        self.allowed = {}
        inputs, targets = [], []
        for reg, pulse, part, words in records:
            allowed = self.allowed.setdefault(part, set())
            allowed.update(words)
            recent = ["<start>", "<start>"]
            for word in [*words, "<end>"]:
                inputs.append([self.ids[t] for t in [reg, pulse, part, *recent]])
                targets.append(self.ids[word])
                recent = [recent[-1], word]

        # Learn to predict the next word/note from two recent tokens and the
        # acoustic context. Separate input tables preserve the five positions.
        count = len(self.words)
        rng = np.random.default_rng(87)
        self.hidden = rng.normal(0, 0.1, (5, count, 48)).astype(np.float32)
        self.output = rng.normal(0, 0.1, (48, count)).astype(np.float32)
        self.bias = np.zeros(count, np.float32)
        x, y = np.array(inputs), np.array(targets)
        for _ in range(4200):
            batch = rng.integers(len(y), size=32)
            keys = x[batch].T
            h = np.tanh(self.hidden[np.arange(5)[:, None], keys].sum(axis=0))
            logits = h @ self.output + self.bias
            error = np.exp(logits - logits.max(axis=1, keepdims=True))
            error /= error.sum(axis=1, keepdims=True)
            error[np.arange(len(batch)), y[batch]] -= 1
            error *= 0.18 / len(batch)
            dh = (error @ self.output.T) * (1 - h*h)
            self.output -= h.T @ error
            self.bias -= error.sum(axis=0)
            for pos, indices in enumerate(keys):
                np.add.at(self.hidden[pos], indices, -dh)
        self.rng = np.random.default_rng()

    def sample(self, reg, pulse, part, limit, anchors):
        start = self.ids["<start>"]
        end = self.ids["<end>"]
        keys = [self.ids[reg], self.ids[pulse], self.ids[part], start, start]
        allowed = [self.ids[w] for w in sorted(self.allowed[part])]
        words = []
        for _ in range(limit):
            h = np.tanh(self.hidden[np.arange(5), keys].sum(axis=0))
            scores = h @ self.output + self.bias
            choices = allowed
            if len(words) >= (3 if part == "verse" else 1):
                choices = [*allowed, end]
            logits = scores[choices].copy()
            for i, token in enumerate(choices):
                if self.words[token] in anchors and self.words[token] not in words:
                    logits[i] += 0.8
            probs = np.exp((logits - logits.max()) / 0.78)
            token = int(self.rng.choice(choices, p=probs / probs.sum()))
            if token == end:
                break
            words.append(self.words[token])
            keys[-2:] = [keys[-1], token]
        return words

    def reply(self, theme, cue, history):
        reg = "midrange"
        for word, value in [("still", "quiet"), ("bright", "bright"), ("low", "low")]:
            if word in cue.lower():
                reg = value
                break
        pulse = "pulsing" if "pulsing" in cue else "steady"
        anchors = set(re.findall(r"[a-z]{3,}", (theme + " " + " ".join(history[-1:])).lower()))
        anchors -= {"the", "and", "with"}
        line = " ".join(self.sample(reg, pulse, "verse", 8, anchors))
        sound = self.sample(reg, pulse, "sound", 1, set())[0]
        notes = self.sample(reg, pulse, "notes", 8, set())
        if all(n == "~" for n in notes):
            notes[-1] = {"quiet": "c3", "low": "d2", "midrange": "c4", "bright": "g5"}[reg]
        return {"line": line, "strudel": f'note("{" ".join(notes)}").s("{sound}")'}
