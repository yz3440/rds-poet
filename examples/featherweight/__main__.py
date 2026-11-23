"""Make a four-line FM-inspired poem with a playable Strudel score."""

import argparse
import json
from pathlib import Path
import re
from time import monotonic
from urllib.error import HTTPError, URLError
import wave

import numpy as np

from radio.live_rds import DEFAULT_LINE_SECONDS
from score import Composer, PATTERN, RATE, render


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "qwen2.5:1.5b"
ARCHIVE_CUE = (
    'Published redsea RDS example, group 2A, MDR JUMP. '
    'Track title: Das Leichteste der Welt (the lightest thing in the world).'
)
LINE_PROMPTS = (
    "Place an everyday object somewhere specific.",
    "Show something moving at the edge of that scene.",
    "Give the sound to a different object.",
    "End on a visible trace left by the movement or sound.",
)


def compose(model, next_cue):
    composer = Composer(model)
    phrases = []
    cues = []
    for index, instruction in enumerate(LINE_PROMPTS):
        theme = f"English poem, line {index + 1} of four. {instruction} Use your own words."
        start = monotonic()
        cue = next_cue()
        print(cue, flush=True)
        phrase = composer.compose(theme, cue, DEFAULT_LINE_SECONDS)
        print(f"{monotonic() - start:.1f}s   {phrase.line}", flush=True)
        cues.append(cue)
        phrases.append(phrase)
    return phrases, cues


def save_take(out, phrases, cues, credit):
    rows = [{"line": phrase.line, "strudel": phrase.strudel, "cue": cue}
            for phrase, cue in zip(phrases, cues)]
    expressions = ",\n  ".join(phrase.strudel for phrase in phrases)
    program = f"setcps({1 / DEFAULT_LINE_SECONDS})\nslowcat(\n  {expressions}\n)\n"

    # The web player builds the same patterns using the public Strudel API.
    notes = []
    for phrase, cue in zip(phrases, cues):
        match = re.fullmatch(PATTERN, phrase.strudel)
        notes.append({"notes": match[1], "synth": match[2], "clip": float(match[3] or 1),
                      "fast": int(match[4] or 1), "line": phrase.line, "cue": cue})
    payload = {"credit": credit, "bars": notes, "seconds": DEFAULT_LINE_SECONDS}
    html = (HERE / "player.html.in").read_text(encoding="utf-8")
    html = html.replace("__TAKE_JSON__", json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c"))

    out.mkdir(parents=True, exist_ok=True)
    (out / "score.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (out / "featherweight.strudel").write_text(program, encoding="utf-8")
    with wave.open(str(out / "preview.wav"), "wb") as wav:
        wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        for phrase in phrases:
            wav.writeframes(np.rint(32767 * phrase.audio).astype("<i2").tobytes())
    (out / "index.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="installed Ollama model for a new take")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--sample-mhz", type=float, nargs="+", metavar="MHZ",
                        help="take a fresh HackRF sample per line, rotating through these stations")
    inputs.add_argument("--reference", action="store_true", help="render the included Codex-written session")
    parser.add_argument("--out", type=Path, default=Path("build/featherweight"))
    args = parser.parse_args()
    try:
        if args.reference:
            source = (HERE / "reference.jsonl").read_text(encoding="utf-8")
            phrases = [render(json.loads(line), DEFAULT_LINE_SECONDS) for line in source.splitlines()]
            cues = [ARCHIVE_CUE] * len(phrases)
            credit = "Reference take written by Codex"
        else:
            next_cue = lambda: ARCHIVE_CUE
            if args.sample_mhz:
                if any(not 87.5 <= freq <= 108 for freq in args.sample_mhz):
                    parser.error("--sample-mhz must be between 87.5 and 108")
                from radio.listen import station_cues
                next_cue = station_cues(args.sample_mhz)
            phrases, cues = compose(args.model, next_cue)
            credit = f"Local model · {args.model}"
        save_take(args.out, phrases, cues, credit)
    except HTTPError as exc:
        parser.exit(1, f"Ollama HTTP {exc.code}: {exc.read(512).decode(errors='replace')}\n")
    except URLError as exc:
        parser.exit(1, f"Could not reach local Ollama: {exc.reason}. Start Ollama and pull {args.model}.\n")
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        parser.exit(1, f"Could not make the take: {exc}\n")
    print(f"Open {(args.out / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
