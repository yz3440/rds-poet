"""Make a four-line FM-inspired poem with a playable Strudel score."""

import argparse
import json
import math
from pathlib import Path
import re
from time import monotonic
from urllib.error import HTTPError, URLError
import wave

import numpy as np

from radio.live_rds import DEFAULT_LINE_SECONDS
from score import PATTERN, RATE, generate, render


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "qwen2.5:1.5b"
CYCLES = 4
ARCHIVE_CUE = (
    'Published redsea RDS example, group 2A, MDR JUMP. '
    'Track title: Das Leichteste der Welt (the lightest thing in the world).'
)
THEME = (
    "Write a four-line sequence called Featherweight. Start with a physical "
    "object suggested by the radio cue. Make it change slightly with each line. "
    "Use original English words. Do not quote song lyrics. Give the final line a landing."
)


def compose(model, cue):
    history = []
    phrases = []
    for index in range(CYCLES):
        start = monotonic()
        for attempt in range(3):
            try:
                reply = generate(model, f"{THEME} This is line {index + 1} of {CYCLES}.", cue, history)
                phrase = render(reply, DEFAULT_LINE_SECONDS)
                if phrase.line in history:
                    raise ValueError("repeated line")
            except (ValueError, TypeError, KeyError) as exc:
                if attempt == 2:
                    raise ValueError(f"Line {index + 1}: three unusable replies ({exc})") from exc
            else:
                break
        print(f"{monotonic() - start:.1f}s   {phrase.line}", flush=True)
        history.append(phrase.line)
        phrases.append(phrase)
    return phrases


def save_take(out, phrases, cue, credit):
    rows = [{"line": phrase.line, "strudel": phrase.strudel} for phrase in phrases]
    expressions = ",\n  ".join(phrase.strudel for phrase in phrases)
    program = f"setcps({1 / DEFAULT_LINE_SECONDS})\nslowcat(\n  {expressions}\n)\n"

    # The web player builds the same patterns using the public Strudel API.
    notes = []
    for phrase in phrases:
        match = re.fullmatch(PATTERN, phrase.strudel)
        notes.append({"notes": match[1], "synth": match[2], "line": phrase.line})
    payload = {"cue": cue, "credit": credit, "bars": notes, "seconds": DEFAULT_LINE_SECONDS}
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
    inputs.add_argument("--sample-mhz", type=float, help="take a short acoustic sample with HackRF")
    inputs.add_argument("--reference", action="store_true", help="render the included Codex-written session")
    parser.add_argument("--out", type=Path, default=Path("build/featherweight"))
    args = parser.parse_args()
    cue = ARCHIVE_CUE
    try:
        if args.reference:
            source = (HERE / "reference.jsonl").read_text(encoding="utf-8")
            phrases = [render(json.loads(line), DEFAULT_LINE_SECONDS) for line in source.splitlines()]
            credit = "Reference take written by Codex"
        else:
            if args.sample_mhz is not None:
                if not math.isfinite(args.sample_mhz) or not 87.5 <= args.sample_mhz <= 108:
                    parser.error("--sample-mhz must be between 87.5 and 108")
                from radio.listen import capture_hackrf
                cue = capture_hackrf(args.sample_mhz)
            print(cue, flush=True)
            phrases = compose(args.model, cue)
            credit = f"Local model · {args.model}"
        save_take(args.out, phrases, cue, credit)
    except HTTPError as exc:
        parser.exit(1, f"Ollama HTTP {exc.code}: {exc.read(512).decode(errors='replace')}\n")
    except URLError as exc:
        parser.exit(1, f"Could not reach local Ollama: {exc.reason}. Start Ollama and pull {args.model}.\n")
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        parser.exit(1, f"Could not make the take: {exc}\n")
    print(f"Open {(args.out / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
