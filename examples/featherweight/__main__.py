"""Make a four-line FM-inspired poem with a playable Strudel score."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic
from urllib.error import HTTPError, URLError
import wave

import numpy as np

from radio.live_rds import DEFAULT_LINE_SECONDS
from score import Composer, RATE, render
from sketch import parse_layers, save_sketch


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


def compose(model, next_cue, sketch=None, edit_size="bold"):
    composer = Composer(model, sketch, edit_size)
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
        notes.append({"layers": [asdict(layer) for layer in parse_layers(phrase.strudel)],
                      "line": phrase.line, "cue": cue})
    payload = {"credit": credit, "bars": notes, "seconds": DEFAULT_LINE_SECONDS}
    html = (HERE / "player.html.in").read_text(encoding="utf-8")
    html = html.replace("__TAKE_JSON__", json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c"))

    out.mkdir(parents=True, exist_ok=True)
    (out / "score.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (out / "featherweight.strudel").write_text(program, encoding="utf-8")
    save_sketch(out / "last-set.strudel", phrases[-1].strudel, DEFAULT_LINE_SECONDS)
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
    parser.add_argument("--scan", action=argparse.BooleanOptionalAction, default=None,
                        help="survey the FM dial for local stations (automatic with --sample-mhz)")
    parser.add_argument("--sketch", type=Path, help="start from this pitched Strudel sketch")
    parser.add_argument("--edit-size", choices=("bold", "subtle"), default="bold",
                        help="size of each generated sketch edit (default: %(default)s)")
    parser.add_argument("--neighbor-offset-khz", type=int, choices=(-200, -100, 100, 200),
                        help="sample this exact offset from each source, including quiet channels")
    parser.add_argument("--transcribe", action="store_true", help="include spoken English from FM samples")
    parser.add_argument("--out", type=Path, default=Path("build/featherweight"))
    args = parser.parse_args()
    if args.scan and args.reference:
        parser.error("--scan requires a generated take")
    if args.scan is None:
        args.scan = bool(args.sample_mhz)
    if args.neighbor_offset_khz and not (args.scan or args.sample_mhz):
        parser.error("--neighbor-offset-khz needs a HackRF source")
    if args.transcribe and not (args.scan or args.sample_mhz):
        parser.error("--transcribe needs --scan or --sample-mhz")
    if args.reference and args.sketch:
        parser.error("--sketch requires a generated take")
    try:
        if args.reference:
            source = (HERE / "reference.jsonl").read_text(encoding="utf-8")
            phrases = [render(json.loads(line), DEFAULT_LINE_SECONDS) for line in source.splitlines()]
            cues = [ARCHIVE_CUE] * len(phrases)
            credit = "Reference take written by Codex"
        else:
            next_cue = lambda: ARCHIVE_CUE
            if args.scan or args.sample_mhz:
                if any(not 87.5 <= freq <= 108 for freq in (args.sample_mhz or [])):
                    parser.error("--sample-mhz must be between 87.5 and 108")
                from radio.listen import Whisper, hackrf_cues
                next_cue = hackrf_cues(args.sample_mhz, scan=args.scan, neighbor_offset_khz=args.neighbor_offset_khz,
                                       recognizer=Whisper() if args.transcribe else None)
            phrases, cues = compose(args.model, next_cue, args.sketch, args.edit_size)
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
