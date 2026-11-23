#!/usr/bin/env python3
"""Perform generated poetry and pitched Strudel patterns on speakers or FM."""

import argparse
from contextlib import ExitStack
from itertools import cycle
import json
import math
from pathlib import Path
from queue import Empty, Queue
import signal
import sys
from textwrap import indent
import threading
from urllib.error import HTTPError

import numpy as np
from gnuradio import audio, blocks, gr

from radio.live_rds import DEFAULT_LINE_SECONDS
from radio.live_tx import DEFAULT_FM_MHZ, Transmitter, hackrf_sink
from score import Composer, RATE, render
from sketch import save_sketch


class Player(gr.sync_block):
    """Advance both the line and the sound on audio cycle boundaries."""

    def __init__(self, first, next_phrase, display, bars=None):
        super().__init__("Poetry score", in_sig=None, out_sig=[np.float32])
        self.current = first
        self.next_phrase = next_phrase
        self.display = display
        self.position = 0
        self.started = False
        self.bars = bars
        self.set_max_output_buffer(4096)

    def work(self, inputs, outputs):
        if self.position == 0:
            if self.bars == 0:
                return -1
            if self.bars is not None:
                self.bars -= 1
            if not self.started:
                self.started = True
                self.display(self.current)
            else:
                try:
                    self.current = self.next_phrase()
                except StopIteration:
                    return -1
                except Empty:
                    pass  # Continue the current cycle during a slow model response.
                else:
                    self.display(self.current)
        data = self.current.audio
        count = min(len(outputs[0]), len(data) - self.position)
        outputs[0][:count] = data[self.position:self.position + count]
        self.position = (self.position + count) % len(data)
        return count


def run(args, cue):
    stopped = threading.Event()
    request_phrase = threading.Event()
    failure = []
    cue_label = "Writing cue"
    if args.score:
        cue_label = "Saved cue"
        with args.score.open(encoding="utf-8") as file:
            phrases = [render(json.loads(line), args.line_seconds) for line in file if line.strip()]
        if not phrases:
            raise ValueError("Score is empty")
        sequence = cycle(phrases) if args.loop_score else iter(phrases)
        next_phrase = sequence.__next__
        first = next_phrase()
    else:
        if args.sample_mhz and args.fm:
            cue_label = "Pre-broadcast radio sample"
        elif args.sample_mhz or args.rtl_mhz:
            cue_label = "Radio sample"
        composer = Composer(args.model, args.sketch)
        upcoming = Queue(maxsize=1)
        next_phrase = upcoming.get_nowait

        def compose():
            return composer.compose(args.theme, cue(), args.line_seconds)

        first = compose()

    def produce():
        while not stopped.is_set():
            if not request_phrase.wait(0.25) or stopped.is_set():
                continue
            request_phrase.clear()
            try:
                upcoming.put_nowait(compose())
            except ValueError as exc:
                print(f"Holding the current phrase: {exc}", file=sys.stderr, flush=True)
                if not stopped.wait(0.5):
                    request_phrase.set()
            except Exception as exc:
                failure.append(exc)
                stopped.set()

    def display(phrase):
        if args.fm:
            graph.rds.set_title(phrase.line)
        if phrase.cue:
            print(f"{cue_label}: {phrase.cue}", file=sys.stderr, flush=True)
        if phrase.edited_layer is not None:
            print(f"Edited sketch layer {phrase.edited_layer + 1}", file=sys.stderr, flush=True)
        print(phrase.line, indent(phrase.strudel, "  "), sep="\n", flush=True)
        if not args.score and player.bars != 0:
            request_phrase.set()

    player = Player(first, next_phrase, display, args.bars)
    if args.fm:
        graph = Transmitter(hackrf_sink(args.fm * 1e6, args.tx_gain_db), first.line,
                            mono=True, audio_source=player)
    else:
        graph = gr.top_block()
        if args.wav_out:
            clock = blocks.throttle(gr.sizeof_float, RATE)
            sink = blocks.wavfile_sink(str(args.wav_out), 1, RATE, blocks.FORMAT_WAV, blocks.FORMAT_PCM_16)
            graph.connect(player, clock)
            output = clock
        else:
            sink = audio.sink(RATE, "", True)
            output = player
        graph.connect(output, sink)

    print(f"setcps({1 / args.line_seconds:.6g})", flush=True)
    graph.start()

    def wait_graph():
        try:
            graph.wait()
        finally:
            stopped.set()

    waiter = threading.Thread(target=wait_graph, name="playback", daemon=True)
    waiter.start()
    try:
        if not args.score:
            threading.Thread(target=produce, name="composer", daemon=True).start()
        stopped.wait()
    finally:
        stopped.set()
        graph.stop()
        waiter.join()
        if args.wav_out:
            sink.close()
        if args.save_sketch and player.started:
            # The composer can be one phrase ahead. Save the last audible revision.
            save_sketch(args.save_sketch, player.current.strudel, args.line_seconds)
    if failure:
        raise failure[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:1.5b", help="local Ollama model name, or pocket for the built-in neural model")
    parser.add_argument("--theme", default="What can be heard through a closed door?")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sample-mhz", type=float, nargs="+", metavar="MHZ",
                        help="rotate through FM stations on HackRF (capture all before FM transmission)")
    source.add_argument("--rtl-mhz", type=float, help="refresh FM cues continuously using an RTL-SDR")
    parser.add_argument("--transcribe", action="store_true", help="use Whisper's English transcript as well as acoustic cues")
    parser.add_argument("--score", type=Path, help="play a saved JSONL score once")
    parser.add_argument("--loop-score", action="store_true", help="repeat the saved score")
    parser.add_argument("--sketch", type=Path, help="start from a Strudel sketch and pick up hand edits between compositions")
    parser.add_argument("--save-sketch", type=Path, help="write the last played sketch to this file on exit")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--fm", nargs="?", const=DEFAULT_FM_MHZ, type=float, metavar="MHZ",
                        help="transmit music and RDS using HackRF (default with --fm: %(const)s MHz)")
    output.add_argument("--wav-out", type=Path, help="record mono audio in real time")
    parser.add_argument("--line-seconds", type=float, default=DEFAULT_LINE_SECONDS,
                        help="seconds per line and Strudel cycle (default: %(default)s)")
    parser.add_argument("--tx-gain-db", type=int, default=8)
    parser.add_argument("--bars", type=int, help="stop a speaker/WAV preview after N cycles")
    args = parser.parse_args()
    if not math.isfinite(args.line_seconds) or not 0.25 <= args.line_seconds <= 60:
        parser.error("choose a cycle duration between 0.25 and 60 seconds")
    for frequency in (args.fm, args.rtl_mhz, *(args.sample_mhz or [])):
        if frequency is not None and not 87.5 <= frequency <= 108:
            parser.error("choose an FM broadcast frequency between 87.5 and 108 MHz")
    if args.bars is not None and (args.bars < 1 or args.fm):
        parser.error("--bars requires a positive count and a speaker or WAV preview")
    if args.score and (args.rtl_mhz or args.sample_mhz):
        parser.error("air cues require model generation")
    if args.transcribe and not (args.sample_mhz or args.rtl_mhz):
        parser.error("--transcribe needs --sample-mhz or --rtl-mhz")
    if args.loop_score and not args.score:
        parser.error("--loop-score requires --score")
    if args.score and args.sketch:
        parser.error("--sketch requires live composition")
    if args.save_sketch and args.save_sketch.resolve() in {
        path.resolve() for path in (args.sketch, args.score, args.wav_out) if path
    }:
        parser.error("use a separate output file for --save-sketch")
    if not 0 <= args.tx_gain_db <= 47:
        parser.error("TX gain must be between 0 and 47 dB")
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        with ExitStack() as stack:
            recognizer = None
            if args.transcribe:
                from radio.transcription import Whisper
                recognizer = Whisper()
            if args.rtl_mhz:
                from radio.listen import rtl_cues
                cue = stack.enter_context(rtl_cues(args.rtl_mhz, recognizer))
            elif args.sample_mhz:
                from radio.listen import station_cues
                cue = station_cues(args.sample_mhz, before_tx=args.fm is not None, recognizer=recognizer)
            else:
                cue = lambda: "Room tone imagined from the theme"
            run(args, cue)
    except KeyboardInterrupt:
        return
    except HTTPError as exc:
        parser.exit(1, f"Ollama HTTP {exc.code}: {exc.read(512).decode(errors='replace')}\n")
    except (OSError, RuntimeError, ValueError, KeyError, TimeoutError) as exc:
        parser.exit(1, f"Performance stopped: {exc}\n")


if __name__ == "__main__":
    main()
