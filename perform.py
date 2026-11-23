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
import threading
from urllib.error import HTTPError

import numpy as np
from gnuradio import audio, blocks, gr

from radio.live_tx import Transmitter, hackrf_sink
from score import RATE, generate, render


class Player(gr.sync_block):
    """Advance both the line and the sound on audio cycle boundaries."""

    def __init__(self, first, queue, display, bars=None):
        super().__init__("Poetry score", in_sig=None, out_sig=[np.float32])
        self.current = first
        self.queue = queue
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
                    self.current = self.queue.get_nowait()
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
    history = []
    pocket = None
    if args.model == "pocket" and not args.score:
        from pocket import PocketPoet
        pocket = PocketPoet()
        print("Loaded the local corpus-trained neural model.", file=sys.stderr)
    if args.score:
        with args.score.open(encoding="utf-8") as file:
            phrases = [render(json.loads(line), args.line_seconds) for line in file if line.strip()]
        if not phrases:
            raise ValueError("Score is empty")
        sequence = cycle(phrases)

    def compose():
        if args.score:
            return next(sequence)
        inspiration = cue()
        print(f"Composing from: {inspiration}", file=sys.stderr, flush=True)
        for attempt in range(3):
            try:
                if pocket is None:
                    reply = generate(args.model, args.theme, inspiration, history)
                else:
                    reply = pocket.reply(args.theme, inspiration, history)
                phrase = render(reply, args.line_seconds)
                if phrase.line in history:
                    raise ValueError("Repeated recent line")
                break
            except (ValueError, KeyError, TypeError) as exc:
                if attempt == 2:
                    raise ValueError(f"No playable phrase after three replies: {exc}") from exc
        history.append(phrase.line)
        del history[:-6]
        return phrase

    first = compose()
    upcoming = Queue(maxsize=1)
    stopped = threading.Event()
    request_phrase = threading.Event()
    failure = []

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
        print(f"{phrase.line}\n  {phrase.strudel}", flush=True)
        request_phrase.set()

    player = Player(first, upcoming, display, args.bars)
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

    worker = threading.Thread(target=produce, name="composer", daemon=True)
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
        worker.start()
        stopped.wait()
    finally:
        stopped.set()
        graph.stop()
        waiter.join()
        if args.wav_out:
            sink.close()
    if failure:
        raise failure[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:1.5b", help="local Ollama model name, or pocket for the built-in neural model")
    parser.add_argument("--theme", default="What can be heard through a closed door?")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sample-mhz", type=float, help="sample FM with HackRF once before playback")
    source.add_argument("--rtl-mhz", type=float, help="refresh FM cues continuously using an RTL-SDR")
    parser.add_argument("--score", type=Path, help="rehearse a JSONL score instead of running the model")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--fm", type=float, metavar="MHZ", help="transmit music and RDS using HackRF")
    output.add_argument("--wav-out", type=Path, help="record mono audio in real time")
    parser.add_argument("--line-seconds", type=float, default=2, help="duration of one Strudel cycle")
    parser.add_argument("--tx-gain-db", type=int, default=8)
    parser.add_argument("--bars", type=int, help="stop a speaker/WAV preview after N cycles")
    args = parser.parse_args()
    if not math.isfinite(args.line_seconds) or not 0.25 <= args.line_seconds <= 60:
        parser.error("choose a cycle duration between 0.25 and 60 seconds")
    for frequency in (args.fm, args.sample_mhz, args.rtl_mhz):
        if frequency is not None and not 87.5 <= frequency <= 108:
            parser.error("choose an FM broadcast frequency between 87.5 and 108 MHz")
    if args.bars is not None and (args.bars < 1 or args.fm):
        parser.error("--bars requires a positive count and a speaker or WAV preview")
    if args.score and (args.rtl_mhz or args.sample_mhz):
        parser.error("air cues require model generation")
    if not 0 <= args.tx_gain_db <= 47:
        parser.error("TX gain must be between 0 and 47 dB")
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        with ExitStack() as stack:
            if args.rtl_mhz:
                from radio.listen import rtl_cues
                cue = stack.enter_context(rtl_cues(args.rtl_mhz))
            else:
                initial = "Room tone imagined from the theme"
                if args.sample_mhz:
                    from radio.listen import capture_hackrf
                    initial = capture_hackrf(args.sample_mhz)
                cue = lambda: initial
            run(args, cue)
    except KeyboardInterrupt:
        return
    except HTTPError as exc:
        parser.exit(1, f"Ollama HTTP {exc.code}: {exc.read(512).decode(errors='replace')}\n")
    except (OSError, RuntimeError, ValueError, KeyError, TimeoutError) as exc:
        parser.exit(1, f"Performance stopped: {exc}\n")


if __name__ == "__main__":
    main()
