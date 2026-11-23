#!/usr/bin/env python3
"""Headless HackRF FM transmitter with line-by-line song title updates."""

import argparse
from itertools import cycle
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

from gnuradio import analog, blocks, filter, gr
from gnuradio.filter import firdes

from radio.live_rds import LiveRds, validate_title


MPX_RATE = 228_000
INTERPOLATION = 36
IQ_RATE = MPX_RATE * INTERPOLATION
DEVIATION = 75_000


class Transmitter(gr.top_block):
    def __init__(self, sink, title, wav=None, *, mono=False):
        super().__init__("Live RadioText")
        self.rds = LiveRds(title)
        # Two chips per bit. The filter interpolates 2375 chips/s to 228 kS/s.
        pulse = filter.interp_fir_filter_fff(
            96, firdes.root_raised_cosine(96, MPX_RATE, 2375, 1, 961)
        )
        carrier = analog.sig_source_f(MPX_RATE, analog.GR_SIN_WAVE, 57000, 0.035)
        rds_band = blocks.multiply_ff()
        multiplex = blocks.add_ff()
        self.connect(self.rds, pulse, (rds_band, 0))
        self.connect(carrier, (rds_band, 1))
        self.connect(rds_band, (multiplex, 0))
        if not mono:
            pilot = analog.sig_source_f(MPX_RATE, analog.GR_SIN_WAVE, 19000, 0.09)
            self.connect(pilot, (multiplex, 2))

        if wav:
            source = blocks.wavfile_source(str(wav), True)
            channels = source.channels()
            if channels not in (1, 2):
                raise ValueError("Use a mono or stereo audio file")
            audio = source
            if channels == 2:
                mix = blocks.add_ff()
                audio = blocks.multiply_const_ff(0.5)
                self.connect((source, 0), (mix, 0))
                self.connect((source, 1), (mix, 1))
                self.connect(mix, audio)
            rate = source.sample_rate()
            if rate != 48_000:
                divisor = math.gcd(rate, 48_000)
                resampler = filter.rational_resampler_fff(
                    interpolation=48_000 // divisor, decimation=rate // divisor
                )
                self.connect(audio, resampler)
                audio = resampler
            self.connect(audio, analog.fm_preemph(48000, 75e-6),
                         analog.rail_ff(-1, 1),
                         filter.fir_filter_fff(1, firdes.low_pass(0.55, 48000, 14500, 1500)),
                         filter.rational_resampler_fff(interpolation=19, decimation=4),
                         (multiplex, 1))
        else:
            tone = analog.sig_source_f(MPX_RATE, analog.GR_SIN_WAVE, 440, 0.12)
            self.connect(tone, (multiplex, 1))

        self.connect(multiplex, analog.frequency_modulator_fc(2 * math.pi * DEVIATION / MPX_RATE),
                     blocks.multiply_const_cc(0.85),
                     filter.rational_resampler_ccc(interpolation=INTERPOLATION, decimation=1), sink)


def hackrf_sink(freq_hz, tx_gain_db=8):
    # Some gr-osmosdr builds crash if libusb cannot initialize.
    check = subprocess.run(["hackrf_info"], capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError((check.stderr or check.stdout).strip() or "hackrf_info failed")

    import osmosdr

    sink = osmosdr.sink(args="numchan=1 hackrf=0")
    if "IF" not in sink.get_gain_names(0):
        raise RuntimeError("HackRF TX gain control unavailable. Check hackrf_info.")
    sink.set_sample_rate(IQ_RATE)
    sink.set_center_freq(freq_hz, 0)
    if abs(sink.get_sample_rate() - IQ_RATE) > 1 or abs(sink.get_center_freq(0) - freq_hz) > 1:
        raise RuntimeError("HackRF did not accept the requested sample rate and frequency")
    sink.set_gain(0, "RF", 0)
    applied_gain = sink.set_gain(tx_gain_db, "IF", 0)
    if applied_gain != tx_gain_db:
        raise RuntimeError(f"HackRF reported {applied_gain:g} dB for the requested {tx_gain_db} dB TX gain")
    sink.set_bandwidth(1.75e6, 0)
    return sink


def load_lyrics(path):
    lines = []
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if line:
            try:
                lines.append(validate_title(line))
            except ValueError as exc:
                raise ValueError(f"{path}, line {number}: {exc}") from exc
    if not lines:
        raise ValueError(f"{path} contains no lyric lines")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq-mhz", type=float, default=87.5)
    text_source = parser.add_mutually_exclusive_group()
    text_source.add_argument("--title", type=validate_title, default="RDS Poet")
    text_source.add_argument("--lyrics", type=Path, metavar="FILE",
                             help="rotate RadioText through a file, one line at a time")
    parser.add_argument("--line-seconds", type=float, metavar="SECONDS",
                        help="time per line with --lyrics (default: 2)")
    parser.add_argument("--audio-file", "--wav", metavar="FILE",
                        help="loop a local audio file supported by libsndfile (default: 440 Hz tone)")
    parser.add_argument("--mono", action="store_true",
                        help="omit the 19 kHz stereo pilot for mono FM reception with RDS")
    parser.add_argument("--tx-gain-db", type=int, default=8, metavar="DB",
                        help="HackRF TX VGA gain, 0 to 47 dB (default: 8)")
    args = parser.parse_args()
    if not 0 <= args.tx_gain_db <= 47:
        parser.error("--tx-gain-db must be an integer from 0 to 47")
    if not math.isfinite(args.freq_mhz) or not 1 <= args.freq_mhz <= 6000:
        parser.error("Use a HackRF frequency between 1 and 6000 MHz")
    if args.line_seconds is not None:
        if args.lyrics is None:
            parser.error("--line-seconds requires --lyrics")
        if not math.isfinite(args.line_seconds) or args.line_seconds <= 0:
            parser.error("--line-seconds must be finite and positive")
    interval = args.line_seconds if args.line_seconds is not None else 2.0
    try:
        lines = load_lyrics(args.lyrics) if args.lyrics else []
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    initial_title = lines[0] if lines else args.title
    try:
        sink = hackrf_sink(args.freq_mhz * 1e6, args.tx_gain_db)
        graph = Transmitter(sink, initial_title, args.audio_file, mono=args.mono)
        graph.start()
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"Could not start transmitter: {exc}\n")

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        print(f"Running at {args.freq_mhz:g} MHz with {args.tx_gain_db} dB TX gain. Ctrl+C stops.", flush=True)
        if lines:
            print(f"Rotating RadioText every {interval:g} seconds.", flush=True)
            for line in cycle(lines):
                graph.rds.set_title(line)
                print(f"Queued RadioText: {line}", flush=True)
                time.sleep(interval)
        print("Type a title and press Enter.", flush=True)
        for line in sys.stdin:
            title = line.rstrip("\r\n")
            try:
                graph.rds.set_title(title)
                print(f"Queued RadioText: {title}", flush=True)
            except ValueError as exc:
                print(exc, file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        graph.stop()
        graph.wait()


if __name__ == "__main__":
    main()
