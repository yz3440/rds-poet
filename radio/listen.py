"""Sample each FM frequency on HackRF and print the measurements and optional speech."""

import argparse
from collections import deque
from contextlib import contextmanager
from itertools import cycle
import math
import subprocess
import sys
import threading

import numpy as np
from scipy import signal
from radio.cues import pair_observations
from radio.transcription import SPEECH_SECONDS, Whisper, speech_cue


AUDIO_RATE = 32_000
IQ_RATE = 8_000_000
TUNE_OFFSET = 250_000


def describe(audio, mhz):
    """Describe audible energy, spectral balance and fluctuations in a short window."""
    x = np.asarray(audio, dtype=np.float32)
    x = x - x.mean()
    rms = np.sqrt(np.mean(x * x))
    label = f"{mhz:g} MHz FM"
    if rms < 0.002:
        return f"{label}: nearly still"
    # A 32 ms Hann window, advanced by 10 ms. Positive spectral flux counts
    # attacks even when a change of sound barely changes the overall level.
    freq, _, magnitude = signal.spectrogram(
        x, AUDIO_RATE, window="hann", nperseg=1024, noverlap=704, scaling="spectrum", mode="magnitude",
    )
    audible = (freq >= 60) & (freq < 14_000)
    band = magnitude[audible]
    center = np.average(freq[audible], weights=(band**2).mean(axis=1) + 1e-20)
    flux = np.maximum(np.diff(band, axis=1), 0).sum(axis=0)
    flux /= max(band.sum(axis=0).mean(), 1e-12)
    background = np.median(flux)
    threshold = max(0.10, background + 2 * np.median(abs(flux - background)))
    peaks, _ = signal.find_peaks(flux, height=threshold, prominence=0.05, distance=8)
    attacks = len(peaks) * AUDIO_RATE / len(x)
    activity = "frequent attacks" if attacks >= 2.5 else "sparse attacks"
    tone = "low and muffled" if center < 900 else "bright and thin" if center > 3200 else "midrange"
    windows = x[:len(x) // 320 * 320].reshape(-1, 320)
    levels = np.sqrt(np.mean(windows**2, axis=1))
    texture = "pulsing" if levels.std() > 0.35 * rms else "steady"
    return (f"{label}: {tone}, {texture}, audio level {20 * math.log10(rms):.0f} dBFS, "
            f"centroid {center:.0f} Hz, {activity} ({attacks:.1f} attacks/s)")


def receive_iq(center_hz, count):
    """Return exactly count interleaved signed I/Q samples, closing the HackRF."""
    try:
        recording = subprocess.run(
            ["hackrf_transfer", "-r", "-", "-f", str(center_hz),
             "-s", str(IQ_RATE), "-n", str(count), "-l", "32", "-g", "24"],
            capture_output=True, timeout=count / IQ_RATE + 15,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out receiving samples from HackRF") from exc
    if recording.returncode or len(recording.stdout) != count * 2:
        detail = recording.stderr.decode(errors="replace")[-500:].strip()
        if "resource busy" in detail.lower():
            raise RuntimeError("HackRF is busy. End the current HackRF session before sampling.")
        raise RuntimeError("Incomplete HackRF capture: " + detail)
    return np.frombuffer(recording.stdout, np.int8)


def capture_hackrf(mhz, recognizer=None):
    seconds = SPEECH_SECONDS if recognizer else 2
    raw = receive_iq(round(mhz * 1e6) - TUNE_OFFSET, seconds * IQ_RATE)
    # Decimate in one-second blocks to bound the size of floating-point I/Q.
    channel = np.concatenate([tune_channel(block.astype(np.float32).view(np.complex64))
                              for block in np.split(raw, seconds)])
    return describe_baseband(channel, mhz, recognizer)


def tune_channel(iq):
    # Mix the station to baseband. The receiver's DC spike then lies outside
    # the FM channel instead of overwhelming a weak broadcast during demodulation.
    period = IQ_RATE // TUNE_OFFSET
    mixer = np.exp(-2j * np.pi * np.arange(period) / period).astype(np.complex64)
    iq.reshape(-1, period)[:] *= mixer
    return signal.resample_poly(iq, 1, 32)  # 250 kHz complex FM channel


def describe_channel(iq, mhz):
    return describe_baseband(tune_channel(iq), mhz)


def describe_baseband(channel, mhz, recognizer=None):
    mpx = np.angle(channel[1:] * channel[:-1].conj()) * 250_000 / (2*np.pi*75_000)
    audio = signal.resample_poly(mpx, 16, 125)[AUDIO_RATE // 4:]
    # An empty FM channel can demodulate into loud noise. Compare out-of-band
    # discriminator noise to the mono programme before giving it to the poet.
    settled = mpx[62_500:]
    freq, power = signal.welch(settled, 250_000, nperseg=1024)
    hiss = power[(freq >= 80_000) & (freq <= 95_000)].mean()
    programme = power[(freq >= 60) & (freq <= 14_000)].mean()
    if abs(settled.mean()) * 75_000 > 25_000:
        cue = f"{mhz:g} MHz FM: off-center signal, likely adjacent-channel bleed"
    elif hiss * 15_000 > 0.12**2 and hiss > 0.4 * programme:
        cue = f"{mhz:g} MHz FM: mostly radio hiss"
    else:
        cue = describe(audio, mhz)
    return speech_cue(cue, audio, recognizer)


def station_cues(frequencies, *, neighbors=None, before_tx=False, recognizer=None):
    """Rotate station pairs, releasing the single HackRF before FM output starts."""
    neighbors = neighbors or {}

    def observe(mhz, capture):
        lead = capture(mhz, recognizer)
        neighbor = neighbors.get(mhz)
        if neighbor is None:
            return lead
        return pair_observations(lead, capture(neighbor, None))

    if before_tx:
        stored = {mhz: capture_hackrf(mhz, recognizer) for mhz in dict.fromkeys(frequencies)}
        for mhz in neighbors.values():
            if mhz not in stored:
                stored[mhz] = capture_hackrf(mhz)
        observations = [observe(mhz, lambda freq, _: stored[freq]) for mhz in frequencies]
        return cycle(observations).__next__
    stations = cycle(frequencies)
    return lambda: observe(next(stations), capture_hackrf)


def hackrf_cues(frequencies=None, *, scan=True, dial=100.0, neighbor_offset_khz=None,
                before_tx=False, recognizer=None):
    from radio.scan import survey

    if not scan and neighbor_offset_khz is None:
        return station_cues(frequencies, before_tx=before_tx, recognizer=recognizer)
    stations = survey() if scan else []
    usable = [s for s in stations if not (before_tx and abs(s.mhz - dial) < 0.05)]
    clear = [s.mhz for s in usable if s.clear]
    detected = [s.mhz for s in usable]
    sources = frequencies or sorted(clear or detected, key=lambda mhz: (round(abs(mhz - dial), 3), mhz))[:2]
    if not sources:
        raise RuntimeError("No source stations detected. Reposition the antenna or try --no-scan --sample-mhz MHZ.")
    neighbors = {}
    for mhz in sources:
        others = [other for other in clear if abs(mhz - other) >= 0.15]
        if not others:
            others = [other for other in detected if abs(mhz - other) >= 0.15]
        if neighbor_offset_khz is not None:
            near = round(mhz + neighbor_offset_khz / 1000, 6)
            if not 87.5 <= near <= 108:
                print(f"Source {mhz:g} MHz, requested neighbor outside the FM band", file=sys.stderr)
                continue
        elif others:
            near = min(others, key=lambda other: (round(abs(mhz - other), 3), other))
        else:
            print(f"Source {mhz:g} MHz, no distinct neighbor detected", file=sys.stderr, flush=True)
            continue
        neighbors[mhz] = near
        print(f"Source {mhz:g} MHz, nearby channel {near:g} MHz ({near - mhz:+.1f} MHz)",
              file=sys.stderr, flush=True)
    return station_cues(sources, neighbors=neighbors, before_tx=before_tx, recognizer=recognizer)


@contextmanager
def rtl_cues(mhz, recognizer=None):
    """Continuously demodulate FM on a second receiver while HackRF transmits."""
    command = ["rtl_fm", "-f", str(round(mhz * 1e6)), "-M", "fm",
               "-s", "200000", "-r", str(AUDIO_RATE), "-"]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    ready = threading.Event()
    windows = deque(maxlen=SPEECH_SECONDS if recognizer else 1)
    lock = threading.Lock()
    state = {"error": None}

    def receive():
        try:
            while True:
                raw = proc.stdout.read(2 * AUDIO_RATE)
                if len(raw) != 2 * AUDIO_RATE:
                    raise RuntimeError("RTL-SDR audio stream ended")
                with lock:
                    windows.append(raw)
                    if len(windows) == windows.maxlen:
                        ready.set()
        except Exception as exc:
            state["error"] = exc
        finally:
            ready.set()

    def latest():
        if state["error"]:
            raise RuntimeError(str(state["error"]))
        with lock:
            window = b"".join(windows)
        audio = np.frombuffer(window, dtype="=i2").astype(np.float32) / 32768
        return speech_cue(describe(audio, mhz), audio, recognizer)

    worker = threading.Thread(target=receive, daemon=True)
    worker.start()
    try:
        if not ready.wait(12):
            raise RuntimeError("Timed out waiting for RTL-SDR FM audio")
        if state["error"]:
            raise RuntimeError(str(state["error"]))
        yield latest
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        worker.join(timeout=3)
        proc.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frequencies", type=float, nargs="+", metavar="MHZ")
    parser.add_argument("--transcribe", action="store_true", help="include English speech from six-second samples")
    args = parser.parse_args()
    if any(not 87.5 <= mhz <= 108 for mhz in args.frequencies):
        parser.error("choose FM broadcast frequencies between 87.5 and 108 MHz")
    try:
        recognizer = Whisper() if args.transcribe else None
        for mhz in args.frequencies:
            print(capture_hackrf(mhz, recognizer), flush=True)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Reception stopped: {exc}\n")


if __name__ == "__main__":
    main()
