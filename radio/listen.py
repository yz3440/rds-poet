"""Print one two-second HackRF reception report per FM frequency and exit."""

import argparse
from contextlib import contextmanager
from itertools import cycle
import math
import subprocess
import threading

import numpy as np
from scipy import signal


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
    freq, power = signal.welch(x, AUDIO_RATE, nperseg=min(1024, len(x)))
    band = (freq >= 60) & (freq < 14_000)
    center = np.average(freq[band], weights=power[band] + 1e-20)
    tone = "low and muffled" if center < 900 else "bright and thin" if center > 3200 else "midrange"
    windows = x[:len(x) // 320 * 320].reshape(-1, 320)
    levels = np.sqrt(np.mean(windows**2, axis=1))
    texture = "pulsing" if levels.std() > 0.35 * rms else "steady"
    return f"{label}: {tone}, {texture}, audio level {20 * math.log10(rms):.0f} dBFS"


def capture_hackrf(mhz):
    # Two seconds of signed eight-bit interleaved I/Q. The subprocess releases
    # the device before the caller can construct the transmitter.
    count = 2 * IQ_RATE
    recording = subprocess.run(
        ["hackrf_transfer", "-r", "-", "-f", str(round(mhz * 1e6) - TUNE_OFFSET),
         "-s", str(IQ_RATE), "-n", str(count), "-l", "32", "-g", "24"],
        capture_output=True, timeout=15,
    )
    if recording.returncode or len(recording.stdout) != count * 2:
        raise RuntimeError("Incomplete HackRF capture: " + recording.stderr.decode(errors="replace")[-500:])
    iq = np.frombuffer(recording.stdout, np.int8).astype(np.float32).view(np.complex64)
    return describe_channel(iq, mhz)


def describe_channel(iq, mhz):
    # Mix the station to baseband. The receiver's DC spike then lies outside
    # the FM channel instead of overwhelming a weak broadcast during demodulation.
    period = IQ_RATE // TUNE_OFFSET
    mixer = np.exp(-2j * np.pi * np.arange(period) / period).astype(np.complex64)
    iq.reshape(-1, period)[:] *= mixer
    channel = signal.resample_poly(iq, 1, 32)  # 250 kHz complex FM channel
    mpx = np.angle(channel[1:] * channel[:-1].conj()) * 250_000 / (2*np.pi*75_000)
    audio = signal.resample_poly(mpx, 16, 125)
    cue = describe(audio[AUDIO_RATE // 4:], mhz)
    # An empty FM channel can demodulate into loud noise. Compare out-of-band
    # discriminator noise to the mono programme before giving it to the poet.
    freq, power = signal.welch(mpx[62_500:], 250_000, nperseg=1024)
    hiss = power[(freq >= 80_000) & (freq <= 95_000)].mean()
    programme = power[(freq >= 60) & (freq <= 14_000)].mean()
    if hiss * 15_000 > 0.12**2 and hiss > 0.4 * programme:
        cue += ", mostly radio hiss"
    return cue


def station_cues(frequencies, *, before_tx=False):
    """Rotate stations, releasing the single HackRF before FM output starts."""
    if before_tx:
        observations = [capture_hackrf(mhz) for mhz in frequencies]
        return cycle(observations).__next__
    stations = cycle(frequencies)
    return lambda: capture_hackrf(next(stations))


@contextmanager
def rtl_cues(mhz):
    """Continuously demodulate FM on a second receiver while HackRF transmits."""
    command = ["rtl_fm", "-f", str(round(mhz * 1e6)), "-M", "fm",
               "-s", "200000", "-r", str(AUDIO_RATE), "-"]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    ready = threading.Event()
    state = {"cue": None, "error": None}

    def receive():
        try:
            while True:
                raw = proc.stdout.read(2 * AUDIO_RATE)
                if len(raw) != 2 * AUDIO_RATE:
                    raise RuntimeError("RTL-SDR audio stream ended")
                state["cue"] = describe(np.frombuffer(raw, dtype="=i2") / 32768, mhz)
                ready.set()
        except Exception as exc:
            state["error"] = exc
        finally:
            ready.set()

    def latest():
        if state["error"]:
            raise RuntimeError(str(state["error"]))
        return state["cue"]

    worker = threading.Thread(target=receive, daemon=True)
    worker.start()
    try:
        if not ready.wait(12):
            raise RuntimeError("Timed out waiting for RTL-SDR FM audio")
        latest()
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
    args = parser.parse_args()
    if any(not 87.5 <= mhz <= 108 for mhz in args.frequencies):
        parser.error("choose FM broadcast frequencies between 87.5 and 108 MHz")
    try:
        for mhz in args.frequencies:
            print(capture_hackrf(mhz), flush=True)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Reception stopped: {exc}\n")


if __name__ == "__main__":
    main()
