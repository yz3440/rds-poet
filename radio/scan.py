"""Survey 87.5–108 MHz with two offset HackRF sweeps and check for broadcast FM."""

from dataclasses import dataclass
import sys

import numpy as np
from scipy import fft, signal

from radio.listen import IQ_RATE, receive_iq


DIAL_KHZ = range(87_500, 108_001, 100)
# Assign each channel to its nearest tuning. Shift the LO on the second pass
# so that a center spike or a fixed receiver spur cannot pass both checks.
CENTERS_KHZ = (90_050, 95_450, 100_850, 106_250)
BLOCK = 64_000
CHANNEL_RATE = 250_000
CHANNEL_BLOCK = BLOCK * CHANNEL_RATE // IQ_RATE
BIN_HZ = IQ_RATE // BLOCK
BLOCKS = 24


@dataclass(frozen=True)
class Reading:
    khz: int
    excess_db: float
    offset_khz: float
    modulation: float
    hiss: float
    pilot_db: float

    @property
    def likely_fm(self):
        # FM quiets the discriminator above the broadcast multiplex. Require
        # actual modulation as well, so a crystal spur can't pose as mono FM.
        if abs(self.offset_khz) >= 15 or self.modulation <= 0.006:
            return False
        if self.pilot_db >= 9:
            return self.excess_db >= 3.5 and self.hiss < 0.22
        return self.excess_db >= 5 and self.hiss < 0.085 and self.modulation > self.hiss


@dataclass(frozen=True)
class Station:
    mhz: float
    excess_db: float
    pilot: bool


def measure_window(iq, center_khz, channels):
    """Channelize one tuner window and measure each 100 kHz dial position."""
    blocks = iq[-BLOCKS * BLOCK:].reshape(-1, BLOCK)
    spectra = fft.fft(blocks - blocks.mean(axis=1, keepdims=True), axis=1)
    # Smooth over independent blocks before estimating the in-window floor.
    rf_psd = np.mean(abs(spectra)**2, axis=0)
    usable = abs(fft.fftfreq(BLOCK, 1 / IQ_RATE)) <= 3_000_000
    floor = max(np.percentile(rf_psd[usable], 20), 1e-20)
    bins = np.rint(fft.fftfreq(CHANNEL_BLOCK) * CHANNEL_BLOCK).astype(int)
    hz = abs(bins) * BIN_HZ
    taper = (0.5 + 0.5 * np.cos(np.pi * np.clip((hz - 90_000) / 24_000, 0, 1))).astype('f')
    results = []
    for khz in channels:
        indices = (bins + (khz - center_khz) * 1000 // BIN_HZ) % BLOCK
        # Both a channel and its central 70 kHz must rise above the floor.
        # Energy only on the skirts can come from two flanking stations.
        power = min(rf_psd[indices[hz < 75_000]].mean(), rf_psd[indices[hz < 35_000]].mean())
        excess_db = 10 * np.log10(max(power / floor, 1e-20))
        channel = fft.ifft(spectra[:, indices] * taper, axis=1)
        # Trim filter edge transients at both ends of each FFT block.
        z = channel[:, 24:-24]
        mpx = np.angle(z[:, 1:] * z[:, :-1].conj()) * (CHANNEL_RATE / (2*np.pi*75_000))
        offset = np.mean(mpx) * 75
        freq, psd = signal.periodogram(mpx, CHANNEL_RATE, window="hann", nfft=CHANNEL_BLOCK)
        psd = psd.mean(axis=0)
        audio = (freq >= 125) & (freq <= 15_000)
        hiss = (freq >= 80_000) & (freq <= 95_000)
        pilot = (freq >= 18_875) & (freq <= 19_125)
        side = (abs(freq - 19_000) >= 500) & (abs(freq - 19_000) <= 1500)
        pilot_db = 10 * np.log10(max(psd[pilot].max(), 1e-24) / max(np.median(psd[side]), 1e-24))
        results.append(Reading(khz, excess_db, offset, np.sqrt(psd[audio].sum() * BIN_HZ),
                               np.sqrt(psd[hiss].sum() * BIN_HZ), pilot_db))
    return results


def sweep(shift_khz):
    centers = [c + shift_khz for c in CENTERS_KHZ]
    readings = {}
    for center in centers:
        channels = [k for k in DIAL_KHZ if min(centers, key=lambda c: abs(k - c)) == center]
        # Allow 160 ms of tuner settling before the measured blocks.
        raw = receive_iq(center * 1000, IQ_RATE * 16 // 100 + BLOCKS * BLOCK)
        iq = raw[-2 * BLOCKS * BLOCK:].astype(np.float32).view(np.complex64)
        readings.update((r.khz, r) for r in measure_window(iq, center, channels))
    return readings


def confirmed(first, second):
    candidates = []
    for khz, a in first.items():
        b = second[khz]
        if a.likely_fm and b.likely_fm:
            candidates.append(Station(khz / 1000, min(a.excess_db, b.excess_db),
                                      min(a.pilot_db, b.pilot_db) >= 9))
    kept = []
    for station in sorted(candidates, key=lambda s: -s.excess_db):
        if all(abs(s.mhz - station.mhz) > 0.15 for s in kept):
            kept.append(station)
    return sorted(kept, key=lambda s: s.mhz)


def survey():
    print("Scanning 87.5–108.0 MHz in 100 kHz steps (pass 1/2)…", file=sys.stderr, flush=True)
    first = sweep(0)
    print("Checking the band again with offset tunings (pass 2/2)…", file=sys.stderr, flush=True)
    stations = confirmed(first, sweep(300))
    print(f"Likely FM stations ({len(stations)}), dB above the estimated RF floor:", file=sys.stderr)
    for station in stations:
        pilot = ", 19 kHz pilot" if station.pilot else ""
        print(f"  {station.mhz:5.1f} MHz   +{station.excess_db:.1f} dB{pilot}", file=sys.stderr)
    return stations


if __name__ == "__main__":
    try:
        survey()
    except (OSError, RuntimeError) as exc:
        sys.exit(f"Scan stopped: {exc}")
