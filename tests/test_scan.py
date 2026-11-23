import unittest

import numpy as np

from radio import scan


class ScanTest(unittest.TestCase):
    def test_sweep_covers_band_edges_and_keeps_them_in_the_usable_band(self):
        for shift in (0, 300):
            distances = [min(abs(k - c - shift) for c in scan.CENTERS_KHZ) for k in scan.DIAL_KHZ]
            self.assertLessEqual(max(distances), 3000)
        self.assertEqual((min(scan.DIAL_KHZ), max(scan.DIAL_KHZ), len(scan.DIAL_KHZ)),
                         (87500, 108000, 206))

    def test_demodulation_separates_mono_fm_and_stereo_fm_from_noise_and_a_carrier(self):
        center = 99_750
        khz = [99_100, 99_200, 99_900, 100_100, 100_200, 100_300]
        rng = np.random.default_rng(10)
        n = 12 * scan.BLOCK
        t = np.arange(n) / scan.IQ_RATE
        iq = rng.normal(0, 0.3, (n, 2)).astype('f').view(np.complex64).ravel()
        for carrier, deviation, pilot in [(99_100, 0, 0), (100_100, 35_000, 6_750), (100_300, 30_000, 0)]:
            phase = (2 * np.pi * (carrier - center) * 1000 * t
                     + deviation / 1000 * np.sin(2*np.pi*1000*t)
                     + pilot / 19000 * np.sin(2*np.pi*19000*t))
            iq += np.exp(1j * phase)
        rows = {r.khz: r for r in scan.measure_window(iq, center, khz)}
        self.assertEqual({r.khz for r in rows.values() if r.likely_fm}, {100_100, 100_300})
        stations = scan.confirmed(rows, rows)
        self.assertEqual([(s.mhz, s.pilot) for s in stations], [(100.1, True), (100.3, False)])

    def test_single_pass_spur_is_rejected(self):
        good = scan.Reading(87_500, 20, 0, 0.2, 0.02, 28)
        noise = scan.Reading(87_500, 1, 0, 0.2, 0.3, 0)
        self.assertEqual(scan.confirmed({87500: good}, {87500: noise}), [])
