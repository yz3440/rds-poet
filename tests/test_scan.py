import io
import sys
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, call, patch

import numpy as np

import perform
from arrangement import HIGH_TAPS, LONG_BASS, choose_section
from radio import listen, scan
from radio.cues import pair_observations, split_cue


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

    def test_spectral_flux_detects_attacks_independently_of_pitch(self):
        t = np.arange(2 * listen.AUDIO_RATE) / listen.AUDIO_RATE
        note = 0.25 * np.sin(2 * np.pi * 400 * t)
        held = listen.describe(note, 88.1)
        staccato = listen.describe(note * ((t % 0.25) < 0.09), 88.1)
        self.assertIn("centroid 400 Hz, sparse attacks (0.0 attacks/s)", held)
        self.assertIn("frequent attacks", staccato)
        self.assertEqual(choose_section(pair_observations(held, staccato), None, set()), HIGH_TAPS)
        self.assertEqual(choose_section(pair_observations(staccato, held), None, set()), LONG_BASS)

    def test_survey_selects_clear_neighbors_and_finishes_capturing_before_tx(self):
        stations = [scan.Station(mhz, 20, True, True) for mhz in (97.1, 99.1, 100.0, 100.5)]
        stations += [scan.Station(99.9, 5, True, False)]
        whisper = Mock()

        def sample(mhz, recognizer=None):
            cue = f"{mhz} MHz FM: midrange, pulsing"
            return cue + (f"\nWhisper transcript: words from {mhz}" if recognizer else "")

        with patch.object(scan, "survey", return_value=stations), redirect_stderr(io.StringIO()), \
                patch.object(listen, "capture_hackrf", side_effect=sample) as capture:
            read = listen.hackrf_cues(dial=100, before_tx=True, recognizer=whisper)
            self.assertEqual(capture.call_args_list, [call(100.5, whisper), call(99.1, whisper)])
            a, b = read(), read()
            self.assertEqual(a, read())
            self.assertEqual(capture.call_count, 2)
            self.assertEqual(split_cue(a), ("100.5 MHz FM: midrange, pulsing",
                                           "99.1 MHz FM: midrange, pulsing", "words from 100.5"))
            self.assertEqual(split_cue(b)[2], "words from 99.1")

    def test_empty_scan_preserves_explicit_tuning_and_reports_auto_failure(self):
        with patch.object(scan, "survey", return_value=[]), redirect_stderr(io.StringIO()), \
                patch.object(listen, "capture_hackrf", return_value="manual") as capture:
            with self.assertRaisesRegex(RuntimeError, "No source stations"):
                listen.hackrf_cues()
            read = listen.hackrf_cues([92.5])
            capture.assert_not_called()
            self.assertEqual(read(), "manual")
            capture.assert_called_once_with(92.5, None)

    def test_performer_starts_with_survey_and_allows_offline_mode(self):
        for flags in ([], ["--no-scan"]):
            with patch.object(sys, "argv", ["perform.py", *flags]), patch.object(perform, "run") as run, \
                    patch.object(listen, "hackrf_cues") as prepare:
                perform.main()
                if flags:
                    prepare.assert_not_called()
                    self.assertIn("Room tone", run.call_args.args[1]())
                else:
                    prepare.assert_called_once_with(None, scan=True, dial=100, neighbor_offset_khz=None,
                                                    before_tx=False, recognizer=None)
                    self.assertIs(run.call_args.args[1], prepare.return_value)

    def test_exact_neighbor_hears_the_next_channel_and_stays_inside_the_band(self):
        with patch.object(scan, "survey") as survey, redirect_stderr(io.StringIO()), \
                patch.object(listen, "capture_hackrf", side_effect=["music", "mostly radio hiss", "edge"]) as capture:
            read = listen.hackrf_cues([105.7, 108.0], scan=False, neighbor_offset_khz=200)
            self.assertEqual(split_cue(read()), ("music", "mostly radio hiss", ""))
            self.assertEqual(split_cue(read()), ("edge", "", ""))
            self.assertEqual(capture.call_args_list, [call(105.7, None), call(105.9, None), call(108, None)])
            survey.assert_not_called()

    def test_off_center_fm_bleed_skips_whisper_and_musical_measurements(self):
        t = np.arange(125_000) / 250_000
        iq = np.exp(2j * np.pi * 65_000 * t + 3j * np.sin(2 * np.pi * 440 * t))
        recognizer = Mock()
        cue = listen.describe_baseband(iq, 105.9, recognizer)
        self.assertIn("adjacent-channel bleed", cue)
        self.assertNotIn("attacks/s", cue)
        recognizer.assert_not_called()
