import io
import json
import re
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from queue import Queue
from unittest.mock import call, patch

import numpy as np

import perform
import score
from radio import listen
from score import Phrase


class PerformanceTest(unittest.TestCase):
    def test_ollama_strudel_survives_json_and_score_rendering(self):
        code = "note('c4 ~ g4').s('triangle')"
        line = "A bird's shadow weighs the air."
        content = json.dumps({"line": line, "strudel": code})
        stream = io.BytesIO(json.dumps({"message": {"content": content}}).encode())
        with patch.object(score, "urlopen", return_value=stream) as post:
            reply = score.generate("qwen2.5:1.5b", "weight", "bright", [])
        schema = json.loads(post.call_args.args[0].data)["format"]
        self.assertRegex(code, schema["properties"]["strudel"]["pattern"])
        phrase = score.render(reply, 0.25)
        self.assertEqual(phrase.line, line)
        self.assertEqual(phrase.strudel, 'note("c4 ~ g4").s("triangle")')

    def test_fm_cli_converts_mhz_before_opening_device(self):
        score = Path(perform.__file__).parent / "examples" / "rehearsal.jsonl"
        for fm, hz in [(["--fm"], 100_000_000), (["--fm", "87.5"], 87_500_000)]:
            argv = ["perform.py", "--score", str(score), *fm, "--tx-gain-db", "40"]
            with self.subTest(fm=fm), patch.object(sys, "argv", argv), patch.object(perform.signal, "signal"), \
                 patch.object(perform, "hackrf_sink", side_effect=RuntimeError("no device")) as sink, \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                perform.main()
            sink.assert_called_once_with(hz, 40)

    def test_late_score_repeats_audio_and_changes_text_only_on_boundary(self):
        first = Phrase("glass", "", np.array([1, 2, 3, 4], np.float32))
        second = Phrase("rain", "", np.array([5, 6, 7, 8], np.float32))
        upcoming = Queue(maxsize=1)
        shown = []
        player = perform.Player(first, upcoming.get_nowait, lambda phrase: shown.append(phrase.line), bars=3)
        output = np.empty(2, dtype=np.float32)
        heard = []
        for step in range(6):
            if step == 3:
                upcoming.put(second)
            self.assertEqual(player.work([], [output]), 2)
            heard.extend(output)
        self.assertEqual(heard, [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(shown, ["glass", "rain"])
        self.assertEqual(player.work([], [output]), -1)

    def test_saved_score_ends_unless_loop_requested(self):
        saved = Path(perform.__file__).parent / "examples" / "featherweight" / "qwen-take.jsonl"
        archived = [json.loads(row)["line"] for row in saved.read_text().splitlines()]
        with tempfile.TemporaryDirectory() as folder:
            wav = Path(folder) / "preview.wav"
            for flags, expected in [([], archived), (["--loop-score", "--bars", "5"], archived + archived[:1])]:
                argv = ["perform.py", "--score", str(saved), "--line-seconds", "0.25", "--wav-out", str(wav), *flags]
                log = io.StringIO()
                with patch.object(sys, "argv", argv), redirect_stdout(log):
                    perform.main()
                lines = [line for line in log.getvalue().splitlines() if not line.startswith((" ", "setcps("))]
                self.assertEqual(lines, expected)
                with wave.open(str(wav)) as audio:
                    self.assertEqual(audio.getnframes(), len(expected) * score.RATE // 4)

    def test_composer_changes_voices_and_rejects_returning_images(self):
        lines = iter(["The dandelion", "A DANDELION opens!", "Oars touch water", "Glass sings",
                      "Who opened the gate?", "Ash collects", "Ink lifts", "Rain", "Brass", "Silk"])
        composer = score.Composer("local")
        def reply(model, theme, cue, history, synth):
            return {"line": next(lines), "strudel": f'note("d3 ~ c4").s("{synth}")'}
        with patch.object(score, "generate", side_effect=reply):
            phrases = [composer.compose("air", "pulsing", 0.25) for _ in range(8)]
        voices = [re.fullmatch(score.PATTERN, phrase.strudel)[2] for phrase in phrases]
        self.assertEqual(set(voices[:4]), set(score.VOICES))
        self.assertEqual(set(voices[4:]), set(score.VOICES))
        self.assertNotEqual(voices[3], voices[4])
        self.assertEqual(phrases[1].line, "Oars touch water")
        self.assertEqual(len(set(p.line for p in phrases)), 8)

    def test_recurring_image_can_echo_once_inside_a_line(self):
        lines = ["A loose dandelion", "Rails carry dandelions west",
                 "Silt covers the dandelion", "Steam lifts the latch"]
        replies = [{"line": line, "strudel": 'note("d3").s("sine")'} for line in lines]
        composer = score.Composer("local")
        with patch.object(score, "generate", side_effect=replies):
            result = [composer.compose("air", "steady", 0.25).line for _ in range(3)]
        self.assertEqual(result, [lines[0], lines[1], lines[3]])

    def test_live_and_saved_phrases_keep_their_own_cue(self):
        row = {"line": "The loose hinge", "strudel": 'note("d3").s("sine")'}
        capture = "105.7 MHz FM: low and muffled, pulsing, audio level -10 dBFS"
        self.assertIsNone(score.render(row, 0.25).cue)
        self.assertEqual(score.render({**row, "cue": capture}, 0.25).cue, capture)
        with patch.object(score, "generate", return_value=row):
            self.assertEqual(score.Composer("local").compose("air", capture, 0.25).cue, capture)

    def test_listen_command_prints_each_capture_in_order(self):
        argv = ["radio.listen", "105.7", "88.1"]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stdout(output), \
             patch.object(listen, "capture_hackrf", side_effect=["first cue", "second cue"]) as capture:
            listen.main()
        self.assertEqual(capture.call_args_list, [call(105.7), call(88.1)])
        self.assertEqual(output.getvalue().splitlines(), ["first cue", "second cue"])

    def test_gate_and_speed_change_audio_in_each_supported_voice(self):
        spectra = []
        for synth in score.VOICES:
            code = f'note("a4").s("{synth}").clip(0.25).fast(2)'
            samples = score.render({"line": "Tap", "strudel": code}, 1).audio
            for step in np.split(samples, 2):
                self.assertGreater(np.max(np.abs(step[:6000])), 0.1)
                self.assertFalse(step[6000:].any())
            # Inspect a whole number of periods in the sustained section.
            spectra.append(np.abs(np.fft.rfft(samples[480:2880]))[[22, 44, 66]])
        sine, tri, square, saw = np.array(spectra) / np.array(spectra)[:, :1]
        self.assertLess(sine[2], 0.01)
        self.assertAlmostEqual(tri[2], 1/9, places=2)
        self.assertAlmostEqual(square[2], 1/3, places=2)
        self.assertAlmostEqual(saw[1], 1/2, places=2)

    def test_station_rotation_captures_fresh_or_releases_device_before_tx(self):
        with patch.object(listen, "capture_hackrf", side_effect=["a", "b", "c"]) as capture:
            cue = listen.station_cues([93.9, 99.5])
            capture.assert_not_called()
            self.assertEqual([cue(), cue(), cue()], ["a", "b", "c"])
            self.assertEqual(capture.call_args_list, [call(93.9), call(99.5), call(93.9)])
        with patch.object(listen, "capture_hackrf", side_effect=["a", "b"]) as capture:
            cue = listen.station_cues([93.9, 99.5], before_tx=True)
            self.assertEqual(capture.call_args_list, [call(93.9), call(99.5)])
            self.assertEqual([cue(), cue(), cue()], ["a", "b", "a"])
            self.assertEqual(capture.call_count, 2)

    def test_weak_offset_station_over_receiver_dc(self):
        t = np.arange(listen.IQ_RATE // 2) / listen.IQ_RATE
        # A 400 Hz FM tone, 20 dB below the receiver's center spike.
        phase = 2 * np.pi * listen.TUNE_OFFSET * t + 50 * np.sin(2 * np.pi * 400 * t)
        iq = (10 + np.exp(1j * phase)).astype(np.complex64)
        cue = listen.describe_channel(iq, 93.9)
        self.assertIn("93.9 MHz FM: low and muffled, steady", cue)
        self.assertNotIn("hiss", cue)

    def test_empty_channel_reports_hiss(self):
        rng = np.random.default_rng(12)
        iq = rng.standard_normal((listen.IQ_RATE // 2, 2), dtype=np.float32).view(np.complex64).ravel()
        self.assertIn("mostly radio hiss", listen.describe_channel(iq, 99.5))
