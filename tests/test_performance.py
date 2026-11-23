import io
import json
import sys
import tempfile
import threading
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from queue import Queue
from unittest.mock import Mock, call, patch

import numpy as np

import perform
import score
import sketch
from radio.cues import pair_observations
from radio.transcription import speech_cue
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
        composer = score.Composer("local", edit_size="subtle")
        calls = []
        def reply(model, theme, cue, history, synth, *, layers, target, step, section):
            calls.append(layers)
            notes = "d3" if layers[target].notes.split()[step] != "d3" else "a4"
            return {"line": next(lines), "strudel": f'note("{notes}").s("{synth}")'}
        with patch.object(score, "generate", side_effect=reply):
            phrases = [composer.compose("air", "pulsing", 0.25) for _ in range(8)]
        previous = calls[0]
        for phrase in phrases:
            layers = sketch.parse_layers(phrase.strudel)
            changed = [i for i, layer in enumerate(layers) if layer != previous[i]]
            self.assertEqual(changed, [phrase.edited_layer])
            np.testing.assert_array_equal(phrase.audio, score.render(
                {"line": phrase.line, "strudel": phrase.strudel}, 0.25).audio)
            previous = layers
        self.assertEqual(phrases[0].edited_layer, 1)
        self.assertEqual({p.edited_layer for p in phrases[:3]}, {0, 1, 2})
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

    def test_default_sections_change_rhythm_register_and_instrument_on_repeated_cue(self):
        composer = score.Composer("local")
        previous = composer.layers
        lines = iter(["Tin", "Ash", "Oars", "Ink"])
        def reply(model, theme, cue, history, synth, *, layers, target, step, section):
            return {"line": next(lines), "strudel": f'note("d4 a4 ~ g4 d4 f4").s("{synth}")'}
        sections = set()
        attacks = []
        with patch.object(score, "generate", side_effect=reply):
            for _ in range(4):
                phrase = composer.compose("port", "88.1 MHz FM: midrange, steady", 0.5)
                layers = sketch.parse_layers(phrase.strudel)
                lead = layers[phrase.edited_layer]
                changed = {i for i in range(3) if layers[i] != previous[i]}
                self.assertEqual(changed, {phrase.edited_layer, phrase.answer_layer})
                sections.add(phrase.section_name)
                attacks.append(lead.fast * sum(n != "~" for n in lead.notes.split()))
                np.testing.assert_array_equal(sketch.mix(layers, 0.5), phrase.audio)
                previous = layers
        self.assertEqual(len(sections), 4)
        self.assertLessEqual(max(attacks[::2]), 4)
        self.assertGreaterEqual(min(attacks[1::2]), 9)

    def test_bold_preserves_muted_and_hand_edited_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "working.strudel"
            source = sketch.STARTER.read_text().replace(".gain(0.52)", ".gain(0)")
            path.write_text(source)
            composer = score.Composer("local", path)
            source = source.replace("~ a4 c5 ~ e5", "g4 ~ a5")
            path.write_text(source)
            before = sketch.parse_layers(source)
            with patch.object(score, "generate", return_value={"line": "Rail", "strudel": 'note("g3").s("sine")'}):
                phrase = composer.compose("air", "bright and thin", 0.25)
            self.assertEqual(phrase.edited_layer, 1)
            self.assertIsNone(phrase.answer_layer)
            self.assertEqual(composer.layers[0], before[0])
            self.assertEqual(composer.layers[2], before[2])

    def test_bold_uses_complete_motif_in_ollama_schema(self):
        composer = score.Composer("local")
        payload = json.dumps({"line": "Glass arc", "strudel": "note('g4 ~ d5').s('triangle')"})
        stream = io.BytesIO(json.dumps({"message": {"content": payload}}).encode())
        with patch.object(score, "urlopen", return_value=stream) as post:
            phrase = composer.compose("port", "88.1 MHz FM: bright and thin, steady", 0.25)
        body = json.loads(post.call_args.args[0].data)
        pattern = body["format"]["properties"]["strudel"]["pattern"]
        self.assertRegex("note('g4 ~ d5').s('triangle')", pattern)
        self.assertNotRegex("note('d2').s('triangle')", pattern)
        self.assertIn("complete", json.loads(body["messages"][-1]["content"])["task"])
        self.assertEqual(phrase.section_name, "spaced bells")

    def test_live_and_saved_phrases_keep_their_own_cue(self):
        row = {"line": "The loose hinge", "strudel": 'note("d3").s("sine")'}
        capture = "105.7 MHz FM: low and muffled, pulsing, audio level -10 dBFS"
        self.assertIsNone(score.render(row, 0.25).cue)
        self.assertEqual(score.render({**row, "cue": capture}, 0.25).cue, capture)
        with patch.object(score, "generate", return_value=row):
            self.assertEqual(score.Composer("local").compose("air", capture, 0.25).cue, capture)

    def test_radio_words_and_current_sketch_reach_ollama(self):
        reply = json.dumps({"line": "River in the window", "strudel": "note('g4').s('triangle')"})
        stream = io.BytesIO(json.dumps({"message": {"content": reply}}).encode())
        cue = speech_cue("88.1 MHz FM: bright and thin, steady", np.ones(100), lambda audio: "river ferry")
        neighbor = "88.9 MHz FM: midrange, steady, sparse attacks (0.0 attacks/s)"
        cue = pair_observations(cue, neighbor)
        composer = score.Composer("local", edit_size="subtle")
        layers = composer.layers
        with patch.object(score, "urlopen", return_value=stream) as post:
            phrase = composer.compose("port", cue, 0.25)
        request = json.loads(post.call_args.args[0].data)
        prompt = json.loads(request["messages"][-1]["content"])
        self.assertEqual(prompt["current_sketch"], [layer.code() for layer in layers])
        self.assertEqual(prompt["edit_layer"], 3)
        self.assertEqual(prompt["air"], cue.splitlines()[0])
        self.assertEqual(prompt["nearby_station"], neighbor)
        self.assertEqual(prompt["overheard_speech"], "river ferry")
        self.assertIn("river ferry", phrase.cue)

    def test_hand_edit_keeps_other_generated_layers_and_bad_edit_can_recover(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "set.strudel"
            seed = sketch.STARTER.read_text()
            source.write_text(seed)
            composer = score.Composer("local", source, edit_size="subtle")
            replies = [{"line": "Turning", "strudel": 'note("a2 ~").s("sine")'},
                       {"line": "Reeds", "strudel": 'note("e5 c5").s("triangle")'}]
            with patch.object(score, "generate", side_effect=replies):
                first = composer.compose("air", "low and muffled", 0.25)
                source.write_text("stack(")
                with self.assertRaises(ValueError):
                    composer.compose("air", "bright and thin", 0.25)
                self.assertEqual(sketch.parse_layers(first.strudel), composer.layers)
                source.write_text(seed.replace("d4 ~ a3 ~ ~ f4 ~", "c4 e4 ~"))
                last = composer.compose("air", "bright and thin", 0.25)
            layers = sketch.parse_layers(last.strudel)
            self.assertEqual([l.notes for l in layers], ["a2 ~ a2 c3", "c4 e4 ~", "e5 c5 c5 ~ e5"])

    def test_silent_and_hiss_windows_skip_recognition(self):
        with patch("radio.transcription.Whisper") as asr:
            for cue in ["88.1 MHz FM: nearly still", "88.1 MHz FM: bright and thin, mostly radio hiss"]:
                self.assertIn("skipped", speech_cue(cue, np.zeros(100), asr))
            asr.assert_not_called()

    def test_rtl_transcribes_six_seconds_while_receiver_keeps_draining(self):
        end = threading.Event()
        chunk = (np.sin(np.arange(listen.AUDIO_RATE) * 0.1) * 8000).astype("=i2").tobytes()
        chunks = iter([chunk] * 6)
        proc = Mock()
        proc.poll.return_value = None
        proc.terminate.side_effect = end.set

        def read(_size):
            packet = next(chunks, None)
            if packet is None:
                end.wait(3)
                return b""
            return packet

        proc.stdout.read.side_effect = read
        recognizer = Mock(return_value="ferry bells")
        with patch.object(listen.subprocess, "Popen", return_value=proc), listen.rtl_cues(88.1, recognizer) as cue:
            self.assertIn("Whisper transcript: ferry bells", cue())
        self.assertEqual(len(recognizer.call_args.args[0]), 6 * listen.AUDIO_RATE)
        recognizer.assert_called_once()

    def test_listen_command_prints_each_capture_in_order(self):
        argv = ["radio.listen", "105.7", "88.1"]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stdout(output), \
             patch.object(listen, "capture_hackrf", side_effect=["first cue", "second cue"]) as capture:
            listen.main()
        self.assertEqual(capture.call_args_list, [call(105.7, None), call(88.1, None)])
        self.assertEqual(output.getvalue().splitlines(), ["first cue", "second cue"])

    def test_gate_and_speed_change_audio_in_each_supported_voice(self):
        spectra = []
        for synth in ("sine", "triangle", "square", "sawtooth"):
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
            self.assertEqual(capture.call_args_list, [call(93.9, None), call(99.5, None), call(93.9, None)])
        with patch.object(listen, "capture_hackrf", side_effect=["a", "b"]) as capture:
            cue = listen.station_cues([93.9, 99.5], before_tx=True)
            self.assertEqual(capture.call_args_list, [call(93.9, None), call(99.5, None)])
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
