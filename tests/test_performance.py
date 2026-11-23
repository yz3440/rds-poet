import io
import json
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from queue import Queue
from unittest.mock import patch

import numpy as np

import perform
import score
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
        argv = ["perform.py", "--score", str(score), "--fm", "87.5", "--tx-gain-db", "40"]
        with patch.object(sys, "argv", argv), patch.object(perform.signal, "signal"), \
             patch.object(perform, "hackrf_sink", side_effect=RuntimeError("no device")) as sink, \
             redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            perform.main()
        sink.assert_called_once_with(87_500_000, 40)

    def test_late_score_repeats_audio_and_changes_text_only_on_boundary(self):
        first = Phrase("glass", "", np.array([1, 2, 3, 4], np.float32))
        second = Phrase("rain", "", np.array([5, 6, 7, 8], np.float32))
        upcoming = Queue(maxsize=1)
        shown = []
        player = perform.Player(first, upcoming, lambda phrase: shown.append(phrase.line), bars=3)
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
