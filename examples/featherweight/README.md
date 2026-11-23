# Featherweight

A four-line radio poem. A feather falls, a scale rocks and a needle rises. Each movement gets a ten-second Strudel phrase. The source for the included reference take is the MDR JUMP song title in redsea’s published example of a decoded FM RadioText group. Codex wrote the reference lines and patterns in response to the title. [Source RadioText](https://github.com/windytan/redsea#readme)

Run from the RDS Poet repository root with the system Python used for GNU Radio. This builds a 40-second read-along without a model download.

```sh
/usr/bin/python3 -m examples.featherweight --reference
```

Open `build/featherweight/index.html`. **Play Strudel** runs the score in the browser with the pinned Strudel web package. The first load needs internet access. **Play WAV preview** plays a local rendering of the same notes and rests, with the poem timed to the audio. The WAV uses this repo’s oscillator envelopes. The Strudel source and an RDS-compatible JSONL score sit next to the player. [Strudel’s browser integration](https://strudel.cc/technical-manual/project-start/)

## Make a fresh take on the ThinkPad

The test machine is a Lenovo ThinkPad T14s Gen 1 running Linux Mint 22.1, with a four-core i5-10310U, integrated Intel UHD graphics and 15 GiB of usable RAM. Start with `qwen2.5:1.5b` under Ollama on CPU. Its Q4_K_M download is 986 MB. The performer requests a 2,048-token context and three inference threads. This leaves CPU capacity for the audio path. The command prints wall time for each accepted line, including initial model loading on the first request. Compare those times with the ten-second reading interval before deciding how far ahead to compose. [Ollama model details](https://ollama.com/library/qwen2.5:1.5b)

Run these commands from the repository root to set up Ollama and generate four new lines and their musical answers. Playback starts when you open the completed take. [Ollama Linux setup](https://docs.ollama.com/linux)

```sh
./setup-ollama.sh
/usr/bin/python3 -m examples.featherweight --out build/featherweight-qwen
```

To draw on the sound at the antenna, choose public FM stations you can receive. The HackRF visits them in order. With `--transcribe` it samples six seconds for each line and Whisper supplies the words it recognizes. The player shows the measurements and transcript as the corresponding three-layer sketch plays. Omit `--transcribe` for two-second acoustic cues.

```sh
/usr/bin/python3 -m examples.featherweight --sample-mhz 105.7 88.1 --transcribe --out build/featherweight-live
```

To keep composing from the fourth line’s arrangement, pass `--sketch build/featherweight-live/last-set.strudel` to `perform.py`. The browser player links to that final sketch and to the full four-line score.

Generated takes use a new musical section for each line. Add `--edit-size subtle` to hear smaller note edits within the existing arrangement.

To play those four lines once on the iPod, tune to 100.0 MHz and use the generated score. Add `--loop-score` to repeat the take.

```sh
/usr/bin/python3 perform.py --score build/featherweight-live/score.jsonl --fm --tx-gain-db 40
```

For a continuing poem, run the performer directly. The single HackRF samples the selected stations before transmission, then Qwen composes fresh lines from the alternating cues.

```sh
/usr/bin/python3 perform.py --sample-mhz 105.7 88.1 --fm --tx-gain-db 40
```

The [two-station take](station-take.jsonl) pairs each generated line with its measured FM cue and playable Strudel. To hear all four voices on speakers without a receiver or model, run `/usr/bin/python3 perform.py --score examples/featherweight/station-take.jsonl`.

The [four-line Qwen take](qwen-take.jsonl) was generated on this ThinkPad from the archived MDR JUMP cue. Across three completed takes with the model already loaded on CPU, the median time for an accepted line and its notes was 4.0 seconds. The slowest took 9.6 seconds. The saved take took 15.6 seconds to compose and plays for 40 seconds. To hear this exact result on speakers, run `/usr/bin/python3 perform.py --score examples/featherweight/qwen-take.jsonl --bars 4` from the repository root.
