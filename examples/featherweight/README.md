# Featherweight

A four-line radio poem. A feather falls, a scale rocks and a needle rises. Each movement gets a two-second Strudel phrase. The source for the included reference take is the MDR JUMP song title in redsea’s published example of a decoded FM RadioText group. Codex wrote the reference lines and patterns in response to the title. [Source RadioText](https://github.com/windytan/redsea#readme)

Run from the RDS Poet repository root with the system Python used for GNU Radio. This builds an eight-second read-along without a model download.

```sh
/usr/bin/python3 -m examples.featherweight --reference
```

Open `build/featherweight/index.html`. **Play Strudel** runs the score in the browser with the pinned Strudel web package. The first load needs internet access. **Play WAV preview** plays a local rendering of the same notes and rests, with the poem timed to the audio. The WAV uses this repo’s oscillator envelopes. The Strudel source and an RDS-compatible JSONL score sit next to the player. [Strudel’s browser integration](https://strudel.cc/technical-manual/project-start/)

## Make a fresh take on the ThinkPad

The test machine is a Lenovo ThinkPad T14s Gen 1 running Linux Mint 22.1, with a four-core i5-10310U, integrated Intel UHD graphics and 15 GiB of usable RAM. Start with `qwen2.5:1.5b` under Ollama on CPU. Its Q4_K_M download is 986 MB. The performer requests a 2,048-token context and three inference threads. This leaves CPU capacity for the audio path. The command prints wall time for each accepted line, including initial model loading on the first request. Compare those times with the two-second reading interval before deciding how far ahead to compose. [Ollama model details](https://ollama.com/library/qwen2.5:1.5b)

With Ollama installed and its local service running, these commands generate four new lines and their musical answers. Playback starts when you open the completed take. [Ollama Linux setup](https://docs.ollama.com/linux)

```sh
ollama pull qwen2.5:1.5b
/usr/bin/python3 -m examples.featherweight --out build/featherweight-qwen
```

To draw on the sound at the antenna, replace 99.5 below with a public FM station you can receive. The HackRF listens for two seconds, closes, and supplies the measured level, brightness and pulse to the model. The player shows the cue used.

```sh
/usr/bin/python3 -m examples.featherweight --sample-mhz 99.5 --out build/featherweight-live
```

To play that take on the iPod, use the existing transmitter with the generated score. It loops until Ctrl+C.

```sh
/usr/bin/python3 perform.py --score build/featherweight-live/score.jsonl --fm 87.5 --tx-gain-db 40
```
